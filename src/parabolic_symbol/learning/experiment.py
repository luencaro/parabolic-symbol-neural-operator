"""Un experimento GRF completo, seccion por seccion, como en 02_calor_1d.ipynb.

`GRFExperiment` reune lo que repiten los notebooks 03-05 (calor 2D y laplaciano
fraccionario en 1D y 2D): datos GRF, banda (c, C) y kappa, barrido, resultados,
modelo seleccionado, prueba, curvas, simbolo aprendido, certificacion de
parabolicidad y clase S^{m,rho}. Cada metodo es una seccion del notebook:
imprime lo mismo que la celda de 02, guarda sus tablas (CSV) y figuras (PNG)
en `results` y devuelve las tablas para mostrarlas con `display`.

Configuracion comun (la de 02): covarianza 7^4 (-Delta + 7^2)^{-2.5}, tiempos
TRAIN_TIMES / TEST_TIMES, 100 + 20 condiciones iniciales, particion por k (80/20),
todos los pares (k, t) con |k|_inf <= k_cap (k_max = k_cap), c y C de los datos de entrenamiento con
orden m, kappa = theoretical_kappa(), C_alpha de los datos de entrenamiento
(diferencias finitas de a_hat) con margen 1.5, y el barrido de SWEEP con early stopping sobre la perdida total de
validacion (trayectoria + parabolicidad + L_class).

En 2D el simbolo es radial, asi que las figuras sobre k se dibujan contra |k| y
se agrega un mapa del error relativo sobre (k1, k2).

Modelo guardado: tras la seleccion, `select` guarda el modelo elegido en
`results/modelo.pt` (`save_experiment_model`): pesos de g_theta y todo lo
necesario para reconstruirlo y regenerar sus datos sin el notebook.
`load_experiment_model` reconstruye el ParabolicMLPSymbol,
`GRFExperiment.from_checkpoint` + `regenerate_data` regeneran los datos con la
misma semilla, y `retrain_selected` reentrena solo la configuracion elegida
cuando hay resultados de una corrida anterior pero no `modelo.pt`.
"""
from __future__ import annotations

import ast
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..symbol import Symbol
from ..symbols.fractional_laplacian import FractionalLaplacianSymbol
from ..symbols.heat import HeatSymbol
from ..symbols.parabolic_mlp_symbol import ParabolicMLPSymbol
from ..data.grf import GRFData, field_rel_l2, grf_datasets
from ..data.grids import k_grid, split_trajectory_data
from ..verification.parabolicity import ParabolicityVerifier
from ..verification.symbol_class import SymbolClassVerifier
from .estimation import (
    Bounds, estimate_bounds, estimate_class_constants, estimate_symbol, oracle_class_constants, symbol_values,
)
from .evaluation import symbol_report, trajectory_report
from .sweep import SweepRun, best_run, run_sweep, sweep_table
from .training import TrainingConfig
from ..utils import DROPOUT_PLACEMENTS, Split, default_device

# El barrido de 02: profundidad x ancho x ubicacion del dropout x weight decay.
SWEEP = dict(
    depths=(2, 3, 4),
    widths=(32, 64, 128),
    dropout=0.2,
    dropout_layers=("none", "middle"),
    weight_decays=(0.0, 1e-4),
)

# lambda con que se estima kappa_hat del simbolo de referencia (celda 6 de 02).
LAM_VALUES = [complex(re, im) for re in (0, 1, 5, 10, 25, 50) for im in (-50, -10, -2, 0, 2, 10, 50)]

# Rejilla de lambda de la CERTIFICACION, independiente de LAM_VALUES:
# |lambda| log-espaciado en [1e-2, 1e3] y 25 angulos en [-pi/2, pi/2] (frontera incluida).
CERT_LAM = (
    np.logspace(-2, 3, 25)[:, None] * np.exp(1j * np.linspace(-np.pi / 2, np.pi / 2, 25))
).ravel()

# Columnas de la tabla del barrido que se muestran.
SWEEP_COLS = ["capas", "ancho", "dropout", "weight_decay", "val_rel_l2", "val_total", "val_par_max_ratio"]


def relative_symbol_error(a_theta: np.ndarray, a_ref: np.ndarray) -> np.ndarray:
    """|a_theta - a| / |a|, con 0 donde a = 0 (k = 0, donde a_theta(0) = 0 por construccion)."""
    err = np.abs(a_theta - a_ref)
    return np.divide(err, np.abs(a_ref), out=np.zeros_like(err), where=np.abs(a_ref) > 0)


def class_C_table(C_data: Dict, C_oracle: Dict, counts: Dict, margin: float) -> pd.DataFrame:
    """Tabla de C_alpha: la de los datos (la que usa L_class), la del oraculo solo
    como comparacion y cuantos k de entrenamiento entraron en cada alpha."""
    tabla = pd.DataFrame({
        "C_alpha datos (x%g)" % margin: {str(a): c for a, c in C_data.items()},
        "C_alpha oraculo (x%g)" % margin: {str(a): C_oracle[a] for a in C_data},
        "k_usados": {str(a): int(counts[a]) for a in C_data},
    })
    tabla.index.name = "alpha"
    return tabla


# ----------------------------------------------------------------------
# Modelo guardado (modelo.pt)
# ----------------------------------------------------------------------
MODEL_FILE = "modelo.pt"
CHECKPOINT_FORMAT = 1

# Simbolos de referencia que se pueden reconstruir desde modelo.pt.
REFERENCE_SYMBOLS = {"HeatSymbol": HeatSymbol, "FractionalLaplacianSymbol": FractionalLaplacianSymbol}


def _reference_spec(reference: Symbol) -> Dict:
    """Clase y argumentos del simbolo de referencia, para reconstruirlo."""
    if isinstance(reference, HeatSymbol):
        return {"clase": "HeatSymbol", "kwargs": {"n": reference.n, "diffusivity": reference.diffusivity}}
    if isinstance(reference, FractionalLaplacianSymbol):
        return {"clase": "FractionalLaplacianSymbol",
                "kwargs": {"n": reference.n, "alpha": float(reference.alpha), "rho": reference.rho}}
    raise TypeError(f"no se sabe guardar el simbolo de referencia {reference!r}")


def _build_reference(spec: Dict) -> Symbol:
    return REFERENCE_SYMBOLS[spec["clase"]](**spec["kwargs"])


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def save_experiment_model(
    path: Union[str, Path],
    run: SweepRun,
    *,
    reference: Symbol,
    m: float,
    resolution: int,
    k_cap: int,
    config: TrainingConfig,
    seed: int = 0,
    val_fraction: float = 0.2,
    dtype: torch.dtype = torch.float64,
    omega: float = 1.0,
    delta: float = 0.1,
    class_margin: float = 1.5,
    estimate_tol: float = 1e-10,
) -> Path:
    """Guarda la corrida elegida con torch.save, lista para reconstruirla sin el notebook.

    Contiene: los pesos de g_theta (state_dict), los argumentos de
    ParabolicMLPSymbol (n, m, c, C, hidden, rho, dropout, dropout_layers,
    dtype), los datos (simbolo de referencia, semilla, resolucion, k_cap,
    val_fraction), las constantes del experimento (omega, delta, margen y tol)
    y la configuracion de entrenamiento usada (`config` con el weight decay de
    la corrida). Todo son tipos simples, asi que se lee con weights_only=True.
    """
    path = Path(path)
    symbol = run.symbol
    config = replace(config, weight_decay=run.weight_decay, class_reference=None)
    checkpoint = {
        "formato": CHECKPOINT_FORMAT,
        "state_dict": {key: value.detach().cpu() for key, value in symbol.net.state_dict().items()},
        "simbolo": {
            "n": symbol.n, "m": float(symbol.m), "c": symbol.c, "C": symbol.C,
            "hidden": tuple(symbol.hidden), "rho": symbol.rho, "dropout": symbol.dropout,
            "dropout_layers": tuple(symbol.dropout_layers), "dtype": _dtype_name(symbol.dtype),
        },
        "seleccion": {
            "capas": run.n_layers, "ancho": run.width, "dropout_layers": run.dropout_layers,
            "dropout_rate": run.dropout_rate, "weight_decay": run.weight_decay,
            "val_rel_l2": run.val_rel_l2, "mejor_epoca": run.best_epoch, "epocas": run.epochs_run,
        },
        "datos": {
            "referencia": _reference_spec(reference), "semilla": seed, "resolucion": resolution,
            "k_cap": k_cap, "val_fraction": val_fraction, "dtype": _dtype_name(dtype),
        },
        "experimento": {
            "m": float(m), "omega": omega, "delta": delta, "class_margin": class_margin,
            "estimate_tol": estimate_tol,
        },
        "entrenamiento": asdict(config),
    }
    torch.save(checkpoint, path)
    print(f"modelo guardado en {path}")
    return path


def load_checkpoint(path: Union[str, Path]) -> Dict:
    """El diccionario de modelo.pt (ver `save_experiment_model`)."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("formato") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: formato de modelo.pt desconocido {checkpoint.get('formato')!r}")
    return checkpoint


def load_experiment_model(
    path: Union[str, Path], device: Optional[torch.device] = None
) -> ParabolicMLPSymbol:
    """Reconstruye el ParabolicMLPSymbol guardado en modelo.pt, en modo eval."""
    checkpoint = load_checkpoint(path)
    spec = checkpoint["simbolo"]
    symbol = ParabolicMLPSymbol(
        n=spec["n"], m=spec["m"], c=spec["c"], C=spec["C"], hidden=spec["hidden"],
        rho=spec["rho"], dropout=spec["dropout"], dropout_layers=spec["dropout_layers"],
        device=device, dtype=getattr(torch, spec["dtype"]),
    )
    symbol.net.load_state_dict(checkpoint["state_dict"])
    symbol.net.eval()
    return symbol


def selected_config(results: Union[str, Path]) -> Dict:
    """La configuracion elegida de una corrida anterior y su val_rel_l2 reportado.

    Se lee de seleccion.csv. Los seleccion.csv anteriores a modelo.pt solo
    traen las metricas, no la configuracion: entonces se toma la fila de menor
    val_rel_l2 de barrido.csv (la que elige `best_run`) y se comprueba que su
    val_rel_l2 es el de seleccion.csv. Sin seleccion.csv (02_calor_1d) basta
    barrido.csv.
    """
    results = Path(results)
    seleccion = None
    if (results / "seleccion.csv").exists():
        seleccion = pd.read_csv(results / "seleccion.csv", index_col=0).iloc[:, 0]
    if seleccion is not None and "capas" in seleccion.index:
        config = {
            "capas": int(seleccion["capas"]), "ancho": int(seleccion["ancho"]),
            "dropout_layers": str(seleccion["dropout_layers"]),
            "weight_decay": float(seleccion["weight_decay"]),
            "val_rel_l2": float(seleccion["rel_l2"]),
        }
    else:
        barrido = pd.read_csv(results / "barrido.csv")
        fila = barrido.loc[barrido["val_rel_l2"].idxmin()]
        config = {
            "capas": int(fila["capas"]), "ancho": int(fila["ancho"]),
            "dropout_layers": str(fila["dropout_layers"]),
            "weight_decay": float(fila["weight_decay"]),
            "val_rel_l2": float(fila["val_rel_l2"]),
        }
        if seleccion is not None and not np.isclose(float(seleccion["rel_l2"]), config["val_rel_l2"], rtol=1e-9):
            raise ValueError(f"{results}: la mejor fila de barrido.csv no es la de seleccion.csv")
    # Ubicacion del dropout: una etiqueta de DROPOUT_PLACEMENTS o indices explicitos "(1, 2)".
    if config["dropout_layers"] not in DROPOUT_PLACEMENTS:
        config["dropout_layers"] = tuple(ast.literal_eval(config["dropout_layers"]))
    return config


@dataclass
class GRFExperiment:
    """Experimento GRF para un simbolo de referencia; ver el docstring del modulo.

    `m` es el orden del simbolo aprendido (m = alpha en el fraccionario, 2 en el
    calor) y fija la banda c < a/<k>^m < C. `results` es la carpeta donde se
    guardan tablas y figuras.
    """

    reference: Symbol
    m: float
    resolution: int
    k_cap: int
    results: Path
    dtype: torch.dtype = torch.float64
    device: Optional[torch.device] = None
    seed: int = 0
    omega: float = 1.0
    delta: float = 0.1            # margen de la banda (c, C)
    class_margin: float = 1.5     # margen de las C_alpha
    val_fraction: float = 0.2
    estimate_tol: float = 1e-10   # tol de estimate_symbol, como en 02

    # Se llenan seccion a seccion.
    grf_train: GRFData = field(init=False, repr=False)
    grf_test: GRFData = field(init=False, repr=False)
    train_data: Split = field(init=False, repr=False)
    val_data: Split = field(init=False, repr=False)
    k_max: int = field(init=False)
    bounds: Bounds = field(init=False)
    kappa_ref: float = field(init=False)
    kappa_theo: float = field(init=False)
    class_C: Dict = field(init=False, repr=False)
    class_C_counts: Dict = field(init=False, repr=False)
    class_C_oracle: Dict = field(init=False, repr=False)
    runs: list = field(init=False, repr=False)
    best: SweepRun = field(init=False, repr=False)
    summary: Dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.results = Path(self.results)
        self.results.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.device) if self.device is not None else default_device()
        self.summary = {}

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> "GRFExperiment":
        """El experimento de un modelo.pt, con `results` en su carpeta. Sin datos:
        `regenerate_data` los regenera con la misma semilla."""
        checkpoint = load_checkpoint(path)
        datos, exp = checkpoint["datos"], checkpoint["experimento"]
        return cls(
            reference=_build_reference(datos["referencia"]), m=exp["m"],
            resolution=datos["resolucion"], k_cap=datos["k_cap"], results=Path(path).parent,
            dtype=getattr(torch, datos["dtype"]), device=device, seed=datos["semilla"],
            omega=exp["omega"], delta=exp["delta"], class_margin=exp["class_margin"],
            val_fraction=datos["val_fraction"], estimate_tol=exp["estimate_tol"],
        )

    @property
    def n(self) -> int:
        return self.reference.n

    @property
    def symbol(self) -> ParabolicMLPSymbol:
        return self.best.symbol

    @property
    def cert_k_max(self) -> int:
        """Ventana de la certificacion: tres veces la de entrenamiento."""
        return 3 * self.k_max

    def _save(self, fig, name: str) -> None:
        plt.tight_layout()
        fig.savefig(self.results / name, dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # 1. Datos
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Datos GRF de entrenamiento y prueba, particion por k y ventana k_max."""
        self.regenerate_data()

        splits = {"entrenamiento": self.train_data, "validacion": self.val_data,
                  "prueba": self.grf_test.split}
        tabla = pd.DataFrame([
            {
                "conjunto": nombre,
                "muestras": int(split["k"].shape[0]),
                "k_unicos": len(np.unique(split["k"].numpy(), axis=0)),
            }
            for nombre, split in splits.items()
        ]).set_index("conjunto")
        tabla.to_csv(self.results / "datos.csv")
        self.summary.update(
            ventana_k=str(self.grf_train.window),
            muestras_entrenamiento=int(self.train_data["k"].shape[0]),
            muestras_validacion=int(self.val_data["k"].shape[0]),
            muestras_prueba=int(self.grf_test.split["k"].shape[0]),
        )

        print()
        print(f"k_cap = {self.k_cap}  ->  k_max = {self.k_max}  (ventana de k: {self.grf_train.window})")
        return tabla

    def regenerate_data(self) -> None:
        """Entrenamiento, validacion y prueba; con la misma semilla dan los mismos datos."""
        self.grf_train, self.grf_test = grf_datasets(
            self.reference, self.resolution, self.seed, k_cap=self.k_cap
        )
        # k_cap fija k_max del resto del experimento (puntos de parabolicidad y certificacion).
        self.k_max = self.k_cap
        self.train_data, self.val_data = split_trajectory_data(
            self.grf_train.split, val_fraction=self.val_fraction, seed=self.seed
        )

    # ------------------------------------------------------------------
    # 2. c, C, kappa y C_alpha
    # ------------------------------------------------------------------
    def estimate_constants(self) -> pd.DataFrame:
        """Banda (c, C), kappa teorico y C_alpha, todo de los datos de entrenamiento."""
        k_values = k_grid(self.n, self.k_max)
        self.kappa_ref = ParabolicityVerifier(self.reference).estimate_kappa_batch(
            k_values, LAM_VALUES, omega=self.omega
        )

        k_est, a_hat = estimate_symbol(self.train_data, tol=self.estimate_tol)
        self.bounds = estimate_bounds(k_est, a_hat, m=self.m, delta=self.delta)
        probe = ParabolicMLPSymbol(
            n=self.n, m=self.m, c=self.bounds.c, C=self.bounds.C, device="cpu", dtype=self.dtype
        )
        self.kappa_theo = probe.theoretical_kappa(self.omega)

        # C_alpha de la Def. 2.1 para L_class, por diferencias finitas del mismo a_hat
        # de entrenamiento, para todos los alpha con |alpha| <= rho del simbolo aprendido.
        self.class_C, self.class_C_counts = estimate_class_constants(
            k_est, a_hat, self.m, rho=probe.rho, margin=self.class_margin, return_counts=True
        )
        # Las del simbolo exacto en |k_i| <= k_max: solo para comparar, no se entrenan con ellas.
        self.class_C_oracle = oracle_class_constants(
            self.reference, self.k_max, margin=self.class_margin, rho=probe.rho
        )

        print(f"kappa_hat referencia = {self.kappa_ref:.4f}")
        print(f"(c, C) de los datos  = ({self.bounds.c:.4f}, {self.bounds.C:.4f})   "
              f"m = {self.m:g}, ventana: {self.bounds.window}")
        print(f"kappa teorico        = {self.kappa_theo:.4f}")
        self.summary.update(c=self.bounds.c, C=self.bounds.C, kappa=self.kappa_theo)

        tabla = class_C_table(self.class_C, self.class_C_oracle, self.class_C_counts, self.class_margin)
        constantes = pd.Series({
            "c": self.bounds.c, "C": self.bounds.C, "m": self.m,
            "kappa_teorico": self.kappa_theo, "kappa_hat_referencia": self.kappa_ref,
        }, name="valor")
        constantes.to_csv(self.results / "constantes.csv")
        tabla.to_csv(self.results / "class_C.csv")
        return tabla

    # ------------------------------------------------------------------
    # 3. Barrido
    # ------------------------------------------------------------------
    def base_config(self) -> TrainingConfig:
        """Configuracion base de TrainingConfig: solo k_max y las C_alpha dependen de los datos."""
        return TrainingConfig(k_max=self.k_max, class_C=self.class_C)

    def sweep(self, **overrides) -> None:
        """El barrido de SWEEP con la configuracion base de TrainingConfig.

        `overrides` reemplaza ejes de SWEEP (p. ej. para una prueba rapida).
        """
        grid = {**SWEEP, **overrides}
        base_config = self.base_config()
        self.runs = run_sweep(
            self.train_data,
            self.val_data,
            n=self.n,
            m=self.m,
            bounds=(self.bounds.c, self.bounds.C),
            base_config=base_config,
            device=self.device,
            dtype=self.dtype,
            **grid,
        )

    # ------------------------------------------------------------------
    # 4. Resultados del barrido
    # ------------------------------------------------------------------
    def sweep_results(self):
        """Tabla del barrido (de mejor a peor val_rel_l2) y resumen por regularizacion."""
        table = sweep_table(self.runs)
        table.to_csv(self.results / "barrido.csv", index=False)
        print(f"{len(table)} corridas, {table['segundos'].sum() / 60:.1f} min de entrenamiento en total")

        ordenada = table[SWEEP_COLS].sort_values("val_rel_l2").reset_index(drop=True)
        por_eje = [
            table.groupby(eje).agg(
                corridas=("val_rel_l2", "size"),
                mejor_val_rel_l2=("val_rel_l2", "min"),
                mejor_val_total=("val_total", "min"),
                peor_ratio=("val_par_max_ratio", "min"),
            )
            for eje in ("dropout", "weight_decay")
            if table[eje].nunique() > 1
        ]
        return ordenada, por_eje

    # ------------------------------------------------------------------
    # 5. Modelo seleccionado
    # ------------------------------------------------------------------
    def select(self) -> pd.Series:
        """La corrida de menor error relativo de validacion y sus metricas."""
        self.best = best_run(self.runs)
        best = self.best

        # kappa que exhibe el simbolo aprendido en la rejilla de certificacion (seccion 8).
        kappa_learned = ParabolicityVerifier(self.symbol).estimate_kappa_batch(
            k_grid(self.n, self.cert_k_max), CERT_LAM, omega=self.omega
        )
        print(f"kappa = theoretical_kappa() = {best.kappa:.4f}")
        print(f"kappa_hat aprendido         = {kappa_learned:.4f}   "
              f"(certificacion, |k_i| <= {self.cert_k_max})")
        print(f"kappa_hat referencia        = {self.kappa_ref:.4f}   (simbolo analitico)")
        print(f"arquitectura elegida    = {best.n_layers} capas x {best.width} neuronas")
        print(f"dropout                 = {best.dropout_desc}")
        print(f"weight decay            = {best.weight_decay:g}  (Linear ocultas interiores)")
        print(f"early stopping          = mejor epoca {best.best_epoch} de {best.epochs_run} corridas")

        configuracion = (f"{best.n_layers}x{best.width}, dropout {best.dropout_desc}, "
                         f"wd={best.weight_decay:g}")
        self.summary.update(configuracion=configuracion, val_rel_l2=best.val_rel_l2)

        val = pd.Series({
            "total": best.val_total,
            "trayectoria": best.val_trajectory,
            "parabolicidad": best.val_parabolicity,
            "clase": best.val_class,
            "rel_l2": best.val_rel_l2,
            "peor ratio": best.val_par_max_ratio,
            "puntos con ratio > 1": best.val_par_violation,
        }, name="validacion (mejor epoca)")
        # seleccion.csv lleva ademas la configuracion elegida, para `retrain_selected`.
        configuracion_elegida = pd.Series({
            "capas": best.n_layers, "ancho": best.width,
            "dropout_layers": best.dropout_layers, "weight_decay": best.weight_decay,
        }, name=val.name)
        pd.concat([val.astype(object), configuracion_elegida]).to_csv(self.results / "seleccion.csv")
        self.save_model()
        return val

    def save_model(self) -> Path:
        """Guarda el modelo elegido en results/modelo.pt (`save_experiment_model`)."""
        return save_experiment_model(
            self.results / MODEL_FILE, self.best, reference=self.reference, m=self.m,
            resolution=self.resolution, k_cap=self.k_cap, config=self.base_config(), seed=self.seed,
            val_fraction=self.val_fraction, dtype=self.dtype, omega=self.omega, delta=self.delta,
            class_margin=self.class_margin, estimate_tol=self.estimate_tol,
        )

    def retrain_selected(self, rtol: float = 1e-3) -> SweepRun:
        """Reentrena SOLO la configuracion elegida de una corrida anterior y guarda modelo.pt.

        Para resultados sin modelo.pt: la configuracion sale de `selected_config`
        (seleccion.csv o barrido.csv), sin repetir el barrido. Los datos, (c, C)
        y las C_alpha se recalculan si hace falta (son deterministas), y se
        entrena con `sweep` restringido a esa configuracion, asi que la red, la
        semilla y TrainingConfig son las del barrido. Avisa si el error de
        validacion no reproduce el reportado (diferencia relativa > rtol).
        """
        config = selected_config(self.results)
        if not hasattr(self, "train_data"):
            self.load_data()
        if not hasattr(self, "class_C"):
            self.estimate_constants()
        print(f"reentrenando solo {config['capas']}x{config['ancho']}, dropout {config['dropout_layers']}, "
              f"wd={config['weight_decay']:g} (sin barrido)")
        self.sweep(
            depths=(config["capas"],), widths=(config["ancho"],),
            dropout_layers=(config["dropout_layers"],), weight_decays=(config["weight_decay"],),
        )
        self.best = best_run(self.runs)
        reportado, obtenido = config["val_rel_l2"], self.best.val_rel_l2
        diferencia = abs(obtenido - reportado) / reportado
        print(f"val_rel_l2 reportado = {reportado:.6e}, reentrenado = {obtenido:.6e} "
              f"(diferencia relativa {diferencia:.2e})")
        if diferencia > rtol:
            warnings.warn(
                f"{self.results}: el reentrenamiento no reproduce el error de validacion reportado "
                f"({obtenido:.6e} frente a {reportado:.6e}); los resultados guardados pueden venir "
                f"de otros datos o de otra version del codigo."
            )
        self.save_model()
        return self.best

    def metrics(self):
        """Metricas de trayectoria (muestras) y del simbolo (k de entrenamiento / validacion)."""
        traj = {"entrenamiento": trajectory_report(self.symbol, self.train_data),
                "validacion": trajectory_report(self.symbol, self.val_data)}
        k_sets = {name: np.unique(split["k"].numpy(), axis=0)
                  for name, split in (("entrenamiento", self.train_data), ("validacion", self.val_data))}
        sym = {name: symbol_report(self.symbol, self.reference, k) for name, k in k_sets.items()}
        n_muestras = {name: int(rep.pop("n")) for name, rep in traj.items()}
        n_k = {name: int(rep.pop("n")) for name, rep in sym.items()}
        print(f"TRAYECTORIA -- muestras: {n_muestras['entrenamiento']:,} entrenamiento / "
              f"{n_muestras['validacion']:,} validacion; SIMBOLO -- k unicos: "
              f"{n_k['entrenamiento']} entrenamiento / {n_k['validacion']} validacion")

        traj, sym = pd.DataFrame(traj), pd.DataFrame(sym)
        traj.to_csv(self.results / "metricas_trayectoria.csv")
        sym.to_csv(self.results / "metricas_simbolo.csv")
        return traj, sym

    # ------------------------------------------------------------------
    # 6. Prueba
    # ------------------------------------------------------------------
    def test(self) -> pd.DataFrame:
        """Error en prueba: pares (k, t) del split y espacio fisico; figura del error vs t."""
        test_traj = trajectory_report(self.symbol, self.grf_test.split)
        test_traj.pop("n")
        print(f"PRUEBA -- {self.grf_test.split['k'].shape[0]:,} pares (k, t), "
              f"{self.grf_test.u0_hat.shape[0]} condiciones iniciales nuevas")
        err_field = field_rel_l2(self.symbol, self.grf_test)   # (n_ic, T)
        err_train = field_rel_l2(self.symbol, self.grf_train)

        print(f"error relativo L2 (pares (k, t))        = {test_traj['rel_l2']:.4e}")
        print(f"error relativo L2 (espacio fisico)     = {err_field.mean():.4e}  "
              f"(medio; max {err_field.max():.4e})")
        self.summary.update(test_rel_l2=test_traj["rel_l2"], test_rel_l2_fisico=float(err_field.mean()))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(self.grf_test.t, err_field.mean(axis=0), "o-", label="prueba (media)")
        ax.fill_between(self.grf_test.t, err_field.min(axis=0), err_field.max(axis=0), alpha=0.2)
        ax.scatter(self.grf_train.t, err_train.mean(axis=0), color="k", marker="x",
                   zorder=3, label="entrenamiento (media)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"error relativo $L^2$ en espacio fisico")
        ax.set_title("Error en prueba vs tiempo")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "error_prueba.png")

        tabla = pd.DataFrame({"prueba": test_traj})
        tabla.loc["rel_l2_fisico_medio"] = float(err_field.mean())
        tabla.loc["rel_l2_fisico_max"] = float(err_field.max())
        tabla.to_csv(self.results / "prueba.csv")
        return tabla

    # ------------------------------------------------------------------
    # 7. Curvas de entrenamiento
    # ------------------------------------------------------------------
    def curves(self) -> None:
        """Perdida total y error relativo de la trayectoria por epoca."""
        history = self.best.result.history
        epochs = np.arange(1, len(history["train"]["total"]) + 1)
        for key, ylabel, title, name, log in (
            ("total", "perdida total", "Perdida vs epoca", "perdida.png", True),
            ("rel_l2", r"error relativo $L^2$ de la trayectoria", "Metrica vs epoca", "rel_l2.png", False),
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            plot = ax.semilogy if log else ax.plot
            plot(epochs, history["train"][key], label="entrenamiento")
            plot(epochs, history["val"][key], label="validacion")
            ax.axvline(self.best.best_epoch, color="k", ls=":", lw=1, label="mejor epoca")
            ax.set_xlabel("epoca")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            self._save(fig, name)

    # ------------------------------------------------------------------
    # 8. Simbolo aprendido vs referencia
    # ------------------------------------------------------------------
    def symbol_figure(self) -> None:
        """a_theta frente a la referencia en |k_i| <= k_max.

        1D: simbolo vs |k|, paridad y error absoluto vs k. 2D: simbolo vs |k|,
        paridad, error absoluto vs |k| y mapa del error relativo sobre (k1, k2).
        """
        k_arr = k_grid(self.n, self.k_max)
        a_ref = symbol_values(self.reference, k_arr)
        a_theta = np.real(self.symbol.evaluate_batch(k_arr))
        norm_k = np.linalg.norm(k_arr, axis=1)
        err = np.abs(a_theta - a_ref)
        rel = relative_symbol_error(a_theta, a_ref)

        fig, axes = plt.subplots(1, 3 if self.n == 1 else 4, figsize=(16 if self.n == 1 else 21, 4.5))

        # (a) La curva del simbolo contra |k| (en 2D el simbolo es radial).
        order = np.argsort(norm_k)
        axes[0].plot(norm_k[order], a_ref[order], "k-", lw=1.5, label="simbolo analitico")
        axes[0].scatter(norm_k, a_theta, s=10 if self.n == 1 else 3, color="C1", label="simbolo aprendido")
        axes[0].set_xlabel(r"$|k|$")
        axes[0].set_ylabel(r"$a(k)$")
        axes[0].set_title(r"Simbolo vs $|k|$")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # (b) Paridad: sobre la diagonal significa que a_theta acerto.
        lo, hi = min(a_ref.min(), a_theta.min()), max(a_ref.max(), a_theta.max())
        pad = 0.05 * (hi - lo)
        lims = [lo - pad, hi + pad]
        axes[1].plot(lims, lims, "k--", lw=1, label=r"$a_\theta = a_{\text{ref}}$")
        axes[1].scatter(a_ref, a_theta, s=10 if self.n == 1 else 3, color="C1")
        axes[1].set_xlim(lims)
        axes[1].set_ylim(lims)
        axes[1].set_aspect("equal")
        axes[1].set_xlabel(r"simbolo analitico $a_{\mathrm{ref}}(k)$")
        axes[1].set_ylabel(r"simbolo aprendido $a_\theta(k)$")
        axes[1].set_title("Paridad")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # (c) Error absoluto: contra k en 1D, contra |k| en 2D.
        if self.n == 1:
            axes[2].plot(k_arr[:, 0], err, "o-", color="C1", ms=4)
            axes[2].set_xlabel(r"$k$")
        else:
            axes[2].scatter(norm_k, err, s=3, color="C1")
            axes[2].set_xlabel(r"$|k|$")
        axes[2].set_title(rf"$|a_\theta(k) - |k|^{{{self.reference.m:g}}}|$")
        axes[2].grid(alpha=0.3)

        # (d) 2D: error relativo sobre (k1, k2).
        if self.n == 2:
            side = 2 * self.k_max + 1
            ext = [-self.k_max - 0.5, self.k_max + 0.5] * 2
            im = axes[3].imshow(rel.reshape(side, side).T, origin="lower", extent=ext)
            axes[3].set_title(r"$|a_\theta(k) - a(k)|\,/\,a(k)$")
            axes[3].set_xlabel(r"$k_1$")
            axes[3].set_ylabel(r"$k_2$")
            fig.colorbar(im, ax=axes[3])
        self._save(fig, "simbolo.png")

        print(f"error relativo medio = {rel.mean():.3f}, maximo = {rel.max():.3f}  "
              f"(|k_i| <= {self.k_max}, k = 0 excluido del cociente)")
        self.summary.update(error_simbolo_medio=float(rel.mean()), error_simbolo_max=float(rel.max()))

    # ------------------------------------------------------------------
    # 9. Certificacion de parabolicidad
    # ------------------------------------------------------------------
    def certification(self) -> pd.DataFrame:
        """Peor ratio de la Def. 2.3 en CERT_LAM x {|k_i| <= K}, dentro y fuera de la ventana."""
        verifier = ParabolicityVerifier(self.symbol)
        kappa = self.best.kappa
        filas = []
        for window in (self.k_max, 2 * self.k_max, self.cert_k_max):
            k_values = k_grid(self.n, window)
            # max ratio = kappa_hat / kappa, con kappa_hat el sup de |(a+lambda)^{-1}| (1+|(k,lambda)|^2)^{m/2}
            ratio = verifier.estimate_kappa_batch(k_values, CERT_LAM, omega=self.omega) / kappa
            filas.append({
                "K": window,
                "region": "dentro" if window <= self.k_max else "fuera",
                "puntos": len(k_values) * len(CERT_LAM),
                "max_ratio": ratio,
                "satisfecha": ratio <= 1.0,
            })
        tabla = pd.DataFrame(filas)
        tabla.to_csv(self.results / "certificacion.csv", index=False)

        peor = float(tabla["max_ratio"].max())
        print(f"{self.symbol!r}")
        print(f"kappa usada = {kappa:.4f}")
        print(f"rejilla de certificacion: |k_i| <= K x {len(CERT_LAM)} lambda "
              f"(|lambda| en [1e-2, 1e3], Re(lambda) >= 0)")
        print(f"kappa minimo que certificaria |k_i| <= {self.cert_k_max}: {kappa * tabla['max_ratio'].iloc[-1]:.4f}")
        self.summary.update(peor_ratio=peor)
        return tabla

    # ------------------------------------------------------------------
    # 10. Clase S^{m,rho}
    # ------------------------------------------------------------------
    def symbol_class(self) -> pd.DataFrame:
        """Evidencia de pertenencia a S^{m,rho} en ventanas crecientes (SymbolClassVerifier)."""
        k_maxs = tuple(sorted({4, 8, 12, 2 * self.k_max}))
        resultados = SymbolClassVerifier(self.symbol).check(k_maxs=k_maxs)
        clase = pd.DataFrame([
            {
                "alpha": str(r.alpha),
                "|alpha|": r.order,
                **{f"C_hat({K})": r.c_hat[K] for K in k_maxs},
                f"crecim.({k_maxs[-2]}->{k_maxs[-1]})": r.growth,
                "pendiente": r.slope,
                "m-|alpha|": r.slope_theoretical,
                "estable": r.stable,
                "consistente": r.consistent,
            }
            for r in resultados
        ])
        clase.to_csv(self.results / "clase.csv", index=False)

        ok = int(clase["consistente"].sum())
        print(f"consistentes: {ok} de {len(clase)} multiindices (rho = {self.symbol.rho}, m = {self.symbol.m:g})")
        print(f"ventana explorada: |k_i| <= {k_maxs[-1]} (evidencia, no prueba sobre todo Z^n)")
        print("pendiente = nan significa Delta^alpha a = 0 en la ventana, asi que C_alpha = 0 basta")
        self.summary.update(multiindices_consistentes=f"{ok}/{len(clase)}")
        return clase

    def summary_row(self) -> Dict:
        """Una fila de la tabla resumen, con lo que fueron guardando las secciones."""
        return dict(self.summary)


# Columnas de la tabla resumen de los notebooks con varios alpha (04 y 05).
SUMMARY_COLS = [
    "alpha", "ventana_k", "c", "C", "kappa", "configuracion", "val_rel_l2", "test_rel_l2",
    "test_rel_l2_fisico", "error_simbolo_medio", "peor_ratio", "multiindices_consistentes",
]


def summary_table(experiments: Dict[float, GRFExperiment], results: Path) -> pd.DataFrame:
    """Una fila por alpha con lo que resumio cada experimento; se guarda en results/resumen.csv."""
    tabla = pd.DataFrame(
        [{"alpha": alpha, **exp.summary_row()} for alpha, exp in experiments.items()]
    )[SUMMARY_COLS]
    tabla.to_csv(Path(results) / "resumen.csv", index=False)
    return tabla

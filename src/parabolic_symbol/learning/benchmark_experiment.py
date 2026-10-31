"""Benchmark PDNO frente a FNO, seccion por seccion, como en 10_benchmark_calor_1d.ipynb.

`BenchmarkExperiment` reune lo que repiten los notebooks 11-13 (calor 2D y
laplaciano fraccionario en 1D y 2D), igual que `PDNOExperiment` para 07-09. Son
las secciones del 10 sin la comparacion con el paper ni la Figura 3, con una
diferencia: el PDNO no se reentrena. Se carga el modelo elegido en el notebook
PDNO correspondiente (07, 08 o 09), se comprueba que da el error de prueba que
ese notebook guardo y sus curvas se leen de su historia.csv.

Datos: los del notebook PDNO (convencion del proyecto, misma referencia,
resolucion, k_cap, semilla y fraccion de validacion), mas la extrapolacion
temporal. Como en el 10, los campos exactos se generan en float64 (numpy) y se
pasan a `dtype` (float32) antes de que los vea cualquier modelo; los float64 solo
se usan para remuestrear en la seccion de invariancia de resolucion.

FNO: `run_fno_sweep` con la OperatorTrainingConfig del notebook PDNO y weight
decay 0 (la de 10), seleccion por menor L2 relativa de validacion, y
`save_fno_model` con la comprobacion de recarga.

Cada grafica es una imagen PNG aparte. En 2D los campos van con imshow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..symbol import Symbol
from ..data.fields import FieldDatasets, field_datasets, resample_fields
from ..data.grf import TEST_TIMES, TRAIN_TIMES
from ..operators.fno_baseline import FNOConfig
from ..operators.pdno import PDNO
from .baseline_training import FNOSweepRun, fno_sweep_table, load_fno_model, run_fno_sweep, save_fno_model
from .estimation import Bounds
from .experiment import SWEEP_COLS, _dtype_name, _reference_spec
from .operator_evaluation import field_errors
from .operator_experiment import HISTORY_FILE, MODEL_FILE, PDNOExperiment
from .operator_training import OperatorTrainingConfig, load_operator_checkpoint, load_operator_model
from .sweep import best_run
from ..utils import default_device

FNO_MODEL_FILE = "modelo_fno.pt"
MODEL_COLORS = {"PDNO": "C0", "FNO": "C1"}

# Campos de OperatorTrainingConfig que pueden diferir entre la corrida guardada del PDNO y TRAIN_CONFIG:
# el weight decay es un eje del barrido y log_every solo cambia lo que se imprime.
_IGNORED_FIELDS = ("weight_decay", "log_every")


@dataclass
class BenchmarkExperiment:
    """Benchmark PDNO (cargado) frente a FNO (barrido) para un simbolo de referencia.

    `pdno_results` es la carpeta del notebook PDNO con modelo.pt, prueba.csv,
    barrido.csv e historia.csv; `pdno_notebook`, el notebook que hay que correr
    si falta (p. ej. "07_pdno_calor_2d.ipynb, seccion alpha = 0.5"). `train_config` es
    la TRAIN_CONFIG de ese notebook; el FNO la usa con weight decay 0.
    `fno_sweep` son los ejes de `run_fno_sweep` (n_modes por eje).
    `resolution_eval_batch` es el lote de evaluacion de la seccion de invariancia
    de resolucion (en 2D, 256 x 256 con lotes de 64 no cabe en 6 GB); por defecto el
    de `train_config`.
    """

    reference: Symbol
    m: float
    resolution: int
    k_cap: int
    results: Path
    pdno_results: Path
    pdno_notebook: str
    train_config: OperatorTrainingConfig
    dtype: torch.dtype
    fno_base: FNOConfig
    fno_sweep: Dict
    extrapolation_times: np.ndarray
    eval_resolutions: Sequence[int]
    device: Optional[torch.device] = None
    seed: int = 0
    val_fraction: float = 0.2
    resolution_eval_batch: Optional[int] = None

    # Se llenan seccion a seccion.
    ds64: FieldDatasets = field(init=False, repr=False)
    ds: FieldDatasets = field(init=False, repr=False)
    pdno: PDNOExperiment = field(init=False, repr=False)
    checkpoint_pdno: Dict = field(init=False, repr=False)
    runs_fno: list = field(init=False, repr=False)
    best_fno: FNOSweepRun = field(init=False, repr=False)
    err_test: Dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.results = Path(self.results)
        self.results.mkdir(parents=True, exist_ok=True)
        self.pdno_results = Path(self.pdno_results)
        self.device = torch.device(self.device) if self.device is not None else default_device()
        if self.resolution_eval_batch is None:
            self.resolution_eval_batch = self.train_config.eval_batch_size
        self.fno_train_config = replace(self.train_config, weight_decay=0.0)
        self.err_test = {}
        # Las secciones que solo miran el PDNO (condiciones iniciales, simbolo, certificacion) son las
        # de PDNOExperiment, con el modelo guardado y las figuras en la carpeta del benchmark.
        self.pdno = PDNOExperiment(
            reference=self.reference, m=self.m, resolution=self.resolution, k_cap=self.k_cap,
            results=self.results, train_config=self.train_config, dtype=self.dtype, device=self.device,
            seed=self.seed, val_fraction=self.val_fraction, extrapolation_times=self.extrapolation_times,
        )

    @property
    def n(self) -> int:
        return self.reference.n

    @property
    def k_max(self) -> int:
        return self.k_cap

    @property
    def model_pdno(self) -> PDNO:
        return self.pdno.model

    @property
    def model_fno(self):
        return self.best_fno.model

    @property
    def models(self) -> Dict:
        return {"PDNO": self.model_pdno, "FNO": self.model_fno}

    @property
    def _modes_label(self) -> str:
        return "|k|" if self.n == 1 else "|k_i|"

    def _save(self, fig, name: str) -> None:
        plt.tight_layout()
        fig.savefig(self.results / name, dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # 1. Datos
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Los conjuntos del notebook PDNO mas la extrapolacion, en float64 (`ds64`) y en `dtype` (`ds`)."""
        self.ds64 = field_datasets(
            self.reference, self.resolution, self.seed, k_cap=self.k_cap, val_fraction=self.val_fraction,
            extrapolation_times=self.extrapolation_times,
        )
        self.ds = self.ds64.to("cpu", self.dtype)      # lo que ven los modelos y la evaluacion
        self.pdno.ds = self.ds
        ds = self.ds
        conjuntos = {"entrenamiento": ds.train, "validacion": ds.val, "prueba": ds.test,
                     "extrapolacion": ds.extrapolation}
        assert np.allclose(ds.train.t, TRAIN_TIMES) and np.allclose(ds.test.t, TEST_TIMES)

        # Disjuntos: entrenamiento / validacion por indice; prueba frente a ambos por distancia entre campos u0.
        assert set(ds.train.ic).isdisjoint(ds.val.ic)
        vistas = torch.cat([ds.train.u0, ds.val.u0]).flatten(1)
        prueba = ds.test.u0.flatten(1)
        distancia = torch.cdist(prueba, vistas).min() / torch.linalg.vector_norm(prueba, dim=1).min()
        assert distancia > 0

        print()
        print(f"k_max = k_cap = {self.k_max}")
        for nombre, data in conjuntos.items():
            print(f"{nombre:14s}: {data.n_ic:3d} condiciones iniciales x {data.n_times:2d} tiempos = "
                  f"{data.n_pairs:4d} pares")
        print(f"prueba: min ||u0_prueba - u0_vista|| / ||u0_prueba|| = {float(distancia):.3f}  (> 0: disjunta)")
        print(f"dtype de los campos: {ds.train.u.dtype}, de las filas: {ds.rows_train['u0'].dtype}")

        tabla = pd.DataFrame([
            {"conjunto": nombre, "condiciones_iniciales": data.n_ic, "tiempos": data.n_times,
             "pares": data.n_pairs, "t_min": float(data.t.min()), "t_max": float(data.t.max())}
            for nombre, data in conjuntos.items()
        ]).set_index("conjunto")
        tabla.to_csv(self.results / "datos.csv")
        return tabla

    def initial_conditions(self) -> None:
        """Dos u0 de entrenamiento y la solucion exacta en algunos tiempos (como en 07-09)."""
        self.pdno.initial_conditions()

    # ------------------------------------------------------------------
    # 2. PDNO guardado
    # ------------------------------------------------------------------
    def load_pdno(self):
        """Carga el PDNO elegido en el notebook PDNO y comprueba datos, configuracion y error de prueba.

        Devuelve (resumen del modelo, tabla de C_alpha usadas). Falla si falta modelo.pt (con el
        notebook que hay que correr), si el modelo es de otros datos, otra precision u otra
        TRAIN_CONFIG (salvo weight decay y log_every), o si su error de prueba no es el de prueba.csv.
        """
        path = self.pdno_results / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"no existe {path}: corre primero notebooks/{self.pdno_notebook} (el benchmark no reentrena el PDNO)"
            )
        checkpoint = load_operator_checkpoint(path)
        datos = checkpoint["datos"]
        esperado = {
            "referencia": _reference_spec(self.reference), "resolucion": self.resolution, "k_cap": self.k_cap,
            "semilla": self.seed, "val_fraction": self.val_fraction,
        }
        distintos = [f"{clave}: {datos[clave]!r} (guardado) != {valor!r}"
                     for clave, valor in esperado.items() if datos[clave] != valor]
        if checkpoint["dtype"] != _dtype_name(self.dtype):
            distintos.append(f"dtype: {checkpoint['dtype']} (guardado) != {_dtype_name(self.dtype)}")
        entrenamiento = asdict(replace(self.train_config, class_reference=None))
        distintos += [f"TRAIN_CONFIG.{clave}: {checkpoint['entrenamiento'][clave]!r} (guardado) != {valor!r}"
                      for clave, valor in entrenamiento.items()
                      if clave not in _IGNORED_FIELDS and checkpoint["entrenamiento"][clave] != valor]
        if distintos:
            raise ValueError(f"{path} no es de este benchmark:\n  " + "\n  ".join(distintos))

        model = load_operator_model(path, device=self.device)
        arch = checkpoint["arquitectura"]
        self.pdno.use_saved_model(model, Bounds(c=arch["c"], C=arch["C"], window=None))
        self.checkpoint_pdno = checkpoint

        err = field_errors(model, self.ds.test, self.train_config.eval_batch_size)
        guardado = pd.read_csv(self.pdno_results / "prueba.csv", index_col=0).loc["global"]
        coincide = bool(np.isclose(err.mean(), float(guardado["media"]), rtol=1e-6, atol=0.0))
        if not coincide:
            raise ValueError(
                f"el PDNO de {path} da un error de prueba {err.mean():.6e}, no el de prueba.csv "
                f"({float(guardado['media']):.6e}); vuelve a correr notebooks/{self.pdno_notebook}"
            )
        self.err_test["PDNO"] = err

        resultado = checkpoint["resultado"]
        print(f"PDNO cargado de {path}")
        capas_dropout = arch["symbol_dropout_layers"]
        dropout = (f"dropout {arch['symbol_dropout']:g} en las capas {capas_dropout}"
                   if capas_dropout and arch["symbol_dropout"] > 0 else "sin dropout")
        print(f"MLP del simbolo        = {arch['symbol_hidden']}, {dropout}, "
              f"weight decay {checkpoint['entrenamiento']['weight_decay']:g}")
        print(f"(c, C) = ({arch['c']:.4f}, {arch['C']:.4f}), m = {arch['m']:g}, "
              f"kappa = theoretical_kappa() = {model.theoretical_kappa(self.train_config.omega):.4f}")
        print(f"mejor epoca            = {resultado['mejor_epoca']} de {resultado['epocas']}")
        print(f"error de prueba        = {err.mean():.6e} (recalculado) | {float(guardado['media']):.6e} "
              f"(prueba.csv) | coincide: {coincide}")

        resumen = pd.Series({
            "c": arch["c"], "C": arch["C"], "m": arch["m"],
            "kappa": model.theoretical_kappa(self.train_config.omega),
            "mlp_simbolo": str(arch["symbol_hidden"]), "weight_decay": checkpoint["entrenamiento"]["weight_decay"],
            "parametros": sum(p.numel() for p in model.parameters()),
            "mejor_epoca": resultado["mejor_epoca"], "epocas": resultado["epocas"],
            "rel_l2_prueba (recalculado)": float(err.mean()), "rel_l2_prueba (prueba.csv)": float(guardado["media"]),
        }, name="PDNO guardado")
        resumen.to_csv(self.results / "pdno.csv")
        class_C = pd.Series({str(alpha): c for alpha, c in resultado["class_C"].items()},
                            name="C_alpha usadas (x%g)" % self.train_config.class_margin)
        class_C.index.name = "alpha"
        class_C.to_csv(self.results / "class_C.csv")
        return resumen, class_C.to_frame()

    # ------------------------------------------------------------------
    # 3. Barridos y seleccion
    # ------------------------------------------------------------------
    def pdno_sweep(self) -> pd.DataFrame:
        """El barrido del PDNO del notebook de origen (barrido.csv), de mejor a peor val_rel_l2."""
        tabla = pd.read_csv(self.pdno_results / "barrido.csv")
        tabla.to_csv(self.results / "barrido_pdno.csv", index=False)
        print(f"barrido del PDNO de {self.pdno_results / 'barrido.csv'}: {len(tabla)} corridas, "
              f"{tabla['segundos'].sum() / 60:.1f} min de entrenamiento en total")
        return tabla[SWEEP_COLS].sort_values("val_rel_l2").reset_index(drop=True)

    def sweep_fno(self) -> None:
        """El barrido del FNO (`run_fno_sweep`) con la TRAIN_CONFIG del PDNO y weight decay 0."""
        self.runs_fno = run_fno_sweep(
            self.ds.train, self.ds.val, replace(self.fno_base, n=self.n), self.fno_train_config,
            device=self.device, **self.fno_sweep,
        )
        tabla = fno_sweep_table(self.runs_fno)
        tabla.to_csv(self.results / "barrido_fno.csv", index=False)
        print(f"{len(tabla)} corridas, {tabla['segundos'].sum() / 60:.1f} min de entrenamiento en total")

    def sweep_fno_results(self):
        """Todas las corridas del FNO (de mejor a peor val_rel_l2) y la mejor por eje del barrido."""
        tabla = fno_sweep_table(self.runs_fno)
        ordenada = tabla.sort_values("val_rel_l2").reset_index(drop=True)
        por_eje = [
            tabla.groupby(eje).agg(corridas=("val_rel_l2", "size"), mejor_val_rel_l2=("val_rel_l2", "min"))
            for eje in ("n_modes", "hidden_channels", "capas")
        ]
        return ordenada, por_eje

    def select_fno(self) -> pd.Series:
        """La corrida del FNO de menor L2 relativa de validacion."""
        self.best_fno = best_run(self.runs_fno)
        best = self.best_fno
        print(f"arquitectura elegida    = n_modes {best.n_modes} ({self._modes_label} <= {best.n_modes // 2}), "
              f"hidden_channels {best.hidden_channels}, {best.n_layers} capas")
        print(f"parametros              = {best.n_parameters:,} (reales; un peso complejo cuenta como dos)")
        print(f"early stopping          = mejor epoca {best.best_epoch} de {best.epochs_run} epocas; "
              f"se detuvo antes del final: {best.stopped_early}")
        print(f"rel_l2 de validacion    = {best.val_rel_l2:.4f}")
        return pd.Series({
            "n_modes": best.n_modes, "hidden_channels": best.hidden_channels, "capas": best.n_layers,
            "parametros": best.n_parameters, "mejor_epoca": best.best_epoch, "epocas": best.epochs_run,
            "val_rel_l2": best.val_rel_l2,
        }, name="FNO elegido")

    def _curve(self, train, val, best_epoch, titulo, nombre) -> None:
        """Una curva de entrenamiento y validacion por epoca, en escala log, en su imagen."""
        train, val = np.asarray(train, dtype=float), np.asarray(val, dtype=float)
        epochs = np.arange(1, len(train) + 1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if np.all(train <= 0) and np.all(val <= 0):
            ax.text(0.5, 0.5, "= 0 en todas las epocas", ha="center", va="center", transform=ax.transAxes)
        else:
            for etiqueta, valores in (("entrenamiento", train), ("validacion", val)):
                ax.semilogy(epochs, np.where(valores > 0, valores, np.nan), label=etiqueta)
            ax.legend()
        ax.axvline(best_epoch, color="k", ls=":", lw=1)
        ax.set_xlabel("epoca")
        ax.set_title(titulo)
        ax.grid(alpha=0.3)
        self._save(fig, nombre)

    def curves(self) -> None:
        """Curvas de las dos corridas elegidas: PDNO (de historia.csv) y FNO; una imagen por termino."""
        historia_path = self.pdno_results / HISTORY_FILE
        if historia_path.exists():
            historia = pd.read_csv(historia_path, index_col=0)
            mejor = self.checkpoint_pdno["resultado"]["mejor_epoca"]
            for key, titulo, sufijo in (
                ("total", "perdida total", "total"),
                ("rel_l2", r"$L^2$ relativa", "rel_l2"),
                ("parabolicity", r"$\sum_\ell \mathcal{L}_{\mathrm{par}}$", "parabolicidad"),
                ("class", r"$\sum_\ell \mathcal{L}_{\mathrm{class}}$", "clase"),
            ):
                self._curve(historia[f"train_{key}"], historia[f"val_{key}"], mejor, f"PDNO: {titulo}",
                            f"perdida_pdno_{sufijo}.png")
        else:
            print(f"no existe {historia_path}: vuelve a correr notebooks/{self.pdno_notebook} para tener las "
                  f"curvas del PDNO (se omiten)")
        history = self.best_fno.result.history
        self._curve(history["train"]["rel_l2"], history["val"]["rel_l2"], self.best_fno.best_epoch,
                    r"FNO: $L^2$ relativa", "perdida_fno_rel_l2.png")

    # ------------------------------------------------------------------
    # 4. Prueba y 5. Extrapolacion
    # ------------------------------------------------------------------
    def _error_table(self, errs: Dict[str, np.ndarray], t: np.ndarray) -> pd.DataFrame:
        """Media y maximo por tiempo de cada modelo, y el global; cada err tiene forma (n_ic, T)."""
        visto = np.isclose(t[:, None], self.ds.train.t.numpy()[None]).any(axis=1)
        columnas = {}
        for nombre, err in errs.items():
            columnas[(nombre, "media")] = np.append(err.mean(axis=0), err.mean())
            columnas[(nombre, "max")] = np.append(err.max(axis=0), err.max())
        tabla = pd.DataFrame(columnas, index=pd.Index([f"{x:.2f}" for x in t] + ["global"], name="t"))
        tabla[("", "en_entrenamiento")] = list(visto) + [None]
        return tabla

    def _error_figure(self, errs: Dict[str, np.ndarray], t: np.ndarray, titulo: str, nombre: str) -> None:
        """Media y banda min-max por tiempo de cada modelo; los tiempos no vistos, con circulos y lineas grises."""
        visto = np.isclose(t[:, None], self.ds.train.t.numpy()[None]).any(axis=1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for tiempo in t[~visto]:
            ax.axvline(tiempo, color="0.85", lw=0.8, zorder=0)
        for nombre_modelo, err in errs.items():
            color = MODEL_COLORS[nombre_modelo]
            media = err.mean(axis=0)
            ax.semilogy(t, media, "-", color=color, lw=1, label=f"{nombre_modelo}: media")
            ax.fill_between(t, err.min(axis=0), err.max(axis=0), color=color, alpha=0.15)
            ax.scatter(t[visto], media[visto], color=color, marker="x", zorder=3)
            ax.scatter(t[~visto], media[~visto], facecolor="none", edgecolor=color, marker="o", zorder=3)
        ax.scatter([], [], color="0.3", marker="x", label="t en TRAIN_TIMES")
        ax.scatter([], [], facecolor="none", edgecolor="0.3", marker="o", label="t no visto")
        ax.set_xlabel("t")
        ax.set_ylabel(r"error relativo $L^2$ (banda: min - max)")
        ax.set_title(titulo)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        self._save(fig, nombre)

    def test(self) -> pd.DataFrame:
        """Error de prueba por condicion inicial y tiempo de los dos modelos; tabla y figura."""
        self.err_test["FNO"] = field_errors(self.model_fno, self.ds.test, self.train_config.eval_batch_size)
        t = self.ds.test.t.numpy()
        print(f"PRUEBA -- {self.ds.test.n_ic} condiciones iniciales nuevas x {self.ds.test.n_times} tiempos")
        for nombre, err in self.err_test.items():
            print(f"  {nombre}: error relativo L2 = {err.mean():.4e}  (medio; max {err.max():.4e})")
        tabla = self._error_table(self.err_test, t)
        tabla.to_csv(self.results / "prueba.csv")
        self._error_figure(self.err_test, t, "Error en prueba vs tiempo", "error_prueba.png")
        return tabla

    def field_figure(self) -> None:
        """Primera u0 de prueba: exacta, PDNO y FNO, y el error puntual de cada modelo.

        1D: campo.png (exacta continua, PDNO --, FNO -.), campo_error_pdno.png y campo_error_fno.png.
        2D: campo_{exacta,pdno,fno}_t<t>.png con una escala de color comun, y
        campo_error_{pdno,fno}_t<t>.png con otra comun a los dos modelos.
        """
        ds, i = self.ds, 0
        idx_t = [0, 4, 9, 19]                                  # t = 0.05, 0.25, 0.5, 1.0
        tiempos = ds.test.t[idx_t].numpy()
        exacta = ds.test.u[i, idx_t].numpy()
        pred = {}
        with torch.no_grad():
            for nombre, m in self.models.items():
                u0_rep = ds.test.u0[i].to(m.device, m.dtype).repeat(len(idx_t), *([1] * self.n))
                pred[nombre] = m(u0_rep, ds.test.t[idx_t].to(m.device, m.dtype)).cpu().numpy()
        ic = ds.test.ic[i]

        if self.n == 1:
            x = 2 * np.pi * np.arange(self.resolution) / self.resolution
            colores = plt.cm.viridis(np.linspace(0, 0.85, len(idx_t)))
            estilos = {"PDNO": "--", "FNO": "-."}
            fig, ax = plt.subplots(figsize=(10, 4))
            for j, color in enumerate(colores):
                ax.plot(x, exacta[j], "-", color=color, lw=1.5, label=f"t = {tiempos[j]:.2f}")
                for nombre in self.models:
                    ax.plot(x, pred[nombre][j], estilos[nombre], color=color, lw=1.2)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(t, x)$")
            ax.set_title(f"condicion inicial de prueba {ic}: exacta (continua), PDNO (--), FNO (-.)")
            ax.legend()
            ax.grid(alpha=0.3)
            self._save(fig, "campo.png")
            for nombre in self.models:
                fig, ax = plt.subplots(figsize=(10, 4))
                for j, color in enumerate(colores):
                    ax.plot(x, pred[nombre][j] - exacta[j], "-", color=color, lw=1, label=f"t = {tiempos[j]:.2f}")
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(rf"{nombre}: $u_\theta(t, x) - u(t, x)$")
                ax.set_title(f"condicion inicial de prueba {ic}: error puntual del {nombre}")
                ax.legend()
                ax.grid(alpha=0.3)
                self._save(fig, f"campo_error_{nombre.lower()}.png")
            return

        lim_campo = max(np.abs(exacta).max(), *(np.abs(p).max() for p in pred.values()))
        lim_error = max(np.abs(p - exacta).max() for p in pred.values())
        campos = [("exacta", "exacta", exacta, lim_campo)]
        campos += [(nombre.lower(), nombre, pred[nombre], lim_campo) for nombre in self.models]
        campos += [(f"error_{nombre.lower()}", f"{nombre} - exacta", pred[nombre] - exacta, lim_error)
                   for nombre in self.models]
        for sufijo, titulo, valores, lim in campos:
            for campo, t in zip(valores, tiempos):
                self.pdno._field_image(campo, f"condicion inicial de prueba {ic}: {titulo}, t = {t:.2f}",
                                       f"campo_{sufijo}_t{t:.2f}.png", lim)

    def extrapolation(self) -> pd.DataFrame:
        """Error de los dos modelos en EXTRAPOLATION_TIMES, fuera del rango de entrenamiento."""
        data = self.ds.extrapolation
        t = data.t.numpy()
        errs = {nombre: field_errors(m, data, self.train_config.eval_batch_size) for nombre, m in self.models.items()}
        print(f"EXTRAPOLACION -- {data.n_ic} condiciones iniciales x {data.n_times} tiempos "
              f"en [{t.min():g}, {t.max():g}]")
        for nombre, err in errs.items():
            print(f"  {nombre}: error relativo L2 = {err.mean():.4e}  (medio; max {err.max():.4e})")
        tabla = self._error_table(errs, t)
        tabla.to_csv(self.results / "extrapolacion.csv")
        self._error_figure(errs, t, "Error en extrapolacion temporal vs tiempo", "error_extrapolacion.png")
        return tabla

    # ------------------------------------------------------------------
    # 6. Invariancia de resolucion
    # ------------------------------------------------------------------
    def resolution_invariance(self) -> pd.DataFrame:
        """Error de prueba de los dos modelos en otras mallas (`resample_fields` desde los campos float64)."""
        fino = resample_fields(self.ds64.test, 2 * self.resolution)
        assert torch.allclose(resample_fields(fino, self.resolution).u, self.ds64.test.u, rtol=0, atol=1e-12)
        del fino

        filas = []
        for R in self.eval_resolutions:
            test_R = resample_fields(self.ds64.test, R).to("cpu", self.dtype)
            fila = {"resolucion": R}
            for nombre, m in self.models.items():
                err = field_errors(m, test_R, self.resolution_eval_batch)
                fila[f"{nombre} media"] = err.mean()
                fila[f"{nombre} max"] = err.max()
            filas.append(fila)
            del test_R
        tabla = pd.DataFrame(filas).set_index("resolucion")
        tabla.to_csv(self.results / "resolucion.csv")

        fig, ax = plt.subplots(figsize=(7, 4))
        R = np.asarray(self.eval_resolutions)
        for nombre in self.models:
            color = MODEL_COLORS[nombre]
            ax.loglog(R, tabla[f"{nombre} media"], "o-", color=color, label=f"{nombre}: media")
            ax.loglog(R, tabla[f"{nombre} max"], "s--", color=color, alpha=0.6, label=f"{nombre}: max")
        ax.axvline(self.resolution, color="k", ls=":", lw=1, label="resolucion de entrenamiento")
        ax.set_xticks(R)
        ax.set_xticklabels([str(r) if self.n == 1 else f"{r}x{r}" for r in R])
        ax.minorticks_off()
        ax.set_xlabel("puntos de la malla" if self.n == 1 else "puntos de la malla (por eje)")
        ax.set_ylabel(r"error relativo $L^2$ en prueba")
        ax.set_title("Invariancia de resolucion")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        self._save(fig, "resolucion.png")
        return tabla

    # ------------------------------------------------------------------
    # 7. Simbolo y certificacion del PDNO (las secciones de PDNOExperiment)
    # ------------------------------------------------------------------
    def symbol(self) -> pd.DataFrame:
        """Simbolo efectivo frente a la referencia, la banda y su error relativo (simbolo*.png, error_simbolo*.png)."""
        self.pdno.symbol_figure()
        return self.pdno.symbol_error()

    def certification(self) -> pd.DataFrame:
        return self.pdno.certification()

    def symbol_class(self) -> pd.DataFrame:
        return self.pdno.symbol_class()

    # ------------------------------------------------------------------
    # 8. Modelo guardado
    # ------------------------------------------------------------------
    def save_fno(self) -> Path:
        """Guarda el FNO elegido en results/modelo_fno.pt (`save_fno_model`)."""
        return save_fno_model(
            self.results / FNO_MODEL_FILE, self.model_fno, reference=self.reference, resolution=self.resolution,
            k_cap=self.k_cap, config=self.best_fno.config, result=self.best_fno.result, seed=self.seed,
            val_fraction=self.val_fraction, extrapolation_times=self.extrapolation_times,
        )

    def check_saved_fno(self) -> bool:
        """Recarga modelo_fno.pt y comprueba que da el mismo error de prueba."""
        cargado = load_fno_model(self.results / FNO_MODEL_FILE, device=self.device)
        err = field_errors(cargado, self.ds.test, self.train_config.eval_batch_size)
        identicos = bool(np.array_equal(err, self.err_test["FNO"]))
        print(f"FNO: error de prueba (cargado) = {err.mean():.6e}  |  (entrenado) = "
              f"{self.err_test['FNO'].mean():.6e}  |  identicos: {identicos}")
        return identicos

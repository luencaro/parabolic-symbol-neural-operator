"""Un experimento PDNO completo, seccion por seccion, como en 06_pdno_calor_1d.ipynb.

`PDNOExperiment` reune lo que repiten los notebooks 07-09 (calor 2D y laplaciano
fraccionario en 1D y 2D con el PDNO), igual que `GRFExperiment` para 03-05.
Cada metodo es una seccion del notebook: imprime lo mismo que la celda de 06,
guarda sus tablas (CSV) y figuras (PNG) en `results` y devuelve las tablas para
mostrarlas con `display`.

Configuracion comun (la de 06): datos de `field_datasets` con particion por
condicion inicial, banda (c, C), kappa = theoretical_kappa() y C_alpha de las
filas espectrales de las u0 de entrenamiento, PDNO en la configuracion lineal
(d = 1, sin W ni sesgos, sigma = identidad, P = Q = identidad, L = 1, tau fijo),
barrido de la MLP del simbolo (`run_operator_sweep`) y seleccion por menor L2
relativa de validacion. Todo corre en `dtype`: campos, filas, modelo,
entrenamiento, evaluacion y comprobaciones. Solo la generacion de los GRF
(numpy, `data/grf.py`) y el analisis posterior en numpy de los valores del
simbolo (banda, C_alpha, verificadores) usan la precision de numpy.

En 2D el simbolo es radial: las figuras sobre k van contra |k|, la ventana de
entrenamiento es el cuadrado |k|_inf <= k_max (dentro / fuera se distingue por
color) y se agrega un mapa del error relativo del simbolo sobre (k1, k2). La
paridad se comprueba en toda la rejilla |k_i| <= R/2 (incluidos k_2 = 0 y
k_2 < 0) y la reduccion se compara ademas con la FFT completa (fftn), que no
usa el medio espacio de rfftn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm

from ..symbol import Symbol
from ..symbols.parabolic_mlp_symbol import ParabolicMLPSymbol
from ..data.fields import FieldData, FieldDatasets, field_datasets
from ..data.grf import TEST_TIMES, TRAIN_TIMES, GRFData
from ..data.grids import k_grid
from ..operators.config import PDNOConfig
from ..operators.pdno import PDNO, EffectiveSymbol
from ..verification.parabolicity import ParabolicityVerifier
from ..verification.symbol_class import SymbolClassVerifier
from .estimation import (
    Bounds, estimate_bounds, estimate_class_constants, estimate_symbol, oracle_class_constants, symbol_values,
)
from .evaluation import symbol_report
from .experiment import CERT_LAM, SUMMARY_COLS, SWEEP, SWEEP_COLS, class_C_table, relative_symbol_error
from .figures import summary_figure
from .operator_evaluation import OPERATOR_METRICS, field_errors
from .operator_sweep import OperatorSweepRun, operator_sweep_table, run_operator_sweep
from .operator_training import (
    OperatorTrainingConfig, OperatorTrainingResult, load_operator_model, save_operator_model,
)
from .sweep import best_run
from ..utils import complex_dtype, default_device

# Configuracion lineal del PDNO (caso calor / laplaciano fraccionario), la de 06.
LINEAR_ARCH = dict(width=1, use_W=False, bias=False, activation="identity", lift="identity", project="identity")

MODEL_FILE = "modelo.pt"
HISTORY_FILE = "historia.csv"   # perdidas por epoca de la corrida elegida (seccion 5)

# Columnas de resumen.csv en 08 y 09: las de 04 / 05 (SUMMARY_COLS) y las propias del PDNO.
PDNO_SUMMARY_COLS = SUMMARY_COLS + [
    "test_rel_l2_fisico_max", "error_simbolo_medio_fuera", "paridad", "semigrupo",
]


def _fft_wavenumbers(resolution: int, n: int) -> np.ndarray:
    """k in Z^n en el layout de np.fft.fftn (todos los modos), forma (resolution ** n, n)."""
    axis = np.rint(np.fft.fftfreq(resolution, d=1.0 / resolution))
    return np.stack(np.meshgrid(*[axis] * n, indexing="ij"), axis=-1).reshape(-1, n)


@dataclass
class PDNOExperiment:
    """Experimento PDNO para un simbolo de referencia; ver el docstring del modulo.

    `m` es el orden del simbolo aprendido (2 en el calor, alpha en el
    fraccionario). `train_config` es la OperatorTrainingConfig de todas las
    corridas (con k_max = k_cap); `sweep_grid`, los ejes del barrido (SWEEP).
    `symbol_hidden` solo se usa en el modelo sin entrenar de las comprobaciones.
    `eval_batch_size` es el lote de `field_errors` en prueba, extrapolacion y
    modelo guardado (por defecto el de `train_config`).
    """

    reference: Symbol
    m: float
    resolution: int
    k_cap: int
    results: Path
    train_config: OperatorTrainingConfig
    dtype: torch.dtype
    sweep_grid: Dict = field(default_factory=lambda: dict(SWEEP))
    device: Optional[torch.device] = None
    seed: int = 0
    val_fraction: float = 0.2
    delta: float = 0.1            # margen de la banda (c, C)
    estimate_tol: float = 1e-10   # filtro de senal de estimate_symbol
    extrapolation_times: Optional[np.ndarray] = None
    arch: Dict = field(default_factory=lambda: dict(LINEAR_ARCH))
    n_layers: int = 1
    learn_tau: bool = False
    symbol_hidden: Tuple[int, ...] = (64, 64, 64)
    model_seed: int = 0
    eval_batch_size: Optional[int] = None

    # Se llenan seccion a seccion.
    ds: FieldDatasets = field(init=False, repr=False)
    bounds: Bounds = field(init=False)
    kappa: float = field(init=False)
    class_C: Dict = field(init=False, repr=False)
    class_C_counts: Dict = field(init=False, repr=False)
    class_C_oracle: Dict = field(init=False, repr=False)
    runs: list = field(init=False, repr=False)
    best: OperatorSweepRun = field(init=False, repr=False)
    err_test: np.ndarray = field(init=False, repr=False)
    summary: Dict = field(init=False, repr=False)
    saved_model: Optional[PDNO] = field(init=False, default=None, repr=False)   # ver `use_saved_model`

    def __post_init__(self) -> None:
        self.results = Path(self.results)
        self.results.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.device) if self.device is not None else default_device()
        if self.train_config.k_max != self.k_cap:
            raise ValueError(f"train_config.k_max ({self.train_config.k_max}) debe ser k_cap ({self.k_cap})")
        if self.eval_batch_size is None:
            self.eval_batch_size = self.train_config.eval_batch_size
        self.summary = {}

    @property
    def n(self) -> int:
        return self.reference.n

    @property
    def k_max(self) -> int:
        return self.k_cap

    @property
    def omega(self) -> float:
        return self.train_config.omega

    @property
    def class_margin(self) -> float:
        return self.train_config.class_margin

    @property
    def model(self) -> PDNO:
        """La corrida elegida del barrido, o el modelo de `use_saved_model`."""
        return self.saved_model if self.saved_model is not None else self.best.model

    def use_saved_model(self, model: PDNO, bounds: Bounds) -> None:
        """Usa un PDNO ya entrenado (p. ej. el de modelo.pt) en lugar de la corrida del barrido.

        Con datos (`ds`) y esto, las secciones que solo miran el modelo (condiciones
        iniciales, simbolo, error del simbolo, multiplicador, certificacion y clase) se
        pueden repetir sin barrido; asi las usan los benchmarks contra el FNO.
        """
        self.saved_model = model
        self.bounds = bounds

    @property
    def result(self) -> OperatorTrainingResult:
        return self.best.result

    @property
    def cert_k_max(self) -> int:
        """Ventana mayor de la certificacion: tres veces la de entrenamiento."""
        return 3 * self.k_max

    @property
    def _ref_label(self) -> str:
        return rf"|k|^{{{self.reference.m:g}}}"

    @property
    def _window_label(self) -> str:
        return f"|k| <= {self.k_max}" if self.n == 1 else f"|k|_inf <= {self.k_max}"

    def _save(self, fig, name: str) -> None:
        plt.tight_layout()
        fig.savefig(self.results / name, dpi=150)
        plt.show()

    def _field_image(self, campo: np.ndarray, titulo: str, nombre: str, lim: float) -> None:
        """Un campo 2D sobre [0, 2 pi)^2 con su barra de color, en su propia imagen."""
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(campo.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
        ax.set_title(titulo)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(im, ax=ax)
        self._save(fig, nombre)

    def _times(self, t) -> torch.Tensor:
        return torch.as_tensor(t, device=self.device, dtype=self.dtype)

    def _sets(self) -> Dict[str, FieldData]:
        conjuntos = {"entrenamiento": self.ds.train, "validacion": self.ds.val, "prueba": self.ds.test}
        if self.ds.extrapolation is not None:
            conjuntos["extrapolacion"] = self.ds.extrapolation   # las mismas u0 de prueba, en otros tiempos
        return conjuntos

    def model_config(self, symbol_hidden: Optional[Tuple[int, ...]] = None) -> PDNOConfig:
        """PDNOConfig fija del experimento; el barrido solo cambia la MLP del simbolo."""
        return PDNOConfig(
            n=self.n, m=self.m, c=self.bounds.c, C=self.bounds.C, n_layers=self.n_layers,
            learn_tau=self.learn_tau, symbol_hidden=symbol_hidden or self.symbol_hidden,
            seed=self.model_seed, **self.arch,
        )

    # ------------------------------------------------------------------
    # 1. Datos
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Campos de entrenamiento, validacion, prueba y (opcional) extrapolacion, en `dtype`."""
        self.ds = field_datasets(
            self.reference, self.resolution, self.seed, k_cap=self.k_cap, val_fraction=self.val_fraction,
            extrapolation_times=self.extrapolation_times, dtype=self.dtype,
        )
        ds = self.ds
        conjuntos = self._sets()
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
        print(f"entrenamiento y validacion disjuntos: {set(ds.train.ic).isdisjoint(ds.val.ic)}")
        print(f"prueba: min ||u0_prueba - u0_vista|| / ||u0_prueba|| = {float(distancia):.3f}  (> 0: disjunta)")
        print(f"filas espectrales para la banda y las C_alpha: {ds.rows_train['k'].shape[0]:,} "
              f"(solo las {ds.train.n_ic} u0 de entrenamiento)")
        print(f"dtype de los campos: {ds.train.u.dtype}, de las filas: {ds.rows_train['u0'].dtype}")
        self.summary.update(ventana_k=str(ds.grf_train.window))

        tabla = pd.DataFrame([
            {"conjunto": nombre, "condiciones_iniciales": data.n_ic, "tiempos": data.n_times,
             "pares": data.n_pairs, "t_min": float(data.t.min()), "t_max": float(data.t.max())}
            for nombre, data in conjuntos.items()
        ]).set_index("conjunto")
        tabla.to_csv(self.results / "datos.csv")
        return tabla

    def initial_conditions(self) -> None:
        """Dos u0 de entrenamiento y la solucion exacta en algunos tiempos de TRAIN_TIMES.

        Una imagen por u0 en 1D (condicion_inicial_<ic>.png) y una por u0 y tiempo en 2D
        (condicion_inicial_<ic>_u0.png, condicion_inicial_<ic>_t<t>.png), con la misma escala de color.
        """
        ds = self.ds
        x = 2 * np.pi * np.arange(self.resolution) / self.resolution
        for i in (0, 1):
            ic = ds.train.ic[i]
            if self.n == 1:
                idx_t = [0, 2, 5, 9]                               # t = 0.05, 0.25, 0.55, 0.95
                colores = plt.cm.viridis(np.linspace(0, 0.85, len(idx_t)))
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(x, ds.train.u0[i].numpy(), "k-", lw=1.5, label=r"$u_0$")
                for j, color in zip(idx_t, colores):
                    ax.plot(x, ds.train.u[i, j].numpy(), color=color, lw=1.2,
                            label=f"t = {float(ds.train.t[j]):.2f}")
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$u(t, x)$")
                ax.set_title(f"condicion inicial de entrenamiento {ic}")
                ax.grid(alpha=0.3)
                ax.legend()
                self._save(fig, f"condicion_inicial_{ic}.png")
                continue
            idx_t = [0, 5, 9]                                      # t = 0.05, 0.55, 0.95
            campos = {"u0": (ds.train.u0[i].numpy(), r"$u_0$")}
            for j in idx_t:
                t = float(ds.train.t[j])
                campos[f"t{t:.2f}"] = (ds.train.u[i, j].numpy(), f"$u(t)$, t = {t:.2f}")
            lim = max(np.abs(campo).max() for campo, _ in campos.values())
            for sufijo, (campo, titulo) in campos.items():
                self._field_image(campo, f"condicion inicial de entrenamiento {ic}: {titulo}",
                                  f"condicion_inicial_{ic}_{sufijo}.png", lim)

    # ------------------------------------------------------------------
    # 2. Banda (c, C), kappa y C_alpha
    # ------------------------------------------------------------------
    def estimate_constants(self) -> pd.DataFrame:
        """Banda (c, C), kappa teorico y C_alpha, de las filas de las u0 de entrenamiento."""
        k_est, a_hat = estimate_symbol(self.ds.rows_train, tol=self.estimate_tol)
        self.bounds = estimate_bounds(k_est, a_hat, m=self.m, delta=self.delta)
        probe = ParabolicMLPSymbol(
            n=self.n, m=self.m, c=self.bounds.c, C=self.bounds.C, device="cpu", dtype=self.dtype
        )
        self.kappa = probe.theoretical_kappa(self.omega)

        # Todos los alpha con |alpha| <= rho del simbolo aprendido (rho = n + 1), con la tolerancia de
        # cero de la precision de las filas, la misma que usa train_operator.
        self.class_C, self.class_C_counts = estimate_class_constants(
            k_est, a_hat, self.m, rho=probe.rho, margin=self.class_margin, return_counts=True,
            dtype=self.ds.rows_train["u_ref"].dtype,
        )
        self.class_C_oracle = oracle_class_constants(
            self.reference, self.k_max, margin=self.class_margin, rho=probe.rho
        )

        print(f"(c, C) de los datos = ({self.bounds.c:.4f}, {self.bounds.C:.4f})   m = {self.m:g}, "
              f"ventana: {self.bounds.window}")
        print(f"kappa teorico       = {self.kappa:.4f}   (omega = {self.omega:g})")
        print(f"C_alpha estimadas   = {len(self.class_C)} multiindices, |alpha| <= rho = {probe.rho}")
        self.summary.update(c=self.bounds.c, C=self.bounds.C, kappa=self.kappa)

        pd.Series({"c": self.bounds.c, "C": self.bounds.C, "m": self.m, "kappa_teorico": self.kappa},
                  name="valor").to_csv(self.results / "constantes.csv")
        tabla = class_C_table(self.class_C, self.class_C_oracle, self.class_C_counts, self.class_margin)
        tabla.to_csv(self.results / "class_C.csv")
        return tabla

    # ------------------------------------------------------------------
    # 3. Comprobaciones antes de entrenar
    # ------------------------------------------------------------------
    def _to_physical(self, grf: GRFData, u_hat: torch.Tensor) -> torch.Tensor:
        """`GRFData.to_physical` en torch (mismo layout, mismo dtype que u_hat)."""
        R, n = self.resolution, self.n
        u_hat = u_hat.reshape(u_hat.shape[:-1] + grf.spectral_shape)
        if n == 1:
            return R * torch.fft.ifft(u_hat, dim=-1).real
        return R**n * torch.fft.irfftn(u_hat, s=(R,) * n, dim=tuple(range(-n, 0)))

    def checks(self) -> pd.DataFrame:
        """Paridad, reduccion al modelo espectral (layout de los datos y fftn completa) y semigrupo."""
        model = PDNO(self.model_config(), device=self.device, dtype=self.dtype)
        assert model.config.is_semigroup, "PDNOExperiment usa la configuracion lineal"
        model.eval()
        print(f"tau = {model.tau.detach().cpu().numpy()}")
        R, n = self.resolution, self.n
        axes = tuple(range(-n, 0))
        cdtype = complex_dtype(self.dtype)

        # Paridad, capa por capa, en todos los k con |k_i| <= R/2 (en 2D, incluidos k_2 = 0 y k_2 < 0).
        k_par = k_grid(n, R // 2)
        rejilla = f"|k| <= {R // 2}" if n == 1 else f"|k_i| <= {R // 2}, {len(k_par):,} k"
        paridad = [
            float(np.max(np.abs(np.real(s.evaluate_batch(k_par)) - np.real(s.evaluate_batch(-k_par)))))
            for s in model.symbols
        ]
        for l, valor in enumerate(paridad):
            print(f"capa {l}: max |a(k) - a(-k)| = {valor:.3e}   ({rejilla})")

        def max_rel(pred: torch.Tensor, exacta: torch.Tensor) -> float:
            # max_x |u_theta - u| / max_x |u|, peor muestra.
            dif = torch.abs(pred - exacta).flatten(1).max(dim=1).values
            return float((dif / torch.abs(exacta).flatten(1).max(dim=1).values).max())

        # Reduccion: PDNO frente a la solucion espectral con el mismo simbolo, (a) en el layout de
        # GRFData (np.fft en 1D, rfftn en 2D) y (b) con la FFT completa, que no usa el medio espacio.
        n_check = 4
        t_check = [0.05, 0.5, 1.0]
        grf = self.ds.grf_test
        u0_check = self.ds.test.u0[:n_check].to(self.device, self.dtype)
        u0_hat = torch.as_tensor(grf.u0_hat[self.ds.test.ic[:n_check]]).to(self.device, cdtype)
        a_data = self._times(model.effective_symbol(grf.k))
        a_full = self._times(model.effective_symbol(_fft_wavenumbers(R, n))).reshape((R,) * n)
        u0_full_hat = torch.fft.fftn(u0_check, dim=axes)
        reduccion = reduccion_fftn = 0.0
        with torch.no_grad():
            for t in t_check:
                pred = model(u0_check, torch.full((n_check,), t, device=self.device, dtype=self.dtype))
                exacta = self._to_physical(grf, torch.exp(-t * a_data)[None] * u0_hat)
                reduccion = max(reduccion, max_rel(pred, exacta))
                completa = torch.fft.ifftn(torch.exp(-t * a_full)[None] * u0_full_hat, dim=axes)
                reduccion_fftn = max(reduccion_fftn, max_rel(pred.to(cdtype), completa))

        # Semigrupo: S(t+s) u0 frente a S(t) S(s) u0.
        pares_ts = [(0.05, 0.05), (0.1, 0.25), (0.25, 0.5), (0.5, 0.45), (0.3, 0.7)]
        semigrupo = 0.0

        def S(u, t):
            return model(u, torch.full((u.shape[0],), t, device=self.device, dtype=self.dtype))

        with torch.no_grad():
            norma_u0 = torch.linalg.vector_norm(u0_check.flatten(1), dim=1)
            for t, s in pares_ts:
                dif = torch.linalg.vector_norm((S(u0_check, t + s) - S(S(u0_check, s), t)).flatten(1), dim=1)
                semigrupo = max(semigrupo, float((dif / norma_u0).max()))

        self.summary.update(paridad=max(paridad), semigrupo=semigrupo)
        esperado = f"~{10 * torch.finfo(self.dtype).eps:.0e}"   # unas pocas unidades de redondeo de dtype
        tabla = pd.DataFrame({
            "valor": [max(paridad), reduccion, reduccion_fftn, semigrupo],
            "esperado": ["0", esperado, esperado, esperado],
        }, index=pd.Index([
            f"paridad: max |a(k) - a(-k)|, {rejilla}",
            f"reduccion: max error relativo ({n_check} u0 de prueba, t = {t_check})",
            f"reduccion (fftn completa): max error relativo ({n_check} u0 de prueba, t = {t_check})",
            f"semigrupo: max ||S(t+s)u0 - S(t)S(s)u0|| / ||u0|| ({len(pares_ts)} pares)",
        ], name="comprobacion"))
        tabla.to_csv(self.results / "comprobaciones.csv")
        return tabla

    # ------------------------------------------------------------------
    # 4. Barrido y seleccion
    # ------------------------------------------------------------------
    def sweep(self, **overrides) -> None:
        """El barrido de `sweep_grid` con la arquitectura fija del PDNO; guarda barrido.csv.

        `overrides` reemplaza ejes de `sweep_grid` (p. ej. para una prueba rapida).
        """
        grid = {**self.sweep_grid, **overrides}
        self.runs = run_operator_sweep(
            self.ds.train, self.ds.val, self.model_config(), self.train_config, rows=self.ds.rows_train,
            device=self.device, dtype=self.dtype, **grid,
        )
        table = operator_sweep_table(self.runs)
        table.to_csv(self.results / "barrido.csv", index=False)
        print(f"{len(table)} corridas, {table['segundos'].sum() / 60:.1f} min de entrenamiento en total")

    def sweep_results(self):
        """Todas las corridas (de mejor a peor val_rel_l2) y resumen por regularizacion."""
        table = operator_sweep_table(self.runs)
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

    def select(self) -> pd.Series:
        """La corrida de menor L2 relativa de validacion y sus metricas en la mejor epoca."""
        self.best = best_run(self.runs)
        best, model, result = self.best, self.model, self.result

        print(f"kappa = theoretical_kappa() = {model.theoretical_kappa(self.omega):.4f}")
        print(f"arquitectura elegida    = {best.n_layers} capas x {best.width} neuronas (MLP del simbolo)")
        print(f"dropout                 = {best.dropout_desc}")
        print(f"weight decay            = {best.weight_decay:g}  (Linear ocultas interiores)")
        print(f"early stopping          = mejor epoca {best.best_epoch} de {best.epochs_run} epocas; "
              f"se detuvo antes del final: {best.stopped_early}")
        print(f"tau                     = {model.tau.detach().cpu().numpy()}")
        print(f"C_alpha de train_operator = las de class_C.csv: {result.class_C == self.class_C}")
        print()
        print("metricas de validacion en la mejor epoca:")
        print(f"  total             = {best.val_total:.4e}")
        print(f"  rel_l2            = {best.val_rel_l2:.4f}")
        print(f"  parabolicidad     = {best.val_parabolicity:.4e}   (0 => ningun punto viola la cota)")
        print(f"  clase             = {best.val_class:.4e}")
        print(f"  peor ratio        = {best.val_par_max_ratio:.4f}")
        print(f"  puntos con >1     = {100 * best.val_par_violation:.2f}%")

        self.summary.update(
            configuracion=f"{best.n_layers}x{best.width}, dropout {best.dropout_desc}, wd={best.weight_decay:g}",
            val_rel_l2=best.val_rel_l2,
        )
        val = pd.Series({
            "total": best.val_total,
            "rel_l2": best.val_rel_l2,
            "parabolicidad": best.val_parabolicity,
            "clase": best.val_class,
            "peor ratio": best.val_par_max_ratio,
            "puntos con ratio > 1": best.val_par_violation,
        }, name="validacion (mejor epoca)")
        configuracion = pd.Series({
            "capas": best.n_layers, "ancho": best.width, "dropout_layers": best.dropout_layers,
            "weight_decay": best.weight_decay, "mejor_epoca": best.best_epoch, "epocas": best.epochs_run,
        }, name=val.name)
        pd.concat([val.astype(object), configuracion]).to_csv(self.results / "seleccion.csv")
        return val

    # ------------------------------------------------------------------
    # 5. Curvas de entrenamiento
    # ------------------------------------------------------------------
    def curves(self) -> None:
        """Perdida total y cada termino por epoca, en entrenamiento y validacion; una imagen por termino.

        Guarda ademas historia.csv (una fila por epoca, columnas train_<metrica> y val_<metrica>), que
        leen los benchmarks contra el FNO sin reentrenar el PDNO.
        """
        history = self.result.history
        epochs = np.arange(1, len(history["train"]["total"]) + 1)
        historia = pd.DataFrame(
            {f"{split}_{key}": history[split][key] for split in ("train", "val") for key in OPERATOR_METRICS},
            index=pd.Index(epochs, name="epoca"),
        )
        historia.to_csv(self.results / HISTORY_FILE)
        paneles = [
            ("total", "perdida total", "perdida_total.png"),
            ("rel_l2", r"$L^2$ relativa", "perdida_rel_l2.png"),
            ("parabolicity", r"$\sum_\ell \mathcal{L}_{\mathrm{par}}$", "perdida_parabolicidad.png"),
            ("class", r"$\sum_\ell \mathcal{L}_{\mathrm{class}}$", "perdida_clase.png"),
        ]
        for key, titulo, nombre_png in paneles:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            curvas = {"entrenamiento": np.asarray(history["train"][key]),
                      "validacion": np.asarray(history["val"][key])}
            if all(np.all(v <= 0) for v in curvas.values()):
                ax.text(0.5, 0.5, "= 0 en todas las epocas", ha="center", va="center", transform=ax.transAxes)
            else:
                for nombre, valores in curvas.items():
                    ax.semilogy(epochs, np.where(valores > 0, valores, np.nan), label=nombre)
                ax.legend()
            ax.axvline(self.result.best_epoch, color="k", ls=":", lw=1, label="mejor epoca")
            ax.set_xlabel("epoca")
            ax.set_title(titulo)
            ax.grid(alpha=0.3)
            self._save(fig, nombre_png)

    # ------------------------------------------------------------------
    # 6. Prueba
    # ------------------------------------------------------------------
    def _error_table(self, err: np.ndarray, t: np.ndarray) -> pd.DataFrame:
        """Media y maximo por tiempo, y el promedio global; err tiene forma (n_ic, T)."""
        visto = np.isclose(t[:, None], self.ds.train.t.numpy()[None]).any(axis=1)
        por_tiempo = pd.DataFrame(
            {"media": err.mean(axis=0), "max": err.max(axis=0), "en_entrenamiento": visto},
            index=pd.Index([f"{x:.2f}" for x in t], name="t"),
        )
        total = pd.DataFrame(
            {"media": [err.mean()], "max": [err.max()], "en_entrenamiento": [None]},
            index=pd.Index(["global"], name="t"),
        )
        return pd.concat([por_tiempo, total])

    def _error_figure(self, err: np.ndarray, t: np.ndarray, titulo: str, nombre: str) -> None:
        """Media y banda min-max por tiempo; los tiempos no vistos en entrenamiento, con otro marcador."""
        visto = np.isclose(t[:, None], self.ds.train.t.numpy()[None]).any(axis=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(t, err.mean(axis=0), "-", color="C0", lw=1, label="media")
        ax.fill_between(t, err.min(axis=0), err.max(axis=0), color="C0", alpha=0.2, label="min - max")
        ax.scatter(t[visto], err.mean(axis=0)[visto], color="k", marker="x", zorder=3, label="t en TRAIN_TIMES")
        ax.scatter(t[~visto], err.mean(axis=0)[~visto], color="C3", marker="o", zorder=3, label="t no visto")
        ax.set_xlabel("t")
        ax.set_ylabel(r"error relativo $L^2$ en espacio fisico")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, nombre)

    def _pair_rel_l2(self) -> float:
        """L2 relativa sobre los pares (k, t) de prueba con |k|_inf <= k_cap, con el simbolo efectivo.

        Es la `rel_l2` de `trajectory_report` (la test_rel_l2 de 04 / 05), en `dtype`:
        sqrt(sum |exp(-t a_bar(k)) u0 - u_ref|^2 / sum |u_ref|^2).
        """
        split = self.ds.grf_test.split
        cdtype = complex_dtype(self.dtype)
        k_unicos, inverso = torch.unique(split["k"], dim=0, return_inverse=True)
        a = self._times(self.model.effective_symbol(k_unicos.cpu().numpy()))[inverso.to(self.device)]
        t = split["t"].to(self.device, self.dtype)
        u0 = split["u0"].to(self.device, cdtype)
        u_ref = split["u_ref"].to(self.device, cdtype)
        residuo = torch.exp(-t * a) * u0 - u_ref
        return float(torch.sqrt(torch.sum(torch.abs(residuo) ** 2) / torch.sum(torch.abs(u_ref) ** 2)))

    def test(self) -> pd.DataFrame:
        """Error relativo L2 en espacio fisico por condicion inicial y tiempo de prueba; tabla y figura."""
        self.err_test = field_errors(self.model, self.ds.test, self.eval_batch_size)   # (n_ic, T)
        err, t = self.err_test, self.ds.test.t.numpy()
        pares = self._pair_rel_l2()
        print(f"PRUEBA -- {self.ds.test.n_ic} condiciones iniciales nuevas x {self.ds.test.n_times} tiempos")
        print(f"error relativo L2 (espacio fisico) = {err.mean():.4e}  (medio; max {err.max():.4e})")
        print(f"error relativo L2 (pares (k, t), {self._window_label}, simbolo efectivo) = {pares:.4e}")
        self.summary.update(
            test_rel_l2=pares, test_rel_l2_fisico=float(err.mean()), test_rel_l2_fisico_max=float(err.max()),
        )

        tabla = self._error_table(err, t)
        tabla.to_csv(self.results / "prueba.csv")
        self._error_figure(err, t, "Error en prueba vs tiempo", "error_prueba.png")
        return tabla

    def field_figure(self) -> None:
        """Primera u0 de prueba: solucion exacta, la del PDNO y el error puntual en algunos tiempos.

        1D: campo.png (exacta y PDNO) y campo_error.png. 2D: una imagen por campo y tiempo,
        campo_{exacta,pdno,error}_t<t>.png; exacta y PDNO comparten la escala de color.
        """
        ds, i = self.ds, 0
        idx_t = [0, 4, 9, 19]                                  # t = 0.05, 0.25, 0.5, 1.0
        t_campo = ds.test.t[idx_t].to(self.device, self.dtype)
        with torch.no_grad():
            u0_rep = ds.test.u0[i].to(self.device, self.dtype).repeat(len(idx_t), *([1] * self.n))
            pred = self.model(u0_rep, t_campo).cpu().numpy()
        exacta = ds.test.u[i, idx_t].numpy()
        tiempos = t_campo.cpu().numpy()

        if self.n == 1:
            x = 2 * np.pi * np.arange(self.resolution) / self.resolution
            colores = plt.cm.viridis(np.linspace(0, 0.85, len(idx_t)))
            fig, ax = plt.subplots(figsize=(10, 4))
            for j, (t, color) in enumerate(zip(tiempos, colores)):
                ax.plot(x, exacta[j], "-", color=color, lw=1.5, label=f"t = {t:.2f}")
                ax.plot(x, pred[j], "--", color=color, lw=1.5)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(t, x)$")
            ax.set_title(f"condicion inicial de prueba {ds.test.ic[i]}: exacta (continua) y PDNO (punteada)")
            ax.legend()
            ax.grid(alpha=0.3)
            self._save(fig, "campo.png")

            fig, ax = plt.subplots(figsize=(10, 4))
            for j, (t, color) in enumerate(zip(tiempos, colores)):
                ax.plot(x, pred[j] - exacta[j], "-", color=color, lw=1, label=f"t = {t:.2f}")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u_\theta(t, x) - u(t, x)$")
            ax.set_title(f"condicion inicial de prueba {ds.test.ic[i]}: error puntual del PDNO")
            ax.legend()
            ax.grid(alpha=0.3)
            self._save(fig, "campo_error.png")
            return

        lim_campo = max(np.abs(exacta).max(), np.abs(pred).max())
        filas = [
            ("exacta", "exacta", exacta, lim_campo),
            ("pdno", "PDNO", pred, lim_campo),
            ("error", "PDNO - exacta", pred - exacta, np.abs(pred - exacta).max()),
        ]
        for sufijo, nombre, campos, lim in filas:
            for campo, t in zip(campos, tiempos):
                self._field_image(campo, f"condicion inicial de prueba {ds.test.ic[i]}: {nombre}, t = {t:.2f}",
                                  f"campo_{sufijo}_t{t:.2f}.png", lim)

    # ------------------------------------------------------------------
    # 7. Simbolo aprendido
    # ------------------------------------------------------------------
    def _symbol_grid(self):
        """Todos los modos de la malla |k_i| <= R/2, el simbolo efectivo, la referencia y la region."""
        efectivo = EffectiveSymbol(self.model)
        k_sym = k_grid(self.n, self.resolution // 2)
        a_eff = np.real(efectivo.evaluate_batch(k_sym))
        a_ref = symbol_values(self.reference, k_sym)
        dentro = np.abs(k_sym).max(axis=1) <= self.k_max
        return efectivo, k_sym, a_eff, a_ref, dentro

    def _symbol_panel(self, ax, norm_k, valores_ref, valores_eff, dentro, etiqueta_ref, etiqueta_eff, log_y):
        """Referencia (continua) y simbolo aprendido contra |k|: curva en 1D, puntos por region en 2D."""
        plot = ax.loglog if log_y else ax.semilogx
        no_cero = norm_k > 0
        orden = np.argsort(norm_k[no_cero])
        plot(norm_k[no_cero][orden], valores_ref[no_cero][orden], "k-", lw=1.5, label=etiqueta_ref)
        if self.n == 1:
            plot(norm_k[no_cero][orden], valores_eff[no_cero][orden], "--", color="C1", lw=1.5, label=etiqueta_eff)
            ax.axvspan(1, self.k_max, color="0.85", zorder=0, label=rf"ventana $|k| \leq {self.k_max}$")
            return
        for region, color, nombre in ((dentro, "C1", "dentro"), (~dentro, "C0", "fuera")):
            sel = region & no_cero
            ax.scatter(norm_k[sel], valores_eff[sel], s=2, color=color,
                       label=rf"{etiqueta_eff}, {nombre} de $|k|_\infty \leq {self.k_max}$")

    def symbol_figure(self) -> None:
        """a_bar(k) frente a la referencia en todos los modos (simbolo.png), y a_bar(k) / <k>^m con la
        banda (simbolo_banda.png). En 1D se dibuja k > 0; a_bar es par."""
        _, k_sym, a_eff, a_ref, dentro = self._symbol_grid()
        if self.n == 1:                                        # a_bar es par: basta k > 0
            positivo = k_sym[:, 0] > 0
            k_sym, a_eff, a_ref, dentro = k_sym[positivo], a_eff[positivo], a_ref[positivo], dentro[positivo]
        norm_k = np.linalg.norm(k_sym, axis=1)
        h_eff = a_eff / (1.0 + norm_k**2) ** (self.m / 2)
        h_ref = a_ref / (1.0 + norm_k**2) ** (self.m / 2)
        ref, m = self._ref_label, self.m

        fig, ax = plt.subplots(figsize=(7, 4.5))
        self._symbol_panel(ax, norm_k, a_ref, a_eff, dentro, rf"${ref}$", r"$\bar a(k)$ aprendido", log_y=True)
        ax.set_ylabel(r"$a(k)$")
        ax.set_title(r"Simbolo vs $|k|$")
        ax.set_xlabel(r"$|k|$")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        self._save(fig, "simbolo.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        self._symbol_panel(ax, norm_k, h_ref, h_eff, dentro, rf"${ref} / \langle k\rangle^{{{m:g}}}$",
                           rf"$\bar a(k) / \langle k\rangle^{{{m:g}}}$", log_y=False)
        ax.axhline(self.bounds.c, color="C3", ls=":", lw=1, label="banda (c, C)")
        ax.axhline(self.bounds.C, color="C3", ls=":", lw=1)
        ax.set_title(r"$\bar a(k) / \langle k\rangle^m$ y la banda")
        ax.set_xlabel(r"$|k|$")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        self._save(fig, "simbolo_banda.png")

    def symbol_error(self) -> pd.DataFrame:
        """|a_bar(k) - a(k)| / a(k) en todos los modos, y metricas dentro y fuera de la ventana.

        error_simbolo.png contra |k|; en 2D, ademas, error_simbolo_mapa.png sobre (k1, k2).
        """
        efectivo, k_sym, a_eff, a_ref, dentro = self._symbol_grid()
        rel = relative_symbol_error(a_eff, a_ref)
        norm_k = np.linalg.norm(k_sym, axis=1)
        no_cero = norm_k > 0
        ylabel = rf"$|\bar a(k) - {self._ref_label}| \,/\, {self._ref_label}$"

        if self.n == 1:
            positivo = k_sym[:, 0] > 0
            orden = np.argsort(k_sym[positivo, 0])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.loglog(k_sym[positivo, 0][orden], rel[positivo][orden], "-", color="C1", lw=1.2)
            ax.axvspan(1, self.k_max, color="0.85", zorder=0, label=rf"ventana $|k| \leq {self.k_max}$")
            ax.set_xlabel(r"$|k|$")
            ax.set_ylabel(ylabel)
            ax.set_title("Error relativo del simbolo")
            ax.legend()
            ax.grid(alpha=0.3, which="both")
            self._save(fig, "error_simbolo.png")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for region, color, nombre in ((dentro, "C1", "dentro"), (~dentro, "C0", "fuera")):
                sel = region & no_cero
                ax.scatter(norm_k[sel], rel[sel], s=2, color=color,
                           label=rf"{nombre} de $|k|_\infty \leq {self.k_max}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$|k|$")
            ax.set_ylabel(ylabel)
            ax.set_title("Error relativo del simbolo")
            ax.legend()
            ax.grid(alpha=0.3, which="both")
            self._save(fig, "error_simbolo.png")

            # Mapa sobre (k1, k2); k = 0 (ambos valen 0) queda en blanco.
            K = self.resolution // 2
            fig, ax = plt.subplots(figsize=(6.5, 5))
            mapa = np.where(no_cero & (rel > 0), rel, np.nan).reshape(2 * K + 1, 2 * K + 1)
            im = ax.imshow(mapa.T, origin="lower", extent=[-K - 0.5, K + 0.5] * 2, norm=LogNorm())
            w = self.k_max + 0.5
            ax.plot([-w, w, w, -w, -w], [-w, -w, w, w, -w], "w--", lw=1,
                    label=rf"ventana $|k|_\infty \leq {self.k_max}$")
            ax.set_xlabel(r"$k_1$")
            ax.set_ylabel(r"$k_2$")
            ax.set_title(r"$|\bar a(k) - a(k)|\,/\,a(k)$ sobre $(k_1, k_2)$")
            ax.legend(loc="upper right")
            fig.colorbar(im, ax=ax)
            self._save(fig, "error_simbolo_mapa.png")

        metricas = {}
        for nombre, region in ((f"dentro ({self._window_label})", dentro),
                               (f"fuera ({self._window_label.replace('<=', '>')})", ~dentro)):
            reporte = symbol_report(efectivo, self.reference, k_sym[region])
            reporte["error_relativo_medio"] = float(rel[region & no_cero].mean())
            reporte["error_relativo_max"] = float(rel[region & no_cero].max())
            metricas[nombre] = reporte
        tabla = pd.DataFrame(metricas)
        tabla.to_csv(self.results / "metricas_simbolo.csv")
        self.summary.update(
            error_simbolo_medio=float(rel[dentro & no_cero].mean()),
            error_simbolo_medio_fuera=float(rel[~dentro & no_cero].mean()),
        )
        return tabla

    def multiplier(self) -> None:
        """exp(-t a_bar(k)) frente a exp(-t a(k)) en 0 <= k <= k_max (en 2D, sobre el eje k = (k1, 0))."""
        efectivo = EffectiveSymbol(self.model)
        t_mult = [0.05, 0.25, 0.5, 1.0]
        colores = plt.cm.viridis(np.linspace(0, 0.85, len(t_mult)))
        k_mult = np.arange(0, self.k_max + 1)
        k_eje = np.zeros((len(k_mult), self.n))
        k_eje[:, 0] = k_mult
        a_eff = np.real(efectivo.evaluate_batch(k_eje))
        a_ref = symbol_values(self.reference, k_eje)

        xlabel = r"$k$" if self.n == 1 else r"$k_1$  ($k_2 = 0$)"

        fig, ax = plt.subplots(figsize=(9, 4))
        for t, color in zip(t_mult, colores):
            ax.plot(k_mult, np.exp(-t * a_ref), "-", color=color, lw=1.5, label=f"t = {t:.2f}")
            ax.plot(k_mult, np.exp(-t * a_eff), "--", color=color, lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$e^{-t a(k)}$")
        ax.set_title("Multiplicador: exacto (continuo) y aprendido (punteado)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        self._save(fig, "multiplicador.png")

        fig, ax = plt.subplots(figsize=(9, 4))
        for t, color in zip(t_mult, colores):
            ax.semilogy(k_mult, np.abs(np.exp(-t * a_eff) - np.exp(-t * a_ref)), "-", color=color, lw=1.2,
                        label=f"t = {t:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rf"$|e^{{-t \bar a(k)}} - e^{{-t {self._ref_label}}}|$")
        ax.set_title("Multiplicador: diferencia absoluta")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        self._save(fig, "multiplicador_error.png")

    # ------------------------------------------------------------------
    # 8. Certificacion
    # ------------------------------------------------------------------
    def _symbols(self) -> Dict[str, Symbol]:
        simbolos: Dict[str, Symbol] = {f"capa {l}": s for l, s in enumerate(self.model.symbols)}
        simbolos["efectivo"] = EffectiveSymbol(self.model)
        return simbolos

    def certification(self) -> pd.DataFrame:
        """Peor ratio de la Def. 2.3 en CERT_LAM x {|k_i| <= K}, K in {k_max, 2 k_max, 3 k_max}."""
        kappa = self.model.theoretical_kappa(self.omega)
        filas = []
        for nombre, simbolo in self._symbols().items():
            verifier = ParabolicityVerifier(simbolo)
            for window in (self.k_max, 2 * self.k_max, self.cert_k_max):
                k_values = k_grid(self.n, window)
                ratio = verifier.estimate_kappa_batch(k_values, CERT_LAM, omega=self.omega) / kappa
                filas.append({
                    "simbolo": nombre, "K": window, "region": "dentro" if window <= self.k_max else "fuera",
                    "puntos": len(k_values) * len(CERT_LAM), "max_ratio": ratio, "satisfecha": ratio <= 1.0,
                })
        tabla = pd.DataFrame(filas)
        tabla.to_csv(self.results / "certificacion.csv", index=False)
        print(f"kappa usada = {kappa:.4f}")
        self.summary.update(peor_ratio=float(tabla["max_ratio"].max()))
        return tabla

    def symbol_class(self) -> pd.DataFrame:
        """Evidencia de pertenencia a S^{m,rho} en ventanas crecientes, por capa y efectivo."""
        k_maxs = tuple(sorted({4, 8, 12, 2 * self.k_max}))
        filas = []
        for nombre, simbolo in self._symbols().items():
            for r in SymbolClassVerifier(simbolo).check(k_maxs=k_maxs):
                filas.append({
                    "simbolo": nombre,
                    "alpha": str(r.alpha),
                    "|alpha|": r.order,
                    **{f"C_hat({K})": r.c_hat[K] for K in k_maxs},
                    f"crecim.({k_maxs[-2]}->{k_maxs[-1]})": r.growth,
                    "pendiente": r.slope,
                    "m-|alpha|": r.slope_theoretical,
                    "estable": r.stable,
                    "consistente": r.consistent,
                })
        clase = pd.DataFrame(filas)
        clase.to_csv(self.results / "clase.csv", index=False)

        for nombre, grupo in clase.groupby("simbolo", sort=False):
            print(f"{nombre}: consistentes {int(grupo['consistente'].sum())} de {len(grupo)} multiindices")
        ventana = f"|k| <= {k_maxs[-1]}" if self.n == 1 else f"|k_i| <= {k_maxs[-1]}"
        print(f"ventana explorada: {ventana} (evidencia, no prueba sobre todo Z^n)")
        print("pendiente = nan significa Delta^alpha a = 0 en la ventana, asi que C_alpha = 0 basta")
        efectivo = clase[clase["simbolo"] == "efectivo"]
        self.summary.update(multiindices_consistentes=f"{int(efectivo['consistente'].sum())}/{len(efectivo)}")
        return clase

    # ------------------------------------------------------------------
    # 9. Extrapolacion temporal
    # ------------------------------------------------------------------
    def extrapolation(self) -> Optional[pd.DataFrame]:
        """Lo mismo que `test` en los tiempos de extrapolacion; None si estan apagados."""
        if self.ds.extrapolation is None:
            print("EXTRAPOLATION_TIMES = None: seccion omitida")
            return None
        data = self.ds.extrapolation
        err = field_errors(self.model, data, self.eval_batch_size)     # (n_ic, T)
        t = data.t.numpy()
        print(f"EXTRAPOLACION -- {data.n_ic} condiciones iniciales x {data.n_times} tiempos")
        print(f"error relativo L2 (espacio fisico) = {err.mean():.4e}  (medio; max {err.max():.4e})")
        tabla = self._error_table(err, t)
        tabla.to_csv(self.results / "extrapolacion.csv")
        self._error_figure(err, t, "Error en extrapolacion temporal vs tiempo", "error_extrapolacion.png")
        return tabla

    # ------------------------------------------------------------------
    # 10. Modelo guardado
    # ------------------------------------------------------------------
    def save_model(self) -> Path:
        """Guarda la corrida elegida en results/modelo.pt (`save_operator_model`)."""
        return save_operator_model(
            self.results / MODEL_FILE, self.model, reference=self.reference, resolution=self.resolution,
            k_cap=self.k_cap, config=self.best.config, result=self.result, seed=self.seed,
            val_fraction=self.val_fraction, extrapolation_times=self.extrapolation_times, delta=self.delta,
            estimate_tol=self.estimate_tol,
        )

    def check_saved_model(self) -> bool:
        """Recarga modelo.pt y comprueba que da el mismo error de prueba que el modelo entrenado."""
        cargado = load_operator_model(self.results / MODEL_FILE, device=self.device)
        err_cargado = field_errors(cargado, self.ds.test, self.eval_batch_size)
        identicos = bool(np.array_equal(err_cargado, self.err_test))
        print(f"dtype del modelo cargado = {cargado.dtype}")
        print(f"error de prueba (cargado) = {err_cargado.mean():.6e}  |  (entrenado) = {self.err_test.mean():.6e}")
        print(f"identicos: {identicos}")
        return identicos

    def summary_row(self) -> Dict:
        """Una fila de la tabla resumen, con lo que fueron guardando las secciones."""
        return dict(self.summary)


def pdno_summary_table(experiments: Dict[float, PDNOExperiment], results: Path) -> pd.DataFrame:
    """Una fila por alpha con lo que resumio cada experimento; se guarda en results/resumen.csv."""
    tabla = pd.DataFrame(
        [{"alpha": alpha, **exp.summary_row()} for alpha, exp in experiments.items()]
    )[PDNO_SUMMARY_COLS]
    tabla.to_csv(Path(results) / "resumen.csv", index=False)
    return tabla


def pdno_summary_figures(resumen, results: Path) -> None:
    """Las graficas de `figures.summary_figure`, cada una en su imagen.

    resumen_error_prueba.png: error en prueba (pares (k, t) y espacio fisico) contra alpha;
    resumen_certificacion.png: peor ratio de la certificacion contra alpha, con la cota ratio = 1.
    `resumen` es la ruta de resumen.csv o la tabla de `pdno_summary_table`.
    """
    summary_figure(resumen, results)

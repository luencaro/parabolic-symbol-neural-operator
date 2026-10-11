"""Datos de la ecuacion del calor 1D al estilo del paper base (Sec. 5.1).

Convencion del repo: toro [0, 2 pi), u(x) = sum_k u_hat(k) e^{i k x} y el
generador tiene el simbolo de HeatSymbol, a(k) = |k|^2 (sin difusividad ni
factores 4 pi^2). La misma convencion fija el campo aleatorio: -Delta tiene
simbolo |k|^2.

- u0 ~ N(0, 7^4 (-Delta + 7^2)^{-2.5}), real, muestreado espectralmente en una
  malla de `resolution` puntos: con autofunciones e_k = e^{ikx} / sqrt(2 pi),

      u0 = sum_k sigma_k xi_k e_k,   sigma_k^2 = 7^4 (|k|^2 + 49)^{-2.5},

  xi_k normal compleja estandar con xi_{-k} = conj(xi_k) (y real en k = 0 y en
  Nyquist), asi que u_hat0(k) = sigma_k xi_k / sqrt(2 pi).
- Solucion exacta: u_hat(t, k) = exp(-t a(k)) u_hat0(k).
- Mascara de recuperabilidad: se descartan los pares (k, t) con
  exp(-t a(k)) <= eps del dtype de entrenamiento, porque ahi u_hat(t, k) cae por
  debajo de la precision de maquina respecto de u_hat0(k) y no informa sobre a(k).

`split` de cada conjunto tiene el mismo formato que `make_trajectory_data`
(claves k, t, u0, u_ref; una fila por condicion inicial y par (k, t)
recuperable), asi que `split_trajectory_data`, `estimate_symbol` y `train`
funcionan sin cambios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch

from ..symbol import Symbol
from ..learning.estimation import KWindow, k_window, symbol_values
from ..utils import LearnedSymbol, Split

# Rejillas temporales del paper: 10 tiempos de entrenamiento, 20 de prueba.
TRAIN_TIMES = 0.05 + 0.1 * np.arange(10)
TEST_TIMES = 0.05 + 0.05 * np.arange(20)


@dataclass
class GRFHeatData:
    """Un conjunto (entrenamiento o prueba) de trayectorias del calor."""

    split: Split            # formato de train(...): solo pares recuperables
    k: np.ndarray           # (N,) frecuencias enteras, orden de np.fft
    t: np.ndarray           # (T,)
    a: np.ndarray           # (N,) simbolo de referencia en k
    u0_hat: np.ndarray      # (n_ic, N) coeficientes de u0, todos los modos
    mask: np.ndarray        # (T, N) pares (k, t) recuperables
    tol: float
    window: KWindow         # k que sobreviven a la mascara

    @property
    def resolution(self) -> int:
        return len(self.k)


def grf_variance(k: np.ndarray, tau: float = 7.0, alpha: float = 2.5) -> np.ndarray:
    """sigma_k^2 = tau^4 (|k|^2 + tau^2)^{-alpha}, autovalores de tau^4 (-Delta + tau^2)^{-alpha}."""
    return tau**4 * (np.asarray(k, dtype=float) ** 2 + tau**2) ** (-alpha)


def sample_grf(
    n_samples: int, resolution: int, rng: np.random.Generator, tau: float = 7.0, alpha: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Muestras reales de u0 en la malla x_j = 2 pi j / resolution.

    Devuelve (u0, u0_hat): u0 real de forma (n_samples, resolution) y sus
    coeficientes u0_hat(k) = fft(u0) / resolution, en el orden de np.fft.
    """
    if resolution % 2:
        raise ValueError(f"la resolucion debe ser par: {resolution}")
    k_half = np.arange(resolution // 2 + 1)            # 0..Nyquist (rfft)
    sigma = np.sqrt(grf_variance(k_half, tau, alpha) / (2.0 * np.pi))
    xi = (
        rng.standard_normal((n_samples, len(k_half)))
        + 1j * rng.standard_normal((n_samples, len(k_half)))
    ) / np.sqrt(2.0)
    xi[:, [0, -1]] = rng.standard_normal((n_samples, 2))  # k = 0 y Nyquist son reales
    # u0(x_j) = sum_k u_hat(k) e^{i k x_j} = resolution * irfft(u_hat)
    u0 = resolution * np.fft.irfft(sigma * xi, n=resolution, axis=-1)
    return u0, np.fft.fft(u0, axis=-1) / resolution


def make_grf_heat_data(
    reference: Symbol,
    n_ic: int,
    times: Sequence[float],
    resolution: int = 1024,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    tau: float = 7.0,
    alpha: float = 2.5,
    verbose: bool = True,
) -> GRFHeatData:
    """n_ic condiciones iniciales GRF evolucionadas con el simbolo `reference` (n = 1).

    `dtype` es la precision con la que se va a entrenar: fija la tolerancia
    eps = finfo(dtype).eps de la mascara de recuperabilidad. Los tensores del
    split se guardan en float64 / complex128, como en `make_trajectory_data`;
    `train` los pasa al dtype del modelo.
    """
    if reference.n != 1:
        raise ValueError(f"los datos GRF son 1D: reference.n = {reference.n}")
    rng = np.random.default_rng(seed)
    _, u0_hat = sample_grf(n_ic, resolution, rng, tau, alpha)
    k = np.rint(np.fft.fftfreq(resolution, d=1.0 / resolution)).astype(int)
    a = symbol_values(reference, k[:, None])
    t = np.asarray(times, dtype=float)

    tol = float(torch.finfo(dtype).eps)
    mask = np.exp(-t[:, None] * a[None, :]) > tol      # (T, N)
    it, ik = np.nonzero(mask)                           # pares (t, k) recuperables

    t_rows = np.tile(t[it], n_ic)
    k_rows = np.tile(k[ik], n_ic)
    u0_rows = u0_hat[:, ik].reshape(-1)
    u_rows = np.exp(-t_rows * np.tile(a[ik], n_ic)) * u0_rows
    split = {
        "k": torch.tensor(k_rows[:, None], dtype=torch.float64),
        "t": torch.tensor(t_rows, dtype=torch.float64),
        "u0": torch.tensor(u0_rows, dtype=torch.complex128),
        "u_ref": torch.tensor(u_rows, dtype=torch.complex128),
    }
    window = k_window(np.unique(k[ik])[:, None])

    if verbose:
        print(
            f"{n_ic} condiciones iniciales x {len(t)} tiempos, resolucion {resolution}: "
            f"{len(it)} de {mask.size} pares (k, t) recuperables (eps={tol:.3g}) "
            f"-> {split['k'].shape[0]:,} muestras"
        )
        print(f"  ventana de k recuperable: {window}")
        k_t = [int(np.abs(k[row]).max()) for row in mask]
        print("  max |k| por tiempo: " + ", ".join(f"t={ti:.2f}:{kt}" for ti, kt in zip(t, k_t)))

    return GRFHeatData(
        split=split, k=k, t=t, a=a, u0_hat=u0_hat, mask=mask, tol=tol, window=window
    )


def heat_grf_datasets(
    reference: Symbol, resolution: int = 1024, dtype: torch.dtype = torch.float32, seed: int = 0
) -> Tuple[GRFHeatData, GRFHeatData]:
    """Entrenamiento (100 u0, TRAIN_TIMES) y prueba (20 u0 nuevas, TEST_TIMES).

    Las condiciones iniciales de prueba salen de un flujo aleatorio distinto
    (SeedSequence.spawn), asi que nunca coinciden con las de entrenamiento.
    """
    seed_train, seed_test = np.random.SeedSequence(seed).spawn(2)
    print("ENTRENAMIENTO")
    train = make_grf_heat_data(
        reference, 100, TRAIN_TIMES, resolution, dtype, np.random.default_rng(seed_train).integers(2**32)
    )
    print("PRUEBA")
    test = make_grf_heat_data(
        reference, 20, TEST_TIMES, resolution, dtype, np.random.default_rng(seed_test).integers(2**32)
    )
    return train, test


def field_rel_l2(symbol: LearnedSymbol, data: GRFHeatData) -> np.ndarray:
    """Error relativo L2 en espacio fisico, como en el paper: ||u_theta - u|| / ||u||.

    Se reconstruye u(t, x) en la malla completa (todos los modos, tambien los
    que la mascara descarta) con a_theta y con el simbolo exacto. Devuelve un
    arreglo (n_ic, T) con el error de cada condicion inicial y tiempo.
    """
    a_theta = np.real(symbol.evaluate_batch(data.k[:, None])).astype(float)
    fields = []
    for a in (a_theta, data.a):
        u_hat = np.exp(-data.t[:, None] * a[None, :])[None] * data.u0_hat[:, None, :]  # (n_ic, T, N)
        fields.append(data.resolution * np.real(np.fft.ifft(u_hat, axis=-1)))
    pred, ref = fields
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)

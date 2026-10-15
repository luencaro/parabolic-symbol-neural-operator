"""Datos GRF al estilo del paper base (Sec. 5.1), para cualquier simbolo y cualquier n.

Convencion del repo: toro T^n = [0, 2 pi)^n, u(x) = sum_k u_hat(k) e^{i k.x} y
el generador tiene el simbolo `reference`, p. ej. HeatSymbol, a(k) = |k|^2 (sin
difusividad ni factores 4 pi^2), o FractionalLaplacianSymbol, a(k) = |k|^alpha.
La misma convencion fija el campo aleatorio: -Delta tiene simbolo |k|^2.

- u0 ~ N(0, 7^4 (-Delta + 7^2)^{-2.5}), real, muestreado espectralmente en una
  malla de `resolution` puntos: con autofunciones e_k = e^{ikx} / sqrt(2 pi),

      u0 = sum_k sigma_k xi_k e_k,   sigma_k^2 = 7^4 (|k|^2 + 49)^{-2.5},

  xi_k normal compleja estandar con xi_{-k} = conj(xi_k) (y real en k = 0 y en
  Nyquist), asi que u_hat0(k) = sigma_k xi_k / sqrt(2 pi).
  `sample_grf_nd` muestrea el mismo campo en T^n (rfftn / irfftn). En n = 1
  los conjuntos salen de `sample_grf` (todos los k, orden de np.fft) y en
  n >= 2 de `sample_grf_nd` (layout de `grf_wavenumbers`: medio espacio de k,
  porque u_hat(-k) = conj(u_hat(k)) no da informacion nueva).
- Solucion exacta: u_hat(t, k) = exp(-t a(k)) u_hat0(k).
- Se guardan todos los pares (k, t) con |k|_inf <= k_cap, como en el paper,
  aunque exp(-t a(k)) caiga por debajo de la precision de maquina. `k_cap` es
  obligatorio (64 en 1D, 32 en 2D en los notebooks) y fija k_max.
- El exponente de la covarianza se llama `decay` (2.5 por defecto) para no
  confundirlo con el alpha del laplaciano fraccionario; las funciones aceptan
  todavia `alpha=` como nombre antiguo.

`split` de cada conjunto tiene claves k, t, u0, u_ref, con una fila por
condicion inicial y par (k, t), el formato de `split_trajectory_data`,
`estimate_symbol` y `train`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..symbol import Symbol
from ..learning.estimation import KWindow, k_window, symbol_values
from ..utils import LearnedSymbol, Split

# Rejillas temporales del paper: 10 tiempos de entrenamiento, 20 de prueba.
TRAIN_TIMES = 0.05 + 0.1 * np.arange(10)
TEST_TIMES = 0.05 + 0.05 * np.arange(20)


@dataclass
class GRFData:
    """Un conjunto (entrenamiento o prueba) de trayectorias de u_t + a(D) u = 0."""

    split: Split            # formato de train(...): pares (k, t) con |k|_inf <= k_cap
    k: np.ndarray           # (N, n) frecuencias enteras, en el layout de `spectral_shape`
    t: np.ndarray           # (T,)
    a: np.ndarray           # (N,) simbolo de referencia en k
    u0_hat: np.ndarray      # (n_ic, N) coeficientes de u0, todos los modos
    mask: np.ndarray        # (T, N) pares (k, t) del split: los de |k|_inf <= k_cap
    k_cap: int
    window: KWindow         # k del split
    resolution: int         # puntos de la malla por eje

    @property
    def n(self) -> int:
        return self.k.shape[1]

    @property
    def spectral_shape(self) -> Tuple[int, ...]:
        """Forma de los coeficientes: (R,) en n = 1 (np.fft), layout de rfftn si n >= 2."""
        if self.n == 1:
            return (self.resolution,)
        return (self.resolution,) * (self.n - 1) + (self.resolution // 2 + 1,)

    def to_physical(self, u_hat: np.ndarray) -> np.ndarray:
        """u(x_j) = sum_k u_hat(k) e^{i k.x_j} en la malla, para u_hat de forma (..., N)."""
        u_hat = u_hat.reshape(u_hat.shape[:-1] + self.spectral_shape)
        if self.n == 1:
            return self.resolution * np.real(np.fft.ifft(u_hat, axis=-1))
        shape = (self.resolution,) * self.n
        axes = tuple(range(-self.n, 0))
        return self.resolution**self.n * np.fft.irfftn(u_hat, s=shape, axes=axes)


# Nombre anterior, cuando los datos eran solo del calor 1D.
GRFHeatData = GRFData


def _decay(decay: float, alpha: Optional[float]) -> float:
    """Compatibilidad: `alpha=` es el nombre antiguo de `decay`."""
    return decay if alpha is None else alpha


def grf_variance(
    k: np.ndarray, tau: float = 7.0, decay: float = 2.5, *, alpha: Optional[float] = None
) -> np.ndarray:
    """sigma_k^2 = tau^4 (|k|^2 + tau^2)^{-decay}, autovalores de tau^4 (-Delta + tau^2)^{-decay}."""
    decay = _decay(decay, alpha)
    return tau**4 * (np.asarray(k, dtype=float) ** 2 + tau**2) ** (-decay)


def sample_grf(
    n_samples: int,
    resolution: int,
    rng: np.random.Generator,
    tau: float = 7.0,
    decay: float = 2.5,
    *,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Muestras reales de u0 en la malla x_j = 2 pi j / resolution.

    Devuelve (u0, u0_hat): u0 real de forma (n_samples, resolution) y sus
    coeficientes u0_hat(k) = fft(u0) / resolution, en el orden de np.fft.
    """
    if resolution % 2:
        raise ValueError(f"la resolucion debe ser par: {resolution}")
    k_half = np.arange(resolution // 2 + 1)            # 0..Nyquist (rfft)
    sigma = np.sqrt(grf_variance(k_half, tau, _decay(decay, alpha)) / (2.0 * np.pi))
    xi = (
        rng.standard_normal((n_samples, len(k_half)))
        + 1j * rng.standard_normal((n_samples, len(k_half)))
    ) / np.sqrt(2.0)
    xi[:, [0, -1]] = rng.standard_normal((n_samples, 2))  # k = 0 y Nyquist son reales
    # u0(x_j) = sum_k u_hat(k) e^{i k x_j} = resolution * irfft(u_hat)
    u0 = resolution * np.fft.irfft(sigma * xi, n=resolution, axis=-1)
    return u0, np.fft.fft(u0, axis=-1) / resolution


def grf_wavenumbers(resolution: int, n: int) -> np.ndarray:
    """Frecuencias enteras k in Z^n en el layout de np.fft.rfftn.

    Forma (resolution,) * (n - 1) + (resolution // 2 + 1, n): los primeros n - 1
    ejes van en el orden de np.fft.fftfreq y el ultimo es 0..Nyquist.
    """
    axes = [np.rint(np.fft.fftfreq(resolution, d=1.0 / resolution))] * (n - 1)
    axes.append(np.arange(resolution // 2 + 1, dtype=float))
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def sample_grf_nd(
    n_samples: int,
    resolution: int,
    n: int,
    rng: np.random.Generator,
    tau: float = 7.0,
    decay: float = 2.5,
    *,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Muestras reales de u0 ~ N(0, tau^4 (-Delta + tau^2)^{-decay}) en T^n = [0, 2 pi)^n.

    Malla x_j = 2 pi j / resolution en cada eje. Con autofunciones
    e_k = e^{i k.x} / (2 pi)^{n/2}, u0_hat(k) = sigma_k xi_k / (2 pi)^{n/2}; en
    n = 1 es la misma distribucion que `sample_grf` (no la misma muestra para
    una semilla dada). xi sale de rfftn de ruido blanco real dividido por
    resolution^{n/2}: asi E|xi_k|^2 = 1 y xi_{-k} = conj(xi_k), y u0 es real sin
    imponer la simetria a mano.

    Devuelve (u0, u0_hat): u0 de forma (n_samples,) + (resolution,) * n y
    u0_hat(k) = rfftn(u0) / resolution^n en el layout de `grf_wavenumbers`.
    u(x) = sum_k u_hat(k) e^{i k.x} se recupera con
    resolution^n * irfftn(u_hat, s=(resolution,) * n).
    """
    if resolution % 2:
        raise ValueError(f"la resolucion debe ser par: {resolution}")
    if n < 1:
        raise ValueError(f"n debe ser >= 1: {n}")
    shape = (resolution,) * n
    axes = tuple(range(-n, 0))
    xi = np.fft.rfftn(rng.standard_normal((n_samples,) + shape), axes=axes) / resolution ** (n / 2)
    k_norm = np.linalg.norm(grf_wavenumbers(resolution, n), axis=-1)
    u0_hat = np.sqrt(grf_variance(k_norm, tau, _decay(decay, alpha))) / (2.0 * np.pi) ** (n / 2) * xi
    u0 = resolution**n * np.fft.irfftn(u0_hat, s=shape, axes=axes)
    return u0, u0_hat


def make_grf_data(
    reference: Symbol,
    n_ic: int,
    times: Sequence[float],
    resolution: int = 1024,
    seed: int = 0,
    tau: float = 7.0,
    decay: float = 2.5,
    verbose: bool = True,
    *,
    k_cap: int,
    alpha: Optional[float] = None,
) -> GRFData:
    """n_ic condiciones iniciales GRF en T^n evolucionadas con el simbolo `reference`.

    n = reference.n. En n = 1 se usa `sample_grf` (todos los k, orden de np.fft);
    en n >= 2, `sample_grf_nd` (layout de `grf_wavenumbers`). El split tiene
    todos los pares (k, t) con |k|_inf <= k_cap (obligatorio, < resolution / 2).
    Los tensores del split se guardan en float64 / complex128, con k de forma
    (filas, n); `train` los pasa al dtype del modelo.
    """
    n = reference.n
    if not 0 <= k_cap < resolution // 2:
        raise ValueError(f"k_cap debe estar en [0, resolution / 2): {k_cap}")
    decay = _decay(decay, alpha)
    rng = np.random.default_rng(seed)
    if n == 1:
        _, u0_hat = sample_grf(n_ic, resolution, rng, tau, decay)
        k = np.rint(np.fft.fftfreq(resolution, d=1.0 / resolution)).astype(int)[:, None]
    else:
        _, u0_hat = sample_grf_nd(n_ic, resolution, n, rng, tau, decay)
        k = grf_wavenumbers(resolution, n).reshape(-1, n).astype(int)
        u0_hat = u0_hat.reshape(n_ic, -1)
    a = symbol_values(reference, k)
    t = np.asarray(times, dtype=float)

    en_cap = np.abs(k).max(axis=1) <= k_cap
    mask = np.broadcast_to(en_cap, (len(t), len(k))).copy()   # (T, N)
    it, ik = np.nonzero(mask)                                 # pares (t, k) del split

    t_rows = np.tile(t[it], n_ic)
    k_rows = np.tile(k[ik], (n_ic, 1))
    u0_rows = u0_hat[:, ik].reshape(-1)
    u_rows = np.exp(-t_rows * np.tile(a[ik], n_ic)) * u0_rows
    split = {
        "k": torch.tensor(k_rows, dtype=torch.float64),
        "t": torch.tensor(t_rows, dtype=torch.float64),
        "u0": torch.tensor(u0_rows, dtype=torch.complex128),
        "u_ref": torch.tensor(u_rows, dtype=torch.complex128),
    }
    window = k_window(np.unique(k[ik], axis=0))

    if verbose:
        malla = "x".join([str(resolution)] * n)
        print(
            f"{n_ic} condiciones iniciales x {len(t)} tiempos, resolucion {malla}: "
            f"{len(it)} pares (k, t) con |k|_inf <= {k_cap} -> {split['k'].shape[0]:,} muestras"
        )
        print(f"  ventana de k: {window}")

    return GRFData(
        split=split, k=k, t=t, a=a, u0_hat=u0_hat, mask=mask, k_cap=k_cap, window=window,
        resolution=resolution,
    )


def grf_datasets(
    reference: Symbol,
    resolution: int = 1024,
    seed: int = 0,
    *,
    k_cap: int,
    tau: float = 7.0,
    decay: float = 2.5,
) -> Tuple[GRFData, GRFData]:
    """Entrenamiento (100 u0, TRAIN_TIMES) y prueba (20 u0 nuevas, TEST_TIMES).

    Las condiciones iniciales de prueba salen de un flujo aleatorio distinto
    (SeedSequence.spawn), asi que nunca coinciden con las de entrenamiento.
    """
    seed_train, seed_test = np.random.SeedSequence(seed).spawn(2)
    print("ENTRENAMIENTO")
    train = make_grf_data(
        reference, 100, TRAIN_TIMES, resolution,
        np.random.default_rng(seed_train).integers(2**32), tau, decay, k_cap=k_cap,
    )
    print("PRUEBA")
    test = make_grf_data(
        reference, 20, TEST_TIMES, resolution,
        np.random.default_rng(seed_test).integers(2**32), tau, decay, k_cap=k_cap,
    )
    return train, test


# Nombres anteriores, cuando los datos eran solo del calor 1D (notebooks 01 y 02).
make_grf_heat_data = make_grf_data
heat_grf_datasets = grf_datasets


def field_rel_l2(symbol: LearnedSymbol, data: GRFData) -> np.ndarray:
    """Error relativo L2 en espacio fisico, como en el paper: ||u_theta - u|| / ||u||.

    Se reconstruye u(t, x) en la malla completa de T^n (todos los modos, tambien
    los de |k|_inf > k_cap) con a_theta y con el simbolo exacto. Devuelve un
    arreglo (n_ic, T) con el error de cada condicion inicial y tiempo.
    """
    a_theta = np.real(symbol.evaluate_batch(data.k)).astype(float)
    space_axes = tuple(range(-data.n, 0))
    err = np.empty((data.u0_hat.shape[0], len(data.t)))
    for j, t in enumerate(data.t):   # un tiempo por vez: en 2D el campo completo pesa
        pred, ref = (data.to_physical(np.exp(-t * a)[None] * data.u0_hat) for a in (a_theta, data.a))
        err[:, j] = np.sqrt(np.sum((pred - ref) ** 2, axis=space_axes) / np.sum(ref**2, axis=space_axes))
    return err

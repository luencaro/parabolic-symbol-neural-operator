"""Datos GRF como campos completos para el PDNO, con particion por condicion inicial.

A partir de un GRFData se construyen, en la malla completa de T^n (todos los
modos, tambien los de |k|_inf > k_cap),

    u0 de forma (n_ic, *malla)   y   u(t) = exp(-t a(D)) u0 de forma (n_ic, T, *malla),

con u(x) = sum_k u_hat(k) e^{ik.x} (`GRFData.to_physical`). Una muestra es un
par (condicion inicial i, tiempo t_j), con indice plano p = i T + j.

Particion (`field_datasets`): todos los tiempos de una condicion inicial van al
mismo conjunto.
- entrenamiento y validacion: las 100 u0 de entrenamiento de `grf_datasets`,
  repartidas por condicion inicial, ambas en TRAIN_TIMES. La validacion mide
  generalizacion a u0 nuevas y es la del early stopping y la seleccion;
- prueba: las 20 u0 de prueba en TEST_TIMES;
- extrapolacion (opcional): las u0 de prueba en tiempos > max(TRAIN_TIMES);
  solo se reporta.
La banda (c, C) y las C_alpha salen de las filas espectrales (k, t, u0, u_ref)
de las u0 de entrenamiento (`ic_rows`), en el formato de siempre.

`resample_fields` lleva un FieldData 1D o 2D a otra malla (submuestreo o relleno
con ceros del espectro) para evaluar la invariancia de resolucion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..symbol import Symbol
from .grf import GRFData, grf_datasets
from .grids import to_device
from ..utils import Split


@dataclass
class FieldData:
    """Campos u0 y u(t) de un conjunto de condiciones iniciales; muestras (i, t_j)."""

    u0: torch.Tensor        # (n_ic, *malla), real
    u: torch.Tensor         # (n_ic, T, *malla), solucion exacta
    t: torch.Tensor         # (T,)
    ic: np.ndarray          # indices de las u0 en el GRFData de origen

    @property
    def n(self) -> int:
        return self.u0.dim() - 1

    @property
    def resolution(self) -> int:
        return self.u0.shape[-1]

    @property
    def n_ic(self) -> int:
        return self.u0.shape[0]

    @property
    def n_times(self) -> int:
        return self.t.shape[0]

    @property
    def n_pairs(self) -> int:
        return self.n_ic * self.n_times

    def pairs(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u0_b, t_b, u_b) de las muestras con indice plano p = i T + j."""
        i = torch.div(idx, self.n_times, rounding_mode="floor")
        j = idx % self.n_times
        return self.u0[i], self.t[j], self.u[i, j]

    def to(self, device: torch.device, dtype: torch.dtype) -> "FieldData":
        return FieldData(
            u0=self.u0.to(device=device, dtype=dtype), u=self.u.to(device=device, dtype=dtype),
            t=self.t.to(device=device, dtype=dtype), ic=self.ic,
        )


@dataclass
class FieldDatasets:
    """Los conjuntos del PDNO; ver el docstring del modulo."""

    train: FieldData
    val: FieldData
    test: FieldData
    extrapolation: Optional[FieldData]
    rows_train: Split       # filas (k, t, u0, u_ref) de las u0 de entrenamiento
    grf_train: GRFData
    grf_test: GRFData

    def to(self, device: torch.device, dtype: torch.dtype) -> "FieldDatasets":
        """Campos y filas espectrales en `device` / `dtype` (complejo asociado en las filas).

        Los GRFData (numpy, float64) no cambian.
        """
        return FieldDatasets(
            train=self.train.to(device, dtype), val=self.val.to(device, dtype),
            test=self.test.to(device, dtype),
            extrapolation=None if self.extrapolation is None else self.extrapolation.to(device, dtype),
            rows_train=to_device(self.rows_train, device, dtype),
            grf_train=self.grf_train, grf_test=self.grf_test,
        )


def grf_fields(
    data: GRFData,
    ic: Optional[Sequence[int]] = None,
    times: Optional[Sequence[float]] = None,
    dtype: torch.dtype = torch.float64,
) -> FieldData:
    """Campos de las condiciones iniciales `ic` (todas por defecto) en `times` (data.t por defecto).

    u(t) = to_physical(exp(-t a(k)) u0_hat(k)), con el simbolo de referencia de
    `data`, en la malla completa. Se calculan con numpy (float64) y se guardan
    en `dtype` en CPU.
    """
    ic = np.arange(data.u0_hat.shape[0]) if ic is None else np.asarray(ic, dtype=int)
    times = data.t if times is None else np.asarray(times, dtype=float)
    u0_hat = data.u0_hat[ic]
    u0 = data.to_physical(u0_hat)
    u = np.stack([data.to_physical(np.exp(-t * data.a)[None] * u0_hat) for t in times], axis=1)
    return FieldData(
        u0=torch.tensor(u0, dtype=dtype), u=torch.tensor(u, dtype=dtype),
        t=torch.tensor(times, dtype=dtype), ic=ic,
    )


def ic_rows(data: GRFData, ic: Sequence[int]) -> Split:
    """Filas (k, t, u0, u_ref) de `data.split` que pertenecen a las condiciones iniciales `ic`.

    `make_grf_data` apila las filas por condicion inicial, con el mismo numero
    de pares (k, t) para cada una, asi que la i-esima ocupa un bloque contiguo.
    """
    ic = np.asarray(ic, dtype=int)
    per_ic = data.split["k"].shape[0] // data.u0_hat.shape[0]
    idx = torch.from_numpy((ic[:, None] * per_ic + np.arange(per_ic)).reshape(-1))
    return {key: value[idx] for key, value in data.split.items()}


def split_by_ic(n_ic: int, val_fraction: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (entrenamiento, validacion) de una particion aleatoria de n_ic condiciones iniciales."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction debe estar en (0, 1): {val_fraction}")
    perm = np.random.default_rng(seed).permutation(n_ic)
    n_val = int(round(val_fraction * n_ic))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def field_datasets(
    reference: Symbol,
    resolution: int = 1024,
    seed: int = 0,
    *,
    k_cap: int,
    val_fraction: float = 0.2,
    extrapolation_times: Optional[Sequence[float]] = None,
    tau: float = 7.0,
    decay: float = 2.5,
    freq_scale: float = 1.0,
    amplitude: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> FieldDatasets:
    """Entrenamiento, validacion, prueba y (opcional) extrapolacion como campos completos.

    Los datos son los de `grf_datasets` con la misma semilla. `k_cap` solo fija
    las filas espectrales con que se estiman la banda y las C_alpha; los campos
    tienen todos los modos. `extrapolation_times` debe ser > max(TRAIN_TIMES).
    `freq_scale` y `amplitude` son los del campo aleatorio (ver `data/grf.py`).
    Campos y filas espectrales quedan en `dtype` (complejo asociado en las
    filas), en CPU; los GRFData se generan con numpy y no cambian.
    """
    grf_train, grf_test = grf_datasets(
        reference, resolution, seed, k_cap=k_cap, tau=tau, decay=decay,
        freq_scale=freq_scale, amplitude=amplitude,
    )
    train_ic, val_ic = split_by_ic(grf_train.u0_hat.shape[0], val_fraction, seed)
    extrapolation = None
    if extrapolation_times is not None:
        extrapolation_times = np.asarray(extrapolation_times, dtype=float)
        if not np.all(extrapolation_times > grf_train.t.max()):
            raise ValueError(
                f"los tiempos de extrapolacion deben ser > {grf_train.t.max():g}: {extrapolation_times}"
            )
        extrapolation = grf_fields(grf_test, times=extrapolation_times, dtype=dtype)
    datasets = FieldDatasets(
        train=grf_fields(grf_train, train_ic, dtype=dtype),
        val=grf_fields(grf_train, val_ic, dtype=dtype),
        test=grf_fields(grf_test, dtype=dtype),
        extrapolation=extrapolation,
        rows_train=to_device(ic_rows(grf_train, train_ic), torch.device("cpu"), dtype),
        grf_train=grf_train,
        grf_test=grf_test,
    )
    print("CAMPOS (condicion inicial x tiempo)")
    for name in ("train", "val", "test", "extrapolation"):
        data = getattr(datasets, name)
        if data is not None:
            print(f"  {name:13s}: {data.n_ic} u0 x {data.n_times} tiempos = {data.n_pairs} pares, "
                  f"malla {tuple(data.u0.shape[1:])}")
    return datasets


def _zero_pad_spectrum(u: torch.Tensor, resolution: int) -> torch.Tensor:
    """El polinomio trigonometrico de u (ultimo eje) evaluado en una malla mas fina.

    Los modos nuevos valen 0. El de Nyquist de la malla gruesa (real) se
    interpreta como coseno: se reparte a la mitad entre +R/2 y -R/2.
    """
    R = u.shape[-1]
    u_hat = torch.fft.rfft(u, dim=-1) / R
    u_hat[..., R // 2] = u_hat[..., R // 2] / 2
    padded = torch.zeros(u.shape[:-1] + (resolution // 2 + 1,), dtype=u_hat.dtype, device=u.device)
    padded[..., : R // 2 + 1] = u_hat
    return resolution * torch.fft.irfft(padded, n=resolution, dim=-1)


def _zero_pad_spectrum_nd(u: torch.Tensor, resolution: int, n: int) -> torch.Tensor:
    """`_zero_pad_spectrum` en los ultimos n ejes (n >= 2), eje por eje con la FFT completa.

    En cada eje, los modos 0..R/2-1 y -R/2+1..-1 se copian, los nuevos valen 0 y
    el de Nyquist se reparte a la mitad entre +R/2 y -R/2, como en 1D; asi el
    espectro sigue siendo hermitico y el resultado es real.
    """
    R = u.shape[-1]
    h = R // 2
    axes = tuple(range(-n, 0))
    u_hat = torch.fft.fftn(u, dim=axes) / R**n
    for axis in axes:
        def sl(start, stop, axis=axis):
            index = [slice(None)] * u_hat.dim()
            index[axis] = slice(start, stop)
            return tuple(index)

        shape = list(u_hat.shape)
        shape[axis] = resolution
        padded = torch.zeros(shape, dtype=u_hat.dtype, device=u.device)
        padded[sl(0, h)] = u_hat[sl(0, h)]
        padded[sl(resolution - h + 1, resolution)] = u_hat[sl(h + 1, R)]
        nyquist = u_hat[sl(h, h + 1)] / 2
        padded[sl(h, h + 1)] = nyquist
        padded[sl(resolution - h, resolution - h + 1)] = nyquist
        u_hat = padded
    return resolution**n * torch.fft.ifftn(u_hat, dim=axes).real


def resample_fields(data: FieldData, resolution: int) -> FieldData:
    """Los campos u0 y u(t) de `data` en una malla de `resolution` puntos por eje (n = 1 o 2).

    - resolution < data.resolution (debe dividirla): submuestreo, los valores
      exactos de las mismas funciones en la malla gruesa. Los modos por encima
      del Nyquist grueso se pliegan (aliasing); con el decaimiento del GRF son
      despreciables.
    - resolution > data.resolution (par): relleno con ceros del espectro de
      u0 y de u(t). Como exp(-t a(D)) es diagonal y los modos nuevos valen 0,
      es la solucion exacta de la misma u0 en la malla fina. En 2D el relleno
      se hace eje por eje (`_zero_pad_spectrum_nd`).
    """
    if data.n not in (1, 2):
        raise ValueError(f"resample_fields solo esta implementado en n = 1 y n = 2: n = {data.n}")
    R = data.resolution
    if resolution == R:
        return data
    if resolution < R:
        if R % resolution:
            raise ValueError(f"la resolucion {resolution} debe dividir a {R}")
        step = R // resolution
        index = (Ellipsis,) + (slice(None, None, step),) * data.n
        u0, u = data.u0[index].clone(), data.u[index].clone()
    else:
        if resolution % 2:
            raise ValueError(f"la resolucion debe ser par: {resolution}")
        if data.n == 1:
            u0, u = _zero_pad_spectrum(data.u0, resolution), _zero_pad_spectrum(data.u, resolution)
        else:
            u0 = _zero_pad_spectrum_nd(data.u0, resolution, data.n)
            u = _zero_pad_spectrum_nd(data.u, resolution, data.n)
    return FieldData(u0=u0, u=u, t=data.t, ic=data.ic)

"""Generacion y particion de datos: muestras de trayectoria y puntos (k, lambda)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..symbol import Symbol
from ..utils import PhasePoints, Split, complex_dtype


def k_grid(n: int, k_max: int) -> np.ndarray:
    """Todos los k in Z^n con |k_i| <= k_max, forma (N, n)."""
    return np.array([np.array(idx) - k_max for idx in np.ndindex(*((2 * k_max + 1,) * n))])


def make_trajectory_data(
    reference: Symbol,
    k_max: int,
    n_samples: int,
    t_min: float,
    t_max: float,
    seed: int = 0,
) -> Split:
    """Genera muestras (k, t, u0, u_ref) con u_ref = exp(-t a_ref(k)) u0(k).

    k se toma al azar de la rejilla entera, t log-uniforme en [t_min, t_max]
    (asi hay senal tanto para |k| pequeno como grande) y u0 ~ normal compleja.
    """
    rng = np.random.default_rng(seed)
    grid = k_grid(reference.n, k_max)
    a_grid = np.array([reference(g)[0, 0] for g in grid])  # se evalua solo la rejilla

    idx = rng.integers(0, len(grid), size=n_samples)
    k = grid[idx]
    t = np.exp(rng.uniform(np.log(t_min), np.log(t_max), size=n_samples))
    u0 = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    u_ref = np.exp(-t * a_grid[idx]) * u0

    return {
        "k": torch.tensor(k, dtype=torch.float64),
        "t": torch.tensor(t, dtype=torch.float64),
        "u0": torch.tensor(u0, dtype=torch.complex128),
        "u_ref": torch.tensor(u_ref, dtype=torch.complex128),
    }


def split_trajectory_data(
    data: Split, val_fraction: float = 0.2, seed: int = 0
) -> Tuple[Split, Split]:
    """Particion por k en entrenamiento y validacion.

    `make_trajectory_data` muestrea k con reemplazo de una rejilla pequena, asi
    que una particion por muestra dejaria casi todos los k de validacion
    tambien en entrenamiento (fuga). Aqui se reparten los k *unicos*: cada k
    cae entero en un solo conjunto, y la validacion mide generalizacion a k no
    vistos. `val_fraction` es entonces la fraccion de k unicos, no de muestras.
    """
    unique_k, inverse = torch.unique(data["k"], dim=0, return_inverse=True)
    n_unique = unique_k.shape[0]
    perm = torch.randperm(n_unique, generator=torch.Generator().manual_seed(seed))
    n_val = int(round(val_fraction * n_unique))
    is_val = torch.zeros(n_unique, dtype=torch.bool)
    is_val[perm[:n_val]] = True
    val_mask = is_val[inverse]
    return (
        {key: value[~val_mask] for key, value in data.items()},
        {key: value[val_mask] for key, value in data.items()},
    )


def to_device(data: Split, device: torch.device, dtype: torch.dtype) -> Split:
    """Mueve un split al device/precision del modelo."""
    cdtype = complex_dtype(dtype)
    return {
        key: value.to(device=device, dtype=cdtype if value.is_complex() else dtype)
        for key, value in data.items()
    }


def sample_phase_points(
    n: int,
    k_max: int,
    batch_size: int,
    lam_min: float,
    lam_max: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> PhasePoints:
    """Muestra (k, lambda) con Re(lambda) >= 0.

    |lambda| es log-uniforme en [lam_min, lam_max] y arg(lambda) in [-pi/2, pi/2].
    La mitad de las muestras cae en la frontera Re(lambda) = 0.
    """
    k = torch.randint(-k_max, k_max + 1, (batch_size, n), generator=generator).to(torch.float64)
    u = torch.rand(batch_size, generator=generator, dtype=torch.float64)
    radius = lam_min * (lam_max / lam_min) ** u
    phase = np.pi * (torch.rand(batch_size, generator=generator, dtype=torch.float64) - 0.5)
    half = batch_size // 2
    phase[:half] = torch.sign(phase[:half]) * np.pi / 2  # frontera Re(lambda) = 0
    lam = torch.polar(radius, phase)
    return (
        k.to(device=device, dtype=dtype),
        lam.to(device=device, dtype=complex_dtype(dtype)),
    )


def sample_k_points(
    n: int,
    k_max: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Muestra k uniformes en {|k_i| <= k_max}, forma (batch, n).

    Para terminos que solo dependen de k (la cota de la Def. 2.1), con un
    muestreo propio, independiente de las muestras de trayectoria.
    """
    k = torch.randint(-k_max, k_max + 1, (batch_size, n), generator=generator)
    return k.to(device=device, dtype=dtype)


def phase_grid(
    n: int,
    k_max: int,
    lam_values: Sequence[complex],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> PhasePoints:
    """Producto cartesiano de la rejilla de k con una lista de lambda.

    Sirve como conjunto fijo y determinista para decidir si un kappa se satisface.
    """
    k = k_grid(n, k_max)
    lam = np.asarray(lam_values, dtype=complex)
    return (
        torch.tensor(np.repeat(k, len(lam), axis=0), dtype=dtype, device=device),
        torch.tensor(np.tile(lam, len(k)), dtype=complex_dtype(dtype), device=device),
    )

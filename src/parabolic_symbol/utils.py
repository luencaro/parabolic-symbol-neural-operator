"""Utilidades compartidas del paquete.

Este modulo no importa nada del resto de `parabolic_symbol`: es la raiz del
grafo de dependencias, asi que cualquier otro modulo puede importarlo sin
riesgo de ciclos. Aqui va tambien `finite_diff` (diferencias finitas genericas),
pendiente de agregar.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterator, List, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
import torch
from torch import nn

# Un conjunto de datos (train o val) y un conjunto de puntos (k, lambda).
# Viven aqui, y no en data/, porque losses.py los necesita sin depender de data/
Split = Dict[str, torch.Tensor]
PhasePoints = Tuple[torch.Tensor, torch.Tensor]


@runtime_checkable
class LearnedSymbol(Protocol):
    """Interfaz del simbolo aprendido (ParabolicMLPSymbol).

    Es lo unico que usan losses.py, training.py, evaluation.py y sweep.py, y
    permite anotar tipos sin importar `symbols/` (evita ciclos).
    `net` es el modulo de torch con los pesos entrenables: se le cambia el
    modo train/eval y se guarda su `state_dict` en el early stopping.
    """

    n: int
    m: float
    rho: int
    hidden: Tuple[int, ...]
    dropout: float
    dropout_layers: Tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    net: nn.Module

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def decay_parameters(self) -> List[nn.Parameter]: ...

    def forward(self, k: torch.Tensor) -> torch.Tensor: ...

    def evaluate_batch(self, k: np.ndarray) -> np.ndarray: ...

    def theoretical_kappa(self, omega: float = 1.0) -> float: ...


DropoutPlacement = Union[str, Sequence[int]]

# Donde colocar el dropout, indexado sobre las capas ocultas (0 = la primera).
# La tasa es fija; lo que se barre es la ubicacion.
DROPOUT_PLACEMENTS = ("none", "all", "middle", "first", "last")


def default_device() -> torch.device:
    """GPU si hay, si no CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dropout_layer_indices(placement: DropoutPlacement, n_hidden: int) -> Tuple[int, ...]:
    """Resuelve una ubicacion de dropout a indices de capa oculta.

    "middle" excluye la primera y la ultima oculta, asi que es vacio si hay
    menos de tres capas. Tambien se acepta una lista explicita de indices.
    """
    if not isinstance(placement, str):
        indices = tuple(sorted({int(i) for i in placement}))
        if any(not 0 <= i < n_hidden for i in indices):
            raise ValueError(f"indices {indices} fuera de rango para {n_hidden} capas ocultas")
        return indices
    if placement == "none":
        return ()
    if placement == "all":
        return tuple(range(n_hidden))
    if placement == "middle":
        return tuple(range(1, n_hidden - 1))
    if placement == "first":
        return (0,) if n_hidden else ()
    if placement == "last":
        return (n_hidden - 1,) if n_hidden else ()
    raise ValueError(f"ubicacion desconocida {placement!r}; usar una de {DROPOUT_PLACEMENTS}")


def bracket(k: np.ndarray) -> np.ndarray:
    """<k> = (1 + |k|^2)^{1/2}, para un lote de k de forma (N, n)."""
    return np.sqrt(1.0 + np.sum(np.asarray(k, dtype=float) ** 2, axis=1))


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def multi_indices(n: int, rho: int) -> List[Tuple[int, ...]]:
    """Todos los multiindices alpha in N_0^n con |alpha| <= rho (Def. 2.1).

    Ordenados por |alpha| y luego lexicograficamente, asi el listado es estable
    entre corridas. Incluye alpha = 0, que es la cota sobre el propio a(k).
    """
    return sorted(
        (alpha for alpha in itertools.product(range(rho + 1), repeat=n) if sum(alpha) <= rho),
        key=lambda alpha: (sum(alpha), alpha),
    )

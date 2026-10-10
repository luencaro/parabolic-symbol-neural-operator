"""Utilidades compartidas del paquete.

Este modulo no importa nada del resto de `parabolic_symbol`: es la raiz del
grafo de dependencias, asi que cualquier otro modulo puede importarlo sin
riesgo de ciclos. Aqui va tambien `finite_diff` (diferencias finitas genericas),
pendiente de agregar.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterator, List, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from torch import nn

# Un conjunto de datos (train o val) y un conjunto de puntos (k, lambda).
# Viven aqui, y no en data.py, porque losses.py los necesita sin depender de data.py.
Split = Dict[str, torch.Tensor]
PhasePoints = Tuple[torch.Tensor, torch.Tensor]


@runtime_checkable
class LearnedSymbol(Protocol):
    """Interfaz comun de los simbolos aprendidos (MLPSymbol, ParabolicMLPSymbol).

    Es lo unico que usan losses.py, training.py, evaluation.py y sweep.py, asi
    que cualquier simbolo que la cumpla se entrena sin tocar esas funciones.
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

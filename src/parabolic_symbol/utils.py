"""Utilidades compartidas del paquete.

Este modulo no importa nada del resto de `parabolic_symbol`: es la raiz del
grafo de dependencias, asi que cualquier otro modulo puede importarlo sin
riesgo de ciclos. Aqui va tambien `finite_diff` (diferencias finitas genericas),
pendiente de agregar.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

# Un conjunto de datos (train o val) y un conjunto de puntos (k, lambda).
# Viven aqui, y no en data.py, porque losses.py los necesita sin depender de data.py.
Split = Dict[str, torch.Tensor]
PhasePoints = Tuple[torch.Tensor, torch.Tensor]


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128

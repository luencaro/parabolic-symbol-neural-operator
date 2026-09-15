"""Simbolo del semigrupo del calor: a(k) = |k|^2, E = C (m = 2)."""
from __future__ import annotations

import numpy as np

from ..Symbol import Symbol


class HeatSymbol(Symbol):
    """a(k) = |k|^2, correspondiente a -Delta en T^n. Caso escalar (E=C)."""

    def __init__(self, n: int):
        super().__init__(n=n, m=2.0, rho=3, dim=1, name="HeatSymbol")

    def evaluate(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        value = np.sum(k**2)
        return np.array([[value]], dtype=complex)

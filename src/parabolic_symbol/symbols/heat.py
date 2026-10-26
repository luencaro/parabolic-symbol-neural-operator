"""Simbolo del semigrupo del calor: a(k) = D |k|^2, E = C (m = 2)."""
from __future__ import annotations

import numpy as np

from ..symbol import Symbol


class HeatSymbol(Symbol):
    """a(k) = D |k|^2, correspondiente a -D Delta en T^n. Caso escalar (E=C).

    `diffusivity` es D (1 por defecto: a(k) = |k|^2, la convencion del repo).
    La convencion del paper base (Sec. 5.1), en T = [0, 1) con difusividad
    0.05, es la del repo con D = 4 pi^2 * 0.05 (cambio x = 2 pi y).
    """

    def __init__(self, n: int, diffusivity: float = 1.0):
        if diffusivity <= 0:
            raise ValueError(f"la difusividad debe ser > 0: {diffusivity}")
        name = "HeatSymbol" if diffusivity == 1.0 else f"HeatSymbol(diffusivity={diffusivity:.6g})"
        super().__init__(n=n, m=2.0, rho=3, dim=1, name=name)
        self.diffusivity = float(diffusivity)

    def evaluate(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        value = self.diffusivity * np.sum(k**2)
        return np.array([[value]], dtype=complex)

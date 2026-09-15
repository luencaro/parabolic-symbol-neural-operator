"""Simbolo del laplaciano fraccionario (-Delta)^{alpha/2}: a(k) = |k|^alpha."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..Symbol import Symbol


class FractionalLaplacianSymbol(Symbol):
    """a(k) = |k|^alpha, E = C, m = alpha.

    Cuando alpha/2 no es entero, (-Delta)^{alpha/2} no es un operador
    diferencial sino genuinamente pseudo-diferencial.
    """

    def __init__(self, n: int, alpha: float, rho: Optional[int] = None):
        if alpha <= 0:
            raise ValueError("alpha debe ser > 0.")
        super().__init__(
            n=n,
            m=alpha,
            rho=rho if rho is not None else int(np.ceil(alpha)) + 1,
            dim=1,
            name=f"FractionalLaplacianSymbol(alpha={alpha})",
        )
        self.alpha = alpha

    def evaluate(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        norm_k = np.sqrt(np.sum(k**2))
        value = norm_k**self.alpha
        return np.array([[value]], dtype=complex)

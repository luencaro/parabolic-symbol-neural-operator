"""Simbolos que NO son parabolicos: Tome de ejemplo Symbols completamente imaginarios
a(k)+lambda*Id deja de ser invertible (biyectiva)
   en puntos concretos del dominio exigido por la definicion (Re(lambda)>=0),
   violando la condicion de biyectividad misma, no solo la cota.
"""
from __future__ import annotations

import numpy as np

from ..Symbol import Symbol

class SlowDecaySymbol(Symbol):
    """a(k) = |k| (crecimiento lineal), pero declara m = 4.

    La resolvente real decae como |k|^{-1}, pero la cota exigida decae como
    |k|^{-4}; para |k| grande el cociente crece sin limite, asi que ninguna
    kappa fija satisface la desigualdad en todo el dominio.
    """

    def __init__(self, n: int):
        super().__init__(n=n, m=4.0, rho=5, dim=1, name="SlowDecaySymbol")

    def evaluate(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        value = np.sqrt(np.sum(k**2))
        return np.array([[value]], dtype=complex)

class PurelyImaginarySymbol(Symbol):
    """a(k) = i|k|, m = 1.

    Para lambda = -i|k| (que tiene Re(lambda) = 0, dentro del dominio
    exigido), a(k) + lambda*Id = 0, que no es invertible: el simbolo falla
    la condicion de biyectividad de la definicion, no solo la cota de norma.
    """

    def __init__(self, n: int):
        super().__init__(n=n, m=1.0, rho=2, dim=1, name="PurelyImaginarySymbol")

    def evaluate(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        norm_k = np.sqrt(np.sum(k**2))
        return np.array([[1j * norm_k]], dtype=complex)

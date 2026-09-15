"""Definicion abstracta de un simbolo a: Z^n -> L(E)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Symbol(ABC):
    """Clase base abstracta para un simbolo a(k) de un multiplicador de
    Fourier operador-valuado sobre el toro T^n, con a(k) in L(E) y
    dim(E) = dim.

    Las subclases deben implementar `evaluate`.

    Parameters
    n : int
        Dimension del toro T^n (k vive en Z^n).
    m : float
        Orden del simbolo.
    rho : int
        Parametro de regularidad de la clase S^{m,rho} (tipicamente rho >= m+1).
    dim : int
        Dimension del espacio E sobre el que actua a(k) (dim=1 => caso escalar).
    name : str, opcional
        Nombre legible para reportes/tests.
    """

    def __init__(
        self,
        n: int,
        m: float,
        rho: int,
        dim: int = 1,
        name: Optional[str] = None,
    ) -> None:
        if n < 1:
            raise ValueError("n debe ser un entero >= 1.")
        if rho < 1:
            raise ValueError("rho debe ser un entero >= 1 (usualmente rho >= m+1).")
        if dim < 1:
            raise ValueError("dim (dimension de E) debe ser >= 1.")

        self.n = n
        self.m = m
        self.rho = rho
        self.dim = dim
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(self, k: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, k: np.ndarray) -> np.ndarray:
        return self.evaluate(np.asarray(k))

    def __repr__(self) -> str:  # pragma: no cover - solo para reportes
        return f"{self.name}(n={self.n}, m={self.m}, rho={self.rho}, dim={self.dim})"
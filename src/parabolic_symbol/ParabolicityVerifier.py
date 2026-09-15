"""Verificacion numerica de la desigualdad de parabolicidad (Capa 1).

Dado un Symbol a, con m su orden, se verifica:

    ||(a(k) + lambda * Id_E)^{-1}||_{L(E)} <= kappa * (1 + |(k, lambda)|^2)^{-m/2}

para (k, lambda) in Z^n x C, con Re(lambda) >= 0 y |(k, lambda)| >= omega,
donde |(k, lambda)| := (|k|^2 + |lambda|^{2/m})^{1/2}.
"""
from __future__ import annotations

from typing import Iterable, List, NamedTuple

import numpy as np

from .Symbol import Symbol


class ParabolicityResult(NamedTuple):
    """Resultado de evaluar la desigualdad en un punto (k, lambda)."""

    k: np.ndarray
    lam: complex
    phase_norm: float          # |(k, lambda)|
    resolvent_norm: float      # ||(a(k) + lambda*Id_E)^{-1}||
    bound: float                # kappa * (1 + |(k, lambda)|^2)^{-m/2}
    ratio: float                 # resolvent_norm / bound (debe ser <= 1)
    holds: bool                  # True si la desigualdad se cumple en este punto


class ParabolicityVerifier:
    """Verifica / estima las constantes de parabolicidad (omega, kappa)
    para un `Symbol` dado.
    """

    def __init__(self, symbol: Symbol):
        self.symbol = symbol


    def phase_space_norm(self, k: np.ndarray, lam: complex) -> float:
        """|(k, lambda)| := (|k|^2 + |lambda|^{2/m})^{1/2}."""
        k = np.asarray(k, dtype=float)
        return float(np.sqrt(np.sum(k**2) + np.abs(lam) ** (2.0 / self.symbol.m)))

    def resolvent(self, k: np.ndarray, lam: complex) -> np.ndarray:
        """Calcula (a(k) + lambda * Id_E)^{-1}"""
        A = self.symbol(k) + lam * np.eye(self.symbol.dim, dtype=complex)
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"a(k) + lambda*Id no es invertible en k={k}, lambda={lam}"
            ) from exc

    @staticmethod
    def operator_norm(A: np.ndarray) -> float:
        """Norma de operador (norma espectral / 2-norma inducida)."""
        return float(np.linalg.norm(A, ord=2))

    # ------------------------------------------------------------------
    # Evaluacion puntual y por region
    # ------------------------------------------------------------------
    def evaluate_point(self, k: np.ndarray, lam: complex, kappa: float) -> ParabolicityResult:
        """Evalua la desigualdad en un unico punto (k, lambda), dado kappa."""
        k = np.asarray(k)
        phase_norm = self.phase_space_norm(k, lam)
        resolvent = self.resolvent(k, lam)
        resolvent_norm = self.operator_norm(resolvent)
        bound = kappa * (1.0 + phase_norm**2) ** (-self.symbol.m / 2.0)
        ratio = resolvent_norm / bound if bound > 0 else np.inf
        return ParabolicityResult(
            k=k,
            lam=lam,
            phase_norm=phase_norm,
            resolvent_norm=resolvent_norm,
            bound=bound,
            ratio=ratio,
            holds=ratio <= 1.0,
        )

    def check_region(
        self,
        k_values: Iterable[np.ndarray],
        lam_values: Iterable[complex],
        omega: float,
        kappa: float,
    ) -> List[ParabolicityResult]:
        """Evalua la desigualdad en una rejilla de (k, lambda), restringida a
        Re(lambda) >= 0 y |(k, lambda)| >= omega. Los puntos fuera de esa
        region se ignoran (la definicion no los exige).
        """
        results: List[ParabolicityResult] = []
        for k in k_values:
            for lam in lam_values:
                lam = complex(lam)
                if lam.real < 0:
                    continue
                phase_norm = self.phase_space_norm(k, lam)
                if phase_norm < omega:
                    continue
                results.append(self.evaluate_point(k, lam, kappa))
        return results

    def estimate_kappa(
        self,
        k_values: Iterable[np.ndarray],
        lam_values: Iterable[complex],
        omega: float,
    ) -> float:
        """Estima (por rejilla, no analiticamente) la kappa minima consistente
        con la region muestreada:

            kappa_hat = sup_{(k,lambda) en la rejilla, region valida}
                        ||(a(k)+lambda*Id)^{-1}|| * (1+|(k,lambda)|^2)^{m/2}

        Nota: esto es una COTA INFERIOR de la kappa real (una rejilla finita
        puede no capturar el peor caso). Sirve para tener un valor de
        referencia con el que despues correr `check_region`.
        """
        worst = 0.0
        for k in k_values:
            for lam in lam_values:
                lam = complex(lam)
                if lam.real < 0:
                    continue
                phase_norm = self.phase_space_norm(k, lam)
                if phase_norm < omega:
                    continue
                resolvent_norm = self.operator_norm(self.resolvent(k, lam))
                candidate = resolvent_norm * (1.0 + phase_norm**2) ** (self.symbol.m / 2.0)
                worst = max(worst, candidate)
        return worst
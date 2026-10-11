"""Verificacion numerica de la pertenencia a la clase discreta S^{m,rho} (Def. 2.1).

Dado un Symbol a de orden m, la Def. 2.1 (Denk-Nau) pide que para todo
multiindice alpha in N_0^n con |alpha| <= rho exista C_alpha >= 0 tal que

    ||Delta^alpha_k a(k)||_{L(E)} <= C_alpha * <k>^{m - |alpha|}   para todo k in Z^n,

con <k> = (1 + |k|^2)^{1/2}, Delta_{k_j} a(k) = a(k + e_j) - a(k) (diferencia
hacia adelante) y Delta^alpha = Delta_{k_1}^{alpha_1} ... Delta_{k_n}^{alpha_n}.
Se toma rho = symbol.rho.

ESTO NO ES UNA PRUEBA. La definicion cuantifica sobre todo Z^n y aqui solo se
mira la ventana finita {|k|_inf <= K}. Un alpha marcado como "consistente" dice
unicamente que, dentro de la ventana explorada, (i) la constante C_alpha_hat(K)
deja de crecer al ampliar K y (ii) el crecimiento observado por capas de radio
no supera <k>^{m - |alpha|}. Fuera de la ventana el simbolo puede hacer
cualquier cosa; en particular una red neuronal solo esta restringida donde se
la entreno.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..symbol import Symbol
from ..data.synthetic import k_grid
from ..utils import bracket, multi_indices


class SymbolClassResult(NamedTuple):
    """Evidencia numerica para un multiindice alpha."""

    alpha: Tuple[int, ...]
    order: int                      # |alpha|
    c_hat: Dict[int, float]         # K -> C_alpha_hat(K), la constante por ventana
    growth: float                   # C_hat(K_ultimo) / C_hat(K_anterior)
    stable: bool                    # growth <= 1 + tol_stable
    slope: float                    # pendiente log-log por capas (nan si Delta^alpha a = 0)
    slope_theoretical: float        # m - |alpha|, la que exige la definicion
    consistent: bool                # stable y la pendiente no supera la teorica


class SymbolClassVerifier:
    """Estima las constantes C_alpha de la Def. 2.1 para un `Symbol` dado.

    Funciona con cualquier `Symbol` (usa `symbol(k)`); si el simbolo es escalar
    y expone `evaluate_batch`, se usa esa via para evaluar por lotes. Los valores
    de a(k) se cachean, asi que compartir un verificador entre varios alpha y
    varias ventanas K no repite evaluaciones.
    """

    def __init__(self, symbol: Symbol):
        self.symbol = symbol
        self._cache: Dict[Tuple[float, ...], np.ndarray] = {}
        self._batch = (
            symbol.evaluate_batch
            if symbol.dim == 1 and hasattr(symbol, "evaluate_batch")
            else None
        )

    # ------------------------------------------------------------------
    # Evaluacion de a(k) y de sus diferencias
    # ------------------------------------------------------------------
    def multi_indices(self, rho: Optional[int] = None) -> List[Tuple[int, ...]]:
        """Todos los alpha in N_0^n con |alpha| <= rho (por defecto symbol.rho)."""
        return multi_indices(self.symbol.n, self.symbol.rho if rho is None else rho)

    def values(self, k_values: np.ndarray) -> np.ndarray:
        """a(k) para un lote de k; forma (N, dim, dim) compleja, con cache."""
        k_values = np.asarray(k_values, dtype=float).reshape(-1, self.symbol.n)
        claves = [tuple(row) for row in k_values]
        nuevas = [clave for clave in dict.fromkeys(claves) if clave not in self._cache]
        if nuevas:
            puntos = np.array(nuevas, dtype=float)
            if self._batch is not None:
                calculadas = np.asarray(self._batch(puntos), dtype=complex).reshape(-1, 1, 1)
            else:
                calculadas = np.array([self.symbol(p) for p in puntos], dtype=complex)
            self._cache.update(zip(nuevas, calculadas))
        return np.array([self._cache[clave] for clave in claves], dtype=complex)

    def norms(self, values: np.ndarray) -> np.ndarray:
        """||.||_{L(E)} punto a punto: modulo si dim = 1, norma espectral si no."""
        if self.symbol.dim == 1:
            return np.abs(values[:, 0, 0])
        return np.array([float(np.linalg.norm(matriz, ord=2)) for matriz in values])

    def difference(self, k_values: np.ndarray, alpha: Sequence[int]) -> np.ndarray:
        """Delta^alpha a(k) por diferencias hacia adelante, formula binomial:

            Delta^alpha a(k) = sum_{beta <= alpha} (-1)^{|alpha - beta|} C(alpha, beta) a(k + beta).
        """
        k_values = np.asarray(k_values, dtype=float).reshape(-1, self.symbol.n)
        alpha = tuple(int(a) for a in alpha)
        total = np.zeros((len(k_values), self.symbol.dim, self.symbol.dim), dtype=complex)
        for beta in itertools.product(*(range(a + 1) for a in alpha)):
            coeficiente = (-1.0) ** (sum(alpha) - sum(beta))
            for a_j, b_j in zip(alpha, beta):
                coeficiente *= comb(a_j, b_j)
            total += coeficiente * self.values(k_values + np.array(beta, dtype=float))
        return total

    def scaled_norms(self, k_values: np.ndarray, alpha: Sequence[int]) -> np.ndarray:
        """<k>^{|alpha| - m} ||Delta^alpha a(k)||: el cociente que debe estar acotado."""
        order = sum(int(a) for a in alpha)
        normas = self.norms(self.difference(k_values, alpha))
        return bracket(k_values) ** (order - self.symbol.m) * normas

    # ------------------------------------------------------------------
    # Constantes, estabilidad y pendiente
    # ------------------------------------------------------------------
    def estimate_constant(self, alpha: Sequence[int], k_max: int) -> float:
        """C_alpha_hat(K) = sup_{|k|_inf <= K} <k>^{|alpha| - m} ||Delta^alpha a(k)||."""
        return float(self.scaled_norms(k_grid(self.symbol.n, k_max), alpha).max())

    def constants(self, alpha: Sequence[int], k_maxs: Sequence[int]) -> Dict[int, float]:
        """C_alpha_hat(K) para una sucesion creciente de ventanas K."""
        return {int(k_max): self.estimate_constant(alpha, k_max) for k_max in k_maxs}

    def log_log_slope(
        self, alpha: Sequence[int], k_max: int, min_radius: Optional[int] = None
    ) -> float:
        """Pendiente log-log de max_{|k|_inf = r} ||Delta^alpha a(k)|| frente a <k>.

        Se parte la rejilla en capas de radio r y, en cada capa, se toma el punto
        donde ||Delta^alpha a|| es maximo; la pendiente sale de ajustar una recta
        a (log <k>, log max). Si el simbolo esta en la clase, esa pendiente no
        supera m - |alpha|.

        El ajuste usa solo la mitad exterior de las capas (r >= k_max // 2 por
        defecto). La cota de la Def. 2.1 es una afirmacion asintotica en |k|, y
        cerca del origen <k> ~ 1 comprime el eje x y sesga la pendiente hacia
        arriba: con a(k) = |k|^3 y todas las capas la recta da 3.29 en vez de 3.

        Devuelve nan si quedan menos de dos capas con diferencia no nula. Eso
        pasa cuando Delta^alpha a se anula en la cola (un polinomio de grado
        < |alpha|, o el pico de un simbolo con esquina en el origen): entonces
        ||Delta^alpha a|| tiene soporte compacto y cualquier C_alpha sirve.
        """
        min_radius = max(1, k_max // 2) if min_radius is None else min_radius
        k_values = k_grid(self.symbol.n, k_max)
        radios = np.max(np.abs(k_values), axis=1)
        normas = self.norms(self.difference(k_values, alpha))
        corchetes = bracket(k_values)

        xs: List[float] = []
        ys: List[float] = []
        for r in range(min_radius, k_max + 1):
            capa = radios == r
            if not capa.any():
                continue
            pico = int(np.argmax(normas[capa]))
            if normas[capa][pico] <= 0.0:
                continue
            xs.append(float(np.log(corchetes[capa][pico])))
            ys.append(float(np.log(normas[capa][pico])))
        if len(xs) < 2:
            return float("nan")
        return float(np.polyfit(xs, ys, 1)[0])

    def check(
        self,
        k_maxs: Sequence[int] = (4, 8, 12, 16),
        tol_slope: float = 0.25,
        tol_stable: float = 0.05,
        atol: float = 1e-8,
    ) -> List[SymbolClassResult]:
        """Evidencia de pertenencia a S^{m,rho}, un resultado por multiindice.

        `k_maxs` son las ventanas crecientes con las que se mira si C_alpha_hat(K)
        se estabiliza (hacen falta al menos dos). `tol_stable` es el crecimiento
        relativo maximo admitido entre las dos ultimas ventanas y `tol_slope` el
        margen que se le da a la pendiente log-log por encima de m - |alpha|.

        Todo C_alpha_hat(K) por debajo de `atol` se toma como 0: es ruido de
        redondeo de una diferencia que deberia anularse (p. ej. |k|^2 calculado
        como (|k|^2)^{1/2} elevado a 2). Sin el umbral, el crecimiento relativo de
        ese ruido (1e-13 -> 1e-11) marcaria el multiindice como inestable.

        El criterio es de una sola cola: la definicion pide una COTA superior, asi
        que una pendiente menor que m - |alpha| (el simbolo es de orden mas bajo
        que el declarado) tambien es consistente.
        """
        k_maxs = [int(k_max) for k_max in k_maxs]
        if len(k_maxs) < 2:
            raise ValueError("hacen falta al menos dos ventanas K para ver estabilidad")
        if sorted(k_maxs) != k_maxs:
            raise ValueError(f"las ventanas K deben ser crecientes: {k_maxs}")

        resultados: List[SymbolClassResult] = []
        for alpha in self.multi_indices():
            c_hat = {
                k_max: (c if c >= atol else 0.0)
                for k_max, c in self.constants(alpha, k_maxs).items()
            }
            ultima, previa = c_hat[k_maxs[-1]], c_hat[k_maxs[-2]]
            if previa > 0.0:
                growth = ultima / previa
            else:  # C_alpha_hat = 0 en ambas ventanas: la cota se cumple trivialmente
                growth = 1.0 if ultima <= 0.0 else float("inf")
            stable = growth <= 1.0 + tol_stable

            order = sum(alpha)
            # Si C_alpha_hat es 0 (salvo redondeo) la pendiente del ruido no dice nada
            slope = self.log_log_slope(alpha, k_maxs[-1]) if ultima > 0.0 else float("nan")
            slope_theoretical = self.symbol.m - order
            consistent = stable and (
                np.isnan(slope) or slope <= slope_theoretical + tol_slope
            )
            resultados.append(
                SymbolClassResult(
                    alpha=alpha,
                    order=order,
                    c_hat=c_hat,
                    growth=growth,
                    stable=stable,
                    slope=slope,
                    slope_theoretical=slope_theoretical,
                    consistent=consistent,
                )
            )
        return resultados

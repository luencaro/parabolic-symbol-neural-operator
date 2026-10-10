"""Estimacion desde los datos de ENTRENAMIENTO: simbolo a(k), banda (c, C) y C_alpha.

Las trayectorias cumplen u(t, k) = exp(-t a(k)) u0(k), asi que para cada k

    log|u(t, k) / u0(k)| = -t a(k)

es una recta por el origen en t. `estimate_symbol` la ajusta por minimos
cuadrados ponderados y da a_hat(k) en cada k de la particion de entrenamiento.
De ahi salen:

- `estimate_bounds`: la banda c < a(k)/<k>^m < C que usa ParabolicMLPSymbol;
- `estimate_class_constants`: las C_alpha de la Def. 2.1 por diferencias finitas
  de a_hat, como alternativa a calcularlas con el simbolo exacto.

`oracle_bounds` y `oracle_class_constants` hacen lo mismo con el simbolo exacto
del repo, solo como referencia. Nada de este modulo mira la validacion.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from .Symbol import Symbol
from .SymbolClassVerifier import SymbolClassVerifier
from .data import k_grid
from .utils import Split, multi_indices

ClassConstantsDict = Dict[Tuple[int, ...], float]

# De donde salen las C_alpha de L_class (ver `resolve_class_constants`).
CLASS_C_SOURCES = ("given", "oracle", "data")


class KWindow(NamedTuple):
    """Resumen de un conjunto de k: cuantos son y que region de Z^n cubren."""

    n_k: int
    k_inf_max: int          # max |k|_inf
    norm_min: float         # min |k|
    norm_max: float         # max |k|

    def __str__(self) -> str:
        return (
            f"{self.n_k} k, |k|_inf <= {self.k_inf_max}, "
            f"|k| in [{self.norm_min:.3g}, {self.norm_max:.3g}]"
        )


class Bounds(NamedTuple):
    """Banda c < a(k)/<k>^m < C y la ventana de k con que se calculo."""

    c: float
    C: float
    window: KWindow


def k_window(k: np.ndarray) -> KWindow:
    """KWindow de un lote de k de forma (N, n)."""
    k = np.asarray(k, dtype=float)
    norms = np.linalg.norm(k, axis=1)
    return KWindow(
        n_k=len(k),
        k_inf_max=int(np.max(np.abs(k))) if len(k) else 0,
        norm_min=float(norms.min()) if len(k) else float("nan"),
        norm_max=float(norms.max()) if len(k) else float("nan"),
    )


def bracket(k: np.ndarray) -> np.ndarray:
    """<k> = (1 + |k|^2)^{1/2}, para un lote de k de forma (N, n)."""
    return np.sqrt(1.0 + np.sum(np.asarray(k, dtype=float) ** 2, axis=1))


# ----------------------------------------------------------------------
# 1. a_hat(k) por minimos cuadrados ponderados
# ----------------------------------------------------------------------
def estimate_symbol(
    train_data: Split, tol: float = 1e-10, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """a_hat(k) en cada k de entrenamiento, por minimos cuadrados ponderados.

    Para cada k se ajusta la recta por el origen y = -t a(k), con
    y = log|u(t,k) / u0(k)|, sobre todos los t y todas las muestras de ese k,
    con pesos w = |u0(k)|^2 (una muestra con u0 chico tiene un cociente ruidoso):

        a_hat(k) = - sum_i w_i t_i y_i / sum_i w_i t_i^2.

    Mascara de recuperabilidad: se descartan los pares con |u| <= tol |u0|,
    porque exp(-t a(k)) puede caer por debajo de la precision de maquina y el
    log deja de medir a(k). Un k sin ningun par recuperable no se estima.

    Devuelve (k, a_hat): k de forma (K, n) (los k unicos con algun par util) y
    a_hat de forma (K,). Con `verbose`, imprime la ventana de k usada y cuantos
    pares se descartaron.
    """
    k = train_data["k"].detach().cpu().numpy().astype(float)
    t = train_data["t"].detach().cpu().numpy().astype(float)
    u0 = np.abs(train_data["u0"].detach().cpu().numpy())
    u = np.abs(train_data["u_ref"].detach().cpu().numpy())

    n_k_total = len(np.unique(k, axis=0))
    keep = (u0 > 0) & (u > tol * u0)
    k, t, u0, u = k[keep], t[keep], u0[keep], u[keep]

    y = np.log(u) - np.log(u0)
    w = u0**2
    k_unique, inverse = np.unique(k, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    num = np.bincount(inverse, weights=w * t * y, minlength=len(k_unique))
    den = np.bincount(inverse, weights=w * t**2, minlength=len(k_unique))
    a_hat = -num / den

    if verbose:
        print(
            f"estimate_symbol: {int(keep.sum()):,} de {len(keep):,} pares (k, t) "
            f"recuperables (tol={tol:g}); {len(k_unique)} de {n_k_total} k de entrenamiento"
        )
        print(f"  ventana usada: {k_window(k_unique)}")
    return k_unique, a_hat


# ----------------------------------------------------------------------
# 2. Banda (c, C)
# ----------------------------------------------------------------------
def _bounds_from_values(k: np.ndarray, a: np.ndarray, m: float, delta: float) -> Bounds:
    """c = (1 - delta) min h, C = (1 + delta) max h, con h(k) = a(k)/<k>^m, k != 0."""
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta debe estar en [0, 1): {delta}")
    k = np.asarray(k, dtype=float).reshape(len(a), -1)
    a = np.real(np.asarray(a))
    nonzero = np.any(k != 0, axis=1)
    if not nonzero.any():
        raise ValueError("no hay ningun k != 0 con que estimar la banda")
    k, a = k[nonzero], a[nonzero]
    h = a / bracket(k) ** m
    c = (1.0 - delta) * float(h.min())
    C = (1.0 + delta) * float(h.max())
    if not 0.0 < c < C:
        raise ValueError(f"la banda estimada no cumple 0 < c < C: c={c:.4g}, C={C:.4g}")
    return Bounds(c=c, C=C, window=k_window(k))


def estimate_bounds(k: np.ndarray, a_hat: np.ndarray, m: float, delta: float = 0.1) -> Bounds:
    """Banda (c, C) desde a_hat, con un margen relativo `delta` a cada lado:

        h_hat(k) = a_hat(k) / <k>^m   (k != 0),
        c = (1 - delta) min h_hat,    C = (1 + delta) max h_hat.

    Valida 0 < c < C. Devuelve Bounds(c, C, window), con window la ventana de
    k != 0 usada.
    """
    return _bounds_from_values(k, a_hat, m, delta)


def symbol_values(symbol: Symbol, k: np.ndarray) -> np.ndarray:
    """Parte real de a(k) sobre un lote, con la convencion del repo: symbol(k)[0, 0]."""
    k = np.asarray(k, dtype=float).reshape(-1, symbol.n)
    if hasattr(symbol, "evaluate_batch"):
        return np.real(symbol.evaluate_batch(k))
    return np.array([np.real(symbol(row)[0, 0]) for row in k])


def oracle_bounds(
    symbol: Symbol, k_window: Union[int, np.ndarray], m: float, delta: float = 0.1
) -> Bounds:
    """La misma banda que `estimate_bounds`, pero con el simbolo exacto (oraculo).

    `k_window` es un entero K (la rejilla |k_i| <= K) o un lote explicito de k,
    p. ej. los k de entrenamiento que devuelve `estimate_symbol`. Solo es una
    referencia: el modelo se entrena con la banda que sale de los datos.
    """
    k = k_grid(symbol.n, k_window) if np.isscalar(k_window) else np.asarray(k_window, dtype=float)
    return _bounds_from_values(k, symbol_values(symbol, k), m, delta)


# ----------------------------------------------------------------------
# 3. Constantes C_alpha de la Def. 2.1
# ----------------------------------------------------------------------
def class_constants_with_margin(
    C_raw: ClassConstantsDict, margin: float = 1.5, zero_tol: float = 1e-8
) -> ClassConstantsDict:
    """Margen y piso sobre unas C_alpha estimadas, como en el notebook original.

    Un C_alpha nulo (Delta^alpha a = 0, p. ej. |alpha| = 3 para el calor) dejaria
    a L_class exigiendo Delta^alpha a_theta = 0 exacto; se sustituye por la menor
    constante positiva (el piso). Se toma como nulo todo C_alpha <= zero_tol *
    max C_alpha, porque las diferencias de a_hat no dan 0 exacto sino ruido de
    redondeo. Despues todo se multiplica por `margin`.
    """
    top = max(C_raw.values())
    positivas = [c for c in C_raw.values() if c > zero_tol * top]
    if not positivas:
        raise ValueError("todas las C_alpha estimadas son nulas")
    piso = min(positivas)
    return {
        alpha: float(margin * (c if c > zero_tol * top else piso)) for alpha, c in C_raw.items()
    }


def estimate_class_constants(
    k: np.ndarray, a_hat: np.ndarray, m: float, rho: int, margin: float = 1.5
) -> ClassConstantsDict:
    """C_alpha de la Def. 2.1 desde a_hat, por diferencias finitas hacia adelante.

        C_alpha_hat = max_k <k>^{|alpha| - m} |Delta^alpha a_hat(k)|,

    sobre los k de entrenamiento cuyo estencil {k + beta : beta <= alpha}
    completo esta en el conjunto (la particion por k deja huecos, asi que no se
    usan todos los k). Despues se aplican el piso y el margen de
    `class_constants_with_margin`.
    """
    k = np.asarray(k, dtype=float)
    n = k.shape[1]
    valores = {tuple(row): float(np.real(a)) for row, a in zip(k.astype(int), a_hat)}
    corchetes = bracket(k)
    C_raw: ClassConstantsDict = {}
    for alpha in multi_indices(n, rho):
        orden = sum(alpha)
        mejor = 0.0
        usados = 0
        for row, corchete in zip(k.astype(int), corchetes):
            diferencia = 0.0
            completo = True
            for beta in itertools.product(*(range(a + 1) for a in alpha)):
                vecino = tuple(int(r + b) for r, b in zip(row, beta))
                if vecino not in valores:
                    completo = False
                    break
                coeficiente = (-1.0) ** (orden - sum(beta))
                for a_j, b_j in zip(alpha, beta):
                    coeficiente *= comb(a_j, b_j)
                diferencia += coeficiente * valores[vecino]
            if completo:
                usados += 1
                mejor = max(mejor, corchete ** (orden - m) * abs(diferencia))
        if usados == 0:
            raise ValueError(f"ningun k de entrenamiento tiene el estencil completo para alpha={alpha}")
        C_raw[alpha] = mejor
    return class_constants_with_margin(C_raw, margin)


def oracle_class_constants(
    reference: Symbol, k_max: int, margin: float = 1.5, rho: Optional[int] = None
) -> ClassConstantsDict:
    """C_alpha del simbolo exacto en |k_i| <= k_max (SymbolClassVerifier), con margen.

    Es la receta del notebook original. `rho` (por defecto el de la referencia)
    fija que multiindices se calculan: pasar el rho del simbolo aprendido.
    """
    verifier = SymbolClassVerifier(reference)
    C_raw = {
        alpha: verifier.estimate_constant(alpha, k_max) for alpha in verifier.multi_indices(rho)
    }
    return class_constants_with_margin(C_raw, margin)


def resolve_class_constants(
    source: str,
    given,
    train_data: Split,
    m: float,
    rho: int,
    k_max: int,
    reference: Optional[Symbol] = None,
    margin: float = 1.5,
):
    """Las C_alpha que usa L_class, segun `source`:

    - "given":  las de `given` tal cual (comportamiento original: un escalar o
                un diccionario {alpha: C_alpha});
    - "oracle": las del simbolo exacto `reference` en |k_i| <= k_max;
    - "data":   las de las diferencias finitas de a_hat sobre los k de
                entrenamiento (`estimate_symbol` + `estimate_class_constants`).

    En "oracle" y "data" se piden todos los alpha con |alpha| <= rho del
    simbolo aprendido.
    """
    if source == "given":
        return given
    if source == "oracle":
        if reference is None:
            raise ValueError("class_C_source='oracle' necesita el simbolo de referencia")
        return oracle_class_constants(reference, k_max, margin, rho)
    if source == "data":
        k, a_hat = estimate_symbol(train_data, verbose=False)
        return estimate_class_constants(k, a_hat, m, rho, margin)
    raise ValueError(f"fuente de C_alpha desconocida {source!r}; usar una de {CLASS_C_SOURCES}")

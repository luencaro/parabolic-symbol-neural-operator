"""Configuracion del PDNO con simbolos parabolicos (Shin et al., 2024, ec. (1)).

    v_0 = P(u0),   v_{l+1} = sigma(W_l v_l + K_l v_l + b_l),   u(t) = Q(v_L)
    K_l v = F^{-1}[ exp(-tau_l t a_par^{(l)}(k)) F v ]

con a_par^{(l)} un ParabolicMLPSymbol par (escalar, aplicado a todos los
canales), tau_l >= 0 y sum_l tau_l = 1. Todas las capas comparten la banda
(c, C) y el orden m.

Los valores por defecto son el caso del calor y del laplaciano fraccionario:
d = 1, sin W, sin sesgos, sigma = identidad, P = Q = identidad. Entonces la red
es el semigrupo u(t) = exp(-t a_bar(D)) u0 con a_bar = sum_l tau_l a_par^{(l)},
una combinacion convexa que sigue en la banda (mismo kappa); con L = 1 es el
modelo espectral de siempre, exp(-t a_par(k)) u0_hat(k).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..utils import DropoutPlacement

ACTIVATIONS = ("identity", "gelu", "tanh")
CHANNEL_MAPS = ("identity", "linear")   # P y Q


@dataclass
class PDNOConfig:
    """Arquitectura del PDNO. Ver el docstring del modulo.

    `width` es d, el numero de canales. `bias` agrega b_l in R^d en cada capa
    y el sesgo de P y Q cuando son lineales. `learn_tau=False` fija
    tau_l = 1/L; con True, tau = softmax(logits), con logits que empiezan en 0.
    P = identidad o Q = identidad exigen d = 1.
    """

    n: int
    m: float
    c: float
    C: float
    width: int = 1
    n_layers: int = 1
    use_W: bool = False
    bias: bool = False
    activation: str = "identity"
    lift: str = "identity"          # P
    project: str = "identity"       # Q
    learn_tau: bool = False
    symbol_hidden: Tuple[int, ...] = (64, 64, 64)
    symbol_dropout: float = 0.0
    symbol_dropout_layers: DropoutPlacement = "none"
    rho: Optional[int] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n debe ser >= 1: {self.n}")
        if self.width < 1:
            raise ValueError(f"el ancho d debe ser >= 1: {self.width}")
        if self.n_layers < 1:
            raise ValueError(f"el numero de capas L debe ser >= 1: {self.n_layers}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activacion desconocida {self.activation!r}; usar una de {ACTIVATIONS}")
        for name, value in (("lift", self.lift), ("project", self.project)):
            if value not in CHANNEL_MAPS:
                raise ValueError(f"{name} desconocido {value!r}; usar uno de {CHANNEL_MAPS}")
            if value == "identity" and self.width != 1:
                raise ValueError(f"{name}='identity' exige d = 1; se recibio d = {self.width}")
        self.symbol_hidden = tuple(self.symbol_hidden)

    @property
    def is_semigroup(self) -> bool:
        """True en el caso lineal: u(t) = exp(-t a_bar(D)) u0 (ver el docstring del modulo)."""
        return (
            self.width == 1 and not self.use_W and not self.bias and self.activation == "identity"
            and self.lift == "identity" and self.project == "identity"
        )

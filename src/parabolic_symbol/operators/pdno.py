"""PDNO con simbolos parabolicos (Shin et al., 2024, ec. (1)).

    v_0 = P(u0),   v_{l+1} = sigma(W_l v_l + K_l v_l + b_l),   u(t) = Q(v_L)
    K_l v = F^{-1}[ exp(-tau_l t a_par^{(l)}(k)) F v ]

con v_l de forma (B, d, *malla). P, Q y W_l actuan punto a punto sobre los
canales; K_l es un PDIO con su propio ParabolicMLPSymbol par. t entra en el
exponente, no como canal. La configuracion esta en `config.py`; con los valores
por defecto (caso lineal) la red es el semigrupo exp(-t a_bar(D)) u0, con
a_bar = sum_l tau_l a_par^{(l)}.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import nn

from ..symbols.parabolic_mlp_symbol import ParabolicMLPSymbol
from .config import PDNOConfig
from .pdio import PDIO
from ..utils import default_device

_ACTIVATIONS = {"identity": nn.Identity, "gelu": nn.GELU, "tanh": nn.Tanh}


def _pointwise(layer: nn.Linear, v: torch.Tensor) -> torch.Tensor:
    """Aplica una Linear sobre el eje de canales de v: (B, d_in, *malla) -> (B, d_out, *malla)."""
    return torch.movedim(layer(torch.movedim(v, 1, -1)), -1, 1)


class PDNO(nn.Module):
    """La red global del PDNO; ver el docstring del modulo.

    `symbols[l]` es el ParabolicMLPSymbol (even=True) de la capa l: lo usan
    L_par y L_class. Sus `net` quedan registradas a traves de `kernels`
    (nn.ModuleList de PDIO), asi que parameters() y state_dict() incluyen todo.
    Se construye directamente en `device` / `dtype` (GPU y float64 por
    defecto); no se debe mover despues con .to(...) (ver PDIO). La semilla se
    fija una vez, antes de construir las capas.
    """

    def __init__(
        self, config: PDNOConfig, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        L, d = config.n_layers, config.width
        factory = dict(device=self.device, dtype=dtype)

        torch.manual_seed(config.seed)
        self.symbols: List[ParabolicMLPSymbol] = [
            ParabolicMLPSymbol(
                n=config.n, m=config.m, c=config.c, C=config.C, hidden=config.symbol_hidden,
                rho=config.rho, dropout=config.symbol_dropout,
                dropout_layers=config.symbol_dropout_layers, device=self.device, dtype=dtype,
                even=True,
            )
            for _ in range(L)
        ]
        self.kernels = nn.ModuleList(PDIO(symbol) for symbol in self.symbols)
        self.W = (
            nn.ModuleList(nn.Linear(d, d, bias=False, **factory) for _ in range(L))
            if config.use_W else None
        )
        self.b = (
            nn.ParameterList(nn.Parameter(torch.zeros(d, **factory)) for _ in range(L))
            if config.bias else None
        )
        self.P = nn.Linear(1, d, bias=config.bias, **factory) if config.lift == "linear" else None
        self.Q = nn.Linear(d, 1, bias=config.bias, **factory) if config.project == "linear" else None
        if config.learn_tau:
            self.tau_logits = nn.Parameter(torch.zeros(L, **factory))
        else:
            self.register_buffer("tau_fixed", torch.full((L,), 1.0 / L, **factory))
        self.activation = _ACTIVATIONS[config.activation]()

    @property
    def n(self) -> int:
        return self.config.n

    @property
    def tau(self) -> torch.Tensor:
        """(tau_1, ..., tau_L): fraccion del tiempo que avanza cada capa, tau_l >= 0, suma 1."""
        if self.config.learn_tau:
            return torch.softmax(self.tau_logits, dim=0)
        return self.tau_fixed

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u0: (B, *malla) real, t: (B,) -> u_theta(t) de forma (B, *malla)."""
        t = t.reshape(-1)
        tau = self.tau
        v = u0.unsqueeze(1)
        if self.P is not None:
            v = _pointwise(self.P, v)
        for l, kernel in enumerate(self.kernels):
            z = kernel(v, t, tau[l])
            if self.W is not None:
                z = z + _pointwise(self.W[l], v)
            if self.b is not None:
                z = z + self.b[l].reshape((1, -1) + (1,) * self.n)
            v = self.activation(z)
        if self.Q is not None:
            v = _pointwise(self.Q, v)
        return v[:, 0]

    def decay_parameters(self) -> List[nn.Parameter]:
        """Pesos con weight decay: Linear interiores de cada simbolo y pesos de W, P y Q."""
        params = [p for symbol in self.symbols for p in symbol.decay_parameters()]
        if self.W is not None:
            params += [layer.weight for layer in self.W]
        params += [layer.weight for layer in (self.P, self.Q) if layer is not None]
        return params

    def theoretical_kappa(self, omega: float = 1.0) -> float:
        """kappa de la Def. 2.3 para todas las capas (comparten la banda, asi que es el mismo)."""
        return max(symbol.theoretical_kappa(omega) for symbol in self.symbols)

    def layer_symbols(self, k: np.ndarray) -> np.ndarray:
        """a_par^{(l)}(k) de cada capa sobre un lote k de forma (N, n); forma (L, N), real."""
        return np.stack([np.real(symbol.evaluate_batch(k)) for symbol in self.symbols])

    def effective_symbol(self, k: np.ndarray) -> np.ndarray:
        """a_bar(k) = sum_l tau_l a_par^{(l)}(k), solo en el caso lineal (config.is_semigroup)."""
        if not self.config.is_semigroup:
            raise ValueError("el simbolo efectivo solo existe en el caso lineal (config.is_semigroup)")
        tau = self.tau.detach().cpu().numpy()
        return tau @ self.layer_symbols(k)

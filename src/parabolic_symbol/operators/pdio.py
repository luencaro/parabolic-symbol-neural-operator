"""PDIO: la capa espectral K del PDNO, con nucleo de semigrupo.

    K v = F^{-1}[ exp(-tau t a_par(k)) F v ],

con a_par un ParabolicMLPSymbol par (even=True), escalar y aplicado a todos los
canales, y la convencion u(x) = sum_k u_hat(k) e^{ik.x}. El multiplicador es
diagonal, asi que la normalizacion de la FFT se cancela y K v =
irfftn(mult * rfftn(v)). No se truncan modos: a_par se evalua una vez por k en
todos los modos de la malla, en el layout de rfftn (`grf_wavenumbers`).

Como a_par es par, el multiplicador es real y par, y rfftn / irfftn lo aplican
exactamente (tambien en la costura del medio espacio y en Nyquist). No se usa
fftn tomando la parte real: eso daria (e^{-ta(k)} + e^{-ta(-k)}) / 2, que no es
un semigrupo.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ..data.grf import grf_wavenumbers
from ..symbols.parabolic_mlp_symbol import ParabolicMLPSymbol


class PDIO(nn.Module):
    """K v = F^{-1}[exp(-tau t a_par(k)) F v] sobre T^n, con a_par = `symbol` (par).

    ParabolicMLPSymbol no es un nn.Module: se guarda como atributo normal y su
    `net` (el mismo objeto) se registra como submodulo, asi que sus pesos entran
    en parameters(), state_dict() y en train() / eval(). El simbolo mueve k a su
    propio device / dtype, asi que la capa se construye ya en el device y dtype
    finales: un .to(...) posterior no actualizaria esos atributos.
    """

    def __init__(self, symbol: ParabolicMLPSymbol) -> None:
        super().__init__()
        if not symbol.even:
            raise ValueError("el PDIO necesita un simbolo par: ParabolicMLPSymbol(..., even=True)")
        self.symbol = symbol
        self.net = symbol.net
        self._k: Dict[Tuple[int, ...], torch.Tensor] = {}

    def wavenumbers(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """k en el layout de rfftn para una malla `shape` = (R,) * n; forma (*espectral, n)."""
        n = self.symbol.n
        if len(shape) != n or len(set(shape)) != 1:
            raise ValueError(f"se esperaba una malla (R,) * {n}: {shape}")
        if shape not in self._k:
            k = grf_wavenumbers(shape[0], n)
            self._k[shape] = torch.from_numpy(k).to(device=self.symbol.device, dtype=self.symbol.dtype)
        return self._k[shape]

    def symbol_on_grid(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """a_par(k) en todos los modos de rfftn de la malla `shape`; forma espectral."""
        k = self.wavenumbers(shape)
        return self.symbol.forward(k.reshape(-1, self.symbol.n)).reshape(k.shape[:-1])

    def forward(self, v: torch.Tensor, t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """v: (B, d, *malla), t: (B,), tau escalar -> K v de forma (B, d, *malla).

        El multiplicador exp(-tau t_b a_par(k)) tiene forma (B, 1, *espectral):
        el simbolo se evalua una vez y se reutiliza en todo el lote y los canales.
        """
        n = self.symbol.n
        shape = tuple(v.shape[-n:])
        axes = tuple(range(-n, 0))
        a = self.symbol_on_grid(shape)
        scale = (tau * t).reshape((-1,) + (1,) * (v.dim() - 1))
        multiplier = torch.exp(-scale * a)
        return torch.fft.irfftn(multiplier * torch.fft.rfftn(v, dim=axes), s=shape, dim=axes)

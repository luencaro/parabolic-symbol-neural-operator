"""Terminos de la perdida: trayectoria y parabolicidad (Def. 2.3).

    L_traj = E[ |exp(-t a_theta(k)) u0(k) - u_ref(k, t)|^2 ]
    L_par  = E[ max(0, |(a_theta(k) + lambda)^{-1}| - kappa (1 + |(k,lambda)|^2)^{-m/2})^2 ]
             sobre Re(lambda) >= 0, |(k,lambda)| >= omega.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .symbols.mlp_symbol import MLPSymbol


def trajectory_loss(symbol: MLPSymbol, k, t, u0, u_ref) -> torch.Tensor:
    """MSE complejo: E[|exp(-t a_theta(k)) u0 - u_ref|^2]."""
    u_pred = torch.exp(-t * symbol.forward(k)) * u0
    return torch.mean(torch.abs(u_pred - u_ref) ** 2)


def phase_space_norm(k: torch.Tensor, lam: torch.Tensor, m: float) -> torch.Tensor:
    """|(k, lambda)| = (|k|^2 + |lambda|^{2/m})^{1/2}."""
    return torch.sqrt(torch.sum(k**2, dim=1) + torch.abs(lam) ** (2.0 / m))


def parabolicity_terms(
    symbol: MLPSymbol, k, lam, omega: float, kappa: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (violacion, ratio) en los puntos dentro del dominio de la Def. 2.3.

    violacion = max(0, |(a+lambda)^{-1}| - cota),  ratio = |(a+lambda)^{-1}| / cota.
    La desigualdad se cumple en un punto si ratio <= 1.
    """
    m = symbol.m
    norm = phase_space_norm(k, lam, m)
    mask = (lam.real >= 0) & (norm >= omega)
    if not mask.any():
        empty = torch.zeros(0, dtype=symbol.dtype, device=symbol.device)
        return empty, empty

    k, lam, norm = k[mask], lam[mask], norm[mask]
    resolvent = 1.0 / torch.abs(symbol.forward(k) + lam)
    bound = kappa * (1.0 + norm**2) ** (-m / 2.0)
    return torch.relu(resolvent - bound), resolvent / bound


def parabolicity_loss(
    symbol: MLPSymbol, k, lam, omega: float, kappa: float
) -> torch.Tensor:
    """Hinge al cuadrado sobre la desigualdad de parabolicidad (Def. 2.3)."""
    violation, _ = parabolicity_terms(symbol, k, lam, omega, kappa)
    if violation.numel() == 0:
        return torch.zeros((), dtype=symbol.dtype, device=symbol.device)
    return torch.mean(violation**2)

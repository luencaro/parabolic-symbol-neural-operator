"""Terminos de la perdida: trayectoria y parabolicidad (Def. 2.3).

    L_traj = E[ |exp(-t a_theta(k)) u0(k) - u_ref(k, t)|^2 ]
    L_par  = E[ max(0, |(a_theta(k) + lambda)^{-1}| - kappa (1 + |(k,lambda)|^2)^{-m/2})^2 ]
             sobre Re(lambda) >= 0, |(k,lambda)| >= omega.
    L_class= E[ max(0, <k>^{|alpha| - m} |Delta^alpha a_theta(k)| - C_alpha)^2 ]
             sobre |alpha| <= rho (Def. 2.1).
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Dict, Sequence, Tuple, Union

import torch

from ..utils import LearnedSymbol, multi_indices

# C_alpha de la Def. 2.1: un escalar para todos los alpha, o uno por multiindice.
ClassConstants = Union[float, Dict[Tuple[int, ...], float]]


def trajectory_loss(symbol: LearnedSymbol, k, t, u0, u_ref) -> torch.Tensor:
    """MSE complejo: E[|exp(-t a_theta(k)) u0 - u_ref|^2]."""
    u_pred = torch.exp(-t * symbol.forward(k)) * u0
    return torch.mean(torch.abs(u_pred - u_ref) ** 2)


def phase_space_norm(k: torch.Tensor, lam: torch.Tensor, m: float) -> torch.Tensor:
    """|(k, lambda)| = (|k|^2 + |lambda|^{2/m})^{1/2}."""
    return torch.sqrt(torch.sum(k**2, dim=1) + torch.abs(lam) ** (2.0 / m))


def parabolicity_terms(
    symbol: LearnedSymbol, k, lam, omega: float, kappa: float
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
    symbol: LearnedSymbol, k, lam, omega: float, kappa: float
) -> torch.Tensor:
    """Hinge al cuadrado sobre la desigualdad de parabolicidad (Def. 2.3)."""
    violation, _ = parabolicity_terms(symbol, k, lam, omega, kappa)
    if violation.numel() == 0:
        return torch.zeros((), dtype=symbol.dtype, device=symbol.device)
    return torch.mean(violation**2)


def class_constants(
    alphas: Sequence[Tuple[int, ...]], C: ClassConstants
) -> Dict[Tuple[int, ...], float]:
    """Normaliza C a un diccionario {alpha: C_alpha} sobre los alpha pedidos."""
    if not isinstance(C, dict):
        return {alpha: float(C) for alpha in alphas}
    faltan = [alpha for alpha in alphas if alpha not in C]
    if faltan:
        raise ValueError(f"faltan constantes C_alpha para los multiindices {faltan}")
    return {alpha: float(C[alpha]) for alpha in alphas}


def class_terms(
    symbol: LearnedSymbol, k, C: ClassConstants
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (violacion, ratio) de la cota de la Def. 2.1, por alpha y por muestra.

    violacion = max(0, <k>^{|alpha| - m} |Delta^alpha a_theta(k)| - C_alpha),
    ratio     = <k>^{|alpha| - m} |Delta^alpha a_theta(k)| / C_alpha.
    La desigualdad se cumple en un punto si ratio <= 1. Ambos tensores tienen
    forma (A, N): A multiindices con |alpha| <= symbol.rho y N muestras de k.

    Las diferencias salen de `symbol.forward`, asi que el termino es
    diferenciable. Los desplazamientos que hacen falta son exactamente los
    mismos multiindices (beta <= alpha implica |beta| <= rho), asi que todos los
    k + beta se evaluan en un unico lote.
    """
    alphas = tuple(multi_indices(symbol.n, symbol.rho))
    constantes = class_constants(alphas, C)
    posicion = {alpha: i for i, alpha in enumerate(alphas)}

    desplazamientos = torch.tensor(alphas, dtype=k.dtype, device=k.device)
    lote = (k.unsqueeze(0) + desplazamientos.unsqueeze(1)).reshape(-1, symbol.n)
    valores = symbol.forward(lote).reshape(len(alphas), -1)  # (A, N), a_theta(k + beta)

    bracket = torch.sqrt(1.0 + torch.sum(k**2, dim=1))
    violacion = []
    ratio = []
    for alpha in alphas:
        diferencia = torch.zeros_like(valores[0])
        for beta in itertools.product(*(range(a + 1) for a in alpha)):
            coeficiente = (-1.0) ** (sum(alpha) - sum(beta))
            for a_j, b_j in zip(alpha, beta):
                coeficiente *= comb(a_j, b_j)
            diferencia = diferencia + coeficiente * valores[posicion[beta]]
        escalada = bracket ** (sum(alpha) - symbol.m) * torch.abs(diferencia)
        violacion.append(torch.relu(escalada - constantes[alpha]))
        ratio.append(escalada / constantes[alpha])
    return torch.stack(violacion), torch.stack(ratio)


def class_loss(symbol: LearnedSymbol, k, C: ClassConstants) -> torch.Tensor:
    """Hinge al cuadrado sobre la cota de la Def. 2.1.

    L_class = E[ max(0, <k>^{|alpha| - m} |Delta^alpha a_theta(k)| - C_alpha)^2 ],
    promediado sobre las muestras k y sobre todos los alpha con |alpha| <= rho.
    """
    violation, _ = class_terms(symbol, k, C)
    if violation.numel() == 0:
        return torch.zeros((), dtype=symbol.dtype, device=symbol.device)
    return torch.mean(violation**2)

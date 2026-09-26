"""Metricas sobre conjuntos completos: trayectoria, simbolo y parabolicidad."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from .Symbol import Symbol
from .data import to_device
from .losses import parabolicity_terms
from .symbols.mlp_symbol import MLPSymbol
from .utils import PhasePoints, Split

if TYPE_CHECKING:  # solo para anotar: en runtime seria el ciclo training <-> evaluation
    from .training import TrainingConfig

# Las seis metricas que `evaluate_split` devuelve y que `train` registra por epoca.
METRICS = ("total", "trajectory", "parabolicity", "rel_l2", "par_max_ratio", "par_violation")


@torch.no_grad()
def evaluate_split(
    symbol: MLPSymbol, data: Split, phase: PhasePoints, config: "TrainingConfig"
) -> Dict[str, float]:
    """Metricas de un conjunto completo, siempre en modo eval (dropout apagado)."""
    was_training = symbol.net.training
    symbol.net.eval()

    residual = torch.exp(-data["t"] * symbol.forward(data["k"])) * data["u0"] - data["u_ref"]
    sq_error = torch.sum(torch.abs(residual) ** 2)
    traj = sq_error / residual.numel()
    rel_l2 = torch.sqrt(sq_error / torch.sum(torch.abs(data["u_ref"]) ** 2))

    violation, ratio = parabolicity_terms(symbol, *phase, config.omega, config.kappa)
    par = torch.mean(violation**2) if violation.numel() else torch.zeros_like(traj)
    max_ratio = ratio.max() if ratio.numel() else torch.zeros_like(traj)
    par_violation = (ratio > 1.0).to(traj.dtype).mean() if ratio.numel() else torch.zeros_like(traj)

    symbol.net.train(was_training)
    return {
        "total": float(config.lambda_trajectory * traj + config.lambda_parabolicity * par),
        "trajectory": float(traj),
        "parabolicity": float(par),
        "rel_l2": float(rel_l2),
        "par_max_ratio": float(max_ratio),
        "par_violation": float(par_violation),
    }


@torch.no_grad()
def trajectory_report(symbol: MLPSymbol, data: Split) -> Dict[str, float]:
    """Metricas de regresion de la trayectoria sobre un conjunto completo.

    La prediccion es compleja, asi que el error de cada muestra es el modulo del
    residuo |u_pred - u_ref| y todas las metricas se derivan de el. `r2` usa la
    varianza compleja de u_ref como referencia.
    """
    was_training = symbol.net.training
    symbol.net.eval()
    data = to_device(data, symbol.device, symbol.dtype)

    pred = torch.exp(-data["t"] * symbol.forward(data["k"])) * data["u0"]
    ref = data["u_ref"]
    err = torch.abs(pred - ref)
    sq_sum = torch.sum(err**2)
    mse = sq_sum / err.numel()

    symbol.net.train(was_training)
    return {
        "n": float(err.numel()),
        "mse": float(mse),
        "rmse": float(torch.sqrt(mse)),
        "mae": float(err.mean()),
        "max_error": float(err.max()),
        "rel_l2": float(torch.sqrt(sq_sum / torch.sum(torch.abs(ref) ** 2))),
        "r2": float(1.0 - sq_sum / torch.sum(torch.abs(ref - ref.mean()) ** 2)),
    }


def symbol_report(symbol: MLPSymbol, reference: Symbol, k_values) -> Dict[str, float]:
    """Metricas de a_theta(k) frente al simbolo de referencia sobre una rejilla."""
    k = np.asarray(k_values, dtype=float).reshape(-1, symbol.n)
    a_ref = np.array([reference(row)[0, 0] for row in k])
    err = np.abs(symbol.evaluate_batch(k) - a_ref)
    return {
        "n": float(err.size),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(err.mean()),
        "max_error": float(err.max()),
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(a_ref)),
    }

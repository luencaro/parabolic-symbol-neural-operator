"""Metricas del PDNO: error relativo L2 en espacio fisico y simbolos de cada capa.

El error de una muestra (u0_i, t_j) es ||u_theta(t_j) - u(t_j)|| / ||u(t_j)||
en la malla completa, como `field_rel_l2`. L_par y L_class se evaluan sobre el
simbolo par a_par^{(l)} de cada capa y se suman sobre las capas.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

from ..data.fields import FieldData
from ..operators.pdno import PDNO
from .losses import class_loss, parabolicity_terms
from ..utils import PhasePoints

if TYPE_CHECKING:  # solo para anotar: en runtime seria el ciclo operator_training <-> evaluation
    from .operator_training import OperatorTrainingConfig

# Las metricas que `evaluate_fields` devuelve y que `train_operator` registra por epoca.
OPERATOR_METRICS = ("total", "rel_l2", "parabolicity", "class", "par_max_ratio", "par_violation")


@torch.no_grad()
def pair_errors(model: PDNO, data: FieldData, batch_size: int = 64) -> torch.Tensor:
    """Error relativo L2 de cada muestra, en el orden plano p = i T + j; forma (n_pairs,).

    Usa el modo en que este el modelo; los lotes se mueven al device del modelo.
    """
    errors = []
    for start in range(0, data.n_pairs, batch_size):
        idx = torch.arange(start, min(start + batch_size, data.n_pairs), device=data.u0.device)
        u0, t, u = (x.to(device=model.device, dtype=model.dtype) for x in data.pairs(idx))
        axes = tuple(range(1, u.dim()))
        diff = torch.linalg.vector_norm(model(u0, t) - u, dim=axes)
        errors.append(diff / torch.linalg.vector_norm(u, dim=axes))
    return torch.cat(errors)


@torch.no_grad()
def field_errors(model: PDNO, data: FieldData, batch_size: int = 64) -> np.ndarray:
    """Error relativo L2 por condicion inicial y tiempo, forma (n_ic, T), en modo eval.

    Para reportar prueba y extrapolacion, como `field_rel_l2` en el modelo espectral.
    """
    was_training = model.training
    model.eval()
    errors = pair_errors(model, data, batch_size).cpu().numpy()
    model.train(was_training)
    return errors.reshape(data.n_ic, data.n_times)


@torch.no_grad()
def evaluate_fields(
    model: PDNO,
    data: FieldData,
    phase: PhasePoints,
    config: "OperatorTrainingConfig",
    class_k: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Metricas de un conjunto completo, siempre en modo eval.

    "rel_l2" es el error relativo L2 por muestra, promediado. "parabolicity" y
    "class" son sum_l L_par(a^{(l)}) y sum_l L_class(a^{(l)}) sobre los puntos
    fijos `phase` y `class_k` ("class" vale 0 si no se dan o si lambda_class = 0).
    "par_max_ratio" es el peor ratio sobre todas las capas y "par_violation" la
    fraccion de puntos (capa, (k, lambda)) con ratio > 1. "total" es
    rel_l2 + lambda_par par, y suma lambda_class class solo si
    `config.early_stopping_class`.
    """
    was_training = model.training
    model.eval()

    rel_l2 = pair_errors(model, data, config.eval_batch_size).mean()
    kappa = model.theoretical_kappa(config.omega)
    par = torch.zeros_like(rel_l2)
    ratios = []
    for symbol in model.symbols:
        violation, ratio = parabolicity_terms(symbol, *phase, config.omega, kappa)
        if violation.numel():
            par = par + torch.mean(violation**2)
            ratios.append(ratio)
    ratio = torch.cat(ratios) if ratios else torch.zeros(0, dtype=model.dtype, device=model.device)
    max_ratio = ratio.max() if ratio.numel() else torch.zeros_like(rel_l2)
    par_violation = (ratio > 1.0).to(model.dtype).mean() if ratio.numel() else torch.zeros_like(rel_l2)
    cls = torch.zeros_like(rel_l2)
    if class_k is not None and config.lambda_class:
        for symbol in model.symbols:
            cls = cls + class_loss(symbol, class_k, config.class_C)

    total = rel_l2 + config.lambda_parabolicity * par
    if config.early_stopping_class:
        total = total + config.lambda_class * cls

    model.train(was_training)
    return {
        "total": float(total),
        "rel_l2": float(rel_l2),
        "parabolicity": float(par),
        "class": float(cls),
        "par_max_ratio": float(max_ratio),
        "par_violation": float(par_violation),
    }

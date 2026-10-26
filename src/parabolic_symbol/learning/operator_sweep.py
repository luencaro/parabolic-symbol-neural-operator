"""Barrido de la MLP del simbolo del PDNO (profundidad, ancho, dropout, weight decay).

Es el barrido de `sweep.py` aplicado al PDNO: las mismas combinaciones
(`sweep_combinations`, con las equivalentes eliminadas) y el mismo criterio de
seleccion (`best_run`: menor L2 relativa de validacion). La arquitectura del
PDNO (d, L, W, sesgos, sigma, P, Q, tau) queda fija en `base_model`; cada corrida
solo cambia la MLP g_theta de los simbolos (symbol_hidden, symbol_dropout,
symbol_dropout_layers) y el weight decay. Todas las corridas usan las mismas
semillas de modelo (base_model.seed) y de entrenamiento (base_config.seed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch

from ..data.fields import FieldData
from ..operators.config import PDNOConfig
from ..operators.pdno import PDNO
from .operator_evaluation import OPERATOR_METRICS
from .operator_training import OperatorTrainingConfig, OperatorTrainingResult, train_operator
from .sweep import sweep_combinations
from ..utils import DropoutPlacement, Split, default_device


@dataclass
class OperatorSweepRun:
    n_layers: int                # capas ocultas de g_theta
    width: int
    dropout_rate: float          # tasa efectiva: 0.0 si ninguna capa lleva dropout
    dropout_layers: str          # etiqueta de la ubicacion ("none", "middle", ...)
    dropout_indices: Tuple[int, ...]  # capas ocultas que efectivamente llevan dropout
    weight_decay: float
    # Las metricas de validacion en la mejor epoca (las de OPERATOR_METRICS).
    val_total: float
    val_rel_l2: float
    val_parabolicity: float
    val_class: float
    val_par_max_ratio: float
    val_par_violation: float
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    seconds: float
    model: PDNO = field(repr=False)
    result: OperatorTrainingResult = field(repr=False)
    config: OperatorTrainingConfig = field(repr=False)  # la de la corrida, con su weight_decay

    @property
    def dropout_desc(self) -> str:
        """Como se imprime la configuracion de dropout en los logs (igual que SweepRun)."""
        if not self.dropout_indices or self.dropout_rate == 0.0:
            return "OFF"
        capas = ",".join(str(i) for i in self.dropout_indices)
        return f"ON p={self.dropout_rate:g} capas[{capas}]"


def run_operator_sweep(
    train_data: FieldData,
    val_data: FieldData,
    base_model: PDNOConfig,
    base_config: OperatorTrainingConfig,
    depths: Sequence[int],
    widths: Sequence[int],
    dropout: float = 0.2,
    dropout_layers: Sequence[DropoutPlacement] = ("none",),
    weight_decays: Sequence[float] = (0.0,),
    rows: Optional[Split] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[OperatorSweepRun]:
    """Entrena un PDNO por cada (profundidad, ancho, ubicacion del dropout, weight decay).

    Cada corrida usa replace(base_model, symbol_hidden=(ancho,) * capas,
    symbol_dropout=dropout, symbol_dropout_layers=indices) y
    replace(base_config, weight_decay=wd), y se entrena con
    `train_operator(model, train_data, val_data, config, rows=rows)`. Los
    argumentos de los ejes son los de `run_sweep` (acepta **SWEEP). Con
    `verbose` se imprime una linea por corrida y se silencian los logs por
    epoca; `OperatorSweepRun.config` guarda la configuracion sin ese cambio.
    """
    device = device if device is not None else default_device()
    combos = sweep_combinations(depths, widths, dropout_layers, weight_decays)
    runs: List[OperatorSweepRun] = []

    total = len(combos)
    for i, (n_layers, width, placement, indices, weight_decay) in enumerate(combos, 1):
        arch = replace(
            base_model, symbol_hidden=(width,) * n_layers, symbol_dropout=dropout,
            symbol_dropout_layers=indices,
        )
        model = PDNO(arch, device=device, dtype=dtype)
        config = replace(base_config, weight_decay=weight_decay)

        start = time.perf_counter()
        result = train_operator(
            model, train_data, val_data, replace(config, log_every=0) if verbose else config, rows=rows,
        )
        seconds = time.perf_counter() - start

        val = {key: result.history["val"][key][result.best_epoch - 1] for key in OPERATOR_METRICS}
        run = OperatorSweepRun(
            n_layers=n_layers,
            width=width,
            dropout_rate=dropout if indices else 0.0,
            dropout_layers=placement if isinstance(placement, str) else str(indices),
            dropout_indices=indices,
            weight_decay=weight_decay,
            val_total=val["total"],
            val_rel_l2=val["rel_l2"],
            val_parabolicity=val["parabolicity"],
            val_class=val["class"],
            val_par_max_ratio=val["par_max_ratio"],
            val_par_violation=val["par_violation"],
            best_epoch=result.best_epoch,
            epochs_run=result.epochs_run,
            stopped_early=result.stopped_early,
            seconds=seconds,
            model=model,
            result=result,
            config=config,
        )
        runs.append(run)

        if verbose:
            capas_drop = ",".join(str(j) for j in indices) if run.dropout_rate else "-"
            print(
                f"[{i:3d}/{total}] {n_layers}x{width:<3d} "
                f"drop[{capas_drop}] wd={weight_decay:.0e} | "
                f"rel_l2={run.val_rel_l2:.4f} | {seconds:5.1f}s"
            )

    return runs


def operator_sweep_table(runs: Sequence[OperatorSweepRun]):
    """Resumen del barrido como DataFrame: las columnas de `sweep_table` sin val_trajectory."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "capas": run.n_layers,
                "ancho": run.width,
                "dropout": run.dropout_desc,
                "dropout_layers": run.dropout_layers,
                "dropout_rate": run.dropout_rate,
                "weight_decay": run.weight_decay,
                "val_total": run.val_total,
                "val_parabolicity": run.val_parabolicity,
                "val_rel_l2": run.val_rel_l2,
                "val_par_max_ratio": run.val_par_max_ratio,
                "val_par_violation": run.val_par_violation,
                "val_class": run.val_class,
                "mejor_epoca": run.best_epoch,
                "epocas": run.epochs_run,
                "segundos": run.seconds,
            }
            for run in runs
        ]
    )

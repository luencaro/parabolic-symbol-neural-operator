"""Barrido de hiperparametros de ParabolicMLPSymbol (profundidad, ancho, dropout, weight decay).

kappa no se barre: la Def. 2.3 se cumple por construccion con
kappa = theoretical_kappa(omega), y ese es el kappa de L_par en cada corrida.
`best_run` elige la corrida de menor error relativo de validacion.
`sweep_combinations` genera las combinaciones (tambien la usa el barrido del
PDNO, `operator_sweep.py`).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch

from ..symbols.parabolic_mlp_symbol import ParabolicMLPSymbol
from .training import METRICS, TrainingConfig, TrainingResult, train
from ..utils import DropoutPlacement, Split, default_device, dropout_layer_indices


@dataclass
class SweepRun:
    kappa: float                 # theoretical_kappa(omega) del simbolo
    n_layers: int
    width: int
    dropout_rate: float          # tasa efectiva: 0.0 si ninguna capa lleva dropout
    dropout_layers: str          # etiqueta de la ubicacion ("none", "all", ...)
    dropout_indices: Tuple[int, ...]  # capas ocultas que efectivamente llevan dropout
    weight_decay: float
    # Las metricas de validacion en la mejor epoca (las mismas de METRICS).
    val_total: float
    val_trajectory: float
    val_parabolicity: float
    val_rel_l2: float
    val_par_max_ratio: float
    val_par_violation: float
    val_class: float
    best_epoch: int
    epochs_run: int
    seconds: float
    symbol: ParabolicMLPSymbol = field(repr=False)
    result: TrainingResult = field(repr=False)

    @property
    def dropout_desc(self) -> str:
        """Como se imprime la configuracion de dropout en los logs."""
        if not self.dropout_indices or self.dropout_rate == 0.0:
            return "OFF"
        capas = ",".join(str(i) for i in self.dropout_indices)
        return f"ON p={self.dropout_rate:g} capas[{capas}]"


def sweep_combinations(
    depths: Sequence[int],
    widths: Sequence[int],
    dropout_layers: Sequence[DropoutPlacement],
    weight_decays: Sequence[float],
) -> List[Tuple[int, int, DropoutPlacement, Tuple[int, ...], float]]:
    """Combinaciones (capas, ancho, ubicacion, indices, weight_decay) del barrido.

    Producto cartesiano de los ejes, en ese orden, sin las combinaciones que
    resuelven a las mismas capas de dropout (p. ej. "middle" con 2 capas ocultas
    no deja ninguna capa interior y equivale a "none"); `indices` son las capas
    ocultas que efectivamente llevan dropout.
    """
    combos = []
    seen = set()
    for n_layers, width, placement, weight_decay in itertools.product(
        depths, widths, dropout_layers, weight_decays
    ):
        indices = dropout_layer_indices(placement, n_layers)
        key = (n_layers, width, indices, weight_decay)
        if key in seen:
            continue
        seen.add(key)
        combos.append((n_layers, width, placement, indices, weight_decay))
    return combos


def run_sweep(
    train_data: Split,
    val_data: Split,
    n: int,
    m: float,
    bounds: Tuple[float, float],
    depths: Sequence[int],
    widths: Sequence[int],
    base_config: TrainingConfig,
    dropout: float = 0.2,
    dropout_layers: Sequence[DropoutPlacement] = ("none",),
    weight_decays: Sequence[float] = (0.0,),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[SweepRun]:
    """Entrena un ParabolicMLPSymbol por cada (profundidad, ancho, ubicacion, weight_decay).

    `bounds = (c, C)` es la banda del simbolo. La tasa de dropout `dropout` es
    fija; el eje que se barre es `dropout_layers`, donde se coloca ese dropout.
    Con `verbose` se imprime una linea por corrida y se silencian los logs por
    epoca de `train`.
    """
    device = device if device is not None else default_device()
    if verbose:
        base_config = replace(base_config, log_every=0)
    runs: List[SweepRun] = []

    # Se materializan las combinaciones antes de entrenar (deduplicando las que
    # resuelven a las mismas capas de dropout) solo para poder imprimir [i/total].
    combos = sweep_combinations(depths, widths, dropout_layers, weight_decays)

    total = len(combos)
    for i, (n_layers, width, placement, indices, weight_decay) in enumerate(combos, 1):
        symbol = ParabolicMLPSymbol(
            n=n, m=m, c=bounds[0], C=bounds[1], hidden=(width,) * n_layers,
            dropout=dropout, dropout_layers=indices,
            seed=base_config.seed, device=device, dtype=dtype,
        )
        config = replace(base_config, weight_decay=weight_decay)

        start = time.perf_counter()
        result = train(symbol, train_data, val_data, config)
        seconds = time.perf_counter() - start

        val = {key: result.history["val"][key][result.best_epoch - 1] for key in METRICS}
        run = SweepRun(
            kappa=symbol.theoretical_kappa(config.omega),
            n_layers=n_layers,
            width=width,
            dropout_rate=symbol.dropout if indices else 0.0,
            dropout_layers=placement if isinstance(placement, str) else str(indices),
            dropout_indices=indices,
            weight_decay=weight_decay,
            val_total=val["total"],
            val_trajectory=val["trajectory"],
            val_parabolicity=val["parabolicity"],
            val_rel_l2=val["rel_l2"],
            val_par_max_ratio=val["par_max_ratio"],
            val_par_violation=val["par_violation"],
            val_class=val["class"],
            best_epoch=result.best_epoch,
            epochs_run=result.epochs_run,
            seconds=seconds,
            symbol=symbol,
            result=result,
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


def best_run(runs: Sequence[SweepRun]) -> Optional[SweepRun]:
    """La corrida con menor val_rel_l2 (None si no hay corridas)."""
    if not runs:
        return None
    return min(runs, key=lambda run: run.val_rel_l2)


def sweep_table(runs: Sequence[SweepRun]):
    """Resumen del barrido como DataFrame (una fila por corrida)."""
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
                "val_trajectory": run.val_trajectory,
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

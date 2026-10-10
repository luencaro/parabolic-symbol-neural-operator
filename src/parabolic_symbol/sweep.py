"""Barrido de hiperparametros (kappa, profundidad, ancho, dropout, weight decay).

Con MLPSymbol (`symbol_type="mlp"`) kappa no se fija a mano: se barre una lista
de valores y se entrena una red por combinacion; `best_run` elige entre las que
cumplen la cota.

Con ParabolicMLPSymbol (`symbol_type="parabolic"`) kappa NO se barre: la Def. 2.3
se cumple por construccion con kappa = theoretical_kappa(omega), y ese es el
kappa de L_par y de la comprobacion. Se siguen barriendo profundidad, ancho,
dropout y weight decay.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch

from .losses import parabolicity_terms
from .symbols.mlp_symbol import (
    DropoutPlacement,
    MLPSymbol,
    default_device,
    dropout_layer_indices,
)
from .symbols.parabolic_mlp_symbol import ParabolicMLPSymbol
from .training import METRICS, TrainingConfig, TrainingResult, train
from .utils import LearnedSymbol, PhasePoints, Split

# Que simbolo entrena el barrido.
SYMBOL_TYPES = ("mlp", "parabolic")


@dataclass
class SweepRun:
    kappa: float
    n_layers: int
    width: int
    dropout_rate: float          # tasa efectiva: 0.0 si ninguna capa lleva dropout
    dropout_layers: str          # etiqueta de la ubicacion ("none", "all", ...)
    dropout_indices: Tuple[int, ...]  # capas ocultas que efectivamente llevan dropout
    weight_decay: float
    satisfied: bool         # la cota se cumple en TODA la rejilla de comprobacion
    max_ratio: float         # sup del ratio sobre esa rejilla (<= 1 <=> satisfied)
    # Las seis metricas de validacion en la mejor epoca (las mismas de METRICS).
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
    symbol: LearnedSymbol = field(repr=False)
    result: TrainingResult = field(repr=False)

    @property
    def dropout_desc(self) -> str:
        """Como se imprime la configuracion de dropout en los logs."""
        if not self.dropout_indices or self.dropout_rate == 0.0:
            return "OFF"
        capas = ",".join(str(i) for i in self.dropout_indices)
        return f"ON p={self.dropout_rate:g} capas[{capas}]"


@torch.no_grad()
def max_ratio_on(symbol: LearnedSymbol, phase: PhasePoints, omega: float, kappa: float) -> float:
    """Peor ratio del simbolo sobre un conjunto fijo de puntos (k, lambda)."""
    _, ratio = parabolicity_terms(symbol, *phase, omega, kappa)
    return float(ratio.max()) if ratio.numel() else 0.0


def run_sweep(
    train_data: Split,
    val_data: Split,
    n: int,
    m: float,
    kappas: Optional[Sequence[float]],
    depths: Sequence[int],
    widths: Sequence[int],
    base_config: TrainingConfig,
    check_phase: PhasePoints,
    dropout: float = 0.2,
    dropout_layers: Sequence[DropoutPlacement] = ("none",),
    weight_decays: Sequence[float] = (0.0,),
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    symbol_type: str = "mlp",
    bounds: Optional[Tuple[float, float]] = None,
) -> List[SweepRun]:
    """Entrena una red por cada (kappa, profundidad, ancho, ubicacion, weight_decay).

    `check_phase` es la rejilla fija sobre la que se decide si un kappa se
    satisface (max_ratio <= 1), independiente de los puntos usados en la perdida.

    La tasa de dropout `dropout` es fija; el eje que se barre es
    `dropout_layers`, donde se coloca ese dropout.

    `symbol_type="parabolic"` entrena ParabolicMLPSymbol con la banda
    `bounds = (c, C)` (obligatoria). Entonces `kappas` debe ser None: cada
    corrida usa kappa = symbol.theoretical_kappa(base_config.omega), y
    `input_scale` / `output_scale` no se usan (la escala la da <k>^m).
    """
    if symbol_type not in SYMBOL_TYPES:
        raise ValueError(f"symbol_type desconocido {symbol_type!r}; usar uno de {SYMBOL_TYPES}")
    if symbol_type == "parabolic":
        if bounds is None:
            raise ValueError("symbol_type='parabolic' necesita bounds=(c, C)")
        if kappas is not None:
            raise ValueError(
                "con symbol_type='parabolic' kappa no se barre (es theoretical_kappa): "
                "pasar kappas=None"
            )
        kappas = (None,)
    elif kappas is None:
        raise ValueError("con symbol_type='mlp' hace falta la malla de kappas")

    device = device if device is not None else default_device()
    runs: List[SweepRun] = []

    # Se materializan las combinaciones antes de entrenar (mismo orden y misma
    # deduplicacion que antes) solo para poder imprimir [i/total].
    combos = []
    seen = set()
    for kappa, n_layers, width, placement, weight_decay in itertools.product(
        kappas, depths, widths, dropout_layers, weight_decays
    ):
        indices = dropout_layer_indices(placement, n_layers)
        key = (kappa, n_layers, width, indices, weight_decay)
        if key in seen:
            continue
        seen.add(key)
        combos.append((kappa, n_layers, width, placement, indices, weight_decay))

    total = len(combos)
    for i, (kappa, n_layers, width, placement, indices, weight_decay) in enumerate(combos, 1):
        if symbol_type == "parabolic":
            symbol = ParabolicMLPSymbol(
                n=n, m=m, c=bounds[0], C=bounds[1], hidden=(width,) * n_layers,
                dropout=dropout, dropout_layers=indices,
                seed=base_config.seed, device=device, dtype=dtype,
            )
            kappa = symbol.theoretical_kappa(base_config.omega)
        else:
            symbol = MLPSymbol(
                n=n, m=m, hidden=(width,) * n_layers,
                dropout=dropout, dropout_layers=indices,
                input_scale=input_scale, output_scale=output_scale,
                seed=base_config.seed, device=device, dtype=dtype,
            )
        config = replace(base_config, kappa=kappa, weight_decay=weight_decay)

        start = time.perf_counter()
        result = train(symbol, train_data, val_data, config)
        seconds = time.perf_counter() - start

        max_ratio = max_ratio_on(symbol, check_phase, config.omega, kappa)
        val = {key: result.history["val"][key][result.best_epoch - 1] for key in METRICS}
        run = SweepRun(
            kappa=kappa,
            n_layers=n_layers,
            width=width,
            dropout_rate=symbol.dropout if indices else 0.0,
            dropout_layers=placement if isinstance(placement, str) else str(indices),
            dropout_indices=indices,
            weight_decay=weight_decay,
            satisfied=max_ratio <= 1.0,
            max_ratio=max_ratio,
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
            # Una linea por corrida: lo demas queda en el SweepRun y en sweep_table.
            capas_drop = ",".join(str(j) for j in indices) if run.dropout_rate else "-"
            print(
                f"[{i:3d}/{total}] kappa={kappa:6.3f} {n_layers}x{width:<3d} "
                f"drop[{capas_drop}] wd={weight_decay:.0e} | "
                f"{'CUMPLE   ' if run.satisfied else 'NO cumple'} "
                f"rel_l2={run.val_rel_l2:.4f} | {seconds:5.1f}s"
            )

    return runs


def best_run(runs: Sequence[SweepRun]) -> Optional[SweepRun]:
    """La corrida con mejor val_rel_l2 entre TODAS las que cumplen la cota.

    kappa ya no ordena la seleccion: cualquier corrida con `satisfied` cumple la
    Def. 2.3 con su propio kappa, asi que entre ellas se elige la que mejor
    aproxima la trayectoria, no la del kappa mas chico.
    """
    feasible = [run for run in runs if run.satisfied]
    if not feasible:
        return None
    return min(feasible, key=lambda run: run.val_rel_l2)


def sweep_table(runs: Sequence[SweepRun]):
    """Resumen del barrido como DataFrame (una fila por corrida)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "kappa": run.kappa,
                "capas": run.n_layers,
                "ancho": run.width,
                "dropout": run.dropout_desc,
                "dropout_layers": run.dropout_layers,
                "dropout_rate": run.dropout_rate,
                "weight_decay": run.weight_decay,
                "cumple": run.satisfied,
                "max_ratio": run.max_ratio,
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

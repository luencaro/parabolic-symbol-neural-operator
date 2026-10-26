"""Entrenamiento, barrido y modelo guardado del FNO de comparacion.

`train_baseline` repite el esquema de `train_operator` sin los terminos del
simbolo (no hay L_par ni L_class):

    L = E_b ||u_theta(t_b) - u(t_b)|| / ||u(t_b)||   (`relative_l2_loss`),

con la misma `OperatorTrainingConfig`: AdamW con `lr`, decaimiento exponencial
hasta `lr_final`, `batch_size` pares (u0, t) por lote, `epochs`, `grad_clip`,
`patience` y `min_delta`, y early stopping sobre la L2 relativa de validacion
(aqui "total" = "rel_l2"). Asi, frente al PDNO, la unica diferencia es la
arquitectura. `weight_decay` se aplica a todos los parametros. Los campos de
`OperatorTrainingConfig` del simbolo (omega, lambdas, C_alpha, k_max, ...) se
ignoran.

`run_fno_sweep` entrena un FNO por cada (n_modes, hidden_channels, n_layers),
todos con las mismas semillas; la seleccion es `best_run` (menor L2 relativa de
validacion), como en el PDNO. `save_fno_model` / `load_fno_model` guardan y
reconstruyen el FNO; la entrada "datos" es la misma que la de
`save_operator_model`.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from ..symbol import Symbol
from ..data.fields import FieldData
from ..operators.fno_baseline import FNOBaseline, FNOConfig
from .experiment import _dtype_name
from .losses import relative_l2_loss
from .operator_evaluation import field_errors
from .operator_training import OperatorTrainingConfig, OperatorTrainingResult, _data_spec
from ..utils import default_device

# Las metricas que `train_baseline` registra por epoca.
BASELINE_METRICS = ("total", "rel_l2")


def train_baseline(
    model: FNOBaseline,
    train_data: FieldData,
    val_data: FieldData,
    config: OperatorTrainingConfig,
) -> OperatorTrainingResult:
    """Entrena el FNO con AdamW y early stopping sobre la L2 relativa de validacion.

    Al terminar, el modelo queda con los pesos de la mejor epoca y en modo eval.
    """
    device, dtype = model.device, model.dtype
    train_data = train_data.to(device, dtype)
    val_data = val_data.to(device, dtype)

    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = None
    if config.lr_final is not None:
        gamma = (config.lr_final / config.lr) ** (1.0 / config.epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    splits = {"train": train_data, "val": val_data}

    if config.log_every:
        arch = model.config
        print(
            f"FNO: n_modes={arch.n_modes} hidden={arch.hidden_channels} capas={arch.n_layers} "
            f"embedding={arch.positional_embedding} | {model.n_parameters():,} parametros | "
            f"weight_decay={config.weight_decay:g}"
        )

    history: Dict = {name: {key: [] for key in BASELINE_METRICS} for name in splits}
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_run = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        model.train()
        perm = torch.randperm(train_data.n_pairs, generator=gen).to(device)

        for start in range(0, train_data.n_pairs, config.batch_size):
            u0, t, u = train_data.pairs(perm[start : start + config.batch_size])
            loss = relative_l2_loss(model(u0, t), u)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        for name, split in splits.items():
            rel_l2 = float(field_errors(model, split, config.eval_batch_size).mean())
            history[name]["rel_l2"].append(rel_l2)
            history[name]["total"].append(rel_l2)

        val_total = history["val"]["total"][-1]
        if val_total < best_val - config.min_delta:
            best_val, best_epoch = val_total, epoch
            best_state = copy.deepcopy(model.state_dict())

        if config.log_every and epoch % config.log_every == 0:
            print(
                f"epoca {epoch:4d} | train: rel_l2={history['train']['rel_l2'][-1]:.4f} "
                f"| val: rel_l2={val_total:.4f}"
            )

        if config.patience and epoch - best_epoch >= config.patience:
            stopped_early = True
            if config.log_every:
                print(f"early stopping en la epoca {epoch} (mejor: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return OperatorTrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_total=best_val,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
    )


# ----------------------------------------------------------------------
# Barrido
# ----------------------------------------------------------------------
@dataclass
class FNOSweepRun:
    n_modes: int
    hidden_channels: int
    n_layers: int
    n_parameters: int
    val_rel_l2: float            # en la mejor epoca
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    seconds: float
    model: FNOBaseline = field(repr=False)
    result: OperatorTrainingResult = field(repr=False)
    config: OperatorTrainingConfig = field(repr=False)


def run_fno_sweep(
    train_data: FieldData,
    val_data: FieldData,
    base_model: FNOConfig,
    base_config: OperatorTrainingConfig,
    n_modes: Sequence[int],
    hidden_channels: Sequence[int],
    n_layers: Sequence[int],
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> List[FNOSweepRun]:
    """Entrena un FNO por cada (n_modes, hidden_channels, n_layers), en ese orden.

    Cada corrida usa replace(base_model, ...) y `train_baseline` con
    `base_config`. Con `verbose` se imprime una linea por corrida y se
    silencian los logs por epoca; `FNOSweepRun.config` guarda la configuracion
    sin ese cambio.
    """
    device = device if device is not None else default_device()
    combos = list(product(n_modes, hidden_channels, n_layers))
    runs: List[FNOSweepRun] = []

    total = len(combos)
    for i, (modes, hidden, layers) in enumerate(combos, 1):
        arch = replace(base_model, n_modes=modes, hidden_channels=hidden, n_layers=layers)
        model = FNOBaseline(arch, device=device)

        start = time.perf_counter()
        result = train_baseline(
            model, train_data, val_data, replace(base_config, log_every=0) if verbose else base_config
        )
        seconds = time.perf_counter() - start

        run = FNOSweepRun(
            n_modes=modes,
            hidden_channels=hidden,
            n_layers=layers,
            n_parameters=model.n_parameters(),
            val_rel_l2=result.history["val"]["rel_l2"][result.best_epoch - 1],
            best_epoch=result.best_epoch,
            epochs_run=result.epochs_run,
            stopped_early=result.stopped_early,
            seconds=seconds,
            model=model,
            result=result,
            config=base_config,
        )
        runs.append(run)

        if verbose:
            print(
                f"[{i:3d}/{total}] n_modes={modes:<3d} hidden={hidden:<3d} capas={layers} "
                f"| rel_l2={run.val_rel_l2:.4f} | {seconds:5.1f}s"
            )

    return runs


def fno_sweep_table(runs: Sequence[FNOSweepRun]):
    """Resumen del barrido del FNO como DataFrame (una fila por corrida)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "n_modes": run.n_modes,
                "k_max_fno": run.n_modes // 2,
                "hidden_channels": run.hidden_channels,
                "capas": run.n_layers,
                "parametros": run.n_parameters,
                "val_rel_l2": run.val_rel_l2,
                "mejor_epoca": run.best_epoch,
                "epocas": run.epochs_run,
                "segundos": run.seconds,
            }
            for run in runs
        ]
    )


# ----------------------------------------------------------------------
# Modelo guardado
# ----------------------------------------------------------------------
FNO_CHECKPOINT_FORMAT = 1


def save_fno_model(
    path: Union[str, Path],
    model: FNOBaseline,
    *,
    reference: Symbol,
    resolution: int,
    k_cap: int,
    config: OperatorTrainingConfig,
    result: Optional[OperatorTrainingResult] = None,
    seed: int = 0,
    val_fraction: float = 0.2,
    extrapolation_times: Optional[Sequence[float]] = None,
    freq_scale: float = 1.0,
    amplitude: float = 1.0,
) -> Path:
    """Guarda el FNO entrenado con torch.save, listo para reconstruirlo sin el notebook.

    Contiene: el state_dict, la FNOConfig, el dtype, la version de
    neuraloperator, los datos (la misma entrada "datos" que `save_operator_model`),
    la OperatorTrainingConfig y, si se da `result`, la mejor epoca. Se lee con
    weights_only=True.
    """
    import neuralop

    path = Path(path)
    checkpoint = {
        "formato": FNO_CHECKPOINT_FORMAT,
        "modelo": "FNO",
        "neuraloperator": neuralop.__version__,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "arquitectura": asdict(model.config),
        "dtype": _dtype_name(model.dtype),
        "datos": _data_spec(
            reference, resolution, k_cap, seed, val_fraction, extrapolation_times, freq_scale, amplitude
        ),
        "entrenamiento": asdict(replace(config, class_reference=None)),
    }
    if result is not None:
        checkpoint["resultado"] = {
            "mejor_epoca": result.best_epoch, "epocas": result.epochs_run,
            "early_stopping": result.stopped_early, "val_rel_l2": result.best_val_total,
        }
    torch.save(checkpoint, path)
    print(f"modelo guardado en {path}")
    return path


def load_fno_checkpoint(path: Union[str, Path]) -> Dict:
    """El diccionario guardado por `save_fno_model`."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("formato") != FNO_CHECKPOINT_FORMAT or checkpoint.get("modelo") != "FNO":
        raise ValueError(f"{path}: no es un modelo de FNO de formato {FNO_CHECKPOINT_FORMAT}")
    return checkpoint


def load_fno_model(path: Union[str, Path], device: Optional[torch.device] = None) -> FNOBaseline:
    """Reconstruye el FNO guardado, en `device` y en modo eval."""
    checkpoint = load_fno_checkpoint(path)
    model = FNOBaseline(
        FNOConfig(**checkpoint["arquitectura"]), device=device, dtype=getattr(torch, checkpoint["dtype"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

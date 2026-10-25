"""Entrenamiento del PDNO con una sola perdida total y un solo optimizador.

    L = L_rel(u_theta(t), u(t)) + lambda_par sum_l L_par(a_par^{(l)})
        + lambda_class sum_l L_class(a_par^{(l)})

L_rel es el error relativo L2 en espacio fisico, por muestra y promediado,
sobre la salida de la red global. L_par y L_class son propiedades del simbolo:
se evaluan sobre el ParabolicMLPSymbol par de cada capa, con las funciones de
`losses.py`. El gradiente de L actualiza a la vez todos los parametros (las
MLP de los simbolos, W, b, P, Q y tau si se aprende). kappa =
model.theoretical_kappa(omega), igual que en `train`; las C_alpha se resuelven
con `resolve_class_constants`, igual que ahora.

Modelo guardado: `save_operator_model` escribe modelo.pt con los pesos, la
PDNOConfig, la OperatorTrainingConfig y lo necesario para regenerar los datos;
`load_operator_model` reconstruye el PDNO y `load_operator_datasets` regenera
los conjuntos con la misma semilla, sin el notebook.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from ..symbol import Symbol
from ..data.fields import FieldData, FieldDatasets, field_datasets
from ..data.grids import sample_k_points, sample_phase_points
from ..operators.config import PDNOConfig
from ..operators.pdno import PDNO
from .estimation import resolve_class_constants
from .experiment import _build_reference, _dtype_name, _reference_spec
from .losses import ClassConstants, class_loss, parabolicity_loss, relative_l2_loss
from .operator_evaluation import OPERATOR_METRICS, evaluate_fields
from ..utils import Split


@dataclass
class OperatorTrainingConfig:
    """Hiperparametros de `train_operator`. Mismos nombres y valores por defecto
    que TrainingConfig, salvo:

    - `batch_size` cuenta pares (condicion inicial, t), no filas (k, t);
    - `n_symbol_points` es cuantos puntos (k, lambda) y k se muestrean por lote
      para L_par y L_class (los mismos para todas las capas);
    - no hay lambda_trajectory: L_rel tiene peso 1.
    """

    epochs: int = 300
    batch_size: int = 20
    eval_batch_size: int = 64
    lr: float = 1e-3
    lr_final: Optional[float] = 1e-5
    weight_decay: float = 0.0    # AdamW, solo en PDNO.decay_parameters()
    omega: float = 1.0
    lambda_parabolicity: float = 1.0
    lambda_class: float = 1.0
    class_C: ClassConstants = 1.0
    class_C_source: str = "given"  # "given", "oracle" o "data" (ver TrainingConfig)
    class_reference: Optional[Symbol] = None
    class_margin: float = 1.5
    early_stopping_class: bool = True
    k_max: int = 8
    lam_min: float = 1e-2
    lam_max: float = 1e2
    n_symbol_points: int = 1024
    grad_clip: Optional[float] = 1.0
    patience: int = 30
    min_delta: float = 0.0
    n_phase_eval: int = 8192
    log_every: int = 20
    seed: int = 0


@dataclass
class OperatorTrainingResult:
    history: Dict[str, Dict[str, List[float]]]  # "train" y "val" por metrica; "tau" por epoca
    best_epoch: int
    best_val_total: float
    epochs_run: int
    stopped_early: bool
    class_C: Optional[ClassConstants] = None


def train_operator(
    model: PDNO,
    train_data: FieldData,
    val_data: FieldData,
    config: OperatorTrainingConfig,
    rows: Optional[Split] = None,
) -> OperatorTrainingResult:
    """Entrena el PDNO con AdamW y early stopping sobre la perdida total de validacion.

    `rows` son las filas espectrales (k, t, u0, u_ref) de las condiciones
    iniciales de entrenamiento (`FieldDatasets.rows_train`); solo hacen falta
    con class_C_source = "data". Al terminar, el modelo queda con los pesos de
    la mejor epoca y en modo eval. El criterio de early stopping es el de
    `train` (ver `early_stopping_class` y `evaluate_fields`).
    """
    if config.class_C_source == "data" and rows is None:
        raise ValueError("class_C_source='data' necesita las filas de entrenamiento `rows`")
    rho = model.symbols[0].rho
    class_C = resolve_class_constants(
        config.class_C_source, config.class_C, rows, model.config.m, rho,
        config.k_max, config.class_reference, config.class_margin,
    )
    config = replace(config, class_C=class_C, class_C_source="given")

    device, dtype, n = model.device, model.dtype, model.n
    kappa = model.theoretical_kappa(config.omega)
    train_data = train_data.to(device, dtype)
    val_data = val_data.to(device, dtype)

    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed)
    params = list(model.parameters())
    decayed = model.decay_parameters()
    decayed_ids = {id(p) for p in decayed}
    optimizer = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": config.weight_decay},
            {"params": [p for p in params if id(p) not in decayed_ids], "weight_decay": 0.0},
        ],
        lr=config.lr,
    )
    scheduler = None
    if config.lr_final is not None:
        gamma = (config.lr_final / config.lr) ** (1.0 / config.epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # Puntos fijos por conjunto para las metricas por epoca, como en `train`.
    phase_eval = {
        name: sample_phase_points(
            n, config.k_max, config.n_phase_eval, config.lam_min, config.lam_max,
            torch.Generator().manual_seed(config.seed + offset), device, dtype,
        )
        for name, offset in (("train", 101), ("val", 202))
    }
    class_eval = {
        name: sample_k_points(
            n, config.k_max, config.n_phase_eval,
            torch.Generator().manual_seed(config.seed + offset), device, dtype,
        ) if config.lambda_class else None
        for name, offset in (("train", 303), ("val", 404))
    }
    splits = {"train": train_data, "val": val_data}

    if config.log_every:
        arch = model.config
        print(
            f"PDNO: L={arch.n_layers} d={arch.width} W={'si' if arch.use_W else 'no'} "
            f"sesgos={'si' if arch.bias else 'no'} sigma={arch.activation} P={arch.lift} "
            f"Q={arch.project} tau={'aprendido' if arch.learn_tau else 'fijo 1/L'} | "
            f"simbolo hidden={arch.symbol_hidden} | weight_decay={config.weight_decay:g}"
        )

    history: Dict = {name: {key: [] for key in OPERATOR_METRICS} for name in splits}
    history["tau"] = []
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

            k_p, lam_p = sample_phase_points(
                n, config.k_max, config.n_symbol_points, config.lam_min, config.lam_max,
                gen, device, dtype,
            )
            loss = loss + config.lambda_parabolicity * sum(
                parabolicity_loss(symbol, k_p, lam_p, config.omega, kappa) for symbol in model.symbols
            )
            if config.lambda_class:
                k_c = sample_k_points(n, config.k_max, config.n_symbol_points, gen, device, dtype)
                loss = loss + config.lambda_class * sum(
                    class_loss(symbol, k_c, config.class_C) for symbol in model.symbols
                )

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        for name, split in splits.items():
            metrics = evaluate_fields(model, split, phase_eval[name], config, class_eval[name])
            for key in OPERATOR_METRICS:
                history[name][key].append(metrics[key])
        history["tau"].append(model.tau.detach().cpu().tolist())

        val_total = history["val"]["total"][-1]
        if val_total < best_val - config.min_delta:
            best_val, best_epoch = val_total, epoch
            best_state = copy.deepcopy(model.state_dict())

        if config.log_every and epoch % config.log_every == 0:
            val = {key: history["val"][key][-1] for key in OPERATOR_METRICS}
            tau = ", ".join(f"{x:.3f}" for x in history["tau"][-1])
            print(
                f"epoca {epoch:4d} | val: total={val['total']:.3e} rel_l2={val['rel_l2']:.4f} "
                f"par={val['parabolicity']:.3e} class={val['class']:.3e} "
                f"max_ratio={val['par_max_ratio']:.3f} viol={100 * val['par_violation']:5.2f}% "
                f"| tau=({tau})"
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
        class_C=class_C,
    )


# ----------------------------------------------------------------------
# Modelo guardado (modelo.pt)
# ----------------------------------------------------------------------
OPERATOR_CHECKPOINT_FORMAT = 1


def save_operator_model(
    path: Union[str, Path],
    model: PDNO,
    *,
    reference: Symbol,
    resolution: int,
    k_cap: int,
    config: OperatorTrainingConfig,
    result: Optional[OperatorTrainingResult] = None,
    seed: int = 0,
    val_fraction: float = 0.2,
    extrapolation_times: Optional[Sequence[float]] = None,
    delta: float = 0.1,
    estimate_tol: float = 1e-10,
) -> Path:
    """Guarda el PDNO entrenado con torch.save, listo para reconstruirlo sin el notebook.

    Contiene: el state_dict (pesos de las MLP de los simbolos, W, b, P, Q y
    tau), la PDNOConfig (con la banda (c, C) y m), el dtype, los datos (simbolo
    de referencia, semilla, resolucion, k_cap, val_fraction, tiempos de
    extrapolacion), las constantes de la estimacion (delta, tol), la
    OperatorTrainingConfig y, si se da `result`, la mejor epoca y las C_alpha
    que se usaron. Todo son tipos simples: se lee con weights_only=True.
    """
    path = Path(path)
    checkpoint = {
        "formato": OPERATOR_CHECKPOINT_FORMAT,
        "modelo": "PDNO",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "arquitectura": asdict(model.config),
        "dtype": _dtype_name(model.dtype),
        "datos": {
            "referencia": _reference_spec(reference), "semilla": seed, "resolucion": resolution,
            "k_cap": k_cap, "val_fraction": val_fraction,
            "extrapolation_times": (
                None if extrapolation_times is None else [float(t) for t in extrapolation_times]
            ),
        },
        "experimento": {"delta": delta, "estimate_tol": estimate_tol},
        "entrenamiento": asdict(replace(config, class_reference=None)),
    }
    if result is not None:
        checkpoint["resultado"] = {
            "mejor_epoca": result.best_epoch, "epocas": result.epochs_run,
            "early_stopping": result.stopped_early, "val_total": result.best_val_total,
            "class_C": result.class_C,
        }
    torch.save(checkpoint, path)
    print(f"modelo guardado en {path}")
    return path


def load_operator_checkpoint(path: Union[str, Path]) -> Dict:
    """El diccionario de modelo.pt (ver `save_operator_model`)."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("formato") != OPERATOR_CHECKPOINT_FORMAT or checkpoint.get("modelo") != "PDNO":
        raise ValueError(f"{path}: no es un modelo.pt de PDNO de formato {OPERATOR_CHECKPOINT_FORMAT}")
    return checkpoint


def load_operator_model(path: Union[str, Path], device: Optional[torch.device] = None) -> PDNO:
    """Reconstruye el PDNO guardado en modelo.pt, en `device` y en modo eval."""
    checkpoint = load_operator_checkpoint(path)
    model = PDNO(
        PDNOConfig(**checkpoint["arquitectura"]), device=device, dtype=getattr(torch, checkpoint["dtype"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_operator_datasets(path: Union[str, Path]) -> FieldDatasets:
    """Regenera los conjuntos de modelo.pt con `field_datasets` (misma semilla, mismos datos)."""
    datos = load_operator_checkpoint(path)["datos"]
    return field_datasets(
        _build_reference(datos["referencia"]), datos["resolucion"], datos["semilla"],
        k_cap=datos["k_cap"], val_fraction=datos["val_fraction"],
        extrapolation_times=datos["extrapolation_times"],
    )

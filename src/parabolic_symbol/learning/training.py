"""Entrenamiento PDNO-PINN del simbolo aprendido a_theta.

Perdida total:
    L = lambda_trajectory * L_traj + lambda_parabolicity * L_par
        + lambda_class * L_class

con kappa = symbol.theoretical_kappa(omega): la Def. 2.3 se cumple por
construccion con ese kappa, asi que no es un hiperparametro. Los terminos estan
en `losses.py`, los datos en `data/` y las metricas en `evaluation.py`. El
barrido de arquitecturas esta en `sweep.py`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import torch

from ..symbol import Symbol
from ..data.synthetic import sample_k_points, sample_phase_points, to_device
from .estimation import resolve_class_constants
from .evaluation import METRICS, evaluate_split
from .losses import ClassConstants, class_loss, parabolicity_loss, trajectory_loss
from ..utils import LearnedSymbol, Split


@dataclass
class TrainingConfig:
    """Hiperparametros de `train`. Los valores por defecto son la configuracion
    base de los notebooks; solo `k_max` y `class_C` dependen de los datos.

    C_alpha de L_class (`class_C_source`):
      - "given"  (por defecto): se usa `class_C` tal cual, como antes;
      - "oracle": se calculan con el simbolo exacto `class_reference` en
                  |k_i| <= k_max, con margen `class_margin`;
      - "data":   se calculan por diferencias finitas de a_hat (estimado de
                  `train_data`, nunca de validacion), con margen `class_margin`.

    `early_stopping_class` decide si L_class entra en la perdida de validacion
    que vigila el early stopping (ver `train`).
    """

    epochs: int = 300
    batch_size: int = 1024
    lr: float = 1e-3
    lr_final: Optional[float] = 1e-5  # si se da, decae exponencialmente hasta este valor
    weight_decay: float = 0.0    # AdamW desacoplado, solo en las Linear ocultas interiores
    omega: float = 1.0
    lambda_trajectory: float = 1.0
    lambda_parabolicity: float = 1.0
    lambda_class: float = 1.0    # peso de L_class (Def. 2.1); 0 lo apaga
    class_C: ClassConstants = 1.0  # C_alpha: un escalar, o {alpha: C_alpha}
    class_C_source: str = "given"  # "given" (class_C), "oracle" o "data"
    class_reference: Optional[Symbol] = None  # simbolo exacto, solo para "oracle"
    class_margin: float = 1.5    # margen sobre las C_alpha de "oracle" / "data"
    early_stopping_class: bool = True  # incluir L_class en la perdida del early stopping
    k_max: int = 8               # ventana de k para los puntos de parabolicidad
    lam_min: float = 1e-2        # rango de |lambda| muestreado
    lam_max: float = 1e2
    grad_clip: Optional[float] = 1.0  # None para desactivar
    patience: int = 30           # early stopping sobre la perdida total de validacion
    min_delta: float = 0.0       # mejora minima para reiniciar la paciencia
    n_phase_eval: int = 8192     # puntos (k, lambda) fijos para las metricas por epoca
    log_every: int = 20
    seed: int = 0


@dataclass
class TrainingResult:
    history: Dict[str, Dict[str, List[float]]]
    best_epoch: int
    best_val_total: float
    epochs_run: int
    stopped_early: bool
    class_C: Optional[ClassConstants] = None  # C_alpha que uso L_class (ya resueltas)


def train(
    symbol: LearnedSymbol, train_data: Split, val_data: Split, config: TrainingConfig
) -> TrainingResult:
    """Entrena a_theta con Adam y early stopping sobre la perdida de validacion.

    Al terminar, los pesos del simbolo quedan en los de la mejor epoca. El kappa
    de L_par es symbol.theoretical_kappa(config.omega).

    Early stopping. Con `lambda_class > 0` la perdida optimizada incluye
    L_class siempre. Lo que cambia `early_stopping_class` es el criterio:
      - False: el "total" de
        validacion que se vigila es solo trayectoria + parabolicidad. L_class
        no tiene datos de validacion propios (sus k se muestrean de la misma
        ventana en ambos conjuntos), asi que no mide generalizacion; incluirla
        puede detener el entrenamiento por una mejora de regularidad que no se
        traduce en mejor trayectoria.
      - True (por defecto): el total incluye lambda_class * L_class, evaluada en un conjunto
        fijo de k. Util cuando importa que la mejor epoca tambien respete las
        C_alpha (p. ej. con C_alpha ajustadas sin mucho margen).
    En ambos casos L_class de validacion se registra en history["val"]["class"].
    """
    class_C = resolve_class_constants(
        config.class_C_source, config.class_C, train_data, symbol.m, symbol.rho,
        config.k_max, config.class_reference, config.class_margin,
    )
    config = replace(config, class_C=class_C, class_C_source="given")

    device, dtype = symbol.device, symbol.dtype
    kappa = symbol.theoretical_kappa(config.omega)
    train_data = to_device(train_data, device, dtype)
    val_data = to_device(val_data, device, dtype)

    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed)
    params = list(symbol.parameters())
    decayed = symbol.decay_parameters()
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

    # Puntos de parabolicidad fijos para las metricas: uno por conjunto, asi las
    # curvas por epoca son comparables y no dependen del muestreo del batch.
    phase_eval = {
        "train": sample_phase_points(
            symbol.n, config.k_max, config.n_phase_eval, config.lam_min, config.lam_max,
            torch.Generator().manual_seed(config.seed + 101), device, dtype,
        ),
        "val": sample_phase_points(
            symbol.n, config.k_max, config.n_phase_eval, config.lam_min, config.lam_max,
            torch.Generator().manual_seed(config.seed + 202), device, dtype,
        ),
    }
    # Igual para L_class: k fijos por conjunto (solo si el termino esta activo).
    class_eval = {
        name: sample_k_points(
            symbol.n, config.k_max, config.n_phase_eval,
            torch.Generator().manual_seed(config.seed + offset), device, dtype,
        ) if config.lambda_class else None
        for name, offset in (("train", 303), ("val", 404))
    }
    splits = {"train": train_data, "val": val_data}

    if config.log_every:
        capas = ",".join(str(i) for i in symbol.dropout_layers) or "-"
        estado = f"p={symbol.dropout:g} en capas[{capas}]" if symbol.dropout_layers else "OFF"
        print(
            f"hidden={symbol.hidden} | dropout {estado} | "
            f"weight_decay={config.weight_decay:g} en {len(decayed)} Linear interiores"
        )

    n_data = train_data["k"].shape[0]
    history = {name: {key: [] for key in METRICS} for name in splits}
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(symbol.net.state_dict())
    epochs_run = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        symbol.net.train()
        perm = torch.randperm(n_data, generator=gen).to(device)

        for start in range(0, n_data, config.batch_size):
            idx = perm[start : start + config.batch_size]

            l_traj = trajectory_loss(
                symbol, train_data["k"][idx], train_data["t"][idx],
                train_data["u0"][idx], train_data["u_ref"][idx],
            )
            k_p, lam_p = sample_phase_points(
                symbol.n, config.k_max, len(idx), config.lam_min, config.lam_max,
                gen, device, dtype,
            )
            l_par = parabolicity_loss(symbol, k_p, lam_p, config.omega, kappa)

            loss = config.lambda_trajectory * l_traj + config.lambda_parabolicity * l_par

            if config.lambda_class:
                # Puntos k propios de este termino, independientes de las muestras
                # de trayectoria y de los puntos de parabolicidad.
                k_c = sample_k_points(symbol.n, config.k_max, len(idx), gen, device, dtype)
                loss = loss + config.lambda_class * class_loss(symbol, k_c, config.class_C)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        for name, split in splits.items():
            metrics = evaluate_split(symbol, split, phase_eval[name], config, class_eval[name])
            for key in METRICS:
                history[name][key].append(metrics[key])

        val_total = history["val"]["total"][-1]
        if val_total < best_val - config.min_delta:
            best_val, best_epoch = val_total, epoch
            best_state = copy.deepcopy(symbol.net.state_dict())

        if config.log_every and epoch % config.log_every == 0:
            val = {key: history["val"][key][-1] for key in METRICS}
            print(
                f"epoca {epoch:4d} | val: total={val['total']:.3e} "
                f"traj={val['trajectory']:.3e} par={val['parabolicity']:.3e} "
                f"class={val['class']:.3e} "
                f"rel_l2={val['rel_l2']:.4f} max_ratio={val['par_max_ratio']:.3f} "
                f"viol={100 * val['par_violation']:5.2f}%"
            )

        if config.patience and epoch - best_epoch >= config.patience:
            stopped_early = True
            if config.log_every:
                print(f"early stopping en la epoca {epoch} (mejor: {best_epoch})")
            break

    symbol.net.load_state_dict(best_state)
    symbol.net.eval()
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_total=best_val,
        epochs_run=epochs_run,
        stopped_early=stopped_early,
        class_C=class_C,
    )

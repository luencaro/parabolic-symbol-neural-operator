"""Entrenamiento PDNO-PINN del simbolo aprendido a_theta.

Perdida total:
    L = lambda_trajectory * L_traj + lambda_parabolicity * L_par

    L_traj = E[ |exp(-t a_theta(k)) u0(k) - u_ref(k, t)|^2 ]
    L_par  = E[ max(0, |(a_theta(k) + lambda)^{-1}| - kappa (1 + |(k,lambda)|^2)^{-m/2})^2 ]
             sobre Re(lambda) >= 0, |(k,lambda)| >= omega.

kappa no se fija a mano: se barre una lista de valores y se elige el mas
pequeno que el entrenamiento consigue satisfacer (ver `run_sweep`/`best_run`).
"""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .Symbol import Symbol
from .symbols.mlp_symbol import MLPSymbol, default_device

Split = Dict[str, torch.Tensor]
PhasePoints = Tuple[torch.Tensor, torch.Tensor]

METRICS = ("total", "trajectory", "parabolicity", "rel_l2", "par_max_ratio", "par_violation")


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
def k_grid(n: int, k_max: int) -> np.ndarray:
    """Todos los k in Z^n con |k_i| <= k_max, forma (N, n)."""
    return np.array([np.array(idx) - k_max for idx in np.ndindex(*((2 * k_max + 1,) * n))])


def make_trajectory_data(
    reference: Symbol,
    k_max: int,
    n_samples: int,
    t_min: float,
    t_max: float,
    seed: int = 0,
) -> Split:
    """Genera muestras (k, t, u0, u_ref) con u_ref = exp(-t a_ref(k)) u0(k).

    k se toma al azar de la rejilla entera, t log-uniforme en [t_min, t_max]
    (asi hay senal tanto para |k| pequeno como grande) y u0 ~ normal compleja.
    """
    rng = np.random.default_rng(seed)
    grid = k_grid(reference.n, k_max)
    a_grid = np.array([reference(g)[0, 0] for g in grid])  # se evalua solo la rejilla

    idx = rng.integers(0, len(grid), size=n_samples)
    k = grid[idx]
    t = np.exp(rng.uniform(np.log(t_min), np.log(t_max), size=n_samples))
    u0 = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    u_ref = np.exp(-t * a_grid[idx]) * u0

    return {
        "k": torch.tensor(k, dtype=torch.float64),
        "t": torch.tensor(t, dtype=torch.float64),
        "u0": torch.tensor(u0, dtype=torch.complex128),
        "u_ref": torch.tensor(u_ref, dtype=torch.complex128),
    }


def split_trajectory_data(
    data: Split, val_fraction: float = 0.2, seed: int = 0
) -> Tuple[Split, Split]:
    """Particion aleatoria en entrenamiento y validacion (sin conjunto de prueba)."""
    n = data["k"].shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(round(val_fraction * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (
        {key: value[train_idx] for key, value in data.items()},
        {key: value[val_idx] for key, value in data.items()},
    )


def to_device(data: Split, device: torch.device, dtype: torch.dtype) -> Split:
    """Mueve un split al device/precision del modelo."""
    cdtype = complex_dtype(dtype)
    return {
        key: value.to(device=device, dtype=cdtype if value.is_complex() else dtype)
        for key, value in data.items()
    }


def sample_phase_points(
    n: int,
    k_max: int,
    batch_size: int,
    lam_min: float,
    lam_max: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> PhasePoints:
    """Muestra (k, lambda) con Re(lambda) >= 0.

    |lambda| es log-uniforme en [lam_min, lam_max] y arg(lambda) in [-pi/2, pi/2].
    La mitad de las muestras cae en la frontera Re(lambda) = 0.

    Se genera en CPU (para que el generador de numeros aleatorios de resultados
    reproducibles en CPU y GPU) y luego se mueve al device.
    """
    k = torch.randint(-k_max, k_max + 1, (batch_size, n), generator=generator).to(torch.float64)
    u = torch.rand(batch_size, generator=generator, dtype=torch.float64)
    radius = lam_min * (lam_max / lam_min) ** u
    phase = np.pi * (torch.rand(batch_size, generator=generator, dtype=torch.float64) - 0.5)
    half = batch_size // 2
    phase[:half] = torch.sign(phase[:half]) * np.pi / 2  # frontera Re(lambda) = 0
    lam = torch.polar(radius, phase)
    return (
        k.to(device=device, dtype=dtype),
        lam.to(device=device, dtype=complex_dtype(dtype)),
    )


def phase_grid(
    n: int,
    k_max: int,
    lam_values: Sequence[complex],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> PhasePoints:
    """Producto cartesiano de la rejilla de k con una lista de lambda.

    Sirve como conjunto fijo y determinista para decidir si un kappa se satisface.
    """
    k = k_grid(n, k_max)
    lam = np.asarray(lam_values, dtype=complex)
    return (
        torch.tensor(np.repeat(k, len(lam), axis=0), dtype=dtype, device=device),
        torch.tensor(np.tile(lam, len(k)), dtype=complex_dtype(dtype), device=device),
    )


# ----------------------------------------------------------------------
# Perdidas y metricas
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------
@dataclass
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 1e-3
    lr_final: Optional[float] = None  # si se da, decae exponencialmente hasta este valor
    omega: float = 1.0
    kappa: float = 3.0
    lambda_trajectory: float = 1.0
    lambda_parabolicity: float = 1.0
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


def train(
    symbol: MLPSymbol, train_data: Split, val_data: Split, config: TrainingConfig
) -> TrainingResult:
    """Entrena a_theta con Adam y early stopping sobre la perdida de validacion.

    Al terminar, los pesos del simbolo quedan en los de la mejor epoca.
    """
    device, dtype = symbol.device, symbol.dtype
    train_data = to_device(train_data, device, dtype)
    val_data = to_device(val_data, device, dtype)

    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed)
    params = list(symbol.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)
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
    splits = {"train": train_data, "val": val_data}

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
            l_par = parabolicity_loss(symbol, k_p, lam_p, config.omega, config.kappa)

            loss = config.lambda_trajectory * l_traj + config.lambda_parabolicity * l_par

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        for name, split in splits.items():
            metrics = evaluate_split(symbol, split, phase_eval[name], config)
            for key in METRICS:
                history[name][key].append(metrics[key])

        val_total = history["val"]["total"][-1]
        if val_total < best_val - config.min_delta:
            best_val, best_epoch = val_total, epoch
            best_state = copy.deepcopy(symbol.net.state_dict())

        if config.log_every and epoch % config.log_every == 0:
            print(
                f"epoca {epoch:4d} | val total {val_total:.3e} | "
                f"val traj {history['val']['trajectory'][-1]:.3e} | "
                f"val par {history['val']['parabolicity'][-1]:.3e} | "
                f"val rel_l2 {history['val']['rel_l2'][-1]:.3e}"
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
    )


# ----------------------------------------------------------------------
# Barrido de hiperparametros (kappa, profundidad, ancho)
# ----------------------------------------------------------------------
@dataclass
class SweepRun:
    kappa: float
    n_layers: int
    width: int
    satisfied: bool          # la cota se cumple en TODA la rejilla de comprobacion
    max_ratio: float         # sup del ratio sobre esa rejilla (<= 1 <=> satisfied)
    val_total: float
    val_rel_l2: float
    best_epoch: int
    epochs_run: int
    seconds: float
    symbol: MLPSymbol = field(repr=False)
    result: TrainingResult = field(repr=False)


@torch.no_grad()
def max_ratio_on(symbol: MLPSymbol, phase: PhasePoints, omega: float, kappa: float) -> float:
    """Peor ratio del simbolo sobre un conjunto fijo de puntos (k, lambda)."""
    _, ratio = parabolicity_terms(symbol, *phase, omega, kappa)
    return float(ratio.max()) if ratio.numel() else 0.0


def run_sweep(
    train_data: Split,
    val_data: Split,
    n: int,
    m: float,
    kappas: Sequence[float],
    depths: Sequence[int],
    widths: Sequence[int],
    base_config: TrainingConfig,
    check_phase: PhasePoints,
    dropout: float = 0.2,
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> List[SweepRun]:
    """Entrena una red por cada (kappa, profundidad, ancho) y reporta si cumple.

    `check_phase` es la rejilla fija sobre la que se decide si un kappa se
    satisface (max_ratio <= 1), independiente de los puntos usados en la perdida.
    """
    device = device if device is not None else default_device()
    runs: List[SweepRun] = []

    for kappa, n_layers, width in itertools.product(kappas, depths, widths):
        config = replace(base_config, kappa=kappa)
        symbol = MLPSymbol(
            n=n, m=m, hidden=(width,) * n_layers, dropout=dropout,
            input_scale=input_scale, output_scale=output_scale,
            seed=base_config.seed, device=device, dtype=dtype,
        )

        start = time.perf_counter()
        result = train(symbol, train_data, val_data, config)
        seconds = time.perf_counter() - start

        max_ratio = max_ratio_on(symbol, check_phase, config.omega, kappa)
        run = SweepRun(
            kappa=kappa,
            n_layers=n_layers,
            width=width,
            satisfied=max_ratio <= 1.0,
            max_ratio=max_ratio,
            val_total=result.history["val"]["total"][result.best_epoch - 1],
            val_rel_l2=result.history["val"]["rel_l2"][result.best_epoch - 1],
            best_epoch=result.best_epoch,
            epochs_run=result.epochs_run,
            seconds=seconds,
            symbol=symbol,
            result=result,
        )
        runs.append(run)

        if verbose:
            flag = "cumple" if run.satisfied else "NO cumple"
            print(
                f"kappa={kappa:6.3f} capas={n_layers} ancho={width:3d} | "
                f"{flag} (max ratio {max_ratio:.3f}) | "
                f"val rel_l2 {run.val_rel_l2:.4f} | "
                f"epocas {run.epochs_run:3d} (mejor {run.best_epoch:3d}) | {seconds:5.1f}s"
            )

    return runs


def best_run(runs: Sequence[SweepRun]) -> Optional[SweepRun]:
    """El kappa mas pequeno que se logro satisfacer; a igual kappa, mejor val_rel_l2."""
    feasible = [run for run in runs if run.satisfied]
    if not feasible:
        return None
    kappa_min = min(run.kappa for run in feasible)
    return min(
        (run for run in feasible if run.kappa == kappa_min), key=lambda run: run.val_rel_l2
    )


def sweep_table(runs: Sequence[SweepRun]):
    """Resumen del barrido como DataFrame (una fila por corrida)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "kappa": run.kappa,
                "capas": run.n_layers,
                "ancho": run.width,
                "cumple": run.satisfied,
                "max_ratio": run.max_ratio,
                "val_rel_l2": run.val_rel_l2,
                "val_total": run.val_total,
                "mejor_epoca": run.best_epoch,
                "epocas": run.epochs_run,
                "segundos": run.seconds,
            }
            for run in runs
        ]
    )

"""FNO de `neuraloperator` con la interfaz del PDNO: forward(u0, t), `device` y `dtype`.

    u_theta(t) = FNO([u0(x), t, x])[0]

en T^n con n = `FNOConfig.n` (1 por defecto, o 2). Entrada del FNO: dos canales
sobre la malla, u0(x) y t constante; el embedding de posicion "grid" de la
libreria agrega x in [0, 1)^n, con linspace(0, 1, R + 1)[:-1] por eje (periodico
y recalculado para cada resolucion).
Salida: u(t), un canal. Sin domain_padding: el problema es periodico y la FFT
tambien. El resto de la arquitectura son los valores por defecto de `FNO`.

En 1D con datos reales la libreria conserva n_modes // 2 + 1 modos de rfft,
es decir |k| <= n_modes / 2: n_modes = 128 ve la misma ventana |k| <= 64 que
las filas espectrales del PDNO. En 2D `n_modes` es el de cada eje, con la misma
regla: n_modes = 64 conserva |k_i| <= 32. Si la malla tiene menos modos, la
libreria usa los que hay.

Precision: SpectralConv (neuraloperator 2.0.0) crea los pesos espectrales y el
buffer de salida en torch.cfloat, asi que en float64 cada capa espectral
volveria a complex64 sin avisar. Ademas Module.to(dtype=...) convertiria los
pesos complejos a reales. Por eso el modelo vive en float32 (el dtype con que
lo crea la libreria) y solo se mueve de device.

Este modulo no se importa desde `operators/__init__.py`: el paquete funciona
sin `neuraloperator` instalado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from neuralop.models import FNO
from torch import nn

from ..utils import default_device


@dataclass
class FNOConfig:
    """Arquitectura del FNO. `positional_embedding` es el de la libreria ("grid" o None).

    `n_modes` es por eje; `n` es la dimension del toro (va al final y vale 1 por
    defecto, asi que los modelos guardados antes de 2D se siguen cargando igual).
    """

    n_modes: int
    hidden_channels: int
    n_layers: int
    positional_embedding: Optional[str] = "grid"
    seed: int = 0
    n: int = 1


class FNOBaseline(nn.Module):
    """El FNO como operador u0 -> u(t); ver el docstring del modulo.

    Misma interfaz que el PDNO, asi que `pair_errors`, `field_errors` y
    `relative_l2_loss` lo evaluan igual. La semilla se fija una vez, antes de
    construir la red.
    """

    def __init__(
        self, config: FNOConfig, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        if dtype != torch.float32:
            raise ValueError(f"el FNO de neuraloperator solo funciona en float32 (ver el modulo): {dtype}")
        self.config = config
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype

        torch.manual_seed(config.seed)
        self.fno = FNO(
            n_modes=(config.n_modes,) * config.n, in_channels=2, out_channels=1,
            hidden_channels=config.hidden_channels, n_layers=config.n_layers,
            positional_embedding=config.positional_embedding,
        )
        self.fno.to(device=self.device)

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u0: (B, *malla) real, t: (B,) -> u_theta(t) de forma (B, *malla)."""
        t = t.reshape((-1,) + (1,) * (u0.dim() - 1)).expand_as(u0)
        return self.fno(torch.stack([u0, t], dim=1))[:, 0]

    def state_dict(self, *args, **kwargs):
        """El de nn.Module sin la entrada "_metadata" que agrega el BaseModel de neuraloperator.

        Son los argumentos del constructor del FNO (no tensores); el modelo se
        reconstruye desde FNOConfig, y load_state_dict no la acepta.
        """
        state = super().state_dict(*args, **kwargs)
        state.pop("_metadata", None)
        return state

    def n_parameters(self) -> int:
        """Numero de parametros reales: un peso complejo cuenta como dos."""
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())

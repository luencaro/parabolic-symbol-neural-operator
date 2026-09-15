"""Simbolo aprendido a_theta: Z^n -> C mediante una MLP sin restricciones.

La red recibe k (vector de Z^n) y devuelve dos numeros reales
(Re a_theta(k), Im a_theta(k)).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..Symbol import Symbol


def default_device() -> torch.device:
    """GPU si hay, si no CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPSymbol(Symbol):
    """a_theta(k) = MLP(k), caso escalar (E = C).

    Parameters
    n : int
        Dimension del toro (k in Z^n).
    m : float
        Orden que se declara para el simbolo (se usa en la cota de parabolicidad).
    hidden : secuencia de int
        Ancho de cada capa oculta.
    rho : int, opcional
        Parametro de regularidad (por defecto ceil(m) + 1).
    input_scale, output_scale : float
        Constantes fijas: la red ve k / input_scale y su salida se multiplica
        por output_scale. Solo cambian unidades, no restringen la forma.
    dropout : float
        Probabilidad de dropout despues de cada activacion oculta (0 = sin dropout).
    device, dtype : torch
        Donde y con que precision vive la red. Por defecto GPU si esta disponible.
    """

    def __init__(
        self,
        n: int,
        m: float,
        hidden: Sequence[int] = (64, 64, 64),
        rho: Optional[int] = None,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        dropout: float = 0.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        hidden = tuple(hidden)
        super().__init__(
            n=n,
            m=m,
            rho=rho if rho is not None else int(np.ceil(m)) + 1,
            dim=1,
            name=f"MLPSymbol(hidden={hidden}, dropout={dropout})",
        )
        if seed is not None:
            torch.manual_seed(seed)

        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.dropout = float(dropout)
        self.hidden = hidden
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype

        # Capas ocultas con Tanh (+ dropout), salida lineal de 2 componentes (Re, Im)
        layers = []
        width_in = n
        for width in hidden:
            layers += [nn.Linear(width_in, width), nn.Tanh()]
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            width_in = width
        layers.append(nn.Linear(width_in, 2))
        self.net = nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)

    def parameters(self):
        """Parametros entrenables (para el optimizador)."""
        return self.net.parameters()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Version en torch, por lotes.

        k: tensor (batch, n). Devuelve a_theta(k) complejo de forma (batch,).
        """
        k = k.to(device=self.device, dtype=self.dtype)
        out = self.net(k / self.input_scale) * self.output_scale
        return torch.complex(out[:, 0], out[:, 1])

    def evaluate_batch(self, k: np.ndarray) -> np.ndarray:
        """a_theta sobre un lote: k de forma (N, n) -> vector complejo (N,).

        Siempre en modo eval, asi el dropout no altera la evaluacion.
        """
        k = np.asarray(k, dtype=float).reshape(-1, self.n)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            values = self.forward(torch.from_numpy(k)).cpu().numpy()
        self.net.train(was_training)
        return values.astype(complex)

    def evaluate(self, k: np.ndarray) -> np.ndarray:
        """Interfaz de Symbol: k in Z^n -> matriz 1x1 compleja."""
        return np.array([[self.evaluate_batch(k)[0]]], dtype=complex)

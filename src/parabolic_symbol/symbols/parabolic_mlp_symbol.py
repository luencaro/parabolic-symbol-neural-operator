"""Simbolo aprendido parabolico por construccion: a_theta = <k>^m * (banda en (c, C)).

Una MLP con salida lineal da un a_theta(k) acotado, y la parabolicidad
(Def. 2.3), que exige |a_theta(k)| >= <k>^m / kappa -> infinito, solo podria
cumplirse dentro de la ventana de entrenamiento. Aqui la escala la da <k>^m y la
red solo aprende un factor acotado:

    x        = k / <k>,                 <k> = (1 + |k|^2)^{1/2}
    a_theta(k) = <k>^m * ( c + (C - c) * sigma(g_theta(x)) )     para k != 0
    a_theta(0) = 0                      (modo cero fijado a mano)

con g_theta: R^n -> R una MLP con Tanh (C^infinito) y sigma la sigmoide logistica.
Asi c <k>^m < a_theta(k) < C <k>^m para todo k != 0, sea cual sea theta, y la
Def. 2.3 se cumple en todo Z^n x {Re(lambda) >= 0} con el kappa explicito de
`theoretical_kappa`.

Pertenencia a S^{m,rho} (Def. 2.1): x(k) = k/<k> es un simbolo de orden 0 con
imagen en la bola unidad abierta, sigma o g_theta es C^infinito y acotada con
todas sus derivadas sobre esa bola, y <k>^m esta en S^m; el producto esta en
S^m. Fijar a_theta(0) = 0 cambia un solo punto, y eso solo altera finitas
diferencias Delta^alpha. La pertenencia esta garantizada cualitativamente, pero
las constantes C_alpha dependen de theta: las controla L_class.

Version par (`even=True`, la que usa el PDNO):

    a_par(k) = (a_theta(k) + a_theta(-k)) / 2
             = <k>^m ( c + (C - c) (sigma(g_theta(x)) + sigma(g_theta(-x))) / 2 ).

El promedio sigue en la banda (c, C), asi que vale el mismo kappa de
`theoretical_kappa`, y sigue en S^m. Un multiplicador real que lleva funciones
reales en funciones reales tiene que ser par; con rfftn / irfftn, ademas, un
a_theta no par haria que el simbolo efectivo cambiara de rama en la costura del
medio espacio (k_n = 0) y saliera de S^{m,rho}.
"""
from __future__ import annotations

from math import sqrt
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..symbol import Symbol
from ..utils import DropoutPlacement, default_device, dropout_layer_indices


class ParabolicMLPSymbol(Symbol):
    """a_theta(k) = <k>^m (c + (C - c) sigma(g_theta(k/<k>))), con a_theta(0) = 0.

    Simbolo escalar real (dim = 1); la escala la da <k>^m.

    Parameters
    n : int
        Dimension del toro (k in Z^n).
    m : float
        Orden del simbolo, m > 0. Fijo, no entrenable.
    c, C : float
        Cotas de la banda, 0 < c < C, fijas y no entrenables. Normalmente salen
        de `estimation.estimate_bounds` sobre los datos de entrenamiento.
    hidden : secuencia de int
        Ancho de cada capa oculta de g_theta.
    rho : int, opcional
        Parametro de regularidad (por defecto n + 1). El Teorema 2.1 exige
        rho >= n + 1; un rho menor levanta ValueError.
    dropout : float
        Tasa de dropout, fija: no es un eje del barrido.
    dropout_layers : str o secuencia de int
        En que capas ocultas se aplica ese dropout (ver DROPOUT_PLACEMENTS).
    device, dtype : torch
        Donde y con que precision vive la red. Por defecto GPU si esta disponible.
    even : bool
        Si True, el simbolo es a_par(k) = (a_theta(k) + a_theta(-k)) / 2: todo
        (`band`, `forward`, `evaluate_batch` y por tanto L_par, L_class y los
        verificadores) ve a_par. Por defecto False (el simbolo de siempre). En
        modo train con dropout, k y -k llevan mascaras distintas, asi que la
        paridad es exacta solo sin dropout o en modo eval.
    """

    def __init__(
        self,
        n: int,
        m: float,
        c: float,
        C: float,
        hidden: Sequence[int] = (64, 64, 64),
        rho: Optional[int] = None,
        dropout: float = 0.2,
        dropout_layers: DropoutPlacement = "none",
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        even: bool = False,
    ) -> None:
        if not m > 0:
            raise ValueError(f"el orden m debe ser > 0: {m}")
        if not 0.0 < c < C:
            raise ValueError(f"las cotas deben cumplir 0 < c < C: c={c}, C={C}")
        hidden = tuple(hidden)
        indices = dropout_layer_indices(dropout_layers, len(hidden))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"la tasa de dropout debe estar en [0, 1): {dropout}")
        rho = n + 1 if rho is None else int(rho)
        if rho < n + 1:
            raise ValueError(
                f"el Teorema 2.1 exige rho >= n + 1 = {n + 1}; se recibio rho={rho}"
            )
        super().__init__(
            n=n,
            m=m,
            rho=rho,
            dim=1,
            name=(
                f"ParabolicMLPSymbol(c={c:.4g}, C={C:.4g}, hidden={hidden}, "
                + ("par, " if even else "")
                + (
                    f"dropout={dropout} en capas ocultas {indices})"
                    if indices and dropout > 0.0
                    else "sin dropout)"
                )
            ),
        )
        if seed is not None:
            torch.manual_seed(seed)

        self.c = float(c)
        self.C = float(C)
        self.dropout = float(dropout)
        self.dropout_layers = indices
        self.hidden = hidden
        self.even = bool(even)
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype

        # g_theta: capas ocultas con Tanh (+ dropout), salida lineal escalar.
        # Tanh y no ReLU/GELU: g_theta tiene que ser C^infinito.
        layers = []
        width_in = n
        for i, width in enumerate(hidden):
            layers += [nn.Linear(width_in, width), nn.Tanh()]
            if self.dropout > 0.0 and i in indices:
                layers.append(nn.Dropout(self.dropout))
            width_in = width
        layers.append(nn.Linear(width_in, 1))
        self.net = nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)

    def parameters(self):
        """Parametros entrenables (para el optimizador). c, C y m no lo son."""
        return self.net.parameters()

    def linear_layers(self):
        """Las capas Linear de g_theta, en orden (entrada, ocultas..., salida)."""
        return [layer for layer in self.net if isinstance(layer, nn.Linear)]

    def decay_parameters(self):
        """Pesos que reciben weight decay: las Linear ocultas interiores.

        Se excluyen la Linear de entrada y la de salida, y los sesgos no decaen.
        """
        return [layer.weight for layer in self.linear_layers()[1:-1]]

    def band(self, k: torch.Tensor) -> torch.Tensor:
        """h_theta(k) = c + (C - c) sigma(g_theta(k/<k>)), en (c, C); forma (batch,).

        Se define tambien en k = 0 (vale lo que diga la red en x = 0); el modo
        cero lo anula `forward`. Con `even`, es el promedio sobre k y -k,
        (h_theta(k) + h_theta(-k)) / 2, con x y -x en un solo lote de la red.
        """
        k = k.to(device=self.device, dtype=self.dtype)
        bracket = torch.sqrt(1.0 + torch.sum(k**2, dim=1, keepdim=True))
        x = k / bracket
        if self.even:
            s = torch.sigmoid(self.net(torch.cat([x, -x]))[:, 0])
            s = 0.5 * (s[: len(x)] + s[len(x) :])
        else:
            s = torch.sigmoid(self.net(x)[:, 0])
        return self.c + (self.C - self.c) * s

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Version en torch, por lotes.

        k: tensor (batch, n). Devuelve a_theta(k) REAL de forma (batch,), en
        `self.dtype`. El modo k = 0 se anula con una mascara multiplicativa (sin
        `if` de Python sobre tensores): la rama de k != 0 es finita tambien en
        k = 0, asi que el gradiente no se contamina con nan.
        """
        k = k.to(device=self.device, dtype=self.dtype)
        norm2 = torch.sum(k**2, dim=1)
        bracket_m = (1.0 + norm2) ** (self.m / 2.0)
        not_zero = (norm2 > 0).to(self.dtype)
        return not_zero * bracket_m * self.band(k)

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

    def theoretical_kappa(self, omega: float = 1.0) -> float:
        """kappa con el que la Def. 2.3 se cumple en TODO Z^n, para cualquier theta.

            M  = max(1, 2^{m/2 - 1})
            kappa_1 = sqrt(2) M / min(c, 1)          (todo k != 0, Re(lambda) >= 0)
            kappa_0 = M (1 + 1/omega^m)              (k = 0, |lambda| >= omega^m)
            kappa   = max(kappa_1, kappa_0)

        Con omega = 1, kappa_0 = 2M. Para m = 2 y c = 1/2: kappa = 2 sqrt(2).

        Derivacion. Hay que ver (1 + |(k,lambda)|^2)^{m/2} <= kappa |a(k) + lambda|
        con |(k,lambda)|^2 = |k|^2 + |lambda|^{2/m}, es decir
        1 + |(k,lambda)|^2 = <k>^2 + |lambda|^{2/m}. Sea a = a_theta(k).

        (A) Si Re(lambda) >= 0 y a > 0:
                |a + lambda|^2 = a^2 + 2a Re(lambda) + |lambda|^2 >= a^2 + |lambda|^2
                               >= (1/2)(a + |lambda|)^2,
            luego a + |lambda| <= sqrt(2) |a + lambda|.
        (B) Como a >= c <k>^m (la banda),
                a + |lambda| >= c <k>^m + |lambda| >= min(c, 1)(<k>^m + |lambda|).
        (C) Lema: (X + Y)^p <= max(1, 2^{p-1})(X^p + Y^p) para X, Y >= 0, p > 0
            (subaditividad si p <= 1, convexidad de t^p si p > 1). Con p = m/2,
            X = <k>^2, Y = |lambda|^{2/m}:
                (1 + |(k,lambda)|^2)^{m/2} <= M (<k>^m + |lambda|).
            Encadenando (C), (B) y (A), para k != 0:
                (1 + |(k,lambda)|^2)^{m/2} <= M/min(c,1) (a + |lambda|)
                                           <= sqrt(2) M/min(c,1) |a + lambda|.
        (k = 0) a = 0, asi que |a + lambda| = |lambda|, y |(0,lambda)| >= omega
            equivale a |lambda| >= omega^m. Por (C) con X = 1:
                (1 + |lambda|^{2/m})^{m/2} <= M (1 + |lambda|)
                                           = M (1/|lambda| + 1) |lambda|
                                           <= M (1 + 1/omega^m) |lambda|.

        En k != 0 la cota no usa omega: vale en todo Re(lambda) >= 0.
        """
        if not omega > 0:
            raise ValueError(f"omega debe ser > 0: {omega}")
        M = max(1.0, 2.0 ** (self.m / 2.0 - 1.0))
        kappa_1 = sqrt(2.0) * M / min(self.c, 1.0)
        kappa_0 = M * (1.0 + 1.0 / omega**self.m)
        return max(kappa_1, kappa_0)

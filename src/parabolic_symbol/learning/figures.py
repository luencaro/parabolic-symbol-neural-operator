"""Figuras de resultados a partir de un modelo guardado (modelo.pt), en n = 1 y n = 2.

Cada funcion recibe el simbolo aprendido (`load_experiment_model`) y los datos
regenerados (`GRFExperiment.from_checkpoint` + `regenerate_data`) y guarda sus
graficas en la carpeta de resultados del experimento, cada una en su propia
imagen. Las que dan una sola grafica la devuelven cerrada, asi que en un
notebook se muestra una sola vez como valor de la celda; las que dan varias las
muestran una por una (`_save_show`) y devuelven None. `name` es el nombre de la
imagen, o la raiz de los nombres cuando son varias (campo.png -> campo_*.png).

1. `field_figure`: campo u(t, x) verdadero, predicho y error, una condicion inicial.
2. `multiplier_figure`: e^{-t a_theta(k)}, e^{-t a(k)} y su diferencia sobre (k, t).
3. `symbol_extrapolation_figure`: a_theta frente a |k|^alpha hasta |k| = 1000, y h_theta.
4. `extrapolation_error_figure`: error L2 en espacio fisico contra t, hasta t = 3.
5. `symbol_error_figure`: error relativo del simbolo contra |k|, entrenamiento / validacion.
6. `power_spectrum_figure`: espectro de potencia medio verdadero frente a predicho.

`summary_figure` resume los notebooks con varios alpha (resumen.csv), en dos imagenes.

Colores: RdBu_r centrada en 0 para campos y errores con signo (misma escala en
cada fila), viridis para magnitudes y curvas ordenadas por t, tab10 solo para
categorias sin orden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ..data.grf import TEST_TIMES, TRAIN_TIMES, GRFData, field_rel_l2
from ..data.grids import k_grid
from ..symbol import Symbol
from ..symbols.parabolic_mlp_symbol import ParabolicMLPSymbol
from ..utils import Split, bracket
from .estimation import symbol_values

PathLike = Union[str, Path]

# Tiempos de los paneles 2D del campo y de las curvas del espectro.
SNAPSHOT_TIMES = (0.05, 0.25, 1.0)
# Categorias (tab10): entrenamiento / validacion, y los tres tipos de tiempo.
CATEGORY_COLORS = {"entrenamiento": "C0", "validacion": "C1", "intermedios": "C1", "extrapolacion": "C2"}


def _save(fig, results: PathLike, name: str):
    fig.savefig(Path(results) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _save_show(fig, results: PathLike, name: str) -> None:
    """Guarda una de varias graficas de una funcion y la muestra (y cierra) en el notebook."""
    fig.savefig(Path(results) / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _stem(name: str) -> str:
    """Raiz de los nombres de varias imagenes: campo.png -> campo."""
    return Path(name).stem


def _a_theta(symbol: ParabolicMLPSymbol, k: np.ndarray) -> np.ndarray:
    return np.real(symbol.evaluate_batch(k)).astype(float)


def _signed(ax, image, vmax: float, extent, **kwargs):
    """imshow con RdBu_r centrada en 0 y escala simetrica [-vmax, vmax]."""
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    return ax.imshow(image, cmap="RdBu_r", norm=norm, extent=extent, **kwargs)


# ----------------------------------------------------------------------
# 1. Campo predicho frente a verdadero
# ----------------------------------------------------------------------
def field_figure(
    symbol: ParabolicMLPSymbol, data: GRFData, results: PathLike, ic: int = 0,
    times: Sequence[float] = SNAPSHOT_TIMES, name: str = "campo.png",
):
    """u(t, x) exacto, el de a_theta y su diferencia, para la condicion inicial `ic`.

    1D: tres mapas sobre (x, t) con t en [0, 1], <raiz>_{verdadero,predicho,error}.png.
    2D: un mapa por campo y por t de `times`, <raiz>_{verdadero,predicho,error}_t<t>.png.
    Verdadero y predicho comparten escala; el error tiene la suya, ambas centradas en 0.
    """
    a_theta = _a_theta(symbol, data.k)
    u0_hat = data.u0_hat[ic]

    def campo(a, t):
        return data.to_physical(np.exp(-np.multiply.outer(t, a)) * u0_hat)

    if data.n == 1:
        t = np.linspace(0.0, 1.0, 201)
        true, pred = campo(data.a, t), campo(a_theta, t)
        filas = [("verdadero", true), ("predicho", pred), ("error (predicho - verdadero)", pred - true)]
        v_field = float(np.abs(true).max())
        v_err = float(np.abs(pred - true).max()) or 1.0
        for (titulo, img), v, sufijo in zip(filas, (v_field, v_field, v_err), ("verdadero", "predicho", "error")):
            fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
            im = _signed(ax, img, v, [0, 2 * np.pi, 0, 1], origin="lower", aspect="auto")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_title(f"Campo u(t, x), condicion inicial {ic} de prueba: {titulo}")
            fig.colorbar(im, ax=ax)
            _save_show(fig, results, f"{_stem(name)}_{sufijo}.png")
        return None

    times = np.asarray(times, dtype=float)
    true = np.stack([campo(data.a, t) for t in times])
    pred = np.stack([campo(a_theta, t) for t in times])
    filas = [("verdadero", true), ("predicho", pred), ("error", pred - true)]
    v_field = float(np.abs(true).max())
    v_err = float(np.abs(pred - true).max()) or 1.0
    for (titulo, imgs), v in zip(filas, (v_field, v_field, v_err)):
        for col, t in enumerate(times):
            fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
            im = _signed(ax, imgs[col].T, v, [0, 2 * np.pi, 0, 2 * np.pi], origin="lower")
            ax.set_title(f"condicion inicial {ic} de prueba: {titulo}, t = {t:g}")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            fig.colorbar(im, ax=ax)
            _save_show(fig, results, f"{_stem(name)}_{titulo}_t{t:g}.png")
    return None


# ----------------------------------------------------------------------
# 2. Multiplicador e^{-t a(k)}
# ----------------------------------------------------------------------
def multiplier_figure(
    symbol: ParabolicMLPSymbol, reference: Symbol, results: PathLike, K: int,
    t_max: float = 1.0, name: str = "multiplicador.png",
):
    """e^{-t a_theta(k)}, e^{-t a(k)} y su diferencia sobre (k, t), k en [-K, K], en tres
    imagenes: <raiz>_aprendido.png, <raiz>_exacto.png y <raiz>_diferencia.png.

    En 2D se dibuja contra |k|: a se promedia sobre los k de la rejilla
    |k_i| <= K con el mismo |k| (el simbolo es radial).
    """
    t = np.linspace(0.0, t_max, 201)
    k = k_grid(symbol.n, K)
    a_theta, a_ref = _a_theta(symbol, k), symbol_values(reference, k)
    if symbol.n == 1:
        x, xlabel = k[:, 0], r"$k$"
    else:
        radios, inverse = np.unique(np.round(np.linalg.norm(k, axis=1), 9), return_inverse=True)
        cuenta = np.bincount(inverse)
        a_theta = np.bincount(inverse, weights=a_theta) / cuenta
        a_ref = np.bincount(inverse, weights=a_ref) / cuenta
        x, xlabel = radios, r"$|k|$"
    M_theta = np.exp(-np.multiply.outer(t, a_theta))
    M_ref = np.exp(-np.multiply.outer(t, a_ref))
    diff = M_theta - M_ref

    v = float(np.abs(diff).max()) or 1.0
    paneles = (
        ("aprendido", M_theta, r"$e^{-t\,a_\theta(k)}$", dict(cmap="viridis", vmin=0.0, vmax=1.0)),
        ("exacto", M_ref, r"$e^{-t\,a(k)}$", dict(cmap="viridis", vmin=0.0, vmax=1.0)),
        ("diferencia", diff, r"$e^{-t\,a_\theta(k)} - e^{-t\,a(k)}$",
         dict(cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -v, v))),
    )
    for sufijo, img, titulo, estilo in paneles:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        im = ax.pcolormesh(x, t, img, shading="nearest", **estilo)
        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        _save_show(fig, results, f"{_stem(name)}_{sufijo}.png")
    return None


# ----------------------------------------------------------------------
# 3. Simbolo fuera de la ventana de entrenamiento
# ----------------------------------------------------------------------
def _radial_points(n: int, r_max: float, n_r: int = 200):
    """k enteros a lo largo de direcciones fijas, |k| de 1 a r_max: {direccion: k (N, n)}."""
    radios = np.unique(np.round(np.logspace(0, np.log10(r_max), n_r)))
    if n == 1:
        return {"k > 0": radios[:, None]}
    eje = np.zeros((len(radios), n))
    eje[:, 0] = radios
    diagonal = np.unique(np.round(radios / np.sqrt(n)))
    diagonal = np.repeat(diagonal[diagonal >= 1][:, None], n, axis=1)
    return {"eje (k, 0)": eje, "diagonal (k, k)": diagonal}


def symbol_extrapolation_figure(
    symbol: ParabolicMLPSymbol, reference: Symbol, results: PathLike, k_window: int,
    r_max: float = 1000.0, name: str = "simbolo_extrapolacion.png",
):
    """a_theta(k) frente a la referencia en log-log hasta |k| = r_max (<raiz>.png), y
    h_theta = a_theta/<k>^m con la banda (<raiz>_banda.png).

    Se sombrea |k| <= k_window (= k_cap), la parte de la ventana de entrenamiento
    |k|_inf <= k_cap que cubre todas las direcciones. En 2D se toman las
    direcciones del eje y de la diagonal.
    """
    puntos = _radial_points(symbol.n, r_max)
    fig_a, ax_a = plt.subplots(figsize=(7, 5), constrained_layout=True)
    fig_h, ax_h = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axes = (ax_a, ax_h)
    todos = np.concatenate(list(puntos.values()))
    orden = np.argsort(np.linalg.norm(todos, axis=1))
    todos = todos[orden]
    norma = np.linalg.norm(todos, axis=1)
    a_ref = symbol_values(reference, todos)
    axes[0].loglog(norma, a_ref, "k-", lw=1.5, label=rf"$|k|^{{{reference.m:g}}}$")
    axes[1].semilogx(norma, a_ref / bracket(todos) ** symbol.m, "k-", lw=1.5,
                     label=rf"$|k|^{{{reference.m:g}}}/\langle k\rangle^m$")
    for i, (direccion, k) in enumerate(puntos.items()):
        norma_k = np.linalg.norm(k, axis=1)
        a = _a_theta(symbol, k)
        axes[0].loglog(norma_k, a, "o", ms=3, color=f"C{i}", label=rf"$a_\theta$, {direccion}")
        axes[1].semilogx(norma_k, a / bracket(k) ** symbol.m, "o", ms=3, color=f"C{i}",
                         label=rf"$h_\theta$, {direccion}")
    for ax in axes:
        ax.axvspan(1, k_window, color="0.85", zorder=0, label=f"ventana de entrenamiento ($|k|_\\infty\\leq{k_window}$)")
        ax.set_xlabel(r"$|k|$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$a(k)$")
    axes[0].set_title(r"Simbolo aprendido fuera de la ventana")
    axes[1].axhline(symbol.c, color="0.3", ls="--", lw=1)
    axes[1].axhline(symbol.C, color="0.3", ls="--", lw=1)
    axes[1].text(1.05, symbol.c, " c", va="bottom")
    axes[1].text(1.05, symbol.C, " C", va="top")
    axes[1].set_ylabel(r"$h_\theta(k) = a_\theta(k)/\langle k\rangle^m$")
    axes[1].set_title(rf"Banda: $c = {symbol.c:.3g}$, $C = {symbol.C:.3g}$, $m = {symbol.m:g}$")
    for ax in axes:
        ax.legend(fontsize=8)
    _save_show(fig_a, results, name)
    _save_show(fig_h, results, f"{_stem(name)}_banda.png")
    return None


# ----------------------------------------------------------------------
# 4. Error en espacio fisico contra t, con extrapolacion temporal
# ----------------------------------------------------------------------
def extrapolation_error_figure(
    symbol: ParabolicMLPSymbol, data: GRFData, results: PathLike, t_max: float = 3.0,
    name: str = "error_extrapolacion.png",
):
    """Error relativo L2 en espacio fisico (media y min-max sobre las condiciones
    iniciales de `data`) en la rejilla fina de prueba extendida hasta t_max.

    Marcadores: tiempos de entrenamiento, intermedios (t <= 1 no vistos) y de
    extrapolacion (t > 1, zona sombreada).
    """
    paso = TEST_TIMES[1] - TEST_TIMES[0]
    times = np.round(TEST_TIMES[0] + paso * np.arange(int(round((t_max - TEST_TIMES[0]) / paso)) + 1), 10)
    err = field_rel_l2(symbol, data, times)                        # (n_ic, T)
    media = err.mean(axis=0)

    es_train = np.isclose(times[:, None], TRAIN_TIMES[None, :]).any(axis=1)
    tipos = {
        "entrenamiento": es_train,
        "intermedios": ~es_train & (times <= 1.0 + 1e-9),
        "extrapolacion": times > 1.0 + 1e-9,
    }
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axvspan(1.0, times[-1], color="0.9", zorder=0, label="zona de extrapolacion (t > 1)")
    ax.fill_between(times, err.min(axis=0), err.max(axis=0), color="0.6", alpha=0.3, label="min-max")
    ax.semilogy(times, media, "-", color="0.2", lw=1)
    for tipo, sel in tipos.items():
        ax.semilogy(times[sel], media[sel], "o", ms=5, color=CATEGORY_COLORS[tipo], label=f"tiempos {tipo}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"error relativo $L^2$ en espacio fisico")
    ax.set_title(f"Error en prueba vs t ({err.shape[0]} condiciones iniciales)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, results, name)


# ----------------------------------------------------------------------
# 5. Error relativo del simbolo contra |k|
# ----------------------------------------------------------------------
def symbol_error_figure(
    symbol: ParabolicMLPSymbol, reference: Symbol, train_data: Split, val_data: Split,
    results: PathLike, name: str = "error_simbolo.png",
):
    """|a_theta(k) - a(k)| / |a(k)| contra |k| en los k unicos de entrenamiento y de validacion."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for conjunto, split in (("entrenamiento", train_data), ("validacion", val_data)):
        k = np.unique(split["k"].numpy(), axis=0)
        a_ref = symbol_values(reference, k)
        k, a_ref = k[a_ref != 0], a_ref[a_ref != 0]       # k = 0: a = 0, sin cociente
        rel = np.abs(_a_theta(symbol, k) - a_ref) / np.abs(a_ref)
        ax.semilogy(np.linalg.norm(k, axis=1), rel, "o", ms=3 if symbol.n == 1 else 2,
                    color=CATEGORY_COLORS[conjunto], alpha=0.8, label=f"{conjunto} ({len(k)} k)")
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$|a_\theta(k) - a(k)|\,/\,|a(k)|$")
    ax.set_title("Error relativo del simbolo")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return _save(fig, results, name)


# ----------------------------------------------------------------------
# 6. Espectro de potencia
# ----------------------------------------------------------------------
def power_spectrum_figure(
    symbol: ParabolicMLPSymbol, data: GRFData, results: PathLike,
    times: Sequence[float] = SNAPSHOT_TIMES, name: str = "espectro.png",
):
    """Potencia media |u_hat(t, k)|^2 sobre las condiciones iniciales de `data`, por capa
    entera de |k| (k >= 1): verdadera (continua) y predicha (punteada), una curva por t."""
    a_theta = _a_theta(symbol, data.k)
    capa = np.rint(np.linalg.norm(data.k, axis=1)).astype(int)
    cuenta = np.bincount(capa)
    r = np.nonzero(cuenta)[0]
    r = r[r >= 1]
    potencia0 = np.mean(np.abs(data.u0_hat) ** 2, axis=0)            # E|u0_hat(k)|^2

    def espectro(a, t):
        p = np.exp(-2.0 * t * a) * potencia0
        return (np.bincount(capa, weights=p) / np.maximum(cuenta, 1))[r]

    colores = plt.cm.viridis(np.linspace(0.0, 0.85, len(times)))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    pico = 0.0
    for t, color in zip(times, colores):
        verdadero = espectro(data.a, t)
        pico = max(pico, float(verdadero.max()))
        ax.loglog(r, verdadero, "-", color=color, lw=1.5, label=f"t = {t:g}, verdadero")
        ax.loglog(r, espectro(a_theta, t), ":", color=color, lw=2, label=f"t = {t:g}, predicho")
    ax.set_ylim(pico * 1e-30, pico * 10)
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$\mathbb{E}\,|\hat u(t, k)|^2$")
    ax.set_title(f"Espectro de potencia medio ({data.u0_hat.shape[0]} condiciones iniciales)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, results, name)


# ----------------------------------------------------------------------
# Resumen de los notebooks con varios alpha
# ----------------------------------------------------------------------
def summary_figure(resumen: Union[PathLike, pd.DataFrame], results: PathLike, name: str = "resumen.png"):
    """Error en prueba (<raiz>_error_prueba.png) y peor ratio de certificacion
    (<raiz>_certificacion.png) contra alpha, de resumen.csv; cada una en su imagen."""
    tabla = resumen if isinstance(resumen, pd.DataFrame) else pd.read_csv(resumen)
    tabla = tabla.sort_values("alpha")

    fig_e, ax_e = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for i, (col, etiqueta) in enumerate((("test_rel_l2", "pares (k, t)"),
                                         ("test_rel_l2_fisico", "espacio fisico"))):
        ax_e.semilogy(tabla["alpha"], tabla[col], "o-", color=f"C{i}", label=etiqueta)
    ax_e.set_ylabel(r"error relativo $L^2$ en prueba")
    ax_e.set_title("Error en prueba")
    ax_e.legend()

    fig_c, ax_c = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax_c.plot(tabla["alpha"], tabla["peor_ratio"], "o-", color="C0")
    ax_c.axhline(1.0, color="0.3", ls="--", lw=1, label="cota de la Def. 2.3 (ratio = 1)")
    ax_c.set_ylabel("peor ratio de la certificacion")
    ax_c.set_title("Certificacion de parabolicidad")
    ax_c.legend()

    for fig, ax, sufijo in ((fig_e, ax_e, "error_prueba"), (fig_c, ax_c, "certificacion")):
        ax.set_xlabel(r"$\alpha$")
        ax.set_xticks(tabla["alpha"])
        ax.grid(alpha=0.3, which="both")
        _save_show(fig, results, f"{_stem(name)}_{sufijo}.png")
    return None

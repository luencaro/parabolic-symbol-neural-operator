# parabolic-symbol-neural-operator

Aprendizaje del símbolo $a_\theta:\mathbb Z^n\to\mathbb R$ del **generador** de un problema de Cauchy parabólico sobre el toro $\mathbb T^n=[0,2\pi)^n$,

$$\partial_t u + \mathrm{op}[a_\theta]\,u = 0,\qquad u(0)=u_0,\qquad \widehat u(t,k)=e^{-t\,a_\theta(k)}\,\widehat u_0(k),$$

con la convención $u(x)=\sum_k \widehat u(k)\,e^{ik\cdot x}$. La referencia es el calor, $a(k)=|k|^2$ (`HeatSymbol`). El operador solución no tiene parámetros propios: se obtiene exponenciando el símbolo aprendido en cada modo de Fourier.

## La arquitectura: `ParabolicMLPSymbol`

**Dónde:** [`src/parabolic_symbol/symbols/parabolic_mlp_symbol.py`](src/parabolic_symbol/symbols/parabolic_mlp_symbol.py).

$$a_\theta(k)=\langle k\rangle^m\Big(c+(C-c)\,\sigma\big(g_\theta(k/\langle k\rangle)\big)\Big)\quad (k\neq0),\qquad a_\theta(0)=0,$$

con $\langle k\rangle=(1+|k|^2)^{1/2}$, $\sigma$ la sigmoide logística y $g_\theta:\mathbb R^n\to\mathbb R$ una MLP con `Tanh`:

```
x = k / <k>  ->  [ Linear(n -> w) -> Tanh (-> Dropout) ] x L  ->  Linear(w -> 1)  =  g_theta(x)
a_theta(k)   =  <k>^m * ( c + (C - c) * sigmoid(g_theta(x)) )
```

El orden $m$ y la banda $0<c<C$ son fijos; solo se entrena $g_\theta$. La escala la da $\langle k\rangle^m$ y la red solo aprende un factor acotado, así que, para **cualquier** $\theta$,

$$c\,\langle k\rangle^m < a_\theta(k) < C\,\langle k\rangle^m\qquad (k\neq0).$$

### Parabolicidad con $\kappa$ explícito (Def. 2.3)

La Def. 2.3 pide $|(a(k)+\lambda)^{-1}|\le\kappa\,(1+|(k,\lambda)|^2)^{-m/2}$ para $\mathrm{Re}\,\lambda\ge0$ y $|(k,\lambda)|\ge\omega$, con $|(k,\lambda)|^2=|k|^2+|\lambda|^{2/m}$. Gracias a la banda se cumple en todo $\mathbb Z^n$ con

$$M=\max(1,2^{m/2-1}),\qquad \kappa=\max\Big(\frac{\sqrt2\,M}{\min(c,1)},\ M\big(1+\omega^{-m}\big)\Big),$$

que calcula `ParabolicMLPSymbol.theoretical_kappa(omega)`. La idea de la prueba, para $k\neq0$:

1. Si $\mathrm{Re}\,\lambda\ge0$ y $a>0$, entonces $a+|\lambda|\le\sqrt2\,|a+\lambda|$.
2. Por la banda, $a+|\lambda|\ge\min(c,1)\,(\langle k\rangle^m+|\lambda|)$.
3. Por la desigualdad $(X+Y)^p\le M(X^p+Y^p)$, $(1+|(k,\lambda)|^2)^{m/2}\le M(\langle k\rangle^m+|\lambda|)$.

El caso $k=0$ da el segundo término del máximo. La derivación completa está en el docstring de `theoretical_kappa`. Por eso $\kappa$ no es un hiperparámetro: el entrenamiento toma `symbol.theoretical_kappa()`.

### Clase $S^{m,\rho}$ (Def. 2.1)

La Def. 2.1 pide $\|\Delta_k^\alpha a(k)\|\le C_\alpha\langle k\rangle^{m-|\alpha|}$ para $|\alpha|\le\rho$, con $\Delta_{k_j}a(k)=a(k+e_j)-a(k)$. Por construcción:

- $x(k)=k/\langle k\rangle$ es un símbolo de orden $0$ con imagen en la bola unidad abierta.
- $\sigma\circ g_\theta$ es $C^\infty$ y está acotada, con todas sus derivadas, sobre esa bola. Por eso se usa `Tanh` y no ReLU.
- $\langle k\rangle^m\in S^m$, y el producto queda en $S^m$. Fijar $a_\theta(0)=0$ solo altera finitas diferencias.

La pertenencia está garantizada cualitativamente, pero las constantes $C_\alpha$ dependen de $\theta$. Las controla la pérdida $\mathcal L_{\text{class}}$. Se usa $\rho=n+1$, que es lo que exige el Teorema 2.1.

### Estimación de $c$ y $C$

**Dónde:** [`estimation.py`](src/parabolic_symbol/learning/estimation.py). Se usan solo datos de entrenamiento.

1. **`estimate_symbol`.** Como $\log|\widehat u(t,k)/\widehat u_0(k)|=-t\,a(k)$, para cada $k$ se ajusta una recta por el origen por mínimos cuadrados ponderados, con pesos $|\widehat u_0|^2$:
   $\hat a(k)=-\sum_i w_i t_i y_i\big/\sum_i w_i t_i^2$.
   Se descartan los pares con $|\widehat u|\le \text{tol}\,|\widehat u_0|$ (máscara de recuperabilidad).
2. **`estimate_bounds`.** Con $h(k)=\hat a(k)/\langle k\rangle^m$ para $k\neq0$, se toma $c=(1-\delta)\min h$ y $C=(1+\delta)\max h$, con $\delta=0.1$.

Para el calor sale $(c,C)\approx(0.45,\,1.10)$, y con $\omega=1$, $\kappa=\sqrt2/0.45\approx3.14$.

## Pérdidas y entrenamiento

**Dónde:** [`losses.py`](src/parabolic_symbol/learning/losses.py) y [`training.py`](src/parabolic_symbol/learning/training.py).

$$\mathcal L=\lambda_{\text{traj}}\,\mathcal L_{\text{traj}}+\lambda_{\text{par}}\,\mathcal L_{\text{par}}+\lambda_{\text{class}}\,\mathcal L_{\text{class}}$$

| Término | Función | Qué calcula |
|---|---|---|
| $\mathcal L_{\text{traj}}$ | `trajectory_loss` | $\mathbb E\,\lvert e^{-ta_\theta(k)}\widehat u_0(k)-\widehat u(t,k)\rvert^2$ |
| $\mathcal L_{\text{par}}$ | `parabolicity_loss` | hinge² de $\lvert(a_\theta(k)+\lambda)^{-1}\rvert\le\kappa(1+\lvert(k,\lambda)\rvert^2)^{-m/2}$, sobre $\mathrm{Re}\,\lambda\ge0$ y $\lvert(k,\lambda)\rvert\ge\omega$ |
| $\mathcal L_{\text{class}}$ | `class_loss` | hinge² de $\langle k\rangle^{\lvert\alpha\rvert-m}\lvert\Delta_k^\alpha a_\theta(k)\rvert\le C_\alpha$ para $\lvert\alpha\rvert\le\rho$ |

- $\mathcal L_{\text{par}}$ y $\mathcal L_{\text{class}}$ se evalúan en puntos $(k,\lambda)$ y $k$ nuevos en cada lote. Las diferencias finitas de $\mathcal L_{\text{class}}$ salen de un solo `forward` por lotes, así que el término es diferenciable.
- Las $C_\alpha$ se calculan del símbolo de referencia en la ventana de entrenamiento, con margen $1.5$.
- **Optimización:** `AdamW` con `lr = 1e-3`, que decae exponencialmente hasta `1e-5`, y recorte de gradiente en 1. El weight decay solo se aplica a las `Linear` ocultas interiores.
- **Early stopping:** paciencia de 30 épocas sobre la pérdida total de validación, **incluyendo $\mathcal L_{\text{class}}$** (`early_stopping_class=True`). Al terminar se restauran los pesos de la mejor época.
- **Configuración base:** los valores por defecto de `TrainingConfig`. En los notebooks solo se pasan `k_max` y `class_C`, que dependen de los datos.

**Barrido** (`run_sweep` en [`sweep.py`](src/parabolic_symbol/learning/sweep.py)). Combina:

- profundidad $\{2,3,4\}$ y ancho $\{32,64,128\}$;
- dropout $0.2$ en `"none"` o `"middle"` (capas ocultas interiores);
- weight decay $\{0,10^{-4}\}$.

Las combinaciones equivalentes se eliminan, así que quedan 30 entrenamientos. `best_run` elige la corrida de **menor error relativo $L^2$ de validación**.

**Partición.** La partición entrenamiento/validación es **por $k$**: 80 % / 20 % de los $k$ únicos. Así, la validación mide generalización a frecuencias que no se vieron. La parabolicidad del modelo elegido se certifica en una rejilla independiente:

- $|k_i|\le3k_{\max}$;
- $|\lambda|\in[10^{-2},10^3]$ con 25 ángulos.

## Estructura

```
pyproject.toml                    # paquete instalable (pip install -e .)
src/parabolic_symbol/
├── __init__.py
├── symbol.py                     # clase abstracta a: Z^n -> L(E)
├── utils.py                      # tipos, <k>, multiindices, device, dropout
├── symbols/
│   ├── parabolic_mlp_symbol.py   # ARQUITECTURA: ParabolicMLPSymbol
│   ├── heat.py                   # a(k) = |k|^2 (referencia)
│   ├── fractional_laplacian.py   # a(k) = |k|^alpha
│   └── counterexample.py         # simbolos que NO son parabolicos (controles)
├── verification/                 # Defs. 2.1 y 2.3
│   ├── parabolicity.py           # ParabolicityVerifier: estima kappa de la Def. 2.3
│   └── symbol_class.py           # SymbolClassVerifier: estima C_alpha de la Def. 2.1
├── data/
│   ├── synthetic.py              # muestras (k, t, u0, u), puntos (k, lambda) y particion por k
│   └── grf.py                    # calor 1D con condiciones iniciales GRF (paper, Sec. 5.1)
└── learning/
    ├── estimation.py             # a_hat(k), banda (c, C) y C_alpha desde los datos
    ├── losses.py                 # L_traj, L_par, L_class
    ├── training.py               # TrainingConfig y bucle de entrenamiento
    ├── evaluation.py             # metricas y peor ratio de parabolicidad
    └── sweep.py                  # barrido de arquitecturas y seleccion
notebooks/
├── 01_verificadores.ipynb        # verificadores sobre simbolos analiticos y contraejemplos
├── 02_barrido.ipynb              # barrido y certificacion, calor en T^2
└── 03_barrido_grf.ipynb          # lo mismo con los datos GRF del calor 1D + conjunto de prueba
results/
├── 02_barrido/                   # barrido.csv y figuras PNG
└── 03_barrido_grf/
book/myst.yml                     # Jupyter Book con los tres notebooks
```

## Notebooks

- **`01_verificadores`:** comprueba `ParabolicityVerifier` y `SymbolClassVerifier` sobre el calor, el laplaciano fraccionario y símbolos que no son parabólicos.
- **`02_barrido`:** calor en $\mathbb T^2$ con $10^5$ muestras $(k,t)$. Tiene $|k_i|\le12$, $t$ log-uniforme en $[10^{-3},1]$ y $\widehat u_0\sim\mathcal N_{\mathbb C}(0,1)$. Incluye la estimación de $(c,C)$, el barrido, las métricas del modelo elegido, las curvas de entrenamiento, el símbolo aprendido frente a $|k|^2$ y la certificación de las Def. 2.3 y 2.1.
- **`03_barrido_grf`:** el mismo flujo con los datos del paper. Es el calor en $\mathbb T^1$ con resolución 1024 y $u_0\sim\mathcal N(0,7^4(-\Delta+7^2)^{-2.5})$:
  - **Entrenamiento:** 100 condiciones iniciales en $t=0.05+0.1j$.
  - **Prueba:** 20 condiciones iniciales nuevas en $t=0.05+0.05j$.
  - Se descartan los pares $(k,t)$ con $e^{-ta(k)}\le\varepsilon_{\texttt{float32}}$, lo que deja $|k|\le17$.
  - Reporta el error relativo $L^2$ en prueba, también en espacio físico.

## Instalación

Con el devcontainer el entorno se crea solo. Hay dos opciones:

- [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json): funciona sin GPU.
- [`.devcontainer/gpu/devcontainer.json`](.devcontainer/gpu/devcontainer.json): usa `--gpus all` y requiere el NVIDIA Container Toolkit.

A mano:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Los notebooks importan `parabolic_symbol` como paquete instalado y se ejecutan desde `notebooks/`. Usan GPU y `float32` si hay CUDA; si no, CPU y `float64`. El notebook `03` exige CUDA. Para el libro: `cd book && jupyter book start`.

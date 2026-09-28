# parabolic-symbol-neural-operator

Aprendizaje del símbolo $a_\theta:\mathbb Z^n\to\mathbb R$ del **generador** de un problema de Cauchy parabólico sobre el toro $\mathbb T^n$,

$$\partial_t u + \mathrm{op}[a_\theta]\,u = 0,\qquad u(0)=u_0,$$

con una red neuronal y una pérdida tipo PINN que penaliza la parabolicidad (Def. 2.3) y la pertenencia a la clase $S^{m,\rho}$ (Def. 2.1). La referencia es el calor, $a(k)=|k|^2$. El repositorio también incluye verificadores numéricos para ambas definiciones.

## Estructura

```
src/parabolic_symbol/
├── Symbol.py                 # clase abstracta a: Z^n -> L(E)
├── symbols/
│   ├── heat.py               # a(k) = |k|^2
│   ├── fractional_laplacian.py  # a(k) = |k|^alpha
│   ├── counterexample.py     # simbolos que NO son parabolicos (controles)
│   └── mlp_symbol.py         # ARQUITECTURA: a_theta(k) = MLP(k)
├── ParabolicityVerifier.py   # estima kappa de la Def. 2.3
├── SymbolClassVerifier.py    # estima C_beta de la Def. 2.1
├── data.py                   # generacion de datos y PARTICION train/val
├── losses.py                 # L_traj, L_par, L_class
├── training.py               # DESCENSO DEL GRADIENTE (bucle de entrenamiento)
├── evaluation.py             # metricas
├── sweep.py                  # barrido de kappa y arquitecturas, seleccion
└── utils.py                  # tipos y multi-indices
notebooks/
├── exploration.ipynb         # verificadores sobre simbolos analiticos y contraejemplos
└── mlp_symbol_training.ipynb # entrenamiento, barrido, certificacion
book/myst.yml                 # Jupyter Book con los dos notebooks
```

## Instalación

Con el devcontainer (`.devcontainer/devcontainer.json`), el entorno se crea solo. A mano:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Los notebooks importan el paquete con `sys.path.insert(0, "../src")`, así que se ejecutan desde `notebooks/`. Para el libro: `cd book && jupyter book start`.

## Arquitectura

**Dónde:** clase `MLPSymbol` en [`src/parabolic_symbol/symbols/mlp_symbol.py`](src/parabolic_symbol/symbols/mlp_symbol.py). Las capas se construyen en `__init__`.

**Cómo es:** un perceptrón multicapa completamente conectado, con entrada $k\in\mathbb Z^n$ y una única salida real.

```
k  ->  k / input_scale
   ->  [ Linear(n -> w) -> Tanh (-> Dropout) ]      capa oculta 1
   ->  [ Linear(w -> w) -> Tanh (-> Dropout) ] x (L-1)
   ->  Linear(w -> 1)
   ->  * output_scale  =  a_theta(k)  (real)
```

- La profundidad `L` y el ancho `w` vienen de `hidden=(w, ..., w)`. En el barrido se usa $L\in\{2,3,4\}$ y $w\in\{32,64,128\}$.
- `input_scale` y `output_scale` son constantes fijas, no entrenables. En el notebook valen `k_max` y `k_max**2`.
- El dropout tiene tasa fija (0.2). Lo que se barre es dónde se aplica: `"none"` o `"middle"`, es decir, las capas ocultas interiores.
- El símbolo es real porque el objetivo $|k|^\alpha$ es real. La conversión a complejo solo se hace en `evaluate` y `evaluate_batch`, para cumplir la interfaz de `Symbol`.

El operador solución **no tiene parámetros propios**: se obtiene exponenciando el símbolo aprendido en cada modo de Fourier,

$$\widehat u(k,t)=e^{-t\,a_\theta(k)}\,\widehat u_0(k).$$

## Forward

**Dónde:** `MLPSymbol.forward` en [`mlp_symbol.py`](src/parabolic_symbol/symbols/mlp_symbol.py).

```python
def forward(self, k):                          # k: (batch, n)
    out = self.net(k / self.input_scale) * self.output_scale
    return out[:, 0]                           # a_theta(k): (batch,)
```

Todas las pérdidas llaman a `forward`, en [`losses.py`](src/parabolic_symbol/losses.py):

| Término | Función | Qué calcula |
|---|---|---|
| $\mathcal L_{\text{traj}}$ | `trajectory_loss` | $\mathbb E\,\lvert e^{-ta_\theta(k)}\widehat u_0(k)-\widehat u(k,t)\rvert^2$ |
| $\mathcal L_{\text{par}}$ | `parabolicity_loss` | hinge² de $\lvert(a_\theta(k)+\lambda)^{-1}\rvert\le\kappa\langle k,\lambda\rangle^{-m}$, sobre $\mathrm{Re}\,\lambda\ge0$ y $\lvert(k,\lambda)\rvert\ge\omega$ |
| $\mathcal L_{\text{class}}$ | `class_loss` | hinge² de $\langle k\rangle^{\lvert\beta\rvert-m}\lvert\triangle_k^\beta a_\theta(k)\rvert\le C_\beta$ para $\lvert\beta\rvert\le\rho$ |

Las diferencias finitas $\triangle_k^\beta$ se calculan evaluando `forward` en $k+\gamma$ para todo $\gamma\le\beta$ en un solo lote. Por eso $\mathcal L_{\text{class}}$ es diferenciable.

## Descenso del gradiente

**Dónde:** función `train` en [`src/parabolic_symbol/training.py`](src/parabolic_symbol/training.py). Los hiperparámetros están en `TrainingConfig`.

- **Optimizador:** `AdamW`, con `lr = 1e-3`. Tiene dos grupos de parámetros: el weight decay solo se aplica a los pesos de las `Linear` ocultas interiores (`MLPSymbol.decay_parameters`). La capa de entrada, la de salida y los sesgos no decaen.
- **Tasa de aprendizaje:** `ExponentialLR`, que decae por época de `lr` a `lr_final = 1e-5`.
- **Mini-lotes:** los datos de trayectoria se barajan en cada época, en lotes de `batch_size = 1024`.

En cada lote:

```python
l_traj = trajectory_loss(symbol, k, t, u0, u_ref)                       # lote de datos
k_p, lam_p = sample_phase_points(n, k_max, B, lam_min, lam_max, ...)    # (k, lambda) nuevos
l_par  = parabolicity_loss(symbol, k_p, lam_p, omega, kappa)
k_c    = sample_k_points(n, k_max, B, ...)                              # k nuevos
loss   = l_traj + l_par + l_class(symbol, k_c, C)                       # con sus pesos lambda_*

optimizer.zero_grad()
loss.backward()
clip_grad_norm_(params, 1.0)        # recorte de gradiente
optimizer.step()
```

- **Early stopping:** con paciencia de 30 épocas, sobre la pérdida total de validación. Al terminar se restauran los pesos de la mejor época.
  - Esa pérdida total es trayectoria + parabolicidad y **no incluye $\mathcal L_{\text{class}}$** (ver `evaluate_split` en [`evaluation.py`](src/parabolic_symbol/evaluation.py)).
- **Barrido:** `run_sweep` en [`sweep.py`](src/parabolic_symbol/sweep.py) llama a `train` una vez por combinación de $\kappa$ × profundidad × ancho × dropout × weight decay.
- **Selección:** `best_run` elige, entre las corridas que cumplen la cota, la de menor error relativo $L^2$ de validación.

## Partición de los datos y data leakage

**Dónde:** `make_trajectory_data` y `split_trajectory_data` en [`src/parabolic_symbol/data.py`](src/parabolic_symbol/data.py).

`make_trajectory_data` genera muestras $(k,t,\widehat u_0,\widehat u)$:

- $k$ se toma **con reemplazo** de la rejilla $\{|k_i|\le k_{\max}\}$.
- $t$ es log-uniforme en $[10^{-3},1]$.
- $\widehat u_0\sim\mathcal N_{\mathbb C}(0,1)$.
- $\widehat u=e^{-t|k|^2}\widehat u_0$.

**El riesgo.** Como hay muchas muestras por cada $k$, una partición aleatoria *por muestra* dejaría casi todos los $k$ de validación también en entrenamiento. La red se evaluaría en frecuencias que ya vio.

**La solución: partición por $k$.** `split_trajectory_data` reparte los $k$ **únicos**: 80 % a entrenamiento y 20 % a validación, con semilla fija. Todas las muestras de un mismo $k$ caen en el mismo conjunto, así que la validación mide generalización a frecuencias nunca vistas. `val_fraction` es entonces una fracción de $k$ únicos, no de muestras.

```python
unique_k, inverse = torch.unique(data["k"], dim=0, return_inverse=True)
is_val = ...                      # 20% de los k unicos al azar
val_mask = is_val[inverse]        # cada muestra hereda el conjunto de su k
```

**Rejillas separadas para decidir y para certificar.** La parabolicidad se evalúa en tres conjuntos distintos de puntos $(k,\lambda)$:

1. **Entrenamiento:** puntos aleatorios nuevos en cada lote (`sample_phase_points`).
2. **Selección:** una rejilla fija (`check_phase`, construida con `phase_grid`). Decide qué corridas "cumplen" y estima el $\hat\kappa$ de la referencia.
3. **Certificación:** una rejilla independiente, con $|\lambda|\in[10^{-2},10^3]$, 25 ángulos y $|k_i|\le 3k_{\max}$. Solo se aplica al modelo ya elegido, nunca durante la selección.

**Lo que la partición no cubre.** Conviene tenerlo presente al leer las métricas:

- $\mathcal L_{\text{par}}$ y $\mathcal L_{\text{class}}$ muestrean $k$ en toda la ventana $|k_i|\le k_{\max}$, incluidos los $k$ de validación. No usan el valor de referencia $|k|^2$ en esos puntos, solo cotas; aun así, la red recibe gradiente allí.
- Las constantes $C_\beta$ de $\mathcal L_{\text{class}}$ se estiman del símbolo analítico en toda la ventana.
- El conjunto de validación se usa tanto para el early stopping como para elegir la mejor corrida del barrido. No hay un conjunto de prueba aparte, así que el error de validación del modelo elegido es algo optimista.
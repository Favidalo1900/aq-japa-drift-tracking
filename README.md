# aq-JAPA — Jackson Regularization for Affine-Projection Online Learning Under Concept Drift

Código de los experimentos del artículo *q-JAPA: Jackson Regularization for
Affine-Projection Online Learning Under Concept Drift* (IDEAL 2026).

---

## Qué reproduce cada cosa

Todo lo que aparece en el artículo — las dos tablas y la Figura 1 — sale de
`experimentos/`, y todo desde el mismo motor, `core.py`.

### `experimentos/`

| script | qué renglón del paper produce |
|---|---|
| `core.py` | el motor: genera el flujo AR(1), define los tres algoritmos y, corrido directo (`python core.py`), produce los dos paneles de la **Figura 1**. Todo lo demás lo importa. |
| `exp1_condicion_signo.py` | `Steps with u_M w_M > 0 (%)` en los dos paneles, y el rango 59 %–74 % de §4. Su PARTE 1 muestra además, paso a paso sobre una realización, el instante en que el signo se voltea. |
| `exp2_tablas.py` | los MSE causales de los tres métodos, `MSE reduction vs. APA` y `MSD reduction vs. APA` |
| `exp4_holm.py` | `Holm-rejected (m=15)` y los intervalos de confianza al 95 % |

```bash
cd experimentos
mkdir -p ../resultados

python exp2_tablas.py 20     # reproduce las tablas ENVIADAS, dígito por dígito
python exp2_tablas.py 500    # las del artículo publicado (~40 min)
python exp1_condicion_signo.py 500
python exp4_holm.py 500
python exp4_holm.py 500 growth
python core.py               # la Figura 1, segundos

Los tres experimentos piden el número de realizaciones como argumento
obligatorio: no hay valor por defecto.
```

**La comprobación que importa:** `python exp2_tablas.py 20` devuelve
`17.24 16.74 15.31 11.03 6.32` y `0.71 0.65 0.53 0.34 0.21`, que es exactamente
la Tabla 1 de la versión enviada. Eso valida el motor; el cambio de números al
pasar a n = 500 es efecto del tamaño de muestra, no del código.

## Sobre la versión enviada

El código que acompañó a la versión que se envió a revisión
(`aq_japa_simulation.py`) ya no está en este repositorio: reproducía la tabla de
aquella versión, con n = 20, solo APA y aq-JAPA y solo el régimen de
contracción. Quedó fuera para que aquí haya **una sola cosa que correr**, y que
sea la que corresponde al artículo publicado. Sigue disponible en el historial
de git.

Conviene saber que aquel script tenía un error: medía `errors[t]` **después** de
actualizar `w`, es decir el error **a posteriori**, mientras que el artículo
declara el **causal**. La diferencia no es cosmética — reportaba una mejora de
**18.43 %** donde el artículo dice **17.24 %**, porque el error a posteriori ya
vio la respuesta del instante y es optimista. `core.py` siempre usó el causal.

---

## Los tres algoritmos

Con $G_t=(X_t^\top X_t+\epsilon I)^{-1}$ y $D_t=\operatorname{diag}(X_tX_t^\top)$:

| clave | nombre | actualización |
|---|---|---|
| `apa` | APA | $w+\mu X_tG_te_t$ |
| `qfix` | q-JAPA, $q$ fijo | $w+\mu X_tG_te_t-\gamma D_tw$, con $\gamma=\tfrac\mu2(q-1)$, $q=1.15$ |
| `aq` | **aq-JAPA** | $w+\mu X_tG_te_t-\gamma_tD_tw$, con $q_t=1+\alpha\psi_t$ |

Los tres corren sobre **la misma semilla**, así que todas las comparaciones
son pareadas.

## Parámetros

Están todos en `experimentos/core.py`, en un solo bloque. Se cambian ahí y los
tres experimentos quedan consistentes entre sí.

```
T = 1200    M = 10      K = 5       rho = 0.99
mu = 0.05   eps = 1e-3  ruido = 0.05
kappa_diag(R_x) = 1e3   H = 100     T_d = 600
alpha = 0.15  beta = 0.97  L = 80   q fijo = 1.15
```

No hay excepciones: los tres experimentos importan todo de `core.py`.

## Requisitos

```bash
pip install -r requirements.txt
```

`numpy`, `scipy` y `matplotlib`.

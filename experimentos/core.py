"""
core.py  --  nucleo compartido de los experimentos probado en los algoritmos apa, q-japa y aq-JAPA 

Contiene el generador de flujo (datos) y TODOS los algoritmos.
Los scripts exp0..exp4 solo importan de aqui.

NOTACION
    x_t in R^M          entrada en el instante t
    X_t in R^{MxK}      bloque [x_t, ..., x_{t-K+1}]
    d_t in R^K          bloque de salidas deseadas
    e_t = d_t - X_t^T w_t
    G_t = (X_t^T X_t + eps I)^{-1}     preacondicionador de APA
    D_t = diag(X_t X_t^T)              energias instantaneas por coordenada
    w*                  vector optimo; cambia en t = T_d

LOS TRES ALGORITMOS (mismo mu, eps, K, rho y MISMA semilla => pareados)
    'apa'    w_{t+1} = w_t + mu X_t G_t e_t
    'qfix'   w_{t+1} = w_t + mu X_t G_t e_t - gamma   D_t w_t   gamma=(mu/2)(q-1), q=1.15
    'aq'     w_{t+1} = w_t + mu X_t G_t e_t - gamma_t D_t w_t   q_t = 1 + alpha psi_t

    Con preconditioning=False se omite G_t: es la ablacion "APA sin
    preacondicionador", que sirve para ver que aporta G_t. El paper corre
    siempre con preconditioning=True.

ESTRUCTURA
    data_generation(seed, c, modo)          el flujo: x_t, y_t y el optimo w*
    algorithm_execution(...)                corre uno de los tres algoritmos
    holm(ps, alpha)                         la correccion por comparaciones multiples
    figura_transitorio(...)                 LA FIGURA 1 del paper, sus dos paneles

QUE USA CADA EXPERIMENTO
    exp1 -> data_generation, algorithm_execution(historial=True)   condicion de signo
    exp2 -> data_generation, algorithm_execution                   las dos tablas
    exp4 -> data_generation, algorithm_execution, holm             familia de 15 + Holm

    Solo exp1 pide el historial. exp2 y exp4 usan la forma corta, que devuelve
    unicamente los dos numeros de la ventana post-drift.

PARA GENERAR LA FIGURA 1
    python core.py
"""
import numpy as np

# =============================================================================
#  PARAMETROS: los cinco experimentos los importan de aquí
# =============================================================================
T, M, K   = 1200, 10, 5        # T pasos totales; M coordenadas; K = orden de proyeccion
RHO       = 0.99               # correlacion temporal del AR(1): cerca de 1 = muy correlacionado
MU, EPS   = 0.05, 1e-3         # mu = paso de adaptacion; eps = regularizacion de la inversa
NOISE     = 0.05               # varianza del ruido de observacion nu_t
KAPPA     = 1e3                # kappa_diag(R_x): razon energia maxima / energia minima
ALPHA, BETA, L = 0.15, 0.97, 80   # aq-JAPA: escala de psi, factor de la EMA, horizonte de decaimiento
Q_FIXED   = 1.15                  # q de la linea base fija = 1 + ALPHA (el maximo que alcanza aq-JAPA)
H         = 100                   # ventana post-drift sobre la que se promedian MSE y MSD
CS        = [0.05, 0.10, 0.20, 0.40, 0.60]   # factores de drift para shrinkage o growth utilizados
ALGORITMOS = ["apa", "aq", "qfix"]           # orden fijo: los experimentos dependen de el
NOMBRE = {"apa": "APA", "aq": "aq-JAPA", "qfix": "q fijo"} #Diccionario de abreviación de algoritmos


def data_generation(seed, c, modo="shrink"):
    """Data: las entradas x_t, las salidas d_t y el optimo w*.

    La semilla entra como argumento para que
    todos los algoritmos se evaluen exactamente sobre los mismos datos: 

    modo 'shrink': w* pasa de e_M  a  c e_M   (el coeficiente debe BAJAR)
    modo 'growth': w* pasa de c e_M a  e_M    (el coeficiente debe SUBIR)
    """
    # --- SEMILLA -------------------------------------------------------------
    # De aquí sale el modelo AR(1) y el ruido de observacion.

    rng = np.random.default_rng(seed)

    # --- ENERGIAS POR COORDENADA ---------------------------------------------
    # sig2[m] es la varianza estacionaria de la coordenada m: progresion
    # geometrica de 0.1 a 0.1*KAPPA. La coordenada 0 es la mas debil y la M-1 la
    # mas fuerte, y la razon max/min vale exactamente KAPPA = 1000, que es el
    # kappa_diag(R_x) que reporta el paper.
    sig2 = 0.1 * KAPPA ** (np.arange(M) / (M - 1))

    # --- AQUI SE GENERA EL MODELO x_t: M procesos AR(1) INDEPENDIENTES -------
    # Recursion:  x_t[m] = RHO * x_{t-1}[m] + eta_t[m],
    # con eta_t[m] ~ N(0, (1-RHO^2) * sig2[m]). Ese factor (1-RHO^2) es lo que
    # hace que la varianza ESTACIONARIA de x_t[m] sea exactamente sig2[m].
    X = np.zeros((T, M))                   # aqui se va guardando el flujo completo
    xp = np.zeros(M)                       # x_{t-1}: arranca en cero
    sc = np.sqrt(sig2 * (1 - RHO ** 2))    # desviaciones estandar de las innovaciones
    for t in range(T):
        # un paso de la recursion AR(1), las M coordenadas de golpe
        xp = RHO * xp + rng.normal(scale=sc, size=M)
        X[t] = xp

    # Normalizacion: se divide TODO por la misma constante, la raiz de la
    # varianza empirica mas grande. Con eso la energia maxima queda en 1, como
    # dice el paper, y la RAZON entre energias no se toca: sigue siendo KAPPA.
    X = X / np.sqrt(np.max(np.var(X, axis=0)))

    # --- EL OPTIMO w* Y EL DRIFT ---------------------------------------------
    # Solo la ultima coordenada (la de energia maxima, e_M) es distinta de cero.
    # Es donde pega el drift y donde la fuga de Jackson tiene el efecto mas
    # grande, porque D_t pondera por energia.
    w1, w2 = np.zeros(M), np.zeros(M)
    if modo == "shrink": w1[-1], w2[-1] = 1.0, c    # 1 -> c : el peso debe BAJAR
    else:                w1[-1], w2[-1] = c, 1.0    # c -> 1 : el peso debe SUBIR
    Td = T // 2                                     # el drift ocurre a la mitad

    # --- AQUI SE GENERAN LAS SALIDAS y_t (en el codigo se llaman d) ----------
    # d_t = x_t^T w*_t + nu_t, con w*_t = w1 antes del drift y w2 despues.
    # nu_t es ruido blanco de varianza NOISE, independiente de x_t.
    d = np.array([X[t] @ (w1 if t < Td else w2) + rng.normal(scale=np.sqrt(NOISE))
                  for t in range(T)])
    return X, d, w1, w2, Td


def algorithm_execution(X, d, w2, Td, algoritmo,
                        preconditioning=True, historial=False):
    """Corre UN algoritmo sobre el flujo ya generado con data_generation y devuelve sus metricas.

    Devuelve (MSE causal post-drift, MSD post-drift). Con historial=True
    devuelve ademas h, el registro paso a paso de w_M, u_M = w_M - w*_M, el
    producto u_M*w_M, q_t, gamma_t, lambda_max(D_t), ||w_t|| y el error causal.
    De ahi salen la condicion de signo de exp1 y los dos paneles de la Figura 1.
    """
    w = np.zeros(M)                     # el estimador arranca en cero
    e_pre = np.zeros(T)                 # error CAUSAL: se mide ANTES de actualizar w
    msd   = np.full(T, np.nan)          # desviacion al optimo; solo se llena post-drift

    Ebar = base = 1.0                   # aq-JAPA: EMA del error y su linea base pre-drift

    I = np.eye(K)
    # "err" guarda el error causal paso a paso: lo necesita la función figura_transitorio
    # para dibujar la curva de MSE rodante de la Figura 1 (panel b). 
    # Que guarda cada clave, y quien la consume:
    #   t      instante RELATIVO al drift (t - T_d): negativo antes, 0 en el
    #          drift, positivo despues. exp1 recorta la ventana con esto.
    #   wM     la coordenada M de w  (la de energia maxima)     -> exp1
    #   uM     su error, w_M - w*_M                             -> exp1
    #   prod   u_M * w_M, la condicion (i) de la Proposicion 1  -> exp1
    #   q      el q_t de ese paso            -> exp1 y el panel (a) de la Figura 1
    #   gamma  el gamma_t de ese paso        -> exp1 (de ahi sale delta_M)
    #   jackM  la fuga de Jackson sobre la coordenada M: gamma * [D_t]_MM * w_M
    #   err    el error causal de ese paso   -> el panel (b) de la Figura 1
    #   lmaxD  lambda_max(D_t): para verificar la cota del Teorema 1. Hoy no lo
    #          consume ningun experimento; se guarda para la tesis.
    #   norma  ||w_t||: para detectar divergencia. Tampoco se consume aqui.
    hist = {k: [] for k in ("t", "wM", "uM", "prod", "q", "gamma",
                            "jackM", "lmaxD", "norma", "err")}
    # divergio: bandera que se enciende si hay que cortar la ejecucion por
    # inestabilidad numerica.
    # t se inicializa aqui SOLO por si el bucle de abajo no llegara a ejecutarse
    # (pasaria si T <= K-1). Una vez que corre, t conserva el valor de la ULTIMA
    # iteracion -- en Python la variable del for sobrevive al for -- y por eso
    # mas abajo sirve para saber EN QUE PASO se corto la ejecucion.
    divergio, t = False, K - 1

    # =========================================================================
    #  BUCLE PRINCIPAL. Arranca en t = K-1 porque antes no hay K muestras para
    #  formar el bloque. Cada vuelta: se arma el bloque, se mide el error
    #  causal, se calcula el paso segun el algoritmo, y se actualiza w.
    # =========================================================================
    for t in range(K - 1, T):
        # --- BLOQUE DE PROYECCION AFIN ---------------------------------------
        # Xn = [x_t, x_{t-1}, ..., x_{t-K+1}], de M x K: toma los K
        # regresores mas recientes.

        Xn = np.column_stack([X[t - j] for j in range(K)])
        dn = np.array([d[t - j] for j in range(K)])
        e  = dn - Xn.T @ w              # error de bloque, vector de K

        # --- AQUI SE MIDE EL MSE (error CAUSAL) ------------------------------
        # ESTA LINEA VA ANTES DE ACTUALIZAR w: 
        e_pre[t] = d[t] - X[t] @ w

        # --- PREACONDICIONADOR Y ENERGIAS ------------------------------------
        # G_t invierte la correlacion temporal DENTRO del bloque: sumar EPS garantiza invertibilidad
        G  = np.linalg.inv(Xn.T @ Xn + EPS * I)
        # D_t = diag(X_t X_t^T). La entrada m es la suma de x_{t-j,m}^2 sobre el
        # bloque, o sea la ENERGIA acumulada a lo largo del bloque.
        Dg = np.sum(Xn * Xn, axis=1)

        # --- QUE q Y QUE gamma USA CADA ALGORITMO ----------------------------
        q, gamma = 1.0, 0.0                       # por defecto: APA puro, sin fuga
        if algoritmo == "qfix":
            # METODO q-JAPA CON q FIJO.
            q = Q_FIXED; gamma = 0.5 * MU * (Q_FIXED - 1.0)
        elif algoritmo == "aq":
            # METODO aq-JAPA:
            #   1) EMA del error de bloque
            Ebar = BETA * Ebar + (1 - BETA) * np.mean(e ** 2)
            #   2) justo antes del drift se toma el valor del baselina para comparacion
            if t == Td - 1: base = Ebar
            #   3) psi_t:
            psi = 0.0 if t < Td else \
                min(max(Ebar / (base + 1e-12) - 1.0, 0.0), 1.0) * np.exp(-(t - Td) / L)
            q = 1.0 + ALPHA * psi; gamma = 0.5 * MU * (q - 1.0)

        # --- HISTORIAL (solo si se pide): de aqui sale la condicion de signo --
        # u_M es el ERROR de la coordenada de energia maxima. El producto
        # u_M * w_M es la condicion (i) de la Proposicion 1: positivo = la fuga
        # empuja en la direccion correcta, negativo = empuja en contra. Por eso
        # se guarda paso a paso.
        if historial:
            uM = w[M - 1] - w2[M - 1]
            hist["t"].append(t - Td); hist["wM"].append(w[M - 1]); hist["uM"].append(uM)
            hist["prod"].append(uM * w[M - 1]); hist["q"].append(q)
            hist["gamma"].append(gamma)
            hist["jackM"].append(gamma * Dg[M - 1] * w[M - 1])
            hist["lmaxD"].append(Dg.max())
            hist["norma"].append(np.linalg.norm(w))
            hist["err"].append(e_pre[t])

        # --- LA ACTUALIZACION ------------------------------------------------
        # 'paso' es el termino de proyeccion afin. Con preconditioning=False se
        # omite G_t (ablacion; el paper corre siempre con G_t).
        paso = Xn @ (G @ e) if preconditioning else Xn @ e
        # Los tres algoritmos comparten esta linea. El segundo termino,
        # gamma * (Dg * w), es la fuga de Jackson
        w = w + MU * paso - gamma * (Dg * w)

        # --- AQUI SE MIDE EL MSD (despues de actualizar) ---------------------
        # MSD = ||w - w*||^2. 
        # Se mide DESPUES de actualizar porque lo que interesa es
        # que tan cerca quedo el estimador del optimo posterior al drift.
        if t >= Td: msd[t] = np.sum((w - w2) ** 2)

        # --- CORTE POR DIVERGENCIA -------------------------------------------
        # Red de seguridad: si ||w|| se dispara, se corta la ejecucion en vez de
        # llenar todo de NaN. Con los parametros del paper no se activa nunca;
        # esta por si se cambia mu o se corre sin preacondicionador.
        nrm = np.linalg.norm(w)
        if not np.isfinite(nrm) or nrm > 1e8:
            divergio = True
            break

    # =========================================================================
    #  METRICAS FINALES
    #  De las T = 1200 iteraciones solo se reportan las H = 100 POSTERIORES AL
    #  DRIFT: la ventana [T_d, T_d+H) = [600, 700). Es el tramo en que el filtro
    #  esta reaccionando al cambio de w*, que es de lo que habla el paper. Lo
    #  anterior al drift y lo posterior a la ventana no entran en ningun numero
    #  de la tabla.
    # =========================================================================
    sl = slice(Td, Td + H)          # las H muestras posteriores al drift

    # MSE causal: promedio de e_pre^2 sobre esa ventana. Es el numero que
    # aparece en los renglones "causal MSE" de la Tabla 1.
    # El caso np.inf: si la ejecucion se corto por divergencia ANTES de
    # completar la ventana, el tramo no ejecutado quedo con los ceros de la
    # inicializacion; promediarlos daria un MSE artificialmente BAJO, y una
    # ejecucion que exploto se veria como la mejor de todas. Devolver infinito
    # hace que arruine el promedio de forma visible, en vez de mentir.
    mse = np.inf if (divergio and t < Td + H) else np.mean(e_pre[sl] ** 2)

    # MSD: mismo promedio, misma ventana. Se usa nanmean porque msd quedo en
    # NaN antes del drift. El if de afuera cubre el caso extremo de que la
    # ventana entera sea NaN (la ejecucion murio antes de t = T_d): ahi nanmean
    # avisaria con un warning y devolveria NaN igual, asi que se devuelve
    # NaN directo.
    msd_post = np.nanmean(msd[sl]) if np.any(np.isfinite(msd[sl])) else np.nan

    # --- QUE SE DEVUELVE -----------------------------------------------------
    # historial=False -> solo los dos numeros de la ventana post-drift.
    # historial=True  -> esos dos numeros MAS h, que NO esta recortado a la
    #                    ventana: h trae las T-(K-1) = 1196 iteraciones
    #                    completas, de t = K-1 hasta t = T-1. h es un
    #                    diccionario con cada lista de hist convertida a arreglo
    #                    de numpy (para poder filtrarlas de golpe), mas dos
    #                    escalares: si la ejecucion divergio y en que iteracion.
    if historial:
        h = {k: np.array(v) for k, v in hist.items()}
        h["divergio"] = divergio
        h["t_divergencia"] = t if divergio else None
        return (mse, msd_post), h
    return (mse, msd_post)


def holm(ps, alpha=0.05):
    """Procedimiento escalonado de Holm (1979).

    PARA QUE SIRVE: cuando se evaluan m pruebas a la vez, mirar cada una al
    nivel alpha hace que la probabilidad de al menos un falso positivo sea
    1-(1-alpha)^m, muy por encima de alpha. Holm la acota a alpha.

    COMO: ordena los p-values de menor a mayor y rechaza mientras p_(k) < alpha/(m-k+1);
    se detiene en el primer fallo. El umbral empieza apretado (alpha/m, para la
    evidencia mas fuerte) y actualiza k hasta llegar a la cota p(k) < alpha.

    QUIEN LO USA: exp4 (sobre m=15, la
    familia completa que reporta el paper).

    Devuelve un vector booleano en el ORDEN ORIGINAL de ps.
    """
    ps = np.asarray(ps, float); m = len(ps)
    idx = np.argsort(ps); rej = np.zeros(m, bool)
    # Se recorren los p de menor a mayor. En cuanto uno falla su umbral, ese y
    # todos los que siguen quedan sin rechazar: eso es el "step-down".
    for k, i in enumerate(idx):
        if ps[i] < alpha / (m - k): rej[i] = True
        else: break
    return rej


def figura_transitorio(c=0.10, seed=3, ventana=60, destino="../resultados"):
    """AQUI SE GENERA LA FIGURA 1 del paper, sus dos paneles.

    PANEL (a)  la trayectoria de q_t: vale 1 exactamente hasta el drift (o sea,
               aq-JAPA ES APA), salta al activarse el disparador, y decae sola.
    PANEL (b)  el MSE de prediccion rodante en dB de APA y de aq-JAPA. Despues
               del drift la curva de aq-JAPA baja antes: esa es la recuperacion
               mas rapida que reporta el paper.

    Los valores por defecto (c = 0.10, semilla 3, ventana de 60) son los de la
    figura publicada. NO es un promedio Monte Carlo: es UNA realizacion, elegida
    para que se vea el transitorio sin que el promedio lo suavice.
    """
    import os
    # matplotlib se importa AQUI DENTRO y no arriba: asi los experimentos que no
    # dibujan (exp1, exp4) no cargan la libreria sin necesidad.
    import matplotlib; matplotlib.use("Agg")   # "Agg": guarda PNG sin abrir ventana
    import matplotlib.pyplot as plt

    os.makedirs(destino, exist_ok=True)

    X, d, w1, w2, Td = data_generation(seed, c, "shrink")
    # historial=True da acceso al registro paso a paso: de ahi salen la serie
    # de error causal (panel b) y la de q_t (panel a). El "_" recoge la tupla
    # (MSE, MSD), que esta figura no necesita.
    _, h_aq  = algorithm_execution(X, d, w2, Td, "aq",  historial=True)
    _, h_apa = algorithm_execution(X, d, w2, Td, "apa", historial=True)

    # El historial arranca en t = K-1, no en 0. Se rellena con NaN al principio
    # para que el eje horizontal sea el instante absoluto y el drift caiga en Td.
    def _completa(v):
        out = np.full(T, np.nan)
        out[K - 1:K - 1 + len(v)] = v
        return out

    q_serie = _completa(h_aq["q"])

    # --- MSE RODANTE EN dB ----------------------------------------------------
    # El error crudo es muy ruidoso y no deja apreciar el transitorio. Promediando
    # sobre una ventana movil y pasando a dB, la caida post-drift se vuelve
    # legible. El +1e-12 evita log(0) si un tramo sale exactamente nulo.
    def _rodante_db(err):
        e = np.nan_to_num(_completa(err))
        out = np.full(T, np.nan)
        # la ventana necesita 'ventana' muestras para llenarse: los primeros
        # valores quedan en NaN y matplotlib simplemente no los dibuja
        for t in range(ventana - 1, T):
            out[t] = 10.0 * np.log10(np.mean(e[t - ventana + 1:t + 1] ** 2) + 1e-12)
        return out

    db_aq, db_apa = _rodante_db(h_aq["err"]), _rodante_db(h_apa["err"])

    # --- PANEL (a): la trayectoria de q_t ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(q_serie)
    # la linea vertical marca el drift: a su izquierda q_t vale exactamente 1
    ax.axvline(Td, linestyle="--", label="abrupt drift")
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"Adaptive $q_t$")
    ax.set_title(r"Adaptive $q_t$ trajectory")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{destino}/Adaptive q_t trayectory.png", dpi=150); plt.close()

    # --- PANEL (b): el MSE rodante de los dos metodos ------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(db_apa, label="APA"); ax.plot(db_aq, label="aq-JAPA")
    ax.axvline(Td, linestyle="--", label="abrupt drift")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Rolling prediction MSE (dB)")
    ax.set_title("APA vs aq-JAPA after shrinkage drift\n"
                 f"c={c}, alpha={ALPHA}, beta={BETA}, L={L}, normalized X")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{destino}/APA vs aq-JAPA after shrinkage drift.png", dpi=150)
    plt.close()

    # --- comprobacion rapida --------------------------------------------------
    # Si q_t no arranca exactamente en 1 o no vuelve a 1, el disparador esta mal.
    print(f"Figura 1 generada con c = {c}, semilla = {seed}")
    print(f"  q_t antes del drift  : {np.nanmin(q_serie[:Td]):.4f} a {np.nanmax(q_serie[:Td]):.4f}")
    print(f"  q_t maximo post-drift: {np.nanmax(q_serie[Td:]):.4f}")
    print(f"  q_t al final         : {q_serie[-1]:.4f}")
    print(f"-> {destino}/Adaptive q_t trayectory.png")
    print(f"-> {destino}/APA vs aq-JAPA after shrinkage drift.png")


if __name__ == "__main__":
    # core.py se puede correr directo para generar la Figura 1 del paper.
    # Importado desde un experimento, este bloque no se ejecuta.
    import sys
    c    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    sd   = int(sys.argv[2])   if len(sys.argv) > 2 else 3
    figura_transitorio(c=c, seed=sd)

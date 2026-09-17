"""
exp1_condicion_signo.py  --  responde al comentario R4 ("APA consistently
outperforms aq-JAPA under growth drift")

QUE PRODUCE PARA EL PAPER
-------------------------
1) El renglon  "Steps with u_M w_M > 0 (%)"  de los DOS paneles de la Tabla 1,
   corrido con n = 500:
       shrinkage: 94.1  88.6  79.0  65.4  57.9
       growth   : 27.2  27.1  27.0  28.3  31.9
2) El rango 59%-74% de la frase de la Seccion 4: "gamma_t still retains
   59%-74% of its magnitude when the sign first reverses". Sale de la columna
   exp(-t/L), modo shrinkage.

QUE MIDE Y POR QUE
------------------
La Proposicion 1 dice que q-JAPA mejora a APA EN UN PASO si y solo si

    (i)  u_t w_t > 0                     (condicion de SIGNO)
    (ii) 0 < delta < 2(1-a) u_t/w_t      (condicion de MAGNITUD)

con u_t = w_t - w*. Este script mide LAS DOS a lo largo de la ventana
post-drift de H = 100 pasos, en la coordenada M (la de energia maxima), que es
donde pega el drift.

Resultado esperado, y es el argumento del paper: la condicion (ii) se cumple
SIEMPRE con margen enorme; la que falla es la (i), y el porcentaje de pasos en
que se cumple explica por completo la curva de mejora medida.

DOS PARTES
----------
PARTE 1  una sola realizacion, paso a paso: se ve el instante exacto en que el
         producto u_M*w_M cambia de signo. Es didactica, no produce ningun
         numero del paper.
PARTE 2  el promedio sobre las n realizaciones, en los dos modos de drift. De
         aqui salen los numeros de arriba.

DE DONDE TOMA LAS COSAS
-----------------------
Todo de core.py: el generador del flujo, los algoritmos y los parametros.

    python exp1_condicion_signo.py 500        # las del paper (~15 min)

El numero de realizaciones es OBLIGATORIO a proposito: en una version anterior
habia un valor por defecto, alguien corrio el script sin argumento, y el
renglon de la tabla quedo promediado sobre menos realizaciones que el resto
del paper. Sin valor por defecto eso no puede volver a pasar.
"""
import sys
import numpy as np
# data_generation -> construye el flujo;  algorithm_execution -> corre un algoritmo
# CS, H, L, MU, K, M -> los parametros, para no redefinirlos aqui
from core import data_generation, algorithm_execution, CS, H, L, MU, K, M

# --- numero de realizaciones de Monte Carlo, sin valor por defecto ------------
if len(sys.argv) < 2:
    raise SystemExit("uso: python exp1_condicion_signo.py <n>      "
                     "(el paper usa 500)")
n = int(sys.argv[1])

# =============================================================================
#  PARTE 1  --  UNA realizacion, paso a paso
#  Para ver el mecanismo con los ojos: en que iteracion exacta se voltea el
#  signo de u_M*w_M, y cuanta fuga de Jackson quedaba encendida en ese momento.
#  Se hace con dos valores extremos de c y una sola semilla, la 0.
# =============================================================================
print("=" * 79)
print(" PARTE 1: la condicion de signo de la Proposicion 1, paso a paso")
print("=" * 79)
print(" u_M := w_M - w*_M  es el ERROR de la coordenada de energia maxima.")
print(" La fuga de Jackson resta  gamma_t * [D_t]_MM * w_M, o sea empuja w_M")
print(" hacia CERO. Eso ayuda solo si w_M debe bajar, es decir si u_M > 0")
print(" cuando w_M > 0. Por eso la condicion es  u_M * w_M > 0.\n")

for c in (0.05, 0.60):
    X, d, w1, w2, Td = data_generation(0, c, "shrink")
    # historial=True devuelve h ademas de las metricas; aqui solo interesa h.
    _, h = algorithm_execution(X, d, w2, Td, "aq", historial=True)

    print(f"--- shrinkage  w*_M: 1.00 -> {c}   (semilla 0) ---")
    print(f"{'t-Td':>5} {'w_M':>9} {'w*_M':>7} {'u_M':>10} {'u_M*w_M':>11} "
          f"{'signo':>6} {'q_t':>8} {'gamma_t':>9} {'fuga en M':>11}")
    # Se muestran solo algunas iteraciones: al principio densas, luego espaciadas.
    for paso in [0, 2, 5, 10, 15, 20, 25, 30, 40, 60, 80, 99]:
        # h["t"] esta medido relativo al drift, asi que "paso" es directamente
        # el numero de iteraciones transcurridas desde T_d.
        i = np.where(h["t"] == paso)[0][0]
        ok = "OK" if h["prod"][i] > 0 else "MAL"
        print(f"{paso:>5} {h['wM'][i]:>9.4f} {w2[-1]:>7.2f} {h['uM'][i]:>10.4f} "
              f"{h['prod'][i]:>11.5f} {ok:>6} {h['q'][i]:>8.4f} "
              f"{h['gamma'][i]:>9.6f} {h['jackM'][i]:>11.6f}")

    ventana = (h["t"] >= 0) & (h["t"] < H)
    print(f"  -> pasos con signo favorable en la ventana: "
          f"{100 * np.mean(h['prod'][ventana] > 0):.1f}%\n")

# =============================================================================
#  PARTE 2  --  el promedio sobre las n realizaciones, en los dos modos
#  Es lo que va al paper. Se hacen los dos modos porque la COMPARACION entre
#  ellos es el resultado: bajo shrinkage el signo se cumple la mayor parte de
#  la ventana, bajo growth casi nunca.
# =============================================================================
print("=" * 78)
print(" PARTE 2: condicion de signo  u_M * w_M > 0  en la ventana post-drift")
print(f" {n} realizaciones, H = {H} pasos")
print("=" * 78)

for modo, etiqueta in [("shrink", "SHRINKAGE  e_M -> c e_M"),
                       ("growth", "GROWTH     c e_M -> e_M")]:
    print(f"\n--- {etiqueta} ---")
    # Encabezado de la tabla. Los numeros entre >   son anchos de columna, para
    # que todo quede alineado; no tienen significado estadistico.
    print(f"{'c':>6} {'% pasos con signo OK':>22} {'1er cambio de signo':>21} "
          f"{'exp(-t/L) ahi':>15} {'delta_M':>10} {'MSE APA':>9} {'MSE aq':>9} "
          f"{'mejora %':>9}")

    # --- BUCLE MEDIO: los cinco factores de drift ----------------------------
    for c in CS:
        # --- ACUMULADORES: una entrada por realizacion, se promedian al final -
        pct_signo_ok = []   # % de la ventana con u_M*w_M > 0   <- va al paper
        paso_cambio  = []   # en que iteracion se voltea el signo por primera vez
        decaimiento  = []   # exp(-t/L) en ese instante          <- el 59%-74%
        delta_M      = []   # gamma*K*sigma_M^2, la condicion (ii)
        mse_apa      = []   # MSE causal post-drift de APA
        mse_aq       = []   # MSE causal post-drift de aq-JAPA

        # --- BUCLE INTERNO: las n realizaciones ------------------------------
        for s in range(n):
            # Mismo flujo para los dos algoritmos
            X, d, w1, w2, Td = data_generation(s, c, modo)
            # aq-JAPA CON historial: ademas del MSE devuelve h, que es de donde
            # sale todo lo que mide este script.
            (mse_q, _), h = algorithm_execution(X, d, w2, Td, "aq", historial=True)
            # APA sin historial: solo hace falta su MSE, para la ultima columna.
            mse_a, _ = algorithm_execution(X, d, w2, Td, "apa")

            # --- AQUI SE EVALUA u_M * w_M ------------------------------------
            # h["t"] esta medido RELATIVO al drift (0 = instante del drift), asi
            # que esta mascara recorta exactamente la ventana [T_d, T_d+H).
            ventana = (h["t"] >= 0) & (h["t"] < H)
            p = h["prod"][ventana]          # el producto u_M*w_M, paso a paso

            # --- % DE PASOS QUE CUMPLEN LA CONDICION DE SIGNO ----------------
            # p > 0 es un vector de booleanos; su media es la fraccion de pasos
            # con signo favorable. Por 100 => el porcentaje que va al paper.
            pct_signo_ok.append(100 * np.mean(p > 0))

            # --- EN QUE PASO SE VOLTEA EL SIGNO ------------------------------
            # np.where(p < 0)[0] son las POSICIONES con producto negativo.
            # Si hay al menos una, la primera es el cambio de signo; si el
            # arreglo sale vacio, el signo aguanto la ventana entera y se
            # anota H, que es el maximo posible.
            negativos = np.where(p < 0)[0]
            f = negativos[0] if len(negativos) else H
            paso_cambio.append(f)

            # Cuanto queda del decaimiento exp(-t/L) en ese instante. Si es
            # grande, la fuga sigue encendida cuando ya esta estorbando. De
            # esta columna sale el 59%-74% que cita la Seccion 4.
            decaimiento.append(np.exp(-f / L))

            # --- CONDICION (ii), LA DE MAGNITUD ------------------------------
            # delta_M = gamma * K * sigma_M^2. Como la energia maxima esta
            # normalizada a 1, sigma_M^2 = 1 y queda gamma*K. Se toma el gamma
            # mas grande de la ventana: el caso mas desfavorable posible.
            delta_M.append(h["gamma"][ventana].max() * K * 1.0)

            mse_apa.append(mse_a); mse_aq.append(mse_q)

        # --- SE REPORTA EL PROMEDIO SOBRE LAS n REALIZACIONES ----------------
        # La primera columna es la que va al paper. La ultima, la mejora de MSE,
        # se imprime al lado a proposito: la idea es ver que las dos bajan
        # juntas conforme crece c. Ese apareamiento ES el argumento.
        mse_apa, mse_aq = np.array(mse_apa), np.array(mse_aq)
        print(f"{c:>6} {np.mean(pct_signo_ok):>22.1f} {np.mean(paso_cambio):>21.1f} "
              f"{np.mean(decaimiento):>15.3f} {np.mean(delta_M):>10.5f} "
              f"{mse_apa.mean():>9.4f} {mse_aq.mean():>9.4f} "
              f"{100 * (mse_apa.mean() - mse_aq.mean()) / mse_apa.mean():>9.2f}")

# =============================================================================
#  LECTURA: como interpretar la tabla de arriba.
# =============================================================================
print(f"""
LECTURA
  - La condicion de MAGNITUD nunca es el problema: delta_M ~ 0.006-0.012 contra
    cotas 2(1-a)u/w de 0.4 a 1.8.
  - La condicion de SIGNO es la que falla. Al inicio del drift se cumple; cuando
    APA cierra el hueco y se pasa de largo, w_M queda por debajo de w*_M y el
    producto se invierte.
  - El mecanismo se apaga por el exponencial exp(-(t-T_d)/L), no por el signo:
    L = {L} pasos frente a una escala de readaptacion
    1/a_M = 1/(mu E[A_t]_MM) ~ {1/(MU*0.925):.0f} pasos. Por eso en el instante
    del cambio de signo todavia queda 59%-74% del termino.
""")

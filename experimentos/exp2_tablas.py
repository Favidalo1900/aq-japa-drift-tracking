"""
exp2_tablas.py  --  AQUI SE GENERAN LAS TABLAS del paper.
                    Exporta CSV, PNG y .npz.

QUE RENGLONES DEL PAPER PRODUCE
-------------------------------
Los cinco renglones de MSE y MSD de la Tabla 1, EN LOS DOS MODOS. Con n = 500,
modo shrinkage:
    "APA causal MSE"        0.2774 0.2636 0.2383 0.1969 0.1677
    "aq-JAPA causal MSE"    0.2295 0.2203 0.2050 0.1851 0.1735
    "Fixed-q causal MSE"    0.1764 0.1724 0.1702 0.1893 0.2399
    "MSE reduction vs APA"  17.25 16.45 14.00 6.01 -3.46
    "MSD reduction vs APA"   1.26  1.19  1.04 0.74  0.49
Los tres primeros son la tabla que se imprime; los dos ultimos salen del bloque
"renglones del paper", que se imprime al cerrar cada modo. El modo growth
produce sus propios cinco renglones en la misma ejecucion.

Los intervalos de confianza y el renglon "Holm-rejected (m=15)" NO salen de
aqui: los produce exp4, que es donde vive toda la parte de pruebas de
hipotesis.

COMPROBACION IMPORTANTE
-----------------------
Corrido con n = 20 reproduce DIGITO POR DIGITO la tabla de la version que se
envio y fue aceptada:
    MSE reduction vs APA: 17.24 16.74 15.31 11.03 6.32
    MSD reduction vs APA:  0.71  0.65  0.53  0.34 0.21
Los numeros del articulo publicado son distintos porque se recalcularon con
n = 500, respondiendo al comentario del revisor sobre el tamano de muestra. Lo
que cambia es n, no el codigo.

DE DONDE TOMA LAS COSAS
-----------------------
De core.py: el flujo, los tres algoritmos, la lista CS de factores de drift y
el orden ALGORITMOS. De scipy: la distribucion t, para los intervalos.

QUE ARCHIVOS OBTENEMOS  (en ../resultados/)
------------------------------------
    tablas_<modo>_n<n>.npz   las n realizaciones individuales de cada celda,
                             MSE y MSD. No lo lee ningun script: es el respaldo
                             de la simulacion, para poder recalcular cualquier
                             estadistica sin volver a simular 40 minutos.
    tabla_<modo>_n<n>.csv    la tabla en texto
    tabla_<modo>_n<n>.png    la misma tabla, renderizada

    python exp2_tablas.py 20          # reproduce la tabla ENVIADA, en segundos
    python exp2_tablas.py 500         # la del articulo publicado (~40 min)

El numero de realizaciones es OBLIGATORIO: no hay valor por defecto, para que
nunca se genere un numero del paper sin haber elegido n explicitamente.
"""
import sys, time, os
import numpy as np
from scipy import stats                       # de aqui salen los IC y los p-value
import matplotlib; matplotlib.use("Agg")      # "Agg": guarda PNG sin abrir ventana
import matplotlib.pyplot as plt
from core import data_generation, algorithm_execution, CS, NOMBRE, ALGORITMOS, H

# --- numero de realizaciones de Monte Carlo, sin valor por defecto ------------
# Es EL parametro que cambia las conclusiones: con 20 salen los numeros de la
# version enviada, con 500 los del articulo publicado.
if len(sys.argv) < 2:
    raise SystemExit("uso: python exp2_tablas.py <n>      "
                     "(20 reproduce la tabla enviada, 500 la publicada)")
n = int(sys.argv[1])
os.makedirs("../resultados", exist_ok=True)   # aqui se deja todo lo que se genera

# =============================================================================
#  Se hace el experimento en los DOS modos de drift, uno por vuelta, y cada
#  vuelta deja su tabla y sus tres archivos.
# =============================================================================
for modo, titulo in [("shrink", "Shrinkage  e_M -> c e_M  (panel a)"),
                     ("growth", "Growth     c e_M -> e_M  (panel b)")]:
    t0 = time.time()

    # --- ALMACENAMIENTO ------------------------------------------------------
    # Un vector de n entradas por cada combinacion (factor de drift, algoritmo).
    # R guarda el MSE causal y S el MSD. Se guardan las ejecuciones INDIVIDUALES,
    # no los promedios, porque las pruebas t necesitan las diferencias entre algoritmos.
    R = {f"{c}_{k}": np.zeros(n) for c in CS for k in ALGORITMOS}   # MSE causal
    S = {f"{c}_{k}": np.zeros(n) for c in CS for k in ALGORITMOS}   # MSD post-update

    # --- AQUI SE SIMULA TODO -------------------------------------------------
    for c in CS:                     # cada factor de drift
        for s in range(n):           # cada semilla
            # UN flujo por semilla, y los tres algoritmos corren sobre EL
            # MISMO. Eso es lo que hace pareadas las comparaciones de abajo.
            X, d, w1, w2, Td = data_generation(s, c, modo)
            for k in ALGORITMOS:
                mse_k, msd_k = algorithm_execution(X, d, w2, Td, k)
                R[f"{c}_{k}"][s] = mse_k     # aqui se guarda el MSE
                S[f"{c}_{k}"][s] = msd_k     # aqui se guarda el MSD
        print(f"  {modo} c={c} listo ({time.time()-t0:.0f}s)", flush=True)

    # El .npz conserva las n realizaciones de cada celda: con el se puede rehacer
    # cualquier estadistica sin volver a simular (que son ~25 min por modo).
    np.savez(f"../resultados/tablas_{modo}_n{n}.npz", **R,
             **{f"MSD_{k}": v for k, v in S.items()})

    # =========================================================================
    #  AQUI SE ARMA LA TABLA. Un renglon por factor de drift.
    # =========================================================================
    filas = []
    for c in CS:
        # En el orden de ALGORITMOS = ["apa","aq","qfix"]:
        #   a = APA,  q = aq-JAPA,  f = q fijo
        a, q, f = (R[f"{c}_{k}"] for k in ALGORITMOS)

        # --- COMPARACION DESTACADA: aq-JAPA frente a q fijo ------------------
        # D_i es la diferencia en la realizacion i entre los dos algoritmos,
        # sobre el MISMO flujo de datos. Positiva = aq-JAPA quedo mejor.
        D = f - q
        # Error estandar de la media de D, e intervalo t al 95% con n-1 grados
        # de libertad. Si el intervalo no contiene al cero, la diferencia es
        # detectable a ese nivel.
        # Aqui NO se calcula ningun p-value ni se aplica ninguna correccion por
        # comparaciones multiples: toda la parte de pruebas de hipotesis vive en
        # exp4, que es el que reporta el renglon "Holm-rejected (m=15)".
        se = stats.sem(D); lo, hi = stats.t.interval(0.95, n-1, loc=D.mean(), scale=se)

        # El renglon: los tres MSE promedio y la reduccion porcentual con su IC.
        filas.append([f"{c:.2f}", f"{a.mean():.4f}",
                      f"{q.mean():.4f}", f"{f.mean():.4f}",
                      f"{100*D.mean()/f.mean():.2f} [{100*lo/f.mean():.1f},"
                      f" {100*hi/f.mean():.1f}]"])

    # =========================================================================
    #  AQUI SE ESCRIBEN LOS ARCHIVOS
    # =========================================================================
    enc = ["c", "APA", "aq-JAPA", "q fijo",
           "aq vs q fijo: reduccion % [IC95]"]
    base = f"../resultados/tabla_{modo}_n{n}"

    # CSV: para abrirlo en Excel o pegarlo en otro lado
    with open(base + ".csv", "w") as fh:
        fh.write(",".join(enc) + "\n")
        for r in filas: fh.write(",".join(f'"{x}"' for x in r) + "\n")

    # PNG: la misma tabla renderizada, para pegarla en una presentacion.
    # Se dibuja como una tabla de matplotlib sobre ejes sin marco.
    fig, ax = plt.subplots(figsize=(11, 2.2)); ax.axis("off")
    tb = ax.table(cellText=filas, colLabels=enc, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for j in range(len(enc)):    # encabezado en negritas y con fondo
        tb[0, j].set_facecolor("#dfe6ee"); tb[0, j].set_text_props(weight="bold")
    ax.set_title(f"{titulo}   MSE causal post-drift, {H} iteraciones, "
                 f"{n} realizaciones", pad=14)
    plt.tight_layout(); plt.savefig(base + ".png", dpi=160); plt.close()

    # --- y la misma tabla a consola -----------------------------------------
    print("\n" + " | ".join(enc))
    for r in filas: print(" | ".join(r))

    # =========================================================================
    #  LOS DOS RENGLONES DEL PAPER QUE SE CALCULAN CONTRA APA
    #  La tabla de arriba compara aq-JAPA contra q fijo. El panel (a) del paper
    #  reporta ademas la reduccion contra APA, en MSE y en MSD, y esos dos
    #  renglones se arman aqui a partir de los mismos vectores ya simulados.
    # =========================================================================
    red_mse = [100*(R[f"{c}_apa"].mean()-R[f"{c}_aq"].mean())/R[f"{c}_apa"].mean()
               for c in CS]
    red_msd = [100*(S[f"{c}_apa"].mean()-S[f"{c}_aq"].mean())/S[f"{c}_apa"].mean()
               for c in CS]
    print("\n  renglones del paper, aq-JAPA contra APA:")
    print("  c                        " + " ".join(f"{c:>8.2f}" for c in CS))
    print("  MSE reduction vs APA (%) " + " ".join(f"{v:>8.2f}" for v in red_mse))
    print("  MSD reduction vs APA (%) " + " ".join(f"{v:>8.2f}" for v in red_msd))
    print(f"\n-> {base}.csv / .png\n")

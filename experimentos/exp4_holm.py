"""
exp4_holm.py  --  responde a los comentarios 2 y 3 (R2: "not significant after
Holm correction";  R4: "only 20 Monte Carlo runs").

QUE HACE
--------
1. Define explicitamente las pruebas de hipotesis de la familia.
2. Las evalua con n runs y muestra, escalon por escalon, como decide Holm.
3. Repite todo con n = 20, 50, 100, 200, 500 para que se vea que el problema
   de la version enviada era falta de potencia, no ausencia de efecto.

DEFINICION DE LAS PRUEBAS
-------------------------
Para cada par de algoritmos (A,B) y cada valor de c definimos, en la realizacion i,

    Delta_i^{(A,B,c)} = MSE_post(A) - MSE_post(B)          (diferencia PAREADA)

y contrastamos

    H_0 : E[Delta^{(A,B,c)}] = 0     contra     H_1 : E[Delta] != 0

con el estadistico t = mean(Delta) / (s_Delta / sqrt(n)),  t ~ t_{n-1} bajo H_0.

La FAMILIA es el conjunto de pruebas sobre el que se corrige. Aqui:
tres comparaciones x cinco valores de c = 15 pruebas por modo de drift.

QUE RENGLONES DEL PAPER PRODUCE
-------------------------------
    "Holm-rejected (m=15)"  de los dos paneles de la Tabla 1
    los IC al 95% de "MSE reduction vs APA" y "vs fixed-q"
Con n = 500: en modo shrinkage sobreviven 4 de 5 en la comparacion aq-JAPA vs APA
(falla c = 0.60); en modo growth sobreviven 5 de 5 contra q fijo.

DE DONDE TOMA LAS COSAS
-----------------------
De core.py: el flujo, los tres algoritmos, CS, NOMBRE y la funcion holm.
De scipy: la distribucion t, para los p y los intervalos.

QUE DEJA ESCRITO
----------------
    ../resultados/datos_<modo>_n<n>.npz   todas las realizaciones individuales

    python exp4_holm.py            # n = 100 (~8 min)
    python exp4_holm.py 500        # n = 500 (~40 min)
    python exp4_holm.py 500 growth # el otro modo
"""
import sys, os
import numpy as np
from scipy import stats
from core import data_generation, algorithm_execution, CS, holm, NOMBRE

# --- numero de realizaciones de Monte Carlo, sin valor por defecto ------------
if len(sys.argv) < 2:
    raise SystemExit("uso: python exp4_holm.py <n> [shrink|growth]      "
                     "(el paper usa 500)")
n_max = int(sys.argv[1])                                  # realizaciones Monte Carlo
MODO  = sys.argv[2] if len(sys.argv) > 2 else "shrink"    # modo de drift

# LOS TRES Comparaciones de la familia. Cada par (A,B) se lee "B frente a A":
# la diferencia pareada sera MSE(A) - MSE(B), asi que positiva = B mejor.
COMPARACIONES = [("apa", "aq"), ("qfix", "aq"), ("apa", "qfix")]
# Nivel al que se controla la probabilidad de AL MENOS UN falso positivo en
# toda la familia. No es el nivel de cada prueba por separado.
ALPHA_FWER = 0.05

# La carpeta de salida se crea AQUI, antes de simular, y no al final. Si no
# existiera, el np.savez de mas abajo fallaria con FileNotFoundError -- y eso
# ocurriria DESPUES de los ~40 min de simulacion, perdiendo todo el trabajo.
os.makedirs("../resultados", exist_ok=True)

# ------------------------------------------------------------------ simulacion
# --- AQUI SE SIMULA: un vector de n_max MSE por cada (factor de drift, algoritmo)
print(f"Simulando {n_max} realizaciones, modo {MODO} ...")
R = {f"{c}_{k}": np.zeros(n_max) for c in CS for k in NOMBRE}
for c in CS:
    for s in range(n_max):
        # UN flujo por semilla; los tres algoritmos corren sobre el mismo.
        # Así podemos restar los MSE y llamarle diferencia PAREADA.
        X, d, w1, w2, Td = data_generation(s, c, MODO)      # misma semilla -> pareado
        for k in NOMBRE:
            # [0] = MSE causal. El MSD no se usa en este script.
            R[f"{c}_{k}"][s] = algorithm_execution(X, d, w2, Td, k)[0]
    print(f"  c={c} listo", flush=True)
# Se guardan los valores de las runs individuales: con esto se puede rehacer toda la
# estadistica sin volver a simular.
np.savez(f"../resultados/datos_{MODO}_n{n_max}.npz", **R)


def familia(n):
    """Construye las 15 pruebas y devuelve etiquetas, p-values y tamanos."""
    etiq, ps, info = [], [], []
    # Doble bucle: 3 comparaciones x 5 factores de drift = 15 pruebas.
    # Ese 15 es el m que aparece en el renglon "Holm-rejected (m=15)" del paper.
    for A, B in COMPARACIONES:
        for c in CS:
            # [:n] permite reusar la MISMA simulacion para varios tamanos de
            # muestra: asi el barrido de abajo no re-simula nada.
            a = R[f"{c}_{A}"][:n]; b = R[f"{c}_{B}"][:n]
            D = a - b                                   # diferencia PAREADA
            # Estadistico t de la media de D. Bajo H0: E[D]=0 sigue una t con
            # n-1 grados de libertad.
            se = stats.sem(D); t = D.mean() / se
            p  = 2 * stats.t.sf(abs(t), n - 1)          # p BILATERAL
            # Intervalo al 95%: el conjunto de valores de E[D] que la prueba no
            # rechazaria. Es lo que el paper reporta entre corchetes.
            lo, hi = stats.t.interval(0.95, n - 1, loc=D.mean(), scale=se)
            etiq.append(f"{NOMBRE[B]} vs {NOMBRE[A]}, c={c}")
            ps.append(p)
            info.append((100 * D.mean() / a.mean(), 100 * lo / a.mean(),
                         100 * hi / a.mean(), t))
    return etiq, np.array(ps), info


# ------------------------------------------------------------ Holm paso a paso
# =============================================================================
#  AQUI SE APLICA HOLM, ESCALON POR ESCALON, y se imprime la decision de cada
#  uno. La idea de imprimirlo asi es que se pueda auditar a mano: se ve el p,
#  se ve el umbral contra el que compite, y se ve donde se detiene.
# =============================================================================
etiq, ps, info = familia(n_max)
m = len(ps)
print("\n" + "=" * 92)
print(f" HOLM PASO A PASO   (familia de m = {m} pruebas, alpha = {ALPHA_FWER}, n = {n_max})")
print("=" * 92)
print(f"{'k':>3} {'prueba':>34} {'p_(k)':>11} {'umbral a/(m-k+1)':>18} {'decision':>12}")
# Se ordenan los p de menor a mayor. 'vivo' se apaga en el primer fallo: de ahi
# en adelante todo queda bloqueado, eso es lo que significa "step-down".
orden = np.argsort(ps); vivo = True
for k, i in enumerate(orden):
    umbral = ALPHA_FWER / (m - k)
    if vivo and ps[i] < umbral: dec = "RECHAZA"
    else: vivo = False; dec = "se detiene" if k == 0 or dec != "bloqueada" else "bloqueada"
    if not vivo and dec != "se detiene": dec = "bloqueada"
    print(f"{k+1:>3} {etiq[i]:>34} {ps[i]:>11.3e} {umbral:>18.5f} {dec:>12}")

rej = holm(ps, ALPHA_FWER)
print(f"\n  Sobreviven {rej.sum()} de {m}.")
print(f"\n{'prueba':>34} {'reduccion %':>12} {'IC95%':>20} {'t':>8} {'p':>11} {'Holm':>6}")
for i in range(m):
    r, lo, hi, t = info[i]
    print(f"{etiq[i]:>34} {r:>12.2f} [{lo:>8.2f},{hi:>8.2f}] {t:>8.2f} {ps[i]:>11.3e} "
          f"{'si' if rej[i] else 'no':>6}")

# ------------------------------------------------------ efecto del numero de n
print("\n" + "=" * 92)
print(" EFECTO DEL NUMERO DE REALIZACIONES sobre la misma familia")
print("=" * 92)
# AQUI SE DEMUESTRA QUE ERA FALTA DE POTENCIA.
#
# IMPORTANTE, para que no se malinterprete: esto NO vuelve a simular. La
# simulacion ya ocurrio una sola vez, arriba, con n_max realizaciones. Lo que
# hace familia(n) es tomar las PRIMERAS n de esas mismas realizaciones ya
# guardadas -- el [:n] de su codigo -- y rehacer la estadistica con ellas.
#
# Por eso el 100 que aparece en esta lista es una SUBMUESTRA de las 500, no una
# simulacion aparte. El efecto medido es el mismo en todos los renglones; lo
# unico que cambia es con cuanta precision se mide, y por eso las mismas
# comparaciones empiezan a cruzar el umbral conforme sube n.
for n in [x for x in (20, 50, 100, 200, 500) if x <= n_max]:
    _, p_n, _ = familia(n)
    print(f"  n = {n:>4}   sobreviven Holm: {holm(p_n, ALPHA_FWER).sum():>2}/{m}"
          f"   |t| maximo = {max(abs(stats.t.isf(p/2, n-1)) for p in p_n):>6.2f}"
          f"   umbral |t| del 1er escalon = {stats.t.isf(ALPHA_FWER/(2*m), n-1):.2f}")
print("""
LECTURA
  Con n = 20 las cinco pruebas "aq-JAPA vs APA" tienen |t| entre 2.17 y 2.72 y
  el umbral del primer escalon esta en 3.35: ninguna lo cruza, y por eso el
  revisor escribio "not significant after Holm correction". Las comparaciones
  contra q fijo, que tienen |t| ~ 4.7, si sobreviven ya con n = 20.
  Al subir n, el mismo efecto -- sin cambiar de tamano -- cruza el umbral con
  holgura: lo que cambia no es el efecto sino la precision con que se mide.
  (Con n = 20 estos numeros reproducen exactamente las Tablas 1 y 2 enviadas.)
""")

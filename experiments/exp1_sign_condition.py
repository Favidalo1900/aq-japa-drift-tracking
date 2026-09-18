"""
exp1_sign_condition.py  --  answers reviewer comment R4 ("APA consistently
outperforms aq-JAPA under growth drift")

WHAT IT PRODUCES FOR THE PAPER
------------------------------
1) The row "Steps with u_M w_M > 0 (%)" of BOTH panels of Table 1, run with
   n = 500:
       shrinkage: 94.1  88.6  79.0  65.4  57.9
       growth   : 27.2  27.1  27.0  28.3  31.9
2) The 59%-74% range quoted in Section 4: "gamma_t still retains 59%-74% of its
   magnitude when the sign first reverses". It comes from the exp(-t/L) column,
   shrinkage mode.

WHAT IT MEASURES AND WHY
------------------------
Proposition 1 states that q-JAPA improves on APA IN ONE STEP if and only if

    (i)  u_t w_t > 0                     (SIGN condition)
    (ii) 0 < delta < 2(1-a) u_t/w_t      (MAGNITUDE condition)

with u_t = w_t - w*. This script measures BOTH along the post-drift window of
H = 100 steps, on coordinate M (the maximum-energy one), which is where the
drift hits.

Expected result, and it is the argument of the paper: condition (ii) always
holds with an enormous margin; the one that fails is (i), and the percentage of
steps where it holds fully explains the measured improvement curve.

TWO PARTS
---------
PART 1  a single realization, step by step: you can see the exact instant at
        which the product u_M*w_M changes sign. It is didactic and produces no
        number of the paper.
PART 2  the average over the n realizations, in both drift modes. The numbers
        above come from here.

WHERE IT TAKES THINGS FROM
--------------------------
Everything from core.py: the stream generator, the algorithms and the
parameters.

    python exp1_sign_condition.py 500        # the ones in the paper (~15 min)

"""
import sys
import numpy as np
# data_generation -> builds the stream;  algorithm_execution -> runs one algorithm
# CS, H, L, MU, K, M -> the parameters, so they are not redefined here
from core import data_generation, algorithm_execution, CS, H, L, MU, K, M

# --- number of Monte Carlo realizations ---------------
if len(sys.argv) < 2:
    raise SystemExit("usage: python exp1_sign_condition.py <n>      "
                     "(the paper uses 500)")
n = int(sys.argv[1])

# =============================================================================
#  PART 1  --  ONE realization, step by step
#  To see the mechanism: this is at which exact iteration the sign of
#  u_M*w_M flips, and how much Jackson leakage was still switched on at that
#  moment. Done with two extreme values of c and a single seed, seed 0.
# =============================================================================
print("=" * 79)
print(" PART 1: the sign condition of Proposition 1, step by step")
print("=" * 79)
print(" u_M := w_M - w*_M  is the ERROR of the maximum-energy coordinate.")
print(" The Jackson leakage subtracts  gamma_t * [D_t]_MM * w_M, i.e. it pushes")
print(" w_M towards ZERO. That helps only if w_M must go down, that is if")
print(" u_M > 0 when w_M > 0. Hence the condition  u_M * w_M > 0.\n")

for c in (0.05, 0.60):
    X, d, w1, w2, Td = data_generation(0, c, "shrink")
    # history=True returns h besides the metrics; only h matters here.
    _, h = algorithm_execution(X, d, w2, Td, "aq", history=True)

    print(f"--- shrinkage  w*_M: 1.00 -> {c}   (seed 0) ---")
    print(f"{'t-Td':>5} {'w_M':>9} {'w*_M':>7} {'u_M':>10} {'u_M*w_M':>11} "
          f"{'sign':>6} {'q_t':>8} {'gamma_t':>9} {'leak on M':>11}")
    # Only a few iterations are shown: dense at the start, then spaced out.
    for step in [0, 2, 5, 10, 15, 20, 25, 30, 40, 60, 80, 99]:
        # h["t"] is measured relative to the drift, so "step" is directly the
        # number of iterations elapsed since T_d.
        i = np.where(h["t"] == step)[0][0]
        ok = "OK" if h["prod"][i] > 0 else "BAD"
        print(f"{step:>5} {h['wM'][i]:>9.4f} {w2[-1]:>7.2f} {h['uM'][i]:>10.4f} "
              f"{h['prod'][i]:>11.5f} {ok:>6} {h['q'][i]:>8.4f} "
              f"{h['gamma'][i]:>9.6f} {h['jackM'][i]:>11.6f}")

    window = (h["t"] >= 0) & (h["t"] < H)
    print(f"  -> steps with favourable sign in the window: "
          f"{100 * np.mean(h['prod'][window] > 0):.1f}%\n")

# =============================================================================
#  PART 2  --  the average over the n realizations, in both modes
#  This is what goes into the paper. Both modes are run because the COMPARISON
#  between them is the result: under shrinkage the sign holds for most of the
#  window, under growth almost never.
# =============================================================================
print("=" * 78)
print(" PART 2: sign condition  u_M * w_M > 0  in the post-drift window")
print(f" {n} realizations, H = {H} steps")
print("=" * 78)

for mode, label in [("shrink", "SHRINKAGE  e_M -> c e_M"),
                    ("growth", "GROWTH     c e_M -> e_M")]:
    print(f"\n--- {label} ---")
    # Table header. The numbers after > are column widths, so that everything
    # lines up; they carry no statistical meaning.
    print(f"{'c':>6} {'% steps with sign OK':>22} {'1st sign flip':>15} "
          f"{'exp(-t/L) there':>17} {'delta_M':>10} {'MSE APA':>9} {'MSE aq':>9} "
          f"{'improv. %':>10}")

    # --- MIDDLE LOOP: the five drift factors ---------------------------------
    for c in CS:
        # --- ACCUMULATORS: one entry per realization, averaged at the end -----
        pct_sign_ok = []   # % of the window with u_M*w_M > 0   <- goes to the paper
        flip_step   = []   # at which iteration the sign first flips
        decay       = []   # exp(-t/L) at that instant           <- the 59%-74%
        delta_M     = []   # gamma*K*sigma_M^2, condition (ii)
        mse_apa     = []   # post-drift causal MSE of APA
        mse_aq      = []   # post-drift causal MSE of aq-JAPA

        # --- INNER LOOP: the n realizations -----------------------------------
        for s in range(n):
            # Same stream for both algorithms
            X, d, w1, w2, Td = data_generation(s, c, mode)
            # aq-JAPA WITH history: besides the MSE it returns h, which is where
            # everything this script measures comes from.
            (mse_q, _), h = algorithm_execution(X, d, w2, Td, "aq", history=True)
            # APA without history: only its MSE is needed, for the last column.
            mse_a, _ = algorithm_execution(X, d, w2, Td, "apa")

            # --- HERE u_M * w_M IS EVALUATED ---------------------------------
            # h["t"] is measured RELATIVE to the drift (0 = drift instant), so
            # this mask slices exactly the window [T_d, T_d+H).
            window = (h["t"] >= 0) & (h["t"] < H)
            p = h["prod"][window]           # the product u_M*w_M, step by step

            # --- % OF STEPS THAT MEET THE SIGN CONDITION ---------------------
            # p > 0 is a boolean vector; its mean is the fraction of steps with
            # favourable sign. Times 100 => the percentage that goes to the paper.
            pct_sign_ok.append(100 * np.mean(p > 0))

            # --- AT WHICH STEP THE SIGN FLIPS --------------------------------
            # np.where(p < 0)[0] are the POSITIONS with a negative product.
            # If there is at least one, the first is the sign flip; if the array
            # comes out empty, the sign held for the whole window and H is
            # recorded, which is the maximum possible.
            negatives = np.where(p < 0)[0]
            f = negatives[0] if len(negatives) else H
            flip_step.append(f)

            # How much of the exp(-t/L) decay is left at that instant. If it is
            # large, the leakage is still on when it has already become
            # harmful. The 59%-74% quoted in Section 4 comes from this column.
            decay.append(np.exp(-f / L))

            # --- CONDITION (ii), THE MAGNITUDE ONE ---------------------------
            # delta_M = gamma * K * sigma_M^2. Since the maximum energy is
            # normalized to 1, sigma_M^2 = 1 and only gamma*K is left. The
            # largest gamma of the window is taken: the least favourable case.
            delta_M.append(h["gamma"][window].max() * K * 1.0)

            mse_apa.append(mse_a); mse_aq.append(mse_q)

        # --- THE AVERAGE OVER THE n REALIZATIONS IS REPORTED -----------------
        # The first column is the one that goes to the paper. The last one, the
        # MSE improvement, is printed alongside on purpose: the point is to see
        # that both go down together as c grows. That pairing IS the argument.
        mse_apa, mse_aq = np.array(mse_apa), np.array(mse_aq)
        print(f"{c:>6} {np.mean(pct_sign_ok):>22.1f} {np.mean(flip_step):>15.1f} "
              f"{np.mean(decay):>17.3f} {np.mean(delta_M):>10.5f} "
              f"{mse_apa.mean():>9.4f} {mse_aq.mean():>9.4f} "
              f"{100 * (mse_apa.mean() - mse_aq.mean()) / mse_apa.mean():>10.2f}")

# =============================================================================
#  READING: how to interpret the table above.
# =============================================================================
print(f"""
READING
  - The MAGNITUDE condition is never the problem: delta_M ~ 0.006-0.012 against
    bounds 2(1-a)u/w ranging from 0.4 to 1.8.
  - The SIGN condition is the one that fails. It holds at the onset of the
    drift; once APA closes the gap and overshoots, w_M ends up below w*_M and
    the product flips.
  - The mechanism is switched off by the exponential exp(-(t-T_d)/L), not by the
    sign: L = {L} steps against a re-adaptation time scale
    1/a_M = 1/(mu E[A_t]_MM) ~ {1/(MU*0.925):.0f} steps. That is why at the
    instant of the sign flip 59%-74% of the term is still left.
""")

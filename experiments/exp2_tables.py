"""
exp2_tables.py  --  HERE THE TABLES of the paper are generated.
                    Exports CSV, PNG and .npz.

WHICH ROWS OF THE PAPER IT PRODUCES
-----------------------------------
The five MSE and MSD rows of Table 1, IN BOTH MODES. With n = 500, shrinkage
mode:
    "APA causal MSE"        0.2774 0.2636 0.2383 0.1969 0.1677
    "aq-JAPA causal MSE"    0.2295 0.2203 0.2050 0.1851 0.1735
    "Fixed-q causal MSE"    0.1764 0.1724 0.1702 0.1893 0.2399
    "MSE reduction vs APA"  17.25 16.45 14.00 6.01 -3.46
    "MSD reduction vs APA"   1.26  1.19  1.04 0.74  0.49
    
The first three are the printed table; the last two come from the "paper rows"
block, printed when each mode is closed. The growth mode produces its own five
rows in the same run.

The confidence intervals and the "Holm-rejected (m=15)" row do NOT come from
here: they are produced by exp4, which is where the whole hypothesis-testing
part lives.

IMPORTANT CHECK
---------------
Run with n = 20 it reproduces DIGIT BY DIGIT the table of the version that was
submitted and accepted:
    MSE reduction vs APA: 17.24 16.74 15.31 11.03 6.32
    MSD reduction vs APA:  0.71  0.65  0.53  0.34 0.21
The numbers in the published article are different because they were recomputed
with n = 500, answering the reviewer's comment about sample size. What changes
is n, not the code.

WHERE IT TAKES THINGS FROM
--------------------------
From core.py: the stream, the three algorithms, the CS list of drift factors
and the ALGORITHMS order. From scipy: the t distribution, for the intervals.

WHICH FILES IT WRITES (into ../results/)
----------------------------------------
    tables_<mode>_n<n>.npz   the n individual realizations of each cell, MSE and
                             MSD. No script reads it: it is the backup of the
                             simulation, so any statistic can be recomputed
                             without simulating 40 minutes again.
    table_<mode>_n<n>.csv    the table as text
    table_<mode>_n<n>.png    the same table, rendered

    python exp2_tables.py 20          # reproduces the SUBMITTED table, in seconds
    python exp2_tables.py 500         # the published article's (~40 min)

The number of realizations is MANDATORY: there is no default value, so that a
number of the paper can never be generated without choosing n explicitly.
"""
import sys, time, os
import numpy as np
from scipy import stats                       # the intervals come from here
import matplotlib; matplotlib.use("Agg")      # "Agg": writes PNG without opening a window
import matplotlib.pyplot as plt
from core import data_generation, algorithm_execution, CS, NAME, ALGORITHMS, H

# --- number of Monte Carlo realizations, with no default value ---------------
# It is THE parameter that changes the conclusions: with 20 you get the numbers
# of the submitted version, with 500 those of the published article.
if len(sys.argv) < 2:
    raise SystemExit("usage: python exp2_tables.py <n>      "
                     "(20 reproduces the submitted table, 500 the published one)")
n = int(sys.argv[1])
os.makedirs("../results", exist_ok=True)   # everything generated is left here

# =============================================================================
#  The experiment is run in BOTH drift modes, one per pass, and each pass leaves
#  its table and its three files.
# =============================================================================
for mode, title in [("shrink", "Shrinkage  e_M -> c e_M  (panel a)"),
                    ("growth", "Growth     c e_M -> e_M  (panel b)")]:
    t0 = time.time()

    # --- STORAGE -------------------------------------------------------------
    # One vector of n entries per (drift factor, algorithm) combination.
    # R holds the causal MSE and S the MSD. The INDIVIDUAL runs are stored, not
    # the averages, because the t tests need the differences between algorithms.
    R = {f"{c}_{k}": np.zeros(n) for c in CS for k in ALGORITHMS}   # causal MSE
    S = {f"{c}_{k}": np.zeros(n) for c in CS for k in ALGORITHMS}   # post-update MSD

    # --- HERE EVERYTHING IS SIMULATED ----------------------------------------
    for c in CS:                     # each drift factor
        for s in range(n):           # each seed
            # ONE stream per seed, and the three algorithms run on THE SAME one.
            # That is what makes the comparisons below paired.
            X, d, w1, w2, Td = data_generation(s, c, mode)
            for k in ALGORITHMS:
                mse_k, msd_k = algorithm_execution(X, d, w2, Td, k)
                R[f"{c}_{k}"][s] = mse_k     # the MSE is stored here
                S[f"{c}_{k}"][s] = msd_k     # the MSD is stored here
        print(f"  {mode} c={c} done ({time.time()-t0:.0f}s)", flush=True)

    # The .npz keeps the n realizations of each cell: with it any statistic can
    # be redone without simulating again (which is ~25 min per mode).
    np.savez(f"../results/tables_{mode}_n{n}.npz", **R,
             **{f"MSD_{k}": v for k, v in S.items()})

    # =========================================================================
    #  HERE THE TABLE IS BUILT. One row per drift factor.
    # =========================================================================
    rows = []
    for c in CS:
        # In the order of ALGORITHMS = ["apa","aq","qfix"]:
        #   a = APA,  q = aq-JAPA,  f = fixed-q
        a, q, f = (R[f"{c}_{k}"] for k in ALGORITHMS)

        # --- HIGHLIGHTED COMPARISON: aq-JAPA against fixed-q -----------------
        # D_i is the difference in realization i between the two algorithms, on
        # THE SAME data stream. Positive = aq-JAPA came out better.
        D = f - q
        # Standard error of the mean of D, and the 95% t interval with n-1
        # degrees of freedom. If the interval does not contain zero, the
        # difference is detectable at that level.
        # NO p-value is computed here and NO multiple-comparison correction is
        # applied: the whole hypothesis-testing part lives in exp4, which is the
        # one that reports the "Holm-rejected (m=15)" row.
        se = stats.sem(D); lo, hi = stats.t.interval(0.95, n-1, loc=D.mean(), scale=se)

        # The row: the three mean MSEs and the percentage reduction with its CI.
        rows.append([f"{c:.2f}", f"{a.mean():.4f}",
                     f"{q.mean():.4f}", f"{f.mean():.4f}",
                     f"{100*D.mean()/f.mean():.2f} [{100*lo/f.mean():.1f},"
                     f" {100*hi/f.mean():.1f}]"])

    # =========================================================================
    #  HERE THE FILES ARE WRITTEN
    # =========================================================================
    header = ["c", "APA", "aq-JAPA", "fixed-q",
              "aq vs fixed-q: reduction % [95% CI]"]
    stem = f"../results/table_{mode}_n{n}"

    # CSV: to open it in a spreadsheet or paste it elsewhere
    with open(stem + ".csv", "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows: fh.write(",".join(f'"{x}"' for x in r) + "\n")

    # PNG: the same table rendered, to drop into a presentation.
    # It is drawn as a matplotlib table on frameless axes.
    fig, ax = plt.subplots(figsize=(11, 2.2)); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for j in range(len(header)):    # header in bold and with a background
        tb[0, j].set_facecolor("#dfe6ee"); tb[0, j].set_text_props(weight="bold")
    ax.set_title(f"{title}   post-drift causal MSE, {H} iterations, "
                 f"{n} realizations", pad=14)
    plt.tight_layout(); plt.savefig(stem + ".png", dpi=160); plt.close()

    # --- and the same table to the console -----------------------------------
    print("\n" + " | ".join(header))
    for r in rows: print(" | ".join(r))

    # =========================================================================
    #  THE TWO ROWS OF THE PAPER THAT ARE COMPUTED AGAINST APA
    #  The table above compares aq-JAPA against fixed-q. Panel (a) of the paper
    #  also reports the reduction against APA, in MSE and in MSD, and those two
    #  rows are built here from the same vectors already simulated.
    # =========================================================================
    red_mse = [100*(R[f"{c}_apa"].mean()-R[f"{c}_aq"].mean())/R[f"{c}_apa"].mean()
               for c in CS]
    red_msd = [100*(S[f"{c}_apa"].mean()-S[f"{c}_aq"].mean())/S[f"{c}_apa"].mean()
               for c in CS]
    print("\n  paper rows, aq-JAPA against APA:")
    print("  c                        " + " ".join(f"{c:>8.2f}" for c in CS))
    print("  MSE reduction vs APA (%) " + " ".join(f"{v:>8.2f}" for v in red_mse))
    print("  MSD reduction vs APA (%) " + " ".join(f"{v:>8.2f}" for v in red_msd))
    print(f"\n-> {stem}.csv / .png\n")

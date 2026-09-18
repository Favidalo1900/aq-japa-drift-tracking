# aq-JAPA — Jackson Regularization for Affine-Projection Online Learning Under Concept Drift

Experiment code for the paper *q-JAPA: Jackson Regularization for
Affine-Projection Online Learning Under Concept Drift* (IDEAL 2026).

Everything that appears in the paper — the two tables and Figure 1 — comes out
of `experiments/`, and all of it from the same engine, `core.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

`numpy`, `scipy` and `matplotlib`. Nothing else.

---

## How to run

Every command below assumes you are standing **inside `experiments/`**:

```bash
cd experiments
```

That matters: the scripts write to `../results/`, a path relative to where you
are, not to where the file lives. The `results/` folder is created
automatically; it is git-ignored, so nothing you generate is ever committed.

### The `n` argument

Three of the four scripts take **one mandatory argument, `n`**: the number of
Monte Carlo realizations to average over.

```bash
python exp2_tables.py 500
#                     ^^^
#                     this is n
```

There is no default value. Run a script without the argument and it stops
immediately, before simulating anything:

```
$ python exp2_tables.py
usage: python exp2_tables.py <n>      (20 reproduces the submitted table, 500 the published one)
```

`n` is the parameter that decides which version of the tables you get:

| `n` | what you get |
|---|---|
| `20` | the tables of the version **submitted** to review, digit by digit |
| `500` | the tables of the **published** article |

Same code, same seeds, same everything else — only the sample size changes.
That is the whole answer to the reviewer's comment about 20 runs being too few.
Any `n` works; those two are the ones that correspond to something printed.

`n` is also written into every output filename (`table_shrink_n500.csv`), so a
result file always says how many realizations produced it.

### The four commands

| command | what it does | output | time |
|---|---|---|---|
| `python core.py` | draws **Figure 1**, both panels | two `.png` in `../results/` | ~1 s |
| `python exp1_sign_condition.py 500` | the sign condition of Proposition 1 | console only | ~11 min |
| `python exp2_tables.py 500` | the two tables, **both drift modes in one run** | `.npz`, `.csv`, `.png` per mode | ~16 min |
| `python exp4_holm.py 500` | the family of 15 tests + Holm, shrinkage mode | `data_shrink_n500.npz` | ~8 min |
| `python exp4_holm.py 500 growth` | the same, growth mode | `data_growth_n500.npz` | ~8 min |

Times measured on the machine that produced the paper's numbers. The full set is
about 45 minutes.

`core.py` is the only one that takes no argument: it has no Monte Carlo loop.
Figure 1 is **one** realization ($c=0.10$, seed 3, moving window of 60), chosen
so the transient is visible instead of being smoothed away by an average.

`exp4_holm.py` is the only one that takes a **second**, optional argument: the
drift mode, `shrink` (the default) or `growth`. `exp2_tables.py` does not need
it because it runs both modes in a single pass.

### The check that matters

Before trusting any of the above, run this:

```bash
python exp2_tables.py 20
```

It takes under a minute and must print

```
MSE reduction vs APA:  17.24 16.74 15.31 11.03 6.32
MSD reduction vs APA:   0.71  0.65  0.53  0.34 0.21
```

which is exactly Table 1 of the submitted version. If those ten numbers come
out, the engine is the one the paper was written with, and the difference at
`n = 500` is an effect of sample size, not of the code.

---

## Which script produces which row of the paper

| script | what it produces |
|---|---|
| `core.py` | the engine: generates the AR(1) stream, defines the three algorithms and, run directly, produces both panels of **Figure 1**. Everything else imports from it. |
| `exp1_sign_condition.py` | `Steps with u_M w_M > 0 (%)` in both panels, and the 59 %–74 % range of §4. Its PART 1 also shows, step by step over a single realization, the instant at which the sign flips. |
| `exp2_tables.py` | the causal MSEs of the three methods, `MSE reduction vs. APA` and `MSD reduction vs. APA` |
| `exp4_holm.py` | `Holm-rejected (m=15)` and the 95 % confidence intervals |

Every parameter lives in one block at the top of `core.py`, so the four scripts
can never disagree with one another:

```
T = 1200    M = 10      K = 5       rho = 0.99
mu = 0.05   eps = 1e-3  noise = 0.05
kappa_diag(R_x) = 1e3   H = 100     T_d = 600
alpha = 0.15  beta = 0.97  L = 80   fixed q = 1.15
```

---

## The three algorithms

With $G_t=(X_t^\top X_t+\epsilon I)^{-1}$ and $D_t=\mathrm{diag}(X_tX_t^\top)$:

| key | name | update |
|---|---|---|
| `apa` | APA | $w+\mu X_tG_te_t$ |
| `qfix` | q-JAPA, fixed $q$ | $w+\mu X_tG_te_t-\gamma D_tw$, with $\gamma=\tfrac\mu2(q-1)$, $q=1.15$ |
| `aq` | **aq-JAPA** | $w+\mu X_tG_te_t-\gamma_tD_tw$, with $q_t=1+\alpha\psi_t$ |

All three run on **the same seed**, so every comparison is paired.

---

## About the submitted version

The code that accompanied the version sent to review (`aq_japa_simulation.py`)
is not in this repository: it reproduced the table of that version, with n = 20,
only APA and aq-JAPA and only the shrinkage mode. This repository holds a single
thing to run, and it is the one that corresponds to the published article.

Worth knowing: that script measured `errors[t]` **after** updating `w`, that is,
the **a posteriori** error, whereas the article declares the **causal** one. The
difference is not cosmetic — it gave an improvement of **18.43 %** where the
submitted table reports **17.24 %**, because the a posteriori error has already
seen the response of that instant and is optimistic. `core.py` always used the
causal one, and that is why `python exp2_tables.py 20` reproduces the submitted
table digit by digit.

## Figures

The two files in `figures/` are the ones produced by `python core.py`, at
150 dpi. The article's PDF embeds the same plots at 100 dpi: same realization,
different resolution.

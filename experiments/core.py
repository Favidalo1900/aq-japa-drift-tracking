"""
core.py  --  shared core of the experiments, for the apa, q-JAPA and aq-JAPA
algorithms.

Holds the data-stream generator and ALL the algorithms. The exp1, exp2, exp4 scripts
import everything from here, so no experiment can end up running with different
parameters from another.

NOTATION
    x_t in R^M          input at time t
    X_t in R^{MxK}      block [x_t, ..., x_{t-K+1}]
    d_t in R^K          block of desired outputs
    e_t = d_t - X_t^T w_t
    G_t = (X_t^T X_t + eps I)^{-1}     APA preconditioner
    D_t = diag(X_t X_t^T)              instantaneous per-coordinate energies
    w*                  optimal vector; it changes at t = T_d

THE THREE ALGORITHMS run with the same mu, eps, K, rho and the SAME seed, we say they're paired.
    'apa'    w_{t+1} = w_t + mu X_t G_t e_t
    'qfix'   w_{t+1} = w_t + mu X_t G_t e_t - gamma   D_t w_t   gamma=(mu/2)(q-1), q=1.15
    'aq'     w_{t+1} = w_t + mu X_t G_t e_t - gamma_t D_t w_t   q_t = 1 + alpha psi_t

    With preconditioning=False the G_t factor is dropped: 
    "APA without preconditioner", useful to see what G_t contributes. The paper
    always runs with preconditioning=True.

STRUCTURE
    data_generation(seed, c, mode)      the stream: x_t, y_t and the optimum w*
    algorithm_execution(...)           runs one of the three algorithms
    holm(ps, alpha)                    the multiple-comparison correction
    transient_figure(...)              FIGURE 1 of the paper, both panels

WHAT EACH EXPERIMENT USES
    exp1 -> data_generation, algorithm_execution(history=True)   sign condition
    exp2 -> data_generation, algorithm_execution                 the two tables
    exp4 -> data_generation, algorithm_execution, holm           family of 15 + Holm

    exp2 and exp4 need only the two post-drift metrics, (MSE, MSD), so they
    call the function as it comes. exp1 also needs the step-by-step record, and
    asks for it with history=True:

        mse, msd = algorithm_execution(X, d, w2, Td, "aq")
        # returns a tuple of two numbers, nothing else

        (mse, msd), h = algorithm_execution(X, d, w2, Td, "aq", history=True)
        # returns the same PLUS h, the record of the 1196 iterations

TO GENERATE FIGURE 1
    python core.py
"""
import numpy as np

# =============================================================================
#  PARAMETERS: the experiments import them from here
# =============================================================================
T, M, K   = 1200, 10, 5        # T total steps; M coordinates; K = projection order
RHO       = 0.99               # AR(1) temporal correlation: close to 1 = strongly correlated
MU, EPS   = 0.05, 1e-3         # mu = adaptation step; eps = regularization of the inverse
NOISE     = 0.05               # variance of the observation noise nu_t
KAPPA     = 1e3                # kappa_diag(R_x): ratio of maximum to minimum energy
ALPHA, BETA, L = 0.15, 0.97, 80   # aq-JAPA: psi scale, EMA factor, decay horizon
Q_FIXED   = 1.15                  # q of the fixed baseline = 1 + ALPHA (the maximum aq-JAPA reaches)
H         = 100                   # post-drift window over which MSE and MSD are averaged
CS        = [0.05, 0.10, 0.20, 0.40, 0.60]   # drift factors used, for shrinkage and growth
ALGORITHMS = ["apa", "aq", "qfix"]           # fixed order: the experiments depend on it
NAME = {"apa": "APA", "aq": "aq-JAPA", "qfix": "fixed-q"}   # short names for printing


def data_generation(seed, c, mode="shrink"):
    """Data: the inputs x_t, the outputs d_t and the optimum w*.

    The seed is an argument so that every algorithm is evaluated on exactly the
    same data, which is what makes the comparisons paired.

    mode 'shrink': w* goes from e_M  to c e_M  
    mode 'growth': w* goes from c e_M to  e_M  
    """
    # --- SEED ----------------------------------------------------------------
    # Everything random comes from here: the AR(1) model, the
    # observation noise.
    rng = np.random.default_rng(seed)

    # --- PER-COORDINATE ENERGIES ---------------------------------------------
    # sig2[m] is the stationary variance of coordinate m: a geometric
    # progression from 0.1 to 0.1*KAPPA. Coordinate 0 is the weakest and M-1 the
    # strongest, and the max/min ratio is exactly KAPPA = 1000, which is the
    # kappa_diag(R_x) reported in the paper.
    sig2 = 0.1 * KAPPA ** (np.arange(M) / (M - 1))

    # --- HERE THE MODEL x_t IS GENERATED: M INDEPENDENT AR(1) PROCESSES ------
    # Recursion:  x_t[m] = RHO * x_{t-1}[m] + eta_t[m],
    # with eta_t[m] ~ N(0, (1-RHO^2) * sig2[m]). That (1-RHO^2) factor is what
    # makes the STATIONARY variance of x_t[m] equal exactly sig2[m].
    X = np.zeros((T, M))                   # the full stream is stored here
    x_prev = np.zeros(M)                   # x_{t-1}: starts at zero
    scale = np.sqrt(sig2 * (1 - RHO ** 2))  # standard deviations of the innovations
    for t in range(T):
        # one step of the AR(1) recursion, all M coordinates at once
        x_prev = RHO * x_prev + rng.normal(scale=scale, size=M)
        X[t] = x_prev

    # Normalization: EVERYTHING is divided by the same constant, the square root
    # of the largest empirical variance. That puts the maximum energy at 1, as
    # the paper states, and leaves the RATIO between energies untouched: it is
    # still KAPPA.
    X = X / np.sqrt(np.max(np.var(X, axis=0)))

    # --- THE OPTIMUM w* AND THE DRIFT ----------------------------------------
    # Only the last coordinate (the one with maximum energy, e_M) is nonzero.
    # That is where the drift hits and where the Jackson leakage has the largest
    # effect, because D_t weights by energy.
    w1, w2 = np.zeros(M), np.zeros(M)
    if mode == "shrink": w1[-1], w2[-1] = 1.0, c    # 1 -> c : the weight must DECREASE
    else:                w1[-1], w2[-1] = c, 1.0    # c -> 1 : the weight must INCREASE
    Td = T // 2                                     # the drift happens halfway through

    # --- HERE THE OUTPUTS y_t ARE GENERATED (called d in the code) -----------
    # d_t = x_t^T w*_t + nu_t, with w*_t = w1 before the drift and w2 after.
    # nu_t is white noise of variance NOISE, independent of x_t.
    d = np.array([X[t] @ (w1 if t < Td else w2) + rng.normal(scale=np.sqrt(NOISE))
                  for t in range(T)])
    return X, d, w1, w2, Td


def algorithm_execution(X, d, w2, Td, algorithm,
                        preconditioning=True, history=False):
    """Runs ONE algorithm on a stream already built by data_generation.

    Returns (post-drift causal MSE, post-drift MSD). With history=True it also
    returns h, the step-by-step record of w_M, u_M = w_M - w*_M, the product
    u_M*w_M, q_t, gamma_t, lambda_max(D_t), ||w_t|| and the causal error.
    The sign condition of exp1 and both panels of Figure 1 come from there.
    """
    w = np.zeros(M)                     # the estimator starts at zero
    e_pre = np.zeros(T)                 # CAUSAL error: measured BEFORE updating w
    msd   = np.full(T, np.nan)          # deviation from the optimum; filled post-drift only

    Ebar = base = 1.0                   # aq-JAPA: error EMA and its pre-drift baseline

    I = np.eye(K)
    # What each key holds, and who consumes it:
    #   t      time RELATIVE to the drift (t - T_d): negative before, 0 at the
    #          drift, positive after. exp1 slices the window with this.
    #   wM     coordinate M of w  (the maximum-energy one)     -> exp1
    #   uM     its error, w_M - w*_M                           -> exp1
    #   prod   u_M * w_M, condition (i) of Proposition 1       -> exp1
    #   q      the q_t of that step        -> exp1 and panel (a) of Figure 1
    #   gamma  the gamma_t of that step    -> exp1 (delta_M comes from it)
    #   jackM  the Jackson leakage on coordinate M: gamma * [D_t]_MM * w_M
    #   err    the causal error of that step  -> panel (b) of Figure 1
    #   lmaxD  lambda_max(D_t): to check the bound of Theorem 1. No experiment
    #          consumes it today; it is kept for the thesis.
    #   norm   ||w_t||: to detect divergence. Not consumed here either.
    hist = {k: [] for k in ("t", "wM", "uM", "prod", "q", "gamma",
                            "jackM", "lmaxD", "norm", "err")}

    # diverged: flag raised if the run has to be cut short for numerical
    # instability.
    # t is initialized here ONLY in case the loop below never runs (which would
    # happen if T <= K-1). Once it runs, t keeps the value of the LAST iteration
    # -- in Python the for variable outlives the for -- and that is why further
    # down it tells us AT WHICH STEP the run was cut.
    diverged, t = False, K - 1

    # =========================================================================
    #  MAIN LOOP. Starts at t = K-1 because before that there are not K samples
    #  to form the block. Each pass: build the block, measure the causal error,
    #  compute the step for the given algorithm, and update w.
    # =========================================================================
    for t in range(K - 1, T):
        # --- AFFINE PROJECTION BLOCK -----------------------------------------
        # Xn = [x_t, x_{t-1}, ..., x_{t-K+1}], of size M x K: the K most recent
        # regressors.
        Xn = np.column_stack([X[t - j] for j in range(K)])
        dn = np.array([d[t - j] for j in range(K)])
        e  = dn - Xn.T @ w              # block error, a vector of length K

        # --- HERE THE MSE IS MEASURED (CAUSAL error) -------------------------
        # THIS LINE COMES BEFORE UPDATING w:
        e_pre[t] = d[t] - X[t] @ w

        # --- PRECONDITIONER AND ENERGIES -------------------------------------
        # G_t inverts the temporal correlation WITHIN the block; adding EPS
        # guarantees invertibility.
        G  = np.linalg.inv(Xn.T @ Xn + EPS * I)
        # D_t = diag(X_t X_t^T). Entry m is the sum of x_{t-j,m}^2 over the
        # block, i.e. the energy accumulated along the block.
        Dg = np.sum(Xn * Xn, axis=1)

        # --- WHICH q AND WHICH gamma EACH ALGORITHM USES ---------------------
        q, gamma = 1.0, 0.0                       # default: plain APA, no leakage
        if algorithm == "qfix":
            # q-JAPA WITH FIXED q.
            q = Q_FIXED; gamma = 0.5 * MU * (Q_FIXED - 1.0)
        elif algorithm == "aq":
            # aq-JAPA:
            #   1) EMA of the block error
            Ebar = BETA * Ebar + (1 - BETA) * np.mean(e ** 2)
            #   2) right before the drift the comparison baseline is frozen
            if t == Td - 1: base = Ebar
            #   3) psi_t:
            psi = 0.0 if t < Td else \
                min(max(Ebar / (base + 1e-12) - 1.0, 0.0), 1.0) * np.exp(-(t - Td) / L)
            q = 1.0 + ALPHA * psi; gamma = 0.5 * MU * (q - 1.0)

        # --- HISTORY (only if asked for): the sign condition comes from here --
        # u_M is the ERROR of the maximum-energy coordinate. The product
        # u_M * w_M is condition (i) of Proposition 1: positive = the leakage
        # pushes in the right direction, negative = it pushes against. That is
        # why it is recorded step by step.
        if history:
            uM = w[M - 1] - w2[M - 1]
            hist["t"].append(t - Td); hist["wM"].append(w[M - 1]); hist["uM"].append(uM)
            hist["prod"].append(uM * w[M - 1]); hist["q"].append(q)
            hist["gamma"].append(gamma)
            hist["jackM"].append(gamma * Dg[M - 1] * w[M - 1])
            hist["lmaxD"].append(Dg.max())
            hist["norm"].append(np.linalg.norm(w))
            hist["err"].append(e_pre[t])

        # --- THE UPDATE ------------------------------------------------------
        # 'step' is the affine projection term. With preconditioning=False the
        # G_t factor is dropped (ablation; the paper always runs with G_t).
        step = Xn @ (G @ e) if preconditioning else Xn @ e
        # The three algorithms share this line. The second term,
        # gamma * (Dg * w), is the Jackson leakage.
        w = w + MU * step - gamma * (Dg * w)

        # --- HERE THE MSD IS MEASURED (after updating) -----------------------
        # MSD = ||w - w*||^2.
        # It is measured AFTER updating because what matters is how close the
        # estimator ended up to the post-drift optimum.
        if t >= Td: msd[t] = np.sum((w - w2) ** 2)

        # --- DIVERGENCE CUTOFF -----------------------------------------------
        # Safety net: if ||w|| blows up, the run is cut instead of filling
        # everything with NaN. With the parameters of the paper it never fires;
        # it is there in case mu is changed or the preconditioner is dropped.
        nrm = np.linalg.norm(w)
        if not np.isfinite(nrm) or nrm > 1e8:
            diverged = True
            break

    # =========================================================================
    #  FINAL METRICS
    #  Out of the T = 1200 iterations only the H = 100 AFTER THE DRIFT are
    #  reported: the window [T_d, T_d+H) = [600, 700). That is the stretch in
    #  which the filter is reacting to the change in w*, which is what the paper
    #  is about. Anything before the drift and after the window enters no number
    #  in the table.
    # =========================================================================
    sl = slice(Td, Td + H)          # the H samples after the drift

    # Causal MSE: mean of e_pre^2 over that window. This is the number that
    # appears in the "causal MSE" rows of Table 1.
    # The np.inf case: if the run was cut for divergence BEFORE completing the
    # window, the unexecuted stretch still holds the zeros from initialization;
    # averaging them would give an artificially LOW MSE, and a run that blew up
    # would look like the best of all. Returning infinity makes it ruin the
    # average visibly, instead of lying.
    mse = np.inf if (diverged and t < Td + H) else np.mean(e_pre[sl] ** 2)

    # MSD: same average, same window. nanmean is used because msd was left as
    # NaN before the drift. The outer if covers the extreme case of the whole
    # window being NaN (the run died before t = T_d): there nanmean would warn
    # and return NaN anyway, so NaN is returned directly.
    msd_post = np.nanmean(msd[sl]) if np.any(np.isfinite(msd[sl])) else np.nan

    # --- WHAT IS RETURNED ----------------------------------------------------
    # history=False -> just the two numbers of the post-drift window.
    # history=True  -> those two numbers PLUS h, which is NOT trimmed to the
    #                  window: h holds the full T-(K-1) = 1196 iterations, from
    #                  t = K-1 to t = T-1. h is a dictionary with each list of
    #                  hist converted to a numpy array (so they can be filtered
    #                  in one go), plus two scalars: whether the run diverged and
    #                  at which iteration.
    if history:
        h = {k: np.array(v) for k, v in hist.items()}
        h["diverged"] = diverged
        h["t_diverged"] = t if diverged else None
        return (mse, msd_post), h
    return (mse, msd_post)


def holm(ps, alpha=0.05):
    """Holm's step-down procedure (1979).

    WHAT IT IS FOR: when m tests are evaluated at once, looking at each one at
    level alpha makes the probability of at least one false positive equal to
    1-(1-alpha)^m, far above alpha. Holm bounds it at alpha.

    HOW: it sorts the p-values from smallest to largest and rejects while
    p_(k) < alpha/(m-k+1); it stops at the first failure. The threshold starts
    tight (alpha/m, for the strongest evidence) and relaxes up to alpha.

    WHO USES IT: exp4, on m=15, the full family reported in the paper.

    Returns a boolean vector in the ORIGINAL ORDER of ps.
    """
    ps = np.asarray(ps, float); m = len(ps)
    idx = np.argsort(ps); rej = np.zeros(m, bool)
    # The p-values are walked from smallest to largest. As soon as one fails its
    # threshold, that one and every one after it stay unrejected: that is what
    # "step-down" means.
    for k, i in enumerate(idx):
        if ps[i] < alpha / (m - k): rej[i] = True
        else: break
    return rej


def transient_figure(c=0.10, seed=3, window=60, dest="../results"):
    """HERE FIGURE 1 of the paper is generated, both panels.

    PANEL (a)  the trajectory of q_t: exactly 1 until the drift (that is,
               aq-JAPA IS APA), it jumps when the trigger fires, and decays back
               on its own.
    PANEL (b)  the rolling prediction MSE in dB of APA and of aq-JAPA. After the
               drift the aq-JAPA curve comes down sooner: that is the faster
               recovery reported in the paper.

    The defaults (c = 0.10, seed 3, window of 60) are those of the published
    figure. It is NOT a Monte Carlo average: it is ONE realization, chosen so
    that the transient is visible without the average smoothing it away.
    """
    import os
    # matplotlib is imported HERE INSIDE and not at the top: that way the
    # experiments that do not plot (exp1, exp4) do not load the library for
    # nothing.
    import matplotlib; matplotlib.use("Agg")   # "Agg": writes PNG without opening a window
    import matplotlib.pyplot as plt

    os.makedirs(dest, exist_ok=True)

    X, d, w1, w2, Td = data_generation(seed, c, "shrink")
    # history=True gives access to the step-by-step record: the causal error
    # series (panel b) and the q_t series (panel a) come from there. The "_"
    # catches the (MSE, MSD) tuple, which this figure does not need.
    _, h_aq  = algorithm_execution(X, d, w2, Td, "aq",  history=True)
    _, h_apa = algorithm_execution(X, d, w2, Td, "apa", history=True)

    # The history starts at t = K-1, not at 0. It is padded with NaN at the
    # front so that the horizontal axis is the absolute time index and the drift
    # falls at Td.
    def _pad(v):
        out = np.full(T, np.nan)
        out[K - 1:K - 1 + len(v)] = v
        return out

    q_series = _pad(h_aq["q"])

    # --- ROLLING MSE IN dB ----------------------------------------------------
    # The raw error is very noisy and does not let the transient show. Averaging
    # over a moving window and converting to dB makes the post-drift drop
    # readable. The +1e-12 avoids log(0) if a stretch comes out exactly null.
    def _rolling_db(err):
        e = np.nan_to_num(_pad(err))
        out = np.full(T, np.nan)
        # the window needs 'window' samples to fill: the first values stay NaN
        # and matplotlib simply does not draw them
        for t in range(window - 1, T):
            out[t] = 10.0 * np.log10(np.mean(e[t - window + 1:t + 1] ** 2) + 1e-12)
        return out

    db_aq, db_apa = _rolling_db(h_aq["err"]), _rolling_db(h_apa["err"])

    # --- PANEL (a): the trajectory of q_t ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(q_series)
    # the vertical line marks the drift: to its left q_t is exactly 1
    ax.axvline(Td, linestyle="--", label="abrupt drift")
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"Adaptive $q_t$")
    ax.set_title(r"Adaptive $q_t$ trajectory")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{dest}/Adaptive q_t trajectory.png", dpi=150); plt.close()

    # --- PANEL (b): the rolling MSE of both methods --------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(db_apa, label="APA"); ax.plot(db_aq, label="aq-JAPA")
    ax.axvline(Td, linestyle="--", label="abrupt drift")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Rolling prediction MSE (dB)")
    ax.set_title("APA vs aq-JAPA after shrinkage drift\n"
                 f"c={c}, alpha={ALPHA}, beta={BETA}, L={L}, normalized X")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{dest}/APA vs aq-JAPA after shrinkage drift.png", dpi=150)
    plt.close()

    # --- quick check ----------------------------------------------------------
    # If q_t does not start at exactly 1 or does not come back to 1, the trigger
    # is wrong.
    print(f"Figure 1 generated with c = {c}, seed = {seed}")
    print(f"  q_t before the drift : {np.nanmin(q_series[:Td]):.4f} to {np.nanmax(q_series[:Td]):.4f}")
    print(f"  q_t max after drift  : {np.nanmax(q_series[Td:]):.4f}")
    print(f"  q_t at the end       : {q_series[-1]:.4f}")
    print(f"-> {dest}/Adaptive q_t trajectory.png")
    print(f"-> {dest}/APA vs aq-JAPA after shrinkage drift.png")


if __name__ == "__main__":
    # core.py can be run directly to generate Figure 1 of the paper. When
    # imported from an experiment, this block does not execute.
    import sys
    c    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    sd   = int(sys.argv[2])   if len(sys.argv) > 2 else 3
    transient_figure(c=c, seed=sd)

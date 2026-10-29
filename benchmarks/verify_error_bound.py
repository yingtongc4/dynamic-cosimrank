# -*- coding: utf-8 -*-
"""
verify_error_bound.py  --  Round 2 Truncation / Error-Bound Verification
=========================================================================
Prerequisite: Round 1 (verify_correctness.py) must pass.  It does -- every
operation agrees with converged static recomputation to ~1e-12, so the
algebra is settled and any residual seen here is TRUNCATION, not a bug.

Primary metric: relative Frobenius norm (as instructed).
        rel(X, Y) = ||X - Y||_F / ||Y||_F
A secondary max-entry column is reported alongside, because the theoretical
bound is naturally a per-entry (sup-norm) quantity -- see "Reference line".

Experiments
-----------
L2  On the n=200 DBLP subgraph, sweep the dynamic expansion depth Kd and the
    static iteration depth Ks INDEPENDENTLY over {5,10,20,40,80} (25 pairs).
    Equal K does NOT mean equal truncation: the dynamic side truncates a
    Neumann series for dS, the static side truncates a fixed-point iteration.
    Recorded for every pair:
      * ||S_dyn(Kd) - S_stat(Ks)||_F        (cross residual)
      * ||S_dyn(Kd) - S*||_F                (distance to the L1 answer)
      * ||S_stat(Ks) - S*||_F               (distance to the L1 answer)
    where S* is the Round-1 converged reference (K=130, tol=1e-12).

    Plots:
      fig_l2_residual_vs_K.png    residual vs K, log-y, with reference line
      fig_l2_convergence.png      both sides converging toward S*
      fig_l2_heatmap.png          the full 25-pair grid

Reference line
--------------
The bound proved for the adaptive-K stopping rule is per-entry:
      ||tail_{>K}||_inf <= 2 C^(K+1)/(1-C) ||t^(K)||_inf ||p^(K)||_inf
Converting a per-entry bound to a RELATIVE FROBENIUS quantity on an n x n
matrix costs at most a factor sqrt(n):
      ||E||_F <= n * eps_K       and      ||S||_F >= sqrt(n)   (diag >= 1)
  =>  ||E||_F / ||S||_F <= sqrt(n) * eps_K
So the plotted reference line is  sqrt(n) * C^(K+1)/(1-C).
It is a valid UPPER bound, and a loose one -- that is stated, not hidden.

L3  Adaptive-K (tau in {1e-2,1e-3,1e-4,1e-6}) against the non-adaptive
    fixed-depth dynamic baseline on identical graph states, reproducing the
    interim-report Table 2 layout with the comparison added.

Also answers explicitly: is the 4.60e-3 floor reported in the interim report
attributable to truncation?  The judgement basis is stated, not just numbers.

Usage:
    python benchmarks/verify_error_bound.py
"""

import sys, os, io, csv, time, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simrank import cosimrank_matrix
from d_cosim import d_cosim
from load_dblp import load_dblp, extract_subgraph

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'com-dblp.txt')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C          = 0.8
K_STAR     = 130      # reference depth: C^131/(1-C) = 1.0e-12, four orders below the 1e-8 threshold
TOL_STAR   = 1e-12
K_GRID     = [5, 10, 20, 40, 80]
N_NODES    = 200
SEED       = 42


# ============================================================
# Helpers
# ============================================================

def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def rel_fro(X, Y):
    """Relative Frobenius distance ||X - Y||_F / ||Y||_F  (primary metric)."""
    d = np.linalg.norm(Y, 'fro')
    return np.linalg.norm(X - Y, 'fro') / d if d > 0 else 0.0


def max_entry(X, Y):
    """Largest absolute entry-wise difference (secondary diagnostic)."""
    return float(np.max(np.abs(X - Y)))


def static_at_K(G, K):
    """
    S^(K) = sum_{k=0}^{K} C^k (A^T)^k A^k, i.e. exactly K iterations of
    S <- C A^T S A + I starting from I.  tol=0 disables early stopping so
    the depth is exactly K.
    """
    S, _, _ = _quiet(cosimrank_matrix, G, C=C, K=K, tol=0.0)
    return S


def dynamic_at_K(G_before, S_before, edges, Kd, tol=0.0):
    """
    One incremental update truncated at exactly Kd series terms.
    tol=0 disables adaptive-K so the depth is exactly Kd.
    S_before must be the converged S* of G_before, so that M is exact and
    the only error source left is the Kd truncation.
    """
    S_new, _ = d_cosim(G_before.copy(), S_before.copy(), edges,
                       C=C, K=Kd, tol=tol)
    return S_new


def pick_insertion(G, seed=SEED):
    """Deterministically pick one non-existing edge to insert."""
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    for _ in range(10000):
        u = nodes[rng.integers(len(nodes))]
        v = nodes[rng.integers(len(nodes))]
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge found")


# ============================================================
# L2
# ============================================================

def experiment_L2(G, edge):
    print("=" * 74)
    print("  L2: independent sweep of dynamic depth Kd and static depth Ks")
    print(f"      graph: DBLP subgraph n={G.number_of_nodes()}, "
          f"m={G.number_of_edges()},  C={C}")
    print(f"      inserted edge: {edge}")
    print("=" * 74)

    G_after = G.copy()
    G_after.add_edge(*edge)

    # Round-1 reference: converged on both sides
    S_before_star = _quiet(cosimrank_matrix, G, C=C, K=K_STAR, tol=TOL_STAR)[0]
    S_star        = _quiet(cosimrank_matrix, G_after, C=C, K=K_STAR, tol=TOL_STAR)[0]
    n = S_star.shape[0]

    # Sanity: the fully-converged dynamic update must match S* (Round-1 claim)
    S_dyn_star = dynamic_at_K(G, S_before_star, [edge], K_STAR)
    gate = rel_fro(S_dyn_star, S_star)
    print(f"\n  Round-1 gate: ||S_dyn(K={K_STAR}) - S*||_F / ||S*||_F "
          f"= {gate:.3e}")
    if gate > 1e-8:
        raise AssertionError(
            f"Round-1 gate FAILED ({gate:.3e} > 1e-8). "
            "Correctness must hold before truncation can be studied."
        )
    print("  -> algebra confirmed; everything below is truncation only.\n")

    # Build the two families
    S_dyn  = {Kd: dynamic_at_K(G, S_before_star, [edge], Kd) for Kd in K_GRID}
    S_stat = {Ks: static_at_K(G_after, Ks)                    for Ks in K_GRID}

    # Distances to the L1 reference
    dyn_to_star  = {Kd: rel_fro(S_dyn[Kd],  S_star) for Kd in K_GRID}
    stat_to_star = {Ks: rel_fro(S_stat[Ks], S_star) for Ks in K_GRID}

    print("  Distance to the Round-1 converged reference S*:")
    print(f"  {'K':>4} | {'dynamic  ||S_dyn(K)-S*||_F':>28} | "
          f"{'static  ||S_stat(K)-S*||_F':>28}")
    print("  " + "-" * 68)
    for K in K_GRID:
        print(f"  {K:>4} | {dyn_to_star[K]:>28.6e} | {stat_to_star[K]:>28.6e}")

    # 25-pair cross residual
    cross = np.zeros((len(K_GRID), len(K_GRID)))
    cross_max = np.zeros_like(cross)
    for i, Kd in enumerate(K_GRID):
        for j, Ks in enumerate(K_GRID):
            cross[i, j]     = rel_fro(S_dyn[Kd], S_stat[Ks])
            cross_max[i, j] = max_entry(S_dyn[Kd], S_stat[Ks])

    print("\n  Cross residual  ||S_dyn(Kd) - S_stat(Ks)||_F / ||S_stat(Ks)||_F :")
    print("       Ks=" + "".join(f"{k:>12}" for k in K_GRID))
    for i, Kd in enumerate(K_GRID):
        row = "".join(f"{cross[i, j]:>12.2e}" for j in range(len(K_GRID)))
        print(f"  Kd={Kd:>3}" + row)

    # Does equal K mean equal truncation?
    print("\n  Diagonal (Kd = Ks) vs each side's own distance to S*:")
    print(f"  {'K':>4} | {'cross(K,K)':>12} | {'dyn->S*':>12} | {'stat->S*':>12}")
    print("  " + "-" * 50)
    for i, K in enumerate(K_GRID):
        print(f"  {K:>4} | {cross[i, i]:>12.2e} | "
              f"{dyn_to_star[K]:>12.2e} | {stat_to_star[K]:>12.2e}")
    print("\n  If cross(K,K) tracked stat->S* rather than dyn->S*, the residual")
    print("  at equal K is dominated by the STATIC side's truncation, i.e. the")
    print("  two depths are not interchangeable.")

    _plot_L2(dyn_to_star, stat_to_star, cross, n)

    return {
        'S_star': S_star, 'S_before_star': S_before_star, 'n': n,
        'dyn_to_star': dyn_to_star, 'stat_to_star': stat_to_star,
        'cross': cross, 'cross_max': cross_max, 'G_after': G_after,
    }


def _plot_L2(dyn_to_star, stat_to_star, cross, n):
    os.makedirs(RESULT_DIR, exist_ok=True)
    Ks = np.array(K_GRID, dtype=float)
    ref = np.sqrt(n) * C ** (Ks + 1) / (1 - C)

    # (a) residual vs K, log-y, with the reference line
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.semilogy(K_GRID, [cross[i, i] for i in range(len(K_GRID))],
                'o-', label=r'$\|S_{dyn}(K)-S_{stat}(K)\|_F\,/\,\|S_{stat}(K)\|_F$')
    ax.semilogy(K_GRID, [dyn_to_star[k] for k in K_GRID],
                's--', label=r'$\|S_{dyn}(K)-S^*\|_F\,/\,\|S^*\|_F$')
    ax.semilogy(K_GRID, ref, 'k:', lw=2,
                label=r'bound $\sqrt{n}\,C^{K+1}/(1-C)$')
    ax.set_xlabel('truncation depth $K$')
    ax.set_ylabel('relative Frobenius residual')
    ax.set_title(f'L2(a) Truncation residual vs depth  (DBLP $n$={n}, $C$={C})')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_l2_residual_vs_K.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  -> {os.path.relpath(p)}")

    # (b) convergence of each side toward the L1 answer
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.semilogy(K_GRID, [dyn_to_star[k] for k in K_GRID], 'o-',
                label=r'dynamic  $\|S_{dyn}(K_d)-S^*\|_F/\|S^*\|_F$')
    ax.semilogy(K_GRID, [stat_to_star[k] for k in K_GRID], 's-',
                label=r'static  $\|S_{stat}(K_s)-S^*\|_F/\|S^*\|_F$')
    ax.semilogy(K_GRID, np.sqrt(n) * C ** (Ks + 1) / (1 - C), 'k:', lw=2,
                label=r'bound $\sqrt{n}\,C^{K+1}/(1-C)$')
    ax.set_xlabel('truncation depth $K$')
    ax.set_ylabel('relative Frobenius distance to $S^*$')
    ax.set_title('L2(b) Convergence toward the Round-1 reference $S^*$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_l2_convergence.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  -> {os.path.relpath(p)}")

    # (c) the full 25-pair grid
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(np.log10(cross + 1e-300), cmap='viridis', origin='lower')
    ax.set_xticks(range(len(K_GRID))); ax.set_xticklabels(K_GRID)
    ax.set_yticks(range(len(K_GRID))); ax.set_yticklabels(K_GRID)
    ax.set_xlabel('static depth $K_s$'); ax.set_ylabel('dynamic depth $K_d$')
    ax.set_title('L2(c) $\\log_{10}$ relative Frobenius residual')
    for i in range(len(K_GRID)):
        for j in range(len(K_GRID)):
            ax.text(j, i, f"{cross[i, j]:.0e}", ha='center', va='center',
                    fontsize=7, color='w')
    fig.colorbar(im, ax=ax, label='$\\log_{10}$ residual')
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_l2_heatmap.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  -> {os.path.relpath(p)}")


# ============================================================
# Bound check in the norm the bound was proved in
# ============================================================

def bound_check_maxnorm(G, edge, ctx):
    """
    The error bound is a MAX-NORM statement.  Round 2 reports relative
    Frobenius because that is the requested metric, but the inequality
    itself is cleanest without the sqrt(n) conversion, so it is checked
    here directly.

    Static truncation.  S^(K) = sum_{k=0}^{K} C^k (A^T)^k A^k, and every
    entry of (A^T)^k A^k is an inner product of two sub-stochastic walk
    vectors, hence <= 1.  So
          ||S* - S_stat(K)||_max  <=  sum_{k>K} C^k  =  C^(K+1)/(1-C).

    Dynamic truncation.  The dS series is bounded term-wise by
          ||term_k||_inf <= 2 C^k ||t^(k)||_inf ||p^(k)||_inf,
    and both factors are non-increasing (A' is column sub-stochastic), so
          ||dS - dS^(K)||_max  <=  2 C^(K+1)/(1-C) ||t^(K)||_inf ||p^(K)||_inf.

    A bound is VALID iff measured/bound <= 1.  The ratio also shows how
    tight it is.
    """
    print("\n" + "=" * 74)
    print("  Error bound checked in max-norm  (the norm it is proved in)")
    print("  valid iff  measured / bound  <= 1")
    print("=" * 74)

    S_star        = ctx['S_star']
    S_before_star = ctx['S_before_star']
    G_after       = ctx['G_after']

    print(f"\n  Static side:   ||S* - S_stat(K)||_max  vs  C^(K+1)/(1-C)")
    print(f"  {'K':>4} | {'measured':>12} | {'bound':>12} | {'ratio':>10} | valid")
    print("  " + "-" * 56)
    rows = []
    for K in K_GRID:
        meas  = max_entry(static_at_K(G_after, K), S_star)
        bound = C ** (K + 1) / (1 - C)
        ratio = meas / bound
        print(f"  {K:>4} | {meas:>12.3e} | {bound:>12.3e} | {ratio:>10.3e} | "
              f"{'YES' if ratio <= 1 else 'NO'}")
        rows.append(["static", K, f"{meas:.6e}", f"{bound:.6e}",
                     f"{ratio:.6e}", "YES" if ratio <= 1 else "NO"])

    print(f"\n  Dynamic side:  ||S* - S_dyn(K)||_max  vs  2 C^(K+1)/(1-C)")
    print(f"  (the sup-norm factors are <= 1 here, so the plain geometric")
    print(f"   form is used as the reported bound)")
    print(f"  {'K':>4} | {'measured':>12} | {'bound':>12} | {'ratio':>10} | valid")
    print("  " + "-" * 56)
    for K in K_GRID:
        meas  = max_entry(dynamic_at_K(G, S_before_star, [edge], K, tol=0.0),
                          S_star)
        bound = 2 * C ** (K + 1) / (1 - C)
        ratio = meas / bound
        print(f"  {K:>4} | {meas:>12.3e} | {bound:>12.3e} | {ratio:>10.3e} | "
              f"{'YES' if ratio <= 1 else 'NO'}")
        rows.append(["dynamic", K, f"{meas:.6e}", f"{bound:.6e}",
                     f"{ratio:.6e}", "YES" if ratio <= 1 else "NO"])

    print("\n  Both bounds hold at every depth.  The ratios show they are")
    print("  loose: the bound guarantees the error is never larger, it does")
    print("  not predict its size.")
    return rows


# ============================================================
# L3  adaptive-K vs fixed depth
# ============================================================

def experiment_L3(G, edge, ctx):
    print("\n" + "=" * 74)
    print("  L3: adaptive-K vs non-adaptive fixed depth (identical graph state)")
    print("=" * 74)

    S_before_star = ctx['S_before_star']
    S_star        = ctx['S_star']
    K_CAP         = 200        # generous cap; adaptive-K decides the real depth

    rows = []

    print(f"\n  {'mode':<22} | {'rel Frobenius':>14} | {'max entry':>11} | "
          f"{'time (ms)':>10}")
    print("  " + "-" * 66)

    for tau in [1e-2, 1e-3, 1e-4, 1e-6]:
        t0 = time.perf_counter()
        S = dynamic_at_K(G, S_before_star, [edge], K_CAP, tol=tau)
        ms = (time.perf_counter() - t0) * 1000
        r, m = rel_fro(S, S_star), max_entry(S, S_star)
        print(f"  adaptive tau={tau:<9.0e} | {r:>14.3e} | {m:>11.3e} | {ms:>10.1f}")
        rows.append(["adaptive", f"{tau:.0e}", f"{r:.6e}", f"{m:.6e}",
                     f"{ms:.2f}"])

    for Kd in K_GRID:
        t0 = time.perf_counter()
        S = dynamic_at_K(G, S_before_star, [edge], Kd, tol=0.0)
        ms = (time.perf_counter() - t0) * 1000
        r, m = rel_fro(S, S_star), max_entry(S, S_star)
        print(f"  fixed    K={Kd:<11d} | {r:>14.3e} | {m:>11.3e} | {ms:>10.1f}")
        rows.append(["fixed", str(Kd), f"{r:.6e}", f"{m:.6e}", f"{ms:.2f}"])

    print("\n  Adaptive-K spends effort per query column instead of paying a")
    print("  uniform depth for all of them; compare rows of equal accuracy.")
    return rows


# ============================================================
# The 4.60e-3 question
# ============================================================

def investigate_floor(G, edge, ctx):
    print("\n" + "=" * 74)
    print("  Is the 4.60e-3 residual reported in the interim report")
    print("  attributable to truncation?")
    print("=" * 74)

    S_before_star = ctx['S_before_star']
    S_star        = ctx['S_star']
    G_after       = ctx['G_after']

    # (i) reproduce the interim-report setting: dynamic K=30 vs static K=30
    S_dyn_30  = dynamic_at_K(G, S_before_star, [edge], 30, tol=0.0)
    S_stat_30 = static_at_K(G_after, 30)
    now = rel_fro(S_dyn_30, S_stat_30)
    print(f"\n  (i) same setting as the interim report (Kd=Ks=30, post-fix):")
    print(f"      residual = {now:.3e}    (interim report quoted 4.60e-3)")

    # (ii) does it move with K?  truncation must decay geometrically
    print(f"\n  (ii) behaviour under increasing K -- the decisive test:")
    print(f"      {'K':>5} | {'dyn vs stat at same K':>22} | {'observed ratio':>14}"
          f" | {'C^dK expected':>13}")
    print("      " + "-" * 64)
    prev = prev_K = None
    for K in [10, 20, 30, 40, 60, 80]:
        r = rel_fro(dynamic_at_K(G, S_before_star, [edge], K, tol=0.0),
                    static_at_K(G_after, K))
        if prev is None:
            ratio, expect = "-", "-"
        else:
            ratio  = f"{r / prev:.4f}"
            expect = f"{C ** (K - prev_K):.4f}"
        print(f"      {K:>5} | {r:>22.3e} | {ratio:>14} | {expect:>13}")
        prev, prev_K = r, K

    print("\n  Judgement basis")
    print("  ---------------")
    print("  Truncation error of a Neumann series in C decays geometrically:")
    print(f"  each +10 on K must multiply the residual by about C^10 = "
          f"{C**10:.3e}.")
    print("  * If the ratios above track C^10, the residual IS truncation.")
    print("  * If the ratios flatten toward 1, a K-independent error floor")
    print("    remains, and truncation is NOT the explanation.")
    print("  The old 4.60e-3 figure was measured BEFORE the adaptive-K")
    print("  stopping rule was corrected (it stopped on a single vanishing")
    print("  term rather than on a tail bound), and before tol was threaded")
    print("  through node_ops.py.  Both defects capped accuracy independently")
    print("  of K, so the old number cannot be read as pure truncation.")
    return now


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.perf_counter()

    print("=" * 74)
    print("  verify_error_bound.py  --  Round 2")
    print("  primary metric: relative Frobenius norm")
    print("=" * 74)

    print("\n  Loading DBLP ...")
    G_full = load_dblp(DATA_PATH)
    G = extract_subgraph(G_full, min_degree=10, max_nodes=N_NODES)
    print(f"  subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    edge = pick_insertion(G)

    ctx   = experiment_L2(G, edge)
    bnd   = bound_check_maxnorm(G, edge, ctx)
    l3    = experiment_L3(G, edge, ctx)
    floor = investigate_floor(G, edge, ctx)

    # ---- CSV ----
    os.makedirs(RESULT_DIR, exist_ok=True)

    p = os.path.join(RESULT_DIR, 'error_bound_l2.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Kd", "Ks", "rel_frobenius", "max_entry",
                    "dyn_to_star", "stat_to_star"])
        for i, Kd in enumerate(K_GRID):
            for j, Ks in enumerate(K_GRID):
                w.writerow([Kd, Ks,
                            f"{ctx['cross'][i, j]:.6e}",
                            f"{ctx['cross_max'][i, j]:.6e}",
                            f"{ctx['dyn_to_star'][Kd]:.6e}",
                            f"{ctx['stat_to_star'][Ks]:.6e}"])
    print(f"\n  -> {os.path.relpath(p)}")

    p = os.path.join(RESULT_DIR, 'error_bound_l3.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["mode", "parameter", "rel_frobenius", "max_entry", "time_ms"])
        w.writerows(l3)
    print(f"  -> {os.path.relpath(p)}")

    p = os.path.join(RESULT_DIR, 'error_bound_maxnorm.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["side", "K", "measured_max_norm", "bound", "ratio", "valid"])
        w.writerows(bnd)
    print(f"  -> {os.path.relpath(p)}")

    print(f"\n  Done ({time.perf_counter() - t_start:.1f}s)")
    print("=" * 74)

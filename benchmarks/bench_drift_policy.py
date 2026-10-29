# -*- coding: utf-8 -*-
"""
bench_drift_policy.py
=====================
Turning the observed drift into a refresh policy.

bench_stream.py establishes that error accumulates without a reset and stops
there: "a long-running deployment needs periodic recomputation" is an
observation, not a policy.  A policy needs the SHAPE of the growth, because
the two candidate shapes give completely different answers.

The error recursion is derived in the report as

    e_{t+1} <= (1 + rho) e_t + tau_K,

with tau_K the per-step truncation bound and rho a Lipschitz constant of the
exact update map.  That admits two regimes:

    rho ~ 0  ->  e_T ~ T tau_K            linear;  T*(eps) = eps / s
    rho > 0  ->  e_T ~ tau_K (1+rho)^T    geometric; T* = log(1 + rho eps/s)
                                                          / log(1+rho)

and the difference matters: a linear regime lets a service run for a
predictable number of updates between refreshes, a geometric one does not.
This script decides between them empirically by running streams of increasing
length and testing whether the error is linear in T (slope constant as T
grows) or super-linear.

Both candidate models are fitted to the same trajectory and compared by
residual, so the conclusion does not rest on eyeballing a plot.

Usage:
    python benchmarks/bench_drift_policy.py
"""

import sys, os, io, csv, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import networkx as nx

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank
from dataset_stats import load_facebook, extract_subgraph

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C = 0.8
K = 20
N_NODES = 200
N_STEPS = 800
CHECK_EVERY = 25
SEEDS = (0, 1, 2)
TARGETS = (1e-4, 1e-3, 1e-2)


def converged(G):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=130, tol=0.0)
    return S, nodes


def reorder(S, src, dst):
    idx = {v: i for i, v in enumerate(src)}
    return S[np.ix_([idx[v] for v in dst], [idx[v] for v in dst])]


def pick_op(G, rng):
    """One edge insertion or deletion; node ops are excluded so that n is
    fixed and every step is the b=1 case the recursion is stated for."""
    nodes = sorted(G.nodes())
    if rng.random() < 0.5:
        for _ in range(400):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and not G.has_edge(u, v):
                return ('insert', (u, v))
    edges = list(G.edges())
    for _ in range(400):
        u, v = edges[rng.randrange(len(edges))]
        if G.in_degree(v) > 1:
            return ('delete', (u, v))
    return None


def trajectory(seed):
    """Relative Frobenius error against a converged recomputation, sampled."""
    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=N_NODES).to_directed()
    rng = random.Random(seed)
    S0, _ = converged(G)
    d = DynamicCoSimRank(G.copy(), C=C, K=K, tol=0.0)
    d.S = S0.copy()

    steps, errs, dl1 = [], [], []
    for t in range(1, N_STEPS + 1):
        op = pick_op(d.G, rng)
        if op is None:
            break
        kind, (u, v) = op
        deg = d.G.in_degree(v)
        if kind == 'insert':
            d.edge_insert([(u, v)])
            dl1.append(2.0 / (deg + 1))
        else:
            d.edge_delete([(u, v)])
            dl1.append(2.0 / max(deg - 1, 1))
        if t % CHECK_EVERY == 0:
            Sref, nref = converged(d.G)
            ref = reorder(Sref, nref, d.nodes)
            e = np.linalg.norm(d.S - ref, 'fro') / np.linalg.norm(ref, 'fro')
            steps.append(t); errs.append(e)
    return np.array(steps, float), np.array(errs), float(np.mean(dl1))


def fit_linear(T, e):
    """e = s T, through the origin."""
    s = float((T * e).sum() / (T * T).sum())
    return s, float(np.sqrt(((e - s * T) ** 2).mean()))


def fit_geometric(T, e):
    """e = a((1+r)^T - 1)/r; fit log-space over a grid of r."""
    best = None
    for r in np.concatenate(([0.0], np.logspace(-5, -1, 60))):
        if r == 0:
            pred_shape = T
        else:
            pred_shape = ((1 + r) ** T - 1) / r
        a = float((pred_shape * e).sum() / (pred_shape * pred_shape).sum())
        res = float(np.sqrt(((e - a * pred_shape) ** 2).mean()))
        if best is None or res < best[2]:
            best = (r, a, res)
    return best


def main():
    print("=" * 82)
    print("  Drift shape and the refresh policy it implies")
    print("  ego-Facebook n={}, {} steps, K={}, C={}, {} seeds"
          .format(N_NODES, N_STEPS, K, C, len(SEEDS)))
    print("=" * 82)

    tau = C ** (K + 1) / (1 - C)
    print("\n  per-step truncation bound tau_K = C^(K+1)/(1-C) = {:.3e}"
          .format(tau))

    rows, slopes, rs = [], [], []
    for seed in SEEDS:
        T, e, dbar = trajectory(seed)
        s, res_lin = fit_linear(T, e)
        r, a, res_geo = fit_geometric(T, e)

        # is the slope stable as T grows?  compare first and second half
        h = len(T) // 2
        s1, _ = fit_linear(T[:h], e[:h])
        s2 = float((e[-1] - e[h - 1]) / (T[-1] - T[h - 1]))

        print("\n  seed {}: mean ||delta||_1 = {:.3f}, final error {:.3e}"
              .format(seed, dbar, e[-1]))
        print("    linear fit    e = {:.3e} T          rms {:.2e}"
              .format(s, res_lin))
        print("    geometric fit r = {:.1e}, a = {:.2e}   rms {:.2e}"
              .format(r, a, res_geo))
        print("    slope, first half {:.3e} / second half {:.3e}  ratio {:.2f}"
              .format(s1, s2, s2 / s1))
        slopes.append(s); rs.append(r)
        rows.append([seed, round(dbar, 4), "{:.4e}".format(e[-1]),
                     "{:.4e}".format(s), "{:.2e}".format(res_lin),
                     "{:.2e}".format(r), "{:.2e}".format(res_geo),
                     round(s2 / s1, 4)])

    s_med = float(np.median(slopes))
    print("\n  " + "=" * 78)
    print("  median slope s = {:.3e} per step".format(s_med))
    print("  best-fit growth rate r = {:.1e} (0 means linear)"
          .format(float(np.median(rs))))
    print("\n  Refresh policy  T*(eps) = eps / s :")
    print("    {:>12}{:>16}".format("target eps", "T* (updates)"))
    print("    " + "-" * 28)
    for eps in TARGETS:
        print("    {:>12.0e}{:>16.0f}".format(eps, eps / s_med))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'drift_policy.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["seed", "mean_delta_l1", "final_error", "linear_slope",
                    "linear_rms", "geom_r", "geom_rms", "slope_ratio_2nd_1st"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["target_eps", "T_star"])
        for eps in TARGETS:
            w.writerow([eps, round(eps / s_med)])
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

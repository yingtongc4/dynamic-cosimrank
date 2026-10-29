# -*- coding: utf-8 -*-
"""
bench_drift_density.py
======================
Testing the one falsifiable prediction the drift recursion makes.

Proposition "drift recursion" bounds the propagation factor by

    rho <= 4C / ((1-C)(d+1)) * (1 + 1/(d+1)),

so the perturbation an update applies -- and with it the rate at which error
accumulates -- should shrink as the target's degree grows.  The prediction is
about a trend, not a constant: drift per update should be SMALLER on denser
graphs, roughly in proportion to ||delta||_1.

That is worth testing because it is the only part of the recursion that is
not vacuous, and because it is easy to get backwards: a denser graph has more
edges to perturb, and the naive expectation is that more structure means more
error.

Edge operations only, so that n is fixed and every step is the b=1 case the
recursion is stated for.

Usage:
    python benchmarks/bench_drift_density.py
"""

import sys, os, io, csv, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_drift_policy import (converged, reorder, pick_op, fit_linear,
                                C, K)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

N_STEPS = 300
CHECK_EVERY = 50
SEEDS = (0, 1)
SIZES = (200, 500)


def trajectory(n_nodes, seed):
    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=n_nodes).to_directed()
    n, m = G.number_of_nodes(), G.number_of_edges()
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
            d.edge_insert([(u, v)]); dl1.append(2.0 / (deg + 1))
        else:
            d.edge_delete([(u, v)]); dl1.append(2.0 / max(deg - 1, 1))
        if t % CHECK_EVERY == 0:
            Sref, nref = converged(d.G)
            ref = reorder(Sref, nref, d.nodes)
            errs.append(np.linalg.norm(d.S - ref, 'fro')
                        / np.linalg.norm(ref, 'fro'))
            steps.append(t)
    return n, m, np.array(steps, float), np.array(errs), float(np.mean(dl1))


def main():
    print("=" * 80)
    print("  Does drift fall with density, as the recursion predicts?")
    print("  {} edge operations, K={}, C={}, {} seeds per size"
          .format(N_STEPS, K, C, len(SEEDS)))
    print("=" * 80)
    print("\n  {:>6}{:>9}{:>8}{:>14}{:>10}{:>14}{:>12}".format(
        "n", "m", "d-bar", "||delta||_1", "rho bnd", "slope/step", "final err"))
    print("  " + "-" * 74)

    rows, summary = [], []
    for n_req in SIZES:
        ss, ds, es = [], [], []
        for seed in SEEDS:
            n, m, T, e, dbar = trajectory(n_req, seed)
            s, _ = fit_linear(T, e)
            ss.append(s); ds.append(dbar); es.append(e[-1])
            rows.append([n, m, round(m / n, 2), round(dbar, 4),
                         "{:.4e}".format(s), "{:.4e}".format(e[-1]), seed])
        d_bar = m / n
        dl1 = float(np.mean(ds))
        rho = 4 * C / ((1 - C) * (d_bar + 1)) * (1 + 1 / (d_bar + 1))
        s_med = float(np.median(ss))
        print("  {:>6}{:>9}{:>8.1f}{:>14.4f}{:>10.2f}{:>14.3e}{:>12.2e}"
              .format(n, m, d_bar, dl1, rho, s_med, float(np.median(es))))
        summary.append((n, d_bar, dl1, rho, s_med))

    print("  " + "-" * 74)
    (n1, d1, l1, r1, s1), (n2, d2, l2, r2, s2) = summary[0], summary[-1]
    print("\n  From n={} to n={}:".format(n1, n2))
    print("    mean degree      x{:.2f}".format(d2 / d1))
    print("    ||delta||_1      x{:.2f}   (predicted x{:.2f} from 2/(d+1))"
          .format(l2 / l1, (d1 + 1) / (d2 + 1)))
    print("    drift per step   x{:.2f}".format(s2 / s1))
    print("\n  Prediction: denser graph drifts LESS per update.  Observed: {}"
          .format("YES, x{:.2f}".format(s1 / s2) if s2 < s1
                  else "NO, drift rose by x{:.2f}".format(s2 / s1)))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'drift_density.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["n", "m", "d_bar", "mean_delta_l1", "slope_per_step",
                    "final_error", "seed"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

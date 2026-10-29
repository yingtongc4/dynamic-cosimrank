# -*- coding: utf-8 -*-
"""
bench_adversarial.py
====================
Testing the explanation given for linear drift.

Section "drift policy" finds that error accumulates linearly rather than
geometrically, and explains it by saying the bound "had to assume that each
step's error aligns with the accumulated one, and over a stream of updates
landing in different parts of the graph it evidently does not".  Every stream
measured is random, so that explanation has never been tested against the
case it implicitly excludes.

It has a clean falsifier.  Concentrate the updates instead of spreading them:

  random     insert and delete edges anywhere, as elsewhere in the report
  one column repeatedly change the in-neighbours of a single node
  one hub    change edges that all point at the highest-degree node

If the explanation is right, the concentrated streams should drift faster per
update, and may show the positive growth rate rho that the random stream does
not.  If they drift the same, the linearity has some other cause and the
explanation in the report is wrong.

Usage:
    python benchmarks/bench_adversarial.py
"""

import sys, os, io, csv, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_drift_policy import converged, reorder, fit_linear, fit_geometric

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C = 0.8
K = 20
N_NODES = 200
N_STEPS = 400
CHECK_EVERY = 25
SEEDS = (0, 1)


def op_random(d, rng, target=None):
    nodes = sorted(d.G.nodes())
    if rng.random() < 0.5:
        for _ in range(300):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and not d.G.has_edge(u, v):
                return ('i', u, v)
    es = list(d.G.edges())
    for _ in range(300):
        u, v = es[rng.randrange(len(es))]
        if d.G.in_degree(v) > 1:
            return ('d', u, v)
    return None


def op_focused(d, rng, target):
    """Every change lands on the in-neighbour set of `target`."""
    nodes = sorted(d.G.nodes())
    preds = set(d.G.predecessors(target))
    if rng.random() < 0.5 or d.G.in_degree(target) <= 1:
        for _ in range(300):
            u = rng.choice(nodes)
            if u != target and u not in preds:
                return ('i', u, target)
    for _ in range(300):
        u = rng.choice(sorted(preds))
        if d.G.in_degree(target) > 1:
            return ('d', u, target)
    return None


def trajectory(G, mode, seed, target):
    rng = random.Random(seed)
    S0, _ = converged(G)
    d = DynamicCoSimRank(G.copy(), C=C, K=K, tol=0.0)
    d.S = S0.copy()
    pick = op_random if mode == 'random' else op_focused
    T, E = [], []
    for t in range(1, N_STEPS + 1):
        op = pick(d, rng, target)
        if op is None:
            break
        kind, u, v = op
        if kind == 'i':
            d.edge_insert([(u, v)])
        else:
            d.edge_delete([(u, v)])
        if t % CHECK_EVERY == 0:
            Sr, nr = converged(d.G)
            ref = reorder(Sr, nr, d.nodes)
            E.append(float(np.linalg.norm(d.S - ref, 'fro')
                           / np.linalg.norm(ref, 'fro')))
            T.append(t)
    return np.array(T, float), np.array(E)


def main():
    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=N_NODES).to_directed()
    n, m = G.number_of_nodes(), G.number_of_edges()
    hub = max(G.nodes(), key=lambda v: G.in_degree(v))
    median = sorted(G.nodes(), key=lambda v: G.in_degree(v))[n // 2]

    print("=" * 84)
    print("  Is the drift linear because the updates are spread out?")
    print("  ego-Facebook n={}, m={}, K={}, {} operations, {} seeds"
          .format(n, m, K, N_STEPS, len(SEEDS)))
    print("  hub in-degree {}, median-node in-degree {}"
          .format(G.in_degree(hub), G.in_degree(median)))
    print("=" * 84)
    print("\n  {:<22}{:>14}{:>15}{:>12}{:>12}".format(
        "stream", "final error", "slope/step", "fitted r", "vs random"))
    print("  " + "-" * 76)

    rows, base = [], None
    for label, mode, target in (("random", "random", None),
                                ("one column (median)", "focus", median),
                                ("one column (hub)", "focus", hub)):
        sl, fin, rr = [], [], []
        for seed in SEEDS:
            T, E = trajectory(G, mode, seed, target)
            if len(T) < 3:
                continue
            s, _ = fit_linear(T, E)
            r, a, _ = fit_geometric(T, E)
            sl.append(s); fin.append(E[-1]); rr.append(r)
        s_med = float(np.median(sl))
        if base is None:
            base = s_med
        print("  {:<22}{:>14.3e}{:>15.3e}{:>12.1e}{:>11.2f}x"
              .format(label, float(np.median(fin)), s_med,
                      float(np.median(rr)), s_med / base))
        rows.append([label, "{:.4e}".format(float(np.median(fin))),
                     "{:.4e}".format(s_med), "{:.2e}".format(float(np.median(rr))),
                     round(s_med / base, 4)])

    print("  " + "-" * 76)
    conc = max(r[4] for r in rows[1:])
    any_geom = any(float(r[3]) > 0 for r in rows)
    print("\n  concentrating the stream changes the drift rate by up to {:.2f}x"
          .format(conc))
    print("  any stream fitting a positive growth rate: {}"
          .format("YES" if any_geom else "no, all linear"))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'adversarial.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["stream", "final_error", "slope", "fitted_r",
                    "slope_vs_random"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

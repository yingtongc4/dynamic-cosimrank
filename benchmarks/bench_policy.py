# -*- coding: utf-8 -*-
"""
bench_policy.py
===============
The cost model and the drift model, composed into one operating policy.

The report derives two thresholds and never multiplies them together.  The
cost model says when an update is cheaper than a recomputation; the drift
recursion says how many updates may be chained before a refresh.  A
deployment faces one question containing both: given an accuracy target, what
depth K and what refresh interval minimise total cost per update?

Writing E(K) for the static truncation floor at depth K and s(K) for the
drift per update, both of which scale as C^K,

    cost/update = 8Km + 4Kn|Q|          (the update)
                + 4Km|Q| s(K)/(eps - E(K))   (the refresh, amortised)

which is defined only for K above K_min, the depth at which the truncation
floor alone already exceeds eps.  This script checks the algebra against a
measured stream: 1500 edge updates under four strategies, timed end to end
and with the final error measured against a converged recomputation.

  recompute   never maintain; recompute the |Q| columns after every update
  policy      maintain at K*, refresh every T* updates
  no-refresh  maintain at K*, never refresh
  K=20        maintain at the report's operating point, never refresh

Usage:
    python benchmarks/bench_policy.py
"""

import sys, os, io, csv, time, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from simrank import cosimrank_matrix
from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_perquery import assemble, static_columns_prebuilt, C

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

N_NODES = 200
QSIZE = 20
N_STEPS = 1500
EPS = 1e-3
SEED = 7

# calibration, both measured at K = 20 and both scaling as C^(K-20)
E20 = 4.48e-3        # static truncation floor, Table "depths"
S20 = 1.077e-6       # drift per edge update, Table "drift policy"


def floor_at(K):
    return E20 * C ** (K - 20)


def slope_at(K):
    return S20 * C ** (K - 20)


def optimum(n, m, q, eps):
    """K minimising the composed cost, and the interval it implies."""
    Kmin = 20 + np.log(eps / E20) / np.log(C)
    best = None
    for K in range(int(np.ceil(Kmin)) + 1, 121):
        budget = eps - floor_at(K)
        if budget <= 0:
            continue
        T = budget / slope_at(K)
        cost = K * (8 * m + 4 * n * q) + 4 * K * m * q / T
        if best is None or cost < best[2]:
            best = (K, T, cost)
    return Kmin, best


def converged(G):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=130, tol=0.0)
    return S, nodes


def col_error(pq, G, Q):
    """Relative error of the maintained columns against a converged run."""
    S, nodes = converged(G)
    pos = {v: i for i, v in enumerate(nodes)}
    num = den = 0.0
    for q in Q:
        want = np.array([S[q][pos[x]] for x in pq.nodes])
        got = pq.column(q)
        num += float(((got - want) ** 2).sum())
        den += float((want ** 2).sum())
    return (num ** 0.5) / (den ** 0.5)


def make_ops(G, rng, count):
    """A fixed stream of edge insertions and deletions, replayed by everyone."""
    H = G.copy()
    ops = []
    nodes = sorted(H.nodes())
    while len(ops) < count:
        if rng.random() < 0.5:
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and not H.has_edge(u, v):
                H.add_edge(u, v); ops.append(('i', u, v)); continue
        es = list(H.edges())
        u, v = es[rng.randrange(len(es))]
        if H.in_degree(v) > 1:
            H.remove_edge(u, v); ops.append(('d', u, v))
    return ops


def run_maintained(G, Q, ops, K, refresh_every):
    pq = PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K, tol=0.0)
    H = G.copy()
    t0 = time.perf_counter()
    n_ref = 0
    for i, (kind, u, v) in enumerate(ops, 1):
        if kind == 'i':
            pq.edge_insert([(u, v)]); H.add_edge(u, v)
        else:
            pq.edge_delete([(u, v)]); H.remove_edge(u, v)
        if refresh_every and i % refresh_every == 0:
            pq = PerQueryCoSimRank(H.copy(), queries=list(Q), C=C, K=K,
                                   tol=0.0)
            n_ref += 1
    return (time.perf_counter() - t0), pq, H, n_ref


def run_recompute(G, Q, ops, K):
    H = G.copy()
    t0 = time.perf_counter()
    out = None
    for kind, u, v in ops:
        if kind == 'i':
            H.add_edge(u, v)
        else:
            H.remove_edge(u, v)
        A, AT, nodes, idx = assemble(H)
        out = static_columns_prebuilt(A, AT, [idx[q] for q in Q], len(nodes),
                                      C, K)
    return (time.perf_counter() - t0), out, H


def main():
    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=N_NODES).to_directed()
    n, m = G.number_of_nodes(), G.number_of_edges()
    rng = random.Random(SEED)
    Q = rng.sample(sorted(G.nodes()), QSIZE)
    ops = make_ops(G, rng, N_STEPS)

    Kmin, (Kstar, Tstar, cost) = optimum(n, m, QSIZE, EPS)
    print("=" * 84)
    print("  A composed operating policy")
    print("  ego-Facebook n={}, m={}, |Q|={}, target eps={:.0e}, {} updates"
          .format(n, m, QSIZE, EPS, N_STEPS))
    print("=" * 84)
    print("\n  truncation floor alone reaches eps at K_min = {:.1f}".format(Kmin))
    print("  cost-optimal depth                     K*    = {}".format(Kstar))
    print("  refresh interval it implies            T*    = {:.0f} updates"
          .format(Tstar))
    print("  floor at K*: {:.2e}   drift budget: {:.2e}   drift/update: {:.2e}"
          .format(floor_at(Kstar), EPS - floor_at(Kstar), slope_at(Kstar)))
    upd = Kstar * (8 * m + 4 * n * QSIZE)
    ref = 4 * Kstar * m * QSIZE / Tstar
    print("  predicted split: update {:.3g} flops, refresh {:.3g}"
          " ({:.2f}% of total)".format(upd, ref, 100 * ref / (upd + ref)))

    print("\n  {:<14}{:>8}{:>10}{:>12}{:>12}{:>10}".format(
        "strategy", "K", "refresh", "total (s)", "final err", "verdict"))
    print("  " + "-" * 68)

    rows = []
    ref_int = int(Tstar) if Tstar < N_STEPS else 0

    t, out, H = run_recompute(G, Q, ops, Kstar)
    S, nodes = converged(H)
    pos = {v: i for i, v in enumerate(nodes)}
    nodes_after = sorted(H.nodes())
    num = den = 0.0
    for j, q in enumerate(Q):
        want = np.array([S[q][pos[x]] for x in nodes_after])
        num += float(((out[j] - want) ** 2).sum())
        den += float((want ** 2).sum())
    e = (num ** 0.5) / (den ** 0.5)
    print("  {:<14}{:>8}{:>10}{:>12.2f}{:>12.2e}{:>10}".format(
        "recompute", Kstar, "n/a", t, e, "ok" if e <= EPS else "OVER"))
    rows.append(["recompute", Kstar, "", round(t, 3), "{:.3e}".format(e),
                 e <= EPS])

    # each maintained strategy is run REPEATS times: a single run of a
    # 1500-update stream carries enough variance to reverse the ordering of
    # two strategies that differ by one refresh
    REPEATS = 3
    for label, K, every in [("policy", Kstar, ref_int),
                            ("no refresh", Kstar, 0),
                            ("K=20", 20, 0)]:
        ts = []
        for _ in range(REPEATS):
            t, pq, H, n_ref = run_maintained(G, Q, ops, K, every)
            ts.append(t)
        e = col_error(pq, H, Q)
        t_med = float(np.median(ts))
        print("  {:<14}{:>8}{:>10}{:>12.2f}{:>12.2e}{:>10}   [{:.2f}-{:.2f}]"
              .format(label, K, n_ref if every else "none", t_med, e,
                      "ok" if e <= EPS else "OVER", min(ts), max(ts)))
        rows.append([label, K, n_ref, round(t_med, 3), "{:.3e}".format(e),
                     e <= EPS, round(min(ts), 3), round(max(ts), 3)])

    print("  " + "-" * 68)
    rec = rows[0]
    pol = rows[1]
    print("\n  policy against recompute-every-update: {:.1f}x faster, both"
          " within eps".format(rec[3] / pol[3]))
    print("  refreshes performed by the policy    : {} in {} updates"
          .format(pol[2], N_STEPS))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'policy.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["strategy", "K", "refreshes", "total_s", "final_error",
                    "within_eps", "min_s", "max_s"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["n", "m", "Q", "eps", "K_min", "K_star", "T_star"])
        w.writerow([n, m, QSIZE, EPS, round(Kmin, 2), Kstar, round(Tstar)])
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

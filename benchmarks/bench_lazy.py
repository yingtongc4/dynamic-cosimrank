# -*- coding: utf-8 -*-
"""
bench_lazy.py
=============
Why maintain at all?  The obvious alternative, and where it wins.

Nothing in this report forces a service to maintain S.  A third strategy is
available and is not obviously worse: keep the graph up to date, maintain
NOTHING, and recompute a single column on demand when a query arrives -- 2K
matrix-vector products, the cost of one column of the static baseline.  Since
the stream study shows maintenance drifts without periodic refresh and lazy
evaluation cannot drift at all, the lazy strategy has to be argued against
rather than ignored.

The comparison is per update window -- the interval between two consecutive
graph changes -- because that is what invalidates cached columns:

    maintain :  8Km + 4Kn|Q|      flops, once, covering all |Q| columns
    lazy     :  d * 4Km           flops, where d is the number of DISTINCT
                                  maintained users who actually query in the
                                  window (a cache makes repeats free)

so maintaining is cheaper exactly when

    d  >  2 + n|Q|/m.

The threshold is on the query-to-update ratio, not on graph size, and it is
small.  This script measures both sides -- one dynamic update, and one static
column recomputation -- and checks the measured d* against that prediction.

Usage:
    python benchmarks/bench_lazy.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            subgraph, C, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
CONFIGS = [(200, 5), (200, 20), (1000, 5), (1000, 20), (1000, 50)]


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def main():
    print("=" * 84)
    print("  Maintain, or recompute on demand?")
    print("  d* is the number of distinct queries per update above which")
    print("  maintaining is cheaper.  K = {}, median of {} runs."
          .format(K_DEFAULT, REPS))
    print("=" * 84)
    print("\n  {:>6}{:>6}{:>11}{:>13}{:>11}{:>11}{:>10}".format(
        "n", "|Q|", "update ms", "1 column ms", "measured", "predicted",
        "gate"))
    print("  {:>6}{:>6}{:>11}{:>13}{:>11}{:>11}{:>10}".format(
        "", "", "(maintain)", "(lazy)", "d*", "2+n|Q|/m", ""))
    print("  " + "-" * 74)

    rows = []
    for n_req, qsize in CONFIGS:
        G = subgraph('facebook', n_req)
        n, m = G.number_of_nodes(), G.number_of_edges()
        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), qsize)
        u, v = insertable_edge(G, rng)

        G_after = G.copy(); G_after.add_edge(u, v)
        worst = gate(lambda qs: _apply(G, qs, u, v), G_after, Q, rng,
                     "n={} |Q|={}".format(n, qsize))

        # --- maintain: one update, covering all |Q| columns ---
        pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                                 tol=0.0) for _ in range(REPS)]
        st = {'i': 0}

        def go():
            p = pre[st['i']]; st['i'] += 1
            p.edge_insert([(u, v)])

        upd = timed(go, REPS)[0]

        # --- lazy: one column, recomputed on demand from the live matrix ---
        A, AT, nodes2, idx2 = assemble(G_after)
        one = [idx2[Q[0]]]
        col = timed(lambda: static_columns_prebuilt(A, AT, one, len(nodes2),
                                                    C, K_DEFAULT), REPS)[0]

        d_meas = upd / col
        d_pred = 2.0 + n * qsize / m
        print("  {:>6}{:>6}{:>11.2f}{:>13.2f}{:>11.2f}{:>11.2f}{:>10.1e}"
              .format(n, qsize, upd, col, d_meas, d_pred, worst))
        rows.append([n, m, qsize, round(upd, 3), round(col, 3),
                     round(d_meas, 3), round(d_pred, 3),
                     round(d_meas / d_pred, 3), "{:.2e}".format(worst)])

    print("  " + "-" * 74)
    r = [x[5] / x[6] for x in rows]
    print("  measured/predicted d*: {:.2f} to {:.2f}".format(min(r), max(r)))
    print("\n  Reading: at n=1000, |Q|=50 the maintained copy pays for itself as")
    print("  soon as roughly {:.0f} distinct users query between two consecutive"
          .format(rows[-1][5]))
    print("  graph changes.  Below that, recompute on demand and accept no drift.")

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'lazy.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["n", "m", "Q", "update_ms", "one_column_ms", "d_measured",
                    "d_predicted", "ratio", "gate_err"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _apply(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
bench_scale.py
==============
The same follow, on the whole of ego-Facebook.

Every timed result elsewhere in this report stops at n = 1000 and uses the
subgraph recipe of data/dataset_stats.py -- degree filter, largest component,
BFS from the highest-degree node -- which the report itself describes as
biased towards a dense hub-centred region.  Two objections follow from that
and are answered here rather than argued away.

  (i)  A claim about evolving-graph scalability that stops at a thousand nodes
       is not a scalability claim.  The last row below is the FULL ego-Facebook
       graph, n = 4039, m = 176,468 directed edges, with no extraction at all.

  (ii) If the conclusions are an artefact of the extraction, they should change
       when the extraction is removed.  The full-graph row shares no step of
       that pipeline with the rows above it, so agreement across the boundary
       is evidence the recipe is not what produced the result.

A first version of this script used one edge and five repetitions per size and
produced a non-monotone dynamic cost -- n = 3000 apparently cheaper than
n = 2000 -- which is impossible on merit and was noise.  Each size is now
measured on EDGES_PER_SIZE independent insertions of ten repetitions each, and
the median across them is reported together with the spread, so that a reader
can see how much of any trend is real.

|Q| = 50 and K = 20 throughout, gated against a converged static recomputation
before timing.

Usage:
    python benchmarks/bench_scale.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            C, K_DEFAULT)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

QSIZE = 50
SEED = 42
REPS = 10
EDGES_PER_SIZE = 3
SIZES = (1000, 2000, 3000)


def graphs():
    """The extracted sizes, then the whole graph with no extraction."""
    full = load_facebook()
    for n in SIZES:
        yield ("subgraph", extract_subgraph(full, min_degree=10,
                                            max_nodes=n).to_directed())
    yield ("FULL GRAPH", full.to_directed())


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def one_edge(G, Q, u, v, rng, label):
    G_after = G.copy()
    G_after.add_edge(u, v)

    worst = gate(lambda qs: _apply(G, qs, u, v), G_after, Q, rng, label)

    pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                             tol=0.0) for _ in range(REPS)]
    st = {'i': 0}

    def go():
        p = pre[st['i']]; st['i'] += 1
        p.edge_insert([(u, v)])

    dyn = timed(go, REPS)[0]

    A, AT, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                 C, K_DEFAULT), REPS)[0]
    return dyn, stat, worst


def main():
    print("=" * 88)
    print("  One follow, from a thousand nodes to the whole graph")
    print("  |Q| = {}, K = {}, {} independent edges x {} repetitions per size"
          .format(QSIZE, K_DEFAULT, EDGES_PER_SIZE, REPS))
    print("=" * 88)
    print("\n  {:<12}{:>6}{:>9}{:>8}{:>10}{:>11}{:>10}{:>10}{:>9}{:>10}".format(
        "source", "n", "m", "n/m", "dyn (ms)", "static (ms)", "measured",
        "spread", "predicted", "over"))
    print("  " + "-" * 96)

    rows = []
    for label, G in graphs():
        n, m = G.number_of_nodes(), G.number_of_edges()
        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), QSIZE)

        ds, ss, sp, gates = [], [], [], []
        for e in range(EDGES_PER_SIZE):
            erng = random.Random(SEED + 100 * e)
            u, v = insertable_edge(G, erng)
            d, s, w = one_edge(G, Q, u, v, erng,
                               "{} n={} edge{}".format(label, n, e))
            ds.append(d); ss.append(s); sp.append(s / d); gates.append(w)

        d_med, s_med = float(np.median(ds)), float(np.median(ss))
        meas = float(np.median(sp))
        pred = 1.0 / (2.0 / QSIZE + n / m)
        print("  {:<12}{:>6}{:>9}{:>8.4f}{:>10.2f}{:>11.2f}{:>9.2f}x"
              "{:>8.2f}-{:.2f}{:>8.2f}x{:>9.2f}x"
              .format(label, n, m, n / m, d_med, s_med, meas,
                      min(sp), max(sp), pred, pred / meas))
        rows.append([label, n, m, round(n / m, 5), round(d_med, 3),
                     round(s_med, 3), round(meas, 4), round(min(sp), 4),
                     round(max(sp), 4), round(pred, 4),
                     round(pred / meas, 4), "{:.2e}".format(max(gates))])

    print("  " + "-" * 96)
    print("\n  Dynamic cost is monotone in m: {}".format(
        all(rows[i][4] <= rows[i + 1][4] for i in range(len(rows) - 1))))
    print("  Speedup on the FULL graph, no extraction : {:.2f}x"
          .format(rows[-1][6]))
    print("  Speedup on the n=1000 extracted subgraph : {:.2f}x"
          .format(rows[0][6]))
    over = [r[10] for r in rows]
    print("  Model overstatement across the range     : {:.2f}x to {:.2f}x"
          .format(min(over), max(over)))
    print("  All gates passed, worst {}".format(max(r[11] for r in rows)))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'scale.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["source", "n", "m", "n_over_m", "dyn_ms", "static_ms",
                    "measured", "min", "max", "predicted", "overstatement",
                    "gate_err"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _apply(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

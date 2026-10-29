# -*- coding: utf-8 -*-
"""
bench_nodeins.py
================
Node insertion, which is the one operation with no timing sweep of its own.

E1 to E6 cover query-set size, graph size, edge batch size, dataset, node
deletion degree and depth.  Node insertion appears only as ten per cent of a
mixed stream and as one point of Example D, so of the four operations the
report claims to cover, three have timing curves and one does not.

Equation (14) makes a prediction that costs nothing to test:

    b_ins(w) = 1[N_in(w) nonempty] + |N_out(w)|,

so the cost of inserting a node should be flat in its in-degree and linear in
its out-degree.  The two sweeps below vary one and hold the other at zero.

Usage:
    python benchmarks/bench_nodeins.py
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
N_NODES = 1000
QSIZE = 5
DEGREES = (1, 2, 4, 8, 16)


def edges_for(G, w, rng, n_in, n_out):
    """n_in edges into w and n_out edges out of w, all endpoints distinct."""
    pool = rng.sample(sorted(G.nodes()), n_in + n_out)
    ins = [(pool[i], w) for i in range(n_in)]
    outs = [(w, pool[n_in + i]) for i in range(n_out)]
    return ins + outs


def measure(G, Q, w, edges, rng, label):
    G_after = G.copy()
    G_after.add_node(w)
    G_after.add_edges_from(edges)

    gate(lambda qs: _probe(G, qs, w, edges), G_after, Q, rng, label)

    pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                             tol=0.0) for _ in range(REPS)]
    st = {'i': 0}

    def go():
        p = pre[st['i']]; st['i'] += 1
        p.node_insert(w, edges)

    dyn = timed(go, REPS)[0]

    A, AT, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                 C, K_DEFAULT), REPS)[0]
    return dyn, stat


def main():
    G = subgraph('facebook', N_NODES)
    n, m = G.number_of_nodes(), G.number_of_edges()
    w = max(G.nodes()) + 1
    print("=" * 82)
    print("  Node insertion.  ego-Facebook n={}, m={}, |Q|={}, K={},"
          " median of {} runs.".format(n, m, QSIZE, K_DEFAULT, REPS))
    print("  Equation (14): b = 1[in-edges] + |N_out|, so cost should be flat")
    print("  in the in-degree and linear in the out-degree.")
    print("=" * 82)

    rows = []
    for sweep, fixed in (("in-degree", "out"), ("out-degree", "in")):
        print("\n  varying the {} ({}-degree held at 0)".format(sweep, fixed))
        print("  {:>8}{:>6}{:>11}{:>13}{:>11}".format(
            "degree", "b", "dyn (ms)", "static (ms)", "speedup"))
        print("  " + "-" * 50)
        base = None
        for d in DEGREES:
            rng = random.Random(SEED + d)
            Q = rng.sample(sorted(G.nodes()), QSIZE)
            n_in, n_out = (d, 0) if sweep == "in-degree" else (0, d)
            edges = edges_for(G, w, rng, n_in, n_out)
            b = (1 if n_in else 0) + n_out
            dyn, stat = measure(G, Q, w, edges, rng,
                                "{} {}".format(sweep, d))
            if base is None:
                base = dyn
            print("  {:>8}{:>6}{:>11.2f}{:>13.2f}{:>10.2f}x".format(
                d, b, dyn, stat, stat / dyn))
            rows.append([sweep, d, b, round(dyn, 3), round(stat, 3),
                         round(stat / dyn, 4), round(dyn / base, 3)])

    ins = [r for r in rows if r[0] == "in-degree"]
    outs = [r for r in rows if r[0] == "out-degree"]
    print("\n  " + "=" * 74)
    print("  in-degree sweep : cost x{:.2f} as the degree goes {} -> {}"
          " (prediction: flat)".format(ins[-1][3] / ins[0][3],
                                       ins[0][1], ins[-1][1]))
    print("  out-degree sweep: cost x{:.2f} as b goes {} -> {}"
          " (prediction: x{:.1f}, linear in b)"
          .format(outs[-1][3] / outs[0][3], outs[0][2], outs[-1][2],
                  outs[-1][2] / outs[0][2]))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'nodeins.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["sweep", "degree", "b", "dyn_ms", "static_ms",
                       "speedup", "cost_relative_to_first"])
        wcsv.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _probe(G, qs, w, edges):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.node_insert(w, edges)
    return pq


if __name__ == "__main__":
    main()

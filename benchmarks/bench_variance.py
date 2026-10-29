# -*- coding: utf-8 -*-
"""
bench_variance.py
=================
How much does the measured speedup move when only the inserted edge and the
query set change?

Two tables in the report quote the same configuration -- planted graph,
n = 200, |Q| = 20, one local edge -- and give 3.86x in one place and
2.63x/2.67x in another.  The scenario table and the scaling sweep draw their
edge and query set from differently-advanced random streams, so they are not
the same trial.  Rather than assert that the gap is sampling, this measures
the spread directly over independent draws.

Usage:
    python benchmarks/bench_variance.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from per_query import PerQueryCoSimRank
from worked_example_advantage import two_communities, free_edge_in
from bench_perquery import assemble, static_columns_prebuilt, timed, C, K_DEFAULT

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

N = 200
QSIZE = 20
REPS = 10
DRAWS = 12


def one_draw(G, A, B, seed):
    """Time one (edge, query set) draw; return the speedup."""
    rng = random.Random(seed)
    edge = free_edge_in(G, A, rng)
    Q = rng.sample(B, QSIZE)

    G_after = G.copy(); G_after.add_edge(*edge)

    pre = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
           for _ in range(REPS)]
    st = {'i': 0}

    def run():
        p = pre[st['i']]; st['i'] += 1
        p.edge_insert([edge])

    dyn = timed(run, REPS)

    A2, AT2, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(
        A2, AT2, qidx, len(nodes2), C, K_DEFAULT), REPS)

    return edge, stat[0] / dyn[0], dyn[0], stat[0]


def main():
    G, A, B = two_communities(N)
    m = G.number_of_edges()
    print("=" * 70)
    print("  Run-to-run spread at a single configuration")
    print("  planted graph, n={}, m={}, |Q|={}, K={}, one local edge"
          .format(N, m, QSIZE, K_DEFAULT))
    print("  {} independent draws of (edge, query set)".format(DRAWS))
    print("=" * 70)
    print("\n  {:>5}  {:>12}  {:>10}  {:>10}  {:>9}".format(
        "draw", "edge", "dyn (ms)", "stat (ms)", "speedup"))
    print("  " + "-" * 54)

    rows, sp = [], []
    for d in range(DRAWS):
        edge, s, dyn_ms, stat_ms = one_draw(G, A, B, seed=1000 + d)
        print("  {:>5}  {:>12}  {:>10.2f}  {:>10.2f}  {:>8.2f}x".format(
            d, "{}->{}".format(edge[0], edge[1]), dyn_ms, stat_ms, s))
        rows.append([d, "{}->{}".format(edge[0], edge[1]),
                     round(dyn_ms, 3), round(stat_ms, 3), round(s, 4)])
        sp.append(s)

    a = np.array(sp)
    pred = 1.0 / (2.0 / QSIZE + N / m)
    print("\n  " + "-" * 54)
    print("  median      : {:.2f}x".format(np.median(a)))
    print("  min / max   : {:.2f}x / {:.2f}x".format(a.min(), a.max()))
    print("  IQR         : {:.2f}x -- {:.2f}x".format(
        np.percentile(a, 25), np.percentile(a, 75)))
    print("  spread      : {:.1f}x between the extremes".format(a.max() / a.min()))
    print("  predicted   : {:.2f}x   (2/|Q| + n/m)".format(pred))
    print(chr(10) + "  The two figures quoted in the report:")
    for v in (2.63, 3.86):
        inside = a.min() <= v <= a.max()
        print("    {:.2f}x  {}".format(
            v, "within the observed range" if inside
               else "OUTSIDE [{:.2f}, {:.2f}]".format(a.min(), a.max())))
    print("  model overstatement against the median: {:.2f}x".format(
        pred / np.median(a)))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'variance_n200.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["draw", "edge", "dyn_ms", "static_ms", "speedup"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["median", round(float(np.median(a)), 4)])
        w.writerow(["min", round(float(a.min()), 4)])
        w.writerow(["max", round(float(a.max()), 4)])
        w.writerow(["predicted", round(pred, 4)])
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

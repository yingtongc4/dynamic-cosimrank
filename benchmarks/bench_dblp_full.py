# -*- coding: utf-8 -*-
"""
bench_dblp_full.py
==================
The second dataset, at full size, with no subgraph extraction.

Section "scaling off the subgraph recipe" removed the extraction for
ego-Facebook and found the conclusion unchanged, leaving dataset independence
resting on subgraphs at one matched size.  The tool for closing that was
already in hand: the per-query mode holds O(m + n|Q|) and never allocates an
n x n array, so com-DBLP at its full 317,080 nodes is a question of a few
hundred megabytes rather than of 800 GB.

n = 317,080, m = 2,099,732 directed edges, |Q| = 50, K = 20.  The maintained
columns alone are 50 x 317,080 x 8 bytes = 127 MB, which is why this could not
be done in full-matrix mode and can be done here.

The correctness gate runs at K = 130 on three columns, as everywhere else.

Usage:
    python benchmarks/bench_dblp_full.py
"""

import sys, os, csv, gc, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from dataset_stats import load_dblp_graph
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            C, K_DEFAULT)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
QSIZE = 50
Q_SWEEP = (1, 2, 3, 5, 10, 20, 50)
REPS = 3            # each repetition builds a fresh object at n = 317k
MB = 1024.0 * 1024.0


def insertable_edge(G, rng):
    nodes = list(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def main():
    t0 = time.perf_counter()
    print("=" * 84)
    print("  com-DBLP at full size, no subgraph extraction")
    print("=" * 84)

    G = load_dblp_graph().to_directed()
    n, m = G.number_of_nodes(), G.number_of_edges()
    print("  n = {:,}   m = {:,} directed edges   mean degree {:.2f}"
          .format(n, m, m / n))
    print("  loaded in {:.1f} s".format(time.perf_counter() - t0))

    rng = random.Random(SEED)
    Q = rng.sample(sorted(G.nodes()), QSIZE)
    u, v = insertable_edge(G, rng)
    G_after = G.copy(); G_after.add_edge(u, v)
    print("  inserting {} -> {}, |Q| = {}, K = {}".format(u, v, QSIZE,
                                                          K_DEFAULT))

    print("\n  correctness gate at K=130 on three columns...")
    t = time.perf_counter()
    worst = gate(lambda qs: _probe(G, qs, u, v), G_after, Q, rng, "com-DBLP full")
    print("    passed, worst relative column error {:.2e}  ({:.0f} s)"
          .format(worst, time.perf_counter() - t))

    A, AT, nodes2, idx2 = assemble(G_after)

    dbar = m / n
    qstar = 2 * dbar / (dbar - 1)
    print("\n  |Q| sweep.  d-bar = {:.2f} gives |Q|* = {:.3f}, the largest"
          .format(dbar, qstar))
    print("  threshold of any graph here and where the formula is most sensitive.")
    print("\n  {:>5}{:>12}{:>14}{:>11}{:>12}{:>10}".format(
        "|Q|", "dyn (ms)", "static (ms)", "measured", "predicted", "verdict"))
    print("  " + "-" * 66)

    sweep = []
    resident = 0.0
    for q in Q_SWEEP:
        rq = random.Random(SEED + q)
        Qs = rq.sample(sorted(G.nodes()), q)
        pre = [PerQueryCoSimRank(G.copy(), queries=list(Qs), C=C,
                                 K=K_DEFAULT, tol=0.0)
               for _ in range(REPS + 1)]
        resident = max(resident,
                       (pre[0].AT.data.nbytes + pre[0].AT.indices.nbytes
                        + pre[0].AT.indptr.nbytes
                        + sum(c.nbytes for c in pre[0].cols.values())) / MB)
        st = {'i': 0}

        def go():
            p = pre[st['i']]; st['i'] += 1
            p.edge_insert([(u, v)])

        go()                                   # warm-up, discarded
        ts = []
        for _ in range(REPS):
            tb = time.perf_counter(); go()
            ts.append((time.perf_counter() - tb) * 1000.0)
        dq = float(np.median(ts))
        del pre
        gc.collect()

        qi = [idx2[x] for x in Qs]
        static_columns_prebuilt(A, AT, qi, len(nodes2), C, K_DEFAULT)
        sq = timed(lambda: static_columns_prebuilt(A, AT, qi, len(nodes2),
                                                   C, K_DEFAULT), REPS)[0]
        meas, pred = sq / dq, 1.0 / (2.0 / q + n / m)
        print("  {:>5}{:>12.1f}{:>14.1f}{:>10.2f}x{:>11.2f}x{:>10}".format(
            q, dq, sq, meas, pred, "win" if meas > 1 else "lose"))
        sweep.append([q, round(dq, 2), round(sq, 2), round(meas, 4),
                      round(pred, 4), "win" if meas > 1 else "lose"])

    wins = [r[0] for r in sweep if r[3] > 1]
    print("  " + "-" * 66)
    print("  measured: wins from |Q| = {}".format(min(wins) if wins else "never"))
    print("  predicted: |Q|* = {:.3f}, so from |Q| = {}"
          .format(qstar, int(np.floor(qstar)) + 1))
    print("  agree on the integer: {}"
          .format(bool(wins) and min(wins) == int(np.floor(qstar)) + 1))

    d, s = sweep[-1][1], sweep[-1][2]

    model = 1.0 / (2.0 / QSIZE + n / m)
    print("\n  {:<26}{:>14}".format("dynamic update", "{:.1f} ms".format(d)))
    print("  {:<26}{:>14}".format("static, {} columns".format(QSIZE),
                                  "{:.1f} ms".format(s)))
    print("  {:<26}{:>13.2f}x".format("measured speedup", s / d))
    print("  {:<26}{:>13.2f}x".format("predicted (2/|Q|+n/m)", model))
    print("  {:<26}{:>13.2f}x".format("overstatement", model / (s / d)))
    print("  {:<26}{:>11.1f} MiB".format("resident (per-query)", resident))
    print("  {:<26}{:>11.1f} MiB".format("dense n^2 would be",
                                         8.0 * n * n / MB))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'dblp_full.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Q", "dyn_ms", "static_ms", "measured", "predicted",
                    "verdict"])
        w.writerows(sweep)
        w.writerow([])
        w.writerow(["dataset", "n", "m", "Q", "K", "dyn_ms", "static_ms",
                    "measured", "predicted", "overstatement", "resident_MiB",
                    "gate_err"])
        w.writerow(["com-DBLP full", n, m, QSIZE, K_DEFAULT, round(d, 2),
                    round(s, 2), round(s / d, 4), round(model, 4),
                    round(model / (s / d), 4), round(resident, 1),
                    "{:.2e}".format(worst)])
    print("\n  total wall clock {:.0f} s".format(time.perf_counter() - t0))
    print("  -> {}".format(os.path.relpath(p)))


def _probe(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

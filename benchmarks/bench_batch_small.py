# -*- coding: utf-8 -*-
"""
bench_batch_small.py
====================
The batch model's sharpest prediction, tested where it is sharp.

E3 swept batch sizes 1, 5, 10 and 50 and so jumped straight over the range the
model actually makes a falsifiable claim about.  With

    ratio(b) = b * (2/|Q| + n/m),      break-even  b* = (2/|Q| + n/m)^-1,

ego-Facebook at n=1000 (m=55,396, n/m=0.0181) gives b* = 2.39 at |Q|=5 and
b* = 8.47 at |Q|=20.  Both crossings lie inside b <= 10, and neither was
measured.  This script measures b = 1..6 at |Q| = 5 and 20, and b = 1..12 at
|Q| = 20, so that the predicted crossing can be compared with the observed one
rather than assumed.

Each batch is b edges with DISTINCT destinations, so b is exactly the number of
column updates -- edges sharing a destination would collapse into one.

Every configuration is gated against a converged static recomputation before
it is timed.

Usage:
    python benchmarks/bench_batch_small.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            subgraph, C, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

N_NODES = 1000
SEED = 42
B_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12)
Q_SIZES = (5, 20)


def distinct_target_edges(G, b, Q, rng):
    """b insertable edges with b distinct destinations -> exactly b columns."""
    nodes = sorted(G.nodes())
    edges, used = [], set()
    tries = 0
    while len(edges) < b and tries < 50000:
        tries += 1
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u == v or v in used or G.has_edge(u, v):
            continue
        used.add(v)
        edges.append((u, v))
    if len(edges) < b:
        raise RuntimeError("could not find {} distinct-target edges".format(b))
    return edges


def run(G, Q, edges, rng, label):
    G_after = G.copy()
    G_after.add_edges_from(edges)

    worst = gate(lambda qs: _apply(G, qs, edges), G_after, Q, rng, label)

    pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                             tol=0.0) for _ in range(REPS)]
    st = {'i': 0}

    def go():
        p = pre[st['i']]; st['i'] += 1
        p.edge_insert(edges)

    dyn = timed(go, REPS)

    A, AT, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                 C, K_DEFAULT), REPS)
    return dyn[0], stat[0], stat[0] / dyn[0], worst


def _apply(G, qs, edges):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert(edges)
    return pq


def main():
    G = subgraph('facebook', N_NODES)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print("=" * 80)
    print("  Batch size where the model changes its mind")
    print("  ego-Facebook n = {}, m = {}, n/m = {:.4f}, K = {}"
          .format(n, m, n / m, K_DEFAULT))
    print("=" * 80)

    rows = []
    for qsize in Q_SIZES:
        unit = 2.0 / qsize + n / m
        bstar = 1.0 / unit
        print("\n  |Q| = {}:  unit ratio {:.4f},  predicted break-even b* = {:.2f}"
              .format(qsize, unit, bstar))
        print("  {:>3}{:>11}{:>12}{:>12}{:>12}{:>11}{:>11}".format(
            "b", "dyn (ms)", "static (ms)", "measured", "predicted",
            "over", "gate"))
        print("  " + "-" * 72)

        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), qsize)
        prev_win = None
        crossing = None
        for b in B_VALUES:
            if qsize == 5 and b > 6:
                continue
            erng = random.Random(SEED + b)
            edges = distinct_target_edges(G, b, Q, erng)
            dyn, stat, meas, worst = run(G, Q, edges, erng,
                                         "|Q|={} b={}".format(qsize, b))
            pred = 1.0 / (b * unit)
            win = meas > 1.0
            if prev_win is True and win is False and crossing is None:
                crossing = b
            prev_win = win
            print("  {:>3}{:>11.2f}{:>12.2f}{:>11.2f}x{:>11.2f}x{:>10.2f}x{:>11.1e}"
                  .format(b, dyn, stat, meas, pred, pred / meas, worst))
            rows.append([qsize, b, round(dyn, 3), round(stat, 3),
                         round(meas, 4), round(pred, 4),
                         round(pred / meas, 3), "{:.2e}".format(worst)])

        print("  " + "-" * 72)
        print("  predicted crossing between b = {} and {};  measured crossing {}"
              .format(int(bstar), int(bstar) + 1,
                      "between b = {} and {}".format(crossing - 1, crossing)
                      if crossing else "not reached in this range"))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'batch_small.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Q", "b", "dyn_ms", "static_ms", "measured", "predicted",
                    "overstatement", "gate_err"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

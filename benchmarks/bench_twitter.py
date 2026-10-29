# -*- coding: utf-8 -*-
"""
bench_twitter.py
================
The motivating operation, on a graph that actually records it.

Every other experiment in this report inserts a single directed edge into a
graph built by symmetrising undirected friendships or co-authorships.  That
operation corresponds to no real user action: a Facebook friendship is two
reciprocal edges, and ego-Facebook has no notion of a one-way follow at all.

SNAP's ego-Twitter is directed as distributed.  "u follows v" is what its
edges mean, and inserting one is exactly the update the introduction is about,
with b = 1 by construction rather than by assumption.  Only 48.2 per cent of
its edges are reciprocated, so the direction carries information.

  n = 81,306    m = 1,768,149 directed edges    d-bar = 21.75

The cost model predicts the break-even at

  |Q|* = 2 d-bar / (d-bar - 1) = 2.096,

so a single follow should be cheaper to repair than to recompute from the
third query onwards, and more expensive for one or two.  That is a sharp
prediction on a graph twenty times larger than the one it was calibrated on,
and this script tests it directly by sweeping |Q| across the crossing.

Usage:
    python benchmarks/bench_twitter.py
"""

import sys, os, csv, gc, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from dataset_stats import load_twitter
from bench_perquery import (assemble, static_columns_prebuilt, gate,
                            C, K_DEFAULT)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
Q_SIZES = (1, 2, 3, 5, 10, 20, 50)
REPS = 5
MB = 1024.0 * 1024.0


def timed_n(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


def insertable_follow(G, rng):
    """A follow that does not already exist, between two active accounts."""
    nodes = [v for v in G if G.out_degree(v) > 0]
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable follow")


def main():
    t0 = time.perf_counter()
    print("=" * 86)
    print("  ego-Twitter: one follow, on a graph whose edges are follows")
    print("=" * 86)

    G = load_twitter()
    n, m = G.number_of_nodes(), G.number_of_edges()
    dbar = m / n
    qstar = 2 * dbar / (dbar - 1)
    print("  n = {:,}   m = {:,} directed edges   d-bar = {:.2f}"
          .format(n, m, dbar))
    print("  predicted break-even |Q|* = 2 d/(d-1) = {:.3f}"
          "  -> smallest winning integer |Q| is {}"
          .format(qstar, int(np.floor(qstar)) + 1))
    print("  loaded in {:.0f} s".format(time.perf_counter() - t0))

    rng = random.Random(SEED)
    u, v = insertable_follow(G, rng)
    G_after = G.copy(); G_after.add_edge(u, v)
    print("  inserting the follow {} -> {}, K = {}, median of {} runs"
          .format(u, v, K_DEFAULT, REPS))

    print("\n  correctness gate at K=130 on three columns ...", end=" ")
    sys.stdout.flush()
    t = time.perf_counter()
    Qg = rng.sample(sorted(G.nodes()), 8)
    worst = gate(lambda qs: _probe(G, qs, u, v), G_after, Qg, rng, "twitter")
    print("passed, worst {:.2e}  ({:.0f} s)".format(worst,
                                                    time.perf_counter() - t))

    A, AT, nodes2, idx2 = assemble(G_after)

    print("\n  {:>5}{:>12}{:>13}{:>11}{:>12}{:>11}".format(
        "|Q|", "dyn (ms)", "static (ms)", "measured", "predicted", "verdict"))
    print("  " + "-" * 64)

    rows = []
    for q in Q_SIZES:
        rq = random.Random(SEED + q)
        Q = rq.sample(sorted(G.nodes()), q)

        pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                                 tol=0.0) for _ in range(REPS + 1)]
        resident = (pre[0].AT.data.nbytes + pre[0].AT.indices.nbytes
                    + pre[0].AT.indptr.nbytes
                    + sum(c.nbytes for c in pre[0].cols.values())) / MB
        st = {'i': 0}

        def go():
            p = pre[st['i']]; st['i'] += 1
            p.edge_insert([(u, v)])

        go()                      # warm-up, discarded
        dyn = timed_n(go, REPS - 1)
        del pre
        gc.collect()

        qidx = [idx2[x] for x in Q]
        static_columns_prebuilt(A, AT, qidx, len(nodes2), C, K_DEFAULT)
        stat = timed_n(lambda: static_columns_prebuilt(
            A, AT, qidx, len(nodes2), C, K_DEFAULT), REPS - 1)

        meas = stat / dyn
        pred = 1.0 / (2.0 / q + n / m)
        verdict = "win" if meas > 1 else "lose"
        agree = ("agrees" if (meas > 1) == (pred > 1) else "DISAGREES")
        print("  {:>5}{:>12.1f}{:>13.1f}{:>10.2f}x{:>11.2f}x{:>11}"
              .format(q, dyn, stat, meas, pred, verdict))
        rows.append([q, round(dyn, 2), round(stat, 2), round(meas, 4),
                     round(pred, 4), verdict, agree, round(resident, 1)])

    print("  " + "-" * 64)
    wins = [r[0] for r in rows if r[3] > 1]
    loses = [r[0] for r in rows if r[3] <= 1]
    print("\n  measured: loses at |Q| = {}, wins from |Q| = {}"
          .format(", ".join(map(str, loses)) or "none",
                  min(wins) if wins else "never"))
    print("  predicted: loses at |Q| <= {:.2f}, so wins from |Q| = {}"
          .format(qstar, int(np.floor(qstar)) + 1))
    print("  prediction and measurement agree on the integer: {}"
          .format(min(wins) == int(np.floor(qstar)) + 1 if wins else False))
    print("\n  resident memory in per-query mode : {:.0f} MiB"
          .format(rows[-1][7]))
    print("  a dense n x n matrix would be     : {:,.0f} GiB"
          .format(8.0 * n * n / MB / 1024))
    print("  total wall clock: {:.0f} s".format(time.perf_counter() - t0))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'twitter.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Q", "dyn_ms", "static_ms", "measured", "predicted",
                    "verdict", "agreement", "resident_MiB"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["n", "m", "d_bar", "Q_star", "gate_err"])
        w.writerow([n, m, round(dbar, 4), round(qstar, 4),
                    "{:.2e}".format(worst)])
    print("  -> {}".format(os.path.relpath(p)))


def _probe(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

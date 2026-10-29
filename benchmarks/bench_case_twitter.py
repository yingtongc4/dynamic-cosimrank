# -*- coding: utf-8 -*-
"""
bench_case_twitter.py
=====================
The end-to-end case study, moved onto the graph the motivation is about.

bench_case_study.py follows one follow through a plausible deployment, and
runs it on ego-Facebook, whose edges are symmetrised friendships.  It is the
section of this report closest to the opening scenario, so it is the one most
worth running on a graph where an edge is a follow.

Same design as the ego-Facebook version: three target accounts of very
different popularity, so that a result cannot be a property of one account,
and two independent query sets.  Every configuration is gated against a
converged static recomputation before it is timed.

Usage:
    python benchmarks/bench_case_twitter.py
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
QSIZE = 50
QSEEDS = (7, 8)
REPS = 3


def timed_n(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


def pick_targets(G, Q):
    """Least, median and most followed accounts outside the query set."""
    cand = [v for v in G if v not in Q and G.in_degree(v) > 0]
    cand.sort(key=lambda v: (G.in_degree(v), v))
    return [("least followed", cand[0]),
            ("median", cand[len(cand) // 2]),
            ("most followed", cand[-1])]


def pick_source(G, v, rng):
    nodes = [x for x in G if G.out_degree(x) > 0]
    for _ in range(100000):
        u = rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return u
    raise RuntimeError("no insertable follower")


def main():
    t0 = time.perf_counter()
    G = load_twitter()
    n, m = G.number_of_nodes(), G.number_of_edges()
    print("=" * 90)
    print("  One follow on ego-Twitter, followed end to end")
    print("  n = {:,}, m = {:,}, mean degree {:.1f}, |Q| = {}, K = {}"
          .format(n, m, m / n, QSIZE, K_DEFAULT))
    print("  three targets x two query sets, median of {} runs each"
          .format(REPS))
    print("=" * 90)

    rows, speeds = [], []
    for qseed in QSEEDS:
        rng = random.Random(qseed)
        Q = rng.sample(sorted(G.nodes()), QSIZE)
        targets = pick_targets(G, set(Q))

        print("\n  query set seed {}".format(qseed))
        print("  {:<16}{:>9}{:>11}{:>13}{:>10}{:>9}{:>11}".format(
            "target", "in-deg", "dyn (ms)", "static (ms)", "speedup",
            "skipped", "gate err"))
        print("  " + "-" * 80)

        for label, v in targets:
            u = pick_source(G, v, rng)
            G_after = G.copy(); G_after.add_edge(u, v)

            worst = gate(lambda qs: _probe(G, qs, u, v), G_after, Q, rng,
                         "{} {}".format(label, v))

            probe = PerQueryCoSimRank(G.copy(), queries=list(Q), C=C,
                                      K=K_DEFAULT, tol=0.0)
            probe.edge_insert([(u, v)])
            skipped = probe.last_skipped
            del probe
            gc.collect()

            pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C,
                                     K=K_DEFAULT, tol=0.0)
                   for _ in range(REPS + 1)]
            st = {'i': 0}

            def go():
                p = pre[st['i']]; st['i'] += 1
                p.edge_insert([(u, v)])

            go()
            dyn = timed_n(go, REPS)
            del pre
            gc.collect()

            A, AT, nodes2, idx2 = assemble(G_after)
            qidx = [idx2[q] for q in Q]
            static_columns_prebuilt(A, AT, qidx, len(nodes2), C, K_DEFAULT)
            stat = timed_n(lambda: static_columns_prebuilt(
                A, AT, qidx, len(nodes2), C, K_DEFAULT), REPS)
            del A, AT
            gc.collect()

            sp = stat / dyn
            speeds.append(sp)
            print("  {:<16}{:>9}{:>11.1f}{:>13.1f}{:>9.2f}x{:>7}/{:<3}{:>11.1e}"
                  .format(label, G.in_degree(v), dyn, stat, sp, skipped,
                          QSIZE, worst))
            rows.append([qseed, label, v, G.in_degree(v), u, round(dyn, 2),
                         round(stat, 2), round(sp, 4), skipped,
                         "{:.2e}".format(worst)])

    a = np.array(speeds)
    print("\n  " + "=" * 80)
    print("  across all {} observations: median {:.2f}x, range {:.2f}--{:.2f}x"
          .format(len(a), np.median(a), a.min(), a.max()))
    pred = 1.0 / (2.0 / QSIZE + n / m)
    print("  predicted by (2/|Q| + n/m): {:.2f}x   model/measured {:.2f}"
          .format(pred, pred / np.median(a)))
    print("  in-degree of the targets   : {} to {}"
          .format(min(r[3] for r in rows), max(r[3] for r in rows)))
    print("  queries skipped by pruning : {} of {} in every run"
          .format(rows[0][8], QSIZE))
    print("  total wall clock: {:.0f} s".format(time.perf_counter() - t0))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'case_twitter.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["qseed", "target", "v", "in_degree", "u", "dyn_ms",
                    "static_ms", "speedup", "skipped", "gate_err"])
        w.writerows(rows)
    print("  -> {}".format(os.path.relpath(p)))


def _probe(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
bench_case_study.py
===================
One Follow, on the real graph, told end to end -- and repeated.

The report's qualitative case study rested on a single edge inserted at a
single node with a single query set.  One observation cannot distinguish a
property of the method from a property of that node, and the target's
in-degree is exactly the quantity the update formula is sensitive to: a
follow of a popular account renormalises a long column, a follow of an
obscure one renormalises a short one.

This script therefore repeats the case study over three target accounts of
very different popularity -- lowest in-degree, median, highest -- and over
three independent query sets, and reports the range rather than a point.

Every configuration is gated: the maintained columns are checked against a
converged static recomputation (K=130) before any timing is reported.

Usage:
    python benchmarks/bench_case_study.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            subgraph, C, K_DEFAULT, K_GATE, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

N_NODES = 1000
QSIZE = 50            # a recommender maintaining lists for fifty users
QSEEDS = (7, 8, 9)    # three independent query sets


def pick_targets(G, Q):
    """Three follow targets: least, median and most followed."""
    cand = [v for v in sorted(G.nodes()) if v not in Q]
    cand.sort(key=lambda v: (G.in_degree(v), v))
    return [("least followed", cand[0]),
            ("median",         cand[len(cand) // 2]),
            ("most followed",  cand[-1])]


def pick_source(G, v, Q, rng):
    """A user who does not already follow v."""
    for _ in range(10000):
        u = rng.choice(sorted(G.nodes()))
        if u != v and not G.has_edge(u, v):
            return u
    raise RuntimeError("no insertable source for {}".format(v))


def one_case(G, Q, u, v, rng):
    """Insert the follow u->v; gate it, then time it against recomputation."""
    G_after = G.copy()
    G_after.add_edge(u, v)

    # ---- correctness gate at converged depth, on three columns ----
    worst = gate(lambda qs: _apply(G, qs, u, v, K_GATE),
                 G_after, Q, rng, "follow {}->{}".format(u, v))

    # ---- scope: how much of the graph the update touches ----
    probe = _apply(G, list(Q), u, v, K_DEFAULT)
    touched, skipped = probe.last_touched, probe.last_skipped

    # ---- timing: dynamic ----
    pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                             tol=0.0) for _ in range(REPS)]
    st = {'i': 0}

    def run():
        p = pre[st['i']]; st['i'] += 1
        p.edge_insert([(u, v)])

    dyn = timed(run, REPS)

    # ---- timing: static, from an already assembled matrix ----
    A, AT, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                 C, K_DEFAULT), REPS)
    return dict(dyn=dyn[0], static=stat[0], speedup=stat[0] / dyn[0],
                touched=touched, skipped=skipped, gate=worst)


def _apply(G, qs, u, v, K):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=K, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


def main():
    G = subgraph('facebook', N_NODES)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print("=" * 78)
    print("  One Follow on ego-Facebook, repeated")
    print("  n = {}, m = {}, mean degree {:.1f}, |Q| = {}, K = {}"
          .format(n, m, m / n, QSIZE, K_DEFAULT))
    print("  three targets x three query sets, median of {} runs each"
          .format(REPS))
    print("=" * 78)

    rows, speeds = [], []
    for qseed in QSEEDS:
        rng = random.Random(qseed)
        Q = rng.sample(sorted(G.nodes()), QSIZE)
        targets = pick_targets(G, Q)

        print("\n  query set seed {}".format(qseed))
        print("  {:<16}{:>7}{:>8}{:>10}{:>11}{:>10}{:>9}{:>11}".format(
            "target", "in-deg", "u", "dyn ms", "static ms", "speedup",
            "skipped", "gate err"))
        print("  " + "-" * 84)
        for label, v in targets:
            u = pick_source(G, v, Q, rng)
            r = one_case(G, Q, u, v, rng)
            speeds.append(r['speedup'])
            print("  {:<16}{:>7}{:>8}{:>10.2f}{:>11.2f}{:>9.2f}x{:>7}/{:<3}{:>11.1e}"
                  .format(label, G.in_degree(v), u, r['dyn'], r['static'],
                          r['speedup'], r['skipped'], QSIZE, r['gate']))
            rows.append([qseed, label, v, G.in_degree(v), u,
                         round(r['dyn'], 3), round(r['static'], 3),
                         round(r['speedup'], 4), r['touched'], r['skipped'],
                         "{:.2e}".format(r['gate'])])

    a = np.array(speeds)
    print("\n  " + "=" * 74)
    print("  across all {} observations".format(len(a)))
    print("  median  {:.2f}x    range  {:.2f}x -- {:.2f}x    spread {:.2f}x"
          .format(np.median(a), a.min(), a.max(), a.max() / a.min()))
    pred = 1.0 / (2.0 / QSIZE + n / m)
    print("  predicted by (2/|Q| + n/m) : {:.2f}x   overstatement {:.2f}x"
          .format(pred, pred / np.median(a)))
    tou = [r[8] for r in rows]
    print("  columns touched            : {}--{} of {}  (pruning never fires)"
          .format(min(tou), max(tou), n))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'case_study.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["qseed", "target", "v", "in_degree", "u", "dyn_ms",
                    "static_ms", "speedup", "touched", "skipped", "gate_err"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
bench_blocked_both.py
=====================
The experiment bench_blocked.py said was owed and did not run.

bench_blocked.py gave the static side a blocked implementation and left the
dynamic side as a Python loop over |Q|, which made that comparison
deliberately one-sided.  It is also the loop identified as where the
interpreter overhead concentrates, so blocking it is the direct test of the
report's least-verified claim: that the gap between the cost model and the
measurements belongs to the implementation and not to the model.

Four implementations, all producing the same numbers:

  dyn  loop     the per-query accumulation as a Python loop over |Q|
  dyn  blocked  the same arithmetic as two dense (n x K+1)(K+1 x |Q|) products
  stat per-col  |Q| independent sparse matrix-vector iterations
  stat blocked  one sparse matrix-matrix iteration over an n x |Q| block

The prediction under test is quantitative.  If the model's overstatement is
interpreter overhead concentrated in the per-query term, then blocking both
sides should move the measured speedup towards the predicted one; if the
overstatement is a modelling error, blocking both sides should leave the gap
roughly where it was.

Usage:
    python benchmarks/bench_blocked_both.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            subgraph, C, K_DEFAULT, REPS)
from bench_blocked import static_columns_blocked

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
N_NODES = 1000
Q_SIZES = (1, 2, 3, 5, 10, 20, 50, 200)


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def time_dynamic(G, Q, edge, blocked):
    pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                             tol=0.0, blocked=blocked) for _ in range(REPS)]
    st = {'i': 0}

    def go():
        p = pre[st['i']]; st['i'] += 1
        p.edge_insert([edge])

    return timed(go, REPS)[0]


def main():
    G = subgraph('facebook', N_NODES)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print("=" * 94)
    print("  Blocking both sides.  ego-Facebook n = {}, m = {}, K = {},"
          " median of {} runs.".format(n, m, K_DEFAULT, REPS))
    print("  'model' is 1/(2/|Q| + n/m); 'over' is model divided by measured.")
    print("=" * 94)
    print("\n  {:>4}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>8}{:>8}".format(
        "|Q|", "dyn loop", "dyn blk", "st col", "st blk", "model",
        "loop/col", "blk/blk", "over"))
    print("  " + "-" * 76)

    rows = []
    for qsize in Q_SIZES:
        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), qsize)
        u, v = insertable_edge(G, rng)
        G_after = G.copy(); G_after.add_edge(u, v)

        gate(lambda qs: _probe(G, qs, u, v), G_after, Q, rng,
             "|Q|={}".format(qsize))

        # the two dynamic modes must agree before either is timed
        a = _probe(G, list(Q), u, v, K_DEFAULT, False)
        b = _probe(G, list(Q), u, v, K_DEFAULT, True)
        agree = max(float(np.abs(a.column(q) - b.column(q)).max()) for q in Q)
        assert agree < 1e-12, "blocked and loop disagree by {:.2e}".format(agree)

        d_loop = time_dynamic(G, Q, (u, v), False)
        d_blk = time_dynamic(G, Q, (u, v), True)

        A, AT, nodes2, idx2 = assemble(G_after)
        qidx = [idx2[q] for q in Q]
        s_col = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                      C, K_DEFAULT), REPS)[0]
        s_blk = timed(lambda: static_columns_blocked(A, AT, qidx, len(nodes2),
                                                     C, K_DEFAULT), REPS)[0]

        model = 1.0 / (2.0 / qsize + n / m)
        loop_col = s_col / d_loop
        blk_blk = s_blk / d_blk
        print("  {:>4}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>8.2f}x{:>8.2f}x"
              "{:>7.2f}x{:>8.2f}".format(qsize, d_loop, d_blk, s_col, s_blk,
                                         model, loop_col, blk_blk,
                                         model / blk_blk))
        rows.append([qsize, round(d_loop, 3), round(d_blk, 3),
                     round(s_col, 3), round(s_blk, 3), round(model, 4),
                     round(loop_col, 4), round(blk_blk, 4),
                     round(model / loop_col, 4), round(model / blk_blk, 4),
                     "{:.1e}".format(agree)])

    print("  " + "-" * 76)
    big = [r for r in rows if r[0] >= 10]
    o_loop = [r[8] for r in big]
    o_blk = [r[9] for r in big]
    print("\n  Model overstatement at |Q| >= 10")
    print("    against the unblocked pair : {:.2f} to {:.2f}"
          .format(min(o_loop), max(o_loop)))
    print("    against the blocked pair   : {:.2f} to {:.2f}"
          .format(min(o_blk), max(o_blk)))
    sp = [r[2] / r[1] for r in rows]
    print("\n  Blocking the dynamic side alone speeds it up {:.2f}x to {:.2f}x"
          .format(1 / max(sp), 1 / min(sp)))
    win_loop = [r[0] for r in rows if r[6] > 1]
    win_blk = [r[0] for r in rows if r[7] > 1]
    print("  Smallest winning |Q|: unblocked {}, blocked {}"
          .format(min(win_loop) if win_loop else "none",
                  min(win_blk) if win_blk else "none"))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'blocked_both.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Q", "dyn_loop_ms", "dyn_blocked_ms", "static_percol_ms",
                    "static_blocked_ms", "model", "loop_over_percol",
                    "blocked_over_blocked", "over_unblocked", "over_blocked",
                    "mode_agreement"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _probe(G, qs, u, v, K=130, blocked=False):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=K, tol=0.0,
                           blocked=blocked)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

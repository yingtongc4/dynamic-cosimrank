# -*- coding: utf-8 -*-
"""
bench_reconcile.py
==================
Two tables in the report quote the same configuration for node deletion at
out-degree 1 -- ego-Facebook n=1000, |Q|=5, K=20, median of 10 runs -- and
disagree by about 1.5x.  The two figures come from different scripts run at
different times.  This one runs both selections back to back in a single
process so that the discrepancy is either reproduced or eliminated, and
reports which node and which query set each picks.

Usage:
    python benchmarks/bench_reconcile.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_perquery import (assemble, static_columns_prebuilt, timed,
                            C, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 42


def pick_breakdown(G, nodes, Q):
    """Selection used by bench_nodeop_breakdown.py."""
    return next(v for v in nodes if v not in Q and G.out_degree(v) == 1)


def pick_constants(G, nodes, Q):
    """Selection used by bench_constants.py."""
    by_out = {}
    for v in nodes:
        if v in Q:
            continue
        by_out.setdefault(G.out_degree(v), []).append(v)
    for d in sorted(by_out):
        if d == 0:
            continue
        return by_out[d][0]
    raise RuntimeError("no candidate")


def measure(G, Q, victim, reps=REPS):
    G_after = G.copy(); G_after.remove_node(victim)

    pre = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
           for _ in range(reps)]
    st = {'i': 0}

    def run():
        p = pre[st['i']]; st['i'] += 1
        p.node_delete(victim)

    dyn = timed(run, reps)

    A2, AT2, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(
        A2, AT2, qidx, len(nodes2), C, K_DEFAULT), reps)
    return dyn, stat


def components(G, Q, victim, reps=REPS):
    """The three parts a node deletion spends its time in."""
    base = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
    d_idx = base.idx[victim]

    pre = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
           for _ in range(reps)]
    st = {'i': 0}

    def one_column():
        p = pre[st['i']]; st['i'] += 1
        p._apply_column_change(p.idx[victim], set())

    col = timed(one_column, reps)
    rebuild = timed(base._build_A, reps)
    reindex = timed(
        lambda: [{i - 1 if i > d_idx else i for i in preds}
                 for preds in base.in_nbrs], reps)
    return col, rebuild, reindex


def main():
    print("=" * 76)
    print("  Reconciling two measurements of the same configuration")
    print("  ego-Facebook n=1000, |Q|=5, K=20, deg_out(v)=1, median of 10 runs")
    print("=" * 76)

    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=1000).to_directed()
    n, m = G.number_of_nodes(), G.number_of_edges()
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 5)
    print("  n={}, m={}, Q={}".format(n, m, sorted(Q)))

    v_b = pick_breakdown(G, nodes, Q)
    v_c = pick_constants(G, nodes, Q)
    print("\n  node chosen by bench_nodeop_breakdown : {}  (out-degree {})"
          .format(v_b, G.out_degree(v_b)))
    print("  node chosen by bench_constants        : {}  (out-degree {})"
          .format(v_c, G.out_degree(v_c)))
    print("  same node: {}".format(v_b == v_c))

    rows = []
    print("\n  {:<28}{:>11}{:>12}{:>10}".format(
        "selection", "dyn (ms)", "static (ms)", "speedup"))
    print("  " + "-" * 62)
    for label, v in [("breakdown selection", v_b), ("constants selection", v_c)]:
        dyn, stat = measure(G, Q, v)
        print("  {:<28}{:>11.2f}{:>12.2f}{:>9.2f}x".format(
            label, dyn[0], stat[0], stat[0] / dyn[0]))
        rows.append([label, v, round(dyn[0], 3), round(stat[0], 3),
                     round(stat[0] / dyn[0], 4)])

    # repeat the first one to gauge within-session drift
    dyn2, stat2 = measure(G, Q, v_b)
    print("  {:<28}{:>11.2f}{:>12.2f}{:>9.2f}x".format(
        "breakdown selection, again", dyn2[0], stat2[0], stat2[0] / dyn2[0]))
    rows.append(["breakdown repeat", v_b, round(dyn2[0], 3), round(stat2[0], 3),
                 round(stat2[0] / dyn2[0], 4)])

    col, rebuild, reindex = components(G, Q, v_b)
    dyn, stat = measure(G, Q, v_b)
    b = G.out_degree(v_b) + 1
    parts = [("column updates (x{})".format(b), col[0] * b),
             ("full sparse rebuild", rebuild[0]),
             ("renumbering", reindex[0])]
    print("\n  Breakdown against the total measured in the same session:")
    print("  {:<34}{:>10}{:>10}".format("component", "ms", "share"))
    print("  " + "-" * 54)
    for lbl, ms in parts:
        print("  {:<34}{:>10.2f}{:>9.0f}%".format(lbl, ms, 100 * ms / dyn[0]))
        rows.append([lbl, "", round(ms, 3), "", round(100 * ms / dyn[0], 1)])
    print("  " + "-" * 54)
    print("  {:<34}{:>10.2f}{:>9.0f}%".format("measured total", dyn[0], 100.0))
    print("  {:<34}{:>10.2f}".format("static baseline", stat[0]))
    rows.append(["measured total", "", round(dyn[0], 3), "", 100.0])
    rows.append(["static baseline", "", round(stat[0], 3), "", ""])

    share = 100 * col[0] * b / dyn[0]
    print("\n  column updates are {:.0f}% of the operation".format(share))
    print("  the batch model understates the cost by {:.1f}x".format(
        dyn[0] / (col[0] * b)))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'nodeop_reconciled.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["item", "node", "ms", "static_ms", "value"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

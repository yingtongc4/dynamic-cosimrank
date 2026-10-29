# -*- coding: utf-8 -*-
"""
bench_nodeop_breakdown.py
=========================
Why node operations lose by more than the batch model predicts.

Equation (batch) counts only column updates, on the assumption that a node
operation is b = deg_out(v)+1 of them and nothing else.  The implementation
does something else as well: because a node operation changes n, the CSR
splice used for edge updates does not apply, so the whole sparse matrix is
rebuilt and every in-neighbour index is renumbered.  Neither step appears in
any term of the flop model.

This script times the parts separately so the discrepancy can be attributed
rather than guessed at.

Usage:
    python benchmarks/bench_nodeop_breakdown.py
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


def reindex_cost(pq, d_idx):
    """The Python-level renumbering a node operation performs."""
    return [{i - 1 if i > d_idx else i for i in preds} for preds in pq.in_nbrs]


def main():
    print("=" * 76)
    print("  Node-operation cost breakdown (ego-Facebook n=1000, |Q|=5, K=20)")
    print("=" * 76)

    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=1000).to_directed()
    n, m = G.number_of_nodes(), G.number_of_edges()
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 5)
    print("  n={}, m={}, n/m={:.5f}".format(n, m, n / m))

    # a node of out-degree 1: the cheapest possible deletion, b = 2
    victim = next(v for v in nodes if v not in Q and G.out_degree(v) == 1)
    d_out = G.out_degree(victim)
    print("  deleting node {} with out-degree {} -> b = {} column updates"
          .format(victim, d_out, d_out + 1))

    base = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
    d_idx = base.idx[victim]

    # --- component 1: one column update ---
    pre = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
           for _ in range(REPS)]
    st = {'i': 0}

    def one_column():
        p = pre[st['i']]; st['i'] += 1
        j = p.idx[victim]
        p._apply_column_change(j, set())

    col = timed(one_column, REPS)

    # --- component 2: the full sparse rebuild ---
    rebuild = timed(base._build_A, REPS)

    # --- component 3: the Python-level renumbering ---
    reindex = timed(lambda: reindex_cost(base, d_idx), REPS)

    # --- component 4: the whole node deletion ---
    pre2 = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
            for _ in range(REPS)]
    st2 = {'i': 0}

    def whole():
        p = pre2[st2['i']]; st2['i'] += 1
        p.node_delete(victim)

    total = timed(whole, REPS)

    # --- the static baseline it is measured against ---
    G_after = G.copy(); G_after.remove_node(victim)
    A2, AT2, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(
        A2, AT2, qidx, len(nodes2), C, K_DEFAULT), REPS)

    b = d_out + 1
    modelled = b * col[0]
    print()
    print("  {:<46}{:>10}{:>10}".format("component", "ms", "share"))
    print("  " + "-" * 66)
    rows = []
    for label, ms in [
        ("one column update (x{})".format(b), col[0] * b),
        ("full sparse rebuild, _build_A", rebuild[0]),
        ("Python renumbering of in-neighbour sets", reindex[0]),
    ]:
        print("  {:<46}{:>10.2f}{:>9.0f}%".format(label, ms, 100 * ms / total[0]))
        rows.append([label, round(ms, 3), round(100 * ms / total[0], 1)])
    print("  " + "-" * 66)
    print("  {:<46}{:>10.2f}{:>9.0f}%".format("measured node_delete total",
                                              total[0], 100.0))
    print("  {:<46}{:>10.2f}".format("static baseline (prebuilt)", stat[0]))
    rows.append(["measured node_delete total", round(total[0], 3), 100.0])
    rows.append(["static baseline", round(stat[0], 3), ""])

    print()
    single_ratio = 2.0 / len(Q) + n / m
    pred_single = 1.0 / single_ratio
    pred_batch = 1.0 / (b * single_ratio)
    meas = stat[0] / total[0]
    print("  Model vs measurement")
    print("  " + "-" * 66)
    print("  single-update speedup, predicted        : {:>7.2f}x".format(pred_single))
    print("  batch of b={} predicted                  : {:>7.2f}x".format(b, pred_batch))
    print("  measured                                : {:>7.2f}x".format(meas))
    print("  overstatement                           : {:>7.2f}x".format(pred_batch / meas))
    print()
    print("  Of the measured time, only {:.0f}% is column updates -- the term the".format(
        100 * modelled / total[0]))
    print("  batch model counts.  The remaining {:.0f}% is the rebuild and".format(
        100 * (total[0] - modelled) / total[0]))
    print("  renumbering that a change in n forces, which no term represents.")

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'nodeop_breakdown.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["component", "ms", "share_pct"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["predicted_single", round(pred_single, 4)])
        w.writerow(["predicted_batch_b{}".format(b), round(pred_batch, 4)])
        w.writerow(["measured", round(meas, 4)])
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

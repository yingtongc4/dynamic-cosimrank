# -*- coding: utf-8 -*-
"""
bench_nodeop_phases.py
======================
Where a node deletion actually spends its time, measured inside a single call.

An earlier attempt timed the parts on separate objects and reported shares of
27%, 55% and 16%.  That was wrong: the parts were measured in different cache
and allocator states, they did not have to sum to the total, and re-running
the identical configuration showed the total itself varying by 1.5x between
sessions.  The shares summed to 140%.

This version instruments one node_delete call, so the phases sum to the total
by construction, and repeats it enough times to report a distribution rather
than a point.

Usage:
    python benchmarks/bench_nodeop_phases.py
"""

import sys, os, csv, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_perquery import (assemble, static_columns_prebuilt, timed,
                            C, K_DEFAULT)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEED = 42
REPS = 40


def instrumented_node_delete(pq, del_node):
    """
    node_delete, with a timer around each phase.  Mirrors per_query.node_delete
    exactly; any divergence would show up as phases not summing to the whole,
    which is checked by the caller.
    """
    t = {}
    t0 = time.perf_counter()

    d_idx = pq.idx[del_node]

    # --- phase 1: column updates (the only term eq:batch counts) ---
    a = time.perf_counter()
    if pq.in_nbrs[d_idx]:
        pq._apply_column_change(d_idx, set())
    succs = [j for j, preds in enumerate(pq.in_nbrs) if d_idx in preds]
    for j in succs:
        preds = set(pq.in_nbrs[j]); preds.discard(d_idx)
        pq._apply_column_change(j, preds)
    t['columns'] = time.perf_counter() - a

    # --- phase 2: renumbering the in-neighbour sets ---
    a = time.perf_counter()
    pq.nodes.pop(d_idx)
    pq.in_nbrs.pop(d_idx)
    pq.in_nbrs = [{i - 1 if i > d_idx else i for i in preds}
                  for preds in pq.in_nbrs]
    pq.idx = {v: i for i, v in enumerate(pq.nodes)}
    t['renumber'] = time.perf_counter() - a

    # --- phase 3: rebuilding the sparse matrix, forced by the change in n ---
    a = time.perf_counter()
    pq._build_A()
    t['rebuild'] = time.perf_counter() - a

    # --- phase 4: shrinking the maintained columns ---
    a = time.perf_counter()
    if del_node in pq.cols:
        del pq.cols[del_node]
        pq.query_nodes.remove(del_node)
    for q in pq.query_nodes:
        pq.cols[q] = np.delete(pq.cols[q], d_idx)
    t['columns_shrink'] = time.perf_counter() - a

    t['total'] = time.perf_counter() - t0
    return t


def main():
    print("=" * 76)
    print("  Node-deletion phases, timed inside a single call")
    print("  ego-Facebook n=1000, |Q|=5, K=20, deg_out(v)=1, {} repetitions"
          .format(REPS))
    print("=" * 76)

    G = extract_subgraph(load_facebook(), min_degree=10,
                         max_nodes=1000).to_directed()
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 5)
    victim = next(v for v in nodes if v not in Q and G.out_degree(v) == 1)
    b = G.out_degree(victim) + 1
    print("  deleting node {} (out-degree {}), b = {} column updates"
          .format(victim, G.out_degree(victim), b))

    samples = []
    for _ in range(REPS):
        pq = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
        samples.append(instrumented_node_delete(pq, victim))

    keys = ['columns', 'renumber', 'rebuild', 'columns_shrink']
    ms = {k: np.array([s[k] for s in samples]) * 1000 for k in keys + ['total']}

    # sanity: phases must account for the total
    acct = sum(ms[k] for k in keys)
    resid = np.median(ms['total'] - acct)
    print("  phases account for {:.1f}% of the total (residual {:.3f} ms)"
          .format(100 * np.median(acct) / np.median(ms['total']), resid))

    print("\n  {:<20}{:>10}{:>10}{:>10}{:>9}".format(
        "phase", "median", "IQR lo", "IQR hi", "share"))
    print("  " + "-" * 60)
    rows = []
    for k in keys:
        a = ms[k]
        share = 100 * np.median(a) / np.median(ms['total'])
        print("  {:<20}{:>9.2f}{:>10.2f}{:>10.2f}{:>8.0f}%".format(
            k, np.median(a), np.percentile(a, 25), np.percentile(a, 75), share))
        rows.append([k, round(float(np.median(a)), 3),
                     round(float(np.percentile(a, 25)), 3),
                     round(float(np.percentile(a, 75)), 3), round(share, 1)])
    tot = ms['total']
    print("  " + "-" * 60)
    print("  {:<20}{:>9.2f}{:>10.2f}{:>10.2f}{:>8}%".format(
        "total", np.median(tot), np.percentile(tot, 25),
        np.percentile(tot, 75), 100))
    print("  total spread: {:.2f}x between min and max"
          .format(tot.max() / tot.min()))
    rows.append(["total", round(float(np.median(tot)), 3),
                 round(float(np.percentile(tot, 25)), 3),
                 round(float(np.percentile(tot, 75)), 3), 100.0])

    # the static baseline, same session
    G_after = G.copy(); G_after.remove_node(victim)
    A2, AT2, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    stat = timed(lambda: static_columns_prebuilt(
        A2, AT2, qidx, len(nodes2), C, K_DEFAULT), REPS)
    print("\n  static baseline (prebuilt) : {:.2f} ms".format(stat[0]))
    print("  measured speedup           : {:.2f}x"
          .format(stat[0] / np.median(tot)))
    rows.append(["static_baseline", round(stat[0], 3), "", "", ""])

    col_share = 100 * np.median(ms['columns']) / np.median(tot)
    print("\n  Column updates -- the only term eq:batch counts -- are {:.0f}% of"
          .format(col_share))
    print("  the operation; the batch model therefore understates its cost by")
    print("  {:.1f}x.".format(np.median(tot) / np.median(ms['columns'])))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'nodeop_phases.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["phase", "median_ms", "iqr_lo_ms", "iqr_hi_ms", "share_pct"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

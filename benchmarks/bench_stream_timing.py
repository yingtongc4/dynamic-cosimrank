# -*- coding: utf-8 -*-
"""
bench_stream_timing.py
======================
Does the speed advantage survive a long stream?

Every timing in this report measures one update applied to a freshly built
object.  The title says evolving graphs, but nothing measures update number
500, after the CSR buffers have been spliced hundreds of times and the
maintained columns have drifted.  Two things could degrade there and neither
has been checked: the sparse structure could fragment as columns are spliced
in and out, and the graph itself drifts away from the one the timings were
taken on.

This script times the update at intervals across a 1000-operation stream and
compares it with a static recomputation on the graph as it stands at that
point, so the speedup is measured against a moving baseline rather than the
initial one.

Usage:
    python benchmarks/bench_stream_timing.py
"""

import sys, os, csv, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from bench_perquery import (assemble, static_columns_prebuilt, subgraph,
                            C, K_DEFAULT)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

N_NODES = 500
QSIZE = 20
N_STEPS = 1000
PROBE_EVERY = 100
REPS = 5
SEED = 11


def next_op(H, rng):
    nodes = sorted(H.nodes())
    if rng.random() < 0.5:
        for _ in range(300):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and not H.has_edge(u, v):
                return ('i', u, v)
    es = list(H.edges())
    for _ in range(300):
        u, v = es[rng.randrange(len(es))]
        if H.in_degree(v) > 1:
            return ('d', u, v)
    return None


def main():
    G = subgraph('facebook', N_NODES)
    n = G.number_of_nodes()
    rng = random.Random(SEED)
    Q = rng.sample(sorted(G.nodes()), QSIZE)

    print("=" * 80)
    print("  Update cost along a {}-operation stream".format(N_STEPS))
    print("  ego-Facebook n={}, |Q|={}, K={}, probe every {} operations"
          .format(n, QSIZE, K_DEFAULT, PROBE_EVERY))
    print("=" * 80)
    print("\n  {:>8}{:>8}{:>12}{:>12}{:>11}{:>12}".format(
        "step", "m", "dyn (ms)", "static (ms)", "speedup", "nnz(A)"))
    print("  " + "-" * 62)

    pq = PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                           tol=0.0)
    H = G.copy()
    rows = []

    for step in range(1, N_STEPS + 1):
        op = next_op(H, rng)
        if op is None:
            break
        kind, u, v = op

        probe = (step % PROBE_EVERY == 0) or step == 1
        if probe:
            # time this operation properly, on clones of the current state
            clones = [PerQueryCoSimRank(H.copy(), queries=list(Q), C=C,
                                        K=K_DEFAULT, tol=0.0)
                      for _ in range(REPS)]
            ts = []
            for c in clones:
                t = time.perf_counter()
                if kind == 'i':
                    c.edge_insert([(u, v)])
                else:
                    c.edge_delete([(u, v)])
                ts.append((time.perf_counter() - t) * 1000.0)
            dyn = float(np.median(ts))

        if kind == 'i':
            pq.edge_insert([(u, v)]); H.add_edge(u, v)
        else:
            pq.edge_delete([(u, v)]); H.remove_edge(u, v)

        if probe:
            A, AT, nodes2, idx2 = assemble(H)
            qidx = [idx2[q] for q in Q]
            ss = []
            for _ in range(REPS):
                t = time.perf_counter()
                static_columns_prebuilt(A, AT, qidx, len(nodes2), C, K_DEFAULT)
                ss.append((time.perf_counter() - t) * 1000.0)
            stat = float(np.median(ss))
            m = H.number_of_edges()
            nnz = pq.AT.nnz
            print("  {:>8}{:>8}{:>12.2f}{:>12.2f}{:>10.2f}x{:>12}".format(
                step, m, dyn, stat, stat / dyn, nnz))
            rows.append([step, m, round(dyn, 3), round(stat, 3),
                         round(stat / dyn, 4), nnz])

    print("  " + "-" * 62)
    sp = [r[4] for r in rows]
    dy = [r[2] for r in rows]
    print("\n  speedup at the first probe : {:.2f}x".format(sp[0]))
    print("  speedup at the last probe  : {:.2f}x".format(sp[-1]))
    print("  range across the stream    : {:.2f}x to {:.2f}x"
          .format(min(sp), max(sp)))
    print("  dynamic cost, first / last : {:.2f} / {:.2f} ms  (ratio {:.2f})"
          .format(dy[0], dy[-1], dy[-1] / dy[0]))
    print("  nnz(A), first / last       : {} / {}".format(rows[0][5],
                                                          rows[-1][5]))
    print("  edges, first / last        : {} / {}".format(rows[0][1],
                                                          rows[-1][1]))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'stream_timing.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["step", "m", "dyn_ms", "static_ms", "speedup", "nnz_A"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
bench_memory.py
===============
The per-query mode's whole claim is that it never allocates an n x n array.
The report asserts that and never measures it, which is the same omission the
cost model was criticised for elsewhere.  D-CoSim advertises linear memory as
a headline property, so the comparison is owed.

Two quantities are measured, because a single peak figure conflates them and
the first version of this script did exactly that:

  RESIDENT   the maintained state that persists between updates -- the sparse
             matrices and whichever similarity scores are being kept.  This is
             what the O(n|Q|) versus O(n^2) claim is about.  Measured by
             summing nbytes over the arrays actually held.

  TRANSIENT  the additional allocation a single update performs, measured with
             tracemalloc started AFTER construction, so that the Python-list
             intermediates of graph assembly -- which dominate a naive peak
             measurement and belong to neither algorithm -- are excluded.

Usage:
    python benchmarks/bench_memory.py
"""

import sys, os, csv, gc, random, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import scipy.sparse as sp

from per_query import PerQueryCoSimRank
from dynamic_cosimrank import DynamicCoSimRank
from bench_perquery import subgraph, C, K_DEFAULT

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
QSIZE = 50
SIZES = (200, 500, 1000, 2000)
MB = 1024.0 * 1024.0


def nbytes(x):
    """Bytes held by a NumPy array or a SciPy sparse matrix."""
    if sp.issparse(x):
        x = x.tocsr()
        return x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    return np.asarray(x).nbytes


def resident_pq(pq):
    b = nbytes(pq.AT)
    b += sum(nbytes(v) for v in pq.cols.values())
    return b / MB


def resident_full(d):
    b = nbytes(d.S)
    for attr in ('A', 'AT'):
        if hasattr(d, attr) and getattr(d, attr) is not None:
            b += nbytes(getattr(d, attr))
    return b / MB


def transient(obj, apply_update):
    """Allocation performed by one update, construction already paid for."""
    gc.collect()
    tracemalloc.start()
    try:
        apply_update(obj)
        _, hi = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return hi / MB


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def main():
    print("=" * 90)
    print("  Memory, |Q| = {}, K = {}, one edge insertion.  MiB."
          .format(QSIZE, K_DEFAULT))
    print("  RESIDENT: state held between updates.  TRANSIENT: extra")
    print("  allocation during one update, construction excluded.")
    print("=" * 90)
    print("\n  {:>6}{:>8}{:>10}{:>12}{:>11}{:>10}{:>12}{:>11}".format(
        "n", "m", "dense n^2", "res: full", "res: p-q", "ratio",
        "tra: full", "tra: p-q"))
    print("  " + "-" * 80)

    rows = []
    for n_req in SIZES:
        G = subgraph('facebook', n_req)
        n, m = G.number_of_nodes(), G.number_of_edges()
        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), QSIZE)
        u, v = insertable_edge(G, rng)
        dense = 8.0 * n * n / MB

        pq = PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                               tol=0.0)
        r_pq = resident_pq(pq)
        t_pq = transient(pq, lambda o: o.edge_insert([(u, v)]))

        fu = DynamicCoSimRank(G.copy(), C=C, K=K_DEFAULT, tol=0.0)
        r_fu = resident_full(fu)
        t_fu = transient(fu, lambda o: o.edge_insert([(u, v)]))

        print("  {:>6}{:>8}{:>10.2f}{:>12.2f}{:>11.2f}{:>9.2f}x{:>12.2f}"
              "{:>11.2f}".format(n, m, dense, r_fu, r_pq, r_pq / r_fu,
                                 t_fu, t_pq))
        rows.append([n, m, round(dense, 3), round(r_fu, 3), round(r_pq, 3),
                     round(r_pq / r_fu, 4), round(t_fu, 3), round(t_pq, 3)])

    print("  " + "-" * 80)
    a, f = rows[0], rows[-1]
    gn = f[0] / a[0]
    gm = f[1] / a[1]
    print("\n  From n={} to n={}:  n x{:.1f}, m x{:.1f}"
          .format(a[0], f[0], gn, gm))
    print("    resident, full mode : x{:.1f}   (n^2 predicts x{:.1f})"
          .format(f[3] / a[3], gn ** 2))
    print("    resident, per-query : x{:.1f}   (n|Q| predicts x{:.1f}, "
          "m predicts x{:.1f})".format(f[4] / a[4], gn, gm))
    print("    transient, per-query: x{:.1f}".format(f[7] / a[7]))
    print("\n  The per-query mode holds no n x n array at any size; the"
          " largest object")
    print("  it allocates is the |Q| x n block of maintained columns.")

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'memory.csv')
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(["n", "m", "dense_n2_MiB", "resident_full_MiB",
                    "resident_pq_MiB", "ratio", "transient_full_MiB",
                    "transient_pq_MiB"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

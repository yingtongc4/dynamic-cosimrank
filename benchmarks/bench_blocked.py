# -*- coding: utf-8 -*-
"""
bench_blocked.py
================
The adversarial baseline the report had not yet built.

Every static baseline used so far recomputes the |Q| query columns one at a
time: |Q| independent sparse matrix-VECTOR products per iteration.  A static
implementation written to win would not do that.  It would stack the |Q|
right-hand sides into one dense n x |Q| array and issue a sparse
matrix-MATRIX product, which SciPy dispatches to a routine that touches each
matrix entry once for all |Q| columns instead of once per column, with far
better cache behaviour.  Section "what the model does not capture" argues that
baselines must be chosen adversarially; this is the baseline that argument
demands.

The flop count is identical -- 2K|Q| products' worth of arithmetic either way,
which is why the cost model cannot tell them apart.  The wall-clock is not,
and the question this script answers is by how much, and whether the reported
speedups survive it.

Note that this cuts both ways: the same blocking could be applied to the
dynamic method's per-query accumulation, which is a Python loop over |Q|.  It
is not applied here, so the comparison below is deliberately unfavourable to
the method under test.

Usage:
    python benchmarks/bench_blocked.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            subgraph, C, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
CONFIGS = [(1000, 1), (1000, 2), (1000, 3), (1000, 5), (1000, 10),
           (1000, 20), (1000, 50), (1000, 200)]


def static_columns_blocked(A, AT, q_indices, n, C=C, K=K_DEFAULT):
    """
    The same computation as static_columns_prebuilt, with the |Q| right-hand
    sides carried together as one dense n x |Q| block.  Each iteration is one
    sparse-times-dense product instead of |Q| sparse-times-vector products.
    """
    P = np.zeros((n, len(q_indices)))
    for c, qi in enumerate(q_indices):
        P[qi, c] = 1.0
    fwd = [P]
    for _ in range(K):
        fwd.append(A @ fwd[-1])
    R = fwd[K].copy()
    for k in range(K - 1, -1, -1):
        R = C * (AT @ R) + fwd[k]
    return R


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def main():
    print("=" * 88)
    print("  A blocked static baseline: same flops, one sparse-times-dense")
    print("  product per iteration instead of |Q| sparse-times-vector ones.")
    print("  ego-Facebook, one edge inserted, K = {}, median of {} runs."
          .format(K_DEFAULT, REPS))
    print("=" * 88)
    print("\n  {:>6}{:>6}{:>10}{:>12}{:>11}{:>12}{:>11}{:>12}".format(
        "n", "|Q|", "dyn (ms)", "per-col (ms)", "vs per-col",
        "blocked (ms)", "vs blocked", "block gain"))
    print("  " + "-" * 80)

    rows = []
    for n_req, qsize in CONFIGS:
        G = subgraph('facebook', n_req)
        n, m = G.number_of_nodes(), G.number_of_edges()
        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), qsize)
        u, v = insertable_edge(G, rng)
        G_after = G.copy(); G_after.add_edge(u, v)

        worst = gate(lambda qs: _apply(G, qs, u, v), G_after, Q, rng,
                     "n={} |Q|={}".format(n, qsize))

        pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                                 tol=0.0) for _ in range(REPS)]
        st = {'i': 0}

        def go():
            p = pre[st['i']]; st['i'] += 1
            p.edge_insert([(u, v)])

        dyn = timed(go, REPS)[0]

        A, AT, nodes2, idx2 = assemble(G_after)
        qidx = [idx2[q] for q in Q]

        # the two baselines must agree, or the comparison is meaningless
        a = static_columns_prebuilt(A, AT, qidx, len(nodes2), C, K_DEFAULT)
        b = static_columns_blocked(A, AT, qidx, len(nodes2), C, K_DEFAULT)
        agree = max(float(np.abs(np.asarray(a[c]) - b[:, c]).max())
                    for c in range(len(qidx)))
        assert agree < 1e-12, "baselines disagree by {:.2e}".format(agree)

        per = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                    C, K_DEFAULT), REPS)[0]
        blk = timed(lambda: static_columns_blocked(A, AT, qidx, len(nodes2),
                                                   C, K_DEFAULT), REPS)[0]

        print("  {:>6}{:>6}{:>10.2f}{:>12.2f}{:>10.2f}x{:>12.2f}{:>10.2f}x"
              "{:>11.2f}x".format(n, qsize, dyn, per, per / dyn, blk,
                                  blk / dyn, per / blk))
        rows.append([n, m, qsize, round(dyn, 3), round(per, 3), round(blk, 3),
                     round(per / dyn, 4), round(blk / dyn, 4),
                     round(per / blk, 4), "{:.1e}".format(agree),
                     "{:.1e}".format(worst)])

    print("  " + "-" * 80)
    print("\n  The blocked baseline is {:.1f}x to {:.1f}x faster than the "
          "per-column one".format(min(r[8] for r in rows),
                                  max(r[8] for r in rows)))
    survive = [r for r in rows if r[7] > 1.0]
    print("  Configurations still won against the blocked baseline: {}/{}"
          .format(len(survive), len(rows)))
    if survive:
        print("  Smallest surviving |Q| : {}".format(min(r[2] for r in survive)))
    lost = [r for r in rows if r[7] <= 1.0]
    if lost:
        print("  Lost at |Q| = {}".format(", ".join(str(r[2]) for r in lost)))
    print("  Both baselines agree to {}".format(max(r[9] for r in rows)))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'blocked.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["n", "m", "Q", "dyn_ms", "percol_ms", "blocked_ms",
                    "vs_percol", "vs_blocked", "block_gain", "agreement",
                    "gate_err"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _apply(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
verify_ranking_ties.py
======================
Why one com-DBLP column keeps a top-10 overlap of 0.7 no matter how deep
the series is taken.

verify_ranking.py reports, on the com-DBLP subgraph, a worst-case overlap@10
of 0.700 that does NOT improve between K=10 and K=30 even as the relative
Frobenius error falls from 2.4e-2 to 3.0e-4.  An error that shrinks by two
orders of magnitude while a ranking stays put is either a bug or a tie, and
the two look identical in aggregate statistics.

NDCG@10 sits at exactly 1.00000 over the same range, which is what a tie
looks like: the items swapped in have the same converged score as the items
swapped out, so the discounted gain is unchanged.  This script checks that
directly by printing the converged scores around the rank-10 boundary of the
worst column, and counting how many nodes share the boundary value.

Usage:
    python benchmarks/verify_ranking_ties.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from per_query import PerQueryCoSimRank
from bench_perquery import static_columns, subgraph, C, K_GATE

SEED = 42
QSIZE = 20
K_TEST = 20


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def main():
    G = subgraph('dblp', 500)
    n, m = G.number_of_nodes(), G.number_of_edges()
    rng = random.Random(SEED)
    Q = rng.sample(sorted(G.nodes()), QSIZE)
    u, v = insertable_edge(G, rng)
    G_after = G.copy(); G_after.add_edge(u, v)

    ref, ref_nodes = static_columns(G_after, Q, C=C, K=K_GATE)
    pos = {x: i for i, x in enumerate(ref_nodes)}

    pq = PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_TEST, tol=0.0)
    pq.edge_insert([(u, v)])

    print("=" * 74)
    print("  com-DBLP n={} m={}, K={}, converged reference at K={}"
          .format(n, m, K_TEST, K_GATE))
    print("=" * 74)

    worst_q, worst_ov = None, 2.0
    for q in Q:
        want = np.array([ref[q][pos[x]] for x in pq.nodes])
        got = pq.column(q)
        mask = np.ones(n, dtype=bool); mask[pq.idx[q]] = False
        od = np.argsort(-got[mask], kind='stable')[:10]
        orr = np.argsort(-want[mask], kind='stable')[:10]
        ov = len(set(od) & set(orr)) / 10.0
        if ov < worst_ov:
            worst_ov, worst_q = ov, q

    print("\n  worst column: query {} with overlap@10 = {:.1f}"
          .format(worst_q, worst_ov))

    want = np.array([ref[worst_q][pos[x]] for x in pq.nodes])
    got = pq.column(worst_q)
    mask = np.ones(n, dtype=bool); mask[pq.idx[worst_q]] = False
    w, g = want[mask], got[mask]
    orr = np.argsort(-w, kind='stable')
    od = np.argsort(-g, kind='stable')

    print("\n  Converged scores around the rank-10 boundary:")
    print("  {:>5}{:>8}{:>22}{:>22}".format("rank", "node", "converged",
                                            "dynamic (K={})".format(K_TEST)))
    print("  " + "-" * 57)
    for r in range(14):
        i = orr[r]
        mark = "  <-- boundary" if r == 9 else ""
        print("  {:>5}{:>8}{:>22.17f}{:>22.17f}{}".format(
            r + 1, i, w[i], g[i], mark))

    boundary = w[orr[9]]
    tied = int(np.sum(w == boundary))
    close = int(np.sum(np.abs(w - boundary) < 1e-12))
    print("\n  nodes sharing the rank-10 converged score EXACTLY : {}".format(tied))
    print("  nodes within 1e-12 of it                          : {}".format(close))

    swapped_out = [i for i in orr[:10] if i not in set(od[:10])]
    swapped_in = [i for i in od[:10] if i not in set(orr[:10])]
    print("\n  dropped from the top 10 : {}".format(swapped_out))
    print("  promoted into it        : {}".format(swapped_in))
    print("\n  {:>8}{:>22}{:>22}".format("node", "converged", "dynamic"))
    print("  " + "-" * 52)
    for i in swapped_out + swapped_in:
        print("  {:>8}{:>22.17f}{:>22.17f}".format(i, w[i], g[i]))

    if swapped_out and swapped_in:
        gaps = [abs(w[a] - w[b]) for a in swapped_out for b in swapped_in]
        print("\n  largest converged-score gap across the swap: {:.3e}"
              .format(max(gaps)))
        print("  verdict: {}".format(
            "TIE -- the swapped items are indistinguishable in the converged "
            "scores" if max(gaps) < 1e-12 else
            "NOT a tie -- the ranking genuinely differs"))


if __name__ == "__main__":
    main()

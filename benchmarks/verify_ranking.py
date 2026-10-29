# -*- coding: utf-8 -*-
"""
verify_ranking.py
=================
Does the truncation error change the answer?

Every accuracy figure in this report is a norm -- Frobenius or max -- on the
matrix S.  A similarity SEARCH system does not return S; it returns a ranked
list.  A norm can be small while the ordering changes, and it can be large
while the ordering does not, so a matrix norm is evidence about the ranking
only indirectly.  This script measures the ranking directly.

For each maintained column the dynamically updated scores at depth K are
compared with a converged static recomputation (K = 130, truncation below
1e-12) on the same graph state:

  overlap@k   |top-k(dynamic) INTERSECT top-k(converged)| / k
  order@k     1 if the top-k appear in exactly the same order, else 0
  NDCG@k      converged scores used as gains, ranked by the dynamic scores
  tau         Kendall's tau over the converged top-k

K is swept so that the depth at which the ranking first breaks can be
located, rather than assumed to be below the operating point.  The query node
itself is excluded from every list, since S(q,q) = 1 by construction and
ranking it would flatter every method equally.

Usage:
    python benchmarks/verify_ranking.py
"""

import sys, os, csv, random, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
from scipy.stats import kendalltau

from per_query import PerQueryCoSimRank
from bench_perquery import (static_columns, subgraph, C, K_GATE)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
QSIZE = 20
K_VALUES = (2, 3, 5, 10, 15, 20, 30)
K_LIST = (1, 5, 10, 20)
CONFIGS = [('facebook', 1000), ('dblp', 500)]


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def ndcg(order_by_dyn, gains_ref, k):
    """NDCG@k with the converged scores as gains."""
    g = gains_ref[order_by_dyn[:k]]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((g * disc).sum())
    ideal = np.sort(gains_ref)[::-1][:k]
    idcg = float((ideal * disc).sum())
    return 1.0 if idcg == 0 else dcg / idcg


def compare(dyn_col, ref_col, self_idx):
    """Ranking agreement between two score vectors, self excluded."""
    mask = np.ones(len(ref_col), dtype=bool)
    mask[self_idx] = False
    d, r = dyn_col[mask], ref_col[mask]
    od, orr = np.argsort(-d, kind='stable'), np.argsort(-r, kind='stable')

    out = {}
    for k in K_LIST:
        kk = min(k, len(d))
        out['overlap@{}'.format(k)] = len(set(od[:kk]) & set(orr[:kk])) / kk
        out['order@{}'.format(k)] = float(np.array_equal(od[:kk], orr[:kk]))
        out['ndcg@{}'.format(k)] = ndcg(od, r, kk)
    top = orr[:20]
    t = kendalltau(d[top], r[top]).correlation
    out['tau@20'] = 1.0 if np.isnan(t) else float(t)
    return out


def main():
    print("=" * 86)
    print("  Does the truncation error change the ranking?")
    print("  Dynamic update at depth K against a converged recomputation"
          " (K={}).".format(K_GATE))
    print("  Means over {} maintained columns; self excluded."
          .format(QSIZE))
    print("=" * 86)

    rows = []
    for dataset, n_req in CONFIGS:
        G = subgraph(dataset, n_req)
        n, m = G.number_of_nodes(), G.number_of_edges()
        rng = random.Random(SEED)
        Q = rng.sample(sorted(G.nodes()), QSIZE)
        u, v = insertable_edge(G, rng)
        G_after = G.copy(); G_after.add_edge(u, v)

        # converged reference on the post-update graph
        ref, ref_nodes = static_columns(G_after, Q, C=C, K=K_GATE)
        pos = {x: i for i, x in enumerate(ref_nodes)}

        print("\n  {} n={} m={}: insert {}->{}".format(dataset, n, m, u, v))
        print("  {:>4}{:>12}{:>11}{:>11}{:>11}{:>11}{:>10}{:>11}".format(
            "K", "rel Frob", "ovl@1", "ovl@10", "order@10", "ndcg@10",
            "tau@20", "worst ovl@10"))
        print("  " + "-" * 82)

        for K in K_VALUES:
            pq = PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K,
                                   tol=0.0)
            pq.edge_insert([(u, v)])

            accs, fro_num, fro_den = [], 0.0, 0.0
            for q in Q:
                want = np.array([ref[q][pos[x]] for x in pq.nodes])
                got = pq.column(q)
                fro_num += float(((got - want) ** 2).sum())
                fro_den += float((want ** 2).sum())
                accs.append(compare(got, want, pq.idx[q]))

            fro = (fro_num ** 0.5) / (fro_den ** 0.5)
            mean = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
            worst10 = min(a['overlap@10'] for a in accs)
            print("  {:>4}{:>12.2e}{:>11.3f}{:>11.3f}{:>11.3f}{:>11.5f}"
                  "{:>10.4f}{:>11.3f}"
                  .format(K, fro, mean['overlap@1'], mean['overlap@10'],
                          mean['order@10'], mean['ndcg@10'], mean['tau@20'],
                          worst10))
            rows.append([dataset, n, K, "{:.3e}".format(fro)]
                        + [round(mean[k], 6) for k in sorted(mean)]
                        + [round(worst10, 4)])

    print("\n  " + "=" * 82)
    print("  Read the ndcg@10 and order@10 columns together: NDCG can sit at")
    print("  1.00000 while order@10 is below 1, which means the list is")
    print("  reordered only among items of indistinguishable score.")

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'ranking.csv')
    keys = sorted(compare(np.zeros(3), np.zeros(3), 0))
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "K", "rel_frobenius"] + keys
                   + ["worst_overlap@10"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

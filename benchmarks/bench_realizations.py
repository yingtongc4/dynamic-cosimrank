# -*- coding: utf-8 -*-
"""
bench_realizations.py
=====================
Three subgraphs per size instead of one.

The extraction in data/dataset_stats.py is deterministic: degree filter,
largest component, breadth-first search from the highest-degree node.  So
"the n=1000 subgraph" is one graph, not a sample of one, and every conclusion
drawn across sizes -- the n-sweep, the monotonicity of the dynamic cost, the
observation that mean degree rises with n -- rests on a single realisation.
Mean degree is the only free variable in the break-even, so this matters more
than it would for most parameters.

Changing the breadth-first starting node gives an independent draw from the
same recipe.  Three starts per size are run here: the highest-degree node as
before, and two chosen at random among nodes of degree at least ten.

Usage:
    python benchmarks/bench_realizations.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import networkx as nx

from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            C, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
QSIZE = 20
SIZES = (200, 500, 1000)
STARTS = 3


def extract_from(G, start, max_nodes, min_degree=10):
    high = [v for v, d in G.degree() if d >= min_degree]
    H = G.subgraph(high).copy()
    comp = max(nx.connected_components(H), key=len)
    largest = H.subgraph(comp).copy()
    if start not in largest:
        start = max(largest.degree(), key=lambda x: x[1])[0]
    order = list(nx.bfs_tree(largest, start).nodes())[:max_nodes]
    sub = largest.subgraph(order).copy()
    return nx.convert_node_labels_to_integers(sub, ordering='sorted')


def starts_for(G, k, rng, min_degree=10):
    high = [v for v, d in G.degree() if d >= min_degree]
    H = G.subgraph(high).copy()
    comp = max(nx.connected_components(H), key=len)
    largest = H.subgraph(comp).copy()
    hub = max(largest.degree(), key=lambda x: x[1])[0]
    others = rng.sample(sorted(largest.nodes()), k - 1)
    return [hub] + others


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def main():
    full = load_facebook()
    rng0 = random.Random(SEED)
    starts = starts_for(full, STARTS, rng0)

    print("=" * 80)
    print("  Three extraction realisations per size, |Q| = {}, K = {}"
          .format(QSIZE, K_DEFAULT))
    print("  Start 1 is the highest-degree node, the recipe used everywhere")
    print("  else in this report; starts 2 and 3 are random.")
    print("=" * 80)
    print("\n  {:>6}{:>8}{:>10}{:>9}{:>11}{:>13}{:>11}".format(
        "n", "start", "m", "d-bar", "dyn (ms)", "static (ms)", "speedup"))
    print("  " + "-" * 68)

    rows = []
    for n_req in SIZES:
        block = []
        for si, st in enumerate(starts, 1):
            G = extract_from(full, st, n_req).to_directed()
            n, m = G.number_of_nodes(), G.number_of_edges()
            rng = random.Random(SEED)
            Q = rng.sample(sorted(G.nodes()), QSIZE)
            u, v = insertable_edge(G, rng)
            G2 = G.copy(); G2.add_edge(u, v)
            gate(lambda qs: _probe(G, qs, u, v), G2, Q, rng,
                 "n={} start{}".format(n, si))

            pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C,
                                     K=K_DEFAULT, tol=0.0)
                   for _ in range(REPS)]
            stt = {'i': 0}

            def go():
                p = pre[stt['i']]; stt['i'] += 1
                p.edge_insert([(u, v)])

            dyn = timed(go, REPS)[0]
            A, AT, nodes2, idx2 = assemble(G2)
            qidx = [idx2[x] for x in Q]
            stat = timed(lambda: static_columns_prebuilt(
                A, AT, qidx, len(nodes2), C, K_DEFAULT), REPS)[0]
            sp = stat / dyn
            print("  {:>6}{:>8}{:>10}{:>9.1f}{:>11.2f}{:>13.2f}{:>10.2f}x"
                  .format(n, si, m, m / n, dyn, stat, sp))
            block.append((m / n, sp))
            rows.append([n, si, m, round(m / n, 2), round(dyn, 3),
                         round(stat, 3), round(sp, 4)])
        d = [x[0] for x in block]
        s = [x[1] for x in block]
        print("  {:>6}{:>8}{:>10}{:>9}{:>11}{:>13}{:>11}".format(
            "", "spread", "", "{:.2f}x".format(max(d) / min(d)), "", "",
            "{:.2f}x".format(max(s) / min(s))))
        print("  " + "-" * 68)

    print("\n  mean degree across all realisations: {:.1f} to {:.1f}"
          .format(min(r[3] for r in rows), max(r[3] for r in rows)))
    print("  the break-even 2d/(d-1) therefore ranges {:.2f} to {:.2f}"
          .format(*sorted([2 * d / (d - 1) for d in
                           (max(r[3] for r in rows),
                            min(r[3] for r in rows))])))
    for n_req in SIZES:
        sub = [r for r in rows if r[0] == n_req or abs(r[0] - n_req) < 5]
        if sub:
            print("  n={:<6} speedup {:.2f} to {:.2f} across starts"
                  .format(n_req, min(r[6] for r in sub),
                          max(r[6] for r in sub)))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'realizations.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["n", "start", "m", "d_bar", "dyn_ms", "static_ms",
                    "speedup"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _probe(G, qs, u, v):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=130, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

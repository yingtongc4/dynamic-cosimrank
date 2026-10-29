# -*- coding: utf-8 -*-
"""
bench_pruning.py
================
Where the exact pruning of Proposition "exact pruning" actually fires.

The case study measures it firing zero times in nine runs on ego-Facebook:
every one of 1000 columns marked touched, no query skipped.  That is the
expected outcome on a graph with mean degree 55 and diameter 4 -- twenty steps
of a backward walk reach everything -- but it leaves the proposition listed as
a contribution with no graph on which it does anything.

The property it needs is large diameter relative to K, not sparsity.  Road
networks are the standard example: approximately planar, degree about 3, and
diameters in the hundreds.  A 2-D grid is the same shape of object and can be
generated rather than downloaded, so this script uses grids of increasing side
length alongside a chain of communities, and reports for each how much of the
graph one edge update touches.

If pruning fires, Proposition "exact pruning" earns its place; if it does not,
it should be recorded as an observation rather than a contribution.  Both
outcomes are reportable, and the script does not choose between them.

Usage:
    python benchmarks/bench_pruning.py
"""

import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import networkx as nx

from per_query import PerQueryCoSimRank
from bench_perquery import (assemble, static_columns_prebuilt, timed, gate,
                            subgraph, C, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
QSIZE = 20


def grid(side):
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G)
    return G.to_directed()


def chain_of_communities(k, size):
    """k cliques of `size`, joined in a line by single edges."""
    G = nx.Graph()
    for c in range(k):
        base = c * size
        for i in range(size):
            for j in range(i + 1, size):
                G.add_edge(base + i, base + j)
        if c:
            G.add_edge(base - 1, base)
    return G.to_directed()


def two_components(side):
    """Two disjoint grids.  No path between them, so S is block-diagonal."""
    A, B = grid(side), grid(side)
    B = nx.relabel_nodes(B, {v: v + A.number_of_nodes() for v in B})
    G = nx.DiGraph()
    G.add_edges_from(A.edges()); G.add_edges_from(B.edges())
    return G


def dag_layers(width, depth):
    """A layered DAG: edges only go forward, so most pairs share no ancestor."""
    G = nx.DiGraph()
    for L in range(depth - 1):
        for i in range(width):
            for j in range(width):
                if (i + j) % 3 == 0:
                    G.add_edge(L * width + i, (L + 1) * width + j)
    return G


def graphs():
    yield ("grid 20x20", grid(20), None)
    yield ("grid 45x45", grid(45), None)
    yield ("chain 40x10", chain_of_communities(40, 10), None)
    yield ("2 grids, far Q", two_components(20), 'far')
    yield ("layered DAG", dag_layers(40, 25), None)
    yield ("ego-Facebook", subgraph('facebook', 1000), None)


def diameter(G):
    U = G.to_undirected()
    if not nx.is_connected(U):
        U = U.subgraph(max(nx.connected_components(U), key=len))
    # exact diameter is O(nm); a double-sweep lower bound is enough here
    a = max(nx.single_source_shortest_path_length(U, next(iter(U))).items(),
            key=lambda x: x[1])[0]
    d = max(nx.single_source_shortest_path_length(U, a).values())
    return d


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(200000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def insertable_edge_within(G, rng, lo, hi):
    cand = [x for x in sorted(G.nodes()) if lo <= x < hi]
    for _ in range(200000):
        u, v = rng.choice(cand), rng.choice(cand)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge in range")


def main():
    print("=" * 92)
    print("  Does exact pruning ever fire?  |Q| = {}, K = {}, one edge"
          " inserted.".format(QSIZE, K_DEFAULT))
    print("  'touched' is the number of columns the update cannot prove zero.")
    print("=" * 92)
    print("\n  {:<15}{:>6}{:>8}{:>7}{:>7}{:>12}{:>10}{:>10}{:>11}".format(
        "graph", "n", "m", "d-bar", "diam", "touched", "%", "skipped",
        "speedup"))
    print("  " + "-" * 88)

    rows = []
    for label, G, mode in graphs():
        n, m = G.number_of_nodes(), G.number_of_edges()
        diam = diameter(G)
        rng = random.Random(SEED)
        if mode == 'far':
            # change an edge in the first component, query only the second
            half = n // 2
            Q = rng.sample(sorted(v for v in G.nodes() if v >= half), QSIZE)
            u, v = insertable_edge_within(G, rng, 0, half)
        else:
            Q = rng.sample(sorted(G.nodes()), QSIZE)
            u, v = insertable_edge(G, rng)
        G_after = G.copy(); G_after.add_edge(u, v)

        gate(lambda qs: _apply(G, qs, u, v), G_after, Q, rng, label)

        probe = _apply(G, list(Q), u, v, K_DEFAULT)
        touched, skipped = probe.last_touched, probe.last_skipped

        pre = [PerQueryCoSimRank(G.copy(), queries=list(Q), C=C, K=K_DEFAULT,
                                 tol=0.0) for _ in range(REPS)]
        st = {'i': 0}

        def go():
            p = pre[st['i']]; st['i'] += 1
            p.edge_insert([(u, v)])

        dyn = timed(go, REPS)[0]
        A, AT, nodes2, idx2 = assemble(G_after)
        qidx = [idx2[q] for q in Q]
        stat = timed(lambda: static_columns_prebuilt(A, AT, qidx, len(nodes2),
                                                     C, K_DEFAULT), REPS)[0]

        print("  {:<15}{:>6}{:>8}{:>7.1f}{:>7}{:>12}{:>9.1f}%{:>7}/{:<3}"
              "{:>10.2f}x".format(label, n, m, m / n, diam, touched,
                                  100.0 * touched / n, skipped, QSIZE,
                                  stat / dyn))
        rows.append([label, n, m, round(m / n, 2), diam, touched,
                     round(100.0 * touched / n, 2), skipped, QSIZE,
                     round(dyn, 3), round(stat, 3), round(stat / dyn, 4)])

    print("  " + "-" * 88)
    fired = [r for r in rows if r[7] > 0]
    print("\n  graphs on which pruning skipped at least one query: {}/{}"
          .format(len(fired), len(rows)))
    for r in fired:
        print("    {:<15} diameter {:>4}, {} of {} queries skipped,"
              " {:.1f}% of columns touched"
              .format(r[0], r[4], r[7], r[8], r[6]))
    if not fired:
        print("    none -- Proposition 'exact pruning' fires on no graph tested")

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'pruning.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["graph", "n", "m", "d_bar", "diameter", "touched",
                    "touched_pct", "skipped", "Q", "dyn_ms", "static_ms",
                    "speedup"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


def _apply(G, qs, u, v, K=130):
    pq = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=K, tol=0.0)
    pq.edge_insert([(u, v)])
    return pq


if __name__ == "__main__":
    main()

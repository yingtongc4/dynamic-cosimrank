"""
Scalability correctness tests: verify dynamic updates match static recomputation
across a range of graph sizes and random seeds.

Tests edge insertion, edge deletion, node insertion, and node deletion on
Barabasi-Albert random graphs with n in {20, 50, 100}.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
import random
from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank

DECAY = 0.8
K     = 30
TOL   = 0.05   # generous: accounts for fixed-K truncation gap
SIZES = [20, 50, 100]
SEEDS = [0, 1, 2]


def make_graph(n, seed):
    """Barabasi-Albert directed graph with m=3 edges per new node."""
    rng = random.Random(seed)
    G_undirected = nx.barabasi_albert_graph(n, 3, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())
    for u, v in G_undirected.edges():
        if rng.random() < 0.5:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)
    return G


def frob(A, B):
    return np.linalg.norm(A - B, 'fro')


def test_edge_insert_random():
    """Edge insertion on random graphs: dynamic matches static within tolerance."""
    for n in SIZES:
        for seed in SEEDS:
            G = make_graph(n, seed)
            rng = random.Random(seed + 100)
            nodes = list(G.nodes())
            dcsr = DynamicCoSimRank(G.copy(), C=DECAY, K=K)

            inserted = False
            for _ in range(200):
                u, v = rng.choice(nodes), rng.choice(nodes)
                if u != v and not dcsr.G.has_edge(u, v):
                    dcsr.edge_insert([(u, v)])
                    inserted = True
                    break

            if not inserted:
                continue

            err = dcsr.frobenius_error()
            assert err < TOL, f"edge_insert n={n} seed={seed}: err={err:.4e}"
    print(f"[PASS] edge_insert  (sizes={SIZES}, seeds={SEEDS})")


def test_edge_delete_random():
    """Edge deletion on random graphs: dynamic matches static within tolerance."""
    for n in SIZES:
        for seed in SEEDS:
            G = make_graph(n, seed)
            if G.number_of_edges() < 5:
                continue
            rng = random.Random(seed + 200)
            dcsr = DynamicCoSimRank(G.copy(), C=DECAY, K=K)

            e = rng.choice(list(dcsr.G.edges()))
            dcsr.edge_delete([e])

            err = dcsr.frobenius_error()
            assert err < TOL, f"edge_delete n={n} seed={seed}: err={err:.4e}"
    print(f"[PASS] edge_delete  (sizes={SIZES}, seeds={SEEDS})")


def test_node_insert_random():
    """Node insertion on random graphs: dynamic matches static within tolerance."""
    for n in SIZES:
        for seed in SEEDS:
            G = make_graph(n, seed)
            rng = random.Random(seed + 300)
            nodes = list(G.nodes())
            new_node = max(nodes) + 1

            new_edges = []
            for _ in range(3):
                v = rng.choice(nodes)
                if rng.random() < 0.5:
                    new_edges.append((new_node, v))
                else:
                    new_edges.append((v, new_node))

            dcsr = DynamicCoSimRank(G.copy(), C=DECAY, K=K)
            dcsr.node_insert(new_node, new_edges)

            err = dcsr.frobenius_error()
            assert err < TOL, f"node_insert n={n} seed={seed}: err={err:.4e}"
    print(f"[PASS] node_insert  (sizes={SIZES}, seeds={SEEDS})")


def test_node_delete_random():
    """Node deletion on random graphs: dynamic matches static within tolerance."""
    for n in SIZES:
        for seed in SEEDS:
            G = make_graph(n, seed)
            rng = random.Random(seed + 400)
            nodes = list(G.nodes())

            target = rng.choice(nodes)
            dcsr = DynamicCoSimRank(G.copy(), C=DECAY, K=K)
            dcsr.node_delete(target)

            err = dcsr.frobenius_error()
            assert err < TOL, f"node_delete n={n} seed={seed}: err={err:.4e}"
    print(f"[PASS] node_delete  (sizes={SIZES}, seeds={SEEDS})")


def test_error_decreases_with_k():
    """
    Frobenius error should decrease as K increases (truncation hypothesis).
    Tests on n=50, one random graph, edge insertion.
    """
    G = make_graph(50, seed=7)
    rng = random.Random(42)
    nodes = list(G.nodes())

    u, v = None, None
    for _ in range(200):
        a, b = rng.choice(nodes), rng.choice(nodes)
        if a != b and not G.has_edge(a, b):
            u, v = a, b
            break

    if u is None:
        print("[SKIP] test_error_decreases_with_k: no insertable edge found")
        return

    errors = {}
    for k in [10, 20, 30, 50]:
        dcsr = DynamicCoSimRank(G.copy(), C=DECAY, K=k)
        dcsr.edge_insert([(u, v)])
        errors[k] = dcsr.frobenius_error()

    print(f"[INFO] Frobenius error vs K: {errors}")
    assert errors[50] <= errors[10] + 1e-3, \
        f"Error should not increase with K: {errors}"
    print("[PASS] error_decreases_with_k")


if __name__ == "__main__":
    print("=" * 55)
    print("  Scalability Correctness Tests")
    print("=" * 55 + "\n")

    test_edge_insert_random()
    test_edge_delete_random()
    test_node_insert_random()
    test_node_delete_random()
    test_error_decreases_with_k()

    print("\n[ALL PASS]\n")

"""
Correctness tests for D-CoSim (edge insertion).

Strategy for each test:
  1. Build a small graph G and compute static CoSimRank S_old.
  2. Insert edges via d_cosim to get S_dynamic.
  3. Recompute static CoSimRank on the updated graph G' to get S_ref.
  4. Verify ||S_dynamic - S_ref||_F < tolerance.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from simrank import cosimrank_matrix
from d_cosim import d_cosim


DECAY = 0.6
ITERS = 30
TOL   = 1e-4


def build_base_graph():
    """Base graph from WWW 2018 Figure 1 (solid arrows)."""
    G = nx.DiGraph()
    G.add_edges_from([
        ('a', 'b'), ('a', 'c'),
        ('b', 'd'),
        ('c', 'd'), ('c', 'e'),
        ('d', 'e'),
    ])
    return G


def recompute(G, C=DECAY, K=ITERS):
    S, nodes, idx = cosimrank_matrix(G, C=C, K=K)
    return S, nodes, idx


def frobenius_error(A, B):
    return np.linalg.norm(A - B, 'fro')


def test_single_edge_insertion():
    G = build_base_graph()
    S_old, nodes, idx = recompute(G)

    new_edge = [('b', 'e')]
    S_dyn, G_new = d_cosim(G, S_old, new_edge, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Single edge] Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Single edge insertion")


def test_multiple_edges_same_target():
    """Multiple inserted edges sharing the same destination (one piece)."""
    G = build_base_graph()
    S_old, nodes, idx = recompute(G)

    new_edges = [('a', 'e'), ('b', 'e')]
    S_dyn, G_new = d_cosim(G, S_old, new_edges, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Same target]  Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Multiple edges same target")


def test_multiple_edges_diff_targets():
    """Multiple inserted edges with different destinations (multiple pieces)."""
    G = build_base_graph()
    S_old, nodes, idx = recompute(G)

    new_edges = [('a', 'e'), ('c', 'b')]
    S_dyn, G_new = d_cosim(G, S_old, new_edges, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Diff targets] Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Multiple edges different targets")


def test_symmetry_preserved():
    G = build_base_graph()
    S_old, nodes, idx = recompute(G)

    new_edges = [('a', 'e'), ('b', 'c')]
    S_dyn, _ = d_cosim(G, S_old, new_edges, C=DECAY, K=ITERS)

    assert np.allclose(S_dyn, S_dyn.T, atol=1e-6), "S_dynamic is not symmetric"
    print("[PASS] Symmetry preserved after insertion")


def test_sequential_insertions():
    """Two sequential single-edge insertions must match a single batch recompute."""
    G = build_base_graph()
    S, _, _ = recompute(G)

    edge1 = [('a', 'e')]
    edge2 = [('b', 'c')]

    S, G = d_cosim(G, S, edge1, C=DECAY, K=ITERS)
    S, G = d_cosim(G, S, edge2, C=DECAY, K=ITERS)

    G_ref = build_base_graph()
    G_ref.add_edges_from(edge1 + edge2)
    S_ref, _, _ = recompute(G_ref)

    err = frobenius_error(S, S_ref)
    print(f"[Sequential]   Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Sequential insertions")


if __name__ == "__main__":
    print("=" * 50)
    print("  D-CoSim Edge Insertion Tests")
    print("=" * 50 + "\n")

    test_single_edge_insertion()
    test_multiple_edges_same_target()
    test_multiple_edges_diff_targets()
    test_symmetry_preserved()
    test_sequential_insertions()

    print("\n[ALL PASS]\n")

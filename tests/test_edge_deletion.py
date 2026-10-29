"""
Correctness tests for edge deletion.

Strategy for each test:
  1. Build a small graph G and compute static CoSimRank S_old.
  2. Delete edges via edge_deletion to get S_dynamic.
  3. Recompute static CoSimRank on the pruned graph G' to get S_ref.
  4. Verify ||S_dynamic - S_ref||_F < tolerance.

Edge deletion is treated as a signed edge insertion (negative rank-one
column perturbation), so it is exact, not approximate.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from simrank import cosimrank_matrix
from edge_deletion import edge_deletion


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


def test_single_edge_deletion():
    G = build_base_graph()
    S_old, nodes, idx = recompute(G)

    del_edge = [('a', 'c')]
    S_dyn, G_new = edge_deletion(G, S_old, del_edge, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Single edge del]   Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Single edge deletion")


def test_multiple_edges_same_target():
    """Multiple in-edges to the same destination deleted at once (one piece)."""
    G = nx.DiGraph()
    G.add_edges_from([
        ('a', 'e'), ('b', 'e'), ('c', 'e'),
        ('a', 'b'), ('b', 'c'),
    ])
    S_old, _, _ = recompute(G)

    del_edges = [('a', 'e'), ('b', 'e')]
    S_dyn, G_new = edge_deletion(G, S_old, del_edges, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Same target del]   Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Multiple edges same target deletion")


def test_multiple_edges_diff_targets():
    """Edges with different destinations deleted at once (multiple pieces)."""
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    del_edges = [('a', 'b'), ('c', 'e')]
    S_dyn, G_new = edge_deletion(G, S_old, del_edges, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Diff targets del]  Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Multiple edges different targets deletion")


def test_delete_to_isolated():
    """Delete all in-edges of a node, reducing its in-degree to 0."""
    G = nx.DiGraph()
    G.add_edges_from([
        ('a', 'c'), ('b', 'c'),
        ('c', 'd'),
    ])
    S_old, _, _ = recompute(G)

    del_edges = [('a', 'c'), ('b', 'c')]
    S_dyn, G_new = edge_deletion(G, S_old, del_edges, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Isolated target]   Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Delete to isolated node (in-degree = 0)")


def test_symmetry_preserved():
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    del_edges = [('a', 'c'), ('b', 'd')]
    S_dyn, _ = edge_deletion(G, S_old, del_edges, C=DECAY, K=ITERS)

    assert np.allclose(S_dyn, S_dyn.T, atol=1e-6), "S_dynamic is not symmetric"
    print("[PASS] Symmetry preserved after deletion")


def test_sequential_deletions():
    """Two sequential single-edge deletions must match a batch recompute."""
    G = build_base_graph()
    S, _, _ = recompute(G)

    edge1 = [('a', 'b')]
    edge2 = [('c', 'd')]

    S, G = edge_deletion(G, S, edge1, C=DECAY, K=ITERS)
    S, G = edge_deletion(G, S, edge2, C=DECAY, K=ITERS)

    G_ref = build_base_graph()
    G_ref.remove_edges_from(edge1 + edge2)
    S_ref, _, _ = recompute(G_ref)

    err = frobenius_error(S, S_ref)
    print(f"[Sequential del]    Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Sequential deletions")


def test_invalid_edge_raises():
    """Deleting a non-existent edge must raise ValueError."""
    G = build_base_graph()
    S, _, _ = recompute(G)

    try:
        edge_deletion(G, S, [('x', 'y')], C=DECAY, K=ITERS)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] Invalid edge raises ValueError")


if __name__ == "__main__":
    print("=" * 50)
    print("  Edge Deletion Tests")
    print("=" * 50 + "\n")

    test_single_edge_deletion()
    test_multiple_edges_same_target()
    test_multiple_edges_diff_targets()
    test_delete_to_isolated()
    test_symmetry_preserved()
    test_sequential_deletions()
    test_invalid_edge_raises()

    print("\n[ALL PASS]\n")

"""
Correctness tests for node insertion and node deletion.

Strategy for each test:
  1. Compute static CoSimRank S_old on the base graph G.
  2. Apply the incremental operation to get S_dynamic.
  3. Recompute static CoSimRank on the modified graph G' to get S_ref.
  4. Verify ||S_dynamic - S_ref||_F < tolerance.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from simrank import cosimrank_matrix
from node_ops import node_insertion, node_deletion


DECAY = 0.6
ITERS = 30
TOL   = 1e-4


def build_base_graph():
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


# ------------------------------------------------------------------
# Node Insertion Tests
# ------------------------------------------------------------------

def test_insert_isolated_node():
    """Inserting an isolated node: new node must have similarity 0 with all others."""
    G = build_base_graph()
    S_old, nodes, idx = recompute(G)

    S_dyn, G_new = node_insertion(G, S_old, 'z', [], C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Insert isolated]     Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Insert isolated node")


def test_insert_node_with_in_edge():
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, G_new = node_insertion(G, S_old, 'z', [('a', 'z')], C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Insert w/ in-edge]   Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Insert node with in-edge")


def test_insert_node_with_out_edge():
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, G_new = node_insertion(G, S_old, 'z', [('z', 'd')], C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Insert w/ out-edge]  Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Insert node with out-edge")


def test_insert_node_with_multiple_edges():
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    new_edges = [('a', 'z'), ('c', 'z'), ('z', 'd')]
    S_dyn, G_new = node_insertion(G, S_old, 'z', new_edges, C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Insert multi-edge]   Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Insert node with multiple edges")


def test_insert_symmetry():
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, _ = node_insertion(G, S_old, 'z', [('a', 'z'), ('z', 'e')], C=DECAY, K=ITERS)
    assert np.allclose(S_dyn, S_dyn.T, atol=1e-6), "S not symmetric after node insertion"
    print("[PASS] Symmetry preserved after node insertion")


def test_insert_invalid_node():
    """Inserting an already-existing node must raise ValueError."""
    G = build_base_graph()
    S_old, _, _ = recompute(G)
    try:
        node_insertion(G, S_old, 'a', [], C=DECAY, K=ITERS)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] Insert existing node raises ValueError")


# ------------------------------------------------------------------
# Node Deletion Tests
# ------------------------------------------------------------------

def test_delete_leaf_node():
    """Delete leaf node e (in-edges only, no out-edges)."""
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, G_new = node_deletion(G, S_old, 'e', C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Delete leaf node]    Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Delete leaf node")


def test_delete_source_node():
    """Delete source node a (out-edges only, no in-edges)."""
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, G_new = node_deletion(G, S_old, 'a', C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Delete source node]  Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Delete source node")


def test_delete_internal_node():
    """Delete internal node c (both in-edges and out-edges)."""
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, G_new = node_deletion(G, S_old, 'c', C=DECAY, K=ITERS)
    S_ref, _, _  = recompute(G_new)

    err = frobenius_error(S_dyn, S_ref)
    print(f"[Delete internal]     Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Delete internal node")


def test_delete_symmetry():
    G = build_base_graph()
    S_old, _, _ = recompute(G)

    S_dyn, _ = node_deletion(G, S_old, 'd', C=DECAY, K=ITERS)
    assert np.allclose(S_dyn, S_dyn.T, atol=1e-6), "S not symmetric after node deletion"
    print("[PASS] Symmetry preserved after node deletion")


def test_delete_invalid_node():
    """Deleting a non-existent node must raise ValueError."""
    G = build_base_graph()
    S_old, _, _ = recompute(G)
    try:
        node_deletion(G, S_old, 'z', C=DECAY, K=ITERS)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] Delete nonexistent node raises ValueError")


def test_insert_then_delete():
    """Insert a node then delete it; S must return to the original state."""
    G = build_base_graph()
    S_old, nodes_old, _ = recompute(G)

    S_ins, G_ins = node_insertion(G, S_old, 'z', [('a', 'z')], C=DECAY, K=ITERS)
    S_del, G_del = node_deletion(G_ins, S_ins, 'z', C=DECAY, K=ITERS)

    assert sorted(G_del.nodes()) == sorted(G.nodes()), "Node set mismatch"

    err = frobenius_error(S_del, S_old)
    print(f"[Insert->Delete]      Frobenius error = {err:.6f}")
    assert err < TOL, f"Error too large: {err}"
    print("[PASS] Insert then delete returns to original S")


if __name__ == "__main__":
    print("=" * 55)
    print("  Node Insertion Tests")
    print("=" * 55 + "\n")

    test_insert_isolated_node()
    test_insert_node_with_in_edge()
    test_insert_node_with_out_edge()
    test_insert_node_with_multiple_edges()
    test_insert_symmetry()
    test_insert_invalid_node()

    print("\n" + "=" * 55)
    print("  Node Deletion Tests")
    print("=" * 55 + "\n")

    test_delete_leaf_node()
    test_delete_source_node()
    test_delete_internal_node()
    test_delete_symmetry()
    test_delete_invalid_node()
    test_insert_then_delete()

    print("\n[ALL PASS]\n")

"""
Integration tests for the DynamicCoSimRank unified API.

Verifies that all four operations are correctly wired together and that
the framework's convenience methods (similarity, frobenius_error, stats)
behave as expected.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank

DECAY = 0.6
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


def frobenius_error(A, B):
    return np.linalg.norm(A - B, 'fro')


def test_initialization():
    """After init, S must equal static CoSimRank on the same graph."""
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)

    S_ref, _, _ = cosimrank_matrix(G, C=DECAY, K=30)
    err = frobenius_error(dcsr.S, S_ref)
    assert err < TOL, f"Init error too large: {err}"
    print("[PASS] Initialization")


def test_edge_insert_api():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)
    dcsr.edge_insert([('b', 'e')])

    G_ref = build_base_graph()
    G_ref.add_edge('b', 'e')
    S_ref, _, _ = cosimrank_matrix(G_ref, C=DECAY, K=30)

    err = frobenius_error(dcsr.S, S_ref)
    assert err < TOL, f"Edge insert error: {err}"
    print(f"[PASS] edge_insert API  (err={err:.2e})")


def test_edge_delete_api():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)
    dcsr.edge_delete([('a', 'b')])

    G_ref = build_base_graph()
    G_ref.remove_edge('a', 'b')
    S_ref, _, _ = cosimrank_matrix(G_ref, C=DECAY, K=30)

    err = frobenius_error(dcsr.S, S_ref)
    assert err < TOL, f"Edge delete error: {err}"
    print(f"[PASS] edge_delete API  (err={err:.2e})")


def test_node_insert_api():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)
    dcsr.node_insert('z', [('a', 'z'), ('z', 'e')])

    G_ref = build_base_graph()
    G_ref.add_edges_from([('a', 'z'), ('z', 'e')])
    S_ref, _, _ = cosimrank_matrix(G_ref, C=DECAY, K=30)

    err = frobenius_error(dcsr.S, S_ref)
    assert err < TOL, f"Node insert error: {err}"
    print(f"[PASS] node_insert API  (err={err:.2e})")


def test_node_delete_api():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)
    dcsr.node_delete('c')

    G_ref = build_base_graph()
    G_ref.remove_node('c')
    S_ref, _, _ = cosimrank_matrix(G_ref, C=DECAY, K=30)

    err = frobenius_error(dcsr.S, S_ref)
    assert err < TOL, f"Node delete error: {err}"
    print(f"[PASS] node_delete API  (err={err:.2e})")


def test_similarity_query():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)

    s_cd = dcsr.similarity('c', 'd')
    nodes = dcsr.nodes
    idx = {n: i for i, n in enumerate(nodes)}
    assert abs(s_cd - dcsr.S[idx['c'], idx['d']]) < 1e-10
    print(f"[PASS] similarity('c','d') = {s_cd:.4f}")


def test_apply_stream_mixed():
    """Mixed update stream: verify final state matches batch recomputation."""
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)

    stream = [
        ('edge_insert', [('b', 'e')]),
        ('edge_delete', [('a', 'b')]),
        ('node_insert', 'z', [('c', 'z')]),
        ('edge_insert', [('z', 'd')]),
        ('node_delete', 'z'),
        ('edge_delete', [('c', 'e')]),
    ]
    timings = dcsr.apply_stream(stream)

    G_ref = build_base_graph()
    G_ref.add_edge('b', 'e')
    G_ref.remove_edge('a', 'b')
    # node z: inserted with c->z and z->d, then deleted -> net effect: z absent
    G_ref.remove_edge('c', 'e')
    S_ref, _, _ = cosimrank_matrix(G_ref, C=DECAY, K=30)

    err = frobenius_error(dcsr.S, S_ref)
    assert err < TOL, f"Stream error: {err}"
    print(f"[PASS] apply_stream (6 ops, err={err:.2e})")
    print(f"       Per-op timings: {[f'{t:.2f}ms' for t in timings]}")


def test_frobenius_error_method():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)
    err = dcsr.frobenius_error()
    assert err < TOL, f"Self-check error: {err}"
    print(f"[PASS] frobenius_error() = {err:.2e}")


def test_stats():
    G = build_base_graph()
    dcsr = DynamicCoSimRank(G, C=DECAY, K=30)
    dcsr.edge_insert([('b', 'e')])
    dcsr.edge_delete([('c', 'e')])
    dcsr.node_insert('z', [])
    dcsr.node_delete('z')

    s = dcsr.stats()
    assert s['edge_insert']['calls'] == 1
    assert s['edge_delete']['calls'] == 1
    assert s['node_insert']['calls'] == 1
    assert s['node_delete']['calls'] == 1
    for op in s:
        assert s[op]['avg_ms'] >= 0
    print("[PASS] stats() tracking correct")
    for op, d in s.items():
        print(f"       {op}: {d['calls']} call(s), avg {d['avg_ms']:.2f} ms")


if __name__ == "__main__":
    print("=" * 55)
    print("  Framework Integration Tests")
    print("=" * 55 + "\n")

    test_initialization()
    test_edge_insert_api()
    test_edge_delete_api()
    test_node_insert_api()
    test_node_delete_api()
    test_similarity_query()
    test_apply_stream_mixed()
    test_frobenius_error_method()
    test_stats()

    print("\n[ALL PASS]\n")

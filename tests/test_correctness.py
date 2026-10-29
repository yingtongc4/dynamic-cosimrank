"""
Correctness tests for SimRank and CoSimRank.

The undirected graph (represented as a bidirectional digraph) is used for
most property tests because row-normalised and column-normalised matrices
coincide, isolating the mathematical properties from directional effects.
A directed polytree from WWW 2018 Example 5 is used for numerical validation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from simrank import simrank, cosimrank_matrix, cosimrank_vector


def build_undirected_as_digraph():
    G = nx.DiGraph()
    edges = [(1,2),(2,1),(2,3),(3,2),(3,4),(4,3),(1,4),(4,1)]
    G.add_edges_from(edges)
    return G


def build_polytree_T():
    """
    Spanning polytree T from WWW 2018 Figure 2 (Example 5).
    Edges derived from the reported A matrices:
      a->c, b->c  (level 1 -> level 2)
      c->e, c->f, c->g, d->f  (level 2 -> level 3)
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ('a', 'c'), ('b', 'c'),
        ('c', 'e'), ('c', 'f'), ('c', 'g'),
        ('d', 'f'),
    ])
    return G


def test_simrank_diagonal():
    G = build_undirected_as_digraph()
    S, nodes = simrank(G, C=0.6, K=30)
    assert np.allclose(np.diag(S), 1.0), "SimRank diagonal must be 1"
    print("[PASS] SimRank diagonal = 1")


def test_simrank_symmetry():
    G = build_undirected_as_digraph()
    S, nodes = simrank(G, C=0.6, K=30)
    assert np.allclose(S, S.T, atol=1e-6), "SimRank must be symmetric"
    print("[PASS] SimRank is symmetric")


def test_cosimrank_symmetry():
    G = build_undirected_as_digraph()
    S, nodes, _ = cosimrank_matrix(G, C=0.6, K=30)
    assert np.allclose(S, S.T, atol=1e-6), "CoSimRank must be symmetric"
    print("[PASS] CoSimRank is symmetric")


def test_cosimrank_nonnegative():
    G = build_undirected_as_digraph()
    S, nodes, _ = cosimrank_matrix(G, C=0.6, K=30)
    assert np.all(S >= -1e-9), "CoSimRank scores must be >= 0"
    print("[PASS] CoSimRank scores are non-negative")


def test_cosimrank_vector_vs_matrix():
    """Vector form and matrix form must agree."""
    G = build_undirected_as_digraph()
    S_mat, nodes, idx = cosimrank_matrix(G, C=0.6, K=30)

    pairs = [(1,2), (1,3), (2,4), (3,4)]
    for u, v in pairs:
        s_vec = cosimrank_vector(G, u, v, C=0.6, K=30)
        s_mat = S_mat[idx[u], idx[v]]
        assert abs(s_vec - s_mat) < 1e-4, \
            f"Mismatch ({u},{v}): vector={s_vec:.5f}, matrix={s_mat:.5f}"
    print("[PASS] Vector form matches matrix form")


def test_cosimrank_www_example5():
    """
    Numerical check against WWW 2018 Paper Example 5 (polytree T, C=0.6).
    Expected values: S(c,c)=1.30, S(e,e)=1.78, S(e,f)=0.39,
                     S(e,g)=0.78, S(f,f)=1.34.
    """
    G = build_polytree_T()
    S, nodes, idx = cosimrank_matrix(G, C=0.6, K=50)

    expected = {
        ('c','c'): 1.30,
        ('e','e'): 1.78,
        ('e','f'): 0.39,
        ('e','g'): 0.78,
        ('f','f'): 1.34,
    }

    print("\nWWW 2018 Example 5 check (polytree T, C=0.6):")
    all_pass = True
    for (u,v), exp in expected.items():
        got = S[idx[u], idx[v]]
        ok = abs(got - exp) < 0.02
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} S({u},{v}) = {got:.4f}  (expected {exp})")
        if not ok:
            all_pass = False

    assert all_pass, "Some values do not match WWW 2018 Example 5"
    print("[PASS] WWW 2018 Example 5 values verified")


def print_matrix(S, nodes, title="CoSimRank"):
    print(f"\n{title}:")
    header = "     " + "  ".join(f"{str(n):>5}" for n in nodes)
    print(header)
    for i, u in enumerate(nodes):
        row = f"{str(u):>4} " + "  ".join(f"{S[i,j]:>5.3f}" for j in range(len(nodes)))
        print(row)


if __name__ == "__main__":
    print("=" * 50)
    print("  Correctness Tests")
    print("=" * 50)

    test_simrank_diagonal()
    test_simrank_symmetry()
    test_cosimrank_symmetry()
    test_cosimrank_nonnegative()
    test_cosimrank_vector_vs_matrix()
    test_cosimrank_www_example5()

    print("\n[ALL PASS]\n")

    G = build_polytree_T()
    S, nodes, idx = cosimrank_matrix(G, C=0.6, K=50)
    print_matrix(S, nodes, title="CoSimRank on Polytree T (C=0.6)")

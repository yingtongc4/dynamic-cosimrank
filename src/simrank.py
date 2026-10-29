"""
SimRank and CoSimRank static implementations.

Convention (consistent with WWW 2018 D-CoSim):
  A = column-normalized adjacency matrix
      A[i,j] = 1/in_deg(j)  if edge i->j,  else 0
  Each column sums to 1.

  CoSimRank matrix equation (WWW 2018 Eq.1):
      S = C * A^T * S * A + I

  Equivalent vector form:
      p_j^(k) = A @ p_j^(k-1),  p_j^(0) = e_j   (backward walk along in-edges)
      s(i,j)  = sum_{k=0}^K  C^k  <p_i^(k), p_j^(k)>
"""

import numpy as np
import networkx as nx


def build_col_normalized(G):
    """
    Build the column-normalized adjacency matrix A (WWW 2018 convention).
    A[i,j] = 1/in_deg(j)  if edge i->j.
    Columns with in_deg = 0 are all-zero.

    Returns:
        A     : (n x n) numpy array
        nodes : sorted node list (row/col ordering)
        idx   : dict mapping node -> matrix index
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    A = np.zeros((n, n))
    for v in G.nodes():
        in_nbrs = list(G.predecessors(v))
        if in_nbrs:
            j = idx[v]
            for u in in_nbrs:
                A[idx[u], j] = 1.0 / len(in_nbrs)
    return A, nodes, idx


def build_row_normalized(G):
    """
    Build the row-normalized adjacency matrix (used as SimRank baseline).
    A[i,j] = 1/out_deg(i)  if edge i->j.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    A = np.zeros((n, n))
    for u in G.nodes():
        out_nbrs = list(G.successors(u))
        if out_nbrs:
            i = idx[u]
            for v in out_nbrs:
                A[i, idx[v]] = 1.0 / len(out_nbrs)
    return A, nodes, idx


def simrank(G, C=0.8, K=10, tol=1e-6):
    """
    SimRank (matrix form, Jeh & Widom 2002):
        S^(0) = I
        S^(k) = max(C * P^T S^(k-1) P, I)  element-wise

    P is the column-normalized matrix (in-neighbour based, consistent with D-CoSim).

    Returns:
        S     : (n x n) similarity matrix
        nodes : sorted node list
    """
    P, nodes, _ = build_col_normalized(G)
    n = len(nodes)
    S = np.eye(n)

    for k in range(K):
        S_new = C * (P.T @ S @ P)
        np.fill_diagonal(S_new, 1.0)
        diff = np.linalg.norm(S_new - S, 'fro')
        if diff < tol:
            print(f"[SimRank] converged at iteration {k+1}")
            break
        S = S_new

    return S, nodes


def cosimrank_matrix(G, C=0.8, K=20, tol=1e-8):
    """
    CoSimRank (matrix form, WWW 2018 / ACL 2014):
        S^(0) = I
        S^(k) = C * A^T * S^(k-1) * A + I

    A is column-normalized.

    Returns:
        S     : (n x n) similarity matrix
        nodes : sorted node list
        idx   : dict mapping node -> matrix index
    """
    A, nodes, idx = build_col_normalized(G)
    n = len(nodes)
    S = np.eye(n)

    for k in range(K):
        S_new = C * (A.T @ S @ A) + np.eye(n)
        diff = np.linalg.norm(S_new - S, 'fro')
        if diff < tol:
            print(f"[CoSimRank matrix] converged at iteration {k+1}")
            break
        S = S_new

    return S, nodes, idx


def cosimrank_vector(G, node_i, node_j, C=0.8, K=20):
    """
    CoSimRank (vector form, ACL 2014 Eq.4):
        s(i,j) = sum_{k=0}^{K} C^k  <p_i^(k), p_j^(k)>

    p^(k) = A @ p^(k-1),  p^(0) = e_i
    A is column-normalized, so p^(k) is a k-step backward walk along in-edges.
    Result is equivalent to cosimrank_matrix[i,j].

    Returns:
        float: CoSimRank score for (node_i, node_j)
    """
    A, nodes, idx = build_col_normalized(G)

    pi = np.zeros(len(nodes)); pi[idx[node_i]] = 1.0
    pj = np.zeros(len(nodes)); pj[idx[node_j]] = 1.0

    score = 0.0
    for k in range(K):
        score += (C ** k) * np.dot(pi, pj)
        pi = A @ pi
        pj = A @ pj

    return score

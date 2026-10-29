"""
Edge Deletion: Incremental CoSimRank Update
Thesis Section III.B.2 & Algorithm 1

Key insight:
  Deleting edge (src -> dst) is mathematically equivalent to a "negative insertion"
  on column dst of the transition matrix P.
  dP = P' - P is rank-one sparse (only column dst changes), so we can reuse
  the D-CoSim framework (WWW 2018 Theorem 1) directly.

  For insertion:  w^(0) = sum(e_src) - delta * A[:,dst]
  For deletion:   w^(0) = -sum(e_src) + num_deleted * A[:,dst]   (sign-flipped)

  IMPORTANT - avoiding the 0/0 degenerate case:
    The D-CoSim derivation scales w^(0) by `total` (new in-degree) and then
    divides by `total` in t^(0). When new_deg = 0 (deleting the last in-edge),
    both numerator and denominator vanish, giving 0/0.
    Fix: work with delta_u = dA[:,dst] directly (unscaled).
    t^(0) = (C/2) * (A + A')^T * r_fu   - no division by total needed.
    This is algebraically equivalent to the scaled form when total > 0,
    and correctly handles total = 0 (isolated node after deletion).

Exactness:
  Same as D-CoSim insertion - EXACT (not approximate) because Theorem 1
  is purely algebraic and does not require delta > 0.

Convention: column-normalised A, S = C * A^T * S * A + I  (WWW 2018).
"""

import numpy as np
import networkx as nx
from collections import defaultdict


def build_adjacency(G, nodes, idx):
    """Build column-normalised adjacency matrix A."""
    n = len(nodes)
    A = np.zeros((n, n))
    for v in G.nodes():
        in_nbrs = list(G.predecessors(v))
        if in_nbrs:
            j = idx[v]
            for u in in_nbrs:
                A[idx[u], j] = 1.0 / len(in_nbrs)
    return A


def edge_deletion(G, S, deleted_edges, C=0.8, K=10, query=None, tol=1e-8):
    """
    Incremental CoSimRank update for edge deletion.

    Args:
        G             : old graph (networkx DiGraph) containing the edges to delete
        S             : old CoSimRank matrix (n x n numpy array)
        deleted_edges : list of edges to remove, e.g. [(src1,dst1), (src2,dst2)]
        C             : decay factor
        K             : max iterations (adaptive K may stop earlier)
        query         : list of query nodes to update; None = all nodes
        tol           : adaptive K threshold; stops when ||increment||_F < tol

    Returns:
        S_new : updated CoSimRank matrix
        G_new : updated networkx DiGraph (edges removed)
    """
    for (src, dst) in deleted_edges:
        if not G.has_edge(src, dst):
            raise ValueError(f"Edge ({src},{dst}) does not exist in graph.")

    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    if query is None:
        query = nodes
    Q = [idx[q] for q in query]

    A = build_adjacency(G, nodes, idx)
    delta_S = np.zeros((n, n))

    # Group deleted edges by destination node (one "piece" per column affected)
    pieces = defaultdict(list)
    for (src, dst) in deleted_edges:
        pieces[dst].append(src)

    for dst, src_list in pieces.items():
        dst_idx = idx[dst]
        num_deleted = len(src_list)
        deg_in_dst = len(list(G.predecessors(dst)))  # old in-degree
        new_deg = deg_in_dst - num_deleted            # new in-degree after deletion

        if new_deg < 0:
            raise ValueError(
                f"Cannot delete {num_deleted} edges from '{dst}' "
                f"(current in-degree = {deg_in_dst})."
            )

        # Old column of A for node dst
        old_col = A[:, dst_idx].copy()

        # New column after removing src_list from predecessors
        new_col = np.zeros(n)
        src_set = set(src_list)
        if new_deg > 0:
            for w in G.predecessors(dst):
                if w not in src_set:
                    new_col[idx[w]] = 1.0 / new_deg

        # Update transition matrix
        A_new = A.copy()
        A_new[:, dst_idx] = new_col

        # Unscaled column perturbation: delta_u = new_col - old_col
        # Avoids the 0/0 singularity when new_deg = 0.
        # (Equivalent to w^(0)/total in the insertion formula, but without the
        # total factor that would vanish.)
        delta_u = new_col - old_col

        # Forward propagation: delta_u^(k) = A^k * delta_u
        u_seq = [delta_u.copy()]
        for _ in range(K):
            u_seq.append(A @ u_seq[-1])

        # Backward iteration: r_fu ~ S * delta_u  (Neumann series, truncated at K)
        r_fu = u_seq[K].copy()
        for k in range(1, K + 1):
            r_fu = C * (A.T @ r_fu) + u_seq[K - k]

        # Backward walk from e_dst using the updated transition matrix A'
        p_seq = [np.zeros(n)]
        p_seq[0][dst_idx] = 1.0
        for _ in range(K):
            p_seq.append(A_new.T @ p_seq[-1])

        # t^(0) = (C/2) * (A + A')^T * r_fu
        # No division by total - handles new_deg = 0 correctly.
        t0 = (C / 2.0) * ((A + A_new).T @ r_fu)
        t_seq = [t0]
        for _ in range(K):
            t_seq.append(A_new.T @ t_seq[-1])

        # Pre-compute sup-norms for the adaptive-K tail bound.
        # A_new is column sub-stochastic, so ||A_new^T x||_inf <= ||x||_inf,
        # i.e. ||p^(k)||_inf and ||t^(k)||_inf are non-increasing in k.
        p_inf = [np.max(np.abs(p)) for p in p_seq]
        t_inf = [np.max(np.abs(t)) for t in t_seq]

        # Accumulate dS[:,q] for each query node q, with adaptive K
        for q_idx in Q:
            s_delta = np.zeros(n)
            for k in range(K + 1):
                ck = C ** k
                increment = ck * (
                    t_seq[k][q_idx] * p_seq[k]
                    + p_seq[k][q_idx] * t_seq[k]
                )
                s_delta += increment
                # Adaptive K: stop only when the whole REMAINING tail is
                # negligible, not when the current term happens to vanish.
                #   ||term_j||_inf <= 2 C^j ||t^(j)||_inf ||p^(j)||_inf
                # and both factors are non-increasing, so
                #   ||tail_{>k}||_inf <= 2 C^{k+1}/(1-C) ||t^(k)||_inf ||p^(k)||_inf.
                # Testing the current term alone truncates prematurely on chain-
                # or cycle-like structures, where a term can be exactly zero
                # while later terms are not.
                if k > 0:
                    tail = (2.0 * ck * C / (1.0 - C)) * t_inf[k] * p_inf[k]
                    if tail < tol:
                        break
            delta_S[:, q_idx] += s_delta

        # Carry forward the updated A for subsequent pieces
        A = A_new.copy()

    S_new = S + delta_S

    G_new = G.copy()
    G_new.remove_edges_from(deleted_edges)

    return S_new, G_new



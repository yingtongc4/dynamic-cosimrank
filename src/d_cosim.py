"""
D-CoSim: Incremental CoSimRank Update for Edge Insertion
Yu & Wang, WWW 2018, Algorithm 1

Given the old graph G, a set of inserted edges delta_G, and the old similarity
matrix S, updates S without recomputing the full matrix.

Key idea:
  Inserting edges (v -> u) only affects column u of A.
  Group delta_G by destination u into pieces {delta_G_u}.
  For each piece, the similarity change delta_S[:,q] is expressed as a linear
  combination of two walk sequences p^(k) and t^(k).

Convention (consistent with simrank.py):
  A = column-normalized adjacency matrix
  S = C * A^T * S * A + I
"""

import numpy as np
import networkx as nx
from collections import defaultdict


def _col_normalize(A_dict, n):
    """
    Build a column-normalized (n x n) array from a {(i,j): weight} dict.
    A_dict[(i,j)] = weight of edge i->j before normalisation.
    """
    A = np.zeros((n, n))
    col_sum = defaultdict(float)
    for (i, j), w in A_dict.items():
        col_sum[j] += w
    for (i, j), w in A_dict.items():
        if col_sum[j] > 0:
            A[i, j] = w / col_sum[j]
    return A


def build_adjacency(G, nodes, idx):
    """Build the column-normalized adjacency matrix from a networkx DiGraph."""
    n = len(nodes)
    A = np.zeros((n, n))
    for v in G.nodes():
        in_nbrs = list(G.predecessors(v))
        if in_nbrs:
            j = idx[v]
            for u in in_nbrs:
                A[idx[u], j] = 1.0 / len(in_nbrs)
    return A


def d_cosim(G, S, delta_G, C=0.8, K=10, query=None, tol=1e-8):
    """
    D-CoSim: incremental CoSimRank update for edge insertion.

    Implements WWW 2018 Algorithm 1 with adaptive-K early stopping.

    Args:
        G       : old graph (networkx DiGraph), edges not yet inserted
        S       : old CoSimRank matrix (n x n numpy array)
        delta_G : list of edges to insert, e.g. [(v1, u1), (v2, u2), ...]
        C       : decay factor
        K       : max iterations (adaptive K may stop earlier)
        query   : list of query nodes to update (None = all nodes)
        tol     : adaptive-K threshold; stops when ||increment||_2 < tol

    Returns:
        S_new : updated CoSimRank matrix
        G_new : updated networkx DiGraph
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    if query is None:
        query = nodes
    Q = [idx[q] for q in query]

    A = build_adjacency(G, nodes, idx)
    delta_S = np.zeros((n, n))

    # Group edges by destination (each group = one rank-one column perturbation)
    # Duplicates would inflate the new in-degree and silently corrupt dS, so
    # they are rejected rather than tolerated.
    pieces = defaultdict(list)
    seen = set()
    for (v, u) in delta_G:
        if u not in idx or v not in idx:
            raise ValueError(f"Edge ({v},{u}) contains unknown node. "
                             "Add new nodes to G before calling d_cosim.")
        if (v, u) in seen:
            raise ValueError(f"Edge ({v},{u}) appears more than once in delta_G.")
        if G.has_edge(v, u):
            raise ValueError(f"Edge ({v},{u}) already exists in G; "
                             "inserting it again would corrupt the in-degree.")
        seen.add((v, u))
        pieces[u].append(v)

    for u, src_list in pieces.items():
        u_idx = idx[u]
        delta_u = len(src_list)
        deg_in_u = len(list(G.predecessors(u)))

        # New column u of A after insertion (Lemma 1 / Eq.4 in WWW 2018)
        A_new = A.copy()
        old_col = A[:, u_idx].copy()
        new_col = np.zeros(n)

        # Rescale existing in-neighbours to new in-degree
        for i in range(n):
            if old_col[i] > 0:
                new_col[i] = deg_in_u * old_col[i]

        # Add new in-neighbours
        for v in src_list:
            new_col[idx[v]] += 1.0

        total_in = deg_in_u + delta_u
        if total_in > 0:
            new_col /= total_in
        A_new[:, u_idx] = new_col

        delta_A_col = new_col - old_col

        # Compute w^(k) sequence (Eq.8)
        # w^(0) = indicator of new neighbours - delta_u * old column
        w = np.zeros(n)
        for v in src_list:
            w[idx[v]] += 1.0
        w -= delta_u * old_col

        w_seq = [w.copy()]
        for k in range(1, K + 1):
            w = A @ w_seq[-1]
            w_seq.append(w.copy())

        # Compute r (Eq.7): backward iteration from w^(K)
        r = w_seq[K].copy()
        for k in range(1, K + 1):
            r = C * (A.T @ r) + w_seq[K - k]

        # Compute p^(k) and t^(k) sequences (Eq.6)
        p_seq = [np.zeros(n)]
        p_seq[0][u_idx] = 1.0
        for k in range(1, K + 1):
            p_seq.append(A_new.T @ p_seq[-1])

        t0 = (C / (2.0 * total_in)) * ((A + A_new).T @ r)
        t_seq = [t0]
        for k in range(1, K + 1):
            t_seq.append(A_new.T @ t_seq[-1])

        # Pre-compute sup-norms for the adaptive-K tail bound.
        # A_new is column sub-stochastic, so ||A_new^T x||_inf <= ||x||_inf,
        # i.e. ||p^(k)||_inf and ||t^(k)||_inf are non-increasing in k.
        p_inf = [np.max(np.abs(p)) for p in p_seq]
        t_inf = [np.max(np.abs(t)) for t in t_seq]

        # Accumulate delta_S[:,q] for each query node (Eq.5) with adaptive K
        for q_idx in Q:
            s_delta = np.zeros(n)
            for k in range(K + 1):
                ck = C ** k
                increment = ck * (t_seq[k][q_idx] * p_seq[k]
                                  + p_seq[k][q_idx] * t_seq[k])
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

        A = A_new.copy()

    S_new = S + delta_S

    G_new = G.copy()
    G_new.add_edges_from(delta_G)

    return S_new, G_new

"""
Node Insertion & Node Deletion: Incremental CoSimRank Update
Thesis Section III.B.5-6

Node Insertion strategy:
  1. Add new_node to the graph (no edges yet).
  2. Expand S from n x n to (n+1) x (n+1) by inserting a zero row/col at the
     sorted-order position of new_node, then set the diagonal entry to 1.
     This is exact: an isolated node has S(v_new, j) = 0 for all j != v_new
     and S(v_new, v_new) = 1 by definition of CoSimRank.
  3. Insert all edges involving new_node using d_cosim (edge insertion).
  Complexity: O(K * d_new)

Node Deletion strategy:
  1. Delete all in-edges of del_node  -> edge_deletion changes A[:,del_node].
  2. Delete all out-edges of del_node -> edge_deletion changes A[:,u] for each u.
     After both steps, del_node is isolated:
       S[del_node, j] = 0 for j != del_node,  S[del_node, del_node] = 1.
  3. Remove the corresponding row/col from S and the node from G.
  Correctness: an isolated node does not affect the CoSimRank of any other
     pair (i, j) with i, j != del_node, so the contracted submatrix is exact.
  Complexity: O(K * d_total)
"""

import numpy as np
import networkx as nx

from d_cosim import d_cosim
from edge_deletion import edge_deletion


def node_insertion(G, S, new_node, new_edges, C=0.8, K=10, tol=1e-8):
    """
    Incremental CoSimRank update for node insertion.

    Args:
        G         : old graph (networkx DiGraph), must NOT contain new_node
        S         : old CoSimRank matrix (n x n numpy array)
        new_node  : node to insert (any hashable value consistent with G's node type)
        new_edges : list of edges involving new_node,
                    e.g. [(a, new_node), (new_node, b)]
        C         : decay factor
        K         : number of iterations
        tol       : adaptive-K threshold, forwarded to d_cosim

    Returns:
        S_new : updated CoSimRank matrix ((n+1) x (n+1))
        G_new : updated networkx DiGraph
    """
    if new_node in G:
        raise ValueError(f"Node '{new_node}' already exists in graph.")
    for (src, dst) in new_edges:
        if src != new_node and dst != new_node:
            raise ValueError(
                f"Edge ({src},{dst}) does not involve new_node '{new_node}'."
            )

    # Step 1: Add isolated new_node to a copy of G
    G_new = G.copy()
    G_new.add_node(new_node)

    # Step 2: Expand S - insert zero row and zero col at the sorted-order index
    new_nodes = sorted(G_new.nodes())
    new_idx = new_nodes.index(new_node)

    S_exp = np.insert(S, new_idx, 0, axis=0)   # new row
    S_exp = np.insert(S_exp, new_idx, 0, axis=1)  # new col
    S_exp[new_idx, new_idx] = 1.0               # diagonal

    # Step 3: Insert edges with d_cosim
    if new_edges:
        S_new, G_new = d_cosim(G_new, S_exp, new_edges, C=C, K=K, tol=tol)
    else:
        S_new = S_exp

    return S_new, G_new


def node_deletion(G, S, del_node, C=0.8, K=10, tol=1e-8):
    """
    Incremental CoSimRank update for node deletion.

    Args:
        G        : old graph (networkx DiGraph) containing del_node
        S        : old CoSimRank matrix (n x n numpy array)
        del_node : node to remove
        C        : decay factor
        K        : number of iterations
        tol      : adaptive-K threshold, forwarded to edge_deletion

    Returns:
        S_new : updated CoSimRank matrix ((n-1) x (n-1))
        G_new : updated networkx DiGraph
    """
    if del_node not in G:
        raise ValueError(f"Node '{del_node}' not in graph.")

    G_curr = G.copy()
    S_curr = S.copy()

    # Step 1: Remove all in-edges  (w -> del_node)
    # All share the same destination, forming one "piece" in edge_deletion.
    in_edges = [(w, del_node) for w in list(G_curr.predecessors(del_node))]
    if in_edges:
        S_curr, G_curr = edge_deletion(G_curr, S_curr, in_edges, C=C, K=K, tol=tol)

    # Step 2: Remove all out-edges  (del_node -> u)
    # Each has a different destination, forming separate pieces.
    out_edges = [(del_node, u) for u in list(G_curr.successors(del_node))]
    if out_edges:
        S_curr, G_curr = edge_deletion(G_curr, S_curr, out_edges, C=C, K=K, tol=tol)

    # After both steps, del_node is isolated:
    #   S_curr[del_node, j] = 0  for j != del_node
    #   S_curr[del_node, del_node] = 1

    # Step 3: Contract S and G by removing the row/col for del_node
    nodes = sorted(G_curr.nodes())
    del_idx = nodes.index(del_node)

    S_new = np.delete(S_curr, del_idx, axis=0)
    S_new = np.delete(S_new, del_idx, axis=1)

    G_new = G_curr.copy()
    G_new.remove_node(del_node)

    return S_new, G_new


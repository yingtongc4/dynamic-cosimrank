"""
Extended Dynamic CoSimRank Framework -- Unified API

Integrates all four dynamic update types:
  - edge_insert  : edge insertion    (D-CoSim, WWW 2018)
  - edge_delete  : edge deletion     (this work)
  - node_insert  : node insertion
  - node_delete  : node deletion

Usage (individual operations):
    dcsr = DynamicCoSimRank(G, C=0.8, K=10)
    dcsr.edge_insert([('a', 'b'), ('c', 'd')])
    dcsr.edge_delete([('x', 'y')])
    dcsr.node_insert('z', [('a', 'z'), ('z', 'b')])
    dcsr.node_delete('z')
    S     = dcsr.S       # current similarity matrix
    G     = dcsr.G       # current graph
    nodes = dcsr.nodes   # current node list (sorted)

Usage (update stream):
    stream = [
        ('edge_insert', [('a', 'b')]),
        ('edge_delete', [('c', 'd')]),
        ('node_insert', 'z', [('a', 'z')]),
        ('node_delete', 'w'),
    ]
    timings = dcsr.apply_stream(stream)   # returns per-op wall-clock times (ms)
"""

import numpy as np
import networkx as nx
import time

from simrank import cosimrank_matrix
from d_cosim import d_cosim
from edge_deletion import edge_deletion
from node_ops import node_insertion, node_deletion


class DynamicCoSimRank:
    """
    Extended Dynamic CoSimRank Framework.

    Maintains a CoSimRank similarity matrix S for a dynamic graph,
    supporting four incremental update types without full recomputation.
    """

    def __init__(self, G, C=0.8, K=10, tol=1e-8):
        """
        Initialise by computing static CoSimRank on the starting graph G.

        Args:
            G   : initial networkx DiGraph
            C   : decay factor (0.6-0.8 recommended)
            K   : max iterations per update (truncation error <= C^(K+1))
            tol : adaptive-K threshold; iteration stops early when
                  ||increment||_2 < tol
        """
        self.C = C
        self.K = K
        self.tol = tol
        self.G = G.copy()
        self._recompute()

        self._stats = {
            'edge_insert': {'calls': 0, 'total_ms': 0.0},
            'edge_delete': {'calls': 0, 'total_ms': 0.0},
            'node_insert': {'calls': 0, 'total_ms': 0.0},
            'node_delete': {'calls': 0, 'total_ms': 0.0},
        }

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def nodes(self):
        """Sorted list of current nodes."""
        return sorted(self.G.nodes())

    @property
    def n(self):
        """Current number of nodes."""
        return len(self.G)

    def similarity(self, u, v):
        """Return the CoSimRank score for node pair (u, v)."""
        idx = {node: i for i, node in enumerate(self.nodes)}
        return self.S[idx[u], idx[v]]

    # ------------------------------------------------------------------
    # Four incremental operations
    # ------------------------------------------------------------------

    def edge_insert(self, edges):
        """
        Insert edges into the graph and update S incrementally.
        All endpoint nodes must already exist in G.
        """
        t0 = time.perf_counter()
        self.S, self.G = d_cosim(self.G, self.S, edges, C=self.C, K=self.K,
                                 tol=self.tol)
        self._record('edge_insert', t0)

    def edge_delete(self, edges):
        """
        Delete edges from the graph and update S incrementally.
        All edges must exist in G.
        """
        t0 = time.perf_counter()
        self.S, self.G = edge_deletion(self.G, self.S, edges, C=self.C, K=self.K,
                                       tol=self.tol)
        self._record('edge_delete', t0)

    def node_insert(self, new_node, new_edges=None):
        """
        Insert a new node (and its edges) and update S incrementally.
        new_node must NOT already exist in G.
        """
        if new_edges is None:
            new_edges = []
        t0 = time.perf_counter()
        self.S, self.G = node_insertion(
            self.G, self.S, new_node, new_edges, C=self.C, K=self.K, tol=self.tol
        )
        self._record('node_insert', t0)

    def node_delete(self, del_node):
        """
        Delete a node and all its incident edges, then update S incrementally.
        del_node must exist in G.
        """
        t0 = time.perf_counter()
        self.S, self.G = node_deletion(
            self.G, self.S, del_node, C=self.C, K=self.K, tol=self.tol
        )
        self._record('node_delete', t0)

    # ------------------------------------------------------------------
    # Update stream
    # ------------------------------------------------------------------

    def apply_stream(self, updates):
        """
        Process a sequence of mixed updates.

        Each item in `updates` is one of:
            ('edge_insert', [(src, dst), ...])
            ('edge_delete', [(src, dst), ...])
            ('node_insert', node)
            ('node_insert', node, [(src, dst), ...])
            ('node_delete', node)

        Returns:
            List of per-operation wall-clock times in milliseconds.
        """
        timings = []
        for item in updates:
            op = item[0]
            t0 = time.perf_counter()

            if op == 'edge_insert':
                self.S, self.G = d_cosim(
                    self.G, self.S, item[1], C=self.C, K=self.K, tol=self.tol
                )
            elif op == 'edge_delete':
                self.S, self.G = edge_deletion(
                    self.G, self.S, item[1], C=self.C, K=self.K, tol=self.tol
                )
            elif op == 'node_insert':
                edges = item[2] if len(item) > 2 else []
                self.S, self.G = node_insertion(
                    self.G, self.S, item[1], edges, C=self.C, K=self.K, tol=self.tol
                )
            elif op == 'node_delete':
                self.S, self.G = node_deletion(
                    self.G, self.S, item[1], C=self.C, K=self.K, tol=self.tol
                )
            else:
                raise ValueError(f"Unknown operation: '{op}'")

            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(elapsed_ms)
            self._stats[op]['calls'] += 1
            self._stats[op]['total_ms'] += elapsed_ms

        return timings

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def recompute(self):
        """Force a full static recomputation (use for baseline or sanity check)."""
        self._recompute()

    def frobenius_error(self):
        """
        Compute ||S_dynamic - S_static||_F against a fresh full recomputation.
        A non-zero value when K < 30 reflects expected truncation, not a bug.
        """
        S_ref, _, _ = cosimrank_matrix(self.G, C=self.C, K=30)
        return np.linalg.norm(self.S - S_ref, 'fro')

    def stats(self):
        """Return per-operation call counts and average wall-clock times (ms)."""
        return {
            op: {
                'calls': d['calls'],
                'avg_ms': d['total_ms'] / d['calls'] if d['calls'] > 0 else 0.0,
            }
            for op, d in self._stats.items()
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self):
        S, _, _ = cosimrank_matrix(self.G, C=self.C, K=30)
        self.S = S

    def _record(self, op, t0):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stats[op]['calls'] += 1
        self._stats[op]['total_ms'] += elapsed_ms

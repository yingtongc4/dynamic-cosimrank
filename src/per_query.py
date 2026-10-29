"""
Per-Query Dynamic CoSimRank  --  sparse, column-only

The full-matrix implementation maintains all of S, so one update costs
O(K n^2) and cannot beat static recomputation asymptotically.  This module
maintains ONLY the columns S[:, q] for q in a query set Q, and never
materialises an n x n matrix anywhere.

Cost per update
---------------
    build the walk sequences p^(k), t^(k)   O(K m)      shared by all queries
    accumulate dS[:, q] for each q          O(K n |Q|)
    ------------------------------------------------------------
    total                                   O(K (m + n |Q|))

against O(K m n) for a sparse static recomputation.  The advantage is real
only when |Q| << n; at |Q| = n the two collapse to the same order, which is
exactly why the full-matrix benchmark never showed an asymptotic gain.

Convention (identical to simrank.py / d_cosim.py)
    A column-normalised,  A[i,j] = 1/in_deg(j)  if edge i->j
    S = C A^T S A + I,  equivalently  S = sum_k C^k (A^T)^k A^k

Column formula
--------------
    S e_q = sum_{k=0}^{K} C^k (A^T)^k (A^k e_q)
Computed by a forward pass followed by a Horner recursion, both O(K m):
    p^(k) = A^k e_q                          (forward, k = 0..K)
    r = p^(K);  r <- C A^T r + p^(k)         (backward, k = K-1..0)
Naively applying (A^T)^k to each p^(k) separately would cost O(K^2 m).

Update
------
Only column v of A changes, by delta = A'[:,v] - A[:,v].  Then
    r    = S delta                           (same Horner recursion, O(K m))
    t^0  = (C/2) (A + A')^T r
    p^k  = (A'^T)^k e_v,   t^k = (A'^T)^k t^0
    dS[:, q] = sum_k C^k ( t^k[q] p^k + p^k[q] t^k )
Each term is a SCALAR times a vector; no outer product is ever formed, and
S itself is never reconstructed.

The unscaled  delta = A'[:,v] - A[:,v]  form is used for insertion and
deletion alike; it stays well defined when the new in-degree is zero, where
the scaled D-CoSim form gives 0/0.
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx


class PerQueryCoSimRank:
    """
    Maintains S[:, q] for q in Q under edge and node updates.

    Args:
        G       : initial networkx DiGraph
        queries : iterable of query nodes (None = all nodes, i.e. |Q| = n)
        C       : decay factor
        K       : maximum series depth
        tol     : adaptive-K threshold on the REMAINING tail; 0 disables
                  adaptive stopping and forces exactly K terms
    """

    def __init__(self, G, queries=None, C=0.8, K=30, tol=1e-8,
                 blocked=False):
        self.C, self.K, self.tol = C, K, tol
        # blocked=True replaces the per-query Python loop of
        # _apply_column_change with two dense matrix products.  Same
        # arithmetic, same result; it exists so that the dynamic side
        # can be compared against a blocked static baseline on equal
        # terms.  Adaptive-K is unavailable in this mode, since the
        # stopping test is per column.
        self.blocked = blocked

        self.nodes = sorted(G.nodes())
        self.idx = {v: i for i, v in enumerate(self.nodes)}

        # in-neighbour lists are the source of truth; A is derived from them
        self.in_nbrs = [set() for _ in self.nodes]
        for (u, v) in G.edges():
            self.in_nbrs[self.idx[v]].add(self.idx[u])
        self._build_A()

        q_nodes = self.nodes if queries is None else sorted(queries)
        self.query_nodes = list(q_nodes)
        self.cols = {q: self._solve_column(self.idx[q]) for q in self.query_nodes}

        # Diagnostics for the last update: how many nodes the change could
        # reach, and how many query columns were provably untouched.
        self.last_touched = len(self.nodes)
        self.last_skipped = 0

    # ------------------------------------------------------------------
    # Sparse matrix construction
    # ------------------------------------------------------------------

    def _build_A(self):
        """
        Assemble the transition matrix from the in-neighbour lists, O(m).

        A^T is the primary store, in CSR.  Row v of A^T is column v of A, so
        the one thing an update needs to change lives in a single contiguous
        slice; see _set_column.  A itself is the transpose view, which shares
        storage and costs nothing to form.

        Full assembly is only used at construction and when n changes.
        Assembling from Python lists costs roughly ten single-column walks
        (measured in benchmarks/bench_constants.py), so doing it on every
        column change -- as an earlier version did -- dominated the update
        and made batched updates and node deletion lose to recomputation.
        """
        n = len(self.nodes)
        rows, cols, vals = [], [], []
        for j, preds in enumerate(self.in_nbrs):
            if preds:
                w = 1.0 / len(preds)
                for i in preds:
                    rows.append(j); cols.append(i); vals.append(w)
        # (j, i) ordering builds A^T directly: entry (j,i) of A^T is A[i,j].
        self.AT = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.AT.sort_indices()
        self.A = self.AT.T          # CSC view over the same buffers

    def _set_column(self, j, preds):
        """
        Replace column j of A (row j of A^T) in place, O(nnz) with NumPy
        buffer copies rather than a Python-level rebuild.

        When the in-degree is unchanged the slice is overwritten directly and
        nothing is reallocated, which is the common case for edge insertion
        into a node that already has in-neighbours.
        """
        AT = self.AT
        start, end = int(AT.indptr[j]), int(AT.indptr[j + 1])
        old_len = end - start

        preds = sorted(preds)
        new_len = len(preds)
        if new_len:
            w = 1.0 / new_len
            idx = np.asarray(preds, dtype=AT.indices.dtype)
            val = np.full(new_len, w, dtype=AT.data.dtype)
        else:
            idx = np.empty(0, dtype=AT.indices.dtype)
            val = np.empty(0, dtype=AT.data.dtype)

        if new_len == old_len:
            AT.indices[start:end] = idx
            AT.data[start:end] = val
        else:
            AT.indices = np.concatenate((AT.indices[:start], idx, AT.indices[end:]))
            AT.data = np.concatenate((AT.data[:start], val, AT.data[end:]))
            AT.indptr[j + 1:] += (new_len - old_len)

        self.A = AT.T               # refresh the transpose view

    def _column_of_A(self, j):
        """Dense column j of A, read from row j of the A^T store."""
        n = len(self.nodes)
        col = np.zeros(n)
        AT = self.AT
        start, end = int(AT.indptr[j]), int(AT.indptr[j + 1])
        if end > start:
            col[AT.indices[start:end]] = AT.data[start:end]
        return col

    # ------------------------------------------------------------------
    # Column solve  (initialisation)
    # ------------------------------------------------------------------

    def _solve_column(self, q_idx):
        """
        S e_q = sum_{k=0}^{K} C^k (A^T)^k (A^k e_q),  in O(K m).
        """
        n = len(self.nodes)
        e = np.zeros(n); e[q_idx] = 1.0
        return self._series_ATk_Ak(e)

    def _series_ATk_Ak(self, x):
        """
        Return sum_{k=0}^{K} C^k (A^T)^k A^k x  without forming S.

        Forward pass builds p^(k) = A^k x; the Horner recursion then folds
        in the (A^T)^k factors.  Both passes are O(K m).
        """
        fwd = [x]
        for _ in range(self.K):
            fwd.append(self.A @ fwd[-1])

        r = fwd[self.K].copy()
        for k in range(self.K - 1, -1, -1):
            r = self.C * (self.AT @ r) + fwd[k]
        return r

    # ------------------------------------------------------------------
    # Core update: one column of A changes
    # ------------------------------------------------------------------

    def _apply_column_change(self, v_idx, new_preds):
        """
        Replace the in-neighbour set of node v_idx and update every
        maintained column of S accordingly.

        Only scalar-times-vector combinations are used; no outer product
        and no n x n intermediate is ever allocated.
        """
        C, K, tol = self.C, self.K, self.tol
        n = len(self.nodes)

        old_col = self._column_of_A(v_idx)
        # The pre-update matrix is needed for r = S delta, and _set_column
        # mutates the buffers in place, so keep an explicit copy of A^T.
        AT_old = self.AT.copy()
        A_old = AT_old.T

        self.in_nbrs[v_idx] = set(new_preds)
        self._set_column(v_idx, self.in_nbrs[v_idx])   # A now holds A'
        new_col = self._column_of_A(v_idx)

        delta = new_col - old_col
        if not np.any(delta):
            return

        # r = S delta, using the PRE-update A (M is defined with the old S)
        fwd = [delta]
        for _ in range(K):
            fwd.append(A_old @ fwd[-1])
        r = fwd[K].copy()
        for k in range(K - 1, -1, -1):
            r = C * (AT_old @ r) + fwd[k]

        # t^(0) = (C/2) (A + A')^T r
        t0 = 0.5 * C * (AT_old @ r + self.AT @ r)

        # walk sequences under the UPDATED transition matrix
        p_seq = [np.zeros(n)]
        p_seq[0][v_idx] = 1.0
        t_seq = [t0]
        for _ in range(K):
            p_seq.append(self.AT @ p_seq[-1])
            t_seq.append(self.AT @ t_seq[-1])

        # sup-norms for the tail bound (A' is column sub-stochastic, so
        # ||A'^T x||_inf <= ||x||_inf and both sequences are non-increasing)
        p_inf = [float(np.max(np.abs(p))) for p in p_seq]
        t_inf = [float(np.max(np.abs(t))) for t in t_seq]

        # Reachability pruning.  dS[:,q] is a sum of terms scaled by t^(k)[q]
        # and p^(k)[q]; if both vanish at every k then dS[:,q] is EXACTLY
        # zero, so the column can be skipped outright.  This is not an
        # approximation -- it is the locality of Section "worked example"
        # turned into work avoided.  Building the mask costs O(Kn), i.e. the
        # price of one query column, so it pays whenever |Q| > 1.
        touched = np.zeros(n, dtype=bool)
        for k in range(K + 1):
            np.logical_or(touched, p_seq[k] != 0.0, out=touched)
            np.logical_or(touched, t_seq[k] != 0.0, out=touched)
        self.last_touched = int(touched.sum())
        self.last_skipped = 0

        if self.blocked:
            # dS[:,q] = sum_k C^k ( t^k[q] p^k + p^k[q] t^k ), so stacking the
            # walk sequences as n x (K+1) matrices P and T turns the whole
            # accumulation into two dense products of shape
            # (n x K+1)(K+1 x |Q|) -- identical flops, one BLAS-3 call each.
            live = [q for q in self.query_nodes if touched[self.idx[q]]]
            self.last_skipped = len(self.query_nodes) - len(live)
            if live:
                Pm = np.column_stack(p_seq)              # n x (K+1)
                Tm = np.column_stack(t_seq)
                ck = C ** np.arange(K + 1)
                qi = np.fromiter((self.idx[q] for q in live), int, len(live))
                Pq = Pm[qi, :] * ck                      # |Q| x (K+1)
                Tq = Tm[qi, :] * ck
                delta = Pm @ Tq.T + Tm @ Pq.T            # n x |Q|
                for j, q in enumerate(live):
                    self.cols[q] += delta[:, j]
            return

        for q in self.query_nodes:
            q_idx = self.idx[q]
            if not touched[q_idx]:
                self.last_skipped += 1
                continue
            col = self.cols[q]
            for k in range(K + 1):
                ck = C ** k
                tq, pq = t_seq[k][q_idx], p_seq[k][q_idx]
                if tq:
                    col += (ck * tq) * p_seq[k]
                if pq:
                    col += (ck * pq) * t_seq[k]
                # stop only when the whole remaining tail is negligible:
                #   ||tail_{>k}||_inf <= 2 C^{k+1}/(1-C) ||t^k||_inf ||p^k||_inf
                if tol > 0 and k > 0:
                    if (2.0 * ck * C / (1.0 - C)) * t_inf[k] * p_inf[k] < tol:
                        break

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def edge_insert(self, edges):
        """Insert edges. Each destination column is updated in turn."""
        by_dst = {}
        for (u, v) in edges:
            if u not in self.idx or v not in self.idx:
                raise ValueError(f"edge ({u},{v}) references an unknown node")
            if v in by_dst and u in by_dst[v]:
                raise ValueError(f"edge ({u},{v}) appears more than once")
            by_dst.setdefault(v, set()).add(u)

        for v, srcs in by_dst.items():
            v_idx = self.idx[v]
            preds = set(self.in_nbrs[v_idx])
            for u in srcs:
                if self.idx[u] in preds:
                    raise ValueError(f"edge ({u},{v}) already exists")
                preds.add(self.idx[u])
            self._apply_column_change(v_idx, preds)

    def edge_delete(self, edges):
        """Delete edges. Each destination column is updated in turn."""
        by_dst = {}
        for (u, v) in edges:
            if u not in self.idx or v not in self.idx:
                raise ValueError(f"edge ({u},{v}) references an unknown node")
            by_dst.setdefault(v, set()).add(u)

        for v, srcs in by_dst.items():
            v_idx = self.idx[v]
            preds = set(self.in_nbrs[v_idx])
            for u in srcs:
                if self.idx[u] not in preds:
                    raise ValueError(f"edge ({u},{v}) does not exist")
                preds.discard(self.idx[u])
            self._apply_column_change(v_idx, preds)

    def node_insert(self, new_node, new_edges=()):
        """
        Insert an isolated node, then its edges.

        An isolated node w satisfies S(w, j) = 0 for j != w and S(w, w) = 1,
        so every maintained column gains a single zero entry, and w's own
        column (if queried) starts as e_w.
        """
        if new_node in self.idx:
            raise ValueError(f"node {new_node} already exists")

        pos = np.searchsorted(self.nodes, new_node)
        self.nodes.insert(pos, new_node)
        self.idx = {v: i for i, v in enumerate(self.nodes)}

        # shift in-neighbour indices at or above pos
        shifted = []
        for j, preds in enumerate(self.in_nbrs):
            shifted.append({i + 1 if i >= pos else i for i in preds})
        shifted.insert(pos, set())
        self.in_nbrs = shifted
        self._build_A()

        for q in self.query_nodes:
            self.cols[q] = np.insert(self.cols[q], pos, 0.0)

        for (u, v) in new_edges:
            if u != new_node and v != new_node:
                raise ValueError(f"edge ({u},{v}) does not involve {new_node}")
        if new_edges:
            self.edge_insert(list(new_edges))

    def node_delete(self, del_node):
        """
        Delete a node: strip its in-edges, then its out-edges, then drop the
        row.  After both steps the node is isolated, so removing its entry
        leaves the remaining entries exact.
        """
        if del_node not in self.idx:
            raise ValueError(f"node {del_node} not in graph")
        d_idx = self.idx[del_node]

        # in-edges: one column change
        if self.in_nbrs[d_idx]:
            self._apply_column_change(d_idx, set())

        # out-edges: one column change per successor
        succs = [j for j, preds in enumerate(self.in_nbrs) if d_idx in preds]
        for j in succs:
            preds = set(self.in_nbrs[j]); preds.discard(d_idx)
            self._apply_column_change(j, preds)

        # drop the row/column
        self.nodes.pop(d_idx)
        self.in_nbrs.pop(d_idx)
        self.in_nbrs = [{i - 1 if i > d_idx else i for i in preds}
                        for preds in self.in_nbrs]
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self._build_A()

        if del_node in self.cols:
            del self.cols[del_node]
            self.query_nodes.remove(del_node)
        for q in self.query_nodes:
            self.cols[q] = np.delete(self.cols[q], d_idx)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def column(self, q):
        """Current S[:, q] as a dense length-n vector."""
        return self.cols[q]

    def similarity(self, u, q):
        """S[u, q]; q must be a maintained query node."""
        return self.cols[q][self.idx[u]]

    @property
    def n(self):
        return len(self.nodes)

    @property
    def m(self):
        return sum(len(p) for p in self.in_nbrs)

    def to_graph(self):
        """Reconstruct the current DiGraph (for reference checks only)."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for j, preds in enumerate(self.in_nbrs):
            for i in preds:
                G.add_edge(self.nodes[i], self.nodes[j])
        return G

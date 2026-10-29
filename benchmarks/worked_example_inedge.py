# -*- coding: utf-8 -*-
"""
worked_example_inedge.py
========================
A ten-node example on which the dynamic update provably beats recomputation,
and on which the break-even predicted by the cost model can be checked on the
same page as the model itself.

The point turns on a detail of Section "Explicit expressions": edges that
share a destination are ONE column update, edges with distinct destinations
are one each.  So the direction in which a new node is attached decides the
cost, even when the number of edges is identical:

    node 11 with OUT-edges  11->1, 11->2   ->  columns 1 and 2   ->  b = 2
    node 11 with IN-edges    1->11, 2->11  ->  column 11 only    ->  b = 1

Same two edges, same two neighbours, same node-insertion operation, twice the
cost one way round.

Graph
-----
Two independent five-node communities, each a complete DAG (i -> j whenever
i < j), with no edge between them.  n = 10, m = 20, mean degree 2.  Being
acyclic, the CoSimRank series terminates, so every value below is an exact
rational and the matrices can be checked by hand.

Usage:
    python benchmarks/worked_example_inedge.py
"""

import sys, os, io, csv, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fractions import Fraction as F
import numpy as np
import networkx as nx

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C_EXACT = F(4, 5)
C_FLOAT = 0.8
K_CONV = 130
TOL_CONV = 1e-12
K_MODEL = 20            # depth used in the operation counts

COMM_A = [0, 1, 2, 3, 4]
COMM_B = [5, 6, 7, 8, 9]
NEW = 10                # the node inserted, printed as 11


def base_edges():
    """Two complete DAGs on five nodes each, mutually disconnected."""
    e = []
    for comm in (COMM_A, COMM_B):
        for a in range(len(comm)):
            for b in range(a + 1, len(comm)):
                e.append((comm[a], comm[b]))
    return e


# ---------------------------------------------------------------
# Exact rational linear algebra
# ---------------------------------------------------------------

def zeros(n):
    return [[F(0)] * n for _ in range(n)]


def eye(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def matmul(X, Y):
    n = len(X)
    out = zeros(n)
    for i in range(n):
        for t in range(n):
            a = X[i][t]
            if a:
                for j in range(n):
                    if Y[t][j]:
                        out[i][j] += a * Y[t][j]
    return out


def transpose(X):
    return [list(c) for c in zip(*X)]


def build_A(n, edges):
    A = zeros(n)
    for j in range(n):
        preds = [i for (i, jj) in edges if jj == j]
        if preds:
            w = F(1, len(preds))
            for i in preds:
                A[i][j] = w
    return A


def cosim_exact(A, C, max_terms=60):
    n = len(A)
    S, Ak, ATk, Ck = zeros(n), eye(n), eye(n), F(1)
    AT = transpose(A)
    for _ in range(max_terms):
        if all(not v for row in Ak for v in row):
            return S
        T = matmul(ATk, Ak)
        for i in range(n):
            for j in range(n):
                if T[i][j]:
                    S[i][j] += Ck * T[i][j]
        Ak, ATk, Ck = matmul(Ak, A), matmul(ATk, AT), Ck * C
    raise ValueError("A is not nilpotent")


def sub(X, Y):
    return [[X[i][j] - Y[i][j] for j in range(len(X))] for i in range(len(X))]


def nonzero(X):
    return sum(1 for row in X for v in row if v)


def show(M, label, rows=None, width=12):
    idx = range(len(M)) if rows is None else rows
    print("\n  " + label)
    print("       " + "".join("{:>{w}}".format(j + 1, w=width) for j in idx))
    for i in idx:
        print("  {:>4} ".format(i + 1)
              + "".join("{:>{w}}".format(str(M[i][j]) if M[i][j] else ".", w=width)
                        for j in idx))


# ---------------------------------------------------------------
# Float cross-check against the shipped implementation
# ---------------------------------------------------------------

def converged(G):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C_FLOAT, K=K_CONV, tol=TOL_CONV)
    return S, nodes


def reorder(S, src, dst):
    idx = {v: i for i, v in enumerate(src)}
    perm = [idx[v] for v in dst]
    return S[np.ix_(perm, perm)]


def dyn_vs_static(G0, op, *args):
    S0, _ = converged(G0)
    d = DynamicCoSimRank(G0.copy(), C=C_FLOAT, K=K_CONV, tol=TOL_CONV)
    d.S = S0.copy()
    getattr(d, op)(*args)
    S1, n1 = converged(d.G)
    ref = reorder(S1, n1, d.nodes)
    return np.linalg.norm(d.S - ref, 'fro') / np.linalg.norm(ref, 'fro')


# ---------------------------------------------------------------
# Pieces: edges sharing a destination are one column update
# ---------------------------------------------------------------

def pieces(edges):
    dests = {}
    for (u, v) in edges:
        dests.setdefault(v, []).append(u)
    return dests


def counts(n, m, K, Q, b):
    """Matvecs and flops; a sparse matvec is 2m, a scaled vector add is 2n."""
    return {
        'dyn_mv': 4 * K * b,
        'stat_mv': 2 * K * Q,
        'dyn_fl': b * (8 * K * m + 4 * K * n * Q),
        'stat_fl': 4 * K * m * Q,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    E = base_edges()
    n0, m0 = 10, len(E)
    print("=" * 78)
    print("  Two independent complete-DAG communities")
    print("  n = {}, m = {}, mean degree = {}".format(n0, m0, F(m0, n0)))
    print("  C = {} exactly".format(C_EXACT))
    print("=" * 78)

    S0 = cosim_exact(build_A(n0, E), C_EXACT)
    show(S0, "S before, community A block (nodes 1-5)", rows=COMM_A)
    blockB_equal = all(S0[COMM_A[i]][COMM_A[j]] == S0[COMM_B[i]][COMM_B[j]]
                       for i in range(5) for j in range(5))
    print("\n  community B block identical to A: {}".format(blockB_equal))
    print("  cross blocks all zero          : {}".format(
        all(not S0[i][j] for i in COMM_A for j in COMM_B)))

    # ---------------- the two attachments ----------------
    IN_EDGES = [(0, NEW), (1, NEW)]      # 1 -> 11, 2 -> 11
    OUT_EDGES = [(NEW, 0), (NEW, 1)]     # 11 -> 1, 11 -> 2

    print("\n" + "=" * 78)
    print("  Same two edges, two orientations")
    print("=" * 78)
    for label, edges in [("IN  (1->11, 2->11)", IN_EDGES),
                         ("OUT (11->1, 11->2)", OUT_EDGES)]:
        pc = pieces(edges)
        print("  {}  destinations {} -> b = {} column update(s)".format(
            label, sorted(d + 1 for d in pc), len(pc)))

    results = []
    for label, edges in [("in-edges", IN_EDGES), ("out-edges", OUT_EDGES)]:
        n1 = n0 + 1
        E1 = E + edges
        S1 = cosim_exact(build_A(n1, E1), C_EXACT)

        S0p = zeros(n1)
        for i in range(n0):
            for j in range(n0):
                S0p[i][j] = S0[i][j]
        S0p[NEW][NEW] = F(1)
        dS = sub(S1, S0p)

        b = len(pieces(edges))
        cells = nonzero(dS)
        b_block = any(dS[i][j] for i in COMM_B for j in COMM_B)

        G0 = nx.DiGraph(); G0.add_nodes_from(range(n0)); G0.add_edges_from(E)
        err = dyn_vs_static(G0, 'node_insert', NEW, edges)

        print("\n" + "-" * 78)
        print("  Insert node 11 via {}  (b = {})".format(label, b))
        print("-" * 78)
        show(dS, "dS", rows=[0, 1, 2, 3, 4, NEW])
        print("\n    cells changed        : {}/{}  ({:.1f}%)".format(
            cells, n1 * n1, 100.0 * cells / (n1 * n1)))
        print("    community B touched  : {}".format(b_block))
        print("    dynamic vs static    : {:.2e}  [{}]".format(
            err, "PASS" if err < 1e-8 else "FAIL"))
        results.append((label, b, cells, b_block, err, S1, dS))

    # ---------------- round trip on the in-edge version ----------------
    print("\n" + "=" * 78)
    print("  Round trip: delete node 11 again (in-edge version)")
    print("=" * 78)
    S_in = results[0][5]
    S_back = [[S_in[i][j] for j in range(n0)] for i in range(n0)]
    same = all(S_back[i][j] == S0[i][j] for i in range(n0) for j in range(n0))
    # Node 11 has in-edges at this point, so its row is NOT e_11; the
    # isolated-node condition applies only once column 11 has been cleared,
    # which is the first step of the deletion.
    E_cleared = [e for e in E + IN_EDGES if e[1] != NEW]
    S_cleared = cosim_exact(build_A(n0 + 1, E_cleared), C_EXACT)
    print("  after clearing column 11, row 11 == e_11 : {}".format(
        all(S_cleared[NEW][j] == (F(1) if j == NEW else F(0))
            for j in range(n0 + 1))))
    print("  contracted 10x10 == original             : {}".format(same))

    G_ins = nx.DiGraph(); G_ins.add_nodes_from(range(n0 + 1))
    G_ins.add_edges_from(E + IN_EDGES)
    err_del = dyn_vs_static(G_ins, 'node_delete', NEW)
    print("  dynamic node_delete vs static : {:.2e}  [{}]".format(
        err_del, "PASS" if err_del < 1e-8 else "FAIL"))

    # ---------------- operation counts ----------------
    print("\n" + "=" * 78)
    print("  Operation counts, K = {}, m = {}, n = {}".format(K_MODEL, m0, n0 + 1))
    print("=" * 78)
    print("\n  Matrix-vector products (the cleanest comparison):")
    print("  {:>5}  {:>14}  {:>14}  {:>14}".format(
        "|Q|", "dynamic b=1", "static 2K|Q|", "dyn/static"))
    print("  " + "-" * 52)
    rows_csv = []
    for Q in [1, 3, 5, 11]:
        c = counts(n0 + 1, m0, K_MODEL, Q, 1)
        r = c['stat_mv'] / c['dyn_mv']
        note = "{:.1f}x fewer".format(r) if r >= 1 else "{:.1f}x MORE".format(1 / r)
        print("  {:>5}  {:>14}  {:>14}  {:>14}".format(
            Q, c['dyn_mv'], c['stat_mv'], note))

    print("\n  Full flop ratio  2/|Q| + 1/dbar   (dbar = {}):".format(F(m0, n0)))
    print("  {:>5}  {:>10}  {:>10}  {:>10}".format(
        "|Q|", "ratio", "speedup", "verdict"))
    print("  " + "-" * 42)
    dbar = m0 / n0
    for Q in [1, 3, 4, 5, 11]:
        ratio = 2.0 / Q + 1.0 / dbar
        print("  {:>5}  {:>10.3f}  {:>9.2f}x  {:>10}".format(
            Q, ratio, 1 / ratio, "update" if ratio < 1 else "recompute"))
        rows_csv.append([Q, round(ratio, 4), round(1 / ratio, 4),
                         "update" if ratio < 1 else "recompute"])
    qstar = 2 * dbar / (dbar - 1)
    print("\n  |Q|* = 2 dbar / (dbar - 1) = {:.1f}".format(qstar))
    print("  so the update wins from |Q| = {}".format(int(qstar) + 1))

    print("\n  For the OUT-edge attachment (b = 2) every ratio doubles:")
    print("  {:>5}  {:>10}  {:>10}".format("|Q|", "ratio", "verdict"))
    print("  " + "-" * 30)
    for Q in [5, 11]:
        ratio = 2 * (2.0 / Q + 1.0 / dbar)
        print("  {:>5}  {:>10.3f}  {:>10}".format(
            Q, ratio, "update" if ratio < 1 else "recompute"))

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'worked_example_inedge.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["attachment", "b", "cells_changed", "total_cells",
                    "B_block_touched", "dyn_vs_static"])
        for (label, b, cells, bb, err, _, _) in results:
            w.writerow([label, b, cells, (n0 + 1) ** 2, bb, "{:.3e}".format(err)])
        w.writerow([])
        w.writerow(["Q", "flop_ratio", "speedup", "verdict"])
        w.writerows(rows_csv)
    print("\n  -> {}".format(os.path.relpath(p)))

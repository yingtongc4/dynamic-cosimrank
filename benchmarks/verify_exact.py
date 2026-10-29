# -*- coding: utf-8 -*-
"""
verify_exact.py  --  Zero-tolerance correctness verification
=============================================================
Verifies S_dyn == S_stat as an EXACT identity, with no tolerance at all.

Why this is possible
--------------------
Every quantity in CoSimRank is rational:
  * A[i,j] = 1/in_deg(j)                    -> Fraction
  * C = 3/5 or 4/5                          -> Fraction
  * every operation is +, -, *, /           -> stays rational
So the whole computation can be carried out in exact rational arithmetic
(fractions.Fraction).  No rounding ever occurs, and `==` is meaningful.

This is what the mathematical claim actually says:
      S_dyn == S_stat        (exact matrix equality)
Round 1 (verify_correctness.py) tests the same claim in float64, where
`==` is unusable because the two computation paths round differently.
Both are needed: this file proves the algebra, Round 1 proves the shipped
float implementation.

Two tests
---------
Test A  Exact Lyapunov identity, on ANY graph (cycles included).
        Solves  dS = C A'^T dS A' + C M  exactly by Gaussian elimination
        over Q, and checks  S_before + dS == S_after  with `==`.
        This verifies the DERIVATION in the dissertation.

Test B  Exact end-to-end run of the p/t series algorithm, on DAGs.
        On a DAG the walk vectors vanish after finitely many steps, so the
        series terminates and the algorithm produces an exact rational
        answer.  Checked with `==`.
        This verifies the ALGORITHM STRUCTURE, not just the equation.

Usage:
    python benchmarks/verify_exact.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fractions import Fraction as F
import networkx as nx
import csv

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


# ============================================================
# Exact rational matrix helpers (plain lists of lists)
# ============================================================

def zeros(n, m=None):
    m = n if m is None else m
    return [[F(0) for _ in range(m)] for _ in range(n)]


def eye(n):
    return [[F(1) if i == j else F(0) for j in range(n)] for i in range(n)]


def matmul(X, Y):
    n, k, m = len(X), len(Y), len(Y[0])
    out = zeros(n, m)
    for i in range(n):
        Xi = X[i]
        for t in range(k):
            a = Xi[t]
            if a:
                Yt = Y[t]
                oi = out[i]
                for j in range(m):
                    if Yt[j]:
                        oi[j] += a * Yt[j]
    return out


def transpose(X):
    return [list(col) for col in zip(*X)]


def matvec(X, v):
    return [sum((X[i][j] * v[j] for j in range(len(v))), F(0))
            for i in range(len(X))]


def add(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def sub(X, Y):
    return [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def scale(c, X):
    return [[c * X[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def is_zero(X):
    return all(not X[i][j] for i in range(len(X)) for j in range(len(X[0])))


def mat_eq(X, Y):
    """Exact matrix equality -- this is the whole point of the file."""
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        return False
    return all(X[i][j] == Y[i][j]
               for i in range(len(X)) for j in range(len(X[0])))


def max_abs_diff(X, Y):
    """Largest |X-Y| entry, as an exact Fraction (0 iff X == Y)."""
    return max((abs(X[i][j] - Y[i][j])
                for i in range(len(X)) for j in range(len(X[0]))), default=F(0))


# ============================================================
# Exact linear solve (Gaussian elimination over Q)
# ============================================================

def gauss_solve(M, b):
    """Solve M x = b exactly.  M is N x N of Fractions, b length N."""
    N = len(M)
    aug = [row[:] + [b[i]] for i, row in enumerate(M)]

    for col in range(N):
        piv = None
        for r in range(col, N):
            if aug[r][col]:
                piv = r
                break
        if piv is None:
            raise ValueError(f"Singular system at column {col}")
        aug[col], aug[piv] = aug[piv], aug[col]

        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]

        for r in range(N):
            if r != col and aug[r][col]:
                f = aug[r][col]
                aug[r] = [aug[r][j] - f * aug[col][j] for j in range(N + 1)]

    return [aug[i][N] for i in range(N)]


# ============================================================
# Exact CoSimRank
# ============================================================

def build_A_exact(G, nodes):
    """Column-normalised A with exact rational entries."""
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = zeros(n)
    for v in G.nodes():
        preds = list(G.predecessors(v))
        if preds:
            w = F(1, len(preds))
            j = idx[v]
            for u in preds:
                A[idx[u]][j] = w
    return A


def cosimrank_exact_solve(A, C):
    """
    Solve  S = C A^T S A + I  exactly.

    Vectorised (column-major):  vec(A^T S A) = (A^T (x) A^T) vec(S),
    so  (I - C (A^T (x) A^T)) vec(S) = vec(I).
    Uses Gaussian elimination over Q -- no iteration, no truncation.
    """
    n = len(A)
    AT = transpose(A)
    N = n * n

    # Build (I - C * kron(AT, AT)) acting on vec(S) with vec column-major:
    #   vec(X)[i + n*j] = X[i][j]
    # (A^T X A)[i][j] = sum_{p,q} AT[i][p] X[p][q] A[q][j]
    M = [[F(0)] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            row = i + n * j
            M[row][row] += F(1)
            for p in range(n):
                aip = AT[i][p]
                if not aip:
                    continue
                for q in range(n):
                    aqj = A[q][j]
                    if not aqj:
                        continue
                    M[row][p + n * q] -= C * aip * aqj

    b = [F(0)] * N
    for i in range(n):
        b[i + n * i] = F(1)

    x = gauss_solve(M, b)
    return [[x[i + n * j] for j in range(n)] for i in range(n)]


def cosimrank_exact_dag(A, C, max_terms=200):
    """
    Sum  S = sum_k C^k (A^T)^k A^k  exactly.
    Only valid when A is nilpotent (DAG): the series terminates.
    Raises if A^k has not vanished within max_terms.
    """
    n = len(A)
    S = zeros(n)
    Ak = eye(n)
    ATk = eye(n)
    Ck = F(1)
    AT = transpose(A)
    for k in range(max_terms):
        if is_zero(Ak):
            return S, k
        S = add(S, scale(Ck, matmul(ATk, Ak)))
        Ak = matmul(Ak, A)
        ATk = matmul(ATk, AT)
        Ck *= C
    raise ValueError("A is not nilpotent -- use cosimrank_exact_solve instead")


# ============================================================
# Test A: exact Lyapunov identity  (works on any graph)
# ============================================================

def _new_column(G_after, nodes, v):
    """Exact column v of A for the updated graph."""
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    col = [F(0)] * n
    preds = list(G_after.predecessors(v))
    if preds:
        w = F(1, len(preds))
        for u in preds:
            col[idx[u]] = w
    return col


def lyapunov_identity(G_before, G_after, v, C):
    """
    Verify, exactly:
        S_after - S_before  ==  dS,
    where dS is the exact solution of the Lyapunov equation
        dS = C A'^T dS A' + C M,      M = x e_v^T + e_v x^T,
        x  = A^T S delta + (1/2)(delta^T S delta) e_v,
        delta = A'[:,v] - A[:,v].

    Only column v of A changes; the node set must be identical.
    Returns (ok, max_abs_diff).
    """
    nodes = sorted(G_before.nodes())
    assert nodes == sorted(G_after.nodes())
    n = len(nodes)
    vi = nodes.index(v)

    A  = build_A_exact(G_before, nodes)
    Ap = build_A_exact(G_after,  nodes)

    S_before = cosimrank_exact_solve(A,  C)
    S_after  = cosimrank_exact_solve(Ap, C)

    # delta = A'[:,v] - A[:,v]
    delta = [Ap[i][vi] - A[i][vi] for i in range(n)]

    # x = A^T S delta + (1/2)(delta^T S delta) e_v
    Sd = matvec(S_before, delta)
    ATSd = matvec(transpose(A), Sd)
    dSd = sum((delta[i] * Sd[i] for i in range(n)), F(0))
    x = list(ATSd)
    x[vi] += F(1, 2) * dSd

    # M = x e_v^T + e_v x^T
    M = zeros(n)
    for i in range(n):
        M[i][vi] += x[i]
        M[vi][i] += x[i]

    # Solve dS = C A'^T dS A' + C M  exactly:
    #   (I - C (A'^T (x) A'^T)) vec(dS) = C vec(M)
    ApT = transpose(Ap)
    N = n * n
    LHS = [[F(0)] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            row = i + n * j
            LHS[row][row] += F(1)
            for p in range(n):
                aip = ApT[i][p]
                if not aip:
                    continue
                for q in range(n):
                    aqj = Ap[q][j]
                    if not aqj:
                        continue
                    LHS[row][p + n * q] -= C * aip * aqj

    rhs = [C * M[i][j] for j in range(n) for i in range(n)]
    # careful: index order must match row = i + n*j
    rhs = [F(0)] * N
    for i in range(n):
        for j in range(n):
            rhs[i + n * j] = C * M[i][j]

    sol = gauss_solve(LHS, rhs)
    dS = [[sol[i + n * j] for j in range(n)] for i in range(n)]

    lhs_mat = sub(S_after, S_before)
    return mat_eq(lhs_mat, dS), max_abs_diff(lhs_mat, dS)


# ============================================================
# Test B: exact end-to-end series algorithm (DAGs only)
# ============================================================

def d_cosim_exact_dag(G_before, G_after, v, C, max_terms=200):
    """
    Run the p/t series algorithm in exact rational arithmetic.
    On a DAG all walk vectors vanish after finitely many steps, so the
    series terminates and the result is EXACT -- no truncation, no tolerance.

    Returns S_before + dS as an exact rational matrix.
    """
    nodes = sorted(G_before.nodes())
    n = len(nodes)
    vi = nodes.index(v)

    A  = build_A_exact(G_before, nodes)
    Ap = build_A_exact(G_after,  nodes)
    AT, ApT = transpose(A), transpose(Ap)

    S_before, _ = cosimrank_exact_dag(A, C, max_terms)

    delta = [Ap[i][vi] - A[i][vi] for i in range(n)]

    # r = S delta  (exact; S_before was itself summed from the terminating series)
    r = matvec(S_before, delta)

    # t^(0) = (C/2) (A + A')^T r
    ApA = [[A[i][j] + Ap[i][j] for j in range(n)] for i in range(n)]
    t = [C * F(1, 2) * e for e in matvec(transpose(ApA), r)]

    # p^(0) = e_v
    p = [F(0)] * n
    p[vi] = F(1)

    dS = zeros(n)
    Ck = F(1)
    for k in range(max_terms):
        if all(not e for e in p) or all(not e for e in t):
            break
        tq, pq = t, p
        for q in range(n):
            if not (tq[q] or pq[q]):
                continue
            for i in range(n):
                val = Ck * (tq[q] * pq[i] + pq[q] * tq[i])
                if val:
                    dS[i][q] += val
        p = matvec(ApT, p)
        t = matvec(ApT, t)
        Ck *= C
    else:
        raise ValueError("series did not terminate -- graph is not a DAG")

    return add(S_before, dS)


# ============================================================
# Fixtures
# ============================================================

def fixtures():
    """(label, G_before, G_after, changed_column_v, is_dag)"""
    out = []

    # --- DAG chain: insert an edge ---
    G = nx.DiGraph(); G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    H = G.copy(); H.add_edge(0, 2)
    out.append(("DAG chain, insert 0->2", G, H, 2, True))

    # --- DAG chain: delete an edge ---
    G = nx.DiGraph(); G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    H = G.copy(); H.remove_edge(1, 2)
    out.append(("DAG chain, delete 1->2 (in-deg -> 0)", G, H, 2, True))

    # --- mixed-sign delta on a DAG ---
    G = nx.DiGraph(); G.add_edges_from([(0, 2), (1, 2), (2, 3)])
    H = G.copy(); H.add_edge(3, 2)     # cycle appears -> not a DAG afterwards
    out.append(("mixed-sign, insert 3->2", G, H, 2, False))

    # --- 2-cycle (definitely not nilpotent) ---
    G = nx.DiGraph(); G.add_edges_from([(0, 1)])
    H = G.copy(); H.add_edge(1, 0)
    out.append(("2-cycle, insert 1->0", G, H, 0, False))

    # --- cyclic graph, delete ---
    G = nx.DiGraph(); G.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1)])
    H = G.copy(); H.remove_edge(1, 0)
    out.append(("cyclic, delete 1->0", G, H, 0, False))

    # --- node insertion, modelled as edges into an isolated node ---
    G = nx.DiGraph(); G.add_edges_from([(0, 1), (1, 2)]); G.add_node(3)
    H = G.copy(); H.add_edge(3, 1)
    out.append(("node-insert step, add 3->1", G, H, 1, True))

    # --- larger DAG ---
    G = nx.DiGraph()
    G.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)])
    H = G.copy(); H.add_edge(0, 5)
    out.append(("wider DAG, insert 0->5", G, H, 5, True))

    return out


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    C = F(4, 5)   # 0.8 exactly

    print("=" * 72)
    print("  verify_exact.py  --  exact rational arithmetic, ZERO tolerance")
    print(f"  C = {C} (exactly),  comparison operator = `==`")
    print("=" * 72)

    rows = []
    n_fail = 0

    print("\n" + "-" * 72)
    print("  Test A: exact Lyapunov identity   S_after - S_before == dS")
    print("          (solved over Q; valid on cyclic graphs too)")
    print("-" * 72)
    for (label, G, H, v, is_dag) in fixtures():
        ok, diff = lyapunov_identity(G, H, v, C)
        flag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  {label:<38} {flag}   max|diff| = {diff}")
        rows.append(["lyapunov_exact", label, str(diff), flag])

    print("\n" + "-" * 72)
    print("  Test B: exact end-to-end series algorithm  S_dyn == S_stat")
    print("          (DAG only -- the series terminates, so no truncation)")
    print("-" * 72)
    for (label, G, H, v, is_dag) in fixtures():
        if not is_dag:
            print(f"  {label:<38} SKIP   (not a DAG: series does not terminate)")
            rows.append(["series_exact", label, "-", "SKIP"])
            continue
        nodes = sorted(H.nodes())
        S_dyn = d_cosim_exact_dag(G, H, v, C)
        S_stat, _ = cosimrank_exact_dag(build_A_exact(H, nodes), C)
        ok = mat_eq(S_dyn, S_stat)
        diff = max_abs_diff(S_dyn, S_stat)
        flag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  {label:<38} {flag}   max|diff| = {diff}")
        rows.append(["series_exact", label, str(diff), flag])

    # Show one exact matrix so the rational nature is visible
    print("\n" + "-" * 72)
    print("  Sample exact result (DAG chain 0->1->2->3, C = 4/5)")
    print("-" * 72)
    G = nx.DiGraph(); G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    S, terms = cosimrank_exact_dag(build_A_exact(G, sorted(G.nodes())), C)
    for row in S:
        print("   [" + "  ".join(f"{str(e):>12}" for e in row) + "]")
    print(f"  (series terminated after {terms} terms; every entry is exact)")

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, 'verify_exact.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["test", "fixture", "max_abs_diff_exact", "result"])
        wr.writerows(rows)
    print(f"\n  -> Saved: {os.path.relpath(path)}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Overall: {'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL(S)'}"
          f"  ({elapsed:.1f}s)")
    print("=" * 72)
    # a verification gate has to report through its exit status, or a build
    # system will read a printed FAIL as a successful step
    sys.exit(1 if n_fail else 0)

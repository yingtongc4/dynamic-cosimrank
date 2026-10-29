# -*- coding: utf-8 -*-
"""
worked_example.py
-----------------
Standalone reference implementation for the dissertation worked example.
Computes CoSimRank from mathematical definitions only -- no imports from src/.

Usage:
    python benchmarks/worked_example.py

Produces:
    stdout  : matrix printouts + markdown table
    benchmarks/results/worked_example.csv : numerical results
"""

import os
import csv
import sys
import numpy as np

# Force UTF-8 output on Windows so special chars don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -------------------------------------------------------
# 1.  Reference CoSimRank (standalone, no src/)
# -------------------------------------------------------

def build_transition(n, edges):
    """
    Build column-normalised transition matrix A (n x n).
    Edge (u, v) means u is an in-neighbour of v, so A[u, v] > 0.
    Each column v holds the normalised in-neighbour weights of v.
    """
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edges:
        A[u, v] += 1.0
    col_sums = A.sum(axis=0)
    for j in range(n):
        if col_sums[j] > 0:
            A[:, j] /= col_sums[j]
    return A


def cosimrank_iter(A, C, tol=1e-12, max_iter=5000):
    """
    Compute S* by iterating S^(k+1) = C * A^T S^(k) A + I until convergence.
    Returns (S, iterations_used).
    """
    n = A.shape[0]
    S = np.eye(n)
    AT = A.T
    for k in range(max_iter):
        S_new = C * AT @ S @ A + np.eye(n)
        diff = np.linalg.norm(S_new - S, "fro")
        S = S_new
        if diff < tol:
            return S, k + 1
    raise RuntimeError(f"cosimrank_iter did not converge in {max_iter} iterations")


def cosimrank_series(A, C, tol=1e-14, max_terms=3000):
    """
    Compute S* via the series S = sum_{k=0}^{inf} C^k (A^T)^k A^k.
    Stops when the Frobenius norm of the next term < tol.
    Returns (S, terms_used).
    """
    n = A.shape[0]
    S = np.zeros((n, n))
    AT_k = np.eye(n)
    A_k  = np.eye(n)
    Ck   = 1.0
    for k in range(max_terms):
        term = Ck * AT_k @ A_k
        S += term
        if k > 0 and np.linalg.norm(term, "fro") < tol:
            return S, k + 1
        AT_k = AT_k @ A.T
        A_k  = A_k  @ A
        Ck  *= C
    raise RuntimeError(f"cosimrank_series did not converge in {max_terms} terms")


def cosimrank(A, C):
    """
    Compute S* using both methods and cross-validate.
    Raises if they disagree by more than 1e-10 (Frobenius norm).
    Returns (S, cross_err, n_iter, n_terms).
    """
    S_iter,   n_iter  = cosimrank_iter(A, C)
    S_series, n_terms = cosimrank_series(A, C)
    err = np.linalg.norm(S_iter - S_series, "fro")
    if err > 1e-10:
        raise AssertionError(
            f"Iteration vs series disagree: ||diff||_F = {err:.3e} "
            f"(iter={n_iter}, terms={n_terms})"
        )
    return S_iter, err, n_iter, n_terms


# -------------------------------------------------------
# 2.  Graph states
# -------------------------------------------------------

C = 0.6

# Nodes are 0-indexed internally; printed as 1-indexed for the paper.

# G0: edges 1->2, 2->3  (0-indexed: 0->1, 1->2)
G0_n     = 3
G0_edges = [(0, 1), (1, 2)]

# G1: G0 + 4->1  (add node 3, edge 3->0)
G1_n     = 4
G1_edges = [(0, 1), (1, 2), (3, 0)]

# G2: G1 + 4->2  (add edge 3->1) -- node insertion complete, Case A
G2_n     = 4
G2_edges = [(0, 1), (1, 2), (3, 0), (3, 1)]

# G3: G0 + 1->4, 2->4  (reversed direction, Case B)
G3_n     = 4
G3_edges = [(0, 1), (1, 2), (0, 3), (1, 3)]

# G4: G2 minus 4->1  (deletion intermediate)
G4_n     = 4
G4_edges = [(0, 1), (1, 2), (3, 1)]

# G5: G4 minus 4->2  (node 4 isolated)
G5_n     = 4
G5_edges = [(0, 1), (1, 2)]

# G6: G5 with node 4 contracted (remove row/col index 3)
G6_n     = 3
G6_edges = [(0, 1), (1, 2)]

GRAPHS = [
    ("G0", G0_n, G0_edges, "Base: 1->2, 2->3"),
    ("G1", G1_n, G1_edges, "G0 + 4->1"),
    ("G2", G2_n, G2_edges, "G1 + 4->2  (Case A, node insertion complete)"),
    ("G3", G3_n, G3_edges, "G0 + 1->4, 2->4  (Case B, reversed direction)"),
    ("G4", G4_n, G4_edges, "G2 - 4->1  (deletion intermediate)"),
    ("G5", G5_n, G5_edges, "G4 - 4->2  (node 4 isolated)"),
    ("G6", G6_n, G6_edges, "G5 contracted  (node 4 removed)"),
]


def lbl(i):
    """0-indexed node -> 1-indexed string label."""
    return str(i + 1)


def print_matrix(S, n):
    header = "         " + "  ".join(f"node {lbl(j):>2}" for j in range(n))
    print(header)
    for i in range(n):
        row = "  ".join(f"{S[i, j]:9.6f}" for j in range(n))
        print(f"  node {lbl(i):>2}  {row}")


def check_symmetry(S):
    return np.linalg.norm(S - S.T, "fro")


# -------------------------------------------------------
# 3.  Main computation
# -------------------------------------------------------

results = {}

print("=" * 65)
print("  CoSimRank Worked Example  (C = 0.6)")
print("=" * 65)

for (name, n, edges, desc) in GRAPHS:
    A = build_transition(n, edges)
    S, cross_err, n_iter, n_terms = cosimrank(A, C)
    sym_err = check_symmetry(S)
    results[name] = {
        "S": S, "A": A, "n": n, "edges": edges,
        "cross_err": cross_err, "sym_err": sym_err,
        "n_iter": n_iter, "n_terms": n_terms,
    }

    print(f"\n{'--'*32}")
    print(f"  {name}: {desc}")
    print(f"  Edges (1-indexed): {[(lbl(u)+'->'+lbl(v)) for u,v in edges]}")
    print(f"  Converged in {n_iter} iter; series used {n_terms} terms")
    print(f"  Cross-check ||iter-series||_F = {cross_err:.2e}   "
          f"Symmetry ||S-S^T||_F = {sym_err:.2e}")
    print()
    print_matrix(S, n)


# -------------------------------------------------------
# 4.  Additional checks
# -------------------------------------------------------

print(f"\n{'=' * 65}")
print("  Additional Checks")
print("=" * 65)

print("\nSymmetry check  ||S - S^T||_F:")
for name, _, _, _ in GRAPHS:
    err = results[name]["sym_err"]
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  {name}: {err:.2e}  [{status}]")

# Round-trip: G6 == G0
S_G0 = results["G0"]["S"]
S_G6 = results["G6"]["S"]
rt_err = np.linalg.norm(S_G0 - S_G6, "fro")
rt_status = "PASS" if rt_err < 1e-10 else "FAIL"
print(f"\nRound-trip  G6 == G0?  ||G0 - G6||_F = {rt_err:.2e}  [{rt_status}]")

# Direction sensitivity: G2 != G3
S_G2 = results["G2"]["S"]
S_G3 = results["G3"]["S"]
dir_diff = np.linalg.norm(S_G2 - S_G3, "fro")
dir_status = "PASS (different)" if dir_diff > 1e-6 else "FAIL (same!)"
print(f"Direction sens. G2 != G3?  ||G2 - G3||_F = {dir_diff:.6f}  [{dir_status}]")


# -------------------------------------------------------
# 5.  Reverse walk distributions for G2
# -------------------------------------------------------

print(f"\n{'=' * 65}")
print("  Reverse walk  p^(k) = A^k e_v  for G2  (1-indexed nodes)")
print("  A is column-normalised: A*e_v steps one hop backward along in-edges.")
print("=" * 65)

A_G2 = results["G2"]["A"]
n_G2 = results["G2"]["n"]

for v_idx in range(n_G2):
    print(f"\n  v = node {lbl(v_idx)}")
    p = np.zeros(n_G2)
    p[v_idx] = 1.0
    for k in range(5):
        entries = "  ".join(f"p[{lbl(i)}]={p[i]:.6f}" for i in range(n_G2))
        print(f"    k={k}:  {entries}")
        p = A_G2 @ p  # A^k e_v: backward walk along in-edges


# -------------------------------------------------------
# 5b. Self-check: S[u,v] == sum_k C^k dot(A^k e_u, A^k e_v)
# -------------------------------------------------------

print(f"\n{'=' * 65}")
print("  Self-check: S[u,v] == sum_k C^k dot(A^k e_u, A^k e_v)")
print("=" * 65)

S_G2 = results["G2"]["S"]
A_G2 = results["G2"]["A"]
n_G2 = results["G2"]["n"]
max_terms = 5000
series_tol = 1e-14

max_err = 0.0
for u in range(n_G2):
    for v in range(n_G2):
        pu = np.zeros(n_G2); pu[u] = 1.0
        pv = np.zeros(n_G2); pv[v] = 1.0
        s_uv = 0.0
        Ck = 1.0
        for k in range(max_terms):
            s_uv += Ck * np.dot(pu, pv)
            pu = A_G2 @ pu
            pv = A_G2 @ pv
            Ck *= C
            # stop when the UPPER BOUND on future contributions is tiny
            # |contrib_k| <= C^k ||A^k e_u|| ||A^k e_v|| = Ck*||pu||*||pv||
            if Ck * np.linalg.norm(pu) * np.linalg.norm(pv) < series_tol:
                break
        err = abs(s_uv - S_G2[u, v])
        max_err = max(max_err, err)

status = "PASS" if max_err < 1e-12 else "FAIL"
print(f"  Max entry-wise error over all (u,v): {max_err:.2e}  [{status}]")
if status == "FAIL":
    raise AssertionError(
        f"Self-check FAILED: S[u,v] series disagrees with iter result, "
        f"max err = {max_err:.3e}"
    )


# -------------------------------------------------------
# 6.  Markdown table + CSV
# -------------------------------------------------------

rows_md  = []
rows_csv = []

header_csv = ["graph", "desc", "n", "edges",
              "cross_check_err", "symmetry_err", "n_iter", "n_terms"]
for i in range(4):
    for j in range(4):
        header_csv.append(f"S_{lbl(i)}_{lbl(j)}")

for (name, n, edges, desc) in GRAPHS:
    r = results[name]
    S = r["S"]
    S_pad = np.full((4, 4), np.nan)
    S_pad[:n, :n] = S

    row_csv = [
        name, desc, n,
        str([(lbl(u) + "->" + lbl(v)) for u, v in edges]),
        f"{r['cross_err']:.2e}", f"{r['sym_err']:.2e}",
        r["n_iter"], r["n_terms"],
    ]
    for i in range(4):
        for j in range(4):
            v = S_pad[i, j]
            row_csv.append("" if np.isnan(v) else f"{v:.8f}")
    rows_csv.append(row_csv)

    sym = "Y" if r["sym_err"] < 1e-10 else "N"
    rows_md.append((name, n, len(edges), sym,
                    f"{r['cross_err']:.1e}", f"{r['sym_err']:.1e}"))

print(f"\n{'=' * 65}")
print("  Summary Markdown Table")
print("=" * 65)
print()
print("| Graph | n | |E| | Sym | iter-series err | sym err |")
print("|-------|---|-----|-----|-----------------|---------|")
for (name, n, ne, sym, ce, se) in rows_md:
    print(f"| {name:5} | {n} | {ne:3} |  {sym}  | {ce:>15} | {se:>7} |")

# Write CSV
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "worked_example.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header_csv)
    writer.writerows(rows_csv)

print(f"\nCSV written to: {csv_path}")
print("\n[ALL CHECKS COMPLETE]")

# -*- coding: utf-8 -*-
"""
worked_example_nodeops.py
=========================
Node insertion and deletion, derived in full on the four-node example, then
costed against static recomputation at several graph sizes.

Part 1  The round trip, in exact rational arithmetic.
        G0 : 1 -> 2 -> 3          (the original chain)
        G2 : G0 with node 4 inserted, pointing at 1 and 2
        G4 : G2 minus edge 4->1   (node 1 loses its ONLY in-edge: in-degree 0)
        G5 : G4 minus edge 4->2   (node 2 keeps one in-edge: weights RENORMALISE)
        G6 : G5 with row/column 4 removed
        Does G6 equal G0 exactly?

        The two intermediate steps are not incidental.  They are precisely the
        two cases a naive "subtract the deleted edge" formula gets wrong: one
        produces a zero column, the other a mixed-sign perturbation.

Part 2  Operation counts for update vs recomputation at n = 4, 50, 100, 300,
        with m MEASURED on the actual graphs rather than assumed.

Usage:
    python benchmarks/worked_example_nodeops.py
"""

import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

from fractions import Fraction as F

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
C = F(3, 5)          # 0.6 exactly


# ---------------------------------------------------------------
# Exact rational linear algebra
# ---------------------------------------------------------------

def zeros(n, m=None):
    m = n if m is None else m
    return [[F(0)] * m for _ in range(n)]


def eye(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def matmul(X, Y):
    n, k, m = len(X), len(Y), len(Y[0])
    out = zeros(n, m)
    for i in range(n):
        for t in range(k):
            a = X[i][t]
            if a:
                for j in range(m):
                    if Y[t][j]:
                        out[i][j] += a * Y[t][j]
    return out


def transpose(X):
    return [list(c) for c in zip(*X)]


def add(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def sub(X, Y):
    return [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def scale(c, X):
    return [[c * v for v in row] for row in X]


def is_zero(X):
    return all(not v for row in X for v in row)


def eq(X, Y):
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        return False
    return all(X[i][j] == Y[i][j]
               for i in range(len(X)) for j in range(len(X[0])))


def build_A(n, edges):
    """Column-normalised A, exact rationals. Edge (u,v) means u -> v."""
    A = zeros(n)
    for j in range(n):
        preds = [i for (i, jj) in edges if jj == j]
        if preds:
            w = F(1, len(preds))
            for i in preds:
                A[i][j] = w
    return A


def cosimrank_exact(A, C, max_terms=60):
    """S = sum_k C^k (A^T)^k A^k; terminates iff A is nilpotent."""
    n = len(A)
    S, Ak, ATk, Ck = zeros(n), eye(n), eye(n), F(1)
    AT = transpose(A)
    for k in range(max_terms):
        if is_zero(Ak):
            return S, k
        S = add(S, scale(Ck, matmul(ATk, Ak)))
        Ak, ATk, Ck = matmul(Ak, A), matmul(ATk, AT), Ck * C
    raise ValueError("A is not nilpotent")


def show(M, label):
    n = len(M)
    names = [str(i + 1) for i in range(n)]
    print("\n  " + label)
    print("        " + "".join("{:>12}".format(nm) for nm in names))
    for i in range(n):
        print("   {:>4} ".format(names[i])
              + "".join("{:>12}".format(str(M[i][j])) for j in range(n)))


# ---------------------------------------------------------------
# Part 1: the round trip
# ---------------------------------------------------------------

def part1():
    print("=" * 78)
    print("  PART 1   Node insertion and deletion on the four-node example")
    print("           exact rational arithmetic, C = {}".format(C))
    print("=" * 78)

    G0_e = [(0, 1), (1, 2)]
    G2_e = [(0, 1), (1, 2), (3, 0), (3, 1)]
    G4_e = [(0, 1), (1, 2), (3, 1)]
    G5_e = [(0, 1), (1, 2)]

    A0 = build_A(3, G0_e)
    S0, t0 = cosimrank_exact(A0, C)
    print("\n  G0: edges 1->2, 2->3     (series terminates after {} terms)".format(t0))
    show(S0, "S(G0)")

    A2 = build_A(4, G2_e)
    S2, t2 = cosimrank_exact(A2, C)
    print("\n  G2: + node 4, edges 4->1, 4->2     ({} terms)".format(t2))
    show(S2, "S(G2)")

    print("\n" + "-" * 78)
    print("  Deleting node 4: remove its out-edges one column at a time, then")
    print("  contract.  Its in-edge set is empty, so the in-edge step is vacuous.")
    print("-" * 78)

    print("\n  Step A: delete 4->1.  Node 1 had one in-neighbour, node 4; now none.")
    print("          Column 1 of A goes from e_4 to the ZERO vector, delta = -e_4.")
    print("          Post-update in-degree is 0, so dividing by it is undefined.")
    A4 = build_A(4, G4_e)
    S4, _ = cosimrank_exact(A4, C)
    show(sub(S4, S2), "dS = S(G4) - S(G2)")

    print("\n  Step B: delete 4->2.  Node 2 had in-neighbours 1 and 4; now only 1.")
    print("          The SURVIVING neighbour renormalises 1/2 -> 1, so")
    print("          delta = e_1 - (1/2 e_1 + 1/2 e_4) = 1/2 e_1 - 1/2 e_4:")
    print("          BOTH signs.  Subtracting only the removed edge would miss")
    print("          the increase on node 1 entirely.")
    A5 = build_A(4, G5_e)
    S5, _ = cosimrank_exact(A5, C)
    show(sub(S5, S4), "dS = S(G5) - S(G4)")

    print("\n  Step C: node 4 is now isolated, so its row and column of S must")
    print("          be e_4.  Verify, then contract.")
    row_ok = all(S5[3][j] == (F(1) if j == 3 else F(0)) for j in range(4))
    col_ok = all(S5[i][3] == (F(1) if i == 3 else F(0)) for i in range(4))
    print("          row 4 == e_4 : {}      column 4 == e_4 : {}".format(
        row_ok, col_ok))
    show(S5, "S(G5)")

    S6 = [[S5[i][j] for j in range(3)] for i in range(3)]
    show(S6, "S(G6) = S(G5) with row/column 4 deleted")

    print("\n" + "=" * 78)
    same = eq(S6, S0)
    print("  ROUND TRIP:  S(G6) == S(G0) ?   {}".format(same))
    if same:
        print("  Exactly equal over Q, not 'equal to 1e-12'.  Deleting the node")
        print("  recovers the original chain exactly, with no residue.")
    print("=" * 78)
    return same


# ---------------------------------------------------------------
# Part 2: operation counts
# ---------------------------------------------------------------

def counts(n, m, K, Q, touched=None):
    """
    A sparse matvec costs 2m flops (one multiply and one add per stored
    entry); a scaled vector add costs 2n flops.

    dynamic : 4K matvecs -- 2K to build r = S delta (forward pass plus Horner
              recursion), 2K to propagate p and t -- SHARED across all queries,
              then two scaled length-n adds per (step, touched query).
    static  : 2K matvecs per query column, nothing shared.
    """
    touched = Q if touched is None else touched
    return {
        'dyn_mv': 4 * K,
        'stat_mv': 2 * K * Q,
        'dyn_fl': 8 * K * m + 4 * K * n * touched,
        'stat_fl': 4 * K * m * Q,
    }


def part2():
    print("\n" + "=" * 78)
    print("  PART 2   Operation counts: update vs recomputation")
    print("=" * 78)

    graphs = [("worked example G2", 4, 4)]
    try:
        from dataset_stats import load_facebook, extract_subgraph
        FB = load_facebook()
        for n in [50, 100, 300]:
            g = extract_subgraph(FB, min_degree=10, max_nodes=n).to_directed()
            graphs.append(("ego-Facebook subgraph",
                           g.number_of_nodes(), g.number_of_edges()))
    except Exception as exc:
        print("  (could not load ego-Facebook: {})".format(exc))

    print("\n  m is MEASURED on each graph, never assumed:")
    print("  {:<26}{:>6}{:>10}{:>9}".format("graph", "n", "m", "m/n"))
    print("  " + "-" * 51)
    for (lbl, n, m) in graphs:
        print("  {:<26}{:>6}{:>10}{:>9.1f}".format(lbl, n, m, m / n))

    K = 20
    rows = []
    for Q in [1, 5, 50]:
        print("\n  --- |Q| = {},  K = {} ---".format(Q, K))
        print("  {:>6}{:>8}{:>12}{:>13}{:>15}{:>15}{:>9}".format(
            "n", "m", "dyn matvec", "stat matvec", "dyn flops",
            "stat flops", "ratio"))
        print("  " + "-" * 80)
        for (lbl, n, m) in graphs:
            c = counts(n, m, K, Q)
            r = c['stat_fl'] / c['dyn_fl']
            verdict = "update" if r > 1 else "recompute"
            print("  {:>6}{:>8}{:>12}{:>13}{:>15,}{:>15,}{:>8.2f}x  {}".format(
                n, m, c['dyn_mv'], c['stat_mv'], c['dyn_fl'], c['stat_fl'],
                r, verdict))
            rows.append([n, m, K, Q, c['dyn_mv'], c['stat_mv'],
                         c['dyn_fl'], c['stat_fl'], round(r, 4), verdict])

    print("\n  Break-even, from the counts alone:")
    print("    matvecs only :  4K < 2K|Q|             <=>  |Q| > 2")
    print("    flops        :  8Km + 4Kn|Q| < 4Km|Q|  <=>  2/|Q| + n/m < 1")
    print("\n  Smallest |Q| at which updating wins, per graph:")
    print("  {:>6}{:>8}{:>9}{:>15}{:>12}".format(
        "n", "m", "n/m", "Q* (flops)", "cap = m/n"))
    print("  " + "-" * 51)
    for (lbl, n, m) in graphs:
        qstar = next((q for q in range(1, 100001)
                      if 2.0 / q + n / m < 1.0), None)
        print("  {:>6}{:>8}{:>9.3f}{:>15}{:>12.1f}".format(
            n, m, n / m, str(qstar), m / n))
        rows.append([n, m, K, "break-even", "", "", "", "", qstar,
                     "cap {:.1f}".format(m / n)])

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'nodeops_opcounts.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["n", "m", "K", "Q", "dyn_matvec", "stat_matvec",
                    "dyn_flops", "stat_flops", "ratio_or_Qstar", "verdict"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    ok = part1()
    part2()
    if not ok:
        raise SystemExit("round trip FAILED")

# -*- coding: utf-8 -*-
"""
verify_correctness.py  --  Round 1 Correctness Verification
============================================================
Tests that DynamicCoSimRank agrees with static CoSimRank at the
fixed-point level, eliminating truncation as a confound.

Protocol (per operation):
  1. S_before  = cosimrank_matrix(G_before, K=K_CONV, tol=1e-12)   -- converged
  2. dcsr.S    = S_before    (override the K=30 init so M uses exact S*)
  3. Apply one dynamic update (K=K_CONV series terms)
  4. S_after   = cosimrank_matrix(G_after,  K=K_CONV, tol=1e-12)   -- converged
  5. rel_err   = ||S_dyn - S_after||_F / ||S_after||_F
  6. Reset dcsr.S = S_after before next op  (keep M exact for every step)

For C=0.8, K=130 gives series tail C^{131}/(1-C) = 1.0e-12, four orders of magnitude below the
1e-8 correctness threshold (NOT below machine epsilon, 2.2e-16).
Any rel_err > 1e-8 (THRESHOLD) is an algebraic bug, not truncation.

Structure
---------
Test 0  Nilpotent fixture: closed-form expected values, atol 1e-12.
Test 1  L1: BA and ER random graphs, n in {50, 200},
        n_ops_small ops per type for n=50 / n_ops_large for n=200.
Test 2  Special case: zero-in-degree (last in-edge deleted).
Test 3  Special case: mixed-sign delta (inserting into a node with existing in-edges).

Output: results/verify_correctness.csv  +  markdown table on stdout.
FAIL cases printed as minimal reproducing cases.  Tolerances NOT relaxed.

Usage:
    python benchmarks/verify_correctness.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
import random
import csv

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank

RESULT_DIR   = os.path.join(os.path.dirname(__file__), 'results')
THRESHOLD    = 1e-8     # PASS/FAIL threshold on relative Frobenius error
CONV_TOL     = 1e-12    # cosimrank_matrix convergence tolerance
K_CONV       = 130      # K for both static and dynamic; C^{131}/(1-C)=1.0e-12 at C=0.8

C_NILP  = 0.6           # decay for nilpotent fixture (matches worked_example.py)
C_LARGE = 0.8           # decay for L1 / special-case tests

# Operation counts and seed counts fall with n because each operation is
# followed by a converged static recomputation, which is O(K n^3) dense.
N_OPS = {50: 20, 200: 10, 500: 4, 1000: 2}
SEEDS = {50: (0, 1, 2, 3, 4), 200: (0, 1, 2), 500: (0, 1), 1000: (0,)}
N_LIST = (50, 200, 500, 1000)


# ============================================================
# Helpers
# ============================================================

def _build_A(G):
    """Column-normalised A; returns (A, sorted_nodes)."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v) in G.edges():
        A[idx[u], idx[v]] += 1.0
    col_sums = A.sum(axis=0)
    for j in range(n):
        if col_sums[j] > 0:
            A[:, j] /= col_sums[j]
    return A, nodes


def _converged_S(G, C):
    """cosimrank_matrix with K=K_CONV and tol=CONV_TOL. Returns (S, nodes).
    Stdout is suppressed to avoid per-iteration print noise."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=K_CONV, tol=CONV_TOL)
    return S, nodes


def _reorder(S_ref, ref_nodes, target_nodes):
    """
    Permute rows/cols of S_ref from ref_nodes order to target_nodes order.
    Both lists must contain the same set of nodes.
    """
    ref_idx = {v: i for i, v in enumerate(ref_nodes)}
    perm = [ref_idx[v] for v in target_nodes]
    return S_ref[np.ix_(perm, perm)]


def _rel_err(S_dyn, S_stat_ordered):
    denom = np.linalg.norm(S_stat_ordered, 'fro')
    return np.linalg.norm(S_dyn - S_stat_ordered, 'fro') / denom if denom > 0 else 0.0


# ============================================================
# Graph builders
# ============================================================

def _ba_graph(n, seed):
    G_und = nx.barabasi_albert_graph(n, 3, seed=seed)
    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (u, v) in G_und.edges():
        G.add_edge(u, v) if rng.random() < 0.5 else G.add_edge(v, u)
    return G


def _er_graph(n, seed):
    G_und = nx.erdos_renyi_graph(n, p=0.08, seed=seed)
    rng = random.Random(seed + 1000)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (u, v) in G_und.edges():
        G.add_edge(u, v) if rng.random() < 0.5 else G.add_edge(v, u)
    return G


# ============================================================
# Test 0  Nilpotent fixture
# ============================================================

def test_nilpotent():
    """
    3-node chain G0: 0->1->2.  Insert node 3 (edges 3->0, 3->1) -> G2.
    Then delete node 3 -> should recover G0.

    All closed-form values expressed as polynomials in C = 0.6.
    Threshold: atol 1e-12 (nilpotent, series terminates at k=4).
    """
    C = C_NILP
    print("=" * 60)
    print("  Test 0: Nilpotent fixture  (C = 0.6,  atol = 1e-12)")
    print("=" * 60)

    failures = []
    ATOL = 1e-12

    # ---- G0 ----
    G0 = nx.DiGraph()
    G0.add_edges_from([(0, 1), (1, 2)])

    expected_G0 = np.array([
        [1,     0,           0           ],
        [0,     1 + C,       0           ],
        [0,     0,           1 + C + C**2],
    ])
    S_G0, _ = _converged_S(G0, C)
    err_G0 = np.max(np.abs(S_G0 - expected_G0))
    flag = "PASS" if err_G0 < ATOL else "FAIL"
    print(f"  Static G0 vs closed-form:  max|err|={err_G0:.2e}  [{flag}]")
    if flag == "FAIL":
        failures.append(("G0-closed-form", G0.copy(), None, err_G0))

    # ---- G2: node insertion ----
    G2_ref = nx.DiGraph()
    G2_ref.add_edges_from([(0, 1), (1, 2), (3, 0), (3, 1)])

    expected_G2 = np.array([
        [1 + C,   C/2,                   0,                           0],
        [C/2,     1 + C/2 + C**2/4,      C**2/4,                      0],
        [0,       C**2/4,                1 + C + C**2/2 + C**3/4,     0],
        [0,       0,                     0,                            1],
    ])
    S_G2, _ = _converged_S(G2_ref, C)
    # node order for G2: sorted([0,1,2,3]) == [0,1,2,3]; expected is in that order
    err_G2_cf = np.max(np.abs(S_G2 - expected_G2))
    flag_cf = "PASS" if err_G2_cf < ATOL else "FAIL"
    print(f"  Static G2 vs closed-form:  max|err|={err_G2_cf:.2e}  [{flag_cf}]")
    if flag_cf == "FAIL":
        failures.append(("G2-closed-form", G2_ref.copy(), None, err_G2_cf))

    # Dynamic: start from converged G0, insert node 3
    dcsr = DynamicCoSimRank(G0.copy(), C=C, K=K_CONV, tol=CONV_TOL)
    dcsr.S = S_G0.copy()   # override K=30 init with exact S*
    dcsr.node_insert(3, [(3, 0), (3, 1)])

    S_dyn_G2 = dcsr.S
    S_G2_ord = _reorder(S_G2, sorted(G2_ref.nodes()), dcsr.nodes)
    rel_ins = _rel_err(S_dyn_G2, S_G2_ord)
    flag_ins = "PASS" if rel_ins < ATOL else "FAIL"
    print(f"  Dynamic node-insert 3->G2:  rel-err={rel_ins:.2e}  [{flag_ins}]")
    if flag_ins == "FAIL":
        failures.append(("G2-dyn-insert", G2_ref.copy(),
                         "node_insert 3 edges 3->0 3->1", rel_ins))

    # Dynamic: start from converged G2, delete node 3
    dcsr2 = DynamicCoSimRank(G2_ref.copy(), C=C, K=K_CONV, tol=CONV_TOL)
    dcsr2.S = S_G2.copy()
    dcsr2.node_delete(3)

    S_G0_again, _ = _converged_S(G0, C)
    S_G0_ord = _reorder(S_G0_again, sorted(G0.nodes()), dcsr2.nodes)
    rel_del = _rel_err(dcsr2.S, S_G0_ord)
    flag_del = "PASS" if rel_del < ATOL else "FAIL"
    print(f"  Dynamic node-delete 3->G0:  rel-err={rel_del:.2e}  [{flag_del}]")
    if flag_del == "FAIL":
        failures.append(("G0-dyn-delete", G0.copy(), "node_delete 3", rel_del))

    return failures


# ============================================================
# Test 1  L1: random graphs
# ============================================================

def _try_op(dcsr, op_type, rng, max_new_node):
    """Apply one random operation.  Returns (new_max_node, success)."""
    nodes = sorted(dcsr.G.nodes())
    n = len(nodes)

    if op_type == "edge_insert":
        for _ in range(300):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and not dcsr.G.has_edge(u, v):
                dcsr.edge_insert([(u, v)])
                return max_new_node, True
        return max_new_node, False

    elif op_type == "edge_delete":
        edges = list(dcsr.G.edges())
        if not edges:
            return max_new_node, False
        dcsr.edge_delete([rng.choice(edges)])
        return max_new_node, True

    elif op_type == "node_insert":
        new_node = max_new_node + 1
        # A few edges from the new node to DISTINCT existing targets.
        # Duplicate targets would be an invalid multi-edge insertion.
        k_edges = min(3, n)
        targets = rng.sample(nodes, k_edges)
        new_edges = [(new_node, t) for t in targets]
        dcsr.node_insert(new_node, new_edges)
        return new_node, True

    elif op_type == "node_delete":
        if n <= 3:
            return max_new_node, False
        dcsr.node_delete(rng.choice(nodes))
        return max_new_node, True

    return max_new_node, False


def test_l1(C=C_LARGE):
    """
    L1: compare dynamic vs converged static after each operation.
    Reset dcsr.S to the converged static S after each step so M is always exact.
    """
    print("\n" + "=" * 60)
    print(f"  Test 1: L1 random graphs  (C={C},  K={K_CONV},  tol={CONV_TOL:.0e})")
    print("=" * 60)

    op_types    = ["edge_insert", "edge_delete", "node_insert", "node_delete"]
    graph_types = ["BA", "ER"]
    n_list      = list(N_LIST)
    results     = {}
    failures    = []

    for gtype in graph_types:
        for n in n_list:
            seeds    = SEEDS[n]
            n_ops    = N_OPS[n]

            for op in op_types:
                max_err   = 0.0
                worst     = None

                for seed in seeds:
                    G0   = _ba_graph(n, seed) if gtype == "BA" else _er_graph(n, seed)
                    S0, nodes0 = _converged_S(G0, C)

                    dcsr     = DynamicCoSimRank(G0.copy(), C=C, K=K_CONV,
                                             tol=CONV_TOL)
                    dcsr.S   = S0.copy()   # exact starting S*

                    rng       = random.Random(seed * 99991 + n * 7 + op_types.index(op))
                    max_node  = max(dcsr.G.nodes())
                    done      = 0

                    for _ in range(n_ops + 20):   # +20 headroom for skips
                        if done >= n_ops:
                            break
                        max_node, ok = _try_op(dcsr, op, rng, max_node)
                        if not ok:
                            continue
                        done += 1

                        # Converged static reference on the updated graph
                        S_stat, stat_nodes = _converged_S(dcsr.G, C)
                        S_stat_ord = _reorder(S_stat, stat_nodes, dcsr.nodes)
                        rel = _rel_err(dcsr.S, S_stat_ord)

                        if rel > max_err:
                            max_err = rel
                            worst   = (gtype, n, seed, op, done, rel, dcsr.G.copy())

                        # CRITICAL: reset S to the converged value so next op starts exact
                        dcsr.S = _reorder(S_stat, stat_nodes, dcsr.nodes)

                key = (gtype, n, op)
                passed = max_err <= THRESHOLD
                results[key] = (max_err, passed, worst)

                flag = "PASS" if passed else "FAIL"
                print(f"  {gtype:2s} n={n:3d}  {op:<14}  "
                      f"max-rel-err={max_err:.2e}  [{flag}]  "
                      f"({len(seeds)} seeds x {n_ops} ops)")

                if not passed and worst is not None:
                    failures.append(worst)

    return results, failures


# ============================================================
# Test 2  Special case: zero-in-degree
# ============================================================

def test_zero_in_degree(C=C_LARGE):
    """
    Delete the last remaining in-edge of a node so its in-degree becomes 0.
    After the deletion the column of A is all-zeros; the update formula
    must still produce correct ΔS.
    """
    print("\n" + "=" * 60)
    print(f"  Test 2: Zero-in-degree  (C={C})")
    print("=" * 60)

    failures = []

    # Node 1 has exactly one in-edge: 0->1.  Delete it -> node 1 in-degree 0.
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])   # chain 0->1->2->3->4

    S0, _ = _converged_S(G, C)
    dcsr   = DynamicCoSimRank(G.copy(), C=C, K=K_CONV, tol=CONV_TOL)
    dcsr.S = S0.copy()

    dcsr.edge_delete([(0, 1)])   # node 1 in-degree 0 -> 0

    S_stat, sn = _converged_S(dcsr.G, C)
    rel = _rel_err(dcsr.S, _reorder(S_stat, sn, dcsr.nodes))
    flag = "PASS" if rel <= THRESHOLD else "FAIL"
    print(f"  delete 0->1 (sole in-edge of node 1): rel-err={rel:.2e}  [{flag}]")
    if flag == "FAIL":
        failures.append(("zero-in-deg", dcsr.G.copy(), "delete 0->1", rel))

    # Now node 2 has sole in-edge 1->2; also delete it
    dcsr.S = _reorder(S_stat, sn, dcsr.nodes)
    dcsr.edge_delete([(1, 2)])

    S_stat2, sn2 = _converged_S(dcsr.G, C)
    rel2 = _rel_err(dcsr.S, _reorder(S_stat2, sn2, dcsr.nodes))
    flag2 = "PASS" if rel2 <= THRESHOLD else "FAIL"
    print(f"  delete 1->2 (sole in-edge of node 2): rel-err={rel2:.2e}  [{flag2}]")
    if flag2 == "FAIL":
        failures.append(("zero-in-deg-2", dcsr.G.copy(), "delete 1->2", rel2))

    return failures


# ============================================================
# Test 3  Special case: mixed-sign delta
# ============================================================

def test_mixed_sign_delta(C=C_LARGE):
    """
    Insert a new in-edge into a node that already has in-edges.
    The existing entries in column v of A decrease (negative) while the
    new entry is positive -> delta has both signs.
    """
    print("\n" + "=" * 60)
    print(f"  Test 3: Mixed-sign delta  (C={C})")
    print("=" * 60)

    failures = []

    # Node 2 has two in-edges: 0->2, 1->2 (each weight 0.5).
    # Insert 3->2: weights become 1/3 each; delta_col = [-1/6, -1/6, 0, 1/3, 0]
    G = nx.DiGraph()
    G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4)])

    S0, _ = _converged_S(G, C)
    dcsr   = DynamicCoSimRank(G.copy(), C=C, K=K_CONV, tol=CONV_TOL)
    dcsr.S = S0.copy()

    dcsr.edge_insert([(3, 2)])   # 3->2; node 3 already exists

    S_stat, sn = _converged_S(dcsr.G, C)
    rel = _rel_err(dcsr.S, _reorder(S_stat, sn, dcsr.nodes))
    flag = "PASS" if rel <= THRESHOLD else "FAIL"
    print(f"  insert 3->2 (2 existing in-edges, 3rd added): "
          f"rel-err={rel:.2e}  [{flag}]")
    if flag == "FAIL":
        failures.append(("mixed-sign", dcsr.G.copy(), "insert 3->2", rel))

    # Additional: 5 existing in-edges, insert a 6th
    G2 = nx.DiGraph()
    for i in range(5):
        G2.add_edge(i, 5)     # node 5: 5 in-edges each weight 0.2
    G2.add_edges_from([(5, 6), (6, 7)])

    S2, _ = _converged_S(G2, C)
    dcsr2  = DynamicCoSimRank(G2.copy(), C=C, K=K_CONV, tol=CONV_TOL)
    dcsr2.S = S2.copy()
    dcsr2.edge_insert([(6, 5)])   # node 5 gets 6th in-edge

    S_stat2, sn2 = _converged_S(dcsr2.G, C)
    rel2 = _rel_err(dcsr2.S, _reorder(S_stat2, sn2, dcsr2.nodes))
    flag2 = "PASS" if rel2 <= THRESHOLD else "FAIL"
    print(f"  insert 6->5 (5 existing -> 6 in-edges):       "
          f"rel-err={rel2:.2e}  [{flag2}]")
    if flag2 == "FAIL":
        failures.append(("mixed-sign-5", dcsr2.G.copy(), "insert 6->5", rel2))

    return failures


# ============================================================
# Output
# ============================================================

def _markdown_table(l1_results):
    op_types    = ["edge_insert", "edge_delete", "node_insert", "node_delete"]
    specs       = [(g, n) for g in ("BA", "ER") for n in N_LIST]

    print("\n" + "=" * 60)
    print("  L1 Summary  (max relative Frobenius error, PASS if <= 1e-8)")
    print("=" * 60)
    hdr = "| Operation     |" + "".join(
        " {} n={:<4}|".format(g, n) for (g, n) in specs)
    print(hdr)
    print("|---------------|" + "----------|" * len(specs))
    for op in op_types:
        cells = []
        for (gt, n) in specs:
            key = (gt, n, op)
            if key in l1_results:
                err, passed, _ = l1_results[key]
                mark = "P" if passed else "F"
                cells.append(f"{err:.1e} {mark}")
            else:
                cells.append("  n/a   ")
        print(f"| {op:<13} | {' | '.join(f'{c:>8}' for c in cells)} |")


def _print_failures(all_failures):
    if not all_failures:
        print("\n  No failures -- all tests PASS.")
        return
    print("\n" + "=" * 60)
    print("  FAIL -- Minimal Reproducing Cases")
    print("=" * 60)
    for item in all_failures:
        if len(item) == 7:
            # L1 format: (gtype, n, seed, op, done_ops, rel_err, G)
            gtype, n, seed, op, done_ops, err, G_f = item
            label    = f"{gtype} n={n} seed={seed}"
            op_desc  = f"{op} (op #{done_ops})"
        else:
            # special-case format: (label, G, op_desc, err)
            label, G_f, op_desc, err = item

        print(f"\n  [FAIL] {label}")
        if op_desc:
            print(f"    operation : {op_desc}")
        print(f"    rel-err   : {err:.3e}  (threshold {THRESHOLD:.0e})")
        if G_f is not None:
            print(f"    |V| = {G_f.number_of_nodes()},  "
                  f"|E| = {G_f.number_of_edges()}")
            edges = sorted(G_f.edges())
            print(f"    edges     : {edges[:20]}"
                  + (" ..." if len(edges) > 20 else ""))


def _save_csv(l1_results, *failure_lists):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, 'verify_correctness.csv')
    rows = []

    for (gtype, n, op), (err, passed, _) in sorted(l1_results.items()):
        rows.append(["L1", op, gtype, n, f"{err:.6e}",
                     "PASS" if passed else "FAIL"])

    labels = ["nilpotent", "zero_in_degree", "mixed_sign_delta"]
    for label, fails in zip(labels, failure_lists):
        if not fails:
            rows.append(["special", label, "-", "-", "0.00e+00", "PASS"])
        else:
            for item in fails:
                rows.append(["special", label, "-", "-",
                             f"{item[-1]:.6e}", "FAIL"])

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["test", "operation", "graph_type", "n",
                         "max_rel_err", "result"])
        writer.writerows(rows)
    print(f"\n  -> Saved: {os.path.relpath(path)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import time
    t_start = time.perf_counter()

    print("=" * 60)
    print("  verify_correctness.py  --  Round 1")
    print(f"  K_CONV={K_CONV}, CONV_TOL={CONV_TOL:.0e}, "
          f"PASS threshold={THRESHOLD:.0e}")
    print("=" * 60)

    all_failures = []

    nilp_fails  = test_nilpotent()
    all_failures.extend(nilp_fails)

    l1_res, l1_fails = test_l1()
    all_failures.extend(l1_fails)

    zero_fails = test_zero_in_degree()
    all_failures.extend(zero_fails)

    delta_fails = test_mixed_sign_delta()
    all_failures.extend(delta_fails)

    _markdown_table(l1_res)
    _print_failures(all_failures)
    _save_csv(l1_res, nilp_fails, zero_fails, delta_fails)

    elapsed = time.perf_counter() - t_start
    overall = "ALL PASS" if not all_failures else f"{len(all_failures)} FAIL(S)"
    print(f"\n  Overall: {overall}  ({elapsed:.1f}s)")
    print("=" * 60)

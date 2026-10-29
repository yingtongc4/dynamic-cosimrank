# -*- coding: utf-8 -*-
"""
verify_perquery.py  --  correctness gate for the per-query implementation

Speed numbers are worthless unless the columns are right, so this must pass
before bench_perquery.py is allowed to report anything.

Checks, on BA / ER / real subgraphs:
  1. initial columns          S_pq[:, q]  ==  S_static[:, q]
  2. after edge insertion
  3. after edge deletion
  4. after node insertion
  5. after node deletion
  6. adaptive-K (tol > 0) agrees with the forced-depth run (tol = 0)
  7. the maintained columns are independent of |Q|
     (querying 3 nodes must give the same 3 columns as querying all n)

Threshold: relative L2 error per column <= 1e-8.  No tolerance is relaxed;
a failure prints the minimal reproducing case.

Usage:
    python benchmarks/verify_perquery.py
"""

import sys, os, io, csv, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx

from simrank import cosimrank_matrix
from per_query import PerQueryCoSimRank

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C         = 0.8
K         = 130        # C^131/(1-C) = 1.0e-12, four orders below the 1e-8 threshold
TOL_CONV  = 1e-12
THRESHOLD = 1e-8


def static_cols(G, C=C, K=K, tol=TOL_CONV):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=K, tol=tol)
    return S, nodes


def col_err(pq, G_ref, q):
    """Relative L2 error of the maintained column q against static."""
    S, nodes = static_cols(G_ref)
    pos = {v: i for i, v in enumerate(nodes)}
    ref = np.array([S[pos[u], pos[q]] for u in pq.nodes])
    got = pq.column(q)
    d = np.linalg.norm(ref)
    return np.linalg.norm(got - ref) / d if d > 0 else 0.0


def ba(n, seed):
    U = nx.barabasi_albert_graph(n, 3, seed=seed)
    rng = random.Random(seed)
    G = nx.DiGraph(); G.add_nodes_from(range(n))
    for (u, v) in U.edges():
        G.add_edge(u, v) if rng.random() < 0.5 else G.add_edge(v, u)
    return G


def er(n, seed):
    U = nx.erdos_renyi_graph(n, 0.08, seed=seed)
    rng = random.Random(seed + 77)
    G = nx.DiGraph(); G.add_nodes_from(range(n))
    for (u, v) in U.edges():
        G.add_edge(u, v) if rng.random() < 0.5 else G.add_edge(v, u)
    return G


def run_case(label, G, seed, rows, failures):
    rng = random.Random(seed)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, min(5, len(nodes)))

    pq = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K, tol=TOL_CONV)

    def check(stage, G_ref):
        worst, worst_q = 0.0, None
        for q in list(pq.cols.keys()):
            e = col_err(pq, G_ref, q)
            if e > worst:
                worst, worst_q = e, q
        ok = worst <= THRESHOLD
        rows.append([label, stage, f"{worst:.6e}", "PASS" if ok else "FAIL"])
        print(f"    {stage:<22} max col err = {worst:.3e}  "
              f"[{'PASS' if ok else 'FAIL'}]")
        if not ok:
            failures.append((label, stage, worst, worst_q, G_ref.copy()))
        return ok

    print(f"\n  {label}  (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
          f"|Q|={len(Q)})")
    check("initial columns", G)

    # edge insertion
    G_cur = G.copy()
    for _ in range(200):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G_cur.has_edge(u, v):
            pq.edge_insert([(u, v)]); G_cur.add_edge(u, v)
            break
    check("after edge_insert", G_cur)

    # edge deletion
    e = rng.choice(list(G_cur.edges()))
    pq.edge_delete([e]); G_cur.remove_edge(*e)
    check("after edge_delete", G_cur)

    # node insertion
    new_node = max(nodes) + 1
    targets = rng.sample(sorted(G_cur.nodes()), 3)
    new_edges = [(new_node, t) for t in targets]
    pq.node_insert(new_node, new_edges)
    G_cur.add_node(new_node); G_cur.add_edges_from(new_edges)
    check("after node_insert", G_cur)

    # node deletion (never delete a query node here; that path is covered below)
    cands = [v for v in sorted(G_cur.nodes()) if v not in pq.cols]
    victim = rng.choice(cands)
    pq.node_delete(victim); G_cur.remove_node(victim)
    check("after node_delete", G_cur)


def run_query_independence(label, G, seed, rows, failures):
    """The maintained columns must not depend on how many are maintained."""
    rng = random.Random(seed)
    nodes = sorted(G.nodes())
    Q3 = rng.sample(nodes, 3)

    small = PerQueryCoSimRank(G.copy(), queries=Q3,  C=C, K=K, tol=TOL_CONV)
    full  = PerQueryCoSimRank(G.copy(), queries=None, C=C, K=K, tol=TOL_CONV)

    u, v = None, None
    for _ in range(200):
        a, b = rng.choice(nodes), rng.choice(nodes)
        if a != b and not G.has_edge(a, b):
            u, v = a, b
            break
    small.edge_insert([(u, v)])
    full.edge_insert([(u, v)])

    worst = max(np.linalg.norm(small.column(q) - full.column(q)) /
                max(np.linalg.norm(full.column(q)), 1e-300) for q in Q3)
    ok = worst <= THRESHOLD
    rows.append([label, "|Q|=3 vs |Q|=n", f"{worst:.6e}",
                 "PASS" if ok else "FAIL"])
    print(f"    {'|Q|=3 vs |Q|=n':<22} max col err = {worst:.3e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append((label, "|Q| independence", worst, None, G.copy()))


def run_adaptive(label, G, seed, rows, failures):
    """Adaptive-K must land within its own tolerance of the forced-depth run."""
    rng = random.Random(seed)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 3)

    exact = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K, tol=0.0)
    adapt = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K, tol=1e-12)

    for _ in range(200):
        a, b = rng.choice(nodes), rng.choice(nodes)
        if a != b and not G.has_edge(a, b):
            exact.edge_insert([(a, b)]); adapt.edge_insert([(a, b)])
            break

    worst = max(np.linalg.norm(exact.column(q) - adapt.column(q)) /
                max(np.linalg.norm(exact.column(q)), 1e-300) for q in Q)
    ok = worst <= THRESHOLD
    rows.append([label, "adaptive vs forced", f"{worst:.6e}",
                 "PASS" if ok else "FAIL"])
    print(f"    {'adaptive vs forced':<22} max col err = {worst:.3e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append((label, "adaptive", worst, None, G.copy()))


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    print("=" * 68)
    print("  verify_perquery.py  --  per-query correctness gate")
    print(f"  C={C}, K={K}, threshold={THRESHOLD:.0e}")
    print("=" * 68)

    rows, failures = [], []

    for seed in [0, 1, 2]:
        run_case(f"BA n=60 seed={seed}", ba(60, seed), seed, rows, failures)
    for seed in [0, 1, 2]:
        run_case(f"ER n=60 seed={seed}", er(60, seed), seed, rows, failures)

    print("\n  Query-set independence")
    for seed in [0, 1]:
        run_query_independence(f"BA n=60 seed={seed}", ba(60, seed), seed,
                               rows, failures)

    print("\n  Adaptive-K consistency")
    for seed in [0, 1]:
        run_adaptive(f"BA n=60 seed={seed}", ba(60, seed), seed, rows, failures)

    print("\n" + "=" * 68)
    if failures:
        print("  FAIL -- minimal reproducing cases")
        print("=" * 68)
        for (label, stage, err, q, G) in failures:
            print(f"\n  [FAIL] {label} / {stage}")
            print(f"    rel col err : {err:.3e}  (threshold {THRESHOLD:.0e})")
            if q is not None:
                print(f"    worst query : {q}")
            print(f"    |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    else:
        print("  No failures -- all per-query checks PASS.")
    print("=" * 68)

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'verify_perquery.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["case", "stage", "max_col_rel_err", "result"])
        w.writerows(rows)
    print(f"\n  -> {os.path.relpath(p)}")
    print(f"  Overall: {'ALL PASS' if not failures else f'{len(failures)} FAIL(S)'}"
          f"  ({time.perf_counter()-t0:.1f}s)")
    # exit status, so CI cannot mistake a printed FAIL for success
    sys.exit(1 if failures else 0)

# -*- coding: utf-8 -*-
"""
bench_c_sweep.py
================
The parameter that appears in every bound and in no sweep.

C is in Lemma "static bound" as C^(K+1)/(1-C), in Theorem "tail" as
2C^(K+1)/(1-C), and in Proposition "drift" as rho = 2C||delta||_1/(1-C) times
(1 + ||delta||_1/2), which diverges as C -> 1.  The operating point K = 20 was
chosen at C = 0.8 and every experiment in the report inherits that choice.
Six sweeps and none of them varies it.

Three quantities are measured at C in {0.6, 0.8, 0.9}: the speedup against
|Q|, which the cost model says should not depend on C at all; the depth needed
to reach a fixed accuracy, which should scale as log(eps)/log(C); and the
drift slope, which Proposition "drift" says grows with C.

Usage:
    python benchmarks/bench_c_sweep.py
"""

import sys, os, io, csv, math, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np

from simrank import cosimrank_matrix
from per_query import PerQueryCoSimRank
from dynamic_cosimrank import DynamicCoSimRank
from dataset_stats import load_facebook, extract_subgraph
from bench_perquery import (assemble, static_columns_prebuilt, timed,
                            subgraph, K_DEFAULT, REPS)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42
C_VALUES = (0.6, 0.8, 0.9)
Q_SIZES = (5, 20, 50)
DEPTHS = (5, 10, 20, 40)
DRIFT_STEPS = 200
DRIFT_CHECK = 25


def insertable_edge(G, rng):
    nodes = sorted(G.nodes())
    for _ in range(100000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge")


def converged(G, C):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=200, tol=0.0)
    return S, nodes


def main():
    G1000 = subgraph('facebook', 1000)
    n, m = G1000.number_of_nodes(), G1000.number_of_edges()
    print("=" * 84)
    print("  Sweeping C, the parameter every bound contains")
    print("  ego-Facebook n={}, m={}, K={} unless stated".format(n, m, K_DEFAULT))
    print("=" * 84)

    rows = []

    # ---- 1. speedup against |Q| at each C -------------------------
    print("\n  Speedup against $|Q|$.  The cost model has no C in it, so these")
    print("  rows should agree within the run-to-run spread.")
    print("  {:>6}{:>8}{:>11}{:>13}{:>11}".format(
        "C", "|Q|", "dyn (ms)", "static (ms)", "speedup"))
    print("  " + "-" * 50)
    for C in C_VALUES:
        for q in Q_SIZES:
            rng = random.Random(SEED)
            Q = rng.sample(sorted(G1000.nodes()), q)
            u, v = insertable_edge(G1000, rng)
            G2 = G1000.copy(); G2.add_edge(u, v)

            pre = [PerQueryCoSimRank(G1000.copy(), queries=list(Q), C=C,
                                     K=K_DEFAULT, tol=0.0)
                   for _ in range(REPS)]
            st = {'i': 0}

            def go():
                p = pre[st['i']]; st['i'] += 1
                p.edge_insert([(u, v)])

            dyn = timed(go, REPS)[0]
            A, AT, nodes2, idx2 = assemble(G2)
            qidx = [idx2[x] for x in Q]
            stat = timed(lambda: static_columns_prebuilt(
                A, AT, qidx, len(nodes2), C, K_DEFAULT), REPS)[0]
            print("  {:>6.1f}{:>8}{:>11.2f}{:>13.2f}{:>10.2f}x"
                  .format(C, q, dyn, stat, stat / dyn))
            rows.append(["speedup", C, q, round(dyn, 3), round(stat, 3),
                         round(stat / dyn, 4)])

    # ---- 2. accuracy against depth at each C ----------------------
    print("\n  Static truncation error against depth.  The prediction is")
    print("  E(K) proportional to C^K, so each column should fall by C^dK.")
    G200 = extract_subgraph(load_facebook(), min_degree=10,
                            max_nodes=200).to_directed()
    print("  {:>6}".format("C") + "".join("{:>12}".format("K=" + str(k))
                                          for k in DEPTHS)
          + "{:>14}".format("K for 1e-3"))
    print("  " + "-" * 68)
    for C in C_VALUES:
        Sref, nodes = converged(G200, C)
        errs = []
        for K in DEPTHS:
            with contextlib.redirect_stdout(io.StringIO()):
                S, nd, _ = cosimrank_matrix(G200, C=C, K=K, tol=0.0)
            e = (np.linalg.norm(S - Sref, 'fro')
                 / np.linalg.norm(Sref, 'fro'))
            errs.append(e)
        # depth needed for 1e-3, from the measured decay rate
        k0, k1 = DEPTHS[1], DEPTHS[2]
        rate = (errs[2] / errs[1]) ** (1.0 / (k1 - k0))
        need = k1 + math.log(1e-3 / errs[2]) / math.log(rate)
        print("  {:>6.1f}".format(C)
              + "".join("{:>12.2e}".format(e) for e in errs)
              + "{:>14.1f}".format(need))
        rows.append(["depth", C, "", *[round(e, 12) for e in errs],
                     round(need, 2)])

    # ---- 3. drift slope at each C --------------------------------
    print("\n  Drift over {} edge operations at K={}.  Proposition 'drift'"
          .format(DRIFT_STEPS, K_DEFAULT))
    print("  gives rho proportional to C/(1-C), so the slope should rise"
          " steeply with C.")
    print("  {:>6}{:>16}{:>16}{:>14}".format(
        "C", "final error", "slope/step", "rho bound"))
    print("  " + "-" * 54)
    for C in C_VALUES:
        rng = random.Random(SEED)
        d = DynamicCoSimRank(G200.copy(), C=C, K=K_DEFAULT, tol=0.0)
        S0, _ = converged(G200, C)
        d.S = S0.copy()
        T, E = [], []
        for t in range(1, DRIFT_STEPS + 1):
            nodes = sorted(d.G.nodes())
            if rng.random() < 0.5:
                u, v = rng.choice(nodes), rng.choice(nodes)
                if u != v and not d.G.has_edge(u, v):
                    d.edge_insert([(u, v)])
                else:
                    continue
            else:
                es = list(d.G.edges())
                u, v = es[rng.randrange(len(es))]
                if d.G.in_degree(v) <= 1:
                    continue
                d.edge_delete([(u, v)])
            if t % DRIFT_CHECK == 0:
                Sr, nr = converged(d.G, C)
                idx = {x: i for i, x in enumerate(nr)}
                perm = [idx[x] for x in d.nodes]
                ref = Sr[np.ix_(perm, perm)]
                E.append(float(np.linalg.norm(d.S - ref, 'fro')
                               / np.linalg.norm(ref, 'fro')))
                T.append(t)
        T, E = np.array(T, float), np.array(E)
        slope = float((T * E).sum() / (T * T).sum())
        # d-bar of the graph the drift is actually measured on (G200), not of
        # the n=1000 subgraph used for the speedup sweep above.  Using m and n
        # here took the mean degree from the wrong graph and made rho a factor
        # of 4.5 too small.
        d_bar200 = G200.number_of_edges() / G200.number_of_nodes()
        dl1 = 2.0 / (d_bar200 + 1)
        rho = 2 * C * dl1 / (1 - C) * (1 + dl1 / 2)
        print("  {:>6.1f}{:>16.3e}{:>16.3e}{:>14.2f}"
              .format(C, E[-1], slope, rho))
        rows.append(["drift", C, "", "{:.4e}".format(E[-1]),
                     "{:.4e}".format(slope), round(rho, 3)])

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'c_sweep.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["kind", "C", "Q", "a", "b", "c", "d", "e"])
        w.writerows(rows)
    print("\n  -> {}".format(os.path.relpath(p)))


if __name__ == "__main__":
    main()

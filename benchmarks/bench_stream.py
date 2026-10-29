# -*- coding: utf-8 -*-
"""
bench_stream.py  --  does error accumulate over a long update stream?

The correctness experiment (verify_correctness.py) resets S to the converged
static value before every operation, so that the driving term M is exact and
the measurement isolates one operation's algebra.  That protocol answers
"is the update formula right?" but says nothing about what happens when
updates are chained, which is the actual deployment mode.

This script removes the reset.  A single S is carried through a long stream
of mixed operations and compared against a converged static recomputation of
the current graph after every step.  Any drift is therefore cumulative.

Two depths are run on identical streams:
    K = 20   a realistic operating point, where each step also truncates
    K = 130  effectively exact, isolating floating-point accumulation alone

Reported per seed: the error trajectory, its maximum, and a least-squares
slope of error against step index (a positive slope indicates growth).

Usage:
    python benchmarks/bench_stream.py
"""

import sys, os, io, csv, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank
from dataset_stats import load_facebook, extract_subgraph

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C        = 0.8
N_NODES  = 200
N_STEPS  = 200
SEEDS    = [0, 1, 2, 3, 4]
K_REAL   = 20
K_EXACT  = 130
TOL      = 1e-12


def converged(G, K):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=K, tol=0.0)
    return S, nodes


def reorder(S, src, dst):
    idx = {v: i for i, v in enumerate(src)}
    return S[np.ix_([idx[v] for v in dst], [idx[v] for v in dst])]


def rel(S_dyn, S_ref):
    d = np.linalg.norm(S_ref, 'fro')
    return np.linalg.norm(S_dyn - S_ref, 'fro') / d if d > 0 else 0.0


def one_step(d, rng, max_node):
    """Apply one random operation; return (new_max_node, op_label or None)."""
    nodes = sorted(d.G.nodes())
    roll = rng.random()

    if roll < 0.40:                                   # edge insert
        for _ in range(200):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v and not d.G.has_edge(u, v):
                d.edge_insert([(u, v)])
                return max_node, 'edge_insert'
        return max_node, None

    if roll < 0.80:                                   # edge delete
        edges = list(d.G.edges())
        if not edges:
            return max_node, None
        d.edge_delete([rng.choice(edges)])
        return max_node, 'edge_delete'

    if roll < 0.90:                                   # node insert
        new_node = max_node + 1
        targets = rng.sample(nodes, min(3, len(nodes)))
        d.node_insert(new_node, [(new_node, t) for t in targets])
        return new_node, 'node_insert'

    if len(nodes) > 10:                               # node delete
        d.node_delete(rng.choice(nodes))
        return max_node, 'node_delete'
    return max_node, None


def run_stream(G0, seed, K):
    rng = random.Random(seed * 7919 + K)
    S0, nodes0 = converged(G0, K)

    d = DynamicCoSimRank(G0.copy(), C=C, K=K, tol=TOL)
    d.S = S0.copy()                      # exact start; never reset again

    max_node = max(d.G.nodes())
    traj, counts = [], {}
    for step in range(N_STEPS):
        max_node, op = one_step(d, rng, max_node)
        if op is None:
            continue
        counts[op] = counts.get(op, 0) + 1
        S_ref, ref_nodes = converged(d.G, K)
        traj.append(rel(d.S, reorder(S_ref, ref_nodes, d.nodes)))
    return np.array(traj), counts


def slope(y):
    """Least-squares slope of error against step index."""
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    print("=" * 76)
    print("  bench_stream.py  --  cumulative error over a mixed update stream")
    print(f"  ego-Facebook n={N_NODES}, {N_STEPS} steps, {len(SEEDS)} seeds, C={C}")
    print("  No reset between operations: drift accumulates.")
    print("=" * 76)

    G0 = extract_subgraph(load_facebook(), min_degree=10,
                          max_nodes=N_NODES).to_directed()
    print(f"  graph: n={G0.number_of_nodes()}, m={G0.number_of_edges()}")

    rows, all_traj = [], {}
    for K in [K_EXACT, K_REAL]:
        print(f"\n  K = {K}"
              f"{'  (effectively exact)' if K == K_EXACT else '  (realistic depth)'}")
        print(f"    {'seed':>5} {'steps':>6} {'first':>11} {'final':>11} "
              f"{'max':>11} {'slope/step':>12}")
        print("    " + "-" * 60)
        for seed in SEEDS:
            traj, counts = run_stream(G0, seed, K)
            all_traj[(K, seed)] = traj
            s = slope(traj)
            print(f"    {seed:>5} {len(traj):>6} {traj[0]:>11.3e} "
                  f"{traj[-1]:>11.3e} {traj.max():>11.3e} {s:>12.3e}")
            rows.append([K, seed, len(traj), f"{traj[0]:.6e}",
                         f"{traj[-1]:.6e}", f"{traj.max():.6e}", f"{s:.6e}",
                         str(counts)])

    # ---- verdict, read off the data ----
    print("\n" + "=" * 76)
    for K in [K_EXACT, K_REAL]:
        sl = [slope(all_traj[(K, s)]) for s in SEEDS]
        mx = [all_traj[(K, s)].max() for s in SEEDS]
        grew = sum(1 for x in sl if x > 0)
        print(f"  K={K:<4} max error over all seeds = {max(mx):.3e}; "
              f"{grew}/{len(SEEDS)} seeds have a positive slope")
    print("=" * 76)

    # ---- figure ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for ax, K in zip(axes, [K_EXACT, K_REAL]):
        for seed in SEEDS:
            ax.semilogy(all_traj[(K, seed)], lw=1, label=f"seed {seed}")
        ax.set_xlabel("operations applied")
        ax.set_ylabel(r"$\|S_{dyn}-S_{stat}\|_F/\|S_{stat}\|_F$")
        ax.set_title(f"$K={K}$" +
                     (" (effectively exact)" if K == K_EXACT
                      else " (realistic depth)"), fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Cumulative error over a mixed update stream, no reset "
                 "between operations", fontsize=11)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_stream_drift.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  -> {os.path.relpath(p)}")

    p = os.path.join(RESULT_DIR, 'bench_stream.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["K", "seed", "steps", "first_err", "final_err",
                    "max_err", "slope_per_step", "op_counts"])
        w.writerows(rows)
    print(f"  -> {os.path.relpath(p)}")
    print(f"  Done ({time.perf_counter()-t0:.1f}s)")

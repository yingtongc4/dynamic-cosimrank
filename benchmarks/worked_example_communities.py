# -*- coding: utf-8 -*-
"""
worked_example_communities.py  --  Two-community worked example
================================================================
Replaces the old 4-node example (1->2->3 plus 4->1,4->2), where the single
update perturbed every entry of S and so could not illustrate anything about
the SCOPE of an update.

This example is built so that the two update types differ in scope:

  LOCAL   insert one edge inside community A.
          Only pairs that already shared ancestors inside A are affected;
          the B x B block is untouched.

  GLOBAL  insert a hub node with edges into BOTH communities.
          Every node gains a common ancestor, so the whole matrix moves.

For each scenario we report what the old example could not:
    * affected cells / total cells        (how far the update reaches)
    * ||dS||_F                            (how much it moves)
    * a per-block breakdown A x A / A x B / B x B
and we run BOTH the dynamic update and a full static recomputation, so the
example doubles as a dynamic-vs-static correctness check (previously the
worked example only ever ran static recomputation).

Outputs:
    stdout
    results/worked_example_communities.csv
    results/fig_worked_example_scope.png

Usage:
    python benchmarks/worked_example_communities.py
"""

import sys, os, io, csv, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C        = 0.8
K_CONV   = 130      # C^131/(1-C) = 1.0e-12, four orders below the 1e-8 threshold -- effectively the fixed point
TOL_CONV = 1e-12
ZERO     = 1e-10    # an entry counts as "changed" above this


# ------------------------------------------------------------------
# The graph
# ------------------------------------------------------------------
# Community A: 0..5      Community B: 6..11
# Each community has two sources feeding two middles feeding two sinks,
# plus feedback edges so that a change inside a community propagates
# through the whole community rather than dying at a sink.
#
# The single bridge runs B -> A (11 -> 0).  Direction matters here:
# CoSimRank walks BACKWARD along in-edges, so changing column w perturbs
# exactly the DESCENDANTS of w.  With the bridge pointing B -> A, no node
# of A has a descendant in B, hence an A-internal update cannot reach the
# B x B block at all.  Reversing the bridge would destroy this separation
# -- an earlier draft used 5 -> 6 and the "local" update moved more of
# B x B than the hub did.

COMM_A = [0, 1, 2, 3, 4, 5]
COMM_B = [6, 7, 8, 9, 10, 11]

BASE_EDGES = [
    # community A
    (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 4), (3, 4), (2, 5), (3, 5),
    (4, 2), (5, 3),                    # feedback inside A
    # community B
    (6, 8), (7, 8), (6, 9), (7, 9),
    (8, 10), (9, 10), (8, 11), (9, 11),
    (10, 8), (11, 9),                  # feedback inside B
    # bridge, B -> A
    (11, 0),
]

# LOCAL update: a new edge entirely inside community A
LOCAL_EDGE = (4, 5)

# GLOBAL update: a hub node reaching into both communities
HUB = 12
HUB_EDGES = [(HUB, 2), (HUB, 3), (HUB, 8), (HUB, 9), (HUB, 4), (HUB, 10)]


def build_base():
    G = nx.DiGraph()
    G.add_nodes_from(COMM_A + COMM_B)
    G.add_edges_from(BASE_EDGES)
    return G


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def converged(G):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=K_CONV, tol=TOL_CONV)
    return S, nodes


def reorder(S, src_nodes, dst_nodes):
    idx = {v: i for i, v in enumerate(src_nodes)}
    perm = [idx[v] for v in dst_nodes]
    return S[np.ix_(perm, perm)]


def block_stats(dS, nodes, rows, cols):
    """Frobenius mass and changed-cell count inside one block of dS."""
    pos = {v: i for i, v in enumerate(nodes)}
    ri = [pos[v] for v in rows if v in pos]
    ci = [pos[v] for v in cols if v in pos]
    if not ri or not ci:
        return 0.0, 0, 0
    B = dS[np.ix_(ri, ci)]
    return float(np.linalg.norm(B, 'fro')), int(np.sum(np.abs(B) > ZERO)), B.size


def scope_report(label, dS, nodes):
    total = dS.size
    changed = int(np.sum(np.abs(dS) > ZERO))
    fro = float(np.linalg.norm(dS, 'fro'))

    aa_f, aa_c, aa_n = block_stats(dS, nodes, COMM_A, COMM_A)
    ab_f, ab_c, ab_n = block_stats(dS, nodes, COMM_A, COMM_B)
    bb_f, bb_c, bb_n = block_stats(dS, nodes, COMM_B, COMM_B)

    print(f"\n  {label}")
    print("  " + "-" * 64)
    print(f"    affected cells / total   : {changed} / {total}"
          f"   ({100.0*changed/total:.1f}%)")
    print(f"    ||dS||_F                 : {fro:.6f}")
    print(f"    block  A x A             : {aa_c:>3}/{aa_n:<4} cells   "
          f"||.||_F = {aa_f:.6f}")
    print(f"    block  A x B             : {ab_c:>3}/{ab_n:<4} cells   "
          f"||.||_F = {ab_f:.6f}")
    print(f"    block  B x B             : {bb_c:>3}/{bb_n:<4} cells   "
          f"||.||_F = {bb_f:.6f}")

    return {
        'scenario': label, 'changed': changed, 'total': total,
        'pct': 100.0 * changed / total, 'fro': fro,
        'AA_cells': aa_c, 'AA_fro': aa_f,
        'AB_cells': ab_c, 'AB_fro': ab_f,
        'BB_cells': bb_c, 'BB_fro': bb_f,
    }


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

def run_local():
    print("=" * 68)
    print(f"  LOCAL update: insert edge {LOCAL_EDGE} inside community A")
    print("=" * 68)

    G = build_base()
    S0, nodes0 = converged(G)

    # dynamic
    d = DynamicCoSimRank(G.copy(), C=C, K=K_CONV, tol=TOL_CONV)
    d.S = S0.copy()
    d.edge_insert([LOCAL_EDGE])

    # static
    G_after = G.copy(); G_after.add_edge(*LOCAL_EDGE)
    S1, nodes1 = converged(G_after)
    S1_ord = reorder(S1, nodes1, d.nodes)

    err = np.linalg.norm(d.S - S1_ord, 'fro') / np.linalg.norm(S1_ord, 'fro')
    print(f"\n  dynamic vs static : rel Frobenius = {err:.3e}  "
          f"[{'PASS' if err < 1e-8 else 'FAIL'}]")
    if err >= 1e-8:
        raise AssertionError(f"dynamic != static ({err:.3e})")

    dS = d.S - reorder(S0, nodes0, d.nodes)
    stats = scope_report("LOCAL  (edge 4->5, inside community A)", dS, d.nodes)
    stats['dyn_vs_static'] = err
    return dS, d.nodes, stats


def run_global():
    print("\n" + "=" * 68)
    print(f"  GLOBAL update: insert hub node {HUB} reaching both communities")
    print(f"  hub edges: {HUB_EDGES}")
    print("=" * 68)

    G = build_base()
    S0, nodes0 = converged(G)

    # dynamic
    d = DynamicCoSimRank(G.copy(), C=C, K=K_CONV, tol=TOL_CONV)
    d.S = S0.copy()
    d.node_insert(HUB, HUB_EDGES)

    # static
    G_after = G.copy()
    G_after.add_node(HUB)
    G_after.add_edges_from(HUB_EDGES)
    S1, nodes1 = converged(G_after)
    S1_ord = reorder(S1, nodes1, d.nodes)

    err = np.linalg.norm(d.S - S1_ord, 'fro') / np.linalg.norm(S1_ord, 'fro')
    print(f"\n  dynamic vs static : rel Frobenius = {err:.3e}  "
          f"[{'PASS' if err < 1e-8 else 'FAIL'}]")
    if err >= 1e-8:
        raise AssertionError(f"dynamic != static ({err:.3e})")

    # S0 must be padded to the new size to form dS
    S0_pad = np.zeros_like(d.S)
    pos = {v: i for i, v in enumerate(d.nodes)}
    for a, u in enumerate(nodes0):
        for b, v in enumerate(nodes0):
            S0_pad[pos[u], pos[v]] = S0[a, b]
    S0_pad[pos[HUB], pos[HUB]] = 1.0     # isolated-node baseline

    dS = d.S - S0_pad
    stats = scope_report("GLOBAL (hub node 12 into both communities)", dS, d.nodes)
    stats['dyn_vs_static'] = err
    return dS, d.nodes, stats


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def make_figure(dS_local, nodes_local, dS_global, nodes_global):
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, dS, nodes, title in [
        (axes[0], dS_local,  nodes_local,
         f"LOCAL: edge {LOCAL_EDGE[0]}$\\to${LOCAL_EDGE[1]} inside community A"),
        (axes[1], dS_global, nodes_global,
         f"GLOBAL: hub node {HUB} into both communities"),
    ]:
        M = np.abs(dS)
        im = ax.imshow(M, cmap='inferno', vmin=0, vmax=max(M.max(), 1e-12))
        ax.set_xticks(range(len(nodes))); ax.set_xticklabels(nodes, fontsize=7)
        ax.set_yticks(range(len(nodes))); ax.set_yticklabels(nodes, fontsize=7)

        # mark the community boundary
        b = len(COMM_A) - 0.5
        ax.axhline(b, color='cyan', lw=1.2, ls='--')
        ax.axvline(b, color='cyan', lw=1.2, ls='--')

        pct = 100.0 * np.sum(M > ZERO) / M.size
        ax.set_title(f"{title}\n{pct:.1f}% of cells changed, "
                     f"$\\|\\Delta S\\|_F$ = {np.linalg.norm(dS,'fro'):.3f}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label='$|\\Delta S_{ij}|$')

    fig.suptitle("Scope of an update: local change stays local, "
                 "a hub moves the whole matrix", fontsize=11)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_worked_example_scope.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  -> {os.path.relpath(p)}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  Two-community worked example   (C = 0.8)")
    print(f"  community A = {COMM_A}")
    print(f"  community B = {COMM_B}")
    print(f"  bridge      = (11, 0)   [B -> A]")
    print("=" * 68)

    dS_l, nodes_l, st_l = run_local()
    dS_g, nodes_g, st_g = run_global()

    print("\n" + "=" * 68)
    print("  Comparison")
    print("=" * 68)
    print(f"  {'scenario':<10} {'cells changed':>16} {'||dS||_F':>12} "
          f"{'B x B moved?':>14}")
    print("  " + "-" * 56)
    for st, tag in [(st_l, 'LOCAL'), (st_g, 'GLOBAL')]:
        moved = "yes" if st['BB_cells'] > 0 else "no"
        print(f"  {tag:<10} {st['changed']:>6}/{st['total']:<6} "
              f"({st['pct']:>4.1f}%) {st['fro']:>12.6f} {moved:>14}")

    # Conclusion is read off the measurements, never asserted in advance.
    print()
    if st_l['BB_cells'] == 0 and st_g['BB_cells'] > 0:
        print("  Measured: the local edge leaves the B x B block bit-identical")
        print(f"  ({st_l['BB_cells']} cells moved, ||.||_F = {st_l['BB_fro']:.1e}),")
        print(f"  while the hub moves {st_g['BB_cells']} B x B cells "
              f"(||.||_F = {st_g['BB_fro']:.3f}).")
        print("  No node of A has a descendant in B, so an A-internal column")
        print("  change cannot enter B; the hub becomes a shared ancestor of")
        print("  both communities and therefore does reach it.")
    else:
        print(f"  Measured: LOCAL moved {st_l['BB_cells']} B x B cells, "
              f"GLOBAL moved {st_g['BB_cells']}.")
        print("  The intended separation is NOT clean -- check the bridge")
        print("  direction and which column the local edge perturbs.")
    print()
    print(f"  Scope ratio (GLOBAL / LOCAL): "
          f"{st_g['pct'] / st_l['pct']:.2f}x more cells, "
          f"{st_g['fro'] / st_l['fro']:.2f}x more Frobenius mass.")
    print("  The old 4-node example admitted no such contrast: every update")
    print("  perturbed every entry, so update scope could not be quantified.")

    make_figure(dS_l, nodes_l, dS_g, nodes_g)

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'worked_example_communities.csv')
    fields = ['scenario', 'changed', 'total', 'pct', 'fro',
              'AA_cells', 'AA_fro', 'AB_cells', 'AB_fro',
              'BB_cells', 'BB_fro', 'dyn_vs_static']
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: st_l[k] for k in fields})
        w.writerow({k: st_g[k] for k in fields})
    print(f"  -> {os.path.relpath(p)}")
    print("=" * 68)

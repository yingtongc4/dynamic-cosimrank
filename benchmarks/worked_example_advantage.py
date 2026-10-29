# -*- coding: utf-8 -*-
"""
worked_example_advantage.py
===========================
A worked example on which the DYNAMIC algorithm is the right choice, and one
on which it is not -- with the reason made explicit rather than asserted.

The earlier example (a 3-node chain plus a node pointing into it) perturbed
every entry of S, so it could show neither locality nor its absence.  Its
replacement showed locality but could not show a speed advantage, because
the implementation ignored it.  This example closes that gap.

Design
------
A planted two-community graph (stochastic block model): communities A and B
are internally dense, joined by a single bridge B -> A.  Direction matters:
CoSimRank walks backward along in-edges, so changing column w perturbs the
DESCENDANTS of w, and with the bridge oriented B -> A no node of A has a
descendant in B.

Three scenarios, all a single edge insertion:

  S1  LOCAL, queries far      edge inside A, all queries inside B.
                              Every dS[:,q] is exactly zero.  The dynamic
                              method detects this and does no per-query work;
                              static recomputation cannot know and redoes
                              everything.  This is the strongest case.
  S2  LOCAL, queries near     edge inside A, all queries inside A.
                              Every query is touched; the advantage comes
                              purely from sharing the O(Km) walk across |Q|.
  S3  GLOBAL                  a hub reaching both communities; every query
                              touched regardless of where it sits.

Two measurements per scenario:
  * operation counts -- exact, hardware-free, checkable by hand
  * wall-clock at increasing n, same planted structure throughout

Correctness is gated first: every reported column is checked against static
recomputation at converged depth.

Usage:
    python benchmarks/worked_example_advantage.py
"""

import sys, os, io, csv, time, random, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.sparse as sp
from per_query import PerQueryCoSimRank
from bench_perquery import static_columns, timed


def assemble(G):
    """Build A (csr) and A^T (csr) once, so a baseline can reuse them."""
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    rows, cols, vals = [], [], []
    for v in nodes:
        preds = list(G.predecessors(v))
        if preds:
            w = 1.0 / len(preds)
            for u in preds:
                rows.append(idx[u]); cols.append(idx[v]); vals.append(w)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A, A.T.tocsr(), nodes, idx


def static_columns_prebuilt(A, AT, q_indices, n, C, K):
    """
    Recompute S[:,q] for the given columns from an ALREADY ASSEMBLED matrix.

    This is the strict algorithmic baseline: it isolates the 2K matvecs per
    column, excluding the sparse assembly that a caller starting from a
    networkx graph would also pay.  Comparing against it is the honest test
    of whether the incremental update saves arithmetic, as opposed to saving
    data-structure construction.
    """
    out = []
    for qi in q_indices:
        e = np.zeros(n); e[qi] = 1.0
        fwd = [e]
        for _ in range(K):
            fwd.append(A @ fwd[-1])
        r = fwd[K].copy()
        for k in range(K - 1, -1, -1):
            r = C * (AT @ r) + fwd[k]
        out.append(r)
    return out

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C       = 0.8
K       = 20
K_GATE  = 130
SEED    = 7
REPS    = 10


# ------------------------------------------------------------------
# The planted two-community graph
# ------------------------------------------------------------------

def two_communities(n, p_in=0.25, p_out=0.0, seed=SEED):
    """
    Directed stochastic block model with two equal communities.

    Community A = 0..n/2-1, community B = n/2..n-1.  Internal edges are drawn
    with probability p_in and given a random direction.  The ONLY link between
    the communities is a single bridge from B to A, so that no node of A has a
    descendant in B and an A-internal update cannot reach the B x B block.
    """
    rng = random.Random(seed)
    h = n // 2
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for lo, hi in ((0, h), (h, n)):
        for u in range(lo, hi):
            for v in range(u + 1, hi):
                if rng.random() < p_in:
                    G.add_edge(u, v) if rng.random() < 0.5 else G.add_edge(v, u)

    G.add_edge(n - 1, 0)          # the bridge, B -> A
    return G, list(range(h)), list(range(h, n))


def free_edge_in(G, nodes, rng):
    for _ in range(20000):
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    raise RuntimeError("no insertable edge inside the community")


# ------------------------------------------------------------------
# Operation counting
# ------------------------------------------------------------------

def op_counts(n, m, K, Q_size, touched_queries):
    """
    Exact operation counts, independent of hardware.

    dynamic : 2K matvecs to form r = S delta (forward + Horner)
              2K matvecs to propagate p^(k) and t^(k)
              -> 4K matvecs, each O(m), shared by every query
              then K scaled vector adds of length n per TOUCHED query
    static  : 2K matvecs per query column, none shared
    """
    return {
        'dyn_matvec':   4 * K,
        'dyn_flops':    4 * K * m + 2 * K * n * touched_queries,
        'stat_matvec':  2 * K * Q_size,
        'stat_flops':   2 * K * m * Q_size,
    }


# ------------------------------------------------------------------
# One scenario
# ------------------------------------------------------------------

def run_scenario(label, G, Q, edges, hub=None, hub_edges=None):
    n, m = G.number_of_nodes(), G.number_of_edges()
    rng = random.Random(SEED)

    # ---- correctness gate at converged depth ----
    probe = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_GATE, tol=0.0)
    if hub is None:
        probe.edge_insert(list(edges))
        G_after = G.copy(); G_after.add_edges_from(edges)
    else:
        probe.node_insert(hub, hub_edges)
        G_after = G.copy(); G_after.add_node(hub); G_after.add_edges_from(hub_edges)

    ref, ref_nodes = static_columns(G_after, Q, C=C, K=K_GATE)
    pos = {v: i for i, v in enumerate(ref_nodes)}
    worst = 0.0
    for q in Q:
        want = np.array([ref[q][pos[u]] for u in probe.nodes])
        d = np.linalg.norm(want)
        worst = max(worst, np.linalg.norm(probe.column(q) - want) / d if d else 0.0)
    if worst > 1e-8:
        raise AssertionError(f"{label}: correctness gate failed ({worst:.3e})")

    # ---- how many query columns are provably untouched ----
    diag = PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K, tol=0.0)
    if hub is None:
        diag.edge_insert(list(edges))
    else:
        diag.node_insert(hub, hub_edges)
    skipped = diag.last_skipped
    touched = len(Q) - skipped

    # ---- timing ----
    pre = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K, tol=0.0)
           for _ in range(REPS)]
    st = {'i': 0}

    def run_dyn():
        p = pre[st['i']]; st['i'] += 1
        if hub is None:
            p.edge_insert(list(edges))
        else:
            p.node_insert(hub, hub_edges)

    dyn = timed(run_dyn, REPS)
    stat = timed(lambda: static_columns(G_after, Q, C=C, K=K), REPS)

    # strict algorithmic baseline: matrix already assembled
    A2, AT2, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    n2 = len(nodes2)
    pre_stat = timed(
        lambda: static_columns_prebuilt(A2, AT2, qidx, n2, C, K), REPS)

    counts = op_counts(n, m, K, len(Q), touched)
    ratio_mv = counts['stat_matvec'] / counts['dyn_matvec']

    print(f"\n  {label}")
    print("  " + "-" * 68)
    print(f"    n={n}, m={m}, |Q|={len(Q)}, gate={worst:.1e}")
    print(f"    query columns touched        : {touched}/{len(Q)}"
          f"   (provably zero, skipped: {skipped})")
    print(f"    matvecs  dynamic / static    : {counts['dyn_matvec']} / "
          f"{counts['stat_matvec']}"
          + (f"   ({ratio_mv:.1f}x fewer)" if ratio_mv >= 1
             else f"   ({1/ratio_mv:.1f}x MORE)"))
    print(f"    time  dynamic                : {dyn[0]:7.2f} ms")
    print(f"    time  static (pre-assembled) : {pre_stat[0]:7.2f} ms"
          f"   -> {pre_stat[0]/dyn[0]:5.2f}x   [algorithmic comparison]")
    print(f"    time  static (from graph)    : {stat[0]:7.2f} ms"
          f"   -> {stat[0]/dyn[0]:5.2f}x   [includes sparse assembly]")

    return {
        'scenario': label, 'n': n, 'm': m, 'Q': len(Q),
        'touched': touched, 'skipped': skipped,
        'gate': worst,
        'dyn_ms': dyn[0], 'stat_ms': stat[0], 'stat_pre_ms': pre_stat[0],
        'speedup': stat[0] / dyn[0],
        'speedup_algo': pre_stat[0] / dyn[0],
        **counts,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    print("=" * 76)
    print("  A worked example where the dynamic algorithm wins -- and where it")
    print(f"  does not.   C={C}, K={K}, median of {REPS} runs")
    print("=" * 76)

    rows = []

    # ---------- illustrative scale ----------
    n0 = 200
    G, A, B = two_communities(n0)
    rng = random.Random(SEED)
    e_local = free_edge_in(G, A, rng)

    QB = rng.sample(B, 20)
    QA = rng.sample(A, 20)

    print(f"\n  Planted two-community graph: |A|={len(A)}, |B|={len(B)}, "
          f"bridge {n0-1}->0")
    print(f"  Local edge inserted: {e_local} (both endpoints in A)")

    # S0: the same two communities with the bridge removed.  Nothing links
    # them, so an A-internal change cannot alter any similarity involving a
    # node of B, and every query column in B is provably zero.
    G_split = G.copy()
    G_split.remove_edge(n0 - 1, 0)
    rows.append(run_scenario(
        "S0  no bridge: local edge, 20 queries in B (unreachable)",
        G_split, QB, [e_local]))

    rows.append(run_scenario(
        "S1  bridged: local edge, 20 queries in B (the far community)",
        G, QB, [e_local]))
    rows.append(run_scenario(
        "S2  bridged: local edge, 20 queries in A (the same community)",
        G, QA, [e_local]))

    hub = n0
    hub_edges = [(hub, A[1]), (hub, A[2]), (hub, B[1]), (hub, B[2])]
    rows.append(run_scenario(
        "S3  hub node into both communities, 20 queries in B",
        G, QB, None, hub=hub, hub_edges=hub_edges))

    # ---------- scaling ----------
    print("\n" + "=" * 76)
    print("  Scaling the same planted structure  (local edge, |Q|=20 in B)")
    print("=" * 76)
    scale_rows = []
    for n in [100, 200, 400, 800]:
        Gn, An, Bn = two_communities(n)
        r = random.Random(SEED)
        e = free_edge_in(Gn, An, r)
        Qn = r.sample(Bn, 20)
        res = run_scenario(f"S1 at n={n}", Gn, Qn, [e])
        res['scenario'] = 'S1_scale'; res['variable'] = n
        scale_rows.append(res); rows.append(res)

    # ---------- |Q| sweep on the winning scenario ----------
    print("\n" + "=" * 76)
    print("  |Q| sweep at n=400  (local edge, queries in B)")
    print("=" * 76)
    q_rows = []
    Gq, Aq, Bq = two_communities(400)
    rq = random.Random(SEED)
    eq = free_edge_in(Gq, Aq, rq)
    for qs in [1, 5, 20, 50, 200]:
        Qq = rq.sample(Bq, qs)
        res = run_scenario(f"|Q|={qs}", Gq, Qq, [eq])
        res['scenario'] = 'S1_Qsweep'; res['variable'] = qs
        q_rows.append(res); rows.append(res)

    # ---------- figure ----------
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ns = [r['n'] for r in scale_rows]
    ax.plot(ns, [r['dyn_ms'] for r in scale_rows], 'o-', label='dynamic update')
    ax.plot(ns, [r['stat_ms'] for r in scale_rows], 's-',
            label='static recomputation')
    ax.set_xlabel('$n$'); ax.set_ylabel('time (ms)')
    ax.set_yscale('log')
    ax.set_title('Local edge, 20 queries in the far community', fontsize=10)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    qs = [r['variable'] for r in q_rows]
    ax.plot(qs, [r['speedup'] for r in q_rows], 'o-', color='#3b7dd8')
    ax.axhline(1.0, color='crimson', ls='--', lw=1.2)
    ax.set_xscale('log'); ax.set_xlabel('$|Q|$')
    ax.set_ylabel('speedup ($\\times$)')
    ax.set_title('$n=400$: advantage grows with $|Q|$', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Where the dynamic update wins: one edge changed, many queries',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_worked_example_advantage.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  -> {os.path.relpath(p)}")

    p = os.path.join(RESULT_DIR, 'worked_example_advantage.csv')
    fields = ['scenario', 'variable', 'n', 'm', 'Q', 'touched', 'skipped',
              'gate', 'dyn_matvec', 'stat_matvec', 'dyn_flops', 'stat_flops',
              'dyn_ms', 'stat_ms', 'speedup']
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  -> {os.path.relpath(p)}")
    print(f"\n  Done ({time.perf_counter()-t0:.1f}s)")

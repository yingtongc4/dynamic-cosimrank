# -*- coding: utf-8 -*-
"""
bench_perquery.py  --  six scalability experiments

Every timing here is gated on correctness: before any configuration is
timed, three randomly chosen query columns are compared against a static
recomputation and the run aborts if the relative error exceeds 1e-8.

Baselines
---------
static-perquery   recompute S[:, q] from scratch for each q in Q, using the
                  same O(K m) Horner recursion.  This is the HONEST baseline:
                  a competent static implementation would never build all of
                  S when only |Q| columns are wanted.       cost O(K m |Q|)

static-full       cosimrank_matrix, the dense n x n fixed-point iteration.
                  This is the baseline the earlier chapters compare against.
                                                            cost O(K n^3)
Dynamic per-query update costs O(K (m + n |Q|)).

Experiments (all report median and IQR over repeated trials)
    E1  |Q|        main result -- the advantage exists only when |Q| << n
    E2  n          growth in graph size
    E3  |dG|       batch size of the update
    E4  dataset    ego-Facebook vs com-DBLP, same n
    E5  deg(v)     node deletion by degree -- locates the crossover
    E6  K          truncation depth

Usage:
    python benchmarks/bench_perquery.py
"""

import sys, os, io, csv, time, random, platform, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import scipy
import scipy.sparse as sp
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simrank import cosimrank_matrix
from per_query import PerQueryCoSimRank
from dataset_stats import load_facebook, load_dblp_graph, extract_subgraph

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

C          = 0.8
K_DEFAULT  = 20        # depth used for TIMING -- a realistic operating point
K_GATE     = 130       # depth used for the CORRECTNESS gate: C^131/(1-C) = 1.0e-12, four orders below the 1e-8 threshold
SEED       = 42
REPS       = 10        # repetitions per timed configuration
REPS_FULL  = 3         # dense n x n baseline is too costly for 10 at n=1000
GATE_TOL   = 1e-8

# The gate runs at K_GATE, not at the timing depth.  Comparing a dynamic
# run truncated at K against a static run truncated at the same K measures
# the DIFFERENCE OF TWO TRUNCATIONS (a Neumann series against a fixed-point
# iteration), not correctness -- Round 2 showed those differ by orders of
# magnitude at equal K.  Correctness is therefore checked where truncation
# is negligible; the timing depth is a separate, independent choice.


# ==================================================================
# Environment
# ==================================================================

def environment():
    return {
        'platform': platform.platform(),
        'processor': platform.processor() or platform.machine(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'networkx': nx.__version__,
        'seed': SEED,
        'C': C,
        'K_default': K_DEFAULT,
        'reps': REPS,
    }


# ==================================================================
# Baselines
# ==================================================================

def static_columns(G, Q, C=C, K=K_DEFAULT):
    """
    Recompute S[:, q] from scratch for each q in Q, sparse, no n x n matrix.
        S e_q = sum_k C^k (A^T)^k (A^k e_q)
    Forward pass then Horner recursion, O(K m) per column.
    """
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
    AT = A.T.tocsr()

    out = {}
    for q in Q:
        e = np.zeros(n); e[idx[q]] = 1.0
        fwd = [e]
        for _ in range(K):
            fwd.append(A @ fwd[-1])
        r = fwd[K].copy()
        for k in range(K - 1, -1, -1):
            r = C * (AT @ r) + fwd[k]
        out[q] = r
    return out, nodes


def assemble(G):
    """Assemble A (csr) and A^T (csr) once, so a baseline can reuse them."""
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


def static_columns_prebuilt(A, AT, q_indices, n, C=C, K=K_DEFAULT):
    """
    Recompute the requested columns from an ALREADY ASSEMBLED matrix.

    This is the primary baseline.  A deployed static implementation would
    maintain its sparse structure across updates exactly as the dynamic one
    does, and pay only the 2K matvecs per column; charging it for a fresh
    assembly on every call would flatter the dynamic side by an amount that
    has nothing to do with the update formula.  `static_columns` above keeps
    the assembly and is reported alongside as the end-to-end cost of a naive
    caller.
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


def static_full(G, C=C, K=K_DEFAULT):
    with contextlib.redirect_stdout(io.StringIO()):
        S, nodes, _ = cosimrank_matrix(G, C=C, K=K, tol=0.0)
    return S, nodes


# ==================================================================
# Timing
# ==================================================================

def timed(fn, reps):
    """Return (median_ms, q1_ms, q3_ms) over `reps` calls."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


# ==================================================================
# Correctness gate
# ==================================================================

def gate(build_probe, G_after, Q, rng, label, n_check=3):
    """
    Verify correctness on `n_check` randomly chosen query columns.

    The probe maintains ONLY those columns, so the gate costs O(n_check K m)
    regardless of |Q|.  Verifying all |Q| columns at the converged depth
    would dominate the whole benchmark at |Q| = n and measures nothing extra:
    columns are maintained independently, as verify_perquery.py checks
    directly ("|Q|=3 vs |Q|=n" is exact to 0).

    Raises if any relative column error exceeds GATE_TOL.
    """
    qs = rng.sample(list(Q), min(n_check, len(Q)))
    pq = build_probe(qs)

    ref, nodes = static_columns(G_after, qs, C=C, K=K_GATE)
    pos = {v: i for i, v in enumerate(nodes)}
    worst = 0.0
    for q in qs:
        want = np.array([ref[q][pos[u]] for u in pq.nodes])
        got = pq.column(q)
        d = np.linalg.norm(want)
        err = np.linalg.norm(got - want) / d if d > 0 else 0.0
        worst = max(worst, err)
    if worst > GATE_TOL:
        raise AssertionError(
            f"CORRECTNESS GATE FAILED at {label}: "
            f"max column rel err = {worst:.3e} > {GATE_TOL:.0e}. "
            "Timings from this configuration are not reportable."
        )
    return worst


# ==================================================================
# Graphs
# ==================================================================

_CACHE = {}

def subgraph(dataset, n):
    """Benchmark subgraph as a DiGraph (both directions of every edge kept)."""
    key = (dataset, n)
    if key in _CACHE:
        return _CACHE[key].copy()
    if dataset == 'facebook':
        G = load_facebook()
    elif dataset == 'dblp':
        G = load_dblp_graph()
    else:
        raise ValueError(dataset)
    D = extract_subgraph(G, min_degree=10, max_nodes=n).to_directed()
    _CACHE[key] = D
    return D.copy()


def pick_new_edges(G, count, rng):
    nodes = sorted(G.nodes())
    out, seen = [], set()
    tries = 0
    while len(out) < count and tries < 20000:
        tries += 1
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v) and (u, v) not in seen:
            seen.add((u, v)); out.append((u, v))
    if len(out) < count:
        raise RuntimeError("could not find enough insertable edges")
    return out


# ==================================================================
# One measurement
# ==================================================================

def measure(G, Q, edges, K=K_DEFAULT, label="", rng=None,
            include_full=True, reps=REPS):
    """
    Time one dynamic update against both baselines, after gating correctness.
    Returns a dict of medians / IQRs.
    """
    rng = rng or random.Random(SEED)
    G_after = G.copy(); G_after.add_edges_from(edges)

    # --- correctness gate (not timed), run at converged depth ---
    def build_probe(qs):
        p = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=K_GATE, tol=0.0)
        p.edge_insert(list(edges))
        return p

    gate_err = gate(build_probe, G_after, Q, rng, label)

    # --- dynamic update (excluding initial build, which is amortised) ---
    prebuilt = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K, tol=0.0)
                for _ in range(reps)]
    state = {'i': 0}

    def run_dynamic():
        pq = prebuilt[state['i']]; state['i'] += 1
        pq.edge_insert(list(edges))

    dyn = timed(run_dynamic, reps)

    # --- baseline 1 (primary): columns recomputed from a prebuilt matrix ---
    A2, AT2, nodes2, idx2 = assemble(G_after)
    qidx = [idx2[q] for q in Q]
    n2 = len(nodes2)
    pre = timed(lambda: static_columns_prebuilt(A2, AT2, qidx, n2, C, K), reps)

    # --- baseline 2: the same, but assembling the matrix each call ---
    spq = timed(lambda: static_columns(G_after, Q, C=C, K=K), reps)

    # --- baseline 3: dense full-matrix static ---
    if include_full:
        full = timed(lambda: static_full(G_after, C=C, K=K), REPS_FULL)
    else:
        full = (float('nan'),) * 3

    return {
        'label': label,
        'n': G.number_of_nodes(), 'm': G.number_of_edges(),
        'Q': len(Q), 'dG': len(edges), 'K': K,
        'gate_err': gate_err,
        'dyn_ms': dyn[0], 'dyn_q1': dyn[1], 'dyn_q3': dyn[2],
        'pre_ms': pre[0], 'pre_q1': pre[1], 'pre_q3': pre[2],
        'spq_ms': spq[0], 'spq_q1': spq[1], 'spq_q3': spq[2],
        'full_ms': full[0], 'full_q1': full[1], 'full_q3': full[2],
        'speedup_algorithmic': pre[0] / dyn[0] if dyn[0] > 0 else float('nan'),
        'speedup_vs_perquery': spq[0] / dyn[0] if dyn[0] > 0 else float('nan'),
        'speedup_vs_full': full[0] / dyn[0] if dyn[0] > 0 else float('nan'),
    }


def show(r):
    print(f"    {r['label']:<26} gate={r['gate_err']:.0e}  "
          f"dyn={r['dyn_ms']:>7.2f}  pre={r['pre_ms']:>8.2f}  "
          f"asm={r['spq_ms']:>8.2f}  full={r['full_ms']:>8.1f}  |  "
          f"x_algo={r['speedup_algorithmic']:>6.2f}  "
          f"x_asm={r['speedup_vs_perquery']:>6.2f}  "
          f"x_full={r['speedup_vs_full']:>7.1f}")


# ==================================================================
# Experiments
# ==================================================================

def E1_query_size(rows):
    print("\n" + "=" * 100)
    print("  E1  |Q| sweep   (ego-Facebook, n=1000, one edge insertion)  "
          "-- MAIN RESULT")
    print("=" * 100)
    G = subgraph('facebook', 1000)
    n = G.number_of_nodes()
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    edges = pick_new_edges(G, 1, rng)

    for q in [1, 5, 10, 50, n]:
        Q = nodes if q == n else rng.sample(nodes, q)
        r = measure(G, Q, edges, label=f"|Q|={q}", rng=rng,
                    include_full=(q in (1, n)))
        r['experiment'] = 'E1_Q'; r['variable'] = q
        show(r); rows.append(r)


def E2_graph_size(rows):
    print("\n" + "=" * 100)
    print("  E2  n sweep   (ego-Facebook, |Q|=5, one edge insertion)")
    print("=" * 100)
    rng = random.Random(SEED)
    for n in [100, 300, 1000]:
        G = subgraph('facebook', n)
        nodes = sorted(G.nodes())
        Q = rng.sample(nodes, 5)
        edges = pick_new_edges(G, 1, rng)
        r = measure(G, Q, edges, label=f"n={G.number_of_nodes()}", rng=rng)
        r['experiment'] = 'E2_n'; r['variable'] = G.number_of_nodes()
        show(r); rows.append(r)


def E3_batch_size(rows):
    print("\n" + "=" * 100)
    print("  E3  |dG| sweep   (ego-Facebook, n=1000, |Q|=5)")
    print("=" * 100)
    G = subgraph('facebook', 1000)
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 5)
    for b in [1, 5, 10, 50]:
        edges = pick_new_edges(G, b, rng)
        r = measure(G, Q, edges, label=f"|dG|={b}", rng=rng,
                    include_full=(b == 1))
        r['experiment'] = 'E3_dG'; r['variable'] = b
        show(r); rows.append(r)


def E4_dataset(rows):
    print("\n" + "=" * 100)
    print("  E4  dataset   (n=500, |Q|=5, one edge insertion)")
    print("=" * 100)
    rng = random.Random(SEED)
    for ds in ['facebook', 'dblp']:
        G = subgraph(ds, 500)
        nodes = sorted(G.nodes())
        Q = rng.sample(nodes, 5)
        edges = pick_new_edges(G, 1, rng)
        r = measure(G, Q, edges,
                    label=f"{ds} n={G.number_of_nodes()} m={G.number_of_edges()}",
                    rng=rng)
        r['experiment'] = 'E4_dataset'; r['variable'] = ds
        show(r); rows.append(r)


def E5_degree(rows):
    """Node deletion: cost grows with deg(v) because each incident edge
    forces one column change.  This is where the crossover lives."""
    print("\n" + "=" * 100)
    print("  E5  deg(v) sweep   (ego-Facebook, n=1000, |Q|=5, node deletion)")
    print("=" * 100)
    G = subgraph('facebook', 1000)
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 5)

    degs = sorted(((G.in_degree(v) + G.out_degree(v), v)
                   for v in nodes if v not in Q))
    picks = [degs[len(degs) // 20], degs[len(degs) // 4],
             degs[len(degs) // 2], degs[3 * len(degs) // 4], degs[-1]]

    for (deg, v) in picks:
        G_after = G.copy(); G_after.remove_node(v)

        def build_probe(qs, _v=v):
            p = PerQueryCoSimRank(G.copy(), queries=qs, C=C, K=K_GATE, tol=0.0)
            p.node_delete(_v)
            return p

        gate_err = gate(build_probe, G_after, Q, rng, f"deg={deg}")

        pre = [PerQueryCoSimRank(G.copy(), queries=Q, C=C, K=K_DEFAULT, tol=0.0)
               for _ in range(REPS)]
        st = {'i': 0}

        def run():
            p = pre[st['i']]; st['i'] += 1
            p.node_delete(v)

        dyn = timed(run, REPS)
        A2, AT2, nodes2, idx2 = assemble(G_after)
        qidx = [idx2[q] for q in Q]
        pre = timed(lambda: static_columns_prebuilt(
            A2, AT2, qidx, len(nodes2), C, K_DEFAULT), REPS)
        spq = timed(lambda: static_columns(G_after, Q, C=C, K=K_DEFAULT), REPS)

        r = {
            'experiment': 'E5_deg', 'variable': deg,
            'label': f"deg(v)={deg}", 'n': G.number_of_nodes(),
            'm': G.number_of_edges(), 'Q': len(Q), 'dG': deg, 'K': K_DEFAULT,
            'gate_err': gate_err,
            'dyn_ms': dyn[0], 'dyn_q1': dyn[1], 'dyn_q3': dyn[2],
            'pre_ms': pre[0], 'pre_q1': pre[1], 'pre_q3': pre[2],
            'spq_ms': spq[0], 'spq_q1': spq[1], 'spq_q3': spq[2],
            'full_ms': float('nan'), 'full_q1': float('nan'),
            'full_q3': float('nan'),
            'speedup_algorithmic': pre[0] / dyn[0],
            'speedup_vs_perquery': spq[0] / dyn[0],
            'speedup_vs_full': float('nan'),
        }
        show(r); rows.append(r)

    xs = [r for r in rows if r['experiment'] == 'E5_deg']
    below = [r for r in xs if r['speedup_algorithmic'] < 1.0]
    if below:
        print(f"\n    crossover: dynamic loses to per-query static from "
              f"deg(v) >= {min(r['variable'] for r in below)}")
    else:
        print("\n    no crossover in the tested degree range; dynamic wins "
              "at every degree measured")


def E6_depth(rows):
    print("\n" + "=" * 100)
    print("  E6  K sweep   (ego-Facebook, n=1000, |Q|=5, one edge insertion)")
    print("=" * 100)
    G = subgraph('facebook', 1000)
    rng = random.Random(SEED)
    nodes = sorted(G.nodes())
    Q = rng.sample(nodes, 5)
    edges = pick_new_edges(G, 1, rng)
    for K in [5, 10, 20, 40, 80]:
        r = measure(G, Q, edges, K=K, label=f"K={K}", rng=rng,
                    include_full=False)
        r['experiment'] = 'E6_K'; r['variable'] = K
        show(r); rows.append(r)


# ==================================================================
# Bar chart
# ==================================================================

def bar_chart(rows):
    os.makedirs(RESULT_DIR, exist_ok=True)
    groups = ['E1_Q', 'E2_n', 'E3_dG', 'E4_dataset', 'E5_deg', 'E6_K']
    titles = {'E1_Q': '|Q|', 'E2_n': 'n', 'E3_dG': '|$\\Delta G$|',
              'E4_dataset': 'dataset', 'E5_deg': 'deg($v$)', 'E6_K': 'K'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, g in zip(axes.ravel(), groups):
        rs = [r for r in rows if r['experiment'] == g]
        labels = [str(r['variable']) for r in rs]
        x = np.arange(len(rs))
        sp_pq = [r['speedup_algorithmic'] for r in rs]

        ax.bar(x, sp_pq, 0.6, color='#3b7dd8', label='vs per-query static')
        ax.axhline(1.0, color='crimson', ls='--', lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yscale('log')
        ax.set_title(f"{titles[g]}", fontsize=11)
        ax.set_ylabel('speedup ($\\times$)', fontsize=9)
        ax.grid(True, axis='y', which='both', alpha=0.3)
        for xi, v in zip(x, sp_pq):
            ax.text(xi, v, f"{v:.1f}", ha='center',
                    va='bottom' if v >= 1 else 'top', fontsize=7)

    fig.suptitle("Dynamic update vs per-query static recomputation from a "
                 "prebuilt matrix\n(median of 10 runs; red line = parity)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(RESULT_DIR, 'fig_scalability_six.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  -> {os.path.relpath(p)}")


# ==================================================================
# Main
# ==================================================================

if __name__ == "__main__":
    t0 = time.perf_counter()
    env = environment()

    print("=" * 100)
    print("  bench_perquery.py  --  six scalability experiments")
    print("=" * 100)
    for k, v in env.items():
        print(f"  {k:<12}: {v}")

    rows = []
    E1_query_size(rows)
    E2_graph_size(rows)
    E3_batch_size(rows)
    E4_dataset(rows)
    E5_degree(rows)
    E6_depth(rows)

    bar_chart(rows)

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, 'bench_perquery.csv')
    fields = ['experiment', 'variable', 'label', 'n', 'm', 'Q', 'dG', 'K',
              'gate_err', 'dyn_ms', 'dyn_q1', 'dyn_q3',
              'pre_ms', 'pre_q1', 'pre_q3',
              'spq_ms', 'spq_q1', 'spq_q3', 'full_ms', 'full_q1', 'full_q3',
              'speedup_algorithmic', 'speedup_vs_perquery', 'speedup_vs_full']
    with open(p, 'w', newline='', encoding='utf-8') as f:
        for k, v in env.items():
            f.write(f"# {k},{v}\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"  -> {os.path.relpath(p)}")
    print(f"\n  Done ({time.perf_counter()-t0:.1f}s)")
    print("=" * 100)

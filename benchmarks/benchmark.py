"""
Full Benchmarking: Extended Dynamic CoSimRank Framework
Section V of the thesis - Experimental Design

Metrics measured:
  1. Update Efficiency  - wall-clock time per update (dynamic vs. full recompute)
  2. Speedup            - full_recompute_time / dynamic_update_time
  3. Frobenius Error    - ||S_dynamic - S_static||_F  (exactness check)
  4. Throughput (UPS)   - updates per second
  5. Scalability        - time vs. graph size on Barabasi-Albert graphs

Datasets used (synthetic, as per thesis Table I):
  - Erdos-Renyi  G(n, p)          - controlled random graph
  - Barabasi-Albert BA(n, m)      - power-law degree distribution (like Twitter)

All results are printed as tables and saved to benchmarks/results/.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
import time
import random
import csv
from collections import defaultdict

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def save_csv(filename, headers, rows):
    """Save results to CSV in the results/ folder."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  -> Saved: results/{filename}")

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def make_directed(G):
    """Convert undirected NetworkX graph to directed (keep both directions)."""
    return G.to_directed()


def static_recompute_time(G, C, K=30):
    """Measure wall-clock time for a full static CoSimRank recomputation."""
    t0 = time.perf_counter()
    cosimrank_matrix(G, C=C, K=K)
    return (time.perf_counter() - t0) * 1000  # ms


def frobenius_error(A, B):
    return np.linalg.norm(A - B, 'fro')


def sample_existing_edge(G, rng):
    edges = list(G.edges())
    return rng.choice(edges) if edges else None


def sample_nonexistent_edge(G, nodes, rng, max_tries=200):
    for _ in range(max_tries):
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            return (u, v)
    return None


def print_table(title, headers, rows):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    print(sep)


# -----------------------------------------------------------------------------
# Benchmark 1: Single-operation timing (all four ops)
# -----------------------------------------------------------------------------

def bench_single_ops(n=50, m=3, C=0.8, K=10, seed=42):
    """
    Measure average time per operation type on a BA(n, m) graph.
    Compare each dynamic update against full recomputation.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark 1: Single-Operation Timing")
    print(f"  Graph: Barabasi-Albert n={n}, m={m}, C={C}, K={K}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    np.random.seed(seed)

    G0 = make_directed(nx.barabasi_albert_graph(n, m, seed=seed))
    nodes = sorted(G0.nodes())

    REPEATS = 10
    results = defaultdict(list)

    for trial in range(REPEATS):
        # Fresh framework each trial
        dcsr = DynamicCoSimRank(G0.copy(), C=C, K=K)

        # --- edge_insert ---
        e = sample_nonexistent_edge(dcsr.G, nodes, rng)
        if e:
            t0 = time.perf_counter()
            dcsr.edge_insert([e])
            results['edge_insert_ms'].append((time.perf_counter() - t0) * 1000)
            err = frobenius_error(dcsr.S, cosimrank_matrix(dcsr.G, C=C, K=30)[0])
            results['edge_insert_err'].append(err)

        # --- edge_delete ---
        e = sample_existing_edge(dcsr.G, rng)
        if e:
            t0 = time.perf_counter()
            dcsr.edge_delete([e])
            results['edge_delete_ms'].append((time.perf_counter() - t0) * 1000)
            err = frobenius_error(dcsr.S, cosimrank_matrix(dcsr.G, C=C, K=30)[0])
            results['edge_delete_err'].append(err)

        # --- node_insert ---
        new_node = max(dcsr.G.nodes()) + 1000 + trial   # unique int id
        neighbors = rng.sample(nodes[:10], min(2, len(nodes)))
        new_edges = [(u, new_node) for u in neighbors]
        t0 = time.perf_counter()
        dcsr.node_insert(new_node, new_edges)
        results['node_insert_ms'].append((time.perf_counter() - t0) * 1000)
        err = frobenius_error(dcsr.S, cosimrank_matrix(dcsr.G, C=C, K=30)[0])
        results['node_insert_err'].append(err)

        # --- node_delete ---
        del_node = rng.choice(nodes[:10])
        if del_node in dcsr.G:
            t0 = time.perf_counter()
            dcsr.node_delete(del_node)
            results['node_delete_ms'].append((time.perf_counter() - t0) * 1000)
            err = frobenius_error(dcsr.S, cosimrank_matrix(dcsr.G, C=C, K=30)[0])
            results['node_delete_err'].append(err)

    # Full recompute baseline
    recompute_ms = np.mean([static_recompute_time(G0, C) for _ in range(5)])

    rows = []
    for op in ['edge_insert', 'edge_delete', 'node_insert', 'node_delete']:
        ms_list = results[f'{op}_ms']
        err_list = results[f'{op}_err']
        if not ms_list:
            continue
        avg_ms = np.mean(ms_list)
        speedup = recompute_ms / avg_ms if avg_ms > 0 else float('inf')
        avg_err = np.mean(err_list)
        rows.append([
            op,
            f"{avg_ms:.3f}",
            f"{recompute_ms:.3f}",
            f"{speedup:.1f}x",
            f"{avg_err:.2e}",
        ])

    print_table(
        f"Single-Op Timing (n={n}, averaged over {REPEATS} trials)",
        ["Operation", "Dyn(ms)", "Static(ms)", "Speedup", "Frob.Error"],
        rows,
    )
    save_csv('single_ops_synthetic.csv',
             ['operation', 'dynamic_ms', 'static_ms', 'speedup', 'frob_error'],
             rows)
    return results, recompute_ms


# -----------------------------------------------------------------------------
# Benchmark 2: Scalability vs. graph size
# -----------------------------------------------------------------------------

def bench_scalability(sizes=None, m=3, C=0.8, K=10, seed=42):
    """
    Measure update time as graph size n grows (Barabasi-Albert model).
    Tests thesis claim: sub-quadratic update complexity O(Kd).
    """
    if sizes is None:
        sizes = [20, 40, 60, 80, 100, 150, 200]

    print(f"\n{'='*60}")
    print(f"  Benchmark 2: Scalability vs. Graph Size")
    print(f"  Model: Barabasi-Albert(n, m={m}), C={C}, K={K}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    rows = []

    for n in sizes:
        np.random.seed(seed)
        G = make_directed(nx.barabasi_albert_graph(n, m, seed=seed))
        nodes = sorted(G.nodes())

        dcsr = DynamicCoSimRank(G.copy(), C=C, K=K)

        # edge insert timing (average over 5 ops)
        ins_times = []
        del_times = []
        for _ in range(5):
            e = sample_nonexistent_edge(dcsr.G, nodes, rng)
            if e:
                t0 = time.perf_counter()
                dcsr.edge_insert([e])
                ins_times.append((time.perf_counter() - t0) * 1000)

            e2 = sample_existing_edge(dcsr.G, rng)
            if e2:
                t0 = time.perf_counter()
                dcsr.edge_delete([e2])
                del_times.append((time.perf_counter() - t0) * 1000)

        static_ms = static_recompute_time(G, C)

        rows.append([
            n,
            G.number_of_edges(),
            f"{np.mean(ins_times):.3f}" if ins_times else "N/A",
            f"{np.mean(del_times):.3f}" if del_times else "N/A",
            f"{static_ms:.3f}",
            f"{static_ms / np.mean(ins_times):.1f}x" if ins_times else "N/A",
        ])

    print_table(
        "Scalability: Dynamic Update Time vs. Static Recomputation",
        ["|V|", "|E|", "Ins(ms)", "Del(ms)", "Static(ms)", "Speedup"],
        rows,
    )
    save_csv('scalability_synthetic.csv',
             ['nodes', 'edges', 'insert_ms', 'delete_ms', 'static_ms', 'speedup'],
             rows)
    return rows


# -----------------------------------------------------------------------------
# Benchmark 3: Update stream simulation (thesis Section V.D)
# -----------------------------------------------------------------------------

def bench_update_stream(n=60, m=3, C=0.8, K=10, num_updates=50, seed=42):
    """
    Simulate a realistic update stream: 40% insert / 40% delete / 20% node ops.
    Measure cumulative Frobenius error and throughput (UPS).
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark 3: Update Stream Simulation")
    print(f"  Graph: BA(n={n}, m={m}), {num_updates} updates")
    print(f"  Mix: 40% edge_ins / 40% edge_del / 20% node_ops, C={C}, K={K}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    np.random.seed(seed)

    G0 = make_directed(nx.barabasi_albert_graph(n, m, seed=seed))
    nodes = list(sorted(G0.nodes()))
    next_node_id = max(G0.nodes()) + 1

    dcsr = DynamicCoSimRank(G0.copy(), C=C, K=K)

    total_time_ms = 0.0
    errors = []
    op_counts = defaultdict(int)

    for i in range(num_updates):
        roll = rng.random()

        if roll < 0.40:
            # Edge insert
            e = sample_nonexistent_edge(dcsr.G, sorted(dcsr.G.nodes()), rng)
            if e:
                t0 = time.perf_counter()
                dcsr.edge_insert([e])
                total_time_ms += (time.perf_counter() - t0) * 1000
                op_counts['edge_insert'] += 1

        elif roll < 0.80:
            # Edge delete
            e = sample_existing_edge(dcsr.G, rng)
            if e:
                t0 = time.perf_counter()
                dcsr.edge_delete([e])
                total_time_ms += (time.perf_counter() - t0) * 1000
                op_counts['edge_delete'] += 1

        elif roll < 0.90:
            # Node insert
            new_node = next_node_id
            next_node_id += 1
            curr_nodes = sorted(dcsr.G.nodes())
            neighbors = rng.sample(curr_nodes, min(2, len(curr_nodes)))
            new_edges = [(u, new_node) for u in neighbors]
            t0 = time.perf_counter()
            dcsr.node_insert(new_node, new_edges)
            total_time_ms += (time.perf_counter() - t0) * 1000
            nodes.append(new_node)
            op_counts['node_insert'] += 1

        else:
            # Node delete
            curr_nodes = sorted(dcsr.G.nodes())
            if len(curr_nodes) > 5:
                del_node = rng.choice(curr_nodes)
                t0 = time.perf_counter()
                dcsr.node_delete(del_node)
                total_time_ms += (time.perf_counter() - t0) * 1000
                op_counts['node_delete'] += 1

        # Check error every 10 updates
        if (i + 1) % 10 == 0:
            err = dcsr.frobenius_error()
            errors.append((i + 1, err))

    ups = op_counts['edge_insert'] + op_counts['edge_delete'] + \
          op_counts['node_insert'] + op_counts['node_delete']
    ups = ups / (total_time_ms / 1000) if total_time_ms > 0 else 0

    print(f"\n  Operations executed:")
    for op, cnt in sorted(op_counts.items()):
        print(f"    {op:<15}: {cnt}")
    print(f"\n  Total update time : {total_time_ms:.2f} ms")
    print(f"  Throughput        : {ups:.1f} updates/sec")
    print(f"\n  Frobenius error at checkpoints:")
    for step, err in errors:
        print(f"    After {step:3d} updates: {err:.2e}")

    final_err = errors[-1][1] if errors else float('nan')
    print(f"\n  Final graph: |V|={dcsr.n}, |E|={dcsr.G.number_of_edges()}")
    print(f"  Final Frobenius error: {final_err:.2e}")

    save_csv('stream_synthetic.csv',
             ['checkpoint_update', 'frobenius_error'],
             [[step, f"{err:.6e}"] for step, err in errors])
    save_csv('stream_summary_synthetic.csv',
             ['n_nodes', 'n_updates', 'total_ms', 'throughput_ups', 'final_frob_error', 'K', 'C'],
             [[n, num_updates, round(total_time_ms, 2), round(ups, 1),
               f"{final_err:.6e}", K, C]])

    return ups, errors


# -----------------------------------------------------------------------------
# Benchmark 4: Effect of K on accuracy vs. speed
# -----------------------------------------------------------------------------

def bench_K_tradeoff(n=60, m=3, C=0.8, seed=42):
    """
    Evaluate the accuracy-speed tradeoff as K (max iterations) varies.
    Validates thesis Section III.C: error bound ||S - S^(K)||_F <= C^(K+1).
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark 4: K vs. Accuracy-Speed Tradeoff")
    print(f"  Graph: BA(n={n}, m={m}), C={C}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    np.random.seed(seed)

    G = make_directed(nx.barabasi_albert_graph(n, m, seed=seed))
    nodes = sorted(G.nodes())
    e = sample_nonexistent_edge(G, nodes, rng)

    rows = []
    for K in [1, 3, 5, 10, 15, 20, 30]:
        dcsr = DynamicCoSimRank(G.copy(), C=C, K=K)
        t0 = time.perf_counter()
        if e:
            dcsr.edge_insert([e])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        err = dcsr.frobenius_error()
        bound = C ** (K + 1)
        rows.append([K, f"{elapsed_ms:.3f}", f"{err:.2e}", f"{bound:.2e}"])

    print_table(
        f"K vs. Accuracy (n={n}, C={C}, edge_insert)",
        ["K", "Time(ms)", "Frob.Error", "Error Bound C^(K+1)"],
        rows,
    )
    save_csv('k_tradeoff_synthetic.csv',
             ['K', 'time_ms', 'frob_error', 'error_bound'],
             rows)
    return rows


# -----------------------------------------------------------------------------
# Benchmark 5: Adaptive K - early stopping vs fixed K
# -----------------------------------------------------------------------------

def bench_adaptive_k(n=60, m=3, C=0.8, K=30, seed=42):
    """
    Compare adaptive K (early stopping) vs fixed K on edge insertion.

    For each tol in a log-scale range, measure:
      - actual iterations used (average over 10 trials)
      - wall-clock time
      - Frobenius error vs K=30 reference

    Validates that adaptive K saves iterations with minimal accuracy loss.
    Results saved to adaptive_k_synthetic.csv.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark 5: Adaptive K - Early Stopping Analysis")
    print(f"  Graph: BA(n={n}, m={m}), C={C}, K_max={K}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    np.random.seed(seed)

    G = make_directed(nx.barabasi_albert_graph(n, m, seed=seed))
    nodes = sorted(G.nodes())

    REPEATS = 10
    tol_values = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 0.0]  # 0.0 = no early stop

    rows = []
    csv_rows = []

    for tol in tol_values:
        times = []
        errors = []

        for trial in range(REPEATS):
            e = sample_nonexistent_edge(G, nodes, rng)
            if e is None:
                continue

            dcsr = DynamicCoSimRank(G.copy(), C=C, K=K, tol=tol if tol > 0 else 1e-300)
            t0 = time.perf_counter()
            dcsr.edge_insert([e])
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            err = dcsr.frobenius_error()
            errors.append(err)

        avg_ms  = np.mean(times)
        avg_err = np.mean(errors)
        label   = f"{tol:.0e}" if tol > 0 else "none (fixed K)"

        rows.append([label, f"{avg_ms:.3f}", f"{avg_err:.2e}"])
        csv_rows.append([label, round(avg_ms, 4), f"{avg_err:.6e}"])

    print_table(
        f"Adaptive K (n={n}, K_max={K}, C={C}, averaged {REPEATS} trials)",
        ["tol", "Time(ms)", "Frob.Error"],
        rows,
    )
    save_csv('adaptive_k_synthetic.csv',
             ['tol', 'time_ms', 'frob_error'],
             csv_rows)
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Extended Dynamic CoSimRank - Full Benchmark Suite")
    print("  Thesis Section V: Experimental Design")
    print("=" * 60)

    # Create output directory
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)

    # Run all benchmarks
    print("\nNote: 'Frob.Error' = ||S_dynamic(K) - S_static(K=30)||_F")
    print("      Error > 0 when K < 30 (expected truncation gap, not a correctness bug).")
    print("      See Benchmark 4 for the K vs. accuracy tradeoff.\n")

    bench_single_ops(n=50, m=3, C=0.8, K=10)
    bench_scalability(sizes=[20, 40, 60, 80, 100, 150, 200], m=3, C=0.8, K=10)
    bench_update_stream(n=60, m=3, C=0.8, K=10, num_updates=50)
    bench_K_tradeoff(n=60, m=3, C=0.8)
    bench_adaptive_k(n=60, m=3, C=0.8, K=30)

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print(f"  Results saved to benchmarks/results/")
    print(f"{'='*60}\n")


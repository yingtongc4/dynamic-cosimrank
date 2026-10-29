"""
DBLP Real-World Benchmark
Thesis Section V - Experimental Design (real dataset)

Uses the SNAP DBLP collaboration network (317K nodes, 1M edges).
We extract a dense subgraph via degree filtering + BFS for feasibility.

Experiments:
  1. Correctness check  - verify dynamic updates match static recomputation
  2. Delta-S efficiency - compare dynamic vs. static via the dS expression:
       * theoretical flop count: dynamic O(Kn) vs. static O(Kn^2)
       * measured nonzero ratio of dS (sparsity of the update)
       * wall-clock time and speedup at different graph sizes
  3. Update stream      - simulate realistic edge insertions and deletions

Results are saved as CSV files to benchmarks/results/ for use in the thesis.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import networkx as nx
import time
import random
import csv

from simrank import cosimrank_matrix
from dynamic_cosimrank import DynamicCoSimRank
from load_dblp import load_dblp, extract_subgraph

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'com-dblp.txt')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def save_csv(filename, headers, rows):
    """Save results to CSV in the results/ folder."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  -> Saved: results/{filename}")


def static_time(G, C, K=30):
    t0 = time.perf_counter()
    cosimrank_matrix(G, C=C, K=K)
    return (time.perf_counter() - t0) * 1000


def print_table(title, headers, rows):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    print(sep)


# -------------------------------------------------------------
# Experiment 1: Correctness on DBLP subgraph
# -------------------------------------------------------------

def exp_correctness(G, C=0.8, K=10, n_ops=10, seed=42):
    """
    Verify that dynamic updates produce the same result as static recomputation
    on a real DBLP subgraph. Results saved to correctness_dblp.csv.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment 1: Correctness Check (DBLP, n={G.number_of_nodes()})")
    print(f"  C={C}, K={K}, testing {n_ops} random operations")
    print(f"{'='*60}")

    rng = random.Random(seed)
    dcsr = DynamicCoSimRank(G.copy(), C=C, K=K)

    csv_rows = []
    for i in range(n_ops):
        op_type = rng.choice(['insert', 'delete'])

        if op_type == 'delete' and dcsr.G.number_of_edges() > 10:
            e = rng.choice(list(dcsr.G.edges()))
            t0 = time.perf_counter()
            dcsr.edge_delete([e])
            elapsed = (time.perf_counter() - t0) * 1000
            label = f"delete ({e[0]},{e[1]})"
        else:
            curr_nodes = sorted(dcsr.G.nodes())
            found = False
            for _ in range(50):
                u = rng.choice(curr_nodes)
                v = rng.choice(curr_nodes)
                if u != v and not dcsr.G.has_edge(u, v):
                    t0 = time.perf_counter()
                    dcsr.edge_insert([(u, v)])
                    elapsed = (time.perf_counter() - t0) * 1000
                    label = f"insert ({u},{v})"
                    found = True
                    break
            if not found:
                continue

        err = dcsr.frobenius_error()
        csv_rows.append([i + 1, label, f"{elapsed:.2f}", f"{err:.6e}",
                         'PASS' if err < 0.05 else 'FAIL'])
        print(f"  Op {i+1:2d}: {label:<30}  time={elapsed:.2f}ms  "
              f"Frob.error={err:.2e}")

    all_pass = all(r[4] == 'PASS' for r in csv_rows)
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'} "
          f"(threshold=0.05, K={K})")

    save_csv('correctness_dblp.csv',
             ['op_id', 'operation', 'time_ms', 'frobenius_error', 'result'],
             csv_rows)
    return csv_rows


# -------------------------------------------------------------
# Experiment 2: Delta-S based efficiency comparison
# -------------------------------------------------------------
#
# Motivation from the dS formula:
#   dS[:,q] = sum_{k=0}^{K} C^k ( t^(k)[q]*p^(k) + p^(k)[q]*t^(k) )
#
# Each p^(k) and t^(k) is an n-dim vector updated by one matrix-vector product.
# For a single edge operation only ONE column of A changes (rank-one),
# so the dynamic update costs O(K*n) flops.
# Static recomputation costs O(K*n^2) flops.
# Theoretical speedup = n / (2*d_affected), where d_affected ~= 1 column.
#
# This experiment measures:
#   (a) Theoretical flop ratio: static_flops / dynamic_flops = n / 2
#   (b) Nonzero ratio of dS:    fraction of entries actually changed
#   (c) Wall-clock speedup:     static_ms / dynamic_ms
# and checks that (c) tracks (a) as n grows.

def _compute_delta_S(G, S, edge, C, K):
    """
    Run one dynamic update and return the resulting dS matrix
    (difference between updated S and old S), used to measure sparsity.
    """
    from d_cosim import d_cosim
    S_new, _ = d_cosim(G.copy(), S.copy(), [edge], C=C, K=K)
    return S_new - S


def exp_efficiency(G_full, sizes, C=0.8, K=10, seed=42, sparsity_threshold=1e-8):
    """
    Compare dynamic update vs. static recomputation using the dS expression.

    For each subgraph size n:
      - Compute theoretical flop counts for static and dynamic updates.
      - Measure the sparsity (nonzero ratio) of dS.
      - Measure wall-clock time for both methods.
      - Compute the theoretical crossover point where dynamic beats static.

    Results saved to speed_dblp.csv.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment 2: Delta-S Efficiency Comparison (DBLP)")
    print(f"  Method: dS expression analysis + wall-clock timing")
    print(f"  C={C}, K={K}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    rows = []
    csv_rows = []

    for n in sizes:
        G = None
        for min_deg in [15, 10, 5, 3, 1]:
            try:
                G = extract_subgraph(G_full, min_degree=min_deg, max_nodes=n)
                if G.number_of_nodes() >= n * 0.8:
                    break
            except Exception:
                continue

        if G is None or G.number_of_nodes() < 10:
            continue

        actual_n = G.number_of_nodes()
        actual_e = G.number_of_edges()

        # --- Theoretical flop counts from dS structure ---
        # Static: K iterations of n x n matrix multiply = K * n^2 flops
        # Dynamic: K iterations of n-dim vector operations (2 walks) = 2 * K * n flops
        # (one column change -> rank-one perturbation -> O(Kn) total)
        static_flops  = K * actual_n ** 2
        dynamic_flops = 2 * K * actual_n
        theoretical_speedup = static_flops / dynamic_flops  # = n / 2

        # --- Sparsity of dS ---
        # Measure what fraction of the n x n matrix actually changes per update.
        # A small fraction confirms the rank-one nature of the perturbation.
        S_ref, _, _ = cosimrank_matrix(G, C=C, K=K)
        dcsr = DynamicCoSimRank(G.copy(), C=C, K=K)
        curr_nodes = sorted(dcsr.G.nodes())

        delta_S = None
        ins_edge = None
        for _ in range(100):
            u, v = rng.choice(curr_nodes), rng.choice(curr_nodes)
            if u != v and not dcsr.G.has_edge(u, v):
                ins_edge = (u, v)
                delta_S = _compute_delta_S(G, S_ref, ins_edge, C, K)
                break

        if delta_S is None:
            continue

        total_entries = actual_n * actual_n
        nonzero_entries = int(np.count_nonzero(np.abs(delta_S) > sparsity_threshold))
        nonzero_ratio = nonzero_entries / total_entries

        # --- Wall-clock timing ---
        t0 = time.perf_counter()
        dcsr.edge_insert([ins_edge])
        dyn_ms = (time.perf_counter() - t0) * 1000

        st_ms = np.mean([static_time(G, C, K=K) for _ in range(3)])
        wall_speedup = st_ms / dyn_ms if dyn_ms > 0 else float('nan')

        rows.append([
            actual_n,
            actual_e,
            f"{static_flops:,}",
            f"{dynamic_flops:,}",
            f"{theoretical_speedup:.1f}x",
            f"{nonzero_ratio:.4f}",
            f"{dyn_ms:.2f}",
            f"{st_ms:.2f}",
            f"{wall_speedup:.2f}x",
        ])
        csv_rows.append([
            actual_n, actual_e,
            static_flops, dynamic_flops,
            round(theoretical_speedup, 2),
            round(nonzero_ratio, 6),
            round(dyn_ms, 4),
            round(st_ms, 4),
            round(wall_speedup, 4),
        ])

        print(f"  n={actual_n:4d}  |  theory speedup={theoretical_speedup:.1f}x  "
              f"|  dS nonzero={nonzero_ratio:.4f}  "
              f"|  wall speedup={wall_speedup:.2f}x  "
              f"({'dyn faster' if wall_speedup > 1 else 'static faster'})")

    print_table(
        "Delta-S Efficiency: Dynamic vs. Static (DBLP)",
        ["|V|", "|E|", "Static flops", "Dynamic flops",
         "Theory x", "dS nonzero%", "Dyn(ms)", "Static(ms)", "Wall x"],
        rows,
    )

    # Crossover point analysis
    crossover = [r for r in csv_rows if r[8] > 1.0]
    if crossover:
        print(f"\n  Crossover: dynamic faster than static at n >= {crossover[0][0]}")
        print(f"  Theoretical crossover at n/2 > overhead_factor")
    else:
        print(f"\n  No crossover observed in tested range; increase n.")

    save_csv('speed_dblp.csv',
             ['nodes', 'edges', 'static_flops', 'dynamic_flops',
              'theoretical_speedup', 'ds_nonzero_ratio',
              'dynamic_ms', 'static_ms', 'wall_speedup'],
             csv_rows)
    return csv_rows


# -------------------------------------------------------------
# Experiment 3: Update stream on DBLP
# -------------------------------------------------------------

def exp_stream(G, C=0.8, K=10, n_updates=30, seed=42):
    """
    Simulate a realistic update stream on a DBLP subgraph.
    Results saved to stream_dblp.csv.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment 3: Update Stream  (DBLP n={G.number_of_nodes()}, "
          f"{n_updates} updates, C={C}, K={K})")
    print(f"{'='*60}")

    rng = random.Random(seed)
    dcsr = DynamicCoSimRank(G.copy(), C=C, K=K)
    curr_nodes = sorted(dcsr.G.nodes())

    total_ms = 0.0
    op_counts = {'insert': 0, 'delete': 0}
    csv_rows = []

    for i in range(n_updates):
        roll = rng.random()
        op_label = ''
        elapsed = 0.0

        if roll < 0.5:
            for _ in range(50):
                u = rng.choice(curr_nodes)
                v = rng.choice(curr_nodes)
                if u != v and not dcsr.G.has_edge(u, v):
                    t0 = time.perf_counter()
                    dcsr.edge_insert([(u, v)])
                    elapsed = (time.perf_counter() - t0) * 1000
                    total_ms += elapsed
                    op_counts['insert'] += 1
                    op_label = f"insert ({u},{v})"
                    break
        else:
            if dcsr.G.number_of_edges() > 20:
                e = rng.choice(list(dcsr.G.edges()))
                t0 = time.perf_counter()
                dcsr.edge_delete([e])
                elapsed = (time.perf_counter() - t0) * 1000
                total_ms += elapsed
                op_counts['delete'] += 1
                op_label = f"delete ({e[0]},{e[1]})"

        err = dcsr.frobenius_error()
        csv_rows.append([i + 1, op_label, round(elapsed, 4), f"{err:.6e}"])

    total_ops = sum(op_counts.values())
    ups = total_ops / (total_ms / 1000) if total_ms > 0 else 0

    print(f"\n  Operations : insert={op_counts['insert']}, "
          f"delete={op_counts['delete']}")
    print(f"  Total time : {total_ms:.1f} ms")
    print(f"  Throughput : {ups:.1f} updates/sec")
    print(f"\n  Frobenius error per update (K={K}):")
    for row in csv_rows:
        if row[2] > 0:
            print(f"    Op {row[0]:2d} {row[1]:<28} err={row[3]}")

    save_csv('stream_dblp.csv',
             ['op_id', 'operation', 'time_ms', 'frobenius_error'],
             csv_rows)

    # Summary row
    save_csv('stream_summary_dblp.csv',
             ['n_nodes', 'n_edges', 'n_inserts', 'n_deletes',
              'total_ms', 'throughput_ups', 'K', 'C'],
             [[G.number_of_nodes(), G.number_of_edges(),
               op_counts['insert'], op_counts['delete'],
               round(total_ms, 2), round(ups, 1), K, C]])

    return ups, csv_rows


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  DBLP Real-World Benchmark")
    print("  Extended Dynamic CoSimRank Framework")
    print("=" * 60)

    print("\nLoading DBLP dataset...")
    G_full = load_dblp(DATA_PATH)
    print(f"  Full DBLP : {G_full.number_of_nodes()} nodes, "
          f"{G_full.number_of_edges()} edges")

    G200 = extract_subgraph(G_full, min_degree=10, max_nodes=200)
    print(f"  Subgraph  : {G200.number_of_nodes()} nodes, "
          f"{G200.number_of_edges()} edges\n")

    exp_correctness(G200, C=0.8, K=10, n_ops=10)
    exp_efficiency(G_full, sizes=[50, 100, 150, 200, 250, 300], C=0.8, K=10)
    exp_stream(G200, C=0.8, K=10, n_updates=30)

    print(f"\n{'='*60}")
    print(f"  All results saved to benchmarks/results/")
    print(f"{'='*60}\n")


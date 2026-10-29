# -*- coding: utf-8 -*-
"""
dataset_stats.py  --  Real statistics for every dataset used in the thesis.

Produces the numbers quoted in the Introduction and the Experimental Setup.
No number in the write-up should be invented: everything here is measured.

Datasets
--------
ego-Facebook (SNAP)  social follow graph -- motivates the follow/unfollow
                     scenario used as the running example.
com-DBLP     (SNAP)  co-authorship graph -- second dataset, checks whether
                     the scalability trend is dataset-independent.

For each dataset we report the full graph and the benchmark subgraph:
  n, m (undirected), m (directed), mean degree, median/max degree,
  degree distribution percentiles, average clustering coefficient, density.

Usage:
    python data/dataset_stats.py
"""

import os
import gzip
import csv
import statistics
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, '..', 'benchmarks', 'results')

FACEBOOK_GZ = os.path.join(HERE, 'facebook_combined.txt.gz')
FACEBOOK_TXT = os.path.join(HERE, 'facebook_combined.txt')
DBLP_TXT = os.path.join(HERE, 'com-dblp.txt')
DBLP_GZ = os.path.join(HERE, 'com-dblp.txt.gz')


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def _read_edgelist(path):
    """Read a SNAP-format undirected edge list (whitespace, '#' comments)."""
    G = nx.Graph()
    opener = gzip.open if path.endswith('.gz') else open
    mode = 'rt' if path.endswith('.gz') else 'r'
    with opener(path, mode) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


def load_facebook():
    path = FACEBOOK_TXT if os.path.exists(FACEBOOK_TXT) else FACEBOOK_GZ
    return _read_edgelist(path)


def load_dblp_graph():
    # prefer the plain file if it is present, fall back to the gzip, so the
    # 14 MB uncompressed copy does not have to be shipped
    path = DBLP_TXT if os.path.exists(DBLP_TXT) else DBLP_GZ
    return _read_edgelist(path)


TWITTER_GZ = os.path.join(HERE, 'twitter_combined.txt.gz')


def load_twitter():
    """
    SNAP ego-Twitter, returned as a DiGraph with its own edge directions.

    This is the only dataset here that is directed as distributed, so it is
    the only one on which "u follows v" is the operation the graph records
    rather than one imposed on it.  It must NOT be symmetrised: 48.2% of its
    edges are reciprocated and the other half carry the asymmetry the follow
    semantics depend on.

    The distributed file merges 973 ego networks and repeats an edge once per
    network it appears in, so 2,420,766 lines reduce to the 1,768,149 unique
    edges SNAP reports; add_edge deduplicates.
    """
    G = nx.DiGraph()
    with gzip.open(TWITTER_GZ, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def describe(G, label, clustering_sample=None):
    """
    Measure an undirected graph and the directed version used by CoSimRank.

    The thesis converts an undirected graph to directed by keeping both
    directions of every edge, so m_directed = 2 * m_undirected, and the
    in-degree of a node equals its undirected degree.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]

    if clustering_sample and n > clustering_sample:
        # Exact average clustering is O(sum d^2); sample on very large graphs.
        import random
        rng = random.Random(42)
        nodes = rng.sample(list(G.nodes()), clustering_sample)
        cc = nx.average_clustering(G, nodes=nodes)
        cc_note = f"sampled {clustering_sample} nodes"
    else:
        cc = nx.average_clustering(G)
        cc_note = "exact"

    degs_sorted = sorted(degs)

    def pct(p):
        if not degs_sorted:
            return 0
        idx = min(int(p / 100 * len(degs_sorted)), len(degs_sorted) - 1)
        return degs_sorted[idx]

    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0

    return {
        'label': label,
        'n': n,
        'm_undirected': m,
        'm_directed': 2 * m,
        'mean_degree': sum(degs) / n if n else 0.0,
        'median_degree': statistics.median(degs) if degs else 0,
        'max_degree': max(degs) if degs else 0,
        'min_degree': min(degs) if degs else 0,
        'p25': pct(25), 'p50': pct(50), 'p75': pct(75),
        'p90': pct(90), 'p99': pct(99),
        'avg_clustering': cc,
        'clustering_note': cc_note,
        'density': density,
        'n_components': nx.number_connected_components(G),
    }


def print_row(s):
    print(f"\n  {s['label']}")
    print("  " + "-" * 62)
    print(f"    nodes  n                      : {s['n']:,}")
    print(f"    edges  m (undirected)         : {s['m_undirected']:,}")
    print(f"    edges  m (directed, both dir) : {s['m_directed']:,}")
    print(f"    mean degree  d_bar            : {s['mean_degree']:.2f}")
    print(f"    median / max degree           : {s['median_degree']} / {s['max_degree']}")
    print(f"    degree pct  25/50/75/90/99    : "
          f"{s['p25']} / {s['p50']} / {s['p75']} / {s['p90']} / {s['p99']}")
    print(f"    avg clustering coefficient    : {s['avg_clustering']:.4f} "
          f"({s['clustering_note']})")
    print(f"    density                       : {s['density']:.6f}")
    print(f"    connected components          : {s['n_components']}")


# ------------------------------------------------------------------
# Benchmark subgraph extraction (shared by both datasets)
# ------------------------------------------------------------------

def extract_subgraph(G, min_degree=10, max_nodes=200, seed=42):
    """
    Same recipe as data/load_dblp.py, applied uniformly to every dataset:
      1. keep nodes with degree >= min_degree
      2. take the largest connected component
      3. BFS from the highest-degree node, keep the first max_nodes
      4. relabel to 0..n-1

    Returns an UNDIRECTED graph, because describe() reports undirected
    statistics and derives the directed edge count from them.  Callers that
    need the directed graph CoSimRank actually runs on should apply
    .to_directed(), which keeps both directions of every edge.
    """
    high_deg = [v for v, d in G.degree() if d >= min_degree]
    H = G.subgraph(high_deg).copy()
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    if not comps:
        raise ValueError("no component survives the degree filter")
    largest = H.subgraph(comps[0]).copy()
    start = max(largest.degree(), key=lambda x: x[1])[0]
    bfs_nodes = list(nx.bfs_tree(largest, start).nodes())[:max_nodes]
    sub = largest.subgraph(bfs_nodes).copy()
    mapping = {old: new for new, old in enumerate(sorted(sub.nodes()))}
    sub = nx.relabel_nodes(sub, mapping)
    return sub


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  Dataset statistics  (all figures measured, none quoted)")
    print("=" * 68)

    rows = []

    # ---------- ego-Facebook ----------
    print("\n" + "=" * 68)
    print("  ego-Facebook  (SNAP, social follow graph)")
    print("=" * 68)
    FB = load_facebook()
    s = describe(FB, "ego-Facebook -- full graph")
    print_row(s); rows.append(s)

    for n_sub in [200, 500, 1000]:
        sub = extract_subgraph(FB, min_degree=10, max_nodes=n_sub)
        s = describe(sub, f"ego-Facebook -- benchmark subgraph (max_nodes={n_sub})")
        print_row(s); rows.append(s)

    # ---------- com-DBLP ----------
    print("\n" + "=" * 68)
    print("  com-DBLP  (SNAP, co-authorship graph)")
    print("=" * 68)
    DB = load_dblp_graph()
    s = describe(DB, "com-DBLP -- full graph", clustering_sample=5000)
    print_row(s); rows.append(s)

    for n_sub in [200, 500, 1000]:
        sub = extract_subgraph(DB, min_degree=10, max_nodes=n_sub)
        s = describe(sub, f"com-DBLP -- benchmark subgraph (max_nodes={n_sub})")
        print_row(s); rows.append(s)

    # ---------- comparison table ----------
    print("\n" + "=" * 68)
    print("  Comparison  (the two datasets differ where it matters)")
    print("=" * 68)
    hdr = f"  {'dataset':<44} {'n':>7} {'m_dir':>9} {'d_bar':>7} {'clust':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in rows:
        print(f"  {s['label']:<44} {s['n']:>7,} {s['m_directed']:>9,} "
              f"{s['mean_degree']:>7.2f} {s['avg_clustering']:>7.4f}")

    # ---------- CSV ----------
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, 'dataset_stats.csv')
    fields = ['label', 'n', 'm_undirected', 'm_directed', 'mean_degree',
              'median_degree', 'min_degree', 'max_degree',
              'p25', 'p50', 'p75', 'p90', 'p99',
              'avg_clustering', 'clustering_note', 'density', 'n_components']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in rows:
            w.writerow({k: s[k] for k in fields})
    print(f"\n  -> {os.path.relpath(path)}")
    print("=" * 68)

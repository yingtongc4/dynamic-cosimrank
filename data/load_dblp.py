"""
DBLP Dataset Loader
Loads the SNAP DBLP collaboration network and extracts a manageable subgraph
for benchmarking the Extended Dynamic CoSimRank framework.

Full graph: 317,080 nodes, 1,049,866 edges (undirected)
Strategy: extract the k-core subgraph (nodes with degree >= k),
          then convert to directed by keeping both edge directions.
          This gives a dense, realistic subgraph for benchmarking.
"""

import networkx as nx


def load_dblp(path="com-dblp.txt"):
    """
    Load the full DBLP undirected graph from SNAP format.
    Returns an undirected Graph.
    """
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
    return G


def extract_subgraph(G, min_degree=10, max_nodes=500, seed=42):
    """
    Extract a dense subgraph suitable for CoSimRank benchmarking.

    Strategy:
      1. Keep only nodes with degree >= min_degree (k-core idea)
      2. Take the largest connected component
      3. Sample up to max_nodes nodes (BFS from a high-degree seed node)
      4. Convert to directed (add both directions per undirected edge)

    Args:
        G         : full undirected DBLP graph
        min_degree: minimum degree threshold for node inclusion
        max_nodes : maximum number of nodes in subgraph
        seed      : random seed for reproducibility

    Returns:
        DiGraph with at most max_nodes nodes
    """
    # Filter by degree
    high_deg = [n for n, d in G.degree() if d >= min_degree]
    H = G.subgraph(high_deg).copy()

    # Take largest connected component
    components = sorted(nx.connected_components(H), key=len, reverse=True)
    if not components:
        raise ValueError("No connected component found after degree filtering.")
    largest = H.subgraph(components[0]).copy()

    # BFS from highest-degree node to get a compact neighbourhood
    start = max(largest.degree(), key=lambda x: x[1])[0]
    bfs_nodes = list(nx.bfs_tree(largest, start).nodes())[:max_nodes]
    sub = largest.subgraph(bfs_nodes).copy()

    # Re-index nodes to 0..n-1 for cleanliness
    mapping = {old: new for new, old in enumerate(sorted(sub.nodes()))}
    sub = nx.relabel_nodes(sub, mapping)

    # Convert to directed (both directions per edge)
    D = sub.to_directed()

    return D


def load_dblp_subgraph(path="com-dblp.txt", min_degree=10, max_nodes=500):
    """Convenience wrapper: load DBLP and return a directed subgraph."""
    print(f"Loading DBLP from {path} ...")
    G_full = load_dblp(path)
    print(f"  Full graph: {G_full.number_of_nodes()} nodes, "
          f"{G_full.number_of_edges()} edges")

    D = extract_subgraph(G_full, min_degree=min_degree, max_nodes=max_nodes)
    print(f"  Subgraph  : {D.number_of_nodes()} nodes, "
          f"{D.number_of_edges()} edges  (min_degree>={min_degree})")
    return D


if __name__ == "__main__":
    import os
    here = os.path.dirname(__file__)
    path = os.path.join(here, "com-dblp.txt")
    D = load_dblp_subgraph(path, min_degree=10, max_nodes=300)
    print(f"\nSample edges: {list(D.edges())[:5]}")

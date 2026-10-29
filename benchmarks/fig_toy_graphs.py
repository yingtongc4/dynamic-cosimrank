# -*- coding: utf-8 -*-
"""
fig_toy_graphs.py
=================
Draw the three worked-example graphs.

The report has 46 tables and 4 figures, and the three graphs a reader most
needs to see -- the four-node round trip, the ten-node in-edge example and the
eight-node scope example -- are described only in words, so the structure has
to be reconstructed from prose before the matrices mean anything.

One figure, three panels, each labelled with the operation the example
performs on it.

Usage:
    python benchmarks/fig_toy_graphs.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

BASE = "#2b6cb0"      # existing edges
ADD = "#c53030"       # the edge or node the example inserts
GREY = "#a0aec0"


def example_a():
    """1->2->3, then node 4 with 4->1 and 4->2."""
    G = nx.DiGraph([(1, 2), (2, 3), (4, 1), (4, 2)])
    pos = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0.5, 1)}
    new_nodes, new_edges = {4}, {(4, 1), (4, 2)}
    return G, pos, new_nodes, new_edges


def example_b():
    """Two complete DAGs on five nodes; node 11 attached by in-edges."""
    G = nx.DiGraph()
    for base in (1, 6):
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(base + i, base + j)
    G.add_edge(1, 11); G.add_edge(2, 11)
    pos = {}
    for k in range(5):
        pos[1 + k] = (k * 0.55, 1.0)
        pos[6 + k] = (k * 0.55, 0.0)
    pos[11] = (1.1, 1.85)
    return G, pos, {11}, {(1, 11), (2, 11)}


def example_c():
    """Two independent diamonds; the local insertion 1->4 is the change."""
    G = nx.DiGraph([(1, 3), (2, 3), (3, 4), (5, 7), (6, 7), (7, 8), (1, 4)])
    pos = {1: (0, 1), 2: (0, 0), 3: (0.8, 0.5), 4: (1.6, 0.5),
           5: (0, -1.6), 6: (0, -2.6), 7: (0.8, -2.1), 8: (1.6, -2.1)}
    return G, pos, set(), {(1, 4)}


def draw(ax, G, pos, new_nodes, new_edges, title, caption):
    old_e = [e for e in G.edges() if e not in new_edges]
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos, edgelist=old_e,
                           edge_color=GREY, width=1.2, arrowsize=11,
                           node_size=430, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos, edgelist=list(new_edges),
                           edge_color=ADD, width=2.2, arrowsize=14,
                           node_size=430, connectionstyle="arc3,rad=0.06")
    cols = [ADD if v in new_nodes else BASE for v in G.nodes()]
    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, node_color=cols,
                           node_size=430, linewidths=0)
    nx.draw_networkx_labels(ax=ax, G=G, pos=pos, font_color="white",
                            font_size=8.5, font_weight="bold")
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.text(0.5, -0.10, caption, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.6, color="#333333")
    ax.set_axis_off()


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    G, pos, nn, ne = example_a()
    draw(axes[0], G, pos, nn, ne, "Example A: round trip ($n=4$)",
         "insert node 4 with $4\\to1,\\,4\\to2$;\nthen delete both and contract")

    G, pos, nn, ne = example_b()
    draw(axes[1], G, pos, nn, ne, "Example B: direction decides cost ($n=11$)",
         "node 11 by in-edges is $b=1$;\nthe same two edges reversed is $b=2$")

    G, pos, nn, ne = example_c()
    draw(axes[2], G, pos, nn, ne, "Example C: scope ($n=8$)",
         "insert $1\\to4$ inside the top community;\nthe lower one is untouched")

    plt.tight_layout()
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, "fig_toy_graphs.png")
    plt.savefig(out, dpi=190, bbox_inches="tight")
    print("  -> {}".format(os.path.relpath(out)))
    for name, (G, _, _, _) in (("A", example_a()), ("B", example_b()),
                               ("C", example_c())):
        print("     Example {}: n={}, m={}".format(name, G.number_of_nodes(),
                                                   G.number_of_edges()))


if __name__ == "__main__":
    main()

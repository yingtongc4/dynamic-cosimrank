# -*- coding: utf-8 -*-
"""
fig_scope8.py -- heatmaps for the eight-node scope example.

Kept separate from worked_example_scope8.py so that the numerical result and
the figure that illustrates it are produced by clearly distinct code; the
numbers in the report come from the former.

Usage:
    python benchmarks/fig_scope8.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from worked_example_scope8 import (
    BASE_EDGES, COMM_A, COMM_B, HUB, C_EXACT,
    build_A, cosim_exact, sub, zeros, nonzero,
)
from fractions import Fraction as F

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LOCAL_EDGE = (0, 3)                                   # 1 -> 4, chosen by search
HUB_EDGES = [(HUB, 0), (HUB, 2), (HUB, 4), (HUB, 6)]  # 9 -> 1, 3, 5, 7


def to_float(M):
    return np.array([[float(v) for v in row] for row in M])


def main():
    S0 = cosim_exact(build_A(8, BASE_EDGES), C_EXACT)
    dS_local = sub(cosim_exact(build_A(8, BASE_EDGES + [LOCAL_EDGE]), C_EXACT), S0)

    S0p = zeros(9)
    for i in range(8):
        for j in range(8):
            S0p[i][j] = S0[i][j]
    S0p[8][8] = F(1)
    dS_glob = sub(cosim_exact(build_A(9, BASE_EDGES + HUB_EDGES), C_EXACT), S0p)

    Ml, Mg = to_float(dS_local), to_float(dS_glob)
    vmax = max(np.abs(Ml).max(), np.abs(Mg).max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    panels = [
        (axes[0], Ml, "LOCAL: edge {}$\\to${} inside community A".format(
            LOCAL_EDGE[0] + 1, LOCAL_EDGE[1] + 1)),
        (axes[1], Mg, "GLOBAL: hub node 9 joined to both communities"),
    ]
    for ax, M, title in panels:
        im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        n = M.shape[0]
        ax.set_xticks(range(n)); ax.set_xticklabels(range(1, n + 1), fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(range(1, n + 1), fontsize=8)
        ax.axhline(3.5, color="k", lw=1.1, ls="--")
        ax.axvline(3.5, color="k", lw=1.1, ls="--")
        for i in range(n):
            for j in range(n):
                if M[i, j]:
                    ax.text(j, i, "{:+.2f}".format(M[i, j]), ha="center",
                            va="center", fontsize=6.5)
        cells = int((M != 0).sum())
        pct = 100.0 * cells / M.size
        ax.set_title("{}\n{}/{} cells changed ({:.1f}%)".format(
            title, cells, M.size, pct), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="$\\Delta S_{ij}$")

    fig.suptitle("Scope of an update on two independent communities: a local "
                 "edge cannot leave its own block", fontsize=11)
    fig.tight_layout()

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "fig_worked_example_scope8.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  local  {}/64 cells".format(nonzero(dS_local)))
    print("  global {}/81 cells".format(nonzero(dS_glob)))
    print("  -> {}".format(os.path.relpath(path)))


if __name__ == "__main__":
    main()

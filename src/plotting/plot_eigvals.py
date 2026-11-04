"""
plot_eigvals.py
==================
Plots lambda2 (exact diagonalization, 20 qubits) and lower_bound vs p from
partials/eigvals_data.json — ggplot style, purple-pink palette matching the
light-cone heatmaps. lambda2 solid/full-opacity (the actual data), lower_bound
dashed/lower-alpha (a reference bound), same solid-vs-faded convention used
for the H2-2-vs-TM overlay plot.

USAGE:
    .butterfly/bin/python src/plot_eigvals.py
"""

import json
import pathlib

import matplotlib.pyplot as plt

DATA_PATH = pathlib.Path(__file__).parent.parent / "partials" / "eigvals_data.json"
FIGURE_PATH = pathlib.Path(__file__).parent.parent / "figs" / "eigvals_vs_p.png"

COLOR_LAMBDA2 = "#6a0dad"       # dark purple
COLOR_LOWER_BOUND = "#e377c2"  # pink
COLOR_LIMIT_POINT = "#4a0033"  # deep maroon-purple, darkest of the RdPu family

if __name__ == "__main__":
    with open(DATA_PATH) as f:
        data = json.load(f)

    p_vals = data["p_vals"]
    lambda2 = data["lambda2"]
    lower_bound = data["lower_bound"]

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(p_vals, lambda2, "-", color=COLOR_LAMBDA2, lw=2, alpha=0.35,
                 label="exact diagonlization (20 qubits)")
        ax.plot(p_vals, lower_bound, "--", color=COLOR_LOWER_BOUND, lw=2, alpha=1.0,
                 label="lower bound")
        ax.scatter([0.0], [(4 / 5) ** 4], color=COLOR_LIMIT_POINT, marker="*", s=200,
                    zorder=6,
                    label=r"$\lim_{p\to0,\,N\to\infty} \lambda = (4/5)^4$")

        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$\lambda$")
        ax.legend(fontsize=13, markerscale=1.8, handlelength=3.5,
                   labelspacing=1.2, borderpad=1.2, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)

    print(f"Figure saved → {FIGURE_PATH}")

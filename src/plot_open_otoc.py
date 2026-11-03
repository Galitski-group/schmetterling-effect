"""
plot_open_otoc.py
====================
4-panel figure from partials/open_otoc.json (N=2000 OTOC data, p=1e-5):
top row is the two full (x, t) OTOC heatmaps (C, C_unitary), bottom row is
the two site-resolved (t) traces (C_site overlaid with the gaussian
reference curve, and C_site_unitary alone) -- same RdPu/purple-pink
palette as the rest of this project's light-cone figures.

USAGE:
    .butterfly/bin/python src/plot_open_otoc.py
"""

import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = pathlib.Path(__file__).parent.parent / "partials" / "open_otoc.json"
FIGURE_PATH = pathlib.Path(__file__).parent.parent / "figs" / "open_otoc.png"

COLOR_C_SITE = "#6a0dad"     # dark purple
COLOR_GAUSSIAN = "#e377c2"   # pink
COLOR_UNITARY = "#6a0dad"

if __name__ == "__main__":
    with open(DATA_PATH) as f:
        data = json.load(f)

    x = np.array(data["x"])
    t = np.array(data["t"])
    C = np.array(data["C"])
    C_unitary = np.array(data["C_unitary"])
    C_site = np.array(data["C_site"])
    C_site_unitary = np.array(data["C_site_unitary"])
    gaussian = np.array(data["gaussian"])
    x_centered = x - data["pert_site"]  # pert_site (1000) -> 0

    LABEL_FONTSIZE = 16
    PANEL_LABEL_FONTSIZE = 18

    with plt.style.context("ggplot"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        vmax = max(C.max(), C_unitary.max())
        mesh0 = axes[0, 0].pcolormesh(t, x_centered, C_unitary, shading="nearest", cmap="RdPu", vmin=0, vmax=vmax)
        axes[0, 0].set_xlabel("t", fontsize=LABEL_FONTSIZE)
        axes[0, 0].set_ylabel("x", fontsize=LABEL_FONTSIZE)
        axes[0, 0].set_title(r"$p=0.0$")
        axes[0, 0].set_ylim(-1000, 1000)
        fig.colorbar(mesh0, ax=axes[0, 0], label="C")

        mesh1 = axes[0, 1].pcolormesh(t, x_centered, C, shading="nearest", cmap="RdPu", vmin=0, vmax=vmax)
        axes[0, 1].set_xlabel("t", fontsize=LABEL_FONTSIZE)
        axes[0, 1].set_ylabel("x", fontsize=LABEL_FONTSIZE)
        axes[0, 1].set_title(r"$p>0.0$")
        axes[0, 1].set_ylim(-1000, 1000)
        fig.colorbar(mesh1, ax=axes[0, 1], label="C")

        axes[1, 0].plot(t, C_site_unitary, "-", color=COLOR_UNITARY, lw=2, alpha=0.35, label="C_site_unitary")
        axes[1, 0].set_xlabel("t", fontsize=LABEL_FONTSIZE)
        axes[1, 0].set_ylabel("C", fontsize=LABEL_FONTSIZE)
        axes[1, 0].set_title(r"$p=0.0$")
        axes[1, 0].legend()

        axes[1, 1].plot(t, C_site, "-", color=COLOR_C_SITE, lw=2, alpha=0.35, label="Monte Carlo OTOC")
        axes[1, 1].plot(t, gaussian, "--", color=COLOR_GAUSSIAN, lw=3.5, alpha=1.0, label="Gaussian Limit")
        axes[1, 1].set_xlabel("t", fontsize=LABEL_FONTSIZE)
        axes[1, 1].set_ylabel("C", fontsize=LABEL_FONTSIZE)
        axes[1, 1].set_title(r"$p>0.0$")
        axes[1, 1].legend()

        for ax, label in zip(axes.flat, [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$", r"$\mathbf{(c)}$", r"$\mathbf{(d)}$"]):
            ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=PANEL_LABEL_FONTSIZE,
                    va="bottom", ha="right")

        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)
    print(f"Figure saved → {FIGURE_PATH}")

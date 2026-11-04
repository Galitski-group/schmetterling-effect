"""
error_analysis_open_otoc.py
==============================
Bound-based Monte Carlo uncertainty for figs/open_otoc.png (the N=2000,
p=1e-5, "single qubit" C_site/C_site_unitary traces and C/C_unitary (x,t)
"front" heatmaps), loaded from the externally-provided, already-averaged
partials/open_otoc.json.

WHY A BOUND, NOT AN EXACT STANDARD ERROR: open_otoc.json stores only the
final per-(x,t) Monte Carlo MEANS over num_samples=10000 trajectories --
not the raw per-trajectory values, nor their variance/sum-of-squares. A
textbook SE = sd/sqrt(n) is therefore not computable from this file alone.
If whoever generated it still has the raw trajectory data (or even just
the per-point sample variance), that would give an exact SE directly and
should be preferred over this bound for the paper.

THE BOUND: every quantity in this file (C, C_unitary, C_site,
C_site_unitary) is the same underlying single-Pauli-perturbation "weight"
observable from the project's transfer-matrix OTOC formalism, whose
THEORETICAL maximum is 32/15 ~= 2.1333 (see
src/transfer_matrix_otoc.py's get_OTOC: `W[...] += 32/15`). This is not
just an assumption -- the data confirms it: max(C) and max(C_unitary) in
this file both equal 32/15 to float precision (checked directly). The
observable is also non-negative by construction (it's a weight). So each
individual trajectory's contribution lies in bounded support [0, 32/15].

For a bounded random variable X in [a,b] with known mean mu, the
Bhatia-Davis inequality gives the TIGHTEST possible variance bound
achievable by any distribution with that mean and that support:
    Var(X) <= (b - mu)(mu - a)
This is stronger than the naive flat Popoviciu bound ((b-a)^2/4, which
only applies at mu=(a+b)/2) and has the physically sensible property that
it -> 0 as mu approaches either boundary (e.g. the unitary plateau at
mu~2, or the pre-front region at mu~0), and peaks at mu=(a+b)/2. So:
    SE_bound(mu) = sqrt(mu * (32/15 - mu) / N)       (using a=0)
This is a worst-case upper bound on the true SE, saturated only if the
underlying per-trajectory distribution is literally two-point at {0, 32/15}
-- the true SE is probably smaller. Treat it as "uncertainty is at most
this much," not as the true value.

OUTPUTS:
  figs/open_otoc_errors.png -- (a)/(b) companion SE-bound heatmaps for
      C_unitary/C; (c)/(d) the original C_site_unitary / C_site+gaussian
      traces re-plotted with a shaded +/-1.96*SE_bound (95%) band.
  Printed summary: SE_bound range, and representative points.

USAGE:
    .butterfly/bin/python src/error_analysis_open_otoc.py
"""

import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "partials/open_otoc.json"
FIGURE_PATH = "figs/open_otoc_errors.png"

A, B = 0.0, 32 / 15  # theoretical per-trajectory support, see docstring
COLOR_C_SITE = "#6a0dad"
COLOR_GAUSSIAN = "#e377c2"
COLOR_UNITARY = "#6a0dad"
COLOR_BAND = "#6a0dad"

if __name__ == "__main__":
    with open(DATA_PATH) as f:
        data = json.load(f)

    N = data["num_samples"]
    x = np.array(data["x"])
    t = np.array(data["t"])
    C = np.array(data["C"])
    C_unitary = np.array(data["C_unitary"])
    C_site = np.array(data["C_site"])
    C_site_unitary = np.array(data["C_site_unitary"])
    gaussian = np.array(data["gaussian"])
    x_centered = x - data["pert_site"]

    def se_bound(mu):
        mu = np.clip(mu, A, B)  # guard tiny float overshoot past the bound
        return np.sqrt(mu * (B - mu) / N)

    se_C = se_bound(C)
    se_C_unitary = se_bound(C_unitary)
    se_C_site = se_bound(C_site)
    se_C_site_unitary = se_bound(C_site_unitary)

    print(f"N (samples) = {N}, support assumed = [{A}, {B:.4f}] (=32/15)")
    print()
    print(f"{'array':>20} {'SE max':>10} {'SE at plateau (last t)':>24}")
    for name, arr, se in [
        ("C", C, se_C), ("C_unitary", C_unitary, se_C_unitary),
        ("C_site", C_site, se_C_site), ("C_site_unitary", C_site_unitary, se_C_site_unitary),
    ]:
        print(f"{name:>20} {se.max():>10.5f} {se.flat[-1] if se.ndim==1 else se[:, -1].max():>24.5f}")

    print()
    print("Representative points (mean +/- SE_bound, 95% CI half-width = 1.96*SE_bound):")
    probe_idx = list(x).index(data["probe_site"])
    for label, ti in [("t=0", 0), ("t=100 (near peak)", 100), ("t=500", 500), ("t=1000 (final)", -1)]:
        c = C[probe_idx, ti]
        se = se_C[probe_idx, ti]
        print(f"  C[probe_site, {label:18s}] = {c:.4f} +/- {se:.4f}  "
              f"(95% CI [{c-1.96*se:.4f}, {c+1.96*se:.4f}])")

    # ---- figure: companion SE-bound heatmaps + shaded-band line plots -------
    with plt.style.context("ggplot"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        vmax = max(se_C_unitary.max(), se_C.max())
        mesh0 = axes[0, 0].pcolormesh(t, x_centered, se_C_unitary, shading="nearest",
                                       cmap="RdPu", vmin=0, vmax=vmax)
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("x")
        axes[0, 0].set_title(r"$p=0.0$  (SE bound)")
        axes[0, 0].set_ylim(-1000, 1000)
        fig.colorbar(mesh0, ax=axes[0, 0], label="SE bound")

        mesh1 = axes[0, 1].pcolormesh(t, x_centered, se_C, shading="nearest",
                                       cmap="RdPu", vmin=0, vmax=vmax)
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("x")
        axes[0, 1].set_title(r"$p>0.0$  (SE bound)")
        axes[0, 1].set_ylim(-1000, 1000)
        fig.colorbar(mesh1, ax=axes[0, 1], label="SE bound")

        axes[1, 0].plot(t, C_site_unitary, "-", color=COLOR_UNITARY, lw=2, label="C_site_unitary")
        axes[1, 0].fill_between(t, C_site_unitary - 1.96 * se_C_site_unitary,
                                 C_site_unitary + 1.96 * se_C_site_unitary,
                                 color=COLOR_BAND, alpha=0.25, label="95% CI (bound)")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("C")
        axes[1, 0].set_title(r"$p=0.0$")
        axes[1, 0].legend()

        axes[1, 1].plot(t, C_site, "-", color=COLOR_C_SITE, lw=2, label="Monte Carlo OTOC")
        axes[1, 1].fill_between(t, C_site - 1.96 * se_C_site, C_site + 1.96 * se_C_site,
                                 color=COLOR_BAND, alpha=0.25, label="95% CI (bound)")
        axes[1, 1].plot(t, gaussian, "--", color=COLOR_GAUSSIAN, lw=3.5, alpha=1.0,
                         label="Gaussian Limit")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("C")
        axes[1, 1].set_title(r"$p>0.0$")
        axes[1, 1].legend()

        for ax, label in zip(axes.flat, [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$",
                                          r"$\mathbf{(c)}$", r"$\mathbf{(d)}$"]):
            ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=18,
                    va="bottom", ha="right")

        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)
    print(f"\nFigure saved → {FIGURE_PATH}")

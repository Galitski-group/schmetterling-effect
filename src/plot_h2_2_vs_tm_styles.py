"""
plot_h2_2_vs_tm_styles.py
============================
Same data and physics as plot_h2_2_results.py (real H2-2 hardware C(t) vs
exact transfer-matrix overlay, per p in P_VALUES) — recolored with the
purple-pink (RdPu-derived) palette used for the light-cone heatmaps
(lightcone_w8_noisy_vs_noiseless.py etc.), and rendered under THREE
different matplotlib style contexts (default / ggplot / seaborn) side by
side, purely to compare presentation styles on the same data. No new
circuits, no waiting on Nexus, purely local — reads
partials/h2_2_tiled_checkpoint.pkl.

COLOR CHOICE: each p gets one hue from RdPu, sampled from [0.4, 0.9] (NOT
the full [0, 1] range) — RdPu's palest end is nearly white and reads fine
as a heatmap CELL (surrounded by context) but is close to invisible as a
thin LINE/marker on a white or light-gray axes background, so it's
avoided here. Lower p gets the darker/more-purple end, higher p the
lighter/pinker end (roughly matching the heatmaps' visual intuition that
p=0 sustains stronger correlations than higher p, though this is a fixed
per-series identity color, not a live encoding of C's instantaneous
value). The transfer-matrix overlay for a given p reuses that SAME hue
(ties data and theory together by color) but drawn semi-transparent and
without markers, so hardware data (opaque, errorbars, markers) and theory
(faint solid reference line) stay visually distinct without needing a
second, disconnected color family.

USAGE:
    .butterfly/bin/python src/plot_h2_2_vs_tm_styles.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
import run_disorder_avg as R

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_checkpoint.pkl")
W = 8
SEED = 1962
P_VALUES = [0.0, 0.05]
TM_BC_TYPE = R.TM_BC_TYPE

STYLES = {
    "default": "default",
    "ggplot": "ggplot",
    "seaborn": "seaborn-v0_8-darkgrid",
}
FIGURE_PATHS = {
    name: str(R.FIGURE_DIR / f"h2_2_vs_tm_C_vs_t_{name}.png") for name in STYLES
}


def _prepare_series():
    checkpoint = R._load_partial(CHECKPOINT_PATH)
    series = {}
    for P in P_VALUES:
        L_values = sorted(L for (w, s, p, L) in checkpoint if w == W and s == SEED and p == P)
        if not L_values:
            print(f"No p={P} points found for (W={W}, seed={SEED}) — skipping.")
            continue

        raw = {P: {L: np.asarray(checkpoint[(W, SEED, P, L)]) for L in L_values}}
        stats = R.compute_C_t(raw)
        C_vals = [stats["C"][P][L] for L in L_values]
        se_vals = [stats["se"][P][L] for L in L_values]

        tf = max(L_values) + 1
        tm = R.ButterflyTransferMatrix(W, bc_type=TM_BC_TYPE)
        t_tm, C_tm = tm.compute_otoc(P, tf, pert_site=None, probe_site=None)
        C_tm_scaled = C_tm / 2

        series[P] = dict(
            plot_L_vals=[0] + L_values,
            plot_C_vals=[0.0] + C_vals,
            plot_se_vals=[0.0] + se_vals,
            t_tm=t_tm,
            C_tm_scaled=C_tm_scaled,
        )
    return series


def _colors_for(p_values):
    cmap = plt.get_cmap("RdPu")
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])  # ascending p
    colors = {}
    for rank, idx in enumerate(order):
        frac = 0.9 - 0.5 * (rank / max(1, n - 1))  # lowest p -> 0.9 (dark), highest p -> 0.4
        colors[p_values[idx]] = cmap(frac)
    return colors


def _make_figure(style_key, series, colors):
    style_name = STYLES[style_key]
    figure_path = FIGURE_PATHS[style_key]
    with plt.style.context(style_name):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for P, s in series.items():
            color = colors[P]
            ax.plot(s["t_tm"], s["C_tm_scaled"], "-", lw=2, color=color, alpha=0.55,
                     label=f"TM  N={W}  p={P}", zorder=5)
            ax.errorbar(
                s["plot_L_vals"], s["plot_C_vals"], yerr=s["plot_se_vals"],
                fmt="o--", capsize=3, color=color, label=f"H2-2  N={W}  p={P}",
            )
        ax.set_xlabel("t  (L, circuit size)")
        ax.set_ylabel(r"$C_t$")
        ax.set_title(f"Real H2-2 hardware OTOC vs exact TM — N={W}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
    print(f"Figure saved → {figure_path}")


if __name__ == "__main__":
    series = _prepare_series()
    if not series:
        print(f"Nothing to plot — no checkpointed points for any p in {P_VALUES}.")
        sys.exit(1)

    colors = _colors_for(list(series))
    for style_key in STYLES:
        _make_figure(style_key, series, colors)

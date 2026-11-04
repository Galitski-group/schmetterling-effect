"""
plot_lightcone_h2_2_tiled.py
==============================
Plots the light-cone heatmap from whatever's currently checkpointed in
partials/lightcone_h2_2_tiled_checkpoint.pkl — no new circuits submitted,
no waiting on Nexus, purely local. Points not yet harvested (still queued
on real hardware, see harvest_lightcone_h2_2_tiled_pending.py) just render
as 0.0 (plot_lightcone_heatmap's own convention for unfilled cells) rather
than blocking the plot.

USAGE:
    .butterfly/bin/python src/plot_lightcone_h2_2_tiled.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
import runs.run_disorder_avg as R

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_checkpoint.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_h2_2.png")
W = 8
SEED = 1962
P_VALUES = [0.0, 0.025, 0.05]

if __name__ == "__main__":
    checkpoint = R._load_partial(CHECKPOINT_PATH)

    points = [(w, s, p, L) for (w, s, p, L) in checkpoint if w == W and s == SEED and p in P_VALUES]
    if not points:
        print(f"No points found for (W={W}, seed={SEED}, p in {P_VALUES}) — nothing to plot.")
        sys.exit(1)

    L_values = sorted({L for (_, _, _, L) in points})
    print(f"Plotting {len(points)} point(s) across p={P_VALUES}, L={L_values} "
          f"(W={W}, seed={SEED}):")
    for p in P_VALUES:
        have = sorted(L for (_, _, pp, L) in points if pp == p)
        missing = [L for L in L_values if L not in have]
        print(f"  p={p:.2f}  have L={have}" + (f"  MISSING L={missing}" if missing else ""))

    raw_by_x: dict[int, dict[float, dict[int, np.ndarray]]] = {}
    for (w, s, p, L) in points:
        point_by_x = checkpoint[(w, s, p, L)]
        for x, vals in point_by_x.items():
            raw_by_x.setdefault(x, {}).setdefault(p, {})[L] = np.array(vals)

    stats_by_x = {x: R.compute_C_t(pd) for x, pd in raw_by_x.items()}

    with plt.style.context("seaborn-v0_8-darkgrid"):
        R.plot_lightcone_heatmap(
            stats_by_x,
            L_values=L_values,
            p_values=P_VALUES,
            stat="C",
            figure_path=FIGURE_PATH,
            cmap="RdPu",
            cone_line_color="black",
            xlabel="t  (forward time)",
            ylabel="x",
            shared_colorbar=True,
            shared_axis_labels=True,
            show_suptitle=False,
        )

"""
plot_h2_2_vs_tm_simulation.py
================================
Single-site OTOC figure: the seaborn version of plot_h2_2_vs_tm_styles.py
(H2-2 hardware C(t), N=8, pert_site=4, probe_site=0), with the theory curves
taken from partials/tm_simulation.json instead of ButterflyTransferMatrix.

The JSON stores C0 (p=0) and C (p=0.05) with no t array; list index = t. This
was checked against the previous curve, whose p=0 values it matches at
t=4, 8, 12, 16. Curves are drawn up to t=16, the last hardware point, as in
the original figure.

Reads the checkpoint directly because src/transfer_matrix_otoc.py currently
does not import, which also breaks run_disorder_avg.

USAGE:
    .butterfly/bin/python src/plot_h2_2_vs_tm_simulation.py
"""

import json
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent
CHECKPOINT_PATH = ROOT / "partials" / "h2_2_tiled_checkpoint.pkl"
TM_PATH = ROOT / "partials" / "tm_simulation.json"
FIGURE_PATH = ROOT / "figs" / "h2_2_vs_tm_simulation_C_vs_t.png"

W, SEED = 8, 1962
P_VALUES = [0.0, 0.05]
T_MAX = 16

if __name__ == "__main__":
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    with open(TM_PATH) as f:
        tm = json.load(f)
    if tm["num_qudits"] != W or tm["p"] != P_VALUES[1]:
        raise ValueError("tm_simulation.json does not match this figure's N or p")
    curves = {P_VALUES[0]: np.array(tm["C0"]), P_VALUES[1]: np.array(tm["C"])}

    cmap = plt.get_cmap("RdPu")
    colors = {P_VALUES[0]: cmap(0.9), P_VALUES[1]: cmap(0.4)}

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for P in P_VALUES:
            L_values = sorted(L for (w, s, p, L) in checkpoint if w == W and s == SEED and p == P)
            samples = [np.asarray(checkpoint[(W, SEED, P, L)], dtype=float) for L in L_values]
            C_vals = [0.0] + [v.mean() for v in samples]
            se_vals = [0.0] + [v.std(ddof=1) / np.sqrt(len(v)) for v in samples]

            t_tm = np.arange(T_MAX + 1)
            ax.plot(t_tm, curves[P][: T_MAX + 1], "-", lw=2, color=colors[P], alpha=0.55,
                    label=f"TM  N={W}  p={P}", zorder=5)
            ax.errorbar([0] + L_values, C_vals, yerr=se_vals, fmt="o--", capsize=3,
                        color=colors[P], label=f"H2-2  N={W}  p={P}")
            print(f"p={P}")
            for L, c, se in zip(L_values, C_vals[1:], se_vals[1:]):
                th = curves[P][L]
                print(f"  L={L:2d}  hardware {c:+.4f} +/- {se:.4f}   TM {th:.4f}   z={(c - th) / se:+.2f}")
        ax.set_xlabel("t  (L, circuit size)")
        ax.set_ylabel(r"$C_t$")
        ax.set_title(f"Real H2-2 hardware OTOC vs exact TM — N={W}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)
    print(f"Figure saved → {FIGURE_PATH}")

"""
plot_h2_2_normalized.py
========================
Same H2-2 hardware vs exact-TM data as plot_h2_2_p0_results.py, but as two
separate panels (one per p) with every curve normalized by its OWN maximum
so that each series' peak value is exactly 1 — QC and TM are each divided
by their own max independently, so this compares GROWTH SHAPE, not
absolute scale (a p=0.05 QC error bar sits at ~0.2 in absolute terms, but
at 1.0 here, same as p=0.0's peak — the two panels are not on a shared
absolute scale, deliberately).

Purely local — no new circuits submitted, no waiting on Nexus. See
plot_h2_2_p0_results.py's docstring for the QC-to-TM time-alignment
derivation (x=L, not L-1) and pert_site/probe_site defaults; unchanged here.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_checkpoint.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "h2_2_vs_tm_normalized.png")
W = 8
SEED = 1962
P_VALUES = [0.0, 0.05]
TM_BC_TYPE = R.TM_BC_TYPE  # "open" — matches the tiled circuits' finite open chain
QC_COLOR = "indigo"
TM_COLOR = "crimson"

if __name__ == "__main__":
    checkpoint = R._load_partial(CHECKPOINT_PATH)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(P_VALUES), figsize=(6 * len(P_VALUES), 5), squeeze=False)
    axes = axes[0]

    for ax, P in zip(axes, P_VALUES):
        L_values = sorted(L for (w, s, p, L) in checkpoint if w == W and s == SEED and p == P)
        if not L_values:
            print(f"No p={P} points found for (W={W}, seed={SEED}) — skipping panel.")
            ax.set_title(f"p = {P}  (no data)")
            continue

        raw = {P: {L: np.asarray(checkpoint[(W, SEED, P, L)]) for L in L_values}}
        stats = R.compute_C_t(raw)

        C_vals  = [stats["C"][P][L] for L in L_values]
        se_vals = [stats["se"][P][L] for L in L_values]
        n_vals  = [stats["n"][P][L] for L in L_values]

        tf = max(L_values) + 1
        tm = R.ButterflyTransferMatrix(W, bc_type=TM_BC_TYPE)
        t_tm, C_tm = tm.compute_otoc(P, tf, pert_site=None, probe_site=None)
        C_tm_scaled = C_tm / 2  # /2 matches run_sequential's own QC-vs-TM scale

        # t=0 is a trivial known point (zero backward layers => perturbation
        # can't have reached the probe yet => C_t=0 exactly), prepended for
        # normalization/plotting only — not a measurement.
        plot_L_vals  = np.array([0] + L_values)
        plot_C_vals  = np.array([0.0] + C_vals)
        plot_se_vals = np.array([0.0] + se_vals)

        qc_max = plot_C_vals.max()
        tm_max = C_tm_scaled.max()
        qc_norm = plot_C_vals / qc_max
        se_norm = plot_se_vals / qc_max
        tm_norm = C_tm_scaled / tm_max

        print(f"p={P}: QC max={qc_max:.4f} (at L={plot_L_vals[plot_C_vals.argmax()]}), "
              f"TM max={tm_max:.4f} (at t={t_tm[C_tm_scaled.argmax()]})")
        for L, C, se in zip(plot_L_vals, qc_norm, se_norm):
            print(f"  L={L:3d}  QC_norm={C:+.4f}±{se:.4f}")

        ax.plot(t_tm, tm_norm, "-", lw=2, color=TM_COLOR, label=f"TM  N={W}", zorder=5)
        ax.errorbar(
            plot_L_vals, qc_norm, yerr=se_norm,
            fmt="o--", capsize=3, color=QC_COLOR, label=f"H2-2  N={W}",
        )
        ax.axhline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("t  (backward layers L)")
        ax.set_ylabel(r"$C_t \, / \, \max(C_t)$")
        ax.set_title(f"p = {P}  (normalized to own max)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Real H2-2 hardware OTOC vs exact TM — N={W}, normalized per-series", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {FIGURE_PATH}")

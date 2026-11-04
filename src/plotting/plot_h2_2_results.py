"""
plot_h2_2_p0_results.py
========================
Plots C(t) vs L for every p in P_VALUES that has checkpointed points in the
real H2-2 tiled run (partials/h2_2_tiled_checkpoint.pkl), one series per p,
each with error bars from the disorder-realization variance and overlaid
with its own exact transfer-matrix C(t) curve (the TM's own transfer
matrix depends on p too, not just the hardware run) — no new circuits
submitted, no waiting on Nexus, purely local. Whatever L's are already
checkpointed for a given p are plotted as-is; a p whose sweep is still
in flight (e.g. its L=16 point not yet returned) just shows fewer points,
nothing here waits on it.

TM pert_site/probe_site are left at their defaults (W//2, 0) because that's
what sweep_single_shot_disorder_tiled itself defaults to and what
run_h2_2_tiled.py actually submitted (it overrides neither) — see
ButterflyTransferMatrix.compute_otoc's own matching defaults.

QC-to-TM time alignment: unlike run_sequential's plot_tm_overlay (which
plots QC at L-1, calibrated for a different sweep/circuit path), THIS
tiled circuit builds a full echo with L_back defaulting to L
(_build_otoc_circuit_impl, __main__.py) — L backward layers after the
perturbation, replaying the L forward layers in reverse. ButterflyTransferMatrix
.compute_otoc's C[0] is the weight immediately after seeding the
perturbation with ZERO evolution steps applied, and C[k] is the weight
after k backward-brickwall layers — and its per-t even/odd pairing parity
(t=1 odd, t=2 even, ...) exactly matches this circuit's reversed-layer
parity sequence for any even L (verified: forward layer L-1 is always odd
for even L, so the reversed/backward sequence starts odd, same as TM's
t=1). So a hardware point built with L backward layers corresponds to
TM's C[L], not C[L-1] — QC is plotted directly at x=L here, no shift.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_checkpoint.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "h2_2_vs_tm_C_vs_t.png")
W = 8
SEED = 1962
P_VALUES = [0.0, 0.05]
TM_BC_TYPE = R.TM_BC_TYPE  # "open" — matches the tiled circuits' finite open chain
P_COLORS = {0.0: "indigo", 0.05: "darkorange"}
TM_COLORS = {0.0: "crimson", 0.05: "forestgreen"}

if __name__ == "__main__":
    checkpoint = R._load_partial(CHECKPOINT_PATH)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))

    any_series = False
    for P in P_VALUES:
        L_values = sorted(L for (w, s, p, L) in checkpoint if w == W and s == SEED and p == P)
        if not L_values:
            print(f"No p={P} points found for (W={W}, seed={SEED}) — skipping.")
            continue
        any_series = True

        raw = {P: {L: np.asarray(checkpoint[(W, SEED, P, L)]) for L in L_values}}
        stats = R.compute_C_t(raw)

        C_vals  = [stats["C"][P][L] for L in L_values]
        se_vals = [stats["se"][P][L] for L in L_values]
        n_vals  = [stats["n"][P][L] for L in L_values]

        print(f"H2-2 hardware, p={P}  (W={W}, seed={SEED}):")
        for L, C, se, n in zip(L_values, C_vals, se_vals, n_vals):
            print(f"  L={L:3d}  C_t={C:+.4f}  se={se:.4f}  n={n}")

        tf = max(L_values) + 1
        tm = R.ButterflyTransferMatrix(W, bc_type=TM_BC_TYPE)
        t_tm, C_tm = tm.compute_otoc(P, tf, pert_site=None, probe_site=None)
        C_tm_scaled = C_tm / 2  # /2 matches run_sequential's own QC-vs-TM scale

        print(f"QC vs TM at matching L (x=L, not L-1), p={P}:")
        for L, C, se in zip(L_values, C_vals, se_vals):
            tm_val = C_tm_scaled[L]
            z = (C - tm_val) / se if se > 0 else float("inf")
            print(f"  L={L:3d}  QC={C:+.4f}±{se:.4f}   TM={tm_val:+.4f}   z={z:+.2f}")
        print()

        # t=0 is a trivial known point, not a measurement: zero backward
        # layers means the perturbation can't have reached the probe yet,
        # so C_t=0 exactly — prepended for the plot only (not the printed
        # table above, which reflects only what was actually recorded).
        plot_L_vals  = [0] + L_values
        plot_C_vals  = [0.0] + C_vals
        plot_se_vals = [0.0] + se_vals

        ax.plot(t_tm, C_tm_scaled, "-", lw=2, color=TM_COLORS.get(P, "gray"),
                 label=f"TM  N={W}  p={P}", zorder=5)
        ax.errorbar(
            plot_L_vals, plot_C_vals, yerr=plot_se_vals,
            fmt="o--", capsize=3, color=P_COLORS.get(P, "black"),
            label=f"H2-2  N={W}  p={P}",
        )

    if not any_series:
        print(f"Nothing to plot — no checkpointed points for any p in {P_VALUES}.")
        sys.exit(1)

    ax.set_xlabel("t  (backward layers L)")
    ax.set_ylabel(r"$C_t$")
    ax.set_title(f"Real H2-2 hardware OTOC vs exact TM — N={W}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {FIGURE_PATH}")

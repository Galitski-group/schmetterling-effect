"""
run_lightcone_probe_all_w18_p05_noiseless.py
=============================================
Adds p=0.05 to the W=18 noiseless light-cone comparison (previously
p=0.0, p=0.01 — see run_lightcone_probe_all_w18_noiseless.py). Same W,
L_values, n_realizations, noiseless config; p=0.05 only.

Why a separate script instead of re-running all three p together:
sweep_lightcone_probe_all had no checkpointing when the p=0.0/p=0.01 run
was launched, so that run's raw per-realization data only exists baked
into its already-rendered PNG, not as reusable arrays — re-running all
three from scratch would redo ~6 hours of already-finished work for no
benefit. checkpoint_path is now wired up (added directly to
sweep_lightcone_probe_all), so THIS run's p=0.05 data — and anything
after it — will be truly incremental from here on.

Output: a standalone 1-panel p=0.05 heatmap (this script does not attempt
to merge with the existing 2-panel p=0.0/p=0.01 PNG — that combination is
done separately, once both new p=0.05 runs, noiseless and noisy, finish).

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path —
nothing here can hit real Quantinuum hardware.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runs.run_disorder_avg as R

R.NOISE_P1Q = None
R.NOISE_P2Q = None
R.NOISE_PM = None
R.NOISE_T1 = None
R.NOISE_T2 = None

W = 18
L_VALUES = [1, 2, 4, 6, 8, 10]
P_VALUES = [0.05]
N_REALIZATIONS = 250
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w18_p05_noiseless.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w18_noiseless_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES}  "
          f"n_realizations={N_REALIZATIONS}  noiseless=True  (local simulation only)")

    stats_by_x = R.sweep_lightcone_probe_all(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_realizations=N_REALIZATIONS,
        checkpoint_path=CHECKPOINT_PATH,
        # device_name intentionally omitted -> local AerBackend, never hardware
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
    )

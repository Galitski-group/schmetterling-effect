"""
run_lightcone_probe_all_w8_hw_noiseless.py
===========================================
Local-simulation-ONLY, NOISELESS light-cone sweep at the ACTUAL real-
hardware target size: W=8, n_realizations=240 (both exactly match what
was used throughout the earlier H2-2 tiled OTOC hardware runs) — a
rehearsal for eventually submitting this same "measure every site" sweep
to real Quantinuum H2-2. L_values EVERY integer step 1..18 (see how far
the p=0.05 decay continues at longer times). p_values now include 0.025
too — a midpoint between the existing 0.0 and 0.05 points, added to see a
clearer/smoother degradation gradient — so this run only computes the 18
new (p=0.025, L) points; everything at p=0.0/0.01/0.05 is already
checkpointed.

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path —
nothing here can hit real Quantinuum hardware.

Noiseless: all five NOISE_* module constants are explicitly set to None
before building the backend (module defaults are noisy — see the sibling
run_lightcone_probe_all_w8_hw_noisy.py, which leaves them at the realistic
H2-spec-derived defaults instead).

checkpoint_path is wired up from the start this time (see
sweep_lightcone_probe_all's own docstring) — any future p addition here
is free, not a re-run.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_disorder_avg as R

R.NOISE_P1Q = None
R.NOISE_P2Q = None
R.NOISE_PM = None
R.NOISE_T1 = None
R.NOISE_T2 = None

W = 8
L_VALUES = list(range(1, 19))
P_VALUES_SWEEP = [0.0, 0.01, 0.025, 0.05]  # computed/checkpointed (0.01 kept, already paid for)
P_VALUES_PLOT = [0.0, 0.025, 0.05]  # the three actually requested for this figure
N_REALIZATIONS = 240
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w8_hw_noiseless.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w8_hw_noiseless_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES_SWEEP}  "
          f"n_realizations={N_REALIZATIONS}  noiseless=True  (local simulation only)")

    stats_by_x = R.sweep_lightcone_probe_all(
        W=W,
        p_values=P_VALUES_SWEEP,
        L_values=L_VALUES,
        n_realizations=N_REALIZATIONS,
        checkpoint_path=CHECKPOINT_PATH,
        # device_name intentionally omitted -> local AerBackend, never hardware
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES_PLOT,
        stat="C",
        figure_path=FIGURE_PATH,
    )

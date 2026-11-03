"""
run_lightcone_probe_all_w8_hw_noisy.py
=======================================
Local-simulation-ONLY light-cone sweep with REALISTIC (H2-spec-derived)
noise, at the ACTUAL real-hardware target size: W=8, n_realizations=240
(both exactly match what was used throughout the earlier H2-2 tiled OTOC
hardware runs) — a rehearsal for eventually submitting this same "measure
every site" sweep to real Quantinuum H2-2. Same p values (0.0, 0.01, 0.05)
as the sibling run_lightcone_probe_all_w8_hw_noiseless.py and the earlier
W=18 comparison, with L_values now EVERY integer step 1..10 (denser than
the original 1,2,4,6,8,10) — the 6 original points are already
checkpointed, so this run only computes the 4 new ones (L=3,5,7,9).

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path — noise
here comes from AerBackend's own noise model (built from the NOISE_*
constants below), NOT real Quantinuum hardware; nothing in this script can
hit real hardware.

Noise values are the module's own H2-spec-derived defaults (same as
run_lightcone_probe_all_w18_noisy.py); set explicitly here so this
script's config is self-contained.

checkpoint_path is wired up from the start (see
sweep_lightcone_probe_all's own docstring) — any future p addition here
is free, not a re-run.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_disorder_avg as R

R.NOISE_P1Q = 7.3e-05
R.NOISE_P2Q = 1.29e-03
R.NOISE_PM = 1.35e-03
R.NOISE_T1 = None
R.NOISE_T2 = None

W = 8
L_VALUES = list(range(1, 11))
P_VALUES = [0.0, 0.01, 0.05]
N_REALIZATIONS = 240
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w8_hw_noisy.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w8_hw_noisy_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES}  "
          f"n_realizations={N_REALIZATIONS}  "
          f"noise: p1q={R.NOISE_P1Q} p2q={R.NOISE_P2Q} pm={R.NOISE_PM} "
          f"T1={R.NOISE_T1} T2={R.NOISE_T2}  (local simulation only)")

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

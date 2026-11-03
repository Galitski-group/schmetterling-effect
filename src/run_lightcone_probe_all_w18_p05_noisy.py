"""
run_lightcone_probe_all_w18_p05_noisy.py
=========================================
Adds p=0.05 to the W=18 realistic-noise light-cone comparison (previously
p=0.0, p=0.01 — see run_lightcone_probe_all_w18_noisy.py). Same W,
L_values, n_realizations, noise config; p=0.05 only.

See run_lightcone_probe_all_w18_p05_noiseless.py's docstring for why this
is a separate incremental run rather than redoing all three p together,
and why checkpoint_path is now wired up going forward.

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path — noise
here comes from AerBackend's own noise model, NOT real Quantinuum
hardware; nothing in this script can hit real hardware.

Noise values match run_lightcone_probe_all_w18_noisy.py exactly (the
module's own H2-spec-derived NOISE_P1Q/NOISE_P2Q/NOISE_PM defaults).
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

W = 18
L_VALUES = [1, 2, 4, 6, 8, 10]
P_VALUES = [0.05]
N_REALIZATIONS = 250
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w18_p05_noisy.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w18_noisy_checkpoint.pkl")

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

"""
run_lightcone_probe_all_w8_hw_anchored_noisy.py
=================================================
Same as run_lightcone_probe_all_w8_hw_noisy.py (W=8, n_realizations=240,
p=0.0/0.01/0.05, L=1..10, realistic H2-spec noise, local-simulation-ONLY)
but with anchor_parity_at_perturbation="even" — see
run_lightcone_probe_all_w8_hw_anchored_noiseless.py's docstring for the
full explanation (consistent-but-still-present left/right bias, not true
symmetry) and why this uses its own separate checkpoint file.

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path — noise
here comes from AerBackend's own noise model, NOT real Quantinuum
hardware; nothing in this script can hit real hardware.

Noise values match the module's own H2-spec-derived defaults (same as the
other *_noisy.py scripts in this project).
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
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w8_hw_anchored_noisy.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w8_hw_anchored_noisy_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES}  "
          f"n_realizations={N_REALIZATIONS}  anchor_parity_at_perturbation='even'  "
          f"noise: p1q={R.NOISE_P1Q} p2q={R.NOISE_P2Q} pm={R.NOISE_PM} "
          f"T1={R.NOISE_T1} T2={R.NOISE_T2}  (local simulation only)")

    stats_by_x = R.sweep_lightcone_probe_all(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_realizations=N_REALIZATIONS,
        checkpoint_path=CHECKPOINT_PATH,
        anchor_parity_at_perturbation="even",
        # device_name intentionally omitted -> local AerBackend, never hardware
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
    )

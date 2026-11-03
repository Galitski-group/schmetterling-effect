"""
run_lightcone_probe_all_w18_noiseless.py
=========================================
Local-simulation-ONLY, NOISELESS light-cone sweep using the "measure every
site" circuit (sweep_lightcone_probe_all / build_*_lightcone_probe_all_circuit):
W=18, 250 disorder realizations, L=1,2,4,6,8,10, p in {0.0, 0.01}.

W=18 (not 8) is chosen so the causal cone has real room to grow
(x_max=min(9,8)=8) without the per-shot cost blowup Aer's statevector
simulator hits around W~20-22 (calibrated earlier in this session).
L=0 is deliberately excluded (degenerate point, see
run_lightcone_probe_all_local.py's docstring) — L=1 is the time reference.

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path —
nothing here can hit real Quantinuum hardware.

Noiseless: all five NOISE_* module constants are explicitly set to None
before building the backend (module defaults are noisy — see the sibling
run_lightcone_probe_all_w18_noisy.py, which leaves them at the realistic
H2-spec-derived defaults instead).

Calibrated cost (20-shot probes, this session): summed ms/shot across
L=1,2,4,6,8,10 at W=18 noiseless ~= 224 ms. At N_SHOTS=100 (module
default), 2 circuits/point, 250 realizations, 2 p values: estimated total
~= 6.2 hours. Runs in the background — this script prints progress line by
line so tmp/*.log can be tailed to check on it.
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

W = 18
L_VALUES = [1, 2, 4, 6, 8, 10]
P_VALUES = [0.0, 0.01]
N_REALIZATIONS = 250
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w18_noiseless.png")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES}  "
          f"n_realizations={N_REALIZATIONS}  noiseless=True  (local simulation only)")

    stats_by_x = R.sweep_lightcone_probe_all(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_realizations=N_REALIZATIONS,
        # device_name intentionally omitted -> local AerBackend, never hardware
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
    )

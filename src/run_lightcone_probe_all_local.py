"""
run_lightcone_probe_all_local.py
=================================
Local-simulation-ONLY light-cone sweep using the "measure every site"
circuit (sweep_lightcone_probe_all / build_*_lightcone_probe_all_circuit):
W=8, 250 disorder realizations, NOISELESS, L=1,2,4,6,8,10, p=0.0 only.

L=0 is deliberately NOT included: it's a degenerate point (perturbation
applied with zero subsequent evolution — x=0 reads a trivial deterministic
2.0, every other x reads exactly 0), useful once to confirm the cone's tip
behaves as expected but not informative to keep plotting. L=1 is the
earliest non-degenerate point and serves as the new time reference instead.
x=0 (pert_site) is still fully measured by this circuit at every L (unlike
the single-probe circuit, where probe_site != pert_site was a hard
constraint) — see sweep_lightcone_probe_all's own comment on why x=0 is
recorded, and plot_lightcone_heatmap's on how it renders real x=0 data
instead of forcing it to the old "never probed" convention.

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path —
nothing here can hit real Quantinuum hardware.

Noiseless: all five NOISE_* module constants are explicitly set to None
before building the backend (module defaults are noisy — see
run_disorder_avg.py's own NOISE_P1Q/NOISE_P2Q/NOISE_PM/NOISE_T1/NOISE_T2).
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
L_VALUES = [1, 2, 4, 6, 8, 10]
P_VALUES = [0.0]
N_REALIZATIONS = 250
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_noiseless_p0.png")

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

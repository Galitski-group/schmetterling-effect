"""
run_lightcone_probe_all_w18_noisy.py
=====================================
Local-simulation-ONLY light-cone sweep with REALISTIC (H2-spec-derived)
noise, using the "measure every site" circuit (sweep_lightcone_probe_all /
build_*_lightcone_probe_all_circuit): W=18, 250 disorder realizations,
L=1,2,4,6,8,10, p in {0.0, 0.01} — same config as the sibling
run_lightcone_probe_all_w18_noiseless.py, but with noise left ON.

SAFETY: device_name is never passed to sweep_lightcone_probe_all below, so
_build_backend always takes its local-simulation (AerBackend) path — noise
here comes from AerBackend's own noise model (built from the NOISE_*
constants below), NOT real Quantinuum hardware; nothing in this script can
hit real hardware.

Noise values are the module's own defaults (run_disorder_avg.py's
NOISE_P1Q/NOISE_P2Q/NOISE_PM), derived from the H2 spec sheet referenced
in that module (single/two-qubit gate depolarizing rates + readout
bitflip probability); set explicitly here anyway so this script's config
is self-contained and doesn't silently drift if the module defaults ever
change. T1/T2 are left None (not given by that spec sheet).

Calibrated cost (20-shot probes, this session): summed ms/shot across
L=1,2,4,6,8,10 at W=18 with this noise config ~= 474 ms (~2.1x the
noiseless cost). At N_SHOTS=100 (module default), 2 circuits/point, 250
realizations, 2 p values: estimated total ~= 13.2 hours.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runs.run_disorder_avg as R

R.NOISE_P1Q = 7.3e-05
R.NOISE_P2Q = 1.29e-03
R.NOISE_PM = 1.35e-03
R.NOISE_T1 = None
R.NOISE_T2 = None

W = 18
L_VALUES = [1, 2, 4, 6, 8, 10]
P_VALUES = [0.0, 0.01]
N_REALIZATIONS = 250
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w18_noisy.png")

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
        # device_name intentionally omitted -> local AerBackend, never hardware
    )

    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
    )

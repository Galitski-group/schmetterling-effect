"""
run_lightcone_probe_all_w8_hw_anchored_noiseless.py
=====================================================
Same as run_lightcone_probe_all_w8_hw_noiseless.py (W=8, n_realizations=240,
p=0.0/0.01/0.05, L=1..10, noiseless, local-simulation-ONLY) but with
anchor_parity_at_perturbation="even": forces the layer immediately adjacent
to the perturbation to always be even-parity regardless of L (previously
it flipped between even/odd depending on whether L was odd or even),
making the light-cone's left/right bias CONSISTENT across every L instead
of alternating unpredictably. This does NOT produce true x<->-x symmetry
(impossible for a single perturbed qubit in a strict nearest-neighbour
brickwall — it can only bond with one neighbour per layer, so one
direction always gets a one-layer head start) — the cone will visibly be
offset to one side, but that offset should now be stable/centered rather
than erratic. See build_brickwall_lightcone_probe_all_circuit's own
docstring for the full derivation, and sweep_lightcone_probe_all's for why
a SEPARATE checkpoint file is required here (the checkpoint key doesn't
encode this flag, so mixing with the un-anchored run's checkpoint would
silently combine data from two different circuit structures).

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

W = 8
L_VALUES = list(range(1, 11))
P_VALUES = [0.0, 0.01, 0.05]
N_REALIZATIONS = 240
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w8_hw_anchored_noiseless.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w8_hw_anchored_noiseless_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  L_values={L_VALUES}  p_values={P_VALUES}  "
          f"n_realizations={N_REALIZATIONS}  noiseless=True  anchor_parity_at_perturbation='even'  "
          f"(local simulation only)")

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

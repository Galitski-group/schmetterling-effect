"""
run_lightcone_probe_all_w8_tiled_local_noisy.py
=================================================
Local-simulation validation of the TILED light-cone sweep
(sweep_lightcone_probe_all_tiled) WITH REALISTIC (H2-spec-derived) noise —
same as run_lightcone_probe_all_w8_tiled_local.py but noisy instead of
noiseless, and extended to L=1..10 (that sibling script covers L=1..8).

n_qubits_budget=16 (width=2, 16 qubits/circuit), not 48 (width=6) — see
run_lightcone_probe_all_w8_tiled_local.py's own docstring for why (Aer's
40-qubit default cap, and 2^48 amplitudes being infeasible regardless；
classical bits, not qubits, bind chunk size here, so width=2 reproduces
the real 48-qubit plan's ~7-realizations/chunk exactly).

n_realizations=180, p=0.0/0.025/0.05, anchor_parity_at_perturbation="odd"
— matches run_lightcone_h2_2_tiled.py's current config, for a local
dry-run/correctness check (this time WITH noise layered on, as a closer
preview of real hardware output) before spending real HQC budget on it.

SAFETY: device_name is never passed to sweep_lightcone_probe_all_tiled
below, so _build_backend always takes its local-simulation (AerBackend)
path — noise here comes from AerBackend's own noise model, NOT real
Quantinuum hardware; nothing in this script can hit real hardware.
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
N_QUBITS_BUDGET = 16  # width=2 -- see module docstring for why not 48
L_VALUES = list(range(1, 11))
P_VALUES = [0.0, 0.025, 0.05]
N_REALIZATIONS = 180
ANCHOR_PARITY_AT_PERTURBATION = "odd"
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w8_tiled_local_noisy.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w8_tiled_local_noisy_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  n_qubits_budget={N_QUBITS_BUDGET} (width={N_QUBITS_BUDGET//W})  "
          f"L_values={L_VALUES}  p_values={P_VALUES}  n_realizations={N_REALIZATIONS}  "
          f"anchor_parity_at_perturbation={ANCHOR_PARITY_AT_PERTURBATION!r}  "
          f"noise: p1q={R.NOISE_P1Q} p2q={R.NOISE_P2Q} pm={R.NOISE_PM}  "
          f"TILED  (local simulation only)")

    stats_by_x = R.sweep_lightcone_probe_all_tiled(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_qubits_budget=N_QUBITS_BUDGET,
        n_realizations=N_REALIZATIONS,
        anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
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

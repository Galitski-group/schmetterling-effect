"""
run_lightcone_probe_all_w8_tiled_local.py
==========================================
Local-simulation-ONLY validation of the TILED light-cone sweep
(sweep_lightcone_probe_all_tiled) — exercises the exact same tiling/
chunking code path intended for the real H2-2 submission
(run_lightcone_h2_2_tiled.py), but at n_qubits_budget=16 (width=2, 16
qubits/circuit) instead of 48 (width=6, 48 qubits/circuit).

Why width=2 here, not 6: AerBackend defaults to a hard 40-qubit cap
(pytket's MaxNQubitsPredicate(40)) and 6*W=48 exceeds it — infeasible to
simulate locally as a plain statevector anyway (2^48 amplitudes). Width=2
(16 qubits) stays comfortably under that cap. Classical bits, not qubits,
are what bind chunk size in this circuit (see
build_parallel_lightcone_probe_all_circuit's docstring — each realization
costs W bits, not 1), and that constraint doesn't depend on width, so
chunk_size (~7 realizations/chunk here) matches the real 48-qubit plan
exactly — this genuinely validates the same chunking behaviour, just
spread over more depth-rows (4 here vs 2 at width=6) to fit the same
7-per-chunk into fewer qubits per row.

n_realizations=180, L=1..8, p=0.0/0.01/0.05, anchor_parity_at_perturbation=
"odd" — matches run_lightcone_h2_2_tiled.py's current config exactly (as
of the last edit to that script), for a local dry-run/correctness check
before spending real HQC budget on it.

SAFETY: device_name is never passed to sweep_lightcone_probe_all_tiled
below, so _build_backend always takes its local-simulation (AerBackend)
path — nothing here can hit real Quantinuum hardware.
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
N_QUBITS_BUDGET = 16  # width=2 -- see module docstring for why not 48
L_VALUES = list(range(1, 9))
P_VALUES = [0.0, 0.01, 0.05]
N_REALIZATIONS = 180
ANCHOR_PARITY_AT_PERTURBATION = "odd"
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_w8_tiled_local.png")
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_probe_all_w8_tiled_local_checkpoint.pkl")

if __name__ == "__main__":
    print(f"W={W}  pert_site={W//2}  n_qubits_budget={N_QUBITS_BUDGET} (width={N_QUBITS_BUDGET//W})  "
          f"L_values={L_VALUES}  p_values={P_VALUES}  n_realizations={N_REALIZATIONS}  "
          f"anchor_parity_at_perturbation={ANCHOR_PARITY_AT_PERTURBATION!r}  "
          f"noiseless=True  TILED  (local simulation only)")

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

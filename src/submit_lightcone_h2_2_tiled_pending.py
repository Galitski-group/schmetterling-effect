"""
submit_lightcone_h2_2_tiled_pending.py
========================================
Submit-and-forget version of run_lightcone_h2_2_tiled.py: submits every
remaining (p, L) point's compile+execute jobs to real Quantinuum hardware
WITHOUT waiting for any of them to finish, then exits. Built after directly
observing a job sit QUEUED for 3 days before being auto-cancelled by
Quantinuum — waiting sequentially for each point before submitting the next
means total wall-clock time is the SUM of every point's queue time; this
way every job queues in parallel from the start, and results get collected
later via harvest_lightcone_h2_2_tiled_pending.py (run that periodically,
e.g. once a day, to pull in whatever's finished).

Same config as run_lightcone_h2_2_tiled.py (W=8, n_qubits_budget=48,
n_realizations=180, L=1..8, anchor="odd", BASE_SEED=1962) and the SAME
CHECKPOINT_PATH/RAW_RESULTS_PATH, so points already completed via the old
sequential script (p=0.05, L=1/2/3) are correctly skipped here too.
NEXUS_PROJECT is "schmetterling-effect-test" (NOT "schmetterling-effect-
light-cone", despite what run_lightcone_h2_2_tiled.py's own file currently
says) — matching the project those 3 already-completed points actually
live in (confirmed via their saved job_info), so this run doesn't fragment
the sweep across two different Nexus projects.

SAFETY: device_name="H2-1" is passed explicitly below (matching what the
already-completed points actually ran on — confirmed via job_info; NOT
"H2-2" despite this project's earlier naming). Submits REAL jobs to REAL
hardware and WILL be billed — this is not a dry run, there is no --confirm
gate here (unlike run_lightcone_h2_2_tiled.py) because submission itself
is cheap/fast (only compile is waited on) and the expensive part (execute)
was already going to happen regardless of which script triggers it. Review
the printed remaining-points list before running if you want a safety
check — it prints what it's about to submit before each one.

USAGE:
    .butterfly/bin/python src/submit_lightcone_h2_2_tiled_pending.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_disorder_avg as R

W = 8
N_QUBITS_BUDGET = 48
N_REALIZATIONS = 180
P_VALUES = [0.025]
L_VALUES = list(range(1, 9))
BASE_SEED = 1962
TILED_MAX_DEPTH = 60
TILED_MAX_CLASSICAL_BITS = 64
TILED_MAX_SU4_PER_CHUNK = 1500
ANCHOR_PARITY_AT_PERTURBATION = "odd"

DEVICE_NAME = "H2-1"
R.NEXUS_PROJECT = "schmetterling-effect-test"
R.OPTIMISATION_LEVEL = 0
R.PERT_OP = "X"

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_checkpoint.pkl")
RAW_RESULTS_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_raw_results.pkl")
PENDING_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_pending_jobs.pkl")

if __name__ == "__main__":
    R.submit_lightcone_probe_all_tiled_pending(
        device_name=DEVICE_NAME,
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_qubits_budget=N_QUBITS_BUDGET,
        n_realizations=N_REALIZATIONS,
        base_seed=BASE_SEED,
        optimisation_level=R.OPTIMISATION_LEVEL,
        use_batch=False,
        anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
        max_depth=TILED_MAX_DEPTH,
        max_classical_bits=TILED_MAX_CLASSICAL_BITS,
        max_su4_per_chunk=TILED_MAX_SU4_PER_CHUNK,
        checkpoint_path=CHECKPOINT_PATH,
        pending_path=PENDING_PATH,
    )

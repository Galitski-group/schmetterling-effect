"""
harvest_lightcone_h2_2_tiled_pending.py
=========================================
Checks every job submitted by submit_lightcone_h2_2_tiled_pending.py and
pulls in results for any that have finished (COMPLETED), flags any that
died (CANCELLED/ERROR — these need resubmission, this script does NOT do
that automatically), and leaves anything still queued/running alone.

Safe to run as often as you like (e.g. once a day, or whenever you think
to check) — read-only against jobs that aren't in a terminal state yet,
only writes to CHECKPOINT_PATH/RAW_RESULTS_PATH/PENDING_PATH for jobs that
actually finished (or died) since the last run.

USAGE:
    .butterfly/bin/python src/harvest_lightcone_h2_2_tiled_pending.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_disorder_avg as R

CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_checkpoint.pkl")
RAW_RESULTS_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_raw_results.pkl")
PENDING_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_pending_jobs.pkl")

if __name__ == "__main__":
    R.harvest_lightcone_probe_all_tiled_pending(
        checkpoint_path=CHECKPOINT_PATH,
        raw_results_path=RAW_RESULTS_PATH,
        pending_path=PENDING_PATH,
    )

#!/bin/bash
# lightcone_w24_cpu.slurm
# =========================
# CPU array job: one task per (p, L, realization-chunk) triple, read from
# slurm/tasks_w24.tsv (generate it first with
# src/lightcone_cluster_plan.py). Each task simulates its own chunk of
# realizations, one dedicated W-qubit circuit per realization (NOT tiled
# — see lightcone_cluster_worker.py's docstring for why tiling would be
# catastrophic here), parallelized across realizations via multiprocessing
# within the node.
#
# EDIT BEFORE SUBMITTING:
#   - --account   : YOUR_ACCOUNT_HERE (Zaratan allocation name)
#   - VENV_PATH   : path to this project's Python environment on Zaratan
#                   (this repo uses .butterfly/ locally; set up the
#                   equivalent on-cluster and point at it here)
#   - OUT_DIR     : shared scratch path worker outputs get written to —
#                   MUST be visible from every node (Zaratan's /scratch or
#                   equivalent Lustre/NFS mount, not local node disk)
#   - --time      : sized for the WORST-CASE chunk (largest L) — run a
#                   small test array first (see lightcone_cluster_plan.py's
#                   own docstring) to calibrate this on real hardware
#                   before trusting the placeholder below
#
# USAGE (after generating slurm/tasks_w24.tsv):
#   sbatch --array=0-<N-1>%50 --export=ALL,MODE=noiseless slurm/lightcone_w24_cpu.slurm
#   sbatch --array=0-<N-1>%50 --export=ALL,MODE=noisy     slurm/lightcone_w24_cpu.slurm

#SBATCH --job-name=lightcone_w24
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/schmetterling-effect/.butterfly}"
OUT_DIR="${OUT_DIR:-/scratch/zt1/project/YOUR_ACCOUNT_HERE/lightcone_w24}"
REPO_DIR="${REPO_DIR:-$HOME/schmetterling-effect}"
TASK_TABLE="$REPO_DIR/slurm/tasks_w24.tsv"

if [[ -z "${MODE:-}" ]]; then
    echo "MODE not set — submit with --export=ALL,MODE=noisy or MODE=noiseless" >&2
    exit 1
fi

mkdir -p "$REPO_DIR/slurm/logs" "$OUT_DIR"
source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"

# SLURM_ARRAY_TASK_ID is 0-indexed here (matches --array=0-N-1 above);
# tasks_w24.tsv is also 0-indexed, one row per task.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_TABLE")
P=$(echo "$LINE" | cut -f1)
L=$(echo "$LINE" | cut -f2)
START=$(echo "$LINE" | cut -f3)
COUNT=$(echo "$LINE" | cut -f4)

echo "task $SLURM_ARRAY_TASK_ID -> p=$P L=$L start=$START count=$COUNT mode=$MODE"

python -u src/lightcone_cluster_worker.py \
    --W 24 --p "$P" --L "$L" \
    --realization-start "$START" --realization-count "$COUNT" \
    --n-realizations-total 2000 \
    --mode "$MODE" --base-seed 1962 --anchor odd \
    --out-dir "$OUT_DIR" \
    --n-workers "${SLURM_CPUS_PER_TASK:-32}" --threads-per-worker 1

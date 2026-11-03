#!/bin/bash
# lightcone_w24_gpu.slurm
# =========================
# GPU variant of lightcone_w24_cpu.slurm — same task table, same worker
# script, --gpu passed through instead of CPU multiprocessing (the worker
# forces --n-workers=1 under --gpu: independent CPU processes sharing one
# GPU would just serialize against each other, so there's no benefit to
# more than one worker per GPU).
#
# NOT VALIDATED against real GPU hardware (written/tested on a machine
# with no GPU) — the device="GPU" path is real qiskit-aer API (see
# lightcone_cluster_worker.py's docstring), but confirm it actually engages
# the GPU (nvidia-smi during a run, or check the worker's stderr for the
# "GPU requested but unavailable" fallback warning) before trusting a full
# array submission to it. Also worth A/B testing against the CPU variant
# at this qubit count (W=24, ~268MB statevector) — GPU dispatch overhead
# may not pay off until W is meaningfully larger; don't assume GPU is
# faster here without measuring.
#
# EDIT BEFORE SUBMITTING: same placeholders as lightcone_w24_cpu.slurm
# (--account, VENV_PATH, OUT_DIR), plus verify --partition/--gres against
# `sinfo`/`scontrol show partition gpu` on Zaratan — naming may differ
# from the guess below.
#
# USAGE:
#   sbatch --array=0-<N-1>%20 --export=ALL,MODE=noiseless slurm/lightcone_w24_gpu.slurm
#   sbatch --array=0-<N-1>%20 --export=ALL,MODE=noisy     slurm/lightcone_w24_gpu.slurm

#SBATCH --job-name=lightcone_w24_gpu
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_TABLE")
P=$(echo "$LINE" | cut -f1)
L=$(echo "$LINE" | cut -f2)
START=$(echo "$LINE" | cut -f3)
COUNT=$(echo "$LINE" | cut -f4)

echo "task $SLURM_ARRAY_TASK_ID -> p=$P L=$L start=$START count=$COUNT mode=$MODE (GPU)"
nvidia-smi -L || echo "nvidia-smi not found / no GPU visible — worker will fall back to CPU"

python -u src/lightcone_cluster_worker.py \
    --W 24 --p "$P" --L "$L" \
    --realization-start "$START" --realization-count "$COUNT" \
    --n-realizations-total 2000 \
    --mode "$MODE" --base-seed 1962 --anchor odd \
    --out-dir "$OUT_DIR" \
    --gpu

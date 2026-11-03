#!/bin/bash
# lightcone_w24_reduce.slurm
# =============================
# Master-node collation step — run AFTER a mode's array job finishes (see
# submit_all.sh for --dependency=afterok wiring). Single task, no array.
# Merges every worker output file for MODE into a checkpoint and produces
# the heatmap figure (RdPu colormap, black cone lines — see
# src/lightcone_cluster_reduce.py).
#
# EDIT BEFORE SUBMITTING: same placeholders as lightcone_w24_cpu.slurm.
#
# USAGE:
#   sbatch --export=ALL,MODE=noiseless slurm/lightcone_w24_reduce.slurm
#   sbatch --export=ALL,MODE=noisy     slurm/lightcone_w24_reduce.slurm
#   # or, chained after the array job completes:
#   sbatch --dependency=afterok:<array_job_id> --export=ALL,MODE=noiseless \
#       slurm/lightcone_w24_reduce.slurm

#SBATCH --job-name=lightcone_w24_reduce
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/schmetterling-effect/.butterfly}"
OUT_DIR="${OUT_DIR:-/scratch/zt1/project/YOUR_ACCOUNT_HERE/lightcone_w24}"
REPO_DIR="${REPO_DIR:-$HOME/schmetterling-effect}"

if [[ -z "${MODE:-}" ]]; then
    echo "MODE not set — submit with --export=ALL,MODE=noisy or MODE=noiseless" >&2
    exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"

python -u src/lightcone_cluster_reduce.py --mode "$MODE" --out-dir "$OUT_DIR" --W 24

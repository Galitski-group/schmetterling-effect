#!/bin/bash
# submit_all.sh
# ===============
# Convenience wrapper: submits both noiseless and noisy array jobs plus
# their dependent reduce jobs. Run src/lightcone_cluster_plan.py FIRST to
# generate slurm/tasks_w24.tsv (this script reads its line count to size
# the array).
#
# Edit ARRAY_SLURM below to lightcone_w24_gpu.slurm once the GPU path is
# validated (see that file's own docstring) — defaults to the CPU variant
# since it's the better-tested starting point.
#
# USAGE:
#   .butterfly/bin/python src/lightcone_cluster_plan.py
#   ./slurm/submit_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TASK_TABLE="slurm/tasks_w24.tsv"
ARRAY_SLURM="slurm/lightcone_w24_cpu.slurm"   # or lightcone_w24_gpu.slurm
REDUCE_SLURM="slurm/lightcone_w24_reduce.slurm"
THROTTLE="%50"

if [[ ! -f "$TASK_TABLE" ]]; then
    echo "Missing $TASK_TABLE — run src/lightcone_cluster_plan.py first." >&2
    exit 1
fi

N=$(wc -l < "$TASK_TABLE")
ARRAY_RANGE="0-$((N - 1))${THROTTLE}"

for MODE in noiseless noisy; do
    echo "Submitting $MODE array ($ARRAY_RANGE) using $ARRAY_SLURM ..."
    ARRAY_JOB_ID=$(sbatch --parsable --array="$ARRAY_RANGE" --export=ALL,MODE="$MODE" "$ARRAY_SLURM")
    echo "  -> array job $ARRAY_JOB_ID"

    REDUCE_JOB_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_JOB_ID" \
        --export=ALL,MODE="$MODE" "$REDUCE_SLURM")
    echo "  -> reduce job $REDUCE_JOB_ID (runs after $ARRAY_JOB_ID completes)"
done

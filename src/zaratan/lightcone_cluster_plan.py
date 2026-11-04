"""
lightcone_cluster_plan.py
============================
Generates the SLURM array task table for the cluster light-cone sweep:
one line per (p, L, realization-chunk) triple, in the exact order SLURM
array task indices will read them. Run this ONCE before submitting the
array jobs (same table is reused for both --noisy and --noiseless, since
the (p, L, chunk) grid is identical — only the noise setting differs,
which the worker script controls via --mode).

Chunk size (REALIZATIONS_PER_NODE) is the "n realizations per node" knob
— tune it based on real per-realization timing measured on the cluster
(NOT the rough, contention-affected timings this was designed from
locally): submit a small test array first (e.g. just L=1 and L=W) to get
real per-task walltime on actual Zaratan hardware before committing to
the full grid, then adjust REALIZATIONS_PER_NODE / --time accordingly.
Cost is wildly non-uniform across L (roughly exponential-ish growth with
L for fixed W, and hard exponential growth with W) — the SAME chunk size
at L=1 finishes in a small fraction of the time it takes at L=W, so the
array's --time must be sized for the WORST-CASE (largest L) chunk; cheap
low-L tasks just finish early and release their allocation, which is not
wasted (SLURM array tasks don't hold time they don't use).

USAGE:
    .butterfly/bin/python src/lightcone_cluster_plan.py
    # writes slurm/tasks_w24.tsv, prints the array size to use, e.g.:
    #   sbatch --array=0-1439%50 --export=ALL,MODE=noiseless slurm/lightcone_w24_cpu.slurm
    #   sbatch --array=0-1439%50 --export=ALL,MODE=noisy     slurm/lightcone_w24_cpu.slurm
    # (or slurm/submit_all.sh, which wraps both plus the dependent reduce jobs)
"""

import pathlib

W = 24
L_VALUES = list(range(1, W + 1))
P_VALUES = [0.0, 0.01, 0.05]
N_REALIZATIONS_TOTAL = 2000
REALIZATIONS_PER_NODE = 100  # <-- tune this after a real timing calibration on Zaratan

TASK_TABLE_PATH = pathlib.Path(__file__).parent.parent / "slurm" / "tasks_w24.tsv"

if __name__ == "__main__":
    rows = []
    for p in P_VALUES:
        for L in L_VALUES:
            start = 0
            while start < N_REALIZATIONS_TOTAL:
                count = min(REALIZATIONS_PER_NODE, N_REALIZATIONS_TOTAL - start)
                rows.append((p, L, start, count))
                start += count

    TASK_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASK_TABLE_PATH, "w") as f:
        for p, L, start, count in rows:
            f.write(f"{p}\t{L}\t{start}\t{count}\n")

    print(f"W={W}  L_VALUES=1..{W}  P_VALUES={P_VALUES}  "
          f"N_REALIZATIONS_TOTAL={N_REALIZATIONS_TOTAL}  "
          f"REALIZATIONS_PER_NODE={REALIZATIONS_PER_NODE}")
    print(f"Wrote {len(rows)} task(s) -> {TASK_TABLE_PATH}")
    print()
    print("Submit with (same task table drives both noise conditions):")
    print(f"    sbatch --array=0-{len(rows) - 1}%50 slurm/lightcone_w24_noiseless.slurm")
    print(f"    sbatch --array=0-{len(rows) - 1}%50 slurm/lightcone_w24_noisy.slurm")
    print()
    print("(%50 throttles to 50 concurrent tasks — adjust to your allocation's limits.)")

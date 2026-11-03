"""
lightcone_cluster_worker.py
==============================
Per-SLURM-task worker for the large-W (default W=24) cluster light-cone
sweep. Each task handles ONE (p, L, realization-chunk) triple: a fixed
number of disorder realizations at a single (p, L) point, computed on its
own node and written to its own output file on shared storage — no
coordination with other tasks needed, so tasks can run in any order, on
any number of nodes, and a failed/requeued task just overwrites its own
file.

NOT TILED, deliberately: build_parallel_lightcone_probe_all_circuits_batched
packs multiple realizations as disjoint qubit blocks into ONE wider
circuit, which is a real win for real hardware (cost driver there is
billed circuit count, qubits are cheap up to the device budget) but is
catastrophic for local exact statevector simulation, where cost is
exponential in the TOTAL simulated qubit count -- tiling even 2
realizations at W=24 would mean simulating a 48-qubit statevector
(2^48 amplitudes). Each realization here gets its own dedicated W-qubit
circuit; node-level throughput instead comes from ordinary process-level
parallelism across realizations (see _run_one_realization / --n-workers
below), which scales memory LINEARLY in the number of concurrent workers
instead of exponentially.

GPU: attempted via pytket AerBackend's underlying qiskit_aer.AerSimulator
(backend._qiskit_backend.set_options(device="GPU")) -- pytket's own
AerBackend.__init__ doesn't expose a device= parameter (verified by
inspecting its signature), but it stores the real AerSimulator as
_qiskit_backend, and AerSimulator.set_options(device="GPU") is the
documented qiskit-aer API for opting into GPU execution (requires the
CUDA-enabled qiskit-aer-gpu package installed on the node). --gpu implies
--n-workers=1: GPU work doesn't benefit from CPU-side multiprocessing the
way independent statevector sims do, and multiple processes sharing one
GPU would just serialize against each other anyway. Falls back to CPU
with a clear warning if GPU isn't actually available (e.g. package not
installed, no CUDA device) -- UNTESTED against real GPU hardware, since
this was written and smoke-tested on a machine with no GPU; verify on an
actual Zaratan gpu-partition node before trusting it at scale.

OUTPUT FORMAT: pickles {"W":.., "base_seed":.., "p":.., "L":.., "mode":..,
"realization_start":.., "realization_count":.., "d_r_by_x": {x: [...]}}
to {out_dir}/{mode}/p{p:.3f}_L{L:03d}_r{start:05d}_{start+count:05d}.pkl
-- see lightcone_cluster_reduce.py for how these get merged.

USAGE (normally invoked by the SLURM array script, not directly):
    .butterfly/bin/python src/lightcone_cluster_worker.py \\
        --W 24 --p 0.01 --L 12 --realization-start 0 --realization-count 100 \\
        --mode noiseless --base-seed 1962 --anchor odd \\
        --out-dir /path/to/shared/scratch/lightcone_w24 [--gpu] [--n-workers 8]
"""

import argparse
import os
import pathlib
import pickle
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import run_disorder_avg as R


def _run_one_realization(args_tuple):
    (W, L, p, pert_site, anchor, base_seed, n_realizations_for_seed, r,
     use_gpu, n_threads) = args_tuple

    from pytket.extensions.qiskit import AerBackend

    backend = AerBackend()
    if n_threads is not None:
        backend._qiskit_backend.set_options(max_parallel_threads=n_threads)
    if use_gpu:
        try:
            backend._qiskit_backend.set_options(device="GPU")
        except Exception as exc:  # pragma: no cover - depends on cluster GPU setup
            print(f"[worker] GPU requested but unavailable ({exc!r}) — falling back to CPU",
                  file=sys.stderr)

    build = R.build_brickwall_lightcone_probe_all_circuit

    def _s(axis):
        return (base_seed + axis) * 1_000_000_000 + L * n_realizations_for_seed + r

    shared = dict(
        L=L, W=W, pert_site=pert_site, pert_op=R.PERT_OP, p=p,
        seed=_s(0), init_seed=_s(1), meas_seed=_s(2), meas_seed_ud=_s(3),
        add_barrier=False, anchor_parity_at_perturbation=anchor,
    )
    qc_p = build(**shared, unperturbed=False)
    qc_u = build(**shared, unperturbed=True)

    sp, su = R._run_circuits(
        [qc_p, qc_u], backend, 1,
        nexus_project=R.NEXUS_PROJECT, optimisation_level=R.OPTIMISATION_LEVEL,
        use_batch=False,
    )
    out_p = 1 - 2 * sp[:, 1 : 1 + W].astype(int)
    out_u = 1 - 2 * su[:, 1 : 1 + W].astype(int)
    d_r = (out_u.mean(axis=0) - out_p.mean(axis=0)).tolist()
    return r, d_r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--realization-start", type=int, required=True)
    ap.add_argument("--realization-count", type=int, required=True)
    ap.add_argument("--n-realizations-total", type=int, default=2000,
                     help="Total realizations per (p, L) point across ALL chunks/tasks "
                          "— must match across every task for the same (p, L), since it's "
                          "baked into the per-realization seed formula (same convention as "
                          "sweep_lightcone_probe_all/front_width_scaling.py).")
    ap.add_argument("--mode", choices=["noisy", "noiseless"], required=True)
    ap.add_argument("--base-seed", type=int, default=1962)
    ap.add_argument("--anchor", default="odd")
    ap.add_argument("--pert-site", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n-workers", type=int, default=None,
                     help="CPU multiprocessing pool size (ignored, forced to 1, under --gpu). "
                          "Defaults to os.cpu_count().")
    ap.add_argument("--threads-per-worker", type=int, default=1,
                     help="AerSimulator max_parallel_threads PER worker process — keep "
                          "n_workers * threads_per_worker <= node core count to avoid "
                          "oversubscription.")
    args = ap.parse_args()

    if args.mode == "noiseless":
        R.NOISE_P1Q = None
        R.NOISE_P2Q = None
        R.NOISE_PM = None
        R.NOISE_T1 = None
        R.NOISE_T2 = None
    # else "noisy": leave run_disorder_avg's module-level NOISE_* defaults as-is.

    pert_site = args.pert_site if args.pert_site is not None else args.W // 2
    n_workers = 1 if args.gpu else (args.n_workers or os.cpu_count() or 1)

    print(f"[worker] W={args.W} p={args.p} L={args.L} "
          f"realizations=[{args.realization_start}, "
          f"{args.realization_start + args.realization_count}) "
          f"mode={args.mode} gpu={args.gpu} n_workers={n_workers} "
          f"NOISE_P1Q={R.NOISE_P1Q} NOISE_P2Q={R.NOISE_P2Q} NOISE_PM={R.NOISE_PM}",
          flush=True)

    task_args = [
        (args.W, args.L, args.p, pert_site, args.anchor, args.base_seed,
         args.n_realizations_total, r, args.gpu, args.threads_per_worker)
        for r in range(args.realization_start, args.realization_start + args.realization_count)
    ]

    t0 = time.time()
    d_r_by_r: dict[int, list[float]] = {}
    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for r, d_r in pool.imap_unordered(_run_one_realization, task_args):
                d_r_by_r[r] = d_r
    else:
        for t in task_args:
            r, d_r = _run_one_realization(t)
            d_r_by_r[r] = d_r
    elapsed = time.time() - t0
    print(f"[worker] done: {len(d_r_by_r)} realizations in {elapsed:.1f}s "
          f"({elapsed / max(1, len(d_r_by_r)):.2f}s/realization)", flush=True)

    d_r_by_x: dict[int, list[float]] = {x: [] for x in range(-pert_site, args.W - pert_site)}
    for r in sorted(d_r_by_r):
        for q, val in enumerate(d_r_by_r[r]):
            d_r_by_x[q - pert_site].append(val)

    out_dir = pathlib.Path(args.out_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"p{args.p:.3f}_L{args.L:03d}_r{args.realization_start:05d}_"
        f"{args.realization_start + args.realization_count:05d}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump({
            "W": args.W, "base_seed": args.base_seed, "p": args.p, "L": args.L,
            "mode": args.mode, "pert_site": pert_site,
            "realization_start": args.realization_start,
            "realization_count": args.realization_count,
            "d_r_by_x": d_r_by_x,
        }, f)
    print(f"[worker] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

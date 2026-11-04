"""
recover_h2_2_job.py
====================
Recovers ONE (N, p, L) point's results from an ALREADY-SUBMITTED, still-
queued/running Nexus execute job, without resubmitting anything — for
exactly the scenario where a run_h2_2_tiled.py process died while waiting
(auth/connection hiccup, etc.) but the job itself kept running server-side.

Waits (resiliently — survives multi-hour device maintenance windows, same
_wait_for_job_resilient used everywhere else) for EXECUTE_JOB_ID to
complete, downloads its results, reduces them to the same C_t/d_r values
sweep_single_shot_disorder_tiled would have produced live, and writes them
into BOTH CHECKPOINT_PATH and RAW_RESULTS_PATH (the same paths
run_h2_2_tiled.py uses) under the correct (W, seed, p, L) key — so a
subsequent `run_h2_2_tiled.py --confirm` sees this point as already done
and skips it instead of re-submitting (and re-paying for) it.

IMPORTANT: the config block below (W, SEED, P, L, N_REALIZATIONS,
N_QUBITS_BUDGET, TILED_MAX_*) MUST match what run_h2_2_tiled.py actually
used for this point — it's used to deterministically reconstruct the exact
realizations list and chunk structure that produced the job being
recovered (no circuits are submitted here, this reconstruction is purely
local/free). If any of these differ from the real submission, the tags
won't match what's in the job's results and this will fail loudly (KeyError
on a missing tag) rather than silently recovering the wrong thing.

Usage (can take hours if the device is still in maintenance — background it):
    nohup .butterfly/bin/python src/recover_h2_2_job.py > recover_L16.log 2>&1 &
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import runs.run_disorder_avg as R

# ── Must match the ORIGINAL run_h2_2_tiled.py submission for this point ──────
W = 8
P = 0.05
L = 16
SEED = 1962
N_REALIZATIONS = 240
N_QUBITS_BUDGET = 48
TILED_MAX_DEPTH = 60
TILED_MAX_CLASSICAL_BITS = 64
TILED_MAX_SU4_PER_CHUNK = 1500
PERT_OP = "X"
PROBE_ANGLE = 0.0

EXECUTE_JOB_ID = "0c25a993-f28e-4453-804a-14e7d76e72de"
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_checkpoint.pkl")
RAW_RESULTS_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_raw_results.pkl")

# WAIT_FOR_COMPLETION=True blocks (resiliently, can take hours) until the
# JOB's overall status reaches a terminal state before fetching anything.
# WAIT_FOR_COMPLETION=False fetches whatever's available RIGHT NOW instead:
# qnexus's own _execute._results does one GET and returns each item's
# current per-item status/results regardless of the job's own overall
# status (it only gates on that when allow_incomplete=False, which
# get_results_by_job_id does not pass) — so a single permanently-stuck
# circuit cannot block retrieval of every other circuit that already
# finished; it only fails that one tag, same as any other per-item
# failure below. Set False when you already know (or suspect) one or more
# circuits won't reach a terminal state on their own.
WAIT_FOR_COMPLETION = False

if __name__ == "__main__":
    # 1) Reconstruct the exact realizations list + chunk structure that
    #    produced the original submission — deterministic from the seed
    #    formula, purely local, no circuits submitted here.
    realizations = []
    for r in range(N_REALIZATIONS):
        def _s(axis, r=r, s=SEED):
            return (s + axis) * 1_000_000_000 + L * N_REALIZATIONS + r
        realizations.append(dict(seed=_s(0), init_seed=_s(1), meas_seed=_s(2), meas_seed_ud=_s(3)))

    common = dict(
        L=L, W=W, n_qubits_budget=N_QUBITS_BUDGET, realizations=realizations,
        max_depth=TILED_MAX_DEPTH, max_classical_bits=TILED_MAX_CLASSICAL_BITS,
        max_su4_per_chunk=TILED_MAX_SU4_PER_CHUNK,
        pert_site=W // 2, pert_op=PERT_OP, probe_site=0, probe_angle=PROBE_ANGLE, p=P,
        add_barrier=True,
    )
    pert_chunks = R.build_parallel_otoc_circuits_batched(**common, unperturbed=False)
    unpert_chunks = R.build_parallel_otoc_circuits_batched(**common, unperturbed=True)
    print(f"Reconstructed {len(pert_chunks)} pert chunk(s) + {len(unpert_chunks)} unpert chunk(s) "
          f"for p={P} L={L} — cross-check this against what run_h2_2_tiled.py originally printed "
          f"('chunks per (N,L)' / circuit count) before trusting the recovery below.")

    tag_base = f"N{W}_p{P:.2f}_L{L:03d}_s{SEED}_tiled"

    # 2) Download the already-submitted job's results, keyed by circuit tag.
    #    A multi-circuit execute job can PARTIALLY fail/stall even when the
    #    overall job never reaches a terminal status (e.g. one circuit
    #    stuck indefinitely while the rest completed) -- results_by_tag/
    #    failures_by_tag reflect that per-item, not all-or-nothing. See
    #    WAIT_FOR_COMPLETION's comment above for what toggling it changes.
    all_tags = (
        [f"{tag_base}_pert_chunk{i}" for i in range(len(pert_chunks))]
        + [f"{tag_base}_unpert_chunk{i}" for i in range(len(unpert_chunks))]
    )
    if WAIT_FOR_COMPLETION:
        print(f"Waiting for execute job {EXECUTE_JOB_ID} to reach a terminal status … "
              f"(this can take hours, or forever if any one circuit never terminates)")
    else:
        print(f"Fetching CURRENT results for execute job {EXECUTE_JOB_ID} — not waiting "
              f"for overall completion, per-item results are available regardless of the "
              f"job's own status")
    results_by_tag, failures_by_tag = R.get_results_by_job_id(
        EXECUTE_JOB_ID, tags=all_tags, wait=WAIT_FOR_COMPLETION
    )
    print(f"Got {len(results_by_tag)} result(s), {len(failures_by_tag)} failure(s).")

    # 3) Keep only chunk indices where BOTH pert and unpert came back --
    #    a lone surviving half of a pair can't produce a valid d_r.
    kept_pert_shots, kept_unpert_shots = [], []
    kept_pert_chunks, kept_unpert_chunks = [], []
    dropped = []
    for i, (pchunk, uchunk) in enumerate(zip(pert_chunks, unpert_chunks)):
        pt, ut = f"{tag_base}_pert_chunk{i}", f"{tag_base}_unpert_chunk{i}"
        if pt in results_by_tag and ut in results_by_tag:
            kept_pert_shots.append(results_by_tag[pt]); kept_pert_chunks.append(pchunk)
            kept_unpert_shots.append(results_by_tag[ut]); kept_unpert_chunks.append(uchunk)
        else:
            dropped.append((i, pt in results_by_tag, ut in results_by_tag))

    if dropped:
        print(f"DROPPING {len(dropped)}/{len(pert_chunks)} chunk(s) (missing pert and/or "
              f"unpert half):")
        for i, has_pert, has_unpert in dropped:
            print(f"    chunk {i}: pert={'ok' if has_pert else 'MISSING'}  "
                  f"unpert={'ok' if has_unpert else 'MISSING'}")

    n_recovered = sum(n for _, _, n in kept_pert_chunks)
    print(f"Usable: {len(kept_pert_chunks)}/{len(pert_chunks)} chunk pairs, "
          f"{n_recovered}/{N_REALIZATIONS} realizations.")

    if not kept_pert_chunks:
        print("Nothing usable was recovered — not writing checkpoint/raw_results. "
              "This point will be resubmitted fresh on the next run_h2_2_tiled.py --confirm.")
        sys.exit(1)

    # 4) Reduce exactly like sweep_single_shot_disorder_tiled does live.
    means_p = np.concatenate([
        R.extract_parallel_probe_means(shots, w, n)
        for shots, (_, w, n) in zip(kept_pert_shots, kept_pert_chunks)
    ])
    means_u = np.concatenate([
        R.extract_parallel_probe_means(shots, w, n)
        for shots, (_, w, n) in zip(kept_unpert_shots, kept_unpert_chunks)
    ])
    d_r_values = (means_u - means_p).tolist()
    print(f"Recovered p={P} L={L}: C_t={(means_u - means_p).mean():.4f}  n={len(d_r_values)}")

    if n_recovered < N_REALIZATIONS:
        print(
            f"NOTE: only {n_recovered}/{N_REALIZATIONS} realizations recovered for this point "
            f"(accepted as final per instruction — not topping up or resubmitting). The "
            f"checkpoint resume check requires an EXACT match to N_REALIZATIONS to skip "
            f"re-submission, so a future run_h2_2_tiled.py --confirm would still see this point "
            f"as 'not done' and try to resubmit it WHOLE unless that check is loosened first."
        )

    # 5) Write into checkpoint + raw_results, exactly as the live sweep would
    #    (with the ACTUAL recovered count, not assumed to be N_REALIZATIONS).
    ckpt_key = (W, SEED, P, L)

    checkpoint = R._load_partial(CHECKPOINT_PATH) if pathlib.Path(CHECKPOINT_PATH).exists() else {}
    checkpoint[ckpt_key] = d_r_values
    R._save_partial(checkpoint, CHECKPOINT_PATH)

    raw_results = R._load_partial(RAW_RESULTS_PATH) if pathlib.Path(RAW_RESULTS_PATH).exists() else {}
    raw_results[ckpt_key] = {
        "pert_shots": [np.asarray(s) for s in kept_pert_shots],
        "unpert_shots": [np.asarray(s) for s in kept_unpert_shots],
        "pert_probe_bits": [
            np.asarray(s)[:, w : w + n] for s, (_, w, n) in zip(kept_pert_shots, kept_pert_chunks)
        ],
        "unpert_probe_bits": [
            np.asarray(s)[:, w : w + n]
            for s, (_, w, n) in zip(kept_unpert_shots, kept_unpert_chunks)
        ],
        "chunk_meta": {
            "pert": [(w, n) for _, w, n in kept_pert_chunks],
            "unpert": [(w, n) for _, w, n in kept_unpert_chunks],
        },
        "job_info": {
            "execute_job_id": EXECUTE_JOB_ID,
            "recovered": True,
            "n_realizations_requested": N_REALIZATIONS,
            "n_realizations_recovered": n_recovered,
            "dropped_chunks": dropped,
            "failures": failures_by_tag,
        },
    }
    R._save_partial(raw_results, RAW_RESULTS_PATH)

    print()
    print(f"Done — checkpoint and raw_results updated for p={P} L={L} "
          f"({n_recovered}/{N_REALIZATIONS} realizations).")
    if n_recovered == N_REALIZATIONS:
        print("Full recovery — re-run run_h2_2_tiled.py --confirm to continue with the "
              "remaining points; this one will now be skipped instead of re-submitted.")
    else:
        print("PARTIAL recovery — see NOTE above before re-running run_h2_2_tiled.py.")

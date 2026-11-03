"""
run_h2_2_tiled.py
==================
Runs the tiled single-shot OTOC sweep (single_shot_tiled mode) on the real
Quantinuum H2-2 device: build_parallel_otoc_circuit packs many independent
disorder realizations into a small number of wide circuits (width x depth
tiling) instead of one circuit per realization.

Chunk size is capped by THREE independent constraints, whichever is
tightest (see build_parallel_otoc_circuits_batched):
  - TILED_MAX_DEPTH            qubit-coherence budget (rows per circuit)
  - TILED_MAX_CLASSICAL_BITS   H2-2's real 64-bit classical register limit
  - TILED_MAX_SU4_PER_CHUNK    Nexus compile-server time budget — scales
                                with L, unlike the other two. Discovered
                                empirically mid-run: L=2/L=4 compiled fine
                                at chunk_size=58 (bits-bound), L=8 got its
                                compile job CANCELLED server-side
                                ("Timeout expired") at the same chunk size
                                (3248 total SU4 gates). 1500 is set with
                                margin below the observed 1624-gate success
                                point. This means chunk count — and total
                                circuits/cost — is NOT uniform across L
                                anymore; re-derive per L if you change
                                anything (see the per-L breakdown this
                                script prints below).

SAFETY:
  - device_name="H2-2" is passed explicitly below — nothing in
    run_disorder_avg.py defaults to hardware on its own (see DEVICE_NAME's
    module comment there).
  - Confirmation is a REQUIRED CLI FLAG (--confirm), not an interactive
    prompt: a backgrounded/nohup'd process normally has no attached stdin,
    so `input()` either hangs forever or raises EOFError immediately and
    exits — either way it can't actually gate a background run. Run once
    WITHOUT --confirm first (in your interactive shell) to review the
    summary risk-free; nothing is submitted unless --confirm is present.
  - Nexus login: run `qnx.login()` (browser-based) or the `qnx login` CLI
    command ONCE, interactively, in your own terminal BEFORE launching this
    script in the background. _run_circuits calls
    qnx.login_with_credentials() internally, which no-ops silently if a
    valid cached session already exists — so a prior external login means
    this script never needs to prompt for credentials itself.
  - N_QUBITS_BUDGET is NOT verified against H2-2's real physical qubit count
    here — confirm that yourself (e.g. via Nexus / QuantinuumBackend device
    info) before adding --confirm.
  - CHECKPOINT_PATH persists progress after every (N, seed, p, L) point —
    that's the natural atomic unit, since one _run_circuits call either
    fully completes or doesn't. If this process dies partway through (crash,
    non-nohup'd disconnect, OOM, or a compile-timeout traceback), rerunning
    this same script will skip every point already checkpointed instead of
    re-submitting (and re-paying for) circuits that already executed on
    hardware — this script prints which points that is below, every time.
  - RAW_RESULTS_PATH additionally persists, per (N, seed, p, L) point: the
    raw per-chunk get_shots() arrays (before reduction to the C_t
    statistic) and, on real hardware, Nexus's own compile_job_id /
    execute_job_id / circuit_ids / tags for that point — so exact raw bits
    or the originating Nexus job are recoverable later without re-running
    anything or hunting through the Nexus UI. Purely additive; doesn't
    affect checkpoint_path's resume behaviour.

USAGE:
    # 1) Dry run first — prints the summary (including what's already
    #    checkpointed from a previous partial run), submits nothing:
    .butterfly/bin/python src/run_h2_2_tiled.py

    # 2) Log in to Nexus once, interactively (only needed if not already
    #    logged in — is_logged_in() is checked automatically):
    .butterfly/bin/python -c "import qnexus as qnx; qnx.login()"

    # 3) Once you've verified N_QUBITS_BUDGET against the real device and
    #    are ready to spend real HQC budget, launch backgrounded with
    #    --confirm (nohup, not just & — a bare & still dies on terminal/SSH
    #    disconnect; nohup survives it). Safe to re-run this exact command
    #    after an interruption — already-checkpointed points are skipped:
    nohup .butterfly/bin/python src/run_h2_2_tiled.py --confirm > h2_2_run.log 2>&1 &

Once submitted, circuit execution happens on Quantinuum's Nexus-managed
infrastructure independent of this local process — killing this process
after a point's job has been submitted does not stop it running remotely,
it just means this script won't be there to receive/checkpoint the result
when it comes back (recoverable from Nexus directly if that happens, but
CHECKPOINT_PATH is what protects the normal case).
"""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_disorder_avg as R

# ── Config overrides ────────────────────────────────────────────────────────
R.N_VALUES = [8]
R.N_REALIZATIONS = 240
R.N_QUBITS_BUDGET = 48
R.TILED_MAX_DEPTH = 60
R.TILED_MAX_CLASSICAL_BITS = 64  # H2-2's real classical-register limit
R.TILED_MAX_SU4_PER_CHUNK = 1500  # compile-server budget; see module docstring above
R.P_VALUES = [0.0,.05]
R.L_VALUES = [12]
R.BASE_SEEDS = [1962]
R.PERT_OP = "X"
R.PROBE_ANGLE = 0.0
R.NEXUS_PROJECT = "schmetterling-effect-test"
R.OPTIMISATION_LEVEL = 0
R.NEXUS_BATCH = False

DEVICE_NAME = "H2-2"
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_checkpoint.pkl")
RAW_RESULTS_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "h2_2_tiled_raw_results.pkl")

if __name__ == "__main__":
    W = R.N_VALUES[0]
    width = R.N_QUBITS_BUDGET // W

    def _chunk_size_for(L):
        depth_chunk = R.N_REALIZATIONS if R.TILED_MAX_DEPTH is None else width * R.TILED_MAX_DEPTH
        bits_chunk = (
            R.N_REALIZATIONS if R.TILED_MAX_CLASSICAL_BITS is None
            else max(1, R.TILED_MAX_CLASSICAL_BITS - width)
        )
        gates_chunk = (
            R.N_REALIZATIONS if R.TILED_MAX_SU4_PER_CHUNK is None
            else max(1, R.TILED_MAX_SU4_PER_CHUNK // max(1, R._n_su4_per_realization(L, W)))
        )
        chunk = min(depth_chunk, bits_chunk, gates_chunk)
        binding = min(
            [("TILED_MAX_DEPTH", depth_chunk), ("TILED_MAX_CLASSICAL_BITS", bits_chunk),
             ("TILED_MAX_SU4_PER_CHUNK", gates_chunk)],
            key=lambda x: x[1],
        )[0]
        return chunk, -(-R.N_REALIZATIONS // chunk), binding

    # What's already checkpointed from a previous (possibly interrupted) run.
    checkpoint = {}
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = R._load_partial(CHECKPOINT_PATH)

    all_points = [(p, L) for p in R.P_VALUES for L in R.L_VALUES]
    remaining_points = [
        (p, L) for p, L in all_points
        if (W, R.BASE_SEEDS[0], p, L) not in checkpoint
        or len(checkpoint[(W, R.BASE_SEEDS[0], p, L)]) != R.N_REALIZATIONS
    ]
    done_points = [pt for pt in all_points if pt not in remaining_points]

    n_circuits_remaining = 0
    print("=" * 64)
    print(f"About to submit to REAL Quantinuum hardware: {DEVICE_NAME}")
    print("=" * 64)
    print(f"  N_VALUES        = {R.N_VALUES}")
    print(f"  N_REALIZATIONS  = {R.N_REALIZATIONS}")
    print(f"  N_QUBITS_BUDGET = {R.N_QUBITS_BUDGET}  (width={width})")
    print(f"  TILED_MAX_DEPTH = {R.TILED_MAX_DEPTH}")
    print(f"  TILED_MAX_CLASSICAL_BITS = {R.TILED_MAX_CLASSICAL_BITS}")
    print(f"  TILED_MAX_SU4_PER_CHUNK  = {R.TILED_MAX_SU4_PER_CHUNK}")
    print(f"  P_VALUES        = {R.P_VALUES}")
    print(f"  L_VALUES        = {R.L_VALUES}")
    print(f"  BASE_SEEDS      = {R.BASE_SEEDS}")
    print(f"  NEXUS_PROJECT   = {R.NEXUS_PROJECT!r}")
    print(f"  NEXUS_BATCH     = {R.NEXUS_BATCH}")
    print(f"  CHECKPOINT_PATH = {CHECKPOINT_PATH}")
    print(f"  RAW_RESULTS_PATH = {RAW_RESULTS_PATH}  (raw shots + Nexus job/circuit IDs, per point)")
    print()
    print(f"  Already checkpointed ({len(done_points)}/{len(all_points)} points) — will be SKIPPED:")
    for p, L in done_points:
        vals = checkpoint[(W, R.BASE_SEEDS[0], p, L)]
        print(f"    p={p:.2f} L={L:3d}  C_t={sum(vals)/len(vals):.4f}  (n={len(vals)})")
    print()
    print(f"  Remaining ({len(remaining_points)}/{len(all_points)} points) — per-L chunk breakdown:")
    for p, L in remaining_points:
        chunk_size, n_chunks, binding = _chunk_size_for(L)
        n_circuits_remaining += n_chunks * 2  # pert + unpert
        print(f"    p={p:.2f} L={L:3d}  chunk_size={chunk_size:3d}  n_chunks={n_chunks:3d}  "
              f"circuits={n_chunks*2:3d}  (binding: {binding})")
    print()
    print(f"  Estimated REMAINING circuit submissions ~= {n_circuits_remaining}")
    print("  Re-derive the HQC estimate if you change N_REALIZATIONS/N_QUBITS_BUDGET/")
    print("  L_VALUES/P_VALUES/any TILED_MAX_* constant (build the circuits locally via")
    print("  build_parallel_otoc_circuits_batched and count native-compiled gates).")
    print()
    print("  !! N_QUBITS_BUDGET has NOT been checked against H2-2's real physical")
    print("  !! qubit count — verify that yourself before adding --confirm.       !!")
    print("=" * 64)

    if "--confirm" not in sys.argv:
        print()
        print("DRY RUN — nothing submitted. Re-run with --confirm to actually")
        print("submit to hardware, e.g.:")
        print(f"    nohup .butterfly/bin/python {pathlib.Path(__file__).name} --confirm "
              f"> h2_2_run.log 2>&1 &")
        sys.exit(0)

    print()
    print("--confirm passed — submitting to real hardware now.")
    R.run_sequential(
        mode="single_shot_tiled",
        device_name=DEVICE_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        raw_results_path=RAW_RESULTS_PATH,
    )

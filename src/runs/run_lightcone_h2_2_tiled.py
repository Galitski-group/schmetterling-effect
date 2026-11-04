"""
run_lightcone_h2_2_tiled.py
=============================
Runs the TILED light-cone "measure every site" sweep
(sweep_lightcone_probe_all_tiled) on the real Quantinuum H2-2 device:
build_parallel_lightcone_probe_all_circuit packs many independent disorder
realizations into a small number of wide circuits (width x depth tiling)
instead of one circuit per realization — the lightcone-probe-all analogue
of run_h2_2_tiled.py.

IMPORTANT COST NOTE: this circuit measures every site (not one probe), so
each realization costs W classical bits instead of 1 — the classical
register limit (TILED_MAX_CLASSICAL_BITS) binds ~W times sooner than in
run_h2_2_tiled.py's sweep. At the config below (W=8, n_qubits_budget=48,
max_classical_bits=64): width=6, only ~7 realizations fit per chunk
(58 bits // 8), giving 35 chunks per (p, L) point — a ~6.9x reduction in
circuit count vs. one-circuit-per-realization (2*240=480), not the ~50x
run_h2_2_tiled.py's sweep gets at the same budget. See
build_parallel_lightcone_probe_all_circuit's own docstring for the full
explanation. Local dry-run at this exact config estimated ~2100 total
circuit submissions and ~14,700 HQC across all L/p — re-derive if you
change anything (this script's own dry-run printout below does that from
the actual chunk-building code, not a hand-derived formula).

Chunk size is capped by THREE independent constraints, whichever is
tightest (see build_parallel_lightcone_probe_all_circuits_batched):
  - TILED_MAX_DEPTH            qubit-coherence budget (rows per circuit)
  - TILED_MAX_CLASSICAL_BITS   H2-2's real 64-bit classical register limit
  - TILED_MAX_SU4_PER_CHUNK    Nexus compile-server time budget — scales
                                with L, unlike the other two (see
                                run_h2_2_tiled.py's own docstring for the
                                empirical cliff that motivated this cap;
                                same 1500 default used here).

SAFETY:
  - device_name="H2-2" is passed explicitly below — nothing in
    run_disorder_avg.py defaults to hardware on its own.
  - Confirmation is a REQUIRED CLI FLAG (--confirm), not an interactive
    prompt — see run_h2_2_tiled.py's own docstring for why (input() can't
    gate a backgrounded/nohup'd process).
  - Nexus login: run `qnx.login()` (browser-based) or the `qnx login` CLI
    command ONCE, interactively, in your own terminal BEFORE launching this
    script in the background.
  - N_QUBITS_BUDGET is NOT verified against H2-2's real physical qubit
    count here — confirm that yourself before adding --confirm.
  - CHECKPOINT_PATH persists progress after every (p, L) point — the
    natural atomic unit here (one _run_circuits call either fully
    completes or doesn't). Safe to re-run this same command after an
    interruption — already-checkpointed points are skipped, never
    re-submitted (and re-paid for).
  - RAW_RESULTS_PATH additionally persists, per (p, L) point: the raw
    per-chunk get_shots() arrays and Nexus job/circuit IDs. Purely
    additive; doesn't affect checkpoint_path's resume behaviour.

USAGE:
    # 1) Dry run first — prints the summary (including what's already
    #    checkpointed), submits nothing:
    .butterfly/bin/python src/run_lightcone_h2_2_tiled.py

    # 2) Log in to Nexus once, interactively (only needed if not already
    #    logged in):
    .butterfly/bin/python -c "import qnexus as qnx; qnx.login()"

    # 3) Once you've verified N_QUBITS_BUDGET against the real device and
    #    the cost estimate below is acceptable, launch backgrounded with
    #    --confirm:
    nohup .butterfly/bin/python src/run_lightcone_h2_2_tiled.py --confirm \\
        > lightcone_h2_2_run.log 2>&1 &
"""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runs.run_disorder_avg as R

# ── Config ───────────────────────────────────────────────────────────────────
W = 8
N_QUBITS_BUDGET = 48
N_REALIZATIONS = 180
P_VALUES = [0.05,0.0]
L_VALUES = list(range(1, 9))
BASE_SEED = 1962
TILED_MAX_DEPTH = 60
TILED_MAX_CLASSICAL_BITS = 64  # H2-2's real classical-register limit
TILED_MAX_SU4_PER_CHUNK = 1500  # compile-server budget; see module docstring above
ANCHOR_PARITY_AT_PERTURBATION = "odd"  # None/"even"/"odd" — see
    # build_brickwall_lightcone_probe_all_circuit's docstring. "even" makes
    # the immediate post-perturbation coupling consistently favour x=+1
    # across every L (not true x<->-x symmetry, just a stable, non-erratic
    # offset) — pick "odd" instead for a stable x=-1-favouring offset, or
    # None to reproduce the original L-parity-dependent (alternating) bias.
NEXUS_PROJECT = "schmetterling-effect-light-cone"
OPTIMISATION_LEVEL = 0
USE_BATCH = False

DEVICE_NAME = "H2-1"
CHECKPOINT_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_checkpoint.pkl")
RAW_RESULTS_PATH = str(pathlib.Path(R.PARTIALS_DIR) / "lightcone_h2_2_tiled_raw_results.pkl")
FIGURE_PATH = str(R.FIGURE_DIR / "lightcone_probe_all_h2_2.png")

if __name__ == "__main__":
    width = N_QUBITS_BUDGET // W

    def _chunk_info_for(L):
        # Mirrors build_parallel_lightcone_probe_all_circuits_batched's own
        # chunk-size math exactly (see that function) — computed here purely
        # for the dry-run printout, not used to build anything.
        depth_chunk = N_REALIZATIONS if TILED_MAX_DEPTH is None else width * TILED_MAX_DEPTH
        bits_chunk = (
            N_REALIZATIONS if TILED_MAX_CLASSICAL_BITS is None
            else max(1, (TILED_MAX_CLASSICAL_BITS - width) // W)
        )
        gates_chunk = (
            N_REALIZATIONS if TILED_MAX_SU4_PER_CHUNK is None
            else max(1, TILED_MAX_SU4_PER_CHUNK // max(1, R._n_su4_per_realization(L, W)))
        )
        chunk = min(depth_chunk, bits_chunk, gates_chunk)
        binding = min(
            [("TILED_MAX_DEPTH", depth_chunk), ("TILED_MAX_CLASSICAL_BITS", bits_chunk),
             ("TILED_MAX_SU4_PER_CHUNK", gates_chunk)],
            key=lambda x: x[1],
        )[0]
        return chunk, -(-N_REALIZATIONS // chunk), binding

    checkpoint = {}
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = R._load_partial(CHECKPOINT_PATH)

    all_points = [(p, L) for p in P_VALUES for L in L_VALUES]
    remaining_points = [
        (p, L) for p, L in all_points
        if (W, BASE_SEED, p, L) not in checkpoint
        or any(len(v) != N_REALIZATIONS for v in checkpoint[(W, BASE_SEED, p, L)].values())
    ]
    done_points = [pt for pt in all_points if pt not in remaining_points]

    n_circuits_remaining = 0
    print("=" * 64)
    print(f"About to submit to REAL Quantinuum hardware: {DEVICE_NAME}")
    print("=" * 64)
    print(f"  W               = {W}")
    print(f"  N_REALIZATIONS  = {N_REALIZATIONS}")
    print(f"  N_QUBITS_BUDGET = {N_QUBITS_BUDGET}  (width={width})")
    print(f"  TILED_MAX_DEPTH = {TILED_MAX_DEPTH}")
    print(f"  TILED_MAX_CLASSICAL_BITS = {TILED_MAX_CLASSICAL_BITS}")
    print(f"  TILED_MAX_SU4_PER_CHUNK  = {TILED_MAX_SU4_PER_CHUNK}")
    print(f"  ANCHOR_PARITY_AT_PERTURBATION = {ANCHOR_PARITY_AT_PERTURBATION!r}")
    print(f"  P_VALUES        = {P_VALUES}")
    print(f"  L_VALUES        = {L_VALUES}")
    print(f"  BASE_SEED       = {BASE_SEED}")
    print(f"  NEXUS_PROJECT   = {NEXUS_PROJECT!r}")
    print(f"  CHECKPOINT_PATH = {CHECKPOINT_PATH}")
    print(f"  RAW_RESULTS_PATH = {RAW_RESULTS_PATH}")
    print()
    print(f"  Already checkpointed ({len(done_points)}/{len(all_points)} points) — will be SKIPPED:")
    for p, L in done_points:
        vals_by_x = checkpoint[(W, BASE_SEED, p, L)]
        n = len(next(iter(vals_by_x.values())))
        print(f"    p={p:.2f} L={L:3d}  n={n}")
    print()
    print(f"  Remaining ({len(remaining_points)}/{len(all_points)} points) — per-L chunk breakdown:")
    seen_L = set()
    for p, L in remaining_points:
        chunk_size, n_chunks, binding = _chunk_info_for(L)
        n_circuits_remaining += n_chunks * 2  # pert + unpert
        if L not in seen_L:
            print(f"    L={L:3d}  chunk_size={chunk_size:3d}  n_chunks={n_chunks:3d}  "
                  f"circuits/p={n_chunks*2:3d}  (binding: {binding})")
            seen_L.add(L)
    print()
    print(f"  Estimated REMAINING circuit submissions ~= {n_circuits_remaining}")
    print("  Re-derive the HQC estimate if you change N_REALIZATIONS/N_QUBITS_BUDGET/")
    print("  L_VALUES/P_VALUES/any TILED_MAX_* constant (build the circuits locally via")
    print("  build_parallel_lightcone_probe_all_circuits_batched and count native gates —")
    print("  see the cost-report computation used to size this exact config: ~2100")
    print("  circuits, ~14,700 HQC estimated for the full W=8/n=240/p={0,0.01,0.05}/L=1..10")
    print("  sweep from scratch).")
    print()
    print("  !! N_QUBITS_BUDGET has NOT been checked against H2-2's real physical")
    print("  !! qubit count — verify that yourself before adding --confirm.       !!")
    print("=" * 64)

    if "--confirm" not in sys.argv:
        print()
        print("DRY RUN — nothing submitted. Re-run with --confirm to actually")
        print("submit to hardware, e.g.:")
        print(f"    nohup .butterfly/bin/python {pathlib.Path(__file__).name} --confirm "
              f"> lightcone_h2_2_run.log 2>&1 &")
        sys.exit(0)

    print()
    print("--confirm passed — submitting to real hardware now.")
    stats_by_x = R.sweep_lightcone_probe_all_tiled(
        W=W,
        p_values=P_VALUES,
        L_values=L_VALUES,
        n_qubits_budget=N_QUBITS_BUDGET,
        n_realizations=N_REALIZATIONS,
        base_seed=BASE_SEED,
        optimisation_level=OPTIMISATION_LEVEL,
        use_batch=USE_BATCH,
        anchor_parity_at_perturbation=ANCHOR_PARITY_AT_PERTURBATION,
        max_depth=TILED_MAX_DEPTH,
        max_classical_bits=TILED_MAX_CLASSICAL_BITS,
        max_su4_per_chunk=TILED_MAX_SU4_PER_CHUNK,
        device_name=DEVICE_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        raw_results_path=RAW_RESULTS_PATH,
    )
    R.plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_VALUES,
        p_values=P_VALUES,
        stat="C",
        figure_path=FIGURE_PATH,
    )

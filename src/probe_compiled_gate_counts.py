"""
probe_compiled_gate_counts.py
==============================
Compiles a small grid of sample circuits on the Quantinuum backend and
reports the actual post-compilation gate counts (N1q, N2q, Nm).

Usage — standalone
------------------
.butterfly/bin/python -u probe_compiled_gate_counts.py

Usage — from a Jupyter notebook
---------------------------------
from probe_compiled_gate_counts import run_probe
results, raw_circuits, compiled_circuits = run_probe()
"""

import functools
import pathlib
import importlib.util

from pytket import Circuit
from pytket.circuit import OpType

# ── Load src/__main__.py ──────────────────────────────────────────────────────
_src  = pathlib.Path(__file__).parent / "__main__.py"
_spec = importlib.util.spec_from_file_location("schmetterling", _src)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

generate       = _mod.generate_time_reversal_breaking_random_brick_wall
validate_nexus = _mod.validate_nexus_connection
qnx            = _mod.qnx

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE_NAME = "H2-1E"
PROJECT     = "measurement-induced butterfly effect"
BASE_SEED   = 0
OPT_LEVEL = 0

# PROBE_CONFIGS = [
#     (4,  5),
#     (4,  10),
#     (4,  20),
#     (8,  5),
#     (8,  10),
#     (8,  20),
#     (12, 5),
#     (12, 10),
# ]

PROBE_CONFIGS = [
    (50,  5),
    (50,  10),
    (50,  25),
    (50,  30),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_circuit(ref) -> Circuit:
    """
    Extract a pytket Circuit from a qnexus CircuitRef.
    Tries every known pattern across SDK versions; raises a descriptive error
    listing available attributes if none work.
    """
    # Try direct attributes on the ref itself
    for attr in ("pytket_circuit", "circuit"):
        if hasattr(ref, attr):
            val = getattr(ref, attr)
            if isinstance(val, Circuit):
                return val

    for method in ("download", "download_circuit", "get_circuit"):
        if hasattr(ref, method):
            result = getattr(ref, method)()
            if isinstance(result, Circuit):
                return result

    # Try fetching a model from qnx.circuits
    try:
        model = qnx.circuits.get(ref)
        for attr in ("pytket_circuit", "circuit"):
            if hasattr(model, attr):
                val = getattr(model, attr)
                if isinstance(val, Circuit):
                    return val
        for method in ("download", "download_circuit", "get_circuit"):
            if hasattr(model, method):
                result = getattr(model, method)()
                if isinstance(result, Circuit):
                    return result
    except Exception:
        pass

    public = [a for a in dir(ref) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot extract pytket Circuit from {type(ref).__name__}.\n"
        f"Available attributes: {public}\n"
        "Inspect the object in a notebook with dir(ref) and update _download_circuit."
    )


def count_gates(circuit: Circuit) -> dict:
    n1q = n2q = nm = 0
    for cmd in circuit.get_commands():
        t = cmd.op.type
        if t in (OpType.Barrier, OpType.Reset):
            continue
        if t == OpType.Measure:
            nm += 1
        elif len(cmd.qubits) == 1:
            n1q += 1
        elif len(cmd.qubits) == 2:
            n2q += 1
    return {"N1q": n1q, "N2q": n2q, "Nm": nm}


def hqc_per_shot(n1q: int, n2q: int, nm: int) -> float:
    return 5 + (n1q + 10 * n2q + 5 * nm) / 5000


def compare_hqc(
    raw_circuits: list[Circuit],
    compiled_circuits: list[Circuit],
    n_shots: int = 1,
    labels: list[str] | None = None,
) -> list[dict]:
    """
    Compute and print HQC costs for a paired list of raw and compiled circuits.

    No Nexus connection required — operates purely on pytket Circuit objects.

    Parameters
    ----------
    raw_circuits      : circuits before compilation (as built by the generator)
    compiled_circuits : circuits after Quantinuum compilation
    n_shots           : shots per circuit (default 1 for single-shot mode)
    labels            : optional names for each circuit pair, used in the table

    Returns
    -------
    List of dicts, one per circuit pair:
        label, raw_N1q, raw_N2q, raw_Nm, raw_hqc,
        comp_N1q, comp_N2q, comp_Nm, comp_hqc, comp_hqc_total
    where comp_hqc_total = comp_hqc * n_shots.
    """
    if len(raw_circuits) != len(compiled_circuits):
        raise ValueError(
            f"raw_circuits ({len(raw_circuits)}) and compiled_circuits "
            f"({len(compiled_circuits)}) must have the same length."
        )

    labels = labels or [str(i) for i in range(len(raw_circuits))]

    print(
        f"\n{'Label':<18} "
        f"{'── raw ──':^26}  "
        f"{'── compiled ──':^38}"
    )
    print(
        f"{'':18} "
        f"{'N1q':>6} {'N2q':>6} {'Nm':>5} {'HQC':>7}  "
        f"{'N1q':>6} {'N2q':>6} {'Nm':>5} {'HQC/shot':>9} {'HQC×Ns':>9}"
    )
    print("-" * 80)

    rows = []
    total_raw = 0.0
    total_comp = 0.0

    for lbl, qc_r, qc_c in zip(labels, raw_circuits, compiled_circuits):
        gr = count_gates(qc_r)
        gc = count_gates(qc_c)
        hr = hqc_per_shot(gr["N1q"], gr["N2q"], gr["Nm"])
        hc = hqc_per_shot(gc["N1q"], gc["N2q"], gc["Nm"])
        hc_total = hc * n_shots
        total_raw  += hr
        total_comp += hc_total

        print(
            f"{lbl:<18} "
            f"{gr['N1q']:>6} {gr['N2q']:>6} {gr['Nm']:>5} {hr:>7.3f}  "
            f"{gc['N1q']:>6} {gc['N2q']:>6} {gc['Nm']:>5} {hc:>9.3f} {hc_total:>9.3f}"
        )
        rows.append(dict(
            label=lbl,
            raw_N1q=gr["N1q"], raw_N2q=gr["N2q"], raw_Nm=gr["Nm"], raw_hqc=hr,
            comp_N1q=gc["N1q"], comp_N2q=gc["N2q"], comp_Nm=gc["Nm"],
            comp_hqc=hc, comp_hqc_total=hc_total,
        ))

    print("-" * 80)
    print(f"{'TOTAL (all circuits)':>53}  {total_raw:>9.3f} {total_comp:>9.3f}")
    print(f"  n_shots = {n_shots}\n")

    return rows


# ── Core ──────────────────────────────────────────────────────────────────────

def run_probe(
    probe_configs=PROBE_CONFIGS,
    device_name=DEVICE_NAME,
    project_name=PROJECT,
    base_seed=BASE_SEED,
    optimization_level=OPT_LEVEL,
    _print=None,
):
    log = _print or (lambda *a, **k: None)

    log("Connecting to Nexus …")
    validate_nexus(nexus_hosted=True)
    project_ref = qnx.projects.get_or_create(name=project_name)
    log(f"Project: {project_name!r}\n")

    raw_circuits = []
    for W, L in probe_configs:
        qc = generate(
            L=L, W=W,
            unperturbed=True,
            n_meas_total=0,
            add_barrier=False,
            seed=base_seed,
            init_seed=base_seed,
            meas_seed=base_seed,
        )
        raw_circuits.append(qc)
        log(f"Built  W={W:2d}  L={L:2d}  ({qc.n_qubits}q, {qc.n_gates} raw gates)")

    log(f"\nUploading {len(raw_circuits)} circuit(s) …")
    refs = [
        qnx.circuits.upload(circuit=qc, name=f"probe_W{W}_L{L}", project=project_ref)
        for qc, (W, L) in zip(raw_circuits, probe_configs)
    ]

    log("Submitting compile job …")
    compile_job = qnx.start_compile_job(
        programs=refs,
        name="probe_compile",
        optimisation_level=optimization_level,
        backend_config=qnx.QuantinuumConfig(device_name=device_name),
        project=project_ref,
    )

    log("Waiting for compilation …")
    qnx.jobs.wait_for(compile_job)

    log("Downloading compiled circuits …")
    compiled_circuits = [
        _download_circuit(item.get_output())
        for item in qnx.jobs.results(compile_job)
    ]
    log("Done.\n")

    results = []
    log(f"{'W':>4} {'L':>4}  {'N1q':>6} {'N2q':>6} {'Nm':>5}  "
        f"{'HQC/shot':>10}  {'n_su4':>7} {'N2q/su4':>9} {'N1q/su4':>9}")
    log("-" * 75)

    for (W, L), qc_c in zip(probe_configs, compiled_circuits):
        g = count_gates(qc_c)
        h = hqc_per_shot(g["N1q"], g["N2q"], g["Nm"])
        L_even = (L + 1) // 2
        L_odd  = L // 2
        n_su4  = 2 * (L_even * (W // 2) + L_odd * (W // 2 - 1))
        n2_per = g["N2q"] / n_su4 if n_su4 else float("nan")
        n1_per = g["N1q"] / n_su4 if n_su4 else float("nan")
        row = dict(W=W, L=L, N1q=g["N1q"], N2q=g["N2q"], Nm=g["Nm"],
                   hqc_per_shot=h, n_su4=n_su4, N2q_per_su4=n2_per, N1q_per_su4=n1_per)
        results.append(row)
        log(f"{W:>4} {L:>4}  {g['N1q']:>6} {g['N2q']:>6} {g['Nm']:>5}  "
            f"{h:>10.4f}  {n_su4:>7} {n2_per:>9.2f} {n1_per:>9.2f}")

    return results, raw_circuits, compiled_circuits


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _print = functools.partial(print, flush=True)
    run_probe(_print=_print)

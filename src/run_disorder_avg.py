"""
run_disorder_avg.py
====================
Orchestrates the disorder-averaged OTOC experiment.

Four partition schemes are available for parallel execution:

  Scheme 1  — partition by N (system size)
              --scheme 1 --job-id {0..len(N_VALUES)-1}
              5 jobs for the default config.

  Scheme 2  — partition by c_idx (angle realization)
              --scheme 2 --job-id {0..N_CIRCUITS-1}
              20 jobs; each handles all N, all p, all init-states for one seed.

  Scheme 3  — partition by (c_idx, i_idx) (single disorder realization)
              --scheme 3 --job-id {c_idx} --secondary-id {i_idx}
              400 jobs; finest grain, best fault tolerance.

  Scheme 4  — partition by (N, p) grid point
              --scheme 4 --job-id {n_idx} --secondary-id {p_idx}
              25 jobs; maps 1-to-1 to grid cells in the C(p,T) figure.

Sequential (no flags)  — runs everything in one process (original behaviour).

Merge and plot after parallel jobs finish:
  python run_disorder_avg.py --scheme {1|2|3|4} --merge

Nohup launch helpers are printed when --help-nohup is passed.
"""

import argparse
import glob
import os
import pathlib
import pickle
import importlib.util
import datetime
from collections import defaultdict

import numpy as np

# ── Load src/__main__.py ───────────────────────────────────────────────────────
_src = pathlib.Path(__file__).parent / "__main__.py"
_spec = importlib.util.spec_from_file_location("schmetterling", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_brickwall_otoc_circuit = _mod.build_brickwall_otoc_circuit
build_bowtie_otoc_circuit = _mod.build_bowtie_otoc_circuit
build_brickwall_lightcone_probe_all_circuit = _mod.build_brickwall_lightcone_probe_all_circuit
build_bowtie_lightcone_probe_all_circuit = _mod.build_bowtie_lightcone_probe_all_circuit
_run_circuits = _mod._run_circuits
compute_C_t = _mod.compute_C_t
plot_results = _mod.plot_results
sweep_over_all_disorder_axes = _mod.sweep_over_all_disorder_axes
sweep_single_shot_disorder = _mod.sweep_single_shot_disorder
build_parallel_otoc_circuit = _mod.build_parallel_otoc_circuit
build_parallel_otoc_circuits_batched = _mod.build_parallel_otoc_circuits_batched
extract_parallel_probe_means = _mod.extract_parallel_probe_means
build_parallel_lightcone_probe_all_circuit = _mod.build_parallel_lightcone_probe_all_circuit
build_parallel_lightcone_probe_all_circuits_batched = _mod.build_parallel_lightcone_probe_all_circuits_batched
extract_parallel_lightcone_probe_all_means = _mod.extract_parallel_lightcone_probe_all_means
get_results_by_job_id = _mod.get_results_by_job_id
_n_su4_per_realization = _mod._n_su4_per_realization
QuantinuumBackend = _mod.QuantinuumBackend
build_aer_noise_model = _mod.build_aer_noise_model
qnx = _mod.qnx

# ── Load transfer_matrix_otoc.py ──────────────────────────────────────────────
_tm_src = pathlib.Path(__file__).parent / "transfer_matrix_otoc.py"
_tm_spec = importlib.util.spec_from_file_location("transfer_matrix_otoc", _tm_src)
_tm_mod = importlib.util.module_from_spec(_tm_spec)
_tm_spec.loader.exec_module(_tm_mod)
ButterflyTransferMatrix = _tm_mod.ButterflyTransferMatrix

# ══════════════════════════════════════════════════════════════════════════════
# Parameters  (edit here; all scheme functions read from these globals)
# ══════════════════════════════════════════════════════════════════════════════
#
#   Three independent disorder axes per realization (c_idx, i_idx):
#     Axis 1 — gate angles      : circuit_seed = BASE_SEED * 10_000 + c_idx
#     Axis 2 — initial Pauli word: init_seed   = BASE_SEED * 10_000_000 + c_idx*1_000 + i_idx
#     Axis 3 — measurement sites : meas_seed   = (MEAS_SEED+1)*10_000_000 + c_idx*1_000 + i_idx
#
DATE = f"{datetime.datetime.now():%Y_%m_%d_%H:%M:%S}"
DEVICE_NAME = "H2-2"  # Quantinuum device name, kept here as a reference value only.
# SAFETY: nothing reads this automatically — every function that
# builds a backend takes its own device_name=None-by-default
# argument (always local simulation unless you pass one
# explicitly), so this constant can sit here configured for
# real hardware without anything accidentally running on it.
# Opt in with device_name=DEVICE_NAME, or `--device-name H2-2`
# on the CLI (or `--device-name {DEVICE_NAME}` to reuse this).
NEXUS_PROJECT = "schmetterling-effect-test"
P_VALUES = [0.0, 0.05]  # measurement rate threshold
L_MAX = 25  # max layers
N_VALUES = [8]  # system sizes (even, 8–16)
N_REALIZATIONS = 240  # single-shot mode: independent circuits per (N, p, L)
N_QUBITS_BUDGET = (
    48  # single_shot_tiled mode: total physical qubits to tile realizations
)
# into (build_parallel_otoc_circuit); must be >= max(N_VALUES)
TILED_MAX_DEPTH = 60  # single_shot_tiled mode: cap on sequential depth-rows per tiled
# circuit. Each row adds real elapsed gate time on the same
# physical qubits (via Reset-and-reuse) even though the quantum
# STATE is refreshed — too many rows and later realizations start
# approaching the device's T1/T2 and stop being clean samples.
# None disables the cap; raise ValueError instead of silently
# building an overly-deep circuit if exceeded.
TILED_MAX_CLASSICAL_BITS = 64  # single_shot_tiled mode: cap on total classical bits per
# tiled circuit (width scratch bits + one probe bit per
# realization in that chunk). Real devices cap classical
# register size independent of qubit count/depth (H2-2's
# limit is 64) — since bit count scales directly with
# realizations-per-chunk, this is often TIGHTER than
# TILED_MAX_DEPTH, not redundant with it. None disables.
TILED_MAX_SU4_PER_CHUNK = 1500  # single_shot_tiled mode: cap on total SU4/TK2 gates per
# tiled circuit. Scales with L (unlike the other two caps) —
# Nexus's compile service can cancel a compile job with a
# server-side timeout once there are too many generic
# two-qubit unitaries to KAK-decompose; empirically 1624
# compiled fine, 3248 got cancelled, so this default sits
# with margin below the observed failure point. Not a
# documented hard limit, just a measured cliff — tune if
# your own runs fail/succeed at different sizes. None disables.
N_SHOTS = 100  # multi-shot mode: shots per fixed circuit
N_SHOTS_FULL_OTOC = 500  # shots per circuit for the full forward+backward OTOC sweep
N_CIRCUITS = 5  # multi-shot mode: gate-angle realizations
N_INIT_STATES = 5  # multi-shot mode: initial-state realizations
BASE_SEED = 1917  # root seed for parallel schemes (single seed)
BASE_SEEDS: list[int] = [1962]  # seeds for sequential sweeps — add more to seed-average
MEAS_SEED = 7  # multi-shot mode: root seed for measurement stream
RECORD_EVERY = 4  # record at L = 5, 10, 15, 20
L_VALUES: list[int] | None = [
    2,
    4,
    8,
    12,
]  # explicit L list; overrides L_MAX + RECORD_EVERY when set
TOPOLOGY = "brickwall"  # "brickwall" or "bowtie"
INIT_OPS: list[str] | None = None  # None → ["I","X","Y","Z"]; e.g. ["I","X"] for bowtie
PERT_OP = "X"  # perturbation gate
PROBE_ANGLE = 0.0  # Ry(π/2) on probe qubit
OPTIMISATION_LEVEL = 0  # pytket compilation level: 0 = minimal, 1 = light, 2 = full
NEXUS_BATCH = (
    False  # True = attempt_batching (requires org feature); False = standard queue
)
# DEVICE_NAME = "H2-1LE"  # uncomment for Quantinuum hardware

# ── Noise model (local AerBackend only; ignored when DEVICE_NAME is set) ──────
# All None → ideal (noiseless) simulation.  Mix and match as needed.
#
# Quantinuum device defaults (as reported for the target device) and how they
# map onto build_aer_noise_model's parameters:
#
#   p1                        = 7.3e-05  → NOISE_P1Q directly (1q depolarizing rate)
#   p2                        = 1.29e-03 → NOISE_P2Q directly (2q depolarizing rate)
#   p_meas (0 outcome)        = 9.0e-04  ┐ build_aer_noise_model only supports a
#   p_meas (1 outcome)        = 1.8e-03  ┘ SYMMETRIC readout bitflip (NOISE_PM), so
#                                          these are averaged below (1.35e-03) as an
#                                          approximation — the 1↔0 asymmetry (the
#                                          real device is ~2x more likely to
#                                          misreport a 1 as a 0) is lost. For an
#                                          exact match, build a qiskit ReadoutError
#                                          per qubit with [[1-p0,p0],[p1,1-p1]]
#                                          directly instead of going through `pm`.
#
# NOT representable in this local Aer model at all (would need a custom
# NoiseModel, or better, Quantinuum's own noisy emulator device e.g. "H2-1E"
# via QuantinuumBackend instead of AerBackend):
#   p_crosstalk_meas           = 8.8e-06  — spectator fault during measurement
#   p_init                     = 4.0e-05  — qubit initialization fault
#   p_crosstalk_init           = 9.6e-06  — spectator fault during initialization
#   p1_emission_ratio          = 0.32     — fraction of p1 due to spontaneous
#   p2_emission_ratio          = 0.59       emission (T1-like) vs. dephasing;
#                                            NOISE_T1/T2 use a plain exponential
#                                            thermal_relaxation_error and this
#                                            sheet gives no T1/T2 in seconds to
#                                            plug in directly — leave them None
#                                            unless you derive T1 from
#                                            p1 * p1_emission_ratio and t_gate_1q.
#   quadratic_dephasing_rate   = 0.043    ┐ Quantinuum's coherent dephasing model
#   linear_dephasing_rate      = 2.8e-03  │ (rate depends non-linearly on idle
#   coherent_to_incoherent_factor = 2.0   ┘ time) — no equivalent in a constant-T2
#                                            exponential model.
NOISE_P1Q = 7.3e-05  # single-qubit gate depolarizing error rate (p1)
NOISE_P2Q = 1.29e-03  # two-qubit gate depolarizing error rate (p2)
NOISE_PM = (
    1.35e-03  # readout bitflip probability (mean of p_meas(0)=9e-4, p_meas(1)=1.8e-3)
)
NOISE_T1 = (
    None  # longitudinal relaxation time T1 (seconds) — not given by this spec sheet
)
NOISE_T2 = None  # transverse relaxation time  T2 (seconds; must be <= 2*T1) — see above
NOISE_T_1Q = 50e-9  # single-qubit gate duration (seconds)
NOISE_T_2Q = 300e-9  # two-qubit gate duration    (seconds)


FIGURE_DIR = pathlib.Path(__file__).parent.parent / "figs"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH_C = str(FIGURE_DIR / f"C_vs_t_{DATE}.png")
FIGURE_PATH_CV = str(FIGURE_DIR / f"cv_vs_t_{DATE}.png")
FIGURE_PATH_TM = str(FIGURE_DIR / f"qc_vs_tm_{DATE}.png")
FIGURE_PATH_LIGHTCONE = str(FIGURE_DIR / f"lightcone_heatmap_{DATE}.png")
FIGURE_PATH_FULL_OTOC = str(FIGURE_DIR / f"full_otoc_heatmap_{DATE}.png")
PARTIALS_DIR = "partials"  # directory for intermediate files

# ── Transfer-matrix overlay  (set TM_NUM_QUBITS to None to skip entirely) ─────
TM_NUM_QUBITS: int | None = 8  # None → TM disabled; set to an int to enable
TM_TF: int = 25  # number of time steps to evolve
TM_PERT_SITE: int = 0  # perturbation qubit index
TM_PROBE_SITE: int = 3  # probe qubit index
TM_P_VALUES: list[float] | None = (
    None  # None → use P_VALUES; or set explicitly e.g. [0.0, 0.02]
)
TM_BC_TYPE: str = "open"  # "open" or "periodic"

# OTOC sweep parameters (used in sequential mode and OTOC-specific runs)
OTOC_W = N_VALUES[-1]
OTOC_P_VALUES = [0.02, 1.0]

# ══════════════════════════════════════════════════════════════════════════════
# Shared infrastructure
# ══════════════════════════════════════════════════════════════════════════════


def _L_steps():
    if L_VALUES is not None:
        return sorted(L_VALUES)
    return list(range(RECORD_EVERY, L_MAX + 1, RECORD_EVERY))


def _circuit_builder():
    if TOPOLOGY == "bowtie":
        return build_bowtie_otoc_circuit
    if TOPOLOGY == "brickwall":
        return build_brickwall_otoc_circuit
    raise ValueError(f"Unknown TOPOLOGY {TOPOLOGY!r}. Choose 'brickwall' or 'bowtie'.")


def _build_backend(n_qubits: int | None = None, device_name: str | None = None):
    """
    Construct and return the backend from device_name / noise config.

    Quantinuum hardware : device_name set → QuantinuumBackend (noise config ignored)
    Local noiseless     : device_name = None, NOISE_* = None → AerBackend()
    Local noisy         : device_name = None, any NOISE_* set → AerBackend(noise_model=...)

    device_name : Quantinuum device to target. SAFETY: defaults to None
        (local simulation) regardless of the module-level DEVICE_NAME
        constant — this function never reads that global. Every caller
        that wants real hardware must pass device_name explicitly (e.g.
        device_name=DEVICE_NAME, or wire a --device-name CLI flag through),
        so an ad hoc call can never silently hit hardware just because
        DEVICE_NAME happens to be set somewhere in the module.
    n_qubits : how many qubits to register noise on; defaults to max(N_VALUES).
        Callers that run wider circuits than N_VALUES (e.g. single_shot_tiled
        mode, whose tiled circuits use N_QUBITS_BUDGET physical qubits) must
        pass that larger width explicitly, or pytket's compiler will reject
        the circuit with "More virtual qubits than physical".
    """
    if device_name is not None:
        return QuantinuumBackend(device_name)

    from pytket.extensions.qiskit import AerBackend

    _n_qubits = n_qubits if n_qubits is not None else max(N_VALUES)

    if any(v is not None for v in (NOISE_P1Q, NOISE_P2Q, NOISE_PM, NOISE_T1, NOISE_T2)):
        noise_model = build_aer_noise_model(
            p1q=NOISE_P1Q,
            p2q=NOISE_P2Q,
            pm=NOISE_PM,
            T1=NOISE_T1,
            T2=NOISE_T2,
            t_gate_1q=NOISE_T_1Q,
            t_gate_2q=NOISE_T_2Q,
            n_qubits=_n_qubits,
        )
        print(
            f"Noisy AerBackend  p1q={NOISE_P1Q}  p2q={NOISE_P2Q}  pm={NOISE_PM}  "
            f"T1={NOISE_T1}  T2={NOISE_T2}  t_1q={NOISE_T_1Q}  t_2q={NOISE_T_2Q}"
        )
        return AerBackend(noise_model=noise_model)

    return AerBackend()


def estimate_hqc_cost(mode: str = "single_shot", verbose: bool = True) -> float:
    """
    Estimates the total HQC cost for a full experiment without running it.

    Formula per circuit (Quantinuum billing):
        HQC = 5 + (N1q + 10*N2q + 5*Nm) / 5000 * Ns

    where N1q = single-qubit gates, N2q = two-qubit gates, Nm = measurements,
    Ns = shots per circuit, all counted after compilation to native gates.

    Compilation estimates for a Haar SU(4) gate (KAK decomposition):
        N2q per gate  ≈ 3  (three ZZPhase/CNOT-equivalent native gates)
        N1q per gate  ≈ 6  (surrounding single-qubit rotations)

    Parameters
    ----------
    mode    : "single_shot" or "multi_shot" — must match intended run mode.
    verbose : Print a breakdown table when True.

    Returns
    -------
    Total estimated HQC as a float.
    """
    N2Q_PER_SU4 = 3
    N1Q_PER_SU4 = 6

    L_values = _L_steps()
    total_hqc = 0.0
    rows = []

    for W in N_VALUES:
        for p in P_VALUES:
            for L in L_values:
                # SU(4) gate count per half-circuit (U or U†)
                L_even = (L + 1) // 2
                L_odd = L // 2
                n_su4_half = L_even * (W // 2) + L_odd * (W // 2 - 1)
                n_su4 = 2 * n_su4_half  # U + U†

                N2q = N2Q_PER_SU4 * n_su4
                N1q = N1Q_PER_SU4 * n_su4 + (W - 1)  # W-1 state-prep Paulis

                # Measurements: U mid-circuit + U† mid-circuit + pert (only if
                # pert_op == "measure") + final probe.  A Pauli perturbation adds
                # no measurement, so perturbed and unperturbed share the same Nm.
                n_meas = min(round(p * W * L), W * max(L - 1, 0))
                n_pert_meas = 1 if PERT_OP == "measure" else 0
                Nm = 2 * n_meas + n_pert_meas + 1  # same for both variants

                if mode == "single_shot":
                    n_circuits_point = 2 * N_REALIZATIONS
                    Ns = 1
                else:
                    n_circuits_point = 2 * N_CIRCUITS * N_INIT_STATES
                    Ns = N_SHOTS

                hqc_point = (
                    5 + (N1q + 10 * N2q + 5 * Nm) / 5000 * Ns
                ) * n_circuits_point
                total_hqc += hqc_point

                rows.append((W, p, L, N1q, N2q, Nm, n_circuits_point, Ns, hqc_point))

    if verbose:
        print(f"\nHQC estimate  (mode={mode!r})")
        print(
            f"{'N':>4} {'p':>5} {'L':>4} {'N1q':>6} {'N2q':>6} {'Nm':>5} {'#circs':>7} {'Ns':>4} {'HQC':>10}"
        )
        print("-" * 62)
        for W, p, L, N1q, N2q, Nm, n_c, Ns, hqc in rows:
            print(
                f"{W:>4} {p:>5.2f} {L:>4} {N1q:>6} {N2q:>6} {Nm:>5} {n_c:>7} {Ns:>4} {hqc:>10.1f}"
            )
        print("-" * 62)
        print(f"{'TOTAL':>50}  {total_hqc:>10.1f}")
        print()

    return total_hqc


def _compute_partial(
    N_values: list[int],
    p_values: list[float],
    L_steps: list[int],
    realizations: list[tuple[int, int]],
    device_name: str | None = None,
) -> dict[tuple, np.ndarray]:
    """
    Core compute kernel shared by all four partition schemes.

    Runs the perturbed and unperturbed echo circuits for every combination of
    (N, p, L) in the given lists, but only for the specified disorder
    realizations (c_idx, i_idx pairs).

    Parameters
    ----------
    N_values      : system sizes to process in this partition
    p_values      : measurement rates to process in this partition
    L_steps       : layer counts to process in this partition
    realizations  : list of (c_idx, i_idx) pairs to iterate over
    device_name   : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.

    Returns
    -------
    dict keyed (N, p, L) → 1-D np.ndarray of per-realization F_vals,
    where F_val = pert_mean - unpert_mean (one entry per realization).
    """
    backend = _build_backend(device_name=device_name)
    add_bar = isinstance(backend, QuantinuumBackend)
    meas_base = MEAS_SEED + 1
    partial = defaultdict(list)

    for W in N_values:
        pert_site = W // 2
        probe_site = 0

        for c_idx, i_idx in realizations:
            circuit_seed = BASE_SEED * 10_000 + c_idx
            init_seed_val = BASE_SEED * 10_000_000 + c_idx * 1_000 + i_idx
            meas_seed_val = meas_base * 10_000_000 + c_idx * 1_000 + i_idx

            for p in p_values:
                for L in L_steps:
                    shared = dict(
                        L=L,
                        W=W,
                        pert_site=pert_site,
                        pert_op=PERT_OP,
                        probe_site=probe_site,
                        probe_angle=PROBE_ANGLE,
                        p=p,
                        seed=circuit_seed,
                        init_seed=init_seed_val,
                        meas_seed=meas_seed_val,
                        add_barrier=add_bar,
                        init_ops=INIT_OPS,
                    )
                    _build = _circuit_builder()
                    qc_pert = _build(**shared, unperturbed=False)
                    qc_unpert = _build(**shared, unperturbed=True)
                    probe_bit = qc_pert.n_bits - 1

                    sp = _run_circuit_backend(qc_pert, backend, N_SHOTS)
                    su = _run_circuit_backend(qc_unpert, backend, N_SHOTS)

                    out_p = 1 - 2 * sp[:, probe_bit].astype(int)
                    out_u = 1 - 2 * su[:, probe_bit].astype(int)
                    F_val = float(out_p.mean() - out_u.mean())

                    partial[(W, p, L)].append(F_val)
                    print(
                        f"N={W:2d}  p={p:.2f}  L={L:3d}  "
                        f"c={c_idx:2d}  i={i_idx:2d}  dF={F_val:.4f}"
                    )

    return {k: np.array(v) for k, v in partial.items()}


# ── File I/O helpers ──────────────────────────────────────────────────────────


def _save_partial(data: dict, path: str) -> None:
    """Pickle a partial result dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"Saved → {path}  ({len(data)} (N,p,L) keys)")


def _load_partial(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _merge_partials(pattern: str) -> dict[tuple, np.ndarray]:
    """
    Load all pickle files matching glob pattern and concatenate their F_val
    arrays along the realization axis.

    Returns dict keyed (N, p, L) → full concatenated np.ndarray.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No partial files matching {pattern!r}")
    print(f"Merging {len(files)} partial files …")
    merged = defaultdict(list)
    for fname in files:
        partial = _load_partial(fname)
        for key, vals in partial.items():
            merged[key].extend(vals.tolist())
    return {k: np.array(v) for k, v in merged.items()}


def _merged_to_all_stats(merged: dict) -> tuple:
    """
    Convert the flat (N, p, L)-keyed merged dict into the nested
    all_stats[N] = compute_F({p: {L: array}}) structure used by plot_results.

    Returns (all_stats, N_values, p_values, L_values) sorted.
    """
    raw_by_N: dict[int, dict] = defaultdict(lambda: defaultdict(dict))
    for (N, p, L), vals in merged.items():
        raw_by_N[N][p][L] = vals

    N_values = sorted(raw_by_N)
    p_values = sorted({p for (N, p, L) in merged})
    L_values = sorted({L for (N, p, L) in merged})

    all_stats = {N: compute_C_t(dict(raw_by_N[N])) for N in N_values}
    return all_stats, N_values, p_values, L_values


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 1  — Partition by N
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_1(n_idx: int, device_name: str | None = None) -> None:
    """
    Run the full disorder average for ONE system size N = N_VALUES[n_idx].
    Covers all p_values, all L_steps, all N_CIRCUITS × N_INIT_STATES realizations.

    Parallel usage:  one nohup process per n_idx in range(len(N_VALUES)).
    Degree of parallelism: len(N_VALUES) = 5.

    device_name : None (default) = local simulation; pass explicitly for hardware.

    Launch:
      for i in 0 1 2 3 4; do
        nohup .butterfly/bin/python run_disorder_avg.py --scheme 1 --job-id $i
              > partials/s1_$i.log 2>&1 &
      done
    """
    N = N_VALUES[n_idx]
    realizations = [(c, i) for c in range(N_CIRCUITS) for i in range(N_INIT_STATES)]

    print(f"\n=== Scheme 1 | job {n_idx} | N={N} ===")
    print(f"  realizations = {len(realizations)}")
    print(f"  L_steps = {_L_steps()}")

    partial = _compute_partial(
        [N], P_VALUES, _L_steps(), realizations, device_name=device_name
    )
    _save_partial(partial, f"{PARTIALS_DIR}/s1_N{N}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 2  — Partition by c_idx (angle realization)
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_2(c_idx: int, device_name: str | None = None) -> None:
    """
    Run ALL N_values and ALL p_values, but only for ONE angle seed (c_idx),
    iterating over all N_INIT_STATES init-state seeds.

    Parallel usage: one process per c_idx in range(N_CIRCUITS).
    Degree of parallelism: N_CIRCUITS = 20.

    device_name : None (default) = local simulation; pass explicitly for hardware.

    Launch:
      for i in $(seq 0 19); do
        nohup .butterfly/bin/python run_disorder_avg.py --scheme 2 --job-id $i
              > partials/s2_c$i.log 2>&1 &
      done
    """
    realizations = [(c_idx, i) for i in range(N_INIT_STATES)]

    print(f"\n=== Scheme 2 | c_idx={c_idx} ===")
    print(
        f"  realizations (c_idx={c_idx}, i_idx=0..{N_INIT_STATES-1}): {len(realizations)}"
    )

    partial = _compute_partial(
        N_VALUES, P_VALUES, _L_steps(), realizations, device_name=device_name
    )
    _save_partial(partial, f"{PARTIALS_DIR}/s2_c{c_idx}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 3  — Partition by (c_idx, i_idx)  — finest grain
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_3(c_idx: int, i_idx: int, device_name: str | None = None) -> None:
    """
    Run ALL N_values and ALL p_values for exactly ONE disorder realization
    (c_idx, i_idx).  Each job submits len(N_VALUES)*len(P_VALUES)*len(L_steps)*2
    circuits.

    Parallel usage: one process per (c_idx, i_idx) pair.
    Degree of parallelism: N_CIRCUITS × N_INIT_STATES = 400.

    device_name : None (default) = local simulation; pass explicitly for hardware.

    Launch:
      for c in $(seq 0 19); do
        for i in $(seq 0 19); do
          nohup .butterfly/bin/python run_disorder_avg.py \\
                --scheme 3 --job-id $c --secondary-id $i \\
                > partials/s3_c${c}_i${i}.log 2>&1 &
        done
      done
    """
    n_circuits_per_job = len(N_VALUES) * len(P_VALUES) * len(_L_steps()) * 2
    print(f"\n=== Scheme 3 | c_idx={c_idx}  i_idx={i_idx} ===")
    print(f"  circuits in this job = {n_circuits_per_job}")

    partial = _compute_partial(
        N_VALUES, P_VALUES, _L_steps(), [(c_idx, i_idx)], device_name=device_name
    )
    _save_partial(partial, f"{PARTIALS_DIR}/s3_c{c_idx}_i{i_idx}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 4  — Partition by (N, p)  — grid point
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_4(n_idx: int, p_idx: int, device_name: str | None = None) -> None:
    """
    Run ALL realizations (N_CIRCUITS × N_INIT_STATES) for exactly ONE (N, p)
    grid point.  Each job independently produces a full disorder average for
    one cell of the C(N, p, T) output matrix.

    Parallel usage: one process per (n_idx, p_idx) pair.
    Degree of parallelism: len(N_VALUES) × len(P_VALUES) = 25.

    device_name : None (default) = local simulation; pass explicitly for hardware.

    Launch:
      for n in $(seq 0 4); do
        for p in $(seq 0 4); do
          nohup .butterfly/bin/python run_disorder_avg.py \\
                --scheme 4 --job-id $n --secondary-id $p \\
                > partials/s4_n${n}_p${p}.log 2>&1 &
        done
      done
    """
    N = N_VALUES[n_idx]
    p = P_VALUES[p_idx]
    realizations = [(c, i) for c in range(N_CIRCUITS) for i in range(N_INIT_STATES)]

    print(f"\n=== Scheme 4 | N={N}  p={p:.2f} ===")
    print(f"  realizations = {len(realizations)}")

    partial = _compute_partial(
        [N], [p], _L_steps(), realizations, device_name=device_name
    )
    _save_partial(partial, f"{PARTIALS_DIR}/s4_N{N}_p{p_idx}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Merge  — combine all partial files for a given scheme and plot
# ══════════════════════════════════════════════════════════════════════════════


def merge_and_plot(scheme: int) -> None:
    """
    Load all partial result files for the given scheme, concatenate their
    F_val arrays, compute F / var / CV, and produce the output figures.

    Run after all parallel jobs have completed:
      python run_disorder_avg.py --scheme 2 --merge
    """
    pattern = f"{PARTIALS_DIR}/s{scheme}_*.pkl"
    merged = _merge_partials(pattern)
    all_stats, N_values, p_values, L_values = _merged_to_all_stats(merged)

    print(f"\nMerged shape:")
    print(f"  N_values = {N_values}")
    print(f"  p_values = {p_values}")
    print(f"  L_values = {L_values}")
    total_realizations = len(next(iter(merged.values())))
    print(f"  realizations per (N, p, L) = {total_realizations}")

    plot_results(
        all_stats,
        L_values,
        p_values,
        N_values,
        figure_path_C=FIGURE_PATH_C,
        figure_path_cv=FIGURE_PATH_CV,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Transfer-matrix overlay plot
# ══════════════════════════════════════════════════════════════════════════════


def plot_tm_overlay(
    all_stats: dict,
    L_values: list[int],
    p_values: list[float],
    tm_results: dict[float, tuple[np.ndarray, np.ndarray]],
    tm_N: int,
    figure_path: str,
) -> None:
    """
    Overlay QC disorder-average results with exact transfer-matrix OTOC curves.

    Produces one subplot per p value in p_values.  Each subplot shows QC
    C(T) errorbars for all N, with the series matching tm_N drawn thicker.
    If a TM result exists for that p in tm_results, its exact C(t) curve is
    drawn as a solid crimson line on the same axes.

    Parameters
    ----------
    all_stats : dict
        Output of compute_C_t, keyed all_stats[N]["C" | "se"][p][L].
    L_values : list[int]
        Recorded layer counts (x-axis positions for QC errorbars).
    p_values : list[float]
        Measurement rates swept in the QC experiment.
    tm_results : dict[float, (t_arr, C_arr)]
        Keyed by p; each value is the (t, C) pair from ButterflyTransferMatrix.
    tm_N : int
        System size (num_qubits) used for the TM runs.
    figure_path : str
        Output file path for the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N_values = sorted(all_stats)
    n_panels = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    N_colors = plt.cm.viridis(np.linspace(0, 1, max(len(N_values), 1)))

    for ax, p in zip(axes, p_values):
        for n_idx, N in enumerate(N_values):
            if N not in all_stats or p not in all_stats[N]["C"]:
                continue
            qc_C = [all_stats[N]["C"][p][L] for L in L_values]
            qc_se = [all_stats[N]["se"][p][L] for L in L_values]
            lw = 2.0 if N == tm_N else 0.8
            ax.errorbar(
                [L - 1 for L in L_values],
                qc_C,
                yerr=qc_se,
                fmt="o",
                capsize=3,
                capthick=lw,
                elinewidth=lw,
                color=N_colors[n_idx],
                label=f"QC  N={N}",
            )

        if p in tm_results:
            t_tm, C_tm = tm_results[p]
            ax.plot(
                t_tm,
                C_tm,
                "-",
                lw=2,
                color="crimson",
                label=f"TM  N={tm_N}",
                zorder=5,
            )

        ax.set_xlabel("t  (layers)")
        ax.set_ylabel(r"$C_t$")
        ax.set_title(f"p = {p:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"QC (errorbars) vs exact transfer matrix (line)  —  N$_{{TM}}$={tm_N}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Light-cone sweep  — probe site x vs time t, restricted to the causal cone
# ══════════════════════════════════════════════════════════════════════════════


def sweep_lightcone_x_and_t(
    W: int | None = None,
    p_values: list[float] | None = None,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    n_shots: int | None = None,
    device_name: str | None = None,
) -> dict[int, dict]:
    """
    Disorder-averaged OTOC sweep over probe position x and time L, restricted
    to the causal light cone spreading from a perturbation at the middle
    qubit.

    The circuit is nearest-neighbour brickwall: each layer moves the causal
    boundary by at most one site, so after L layers of U† only probe sites
    with |x| <= L (x = probe_site - pert_site) can carry any signal from the
    perturbation. Sites outside that cone are skipped entirely rather than
    simulated and discarded.

    Parameters
    ----------
    W          : system size; defaults to N_VALUES[0].
    p_values   : measurement rates to sweep; defaults to P_VALUES.
    L_values   : recorded layer counts (time axis); defaults to _L_steps().
    pert_site  : perturbation qubit; defaults to W // 2 (the middle site).
    n_shots    : shots per circuit; defaults to N_SHOTS.
    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.

    Returns
    -------
    stats_by_x : dict[int, dict]
        x → compute_C_t output, i.e. stats_by_x[x]["C"][p][L], ["se"][p][L], …
        x = 0 (the perturbation site itself) is never probed and is absent.
    """
    W = W if W is not None else N_VALUES[0]
    p_values = p_values if p_values is not None else P_VALUES
    L_steps = L_values if L_values is not None else _L_steps()
    pert_site = pert_site if pert_site is not None else W // 2
    shots = n_shots if n_shots is not None else N_SHOTS

    backend = _build_backend(n_qubits=W, device_name=device_name)
    add_bar = isinstance(backend, QuantinuumBackend)
    meas_base = MEAS_SEED + 1

    raw_by_x: dict[int, dict] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    print(
        f"\n=== Light-cone sweep | W={W}  pert_site={pert_site}  L_values={L_steps} ==="
    )

    for c_idx in range(N_CIRCUITS):
        circuit_seed = BASE_SEED * 10_000 + c_idx
        for i_idx in range(N_INIT_STATES):
            init_seed_val = BASE_SEED * 10_000_000 + c_idx * 1_000 + i_idx
            meas_seed_val = meas_base * 10_000_000 + c_idx * 1_000 + i_idx

            for L in L_steps:
                x_max = min(L, pert_site, W - 1 - pert_site)
                xs = [x for x in range(-x_max, x_max + 1) if x != 0]

                for p in p_values:
                    for x in xs:
                        probe_site = pert_site + x
                        shared = dict(
                            L=L,
                            W=W,
                            pert_site=pert_site,
                            pert_op=PERT_OP,
                            probe_site=probe_site,
                            probe_angle=PROBE_ANGLE,
                            p=p,
                            seed=circuit_seed,
                            init_seed=init_seed_val,
                            meas_seed=meas_seed_val,
                            add_barrier=add_bar,
                            init_ops=INIT_OPS,
                        )
                        _build = _circuit_builder()
                        qc_pert = _build(**shared, unperturbed=False)
                        qc_unpert = _build(**shared, unperturbed=True)
                        probe_bit = qc_pert.n_bits - 1

                        sp, su = _run_circuits(
                            [qc_pert, qc_unpert],
                            backend,
                            shots,
                            nexus_project=NEXUS_PROJECT,
                            optimisation_level=OPTIMISATION_LEVEL,
                            use_batch=NEXUS_BATCH,
                        )

                        out_p = 1 - 2 * sp[:, probe_bit].astype(int)
                        out_u = 1 - 2 * su[:, probe_bit].astype(int)
                        d_r = float(out_u.mean() - out_p.mean())

                        raw_by_x[x][p][L].append(d_r)
                        print(
                            f"x={x:+3d}  p={p:.2f}  L={L:3d}  "
                            f"c={c_idx:2d}  i={i_idx:2d}  d={d_r:.4f}"
                        )

    return {
        x: compute_C_t(
            {p: {L: np.array(v) for L, v in Ld.items()} for p, Ld in pd.items()}
        )
        for x, pd in raw_by_x.items()
    }


def sweep_lightcone_probe_all(
    W: int | None = None,
    p_values: list[float] | None = None,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    n_shots: int | None = None,
    n_realizations: int = 250,
    base_seed: int = 42,
    device_name: str | None = None,
    checkpoint_path: str | None = None,
    anchor_parity_at_perturbation: str | None = None,
) -> dict[int, dict]:
    """
    Disorder-averaged OTOC sweep over probe position x and time L, using the
    "measure every site" circuit (build_*_lightcone_probe_all_circuit) —
    ONE (pert, unpert) circuit pair per (p, L, realization) yields every x
    at once, instead of one pair per (x, p, L, realization) like
    sweep_lightcone_x_and_t. State prep is restricted to I/X (see that
    circuit's docstring for the per-site flip-correction protocol), so
    unlike sweep_lightcone_x_and_t this does NOT skip cells outside the
    nominal causal cone — every x in range gets a value every L (the cone
    boundary shows up as that value being ~0, not as a missing cell).

    Parameters
    ----------
    W          : system size; defaults to N_VALUES[0].
    p_values   : measurement rates to sweep; defaults to P_VALUES.
    L_values   : recorded layer counts (time axis); defaults to _L_steps().
    pert_site  : perturbation qubit; defaults to W // 2 (the middle site).
    n_shots    : shots per circuit; defaults to N_SHOTS.
    n_realizations : disorder realizations per (p, L) point.
    base_seed  : root seed for the per-realization seed formula (independent
        per (L, realization), matching sweep_single_shot_disorder_tiled's
        deterministic-seed convention).
    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.
    checkpoint_path : if given, pickles {(W, base_seed, p, L): {x: [d_r, ...]}}
        to this path after EVERY (p, L) point finishes — the natural atomic
        unit here, since one point's n_realizations circuit-pairs either
        fully complete or don't. Points already present (with the expected
        n_realizations count, for every x) are skipped rather than
        re-submitted, so adding a new p value later (e.g. sweeping in
        p=0.05 after already having p=0.0 and p=0.01 checkpointed) is
        additive — existing points are loaded and reused, not recomputed.
        Matches sweep_single_shot_disorder_tiled's checkpoint_path
        convention (same _load_partial/_save_partial helpers). Delete the
        file to force a full fresh run.
    anchor_parity_at_perturbation : None (default), "even", or "odd" — passed
        through to build_brickwall_lightcone_probe_all_circuit (brickwall
        topology only — ignored under TOPOLOGY="bowtie", which doesn't have
        an analogous parity concept). See that function's docstring: forces
        the layer immediately adjacent to the perturbation to always be the
        CHOSEN parity regardless of L ("even" couples pert_site to x=+1
        first every time, "odd" to x=-1 first every time), instead of it
        flipping unpredictably between even/odd depending on whether L is
        odd or even. Note this does NOT produce full x<->-x symmetry (a
        single perturbed qubit can only bond with one neighbour per layer,
        so one direction always gets a one-layer head start) — it makes
        the resulting left/right bias CONSISTENT across every L instead of
        alternating, so the light-cone is offset but that offset is
        centered/stable rather than erratic. Use a DIFFERENT checkpoint_path
        for each distinct value of this argument — the checkpoint key
        doesn't encode it, so mixing runs made with different settings
        would silently combine data from different circuit structures
        under the same key.

    Returns
    -------
    stats_by_x : dict[int, dict]
        x → compute_C_t output, i.e. stats_by_x[x]["C"][p][L], ["se"][p][L], …
        x = 0 (the perturbation site itself) is never probed and is absent
        — same shape as sweep_lightcone_x_and_t's return, so
        plot_lightcone_heatmap works unmodified on this function's output.
    """
    W = W if W is not None else N_VALUES[0]
    p_values = p_values if p_values is not None else P_VALUES
    L_steps = L_values if L_values is not None else _L_steps()
    pert_site = pert_site if pert_site is not None else W // 2
    shots = n_shots if n_shots is not None else N_SHOTS

    backend = _build_backend(n_qubits=W, device_name=device_name)
    add_bar = isinstance(backend, QuantinuumBackend)
    _build = (
        build_bowtie_lightcone_probe_all_circuit
        if TOPOLOGY == "bowtie"
        else build_brickwall_lightcone_probe_all_circuit
    )

    checkpoint: dict[tuple, dict[int, list]] = {}
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = _load_partial(checkpoint_path)
        print(f"Loaded checkpoint ← {checkpoint_path}  ({len(checkpoint)} point(s) already done)")

    raw_by_x: dict[int, dict] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    print(
        f"\n=== Light-cone (probe-all) sweep | W={W}  pert_site={pert_site}  "
        f"L_values={L_steps}  n_realizations={n_realizations} ==="
    )

    for p in p_values:
        for L in L_steps:
            ckpt_key = (W, base_seed, p, L)
            cached = checkpoint.get(ckpt_key)
            if (
                checkpoint_path is not None
                and cached is not None
                and all(len(v) == n_realizations for v in cached.values())
            ):
                for x, vals in cached.items():
                    raw_by_x[x][p][L].extend(vals)
                print(f"p={p:.2f}  L={L:3d}  [checkpointed, skipping submission]  "
                      f"n_realizations={n_realizations}")
                continue

            point_by_x: dict[int, list] = defaultdict(list)
            for r in range(n_realizations):

                def _s(axis, r=r, L=L, s=base_seed):
                    return (s + axis) * 1_000_000_000 + L * n_realizations + r

                shared = dict(
                    L=L,
                    W=W,
                    pert_site=pert_site,
                    pert_op=PERT_OP,
                    p=p,
                    seed=_s(0),
                    init_seed=_s(1),
                    meas_seed=_s(2),
                    meas_seed_ud=_s(3),
                    add_barrier=add_bar,
                )
                if TOPOLOGY != "bowtie":
                    shared["anchor_parity_at_perturbation"] = anchor_parity_at_perturbation
                qc_pert = _build(**shared, unperturbed=False)
                qc_unpert = _build(**shared, unperturbed=True)

                sp, su = _run_circuits(
                    [qc_pert, qc_unpert],
                    backend,
                    shots,
                    nexus_project=NEXUS_PROJECT,
                    optimisation_level=OPTIMISATION_LEVEL,
                    use_batch=NEXUS_BATCH,
                )

                # Bits 1..W are the per-site readouts (bit 0 is scratch) —
                # already flip-corrected in-circuit, so 1-2*bit is directly
                # comparable across sites regardless of each site's own I/X
                # initial condition.
                out_p = 1 - 2 * sp[:, 1 : 1 + W].astype(int)
                out_u = 1 - 2 * su[:, 1 : 1 + W].astype(int)
                d_r_per_site = out_u.mean(axis=0) - out_p.mean(axis=0)

                # Unlike sweep_lightcone_x_and_t, pert_site itself IS
                # recorded here (x=0): this circuit measures every qubit,
                # including pert_site, and its own flip-corrected outcome
                # is meaningful (in fact it's the cone's tip — maximally
                # deterministic right at the perturbation, decaying/
                # spreading outward as L grows) rather than an unmeasurable
                # site the way the single-probe circuit's probe_site !=
                # pert_site constraint made it.
                for q in range(W):
                    x = q - pert_site
                    val = float(d_r_per_site[q])
                    point_by_x[x].append(val)
                    raw_by_x[x][p][L].append(val)

                print(f"p={p:.2f}  L={L:3d}  r={r:3d}  (all {W} sites from one pair)")

            if checkpoint_path is not None:
                checkpoint[ckpt_key] = dict(point_by_x)
                _save_partial(checkpoint, checkpoint_path)

    return {
        x: compute_C_t(
            {p: {L: np.array(v) for L, v in Ld.items()} for p, Ld in pd.items()}
        )
        for x, pd in raw_by_x.items()
    }


def sweep_lightcone_probe_all_tiled(
    W: int | None = None,
    p_values: list[float] | None = None,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    n_qubits_budget: int = 48,
    n_realizations: int = 250,
    base_seed: int = 42,
    optimisation_level: int = 0,
    use_batch: bool = True,
    topology: str = "brickwall",
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    anchor_parity_at_perturbation: str | None = None,
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_per_chunk: int | None = None,
    device_name: str | None = None,
    checkpoint_path: str | None = None,
    raw_results_path: str | None = None,
) -> dict[int, dict]:
    """
    Tiled analogue of sweep_lightcone_probe_all — IDENTICAL per-realization
    seed formula (same circuits, same physics, same seed→realization
    mapping), but packs many realizations per (p, L) point onto shared
    circuits via build_parallel_lightcone_probe_all_circuits_batched
    instead of submitting one (pert, unpert) circuit pair per realization.
    Built for real-hardware cost savings the same way
    sweep_single_shot_disorder_tiled is for the single-probe circuit — see
    that function's docstring for the general tiling rationale (width/depth
    packing, checkpoint/resume behaviour, the three independent chunking
    constraints).

    IMPORTANT: classical bits are consumed ~W times faster per realization
    here than in the single-probe tiled sweep — each realization needs W
    readout bits (one per qubit), not 1 — so max_classical_bits binds much
    sooner (see build_parallel_lightcone_probe_all_circuit's docstring),
    meaning noticeably smaller chunks and a smaller cost saving per point
    than sweep_single_shot_disorder_tiled gets at the same W. Still far
    fewer submissions than one pair per realization (e.g. at W=8,
    n_qubits_budget=48, max_classical_bits=64: width=6, ~7 realizations
    per chunk instead of 1 — an ~7x reduction in circuit count, not the
    ~50x the single-probe circuit gets at the same budget, but still real).

    Parameters
    ----------
    W, p_values, L_values, pert_site : same meaning/defaults as
        sweep_lightcone_probe_all.
    n_qubits_budget : total physical qubits available to tile into (e.g.
        the real device's qubit count) — width = n_qubits_budget // W.
    n_realizations  : disorder realizations per (p, L) point.
    base_seed  : root seed for the per-realization seed formula — SAME
        formula as sweep_lightcone_probe_all, so a fixed base_seed draws
        the exact same realizations regardless of whether they're tiled.
    optimisation_level, use_batch : passed through to _run_circuits.
    topology, init_ops, L_back, anchor_parity_at_perturbation : passed
        through to build_parallel_lightcone_probe_all_circuits_batched;
        anchor_parity_at_perturbation is brickwall-only (see
        build_brickwall_lightcone_probe_all_circuit), ignored under
        topology="bowtie".
    max_depth, max_classical_bits, max_su4_per_chunk : the three
        independent chunking constraints — see
        build_parallel_lightcone_probe_all_circuits_batched's docstring.
    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.
    checkpoint_path : if given, pickles {(W, base_seed, p, L): {x: [d_r, ...]}}
        to this path after EVERY (p, L) point finishes — same format and
        resume semantics as sweep_lightcone_probe_all's own checkpoint_path.
        Use a DIFFERENT file than any sweep_lightcone_probe_all (untiled)
        run or a tiled run with different n_qubits_budget/chunking
        settings — the key doesn't encode those, so mixing them would
        silently combine results from differently-submitted circuits under
        the same key (harmless statistically, since it's the identical
        underlying circuit either way, but keep them separate for clean
        provenance).
    raw_results_path : if given, pickles {(W, base_seed, p, L): record} to
        this path after EVERY point finishes, same additive/non-resume-
        affecting semantics as sweep_single_shot_disorder_tiled's own
        raw_results_path. Each record is
        {"pert_shots": [...], "unpert_shots": [...],
         "chunk_meta": {"pert": [(width, n_realizations), ...], "unpert": [...]},
         "job_info": {...}}.

    Returns
    -------
    stats_by_x : dict[int, dict] — same shape as sweep_lightcone_probe_all's
        own return (x → compute_C_t output), so plot_lightcone_heatmap
        works unmodified on this function's output too.
    """
    W = W if W is not None else N_VALUES[0]
    p_values = p_values if p_values is not None else P_VALUES
    L_steps = L_values if L_values is not None else _L_steps()
    pert_site = pert_site if pert_site is not None else W // 2

    backend = _build_backend(n_qubits=n_qubits_budget, device_name=device_name)
    add_bar = isinstance(backend, QuantinuumBackend)

    checkpoint: dict[tuple, dict[int, list]] = {}
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = _load_partial(checkpoint_path)
        print(f"Loaded checkpoint ← {checkpoint_path}  ({len(checkpoint)} point(s) already done)")

    raw_results: dict[tuple, dict] = {}
    if raw_results_path is not None and os.path.exists(raw_results_path):
        raw_results = _load_partial(raw_results_path)

    raw_by_x: dict[int, dict] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    print(
        f"\n=== Light-cone (probe-all, TILED) sweep | W={W}  pert_site={pert_site}  "
        f"L_values={L_steps}  n_realizations={n_realizations}  "
        f"n_qubits_budget={n_qubits_budget} ==="
    )

    for p in p_values:
        for L in L_steps:
            ckpt_key = (W, base_seed, p, L)
            cached = checkpoint.get(ckpt_key)
            if (
                checkpoint_path is not None
                and cached is not None
                and all(len(v) == n_realizations for v in cached.values())
            ):
                for x, vals in cached.items():
                    raw_by_x[x][p][L].extend(vals)
                print(f"p={p:.2f}  L={L:3d}  [checkpointed, skipping submission]  "
                      f"n_realizations={n_realizations}")
                continue

            realizations = []
            for r in range(n_realizations):

                def _s(axis, r=r, L=L, s=base_seed):
                    return (s + axis) * 1_000_000_000 + L * n_realizations + r

                realizations.append(dict(
                    seed=_s(0), init_seed=_s(1), meas_seed=_s(2), meas_seed_ud=_s(3),
                ))

            common = dict(
                L=L,
                W=W,
                n_qubits_budget=n_qubits_budget,
                realizations=realizations,
                pert_site=pert_site,
                pert_op=PERT_OP,
                add_barrier=add_bar,
                p=p,
                init_ops=init_ops,
                L_back=L_back,
                topology=topology,
                anchor_parity_at_perturbation=anchor_parity_at_perturbation,
                max_depth=max_depth,
                max_classical_bits=max_classical_bits,
                max_su4_per_chunk=max_su4_per_chunk,
            )
            # Each of these is a list of (circuit, width, n_realizations)
            # chunks — more than one iff max_depth/max_classical_bits/
            # max_su4_per_chunk forced a split.
            pert_chunks = build_parallel_lightcone_probe_all_circuits_batched(
                **common, unperturbed=False
            )
            unpert_chunks = build_parallel_lightcone_probe_all_circuits_batched(
                **common, unperturbed=True
            )

            tag_base = f"N{W}_p{p:.2f}_L{L:03d}_s{base_seed}_lightcone_tiled"
            all_circuits = [qc for qc, _, _ in pert_chunks] + [
                qc for qc, _, _ in unpert_chunks
            ]
            all_tags = (
                [f"{tag_base}_pert_chunk{i}" for i in range(len(pert_chunks))]
                + [f"{tag_base}_unpert_chunk{i}" for i in range(len(unpert_chunks))]
            )
            job_info: dict = {}
            all_shots = _run_circuits(
                all_circuits,
                backend,
                1,
                tags=all_tags,
                nexus_project=NEXUS_PROJECT,
                optimisation_level=optimisation_level,
                use_batch=use_batch,
                job_info_out=job_info if raw_results_path is not None else None,
            )
            pert_shots = all_shots[: len(pert_chunks)]
            unpert_shots = all_shots[len(pert_chunks):]

            means_p = np.concatenate(
                [
                    extract_parallel_lightcone_probe_all_means(shots, w, n, W)
                    for shots, (_, w, n) in zip(pert_shots, pert_chunks)
                ],
                axis=0,
            )  # (n_realizations, W)
            means_u = np.concatenate(
                [
                    extract_parallel_lightcone_probe_all_means(shots, w, n, W)
                    for shots, (_, w, n) in zip(unpert_shots, unpert_chunks)
                ],
                axis=0,
            )
            d_r = means_u - means_p  # (n_realizations, W)

            point_by_x: dict[int, list] = {}
            for q in range(W):
                x = q - pert_site
                vals = d_r[:, q].tolist()
                point_by_x[x] = vals
                raw_by_x[x][p][L].extend(vals)

            print(
                f"p={p:.2f}  L={L:3d}  [tiled: {len(pert_chunks)} chunk(s), "
                f"width={pert_chunks[0][1]}]  n_realizations={n_realizations}"
            )

            if checkpoint_path is not None:
                checkpoint[ckpt_key] = point_by_x
                _save_partial(checkpoint, checkpoint_path)

            if raw_results_path is not None:
                raw_results[ckpt_key] = {
                    "pert_shots": [np.asarray(s) for s in pert_shots],
                    "unpert_shots": [np.asarray(s) for s in unpert_shots],
                    "chunk_meta": {
                        "pert": [(w, n) for _, w, n in pert_chunks],
                        "unpert": [(w, n) for _, w, n in unpert_chunks],
                    },
                    "job_info": job_info,
                }
                _save_partial(raw_results, raw_results_path)

    return {
        x: compute_C_t(
            {p: {L: np.array(v) for L, v in Ld.items()} for p, Ld in pd.items()}
        )
        for x, pd in raw_by_x.items()
    }


def submit_lightcone_probe_all_tiled_pending(
    device_name: str,
    W: int | None = None,
    p_values: list[float] | None = None,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    n_qubits_budget: int = 48,
    n_realizations: int = 250,
    base_seed: int = 42,
    optimisation_level: int = 0,
    use_batch: bool = True,
    topology: str = "brickwall",
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    anchor_parity_at_perturbation: str | None = None,
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_per_chunk: int | None = None,
    checkpoint_path: str | None = None,
    pending_path: str = "partials/lightcone_pending_jobs.pkl",
) -> None:
    """
    Submit-and-forget analogue of sweep_lightcone_probe_all_tiled: for every
    (p, L) point not already in checkpoint_path AND not already sitting in
    pending_path, submits its compile+execute jobs (via
    _run_circuits(..., wait_for_execute=False)) and moves straight to the
    next point WITHOUT waiting for any of them to finish — for real-device
    queue times that run hours to multiple DAYS per job (observed directly:
    one job sat queued 3 days before being auto-cancelled), waiting
    sequentially for each point before submitting the next means the whole
    sweep's wall-clock time is the SUM of every point's queue time, when it
    could instead be however long the SLOWEST single point takes, with every
    job queued in parallel on Quantinuum's side from the start.

    device_name is REQUIRED (unlike sweep_lightcone_probe_all_tiled's own
    optional one) — this only makes sense for real hardware; local
    simulation is fast/synchronous already, nothing to gain by not waiting.

    Every other parameter matches sweep_lightcone_probe_all_tiled exactly
    (same seed formula, same chunking, same checkpoint_path format/resume
    semantics) — running this and sweep_lightcone_probe_all_tiled against
    the SAME checkpoint_path is safe and composable.

    pending_path : pickles {(W, base_seed, p, L): {"execute_job_id":...,
        "compile_job_id":..., "tags": [...], "chunk_meta": {"pert": [(w,n),
        ...], "unpert": [...]}, "nexus_project":..., "device_name":...}}
        after EVERY point's jobs are submitted. harvest_lightcone_probe_all_
        tiled_pending reads this file to collect results once jobs finish —
        it's the ONLY record of what's in flight, so don't delete it before
        harvesting. Entries are removed once successfully harvested (or
        found dead/cancelled — see that function).

    Prints a summary at the end: how many points were submitted, how many
    were already checkpointed (skipped), how many were already pending from
    a previous submit call (skipped, not re-submitted).
    """
    if not device_name:
        raise ValueError(
            "submit_lightcone_probe_all_tiled_pending requires device_name — "
            "local simulation is fast/synchronous already, there's nothing to "
            "gain by not waiting for it. Use sweep_lightcone_probe_all_tiled "
            "for local runs."
        )

    W = W if W is not None else N_VALUES[0]
    p_values = p_values if p_values is not None else P_VALUES
    L_steps = L_values if L_values is not None else _L_steps()
    pert_site = pert_site if pert_site is not None else W // 2

    backend = _build_backend(n_qubits=n_qubits_budget, device_name=device_name)
    add_bar = isinstance(backend, QuantinuumBackend)

    checkpoint: dict[tuple, dict[int, list]] = {}
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = _load_partial(checkpoint_path)

    pending: dict[tuple, dict] = {}
    if os.path.exists(pending_path):
        pending = _load_partial(pending_path)

    n_submitted = n_skipped_checkpointed = n_skipped_pending = 0

    print(
        f"\n=== Light-cone (probe-all, TILED, SUBMIT-ONLY) | W={W}  pert_site={pert_site}  "
        f"L_values={L_steps}  n_realizations={n_realizations}  device_name={device_name!r} ==="
    )

    for p in p_values:
        for L in L_steps:
            ckpt_key = (W, base_seed, p, L)
            cached = checkpoint.get(ckpt_key)
            if (
                checkpoint_path is not None
                and cached is not None
                and all(len(v) == n_realizations for v in cached.values())
            ):
                print(f"p={p:.2f}  L={L:3d}  [already checkpointed, skipping]")
                n_skipped_checkpointed += 1
                continue
            if ckpt_key in pending:
                print(f"p={p:.2f}  L={L:3d}  [already pending "
                      f"(execute_job_id={pending[ckpt_key]['execute_job_id']}), skipping]")
                n_skipped_pending += 1
                continue

            realizations = []
            for r in range(n_realizations):

                def _s(axis, r=r, L=L, s=base_seed):
                    return (s + axis) * 1_000_000_000 + L * n_realizations + r

                realizations.append(dict(
                    seed=_s(0), init_seed=_s(1), meas_seed=_s(2), meas_seed_ud=_s(3),
                ))

            common = dict(
                L=L,
                W=W,
                n_qubits_budget=n_qubits_budget,
                realizations=realizations,
                pert_site=pert_site,
                pert_op=PERT_OP,
                add_barrier=add_bar,
                p=p,
                init_ops=init_ops,
                L_back=L_back,
                topology=topology,
                anchor_parity_at_perturbation=anchor_parity_at_perturbation,
                max_depth=max_depth,
                max_classical_bits=max_classical_bits,
                max_su4_per_chunk=max_su4_per_chunk,
            )
            pert_chunks = build_parallel_lightcone_probe_all_circuits_batched(
                **common, unperturbed=False
            )
            unpert_chunks = build_parallel_lightcone_probe_all_circuits_batched(
                **common, unperturbed=True
            )

            tag_base = f"N{W}_p{p:.2f}_L{L:03d}_s{base_seed}_lightcone_tiled"
            all_circuits = [qc for qc, _, _ in pert_chunks] + [
                qc for qc, _, _ in unpert_chunks
            ]
            all_tags = (
                [f"{tag_base}_pert_chunk{i}" for i in range(len(pert_chunks))]
                + [f"{tag_base}_unpert_chunk{i}" for i in range(len(unpert_chunks))]
            )
            job_info: dict = {}
            _run_circuits(
                all_circuits,
                backend,
                1,
                tags=all_tags,
                nexus_project=NEXUS_PROJECT,
                optimisation_level=optimisation_level,
                use_batch=use_batch,
                job_info_out=job_info,
                wait_for_execute=False,
            )

            pending[ckpt_key] = {
                **job_info,
                "chunk_meta": {
                    "pert": [(w, n) for _, w, n in pert_chunks],
                    "unpert": [(w, n) for _, w, n in unpert_chunks],
                },
            }
            _save_partial(pending, pending_path)
            n_submitted += 1
            print(f"p={p:.2f}  L={L:3d}  [submitted: {len(pert_chunks)} pert + "
                  f"{len(unpert_chunks)} unpert chunk(s), execute_job_id="
                  f"{job_info['execute_job_id']}]")

    print(
        f"\nSubmitted {n_submitted} new point(s)  |  skipped {n_skipped_checkpointed} "
        f"already-checkpointed  |  skipped {n_skipped_pending} already-pending."
    )
    print(f"Pending jobs saved → {pending_path}")
    print("Run harvest_lightcone_probe_all_tiled_pending(...) later to collect "
          "results as jobs finish.")


def harvest_lightcone_probe_all_tiled_pending(
    checkpoint_path: str,
    raw_results_path: str | None = None,
    pending_path: str = "partials/lightcone_pending_jobs.pkl",
) -> None:
    """
    Checks every job recorded by submit_lightcone_probe_all_tiled_pending in
    pending_path, and for any that have reached a terminal Nexus status:

    - COMPLETED: fetches results via get_results_by_job_id(wait=False)
      (instant — no blocking, the job is already done), reduces them exactly
      like sweep_lightcone_probe_all_tiled does live, writes into
      checkpoint_path (and raw_results_path if given), and removes the entry
      from pending_path.
    - CANCELLED / ERROR: prints a clear warning (this point did NOT complete
      and its data is NOT recoverable — see recover_h2_2_job.py-style manual
      recovery if you want to salvage a PARTIALLY-completed job instead of a
      fully cancelled one) and removes the entry from pending_path so it
      stops being checked — it will show up as "remaining" again next time
      you run submit_lightcone_probe_all_tiled_pending, i.e. it needs
      resubmission, which this function does NOT do automatically.
    - Anything else (QUEUED / SUBMITTED / RUNNING): left in pending_path
      untouched, printed for visibility, checked again next harvest call.

    Safe to run as many times as you like (e.g. once a day) — every call is
    read-only against still-pending jobs and only mutates pending_path/
    checkpoint_path/raw_results_path for jobs that reached a terminal state
    since the last call.
    """
    if not os.path.exists(pending_path):
        print(f"No pending jobs file at {pending_path} — nothing to harvest.")
        return

    pending: dict[tuple, dict] = _load_partial(pending_path)
    if not pending:
        print(f"{pending_path} is empty — nothing to harvest.")
        return

    checkpoint: dict[tuple, dict[int, list]] = {}
    if os.path.exists(checkpoint_path):
        checkpoint = _load_partial(checkpoint_path)

    raw_results: dict[tuple, dict] = {}
    if raw_results_path is not None and os.path.exists(raw_results_path):
        raw_results = _load_partial(raw_results_path)

    n_harvested = n_dead = n_still_pending = 0
    still_pending: dict[tuple, dict] = {}

    print(f"\n=== Harvesting {len(pending)} pending job(s) from {pending_path} ===")

    for ckpt_key, info in pending.items():
        W, seed, p, L = ckpt_key
        execute_job_id = info["execute_job_id"]
        tags = info["tags"]
        chunk_meta = info["chunk_meta"]

        job_ref = qnx.jobs.get(id=execute_job_id)
        status = qnx.jobs.status(job_ref)
        st = status.status.name if hasattr(status.status, "name") else str(status.status)

        if st == "COMPLETED":
            results_by_tag, failures_by_tag = get_results_by_job_id(
                execute_job_id, tags=tags, wait=False
            )
            n_pert = len(chunk_meta["pert"])
            n_unpert = len(chunk_meta["unpert"])
            pert_shots = [results_by_tag.get(t) for t in tags[:n_pert]]
            unpert_shots = [results_by_tag.get(t) for t in tags[n_pert:]]

            kept_p_shots, kept_p_meta = [], []
            kept_u_shots, kept_u_meta = [], []
            for i, ((w, n), sp, (uw, un), su) in enumerate(
                zip(chunk_meta["pert"], pert_shots, chunk_meta["unpert"], unpert_shots)
            ):
                if sp is not None and su is not None:
                    kept_p_shots.append(sp); kept_p_meta.append((w, n))
                    kept_u_shots.append(su); kept_u_meta.append((uw, un))

            if not kept_p_shots:
                print(f"p={p:.2f}  L={L:3d}  COMPLETED but 0 usable chunk pairs "
                      f"({len(failures_by_tag)} failure(s)) — leaving pending, "
                      f"inspect manually.")
                still_pending[ckpt_key] = info
                n_still_pending += 1
                continue

            means_p = np.concatenate([
                extract_parallel_lightcone_probe_all_means(s, w, n, W)
                for s, (w, n) in zip(kept_p_shots, kept_p_meta)
            ], axis=0)
            means_u = np.concatenate([
                extract_parallel_lightcone_probe_all_means(s, w, n, W)
                for s, (w, n) in zip(kept_u_shots, kept_u_meta)
            ], axis=0)
            d_r = means_u - means_p

            pert_site = W // 2
            point_by_x = {q - pert_site: d_r[:, q].tolist() for q in range(W)}
            n_recovered = d_r.shape[0]

            checkpoint[ckpt_key] = point_by_x
            _save_partial(checkpoint, checkpoint_path)
            if raw_results_path is not None:
                raw_results[ckpt_key] = {
                    "pert_shots": [np.asarray(s) for s in kept_p_shots],
                    "unpert_shots": [np.asarray(s) for s in kept_u_shots],
                    "chunk_meta": {"pert": kept_p_meta, "unpert": kept_u_meta},
                    "job_info": {**info, "recovered": True,
                                 "n_realizations_recovered": n_recovered,
                                 "failures": failures_by_tag},
                }
                _save_partial(raw_results, raw_results_path)

            print(f"p={p:.2f}  L={L:3d}  HARVESTED  n={n_recovered}"
                  + (f"  ({len(failures_by_tag)} chunk failure(s))" if failures_by_tag else ""))
            n_harvested += 1

        elif st in ("CANCELLED", "ERROR"):
            print(f"p={p:.2f}  L={L:3d}  {st} — dead, NOT recoverable, needs "
                  f"resubmission (execute_job_id={execute_job_id}).")
            n_dead += 1

        else:
            print(f"p={p:.2f}  L={L:3d}  still {st} (execute_job_id={execute_job_id})")
            still_pending[ckpt_key] = info
            n_still_pending += 1

    _save_partial(still_pending, pending_path)
    print(
        f"\nHarvested {n_harvested}  |  dead/needs-resubmission {n_dead}  |  "
        f"still in flight {n_still_pending}."
    )


def plot_lightcone_heatmap(
    stats_by_x: dict[int, dict],
    L_values: list[int],
    p_values: list[float],
    stat: str = "C",
    figure_path: str = FIGURE_PATH_LIGHTCONE,
    cmap: str = "viridis",
    cone_line_color: str = "white",
    xlabel: str = "t  (layers, L)",
    ylabel: str = "x  (probe − pert site)",
    shared_colorbar: bool = False,
    shared_axis_labels: bool = False,
    show_suptitle: bool = True,
) -> None:
    """
    Heat map of stat(x, t) restricted to the causal light cone.

    x = probe_site - pert_site (signed, y-axis) vs t = L in layers (x-axis),
    one subplot per p in p_values. Cells outside the light cone (never
    simulated, since no operator can reach there by causality) are set to
    0.0 rather than left blank/NaN — physically the correct value (no
    signal), and it renders as the colormap's zero-color so the cone's
    edge reads as a clean visual boundary (e.g. viridis: yellow inside,
    dark purple outside) instead of a blank/background patch. Dashed white
    lines mark the |x| = L cone edges for the same boundary, drawn on top.

    Parameters
    ----------
    stats_by_x  : output of sweep_lightcone_x_and_t.
    L_values    : recorded layer counts (must match the sweep).
    p_values    : measurement rates to draw, one panel each.
    stat        : which compute_C_t statistic to plot ("C", "se", "var", "cv").
    figure_path : output file path for the saved figure.
    cmap        : matplotlib colormap name, e.g. "RdPu" (pale pink -> dark
        purple) instead of the default "viridis".
    cone_line_color : color of the dashed |x|=L cone-edge lines. Default
        "white" is tuned for viridis (0 -> dark purple, so white reads
        clearly); RdPu-style colormaps put 0 at the PALE end instead, so
        pass e.g. "black" there or the lines will nearly vanish right where
        they matter most (the cone edge).
    xlabel      : x-axis label text.
    ylabel      : y-axis label text.
    shared_colorbar : if True, one colorbar for the whole figure (fixed
        0..1 range, since that's C's physical range) instead of one per
        panel with its own auto-scaled range.
    shared_axis_labels : if True, xlabel is centered under the whole
        figure and ylabel only appears on the leftmost panel, instead of
        every panel repeating both.
    show_suptitle : if False, skip the whole-figure title (e.g. when a
        caption elsewhere already says what the figure is).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = sorted(set(stats_by_x) | {0})
    L_sorted = sorted(L_values)
    x_arr = np.array(x_values)
    L_arr = np.array(L_sorted)

    n_panels = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, p in zip(axes, p_values):
        grid = np.full((len(x_arr), len(L_arr)), 0.0)
        for i, x in enumerate(x_arr):
            if x == 0 and x not in stats_by_x:
                # Perturbation site itself was never probed by this sweep
                # (e.g. sweep_lightcone_x_and_t's probe_site != pert_site
                # circuit constraint) -- C = 0 by convention. Sweeps that DO
                # measure pert_site (e.g. sweep_lightcone_probe_all) supply
                # real x=0 data instead, used via the normal path below.
                grid[i, :] = 0.0
                continue
            stat_p = stats_by_x[x][stat].get(p, {})
            for j, L in enumerate(L_arr):
                if L in stat_p:
                    grid[i, j] = stat_p[L]

        if shared_colorbar:
            mesh = ax.pcolormesh(L_arr, x_arr, grid, shading="nearest", cmap=cmap,
                                  vmin=0, vmax=1)
        else:
            mesh = ax.pcolormesh(L_arr, x_arr, grid, shading="nearest", cmap=cmap)
            fig.colorbar(mesh, ax=ax, label=stat)
        ax.plot(L_arr, L_arr, "--", color=cone_line_color, lw=1, alpha=0.6)
        ax.plot(L_arr, -L_arr, "--", color=cone_line_color, lw=1, alpha=0.6)
        # Clamp to the physically probed range — the cone-edge lines above are
        # not clipped to the system size, so without this the axes autoscale
        # past the actual qubit count whenever the nominal cone (|x| <= L)
        # would extend beyond the boundary.
        ax.set_xlim(L_arr.min() - 0.5, L_arr.max() + 0.5)
        ax.set_ylim(x_arr.min() - 0.5, x_arr.max() + 0.5)
        ax.set_aspect("equal", adjustable="box")
        if shared_axis_labels:
            if ax is axes[0]:
                ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        ax.set_title(f"p = {p:.3f}")

    if shared_axis_labels:
        fig.supxlabel(xlabel)
    if show_suptitle:
        fig.suptitle(f"OTOC light-cone heat map — {stat}(x, t)")

    if shared_colorbar:
        # A shared colorbar via ax=<list of axes> doesn't play well with
        # tight_layout (verified: the colorbar lands on top of the last
        # panel instead of beside it) -- reserve space with
        # subplots_adjust and place the colorbar in its own fixed axes
        # instead, done AFTER tight_layout so tight_layout's own
        # resizing doesn't get re-clobbered.
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(mesh, cax=cax, label=stat)
    else:
        fig.tight_layout()

    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path}")


def run_lightcone_sweep(
    W: int | None = None,
    p_values: list[float] | None = None,
    pert_site: int | None = None,
    L_values: list[int] | None = None,
    n_shots: int | None = None,
    stat: str = "C",
    device_name: str | None = None,
) -> None:
    """
    Run the light-cone (x, t) sweep for one system size and plot the heat map.

    L_values defaults to 1 .. x_max, where x_max = min(pert_site, W-1-pert_site)
    is the number of sites between the middle qubit and the nearer edge —
    i.e. exactly the t range over which the cone is still expanding. The
    module-wide L config (which runs out to L_MAX=25 for the C-vs-t plots)
    would mostly just re-probe an already-saturated, boundary-clipped cone.

    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.

    Launch:  python run_disorder_avg.py --lightcone
    """
    W = W if W is not None else N_VALUES[0]
    pert_site = pert_site if pert_site is not None else W // 2
    x_max = min(pert_site, W - 1 - pert_site)
    L_steps = L_values if L_values is not None else list(range(1, x_max + 1))

    stats_by_x = sweep_lightcone_x_and_t(
        W=W,
        p_values=p_values,
        L_values=L_steps,
        pert_site=pert_site,
        n_shots=n_shots,
        device_name=device_name,
    )
    plot_lightcone_heatmap(
        stats_by_x,
        L_values=L_steps,
        p_values=p_values if p_values is not None else P_VALUES,
        stat=stat,
        figure_path=FIGURE_PATH_LIGHTCONE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Full OTOC evolution  — forward (trivially zero) + backward (partial echo)
# ══════════════════════════════════════════════════════════════════════════════


def sweep_full_otoc_evolution(
    W: int | None = None,
    p_values: list[float] | None = None,
    L: int | None = None,
    pert_site: int | None = None,
    n_shots: int | None = None,
    device_name: str | None = None,
) -> tuple[dict[int, dict], int]:
    """
    Disorder-averaged OTOC sweep tracing the correlator across the FULL
    circuit duration: forward evolution U (t = 0..L) followed by backward
    evolution U† (t = L..2L), for a fixed total depth L.

    t = 0     : start of the circuit.
    t = L     : right after the perturbation, before any U† layer.
    t = L+tau : tau layers of U† applied (0 <= tau <= L).
    t = 2L    : the standard, full-echo OTOC circuit (tau = L).

    For t <= L, C(x, t) = 0 EXACTLY, by a no-signaling argument: with zero
    backward layers applied, the perturbed and unperturbed circuits differ
    only by a local operation on pert_site, and no gate couples pert_site to
    probe_site before the readout — a local operation cannot change another
    qubit's marginal statistics. Those points are filled in analytically,
    with no circuits run, rather than "measured" with sampling noise.

    For t > L, circuits are built with L_back = t - L (see
    build_brickwall_otoc_circuit) so U is always the same full-L forward
    evolution and only the backward depth varies.

    Parameters
    ----------
    W         : system size; defaults to N_VALUES[0].
    p_values  : measurement rates to sweep; defaults to P_VALUES.
    L         : total forward depth (fixed); defaults to the light-cone
                reach x_max = min(pert_site, W-1-pert_site).
    pert_site : perturbation qubit; defaults to W // 2 (the middle site).
    n_shots   : shots per circuit; defaults to N_SHOTS_FULL_OTOC.
    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.

    Returns
    -------
    (stats_by_x, L) : stats_by_x[x] is compute_C_t output keyed by
        t = 0..2L instead of L; L is the depth actually used (echoes back
        the resolved default so callers/plotters agree on the range).
    """
    W = W if W is not None else N_VALUES[0]
    p_values = p_values if p_values is not None else P_VALUES
    pert_site = pert_site if pert_site is not None else W // 2
    x_max = min(pert_site, W - 1 - pert_site)
    L = L if L is not None else x_max
    shots = n_shots if n_shots is not None else N_SHOTS_FULL_OTOC
    xs = [x for x in range(-x_max, x_max + 1) if x != 0]

    backend = _build_backend(device_name=device_name)
    add_bar = isinstance(backend, QuantinuumBackend)
    meas_base = MEAS_SEED + 1

    raw_by_x: dict[int, dict] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    print(
        f"\n=== Full OTOC evolution | W={W}  pert_site={pert_site}  L={L}  "
        f"n_shots={shots}  x in {[-x_max, x_max]} ==="
    )

    # t <= L: no backward evolution yet — C is identically 0 (no-signaling).
    for x in xs:
        for p in p_values:
            for t in range(0, L + 1):
                raw_by_x[x][p][t].append(0.0)

    for c_idx in range(N_CIRCUITS):
        circuit_seed = BASE_SEED * 10_000 + c_idx
        for i_idx in range(N_INIT_STATES):
            init_seed_val = BASE_SEED * 10_000_000 + c_idx * 1_000 + i_idx
            meas_seed_val = meas_base * 10_000_000 + c_idx * 1_000 + i_idx

            for p in p_values:
                for x in xs:
                    probe_site = pert_site + x
                    for tau in range(1, L + 1):
                        shared = dict(
                            L=L,
                            W=W,
                            pert_site=pert_site,
                            pert_op=PERT_OP,
                            probe_site=probe_site,
                            probe_angle=PROBE_ANGLE,
                            p=p,
                            seed=circuit_seed,
                            init_seed=init_seed_val,
                            meas_seed=meas_seed_val,
                            add_barrier=add_bar,
                            init_ops=INIT_OPS,
                            L_back=tau,
                        )
                        _build = _circuit_builder()
                        qc_pert = _build(**shared, unperturbed=False)
                        qc_unpert = _build(**shared, unperturbed=True)
                        probe_bit = qc_pert.n_bits - 1

                        sp, su = _run_circuits(
                            [qc_pert, qc_unpert],
                            backend,
                            shots,
                            nexus_project=NEXUS_PROJECT,
                            optimisation_level=OPTIMISATION_LEVEL,
                            use_batch=NEXUS_BATCH,
                        )

                        out_p = 1 - 2 * sp[:, probe_bit].astype(int)
                        out_u = 1 - 2 * su[:, probe_bit].astype(int)
                        d_r = float(out_u.mean() - out_p.mean())

                        t = L + tau
                        raw_by_x[x][p][t].append(d_r)
                        print(
                            f"x={x:+3d}  p={p:.2f}  t={t:3d} (tau={tau:2d}/{L})  "
                            f"c={c_idx:2d}  i={i_idx:2d}  d={d_r:.4f}"
                        )

    stats_by_x = {
        x: compute_C_t(
            {p: {t: np.array(v) for t, v in td.items()} for p, td in pd.items()}
        )
        for x, pd in raw_by_x.items()
    }
    return stats_by_x, L


def plot_full_otoc_heatmap(
    stats_by_x: dict[int, dict],
    L: int,
    p_values: list[float],
    stat: str = "C",
    figure_path: str = FIGURE_PATH_FULL_OTOC,
) -> None:
    """
    Heat map of stat(x, t) across the FULL echo circuit duration t = 0..2L:
    forward evolution (t <= L, exactly zero) followed by backward evolution
    (t > L, cone edges at |x| = t - L). A solid white line marks t = L, the
    moment of the perturbation.

    Parameters
    ----------
    stats_by_x  : output of sweep_full_otoc_evolution (the stats_by_x half).
    L           : total forward depth used by the sweep (its second output).
    p_values    : measurement rates to draw, one panel each.
    stat        : which compute_C_t statistic to plot ("C", "se", "var", "cv").
    figure_path : output file path for the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = sorted(set(stats_by_x) | {0})
    t_values = list(range(0, 2 * L + 1))
    x_arr = np.array(x_values)
    t_arr = np.array(t_values)

    n_panels = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, p in zip(axes, p_values):
        grid = np.full((len(x_arr), len(t_arr)), np.nan)
        for i, x in enumerate(x_arr):
            if x == 0:
                # Perturbation site itself: never probed, C = 0 by convention.
                grid[i, :] = 0.0
                continue
            stat_p = stats_by_x[x][stat].get(p, {})
            for j, t in enumerate(t_arr):
                if t in stat_p:
                    grid[i, j] = stat_p[t]

        mesh = ax.pcolormesh(t_arr, x_arr, grid, shading="nearest", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label=stat)
        ax.axvline(L, color="white", lw=1.4, ls="-", alpha=0.85)
        cone_t = np.arange(L, 2 * L + 1)
        ax.plot(cone_t, cone_t - L, "--", color="white", lw=1, alpha=0.6)
        ax.plot(cone_t, -(cone_t - L), "--", color="white", lw=1, alpha=0.6)
        ax.set_xlim(t_arr.min() - 0.5, t_arr.max() + 0.5)
        ax.set_ylim(x_arr.min() - 0.5, x_arr.max() + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("t   (0=start · L=perturbation · 2L=full echo)")
        ax.set_ylabel("x  (probe − pert site)")
        ax.set_title(f"p = {p:.3f}")

    fig.suptitle(
        f"Full OTOC evolution — {stat}(x, t),  forward (0..{L}) + backward ({L}..{2*L})"
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path}")


def run_full_otoc_sweep(
    W: int | None = None,
    p_values: list[float] | None = None,
    pert_site: int | None = None,
    L: int | None = None,
    n_shots: int | None = None,
    stat: str = "C",
    device_name: str | None = None,
) -> None:
    """
    Run the full forward+backward OTOC evolution sweep and plot the heat map.

    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware.

    Launch:  python run_disorder_avg.py --full-otoc
    """
    stats_by_x, L_used = sweep_full_otoc_evolution(
        W=W,
        p_values=p_values,
        L=L,
        pert_site=pert_site,
        n_shots=n_shots,
        device_name=device_name,
    )
    plot_full_otoc_heatmap(
        stats_by_x,
        L=L_used,
        p_values=p_values if p_values is not None else P_VALUES,
        stat=stat,
        figure_path=FIGURE_PATH_FULL_OTOC,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tiled single-shot sweep  — same statistics as sweep_single_shot_disorder,
# packed via build_parallel_otoc_circuit instead of one circuit per realization
# ══════════════════════════════════════════════════════════════════════════════


def sweep_single_shot_disorder_tiled(
    p_values: list[float],
    L_values: list[int],
    n_qubits_budget: int,
    N_values: list[int] | None = None,
    n_realizations: int = 400,
    base_seed: int = 42,
    base_seeds: list[int] | None = None,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    backend=None,
    optimisation_level: int = 0,
    use_batch: bool = True,
    topology: str = "brickwall",
    init_ops: list[str] | None = None,
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_per_chunk: int | None = None,
    device_name: str | None = None,
    checkpoint_path: str | None = None,
    raw_results_path: str | None = None,
) -> dict[int, dict]:
    """
    Tiled analogue of sweep_single_shot_disorder (__main__.py): IDENTICAL
    per-realization seed formula, so for a fixed base_seed the two functions
    draw the exact same n_realizations random circuits per (N, p, L) — the
    only difference is how those circuits are packaged for execution.

    sweep_single_shot_disorder : n_realizations separate 1-shot circuit
        pairs (2*n_realizations total circuit submissions) per (N, p, L).
    This function                 : ONE tiled pert circuit + ONE tiled
        unpert circuit (2 submissions total) per (N, p, L), each holding
        all n_realizations instances via build_parallel_otoc_circuit.

    Per-realization seeds (matching sweep_single_shot_disorder exactly, for
    each _seed in base_seeds):
      gate angles   → (_seed + 0) * 10^9 + L * n_realizations + r
      initial state → (_seed + 1) * 10^9 + L * n_realizations + r
      U meas sites  → (_seed + 2) * 10^9 + L * n_realizations + r
      U† meas sites → (_seed + 3) * 10^9 + L * n_realizations + r

    base_seeds : list of root seeds to average over (overrides base_seed
        when set), same semantics as sweep_single_shot_disorder — each seed
        contributes n_realizations independent samples, concatenated before
        computing statistics.
    max_depth : hard cap on build_parallel_otoc_circuit's sequential
        depth-rows (None = uncapped). See that function's docstring — too
        many rows means too much real elapsed gate time per qubit relative
        to its T1/T2, silently turning later "realizations" into noise
        instead of signal. Raises ValueError rather than building an
        overly-deep circuit; increase n_qubits_budget or reduce
        n_realizations if you hit it.
    max_classical_bits : hard cap on total classical bits per tiled circuit
        (width scratch bits + one probe bit per realization in that chunk).
        Real devices cap classical register size independent of qubit count
        or depth (e.g. 64 on H2-2) — since bit count scales directly with
        realizations-per-chunk, this is often the TIGHTER constraint than
        max_depth, not a redundant one. Whichever cap is tighter determines
        the actual chunk size (see build_parallel_otoc_circuits_batched).
    max_su4_per_chunk : hard cap on total SU4/TK2 gates per tiled circuit
        (_n_su4_per_realization(L, W) * realizations-in-that-chunk). Unlike
        max_depth/max_classical_bits, this scales with L — the compile
        service (e.g. Nexus targeting H2-2) can cancel a compile job with a
        server-side timeout once there are too many generic two-qubit
        unitaries to KAK-decompose, and that threshold is independent of
        qubit/bit budget. Chunk size shrinks automatically as L grows to
        stay under this cap. Not a documented hard limit like the other
        two — set it conservatively below wherever your own compiles start
        failing.
    device_name : Quantinuum device to target; None (default) = local
        simulation. Never inherited from the module-level DEVICE_NAME —
        pass it explicitly to run on hardware. Ignored if backend is given.
    checkpoint_path : if given, pickles {(W, seed, p, L): [d_r, ...]} to this
        path after EVERY (W, seed, p, L) point finishes — that's the natural
        atomic unit here, since one _run_circuits call either fully
        completes (job succeeds, results downloaded) or doesn't (exception,
        process killed), so there's no partial-point state to track. On
        start, any existing checkpoint at this path is loaded first, and
        points already present (with the expected n_realizations count) are
        skipped rather than re-submitted — protects against re-paying for
        already-executed real-hardware circuits if a previous run crashed
        or was interrupted partway through. Delete the file to force a full
        fresh run.
    raw_results_path : if given, pickles {(W, seed, p, L): record} to this
        path after EVERY point finishes (purely additive alongside
        checkpoint_path — does not change checkpoint_path's own format or
        resume behaviour). Each record is
        {"pert_shots": [...], "unpert_shots": [...],
         "pert_probe_bits": [...], "unpert_probe_bits": [...],
         "chunk_meta": {"pert": [(width, n_realizations), ...], "unpert": [...]},
         "job_info": {...}}
        — "*_shots" are the raw per-chunk get_shots() arrays exactly as
        returned by _run_circuits (columns 0..width-1 are scratch bits,
        columns width..width+n_realizations-1 are probe bits — NOT one bit
        per qubit, one bit per REALIZATION, all scratch bits grouped first);
        "*_probe_bits" are those probe-bit columns already sliced out per
        chunk, so no index math is needed to use them directly; "job_info"
        carries the Nexus compile_job_id/execute_job_id/circuit_ids/tags
        that produced this point on real hardware. Use
        rebuild_stats_from_raw_results() to turn a whole raw_results_path
        file back into the same all_stats shape this function returns,
        using the exact same reduction this function uses live.

    Returns
    -------
    all_stats[N] = compute_C_t output, same shape as sweep_single_shot_disorder.
    """
    if N_values is None:
        N_values = list(range(8, 17, 2))
    if backend is None:
        backend = _build_backend(n_qubits=n_qubits_budget, device_name=device_name)
    _add_barrier = isinstance(backend, QuantinuumBackend)
    _seeds = base_seeds if base_seeds is not None else [base_seed]

    checkpoint: dict[tuple, list] = {}
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = _load_partial(checkpoint_path)
        print(f"Loaded checkpoint ← {checkpoint_path}  ({len(checkpoint)} point(s) already done)")

    raw_results: dict[tuple, dict] = {}
    if raw_results_path is not None and os.path.exists(raw_results_path):
        raw_results = _load_partial(raw_results_path)

    all_stats: dict[int, dict] = {}

    for W in N_values:
        _pert_site = pert_site if pert_site is not None else W // 2
        _probe_site = probe_site if probe_site is not None else 0
        raw: dict[float, dict[int, list]] = {
            p: {L: [] for L in L_values} for p in p_values
        }

        for _seed in _seeds:
            for p in p_values:
                for L in L_values:
                    ckpt_key = (W, _seed, p, L)
                    cached = checkpoint.get(ckpt_key)
                    if checkpoint_path is not None and cached is not None and len(cached) == n_realizations:
                        raw[p][L].extend(cached)
                        print(
                            f"N={W:2d}  p={p:.2f}  L={L:3d}  seed={_seed}  "
                            f"[checkpointed, skipping submission]  "
                            f"C_t={float(np.mean(cached)):.4f}  n_total={len(raw[p][L])}"
                        )
                        continue

                    realizations = []
                    for r in range(n_realizations):

                        def _s(axis, r=r, s=_seed):
                            return (s + axis) * 1_000_000_000 + L * n_realizations + r

                        realizations.append(
                            dict(
                                seed=_s(0),
                                init_seed=_s(1),
                                meas_seed=_s(2),
                                meas_seed_ud=_s(3),
                            )
                        )

                    common = dict(
                        L=L,
                        W=W,
                        pert_site=_pert_site,
                        pert_op=pert_op,
                        probe_site=_probe_site,
                        probe_angle=probe_angle,
                        p=p,
                        add_barrier=_add_barrier,
                        init_ops=init_ops,
                        topology=topology,
                        n_qubits_budget=n_qubits_budget,
                        realizations=realizations,
                        max_depth=max_depth,
                        max_classical_bits=max_classical_bits,
                        max_su4_per_chunk=max_su4_per_chunk,
                    )
                    # Each of these is a list of (circuit, width, n_realizations)
                    # chunks — more than one iff max_depth/max_classical_bits/
                    # max_su4_per_chunk forced a split.
                    pert_chunks = build_parallel_otoc_circuits_batched(
                        **common, unperturbed=False
                    )
                    unpert_chunks = build_parallel_otoc_circuits_batched(
                        **common, unperturbed=True
                    )

                    tag_base = f"N{W}_p{p:.2f}_L{L:03d}_s{_seed}_tiled"
                    all_circuits = [qc for qc, _, _ in pert_chunks] + [
                        qc for qc, _, _ in unpert_chunks
                    ]
                    all_tags = [
                        f"{tag_base}_pert_chunk{i}" for i in range(len(pert_chunks))
                    ] + [
                        f"{tag_base}_unpert_chunk{i}" for i in range(len(unpert_chunks))
                    ]
                    job_info: dict = {}
                    all_shots = _run_circuits(
                        all_circuits,
                        backend,
                        1,
                        tags=all_tags,
                        nexus_project=NEXUS_PROJECT,
                        optimisation_level=optimisation_level,
                        use_batch=use_batch,
                        job_info_out=job_info if raw_results_path is not None else None,
                    )
                    pert_shots = all_shots[: len(pert_chunks)]
                    unpert_shots = all_shots[len(pert_chunks) :]

                    means_p = np.concatenate(
                        [
                            extract_parallel_probe_means(shots, w, n)
                            for shots, (_, w, n) in zip(pert_shots, pert_chunks)
                        ]
                    )
                    means_u = np.concatenate(
                        [
                            extract_parallel_probe_means(shots, w, n)
                            for shots, (_, w, n) in zip(unpert_shots, unpert_chunks)
                        ]
                    )
                    d_r_values = (means_u - means_p).tolist()
                    raw[p][L].extend(d_r_values)
                    print(
                        f"N={W:2d}  p={p:.2f}  L={L:3d}  seed={_seed}  "
                        f"[tiled: {len(pert_chunks)} chunk(s), width={pert_chunks[0][1]}]  "
                        f"C_t={(means_u - means_p).mean():.4f}  n_total={len(raw[p][L])}"
                    )

                    if checkpoint_path is not None:
                        checkpoint[ckpt_key] = d_r_values
                        _save_partial(checkpoint, checkpoint_path)

                    if raw_results_path is not None:
                        # Probe bits occupy columns width .. width+n_realizations-1 of
                        # each chunk's raw shots (see build_parallel_otoc_circuit) —
                        # bits 0..width-1 are scratch, one per block, NOT one per
                        # qubit-0-of-block; probe bits are a single contiguous range
                        # AFTER all scratch bits, one per realization, not interleaved
                        # by block. Sliced out here so nobody has to re-derive that
                        # indexing later from raw shots + width alone.
                        pert_probe_bits = [
                            np.asarray(s)[:, w : w + n]
                            for s, (_, w, n) in zip(pert_shots, pert_chunks)
                        ]
                        unpert_probe_bits = [
                            np.asarray(s)[:, w : w + n]
                            for s, (_, w, n) in zip(unpert_shots, unpert_chunks)
                        ]
                        raw_results[ckpt_key] = {
                            "pert_shots": [np.asarray(s) for s in pert_shots],
                            "unpert_shots": [np.asarray(s) for s in unpert_shots],
                            "pert_probe_bits": pert_probe_bits,
                            "unpert_probe_bits": unpert_probe_bits,
                            "chunk_meta": {
                                "pert": [(w, n) for _, w, n in pert_chunks],
                                "unpert": [(w, n) for _, w, n in unpert_chunks],
                            },
                            "job_info": job_info,
                        }
                        _save_partial(raw_results, raw_results_path)

        for p in p_values:
            for L in L_values:
                raw[p][L] = np.array(raw[p][L])

        all_stats[W] = compute_C_t(raw)

    return all_stats


def rebuild_stats_from_raw_results(raw_results_path: str) -> dict[int, dict]:
    """
    Rebuild all_stats (same shape sweep_single_shot_disorder_tiled returns)
    entirely from a raw_results_path file written by that function — i.e.
    from raw probe bits alone, not from the (already-reduced) checkpoint.
    This is the answer to "given the raw classical bits, how do I get back
    to the final C(t) graph": it's exactly extract_parallel_probe_means
    applied per chunk, concatenated across chunks, differenced between
    perturbed/unperturbed, then handed to compute_C_t — the identical
    reduction sweep_single_shot_disorder_tiled itself uses live, so a
    reconstruction from this function is guaranteed consistent with
    whatever checkpoint_path already recorded for the same points (useful
    as an independent check that the checkpointed d_r values weren't
    corrupted, or as the sole recovery path if checkpoint_path was lost but
    raw_results_path survived).

    Bit layout reminder (see sweep_single_shot_disorder_tiled's docstring):
    within one chunk's raw shots, columns 0..width-1 are scratch bits (one
    per width-block, not one per qubit), columns width..width+n_real-1 are
    probe bits (one per realization, all grouped together AFTER every
    scratch bit — not interleaved by block/qubit). record["pert_probe_bits"]
    / record["unpert_probe_bits"] already have this slice applied; this
    function uses extract_parallel_probe_means on the full "*_shots" arrays
    directly (recomputing that same slice from "chunk_meta") purely so the
    reduction logic has one canonical implementation instead of two.

    Parameters
    ----------
    raw_results_path : path written by sweep_single_shot_disorder_tiled's
        own raw_results_path argument.

    Returns
    -------
    all_stats[N] = compute_C_t output, keyed exactly like
    sweep_single_shot_disorder_tiled's return value — pass straight to
    plot_results / plot_tm_overlay.
    """
    raw_results = _load_partial(raw_results_path)

    raw_by_W: dict[int, dict[float, dict[int, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (W, _, p, L), record in raw_results.items():
        means_p = np.concatenate([
            extract_parallel_probe_means(shots, w, n)
            for shots, (w, n) in zip(record["pert_shots"], record["chunk_meta"]["pert"])
        ])
        means_u = np.concatenate([
            extract_parallel_probe_means(shots, w, n)
            for shots, (w, n) in zip(record["unpert_shots"], record["chunk_meta"]["unpert"])
        ])
        raw_by_W[W][p][L].extend((means_u - means_p).tolist())

    all_stats: dict[int, dict] = {}
    for W, raw in raw_by_W.items():
        for p in raw:
            for L in raw[p]:
                raw[p][L] = np.array(raw[p][L])
        all_stats[W] = compute_C_t(raw)
    return all_stats


# ══════════════════════════════════════════════════════════════════════════════
# Sequential  — original single-process run (no partitioning)
# ══════════════════════════════════════════════════════════════════════════════


def run_sequential(
    mode: str = "single_shot",
    device_name: str | None = None,
    checkpoint_path: str | None = None,
    raw_results_path: str | None = None,
) -> None:
    """
    Run the full disorder average in one process.

    Parameters
    ----------
    mode : "single_shot" (default), "single_shot_tiled", or "multi_shot"
        single_shot        — sweep_single_shot_disorder: each of N_REALIZATIONS
            samples is a fully independent circuit (fresh angles, initial
            state, U and U† measurement sites) executed for 1 shot.
        single_shot_tiled  — sweep_single_shot_disorder_tiled: identical
            statistics to single_shot (same seed formula), but all
            N_REALIZATIONS samples for a given (N, p, L) are packed into
            ONE tiled circuit (build_parallel_otoc_circuit, width x depth
            on N_QUBITS_BUDGET physical qubits) instead of one circuit per
            realization — 2 circuit submissions per (N, p, L) instead of
            2*N_REALIZATIONS. Requires N_QUBITS_BUDGET >= max(N_VALUES).
        multi_shot         — sweep_over_all_disorder_axes: N_CIRCUITS x
            N_INIT_STATES realizations, each circuit run for N_SHOTS shots.
    device_name : Quantinuum device to target; None (default) = local
        simulation. SAFETY: never inherited from the module-level
        DEVICE_NAME constant, even though DEVICE_NAME may be set — pass
        device_name explicitly (e.g. device_name=DEVICE_NAME, or via the
        CLI's --device-name flag) to actually run on hardware.
    checkpoint_path : single_shot_tiled mode only — see
        sweep_single_shot_disorder_tiled's docstring. Persists progress to
        disk after every (N, seed, p, L) point so a crash/kill mid-run
        doesn't lose already-paid-for hardware results; resuming from the
        same path skips points already completed instead of re-submitting.
    raw_results_path : single_shot_tiled mode only — see
        sweep_single_shot_disorder_tiled's docstring. Additive alongside
        checkpoint_path: persists the raw per-chunk shot arrays plus (on
        real hardware) Nexus compile_job_id/execute_job_id/circuit_ids,
        labeled by (N, seed, p, L), so exact raw results are recoverable
        without re-running anything or needing to look anything up on the
        Nexus side.
    """
    L_steps = _L_steps()
    _qubits_needed = N_QUBITS_BUDGET if mode == "single_shot_tiled" else max(N_VALUES)
    backend = _build_backend(n_qubits=_qubits_needed, device_name=device_name)
    backend_label = str(backend)

    print("=" * 64)
    print(f"Sequential run  —  mode={mode!r}")
    print("=" * 64)
    print(f"  p_values       = {P_VALUES}")
    print(f"  L_max          = {L_MAX}   →  recorded at L = {L_steps}")
    print(f"  N_values       = {N_VALUES}")
    print(f"  base_seeds     = {BASE_SEEDS}")
    print(f"  pert_op        = {PERT_OP!r}")
    print(f"  backend        = {backend_label!r}")

    if mode == "single_shot":
        n_total = (
            len(N_VALUES)
            * len(P_VALUES)
            * len(L_steps)
            * N_REALIZATIONS
            * len(BASE_SEEDS)
            * 2
        )
        print(
            f"  n_realizations = {N_REALIZATIONS} × {len(BASE_SEEDS)} seed(s) = {N_REALIZATIONS * len(BASE_SEEDS)}"
        )
        print(f"  Total circuits = {n_total}  (1 shot each)")
        print("=" * 64)
        all_stats = sweep_single_shot_disorder(
            p_values=P_VALUES,
            L_max=L_MAX,
            N_values=N_VALUES,
            n_realizations=N_REALIZATIONS,
            base_seeds=BASE_SEEDS,
            record_every=RECORD_EVERY,
            L_values=L_steps if L_VALUES is not None else None,
            pert_op=PERT_OP,
            probe_angle=PROBE_ANGLE,
            backend=backend,
            optimisation_level=OPTIMISATION_LEVEL,
            use_batch=NEXUS_BATCH,
            circuit_builder=_circuit_builder(),
            init_ops=INIT_OPS,
            nexus_project=NEXUS_PROJECT,
        )
    elif mode == "single_shot_tiled":
        # Per-(N,L) chunk count if TILED_MAX_DEPTH, TILED_MAX_CLASSICAL_BITS, or
        # TILED_MAX_SU4_PER_CHUNK forces build_parallel_otoc_circuits_batched to
        # split realizations across more than one circuit (see that function) —
        # whichever cap is tightest wins. Unlike the other two, the gate-count
        # cap depends on L, so chunk count isn't uniform across L_steps anymore.
        def _chunks_for(n_qubits, L):
            width_n = max(1, N_QUBITS_BUDGET // n_qubits)
            depth_chunk = N_REALIZATIONS if TILED_MAX_DEPTH is None else width_n * TILED_MAX_DEPTH
            bits_chunk = (
                N_REALIZATIONS if TILED_MAX_CLASSICAL_BITS is None
                else max(1, TILED_MAX_CLASSICAL_BITS - width_n)
            )
            gates_chunk = (
                N_REALIZATIONS if TILED_MAX_SU4_PER_CHUNK is None
                else max(1, TILED_MAX_SU4_PER_CHUNK // max(1, _n_su4_per_realization(L, n_qubits)))
            )
            chunk = min(depth_chunk, bits_chunk, gates_chunk)
            return max(1, -(-N_REALIZATIONS // chunk))

        n_circuits = (
            sum(_chunks_for(n, L) for n in N_VALUES for L in L_steps)
            * len(P_VALUES)
            * 2
            * len(BASE_SEEDS)
        )
        print(
            f"  n_realizations = {N_REALIZATIONS} × {len(BASE_SEEDS)} seed(s) = {N_REALIZATIONS * len(BASE_SEEDS)}"
        )
        print(
            f"  n_qubits_budget = {N_QUBITS_BUDGET}  (max depth-rows = {TILED_MAX_DEPTH}, "
            f"max classical bits = {TILED_MAX_CLASSICAL_BITS}, max su4/chunk = {TILED_MAX_SU4_PER_CHUNK})"
        )
        print(
            f"  chunks per (N,L): "
            f"{ {L: sum(_chunks_for(n, L) for n in N_VALUES) for L in L_steps} }"
        )
        print(
            f"  Total circuits  = {n_circuits}  (vs. "
            f"{len(N_VALUES)*len(P_VALUES)*len(L_steps)*N_REALIZATIONS*len(BASE_SEEDS)*2} untiled)"
        )
        print("=" * 64)
        all_stats = sweep_single_shot_disorder_tiled(
            p_values=P_VALUES,
            L_values=L_steps,
            n_qubits_budget=N_QUBITS_BUDGET,
            N_values=N_VALUES,
            n_realizations=N_REALIZATIONS,
            base_seeds=BASE_SEEDS,
            pert_op=PERT_OP,
            probe_angle=PROBE_ANGLE,
            backend=backend,
            optimisation_level=OPTIMISATION_LEVEL,
            use_batch=NEXUS_BATCH,
            topology=TOPOLOGY,
            init_ops=INIT_OPS,
            max_classical_bits=TILED_MAX_CLASSICAL_BITS,
            max_su4_per_chunk=TILED_MAX_SU4_PER_CHUNK,
            max_depth=TILED_MAX_DEPTH,
            checkpoint_path=checkpoint_path,
            raw_results_path=raw_results_path,
        )
    elif mode == "multi_shot":
        n_real = N_CIRCUITS * N_INIT_STATES * len(BASE_SEEDS)
        n_total = len(N_VALUES) * len(P_VALUES) * n_real * len(L_steps) * 2
        print(f"  n_circuits     = {N_CIRCUITS}")
        print(f"  n_init_states  = {N_INIT_STATES}")
        print(f"  n_shots        = {N_SHOTS}")
        print(f"  meas_seed      = {MEAS_SEED}")
        print(
            f"  Total circuits = {n_total}  ({N_SHOTS} shots each, {len(BASE_SEEDS)} seed(s))"
        )
        print("=" * 64)
        all_stats = sweep_over_all_disorder_axes(
            p_values=P_VALUES,
            L_max=L_MAX,
            N_values=N_VALUES,
            n_shots=N_SHOTS,
            n_circuits=N_CIRCUITS,
            n_init_states=N_INIT_STATES,
            base_seeds=BASE_SEEDS,
            meas_seed=MEAS_SEED,
            record_every=RECORD_EVERY,
            L_values=L_steps if L_VALUES is not None else None,
            pert_op=PERT_OP,
            probe_angle=PROBE_ANGLE,
            backend=backend,
            optimisation_level=OPTIMISATION_LEVEL,
            use_batch=NEXUS_BATCH,
            circuit_builder=_circuit_builder(),
            init_ops=INIT_OPS,
            nexus_project=NEXUS_PROJECT,
        )
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Choose 'single_shot', 'single_shot_tiled', or 'multi_shot'."
        )

    plot_results(
        all_stats=all_stats,
        L_values=L_steps,
        p_values=P_VALUES,
        N_values=N_VALUES,
        figure_path_C=FIGURE_PATH_C,
        figure_path_cv=FIGURE_PATH_CV,
    )

    if TM_NUM_QUBITS is not None:
        _tm_p_values = TM_P_VALUES if TM_P_VALUES is not None else P_VALUES
        print(
            f"\nRunning transfer matrix: N={TM_NUM_QUBITS}  p_values={_tm_p_values}  "
            f"tf={TM_TF}  pert={TM_PERT_SITE}  probe={TM_PROBE_SITE}  "
            f"bc={TM_BC_TYPE!r}"
        )
        tm = ButterflyTransferMatrix(TM_NUM_QUBITS, bc_type=TM_BC_TYPE)
        tm_results = {
            p: (t, C / 2)
            for p in _tm_p_values
            for t, C in [tm.compute_otoc(p, TM_TF, TM_PERT_SITE, TM_PROBE_SITE)]
        }
        plot_tm_overlay(
            all_stats=all_stats,
            L_values=L_steps,
            p_values=P_VALUES,
            tm_results=tm_results,
            tm_N=TM_NUM_QUBITS,
            figure_path=FIGURE_PATH_TM,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def _print_nohup_commands() -> None:
    venv = ".butterfly/bin/python"
    script = "run_disorder_avg.py"
    d = PARTIALS_DIR

    print(f"""
Nohup launch commands
─────────────────────
mkdir -p {d}

Scheme 1 — 5 jobs (one per N):
  for i in 0 1 2 3 4; do
    nohup {venv} {script} --scheme 1 --job-id $i > {d}/s1_$i.log 2>&1 &
  done

Scheme 2 — {N_CIRCUITS} jobs (one per c_idx):
  for i in $(seq 0 {N_CIRCUITS-1}); do
    nohup {venv} {script} --scheme 2 --job-id $i > {d}/s2_c$i.log 2>&1 &
  done

Scheme 3 — {N_CIRCUITS*N_INIT_STATES} jobs (one per realization):
  for c in $(seq 0 {N_CIRCUITS-1}); do
    for i in $(seq 0 {N_INIT_STATES-1}); do
      nohup {venv} {script} --scheme 3 --job-id $c --secondary-id $i \\
            > {d}/s3_c${{c}}_i${{i}}.log 2>&1 &
    done
  done

Scheme 4 — {len(N_VALUES)*len(P_VALUES)} jobs (one per (N,p) grid point):
  for n in $(seq 0 {len(N_VALUES)-1}); do
    for p in $(seq 0 {len(P_VALUES)-1}); do
      nohup {venv} {script} --scheme 4 --job-id $n --secondary-id $p \\
            > {d}/s4_n${{n}}_p${{p}}.log 2>&1 &
    done
  done

Merge + plot after jobs complete:
  {venv} {script} --scheme 1 --merge
  {venv} {script} --scheme 2 --merge   # etc.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disorder-averaged OTOC experiment runner.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scheme",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help=(
            "Partition scheme:\n"
            "  1 = by N          (--job-id = n_idx)\n"
            "  2 = by c_idx      (--job-id = c_idx)\n"
            "  3 = by (c,i)      (--job-id = c_idx, --secondary-id = i_idx)\n"
            "  4 = by (N,p)      (--job-id = n_idx, --secondary-id = p_idx)\n"
            "  omit = sequential full run"
        ),
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=0,
        help="Primary partition index (meaning depends on --scheme).",
    )
    parser.add_argument(
        "--secondary-id",
        type=int,
        default=0,
        help="Secondary partition index for schemes 3 and 4.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Load all partial files for --scheme and produce final figures.",
    )
    parser.add_argument(
        "--mode",
        choices=["single_shot", "single_shot_tiled", "multi_shot"],
        default="single_shot",
        help=(
            "Sequential sweep mode (ignored when --scheme is set):\n"
            "  single_shot (default) — sweep_single_shot_disorder:\n"
            "      N_REALIZATIONS fully independent circuits, 1 shot each.\n"
            "  single_shot_tiled     — sweep_single_shot_disorder_tiled:\n"
            "      same statistics as single_shot, but all N_REALIZATIONS\n"
            "      samples per (N,p,L) are packed into one tiled circuit on\n"
            "      N_QUBITS_BUDGET qubits (build_parallel_otoc_circuit).\n"
            "  multi_shot            — sweep_over_all_disorder_axes:\n"
            "      N_CIRCUITS x N_INIT_STATES realizations, N_SHOTS shots each."
        ),
    )
    parser.add_argument(
        "--L-values",
        type=str,
        default=None,
        metavar="L1,L2,...",
        help=(
            "Comma-separated explicit list of layer counts to evaluate "
            "(e.g. 1,2,5,10,25). Overrides L_MAX and RECORD_EVERY."
        ),
    )
    parser.add_argument(
        "--estimate-hqc",
        action="store_true",
        help="Print an HQC cost estimate for the current config and exit (no circuits run).",
    )
    parser.add_argument(
        "--help-nohup",
        action="store_true",
        help="Print ready-to-paste nohup launch commands and exit.",
    )
    parser.add_argument(
        "--lightcone",
        action="store_true",
        help=(
            "Sweep probe position x vs time t restricted to the causal light "
            "cone from a perturbation at the middle qubit, and plot the "
            "resulting C(x, t) heat map."
        ),
    )
    parser.add_argument(
        "--full-otoc",
        action="store_true",
        help=(
            "Sweep probe position x vs the FULL echo circuit duration t = 0..2L "
            "(forward evolution, trivially zero, then backward evolution "
            "unwinding the perturbation), and plot the resulting C(x, t) heat "
            "map. Uses N_SHOTS_FULL_OTOC shots per circuit."
        ),
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Quantinuum device to run on (e.g. H2-2). SAFETY: omitting this "
            "always runs LOCAL simulation, regardless of the module-level "
            "DEVICE_NAME constant — that constant is never read automatically "
            "by any code path. Pass --device-name explicitly (or "
            "--device-name %s to reuse the configured value) to run on "
            "hardware." % DEVICE_NAME
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "single_shot_tiled mode only: persist progress to this path after "
            "every (N, seed, p, L) point, and skip points already completed "
            "there on a rerun instead of re-submitting (and re-paying for) them."
        ),
    )
    parser.add_argument(
        "--raw-results-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "single_shot_tiled mode only: additionally persist the raw per-chunk "
            "shot arrays and (on real hardware) Nexus job/circuit IDs to this "
            "path after every (N, seed, p, L) point, labeled by that key."
        ),
    )
    args = parser.parse_args()

    if args.L_values:
        L_VALUES = [int(x) for x in args.L_values.split(",")]

    if args.estimate_hqc:
        estimate_hqc_cost(mode=args.mode)

    elif args.help_nohup:
        _print_nohup_commands()

    elif args.lightcone:
        run_lightcone_sweep(device_name=args.device_name)

    elif args.full_otoc:
        run_full_otoc_sweep(device_name=args.device_name)

    elif args.merge:
        if args.scheme is None:
            parser.error("--merge requires --scheme {1|2|3|4}")
        merge_and_plot(args.scheme)

    elif args.scheme == 1:
        run_scheme_1(args.job_id, device_name=args.device_name)

    elif args.scheme == 2:
        run_scheme_2(args.job_id, device_name=args.device_name)

    elif args.scheme == 3:
        run_scheme_3(args.job_id, args.secondary_id, device_name=args.device_name)

    elif args.scheme == 4:
        run_scheme_4(args.job_id, args.secondary_id, device_name=args.device_name)

    else:
        run_sequential(
            mode=args.mode,
            device_name=args.device_name,
            checkpoint_path=args.checkpoint_path,
            raw_results_path=args.raw_results_path,
        )

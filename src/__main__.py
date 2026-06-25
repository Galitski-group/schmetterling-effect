import datetime
from collections import Counter
import numpy as np
import qnexus as qnx
from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend


def calculate_hamming_distance_pdf(data: dict) -> tuple[list, list]:
    """
    Calculates the hamming distance between number of bitstrings and
    ground state |00..0> .

    Parameters:
    data (dict):  Contains bitstring as key, count of bitstring as item

    Returns:
    tuple: The


    """
    distances = Counter()
    for key, item in data.items():
        dist = np.sum(key)
        distances.update({dist: item})

    xs = sorted(distances.keys())
    total = sum(distances.values())
    pdf = [distances[x] / total for x in xs]
    return xs, pdf


def validate_nexus_connection(nexus_hosted=True) -> bool:
    """
    Returns True if we can auth and reach Nexus (and therefore can submit jobs).

    Parameters:
    nexus_hosted (bool) : Checks that local machine connected to nexus hosted api

    Returns:
    bool: Success of connection (true).

    """
    try:
        # (https://docs.quantinuum.com/nexus/trainings/notebooks/basics/getting_started.html
        qnx.login_with_credentials()  # prompt-based

        # This will fail if tokens are invalid, network is blocked, etc.
        df = qnx.devices.get_all(nexus_hosted=nexus_hosted).df()
        return (df is not None) and (len(df) > 0)

    except ConnectionError as ce:
        raise ConnectionError("Failed to connect to quantinuum machine...") from ce


_PERT_UNITARY_OPS = {"X", "Y", "Z", "H", "S", "T"}
_INIT_PAULIS = ["X", "Y", "Z"]  # identity handled by skipping


def generate_time_reversal_breaking_random_brick_wall(
    L: int,
    W: int,
    unperturbed: bool = False,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    add_barrier: bool = True,
    do_reset: bool = False,
    seed: int | None = None,
    init_seed: int | None = None,
    meas_seed: int | None = None,
    meas_seed_ud: int | None = None,
    n_meas_total: int = 0,
) -> Circuit:
    """
    Parameters:
    L (int): The length of circuit or number of layers.
    W (int): The width of the circuit or number of qubits.
    unperturbed (bool): When True the mid-circuit perturbation is suppressed;
    stochastic measurements are unaffected.
    pert_site (int): Qubit index where the mid-circuit perturbation is inserted.
    Defaults to W // 2.
    pert_op (str): Operation applied at pert_site when unperturbed=False.
    Use "measure" for a projective measurement or any single-qubit gate name
    supported by pytket Circuit (e.g. "X", "Y", "Z", "H", "S", "T").
    probe_site (int): Qubit prepared in an equal superposition before U and
    measured after U†. Must differ from pert_site. Defaults to 0.
    probe_angle (float): Ry rotation angle (in pytket half-turns, i.e. units of π)
    applied to probe_site during state preparation. 0.5 → Ry(π/2), -0.5 → Ry(-π/2).
    do_reset (bool): After a "measure" perturbation, reset the qubit to |0>.
    meas_seed (int): Seed for U measurement site selection.
    meas_seed_ud (int): Seed for U† measurement site selection. Independent of
    meas_seed so the forward and backward channels have uncorrelated measurement
    patterns. Defaults to meas_seed + 1 when meas_seed is given, else seed + 1.
    n_meas_total (int): Exact number of stochastic measurements per channel
    (U and U† each get n_meas_total independently placed measurements).
    Distributed uniformly across L-1 eligible layers; sites chosen without
    replacement per layer via the respective meas_rng.

    Returns:
    Circuit: State-prep → U → Perturbation → U† → probe measurement circuit.

    """

    if pert_op != "measure" and pert_op not in _PERT_UNITARY_OPS:
        raise ValueError(
            f"Unsupported pert_op {pert_op!r}. Use 'measure' or one of {_PERT_UNITARY_OPS}."
        )

    if W % 2 != 0:
        raise ValueError("Circuit only defined for even # of qubits.")

    rng = np.random.default_rng(seed=seed)
    init_rng = np.random.default_rng(seed=init_seed if init_seed is not None else seed)
    _ms_u = meas_seed if meas_seed is not None else seed
    _ms_ud = meas_seed_ud if meas_seed_ud is not None else (_ms_u + 1 if _ms_u is not None else 1)
    meas_rng_u = np.random.default_rng(seed=_ms_u)
    meas_rng_ud = np.random.default_rng(seed=_ms_ud)

    if pert_site is None:
        pert_site = W // 2
    if probe_site is None:
        probe_site = 0
    if probe_site == pert_site:
        raise ValueError(
            f"probe_site and pert_site must differ (both are {probe_site})."
        )

    def layer_pairs(l):
        if l % 2 == 0:
            return [(2 * w, 2 * w + 1) for w in range(W // 2)]
        else:
            return [(2 * w + 1, 2 * w + 2) for w in range(W // 2 - 1)]

    # Stochastic measurements are placed only on qubits that participate in a
    # two-qubit gate that layer (paired sites). Unpaired boundary qubits (0 and
    # W-1 in odd layers) are idle and never receive stochastic measurements.
    #
    # U and U† each receive n_meas_total measurements distributed independently:
    # each channel gets its own floor/ceil split across L-1 eligible layers and
    # its own random site selection within each layer. The two RNGs are seeded
    # from meas_seed and meas_seed_ud respectively, ensuring zero correlation.
    #
    # Tuple layout in forward:
    #   pairs_data[k] = (a,b,g,i,j, ai,bi,ci,aj,bj,cj, mi_u,mj_u, mi_ud,mj_ud)
    num_eligible = max(L - 1, 0)

    def _layer_counts(rng_):
        counts = [0] * L
        if num_eligible > 0 and n_meas_total > 0:
            clamped = min(n_meas_total, W * num_eligible)
            base, rem = divmod(clamped, num_eligible)
            for rank, l in enumerate(rng_.permutation(num_eligible)):
                counts[l] = base + (1 if rank < rem else 0)
        return counts

    layer_meas_u  = _layer_counts(meas_rng_u)
    layer_meas_ud = _layer_counts(meas_rng_ud)

    forward = []
    for l in range(L):
        pairs_in_layer = layer_pairs(l)
        paired = sorted({q for pair in pairs_in_layer for q in pair})

        def _pick(rng_, count):
            n = min(count, len(paired))
            return set(rng_.choice(len(paired), size=n, replace=False).tolist()) if n > 0 else set()

        chosen_u  = {paired[k] for k in _pick(meas_rng_u,  layer_meas_u[l])}
        chosen_ud = {paired[k] for k in _pick(meas_rng_ud, layer_meas_ud[l])}

        pairs_data = []
        for i, j in pairs_in_layer:
            a, b, g = rng.normal(size=3)
            ai, bi, ci = rng.uniform(0, 2, size=3)  # TK1 angles for qubit i
            aj, bj, cj = rng.uniform(0, 2, size=3)  # TK1 angles for qubit j
            pairs_data.append((
                a, b, g, i, j, ai, bi, ci, aj, bj, cj,
                i in chosen_u, j in chosen_u,
                i in chosen_ud, j in chosen_ud,
            ))

        forward.append(pairs_data)

    # bit 0 : scratch — receives every mid-circuit measurement (outcomes discarded)
    # bit 1 : probe — final readout only
    n_bits = 2
    probe_bit = 1
    qc = Circuit(W, n_bits)

    barrier_sequence = list(range(W))
    all_bits = list(range(n_bits))

    # --- State preparation ---
    # probe_site → equal superposition via Ry; all other sites → random {I, X, Y, Z}.
    qc.Ry(probe_angle, probe_site)
    for q in range(W):
        if q == probe_site:
            continue
        op = init_rng.choice(["I"] + _INIT_PAULIS)
        if op != "I":
            getattr(qc, op)(q)

    # Build U: gates then U-channel measurements for paired qubits only.
    # Unpaired boundary qubits are idle in odd layers and never measured.
    # All mid-circuit outcomes go to scratch bit 0 (overwritten, never read back).
    for pairs_data in forward:
        for a, b, g, i, j, ai, bi, ci, aj, bj, cj, mi_u, mj_u, _, _ in pairs_data:
            qc.TK2(a, b, g, i, j)
            qc.TK1(ai, bi, ci, i)
            qc.TK1(aj, bj, cj, j)
            if mi_u:
                qc.Measure(i, 0)
            if mj_u:
                qc.Measure(j, 0)
        if add_barrier:
            qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Mid-circuit perturbation (toggled off when unperturbed=True).
    if not unperturbed:
        if pert_op == "measure":
            qc.Measure(pert_site, 0)
            if do_reset:
                qc.Reset(pert_site)
        else:
            getattr(qc, pert_op)(pert_site)

    if add_barrier:
        qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Apply U†: inverse gates then U†-channel measurements (independent of U).
    for pairs_data in reversed(forward):
        for a, b, g, i, j, ai, bi, ci, aj, bj, cj, _, _, mi_ud, mj_ud in reversed(pairs_data):
            qc.TK1(-cj, -bj, -aj, j)
            qc.TK1(-ci, -bi, -ai, i)
            qc.TK2(-a, -b, -g, i, j)
            if mj_ud:
                qc.Measure(j, 0)
            if mi_ud:
                qc.Measure(i, 0)
        if add_barrier:
            qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Final probe measurement.
    qc.Measure(probe_site, probe_bit)

    return qc

def _run_circuits(
    circuits: list[Circuit],
    backend,
    n_shots: int,
    tags: list[str] | None = None,
) -> list[np.ndarray]:
    """
    Compile and execute circuits as a single batch on `backend`.

    All circuits are compiled and submitted together to minimise queue time.
    The trade-off is all-or-nothing fault tolerance: a backend failure loses
    results for every circuit in the batch.

    Circuit names (shown in the Quantinuum portal and local logs) are set from
    `tags` when provided. Auth and machine selection live on the backend object.

    Parameters:
    circuits (list[Circuit]): Circuits to run; output order matches input order.
    backend: Any pytket Backend (e.g. AerBackend for simulation,
    QuantinuumBackend for hardware). Credentials and device are
    configured on the backend object before calling this function.
    n_shots (int): Shots per circuit.
    tags (list[str]): Optional names for each circuit, one per entry in
    circuits. Used for portal tracking and log output.

    Returns:
    list of np.ndarray of shape (n_shots, n_bits), one per input circuit.
    """
    if tags is not None:
        if len(tags) != len(circuits):
            raise ValueError(f"len(tags)={len(tags)} must equal len(circuits)={len(circuits)}")
        for qc, name in zip(circuits, tags):
            qc.name = name

    backend_label = getattr(backend, "device_name", type(backend).__name__)
    tag_summary = tags[0] if tags else "unnamed"
    print(f"[{backend_label}] compiling {len(circuits)} circuit(s)  tag={tag_summary!r}")

    compiled = [
        backend.get_compiled_circuit(qc, optimisation_level=1) for qc in circuits
    ]

    print(f"[{backend_label}] submitting batch of {len(circuits)} circuit(s)  n_shots={n_shots}")
    handles = backend.process_circuits(compiled, n_shots=[n_shots] * len(circuits))

    print(f"[{backend_label}] waiting for results...")
    results = [backend.get_result(h).get_shots() for h in handles]
    print(f"[{backend_label}] done — {len(results)} result(s) received")
    return results


def compute_C_t(
    raw: dict[float, dict[int, np.ndarray]],
) -> dict[str, dict[float, dict[int, float]]]:
    """
    Computes C_t and associated statistics from the raw realization arrays
    produced by sweep_over_all_disorder_axes.

    raw[p][L] is a 1-D array of per-realization values
        d_r = mean_shots(out_u) - mean_shots(out_p)
    where out_u / out_p are the ±1-encoded probe outcomes for the unperturbed
    and perturbed circuits of realization r.

    For each (p, L):
      C_t  = mean over realizations of d_r
           = grand average over (realizations × shots) of (out_u - out_p)
      var  = sample variance of {d_r}  (disorder fluctuation; ddof=1)
      cv   = sqrt(var) / |C_t|          (relative uncertainty; inf when C_t == 0)

    Returns:
    dict with keys "C", "var", "cv", each mapping p → L → scalar float.
    For fixed N and p, all_stats[N]["C"][p] is the C_t time series {L: value}.
    """

    C_t_out: dict[float, dict[int, float]] = {}
    var_out: dict[float, dict[int, float]] = {}
    se_out:  dict[float, dict[int, float]] = {}
    cv_out:  dict[float, dict[int, float]] = {}
    n_out:   dict[float, dict[int, int]]   = {}

    for p, L_data in raw.items():
        C_t_out[p] = {}
        var_out[p] = {}
        se_out[p]  = {}
        cv_out[p]  = {}
        n_out[p]   = {}

        for L, vals in L_data.items():
            n   = len(vals)
            C   = float(np.mean(vals))
            var = float(np.var(vals, ddof=1)) if n > 1 else 0.0
            se  = float(np.sqrt(var / n))     if n > 0 else 0.0
            cv  = float(np.sqrt(var) / abs(C)) if C != 0.0 else np.inf

            C_t_out[p][L] = C
            var_out[p][L] = var
            se_out[p][L]  = se
            cv_out[p][L]  = cv
            n_out[p][L]   = n

    return {"C": C_t_out, "var": var_out, "se": se_out, "cv": cv_out, "n": n_out}


def debug_after_measurement(qc: Circuit, n: int = 20) -> None:
    """
    Auxiliary function which checks that measurement is at
    middle of the time process i.e. between U and U^dagger.

    qc (Circuit): Circuit to be investigated
    n (int): Number of layers in system
    """

    cmds = qc.get_commands()

    # find the (first) Measure command
    m_idx = None
    for idx, cmd in enumerate(cmds):
        if cmd.op.type.name == "Measure":
            m_idx = idx
            break
    if m_idx is None:
        raise RuntimeError("No Measure found in circuit")

    print("MEASURE at command index:", m_idx)
    print("Next commands:")
    for k in range(m_idx + 1, min(m_idx + 1 + n, len(cmds))):
        cmd = cmds[k]
        optype = cmd.op.type.name
        qbs = [q.index[0] for q in cmd.qubits]  # qubit indices
        print(f"{k:4d}  {optype:10s}  qubits={qbs}")


def count_consecutive_tk2_after_measure(qc):
    """
    Docstring for count_consecutive_tk2_after_measure

    :param qc: Description
    """

    cmds = qc.get_commands()
    m_idx = next(i for i, c in enumerate(cmds) if c.op.type.name == "Measure")
    cnt = 0
    pairs = []
    for cmd in cmds[m_idx + 1 :]:
        if cmd.op.type.name != "TK2":
            break
        cnt += 1
        qbs = [q.index[0] for q in cmd.qubits]
        pairs.append(tuple(qbs))
    return cnt, pairs


def sweep_over_all_disorder_axes(
    p_values: list[float],
    L_max: int,
    N_values: list[int] | None = None,
    n_shots: int = 100,
    n_circuits: int = 3,
    n_init_states: int = 3,
    base_seed: int = 42,
    meas_seed: int | None = None,
    record_every: int = 5,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    backend=None,
    device_name: str | None = None,
) -> dict[int, dict[str, dict[float, dict[int, float]]]]:
    """
    Top-level controller: disorder-averaged sweep over (p, T, N).

    For each system size N in N_values, sweeps all p in p_values and all
    recorded L values, averaging over n_circuits gate-angle realizations and
    n_init_states initial Pauli words per realization.  Returns computed
    statistics; call plot_results separately to visualise.

    Parameters:
    p_values (list[float]): Measurement rates to sweep over.
    L_max (int): Maximum number of layers (time steps).
    N_values (list[int]): System sizes (number of qubits, must be even).
    Defaults to [8, 10, 12, 14, 16].
    n_shots (int): Shots per circuit per variant.
    n_circuits (int): Independent gate-angle realizations per (p, N).
    n_init_states (int): Initial Pauli-word realizations per circuit realization.
    base_seed (int): Root seed; circuit and init seeds are derived from it.
    meas_seed (int): Root seed for per-realization measurement uniforms so that
    p acts as a pure threshold. Defaults to base_seed + 1 to keep the
    three disorder axes (gates, init, meas) independent.
    record_every (int): Only record/execute at L = record_every, 2*record_every, ...
    pert_site (int): Perturbation qubit. Defaults to W // 2 per system size.
    pert_op (str): Perturbation type ("measure" or a single-qubit gate name).
    probe_site (int): Probe qubit. Defaults to 0 per system size.
    probe_angle (float): Ry half-turn angle applied to the probe qubit.
    backend: Any compiled pytket Backend. Defaults to AerBackend() when neither
    backend nor device_name is given. Pass QuantinuumBackend for hardware runs.
    device_name (str): Quantinuum device name; only used when backend is None.

    Returns:
    all_stats[N] = compute_C_t output with keys "C", "var", "cv",
    each mapping p → L → scalar float.
    """
    if N_values is None:
        N_values = list(range(8, 17, 2))

    if backend is None:
        if device_name is not None:
            backend = QuantinuumBackend(device_name)
        else:
            from pytket.extensions.qiskit import AerBackend
            backend = AerBackend()

    _add_barrier = isinstance(backend, QuantinuumBackend)
    print(f"Backend: {backend}  |  add_barrier: {_add_barrier}")

    L_values = list(range(record_every, L_max + 1, record_every))
    _meas_base = meas_seed if meas_seed is not None else base_seed + 1

    all_stats: dict[int, dict] = {}

    for W in N_values:
        _pert_site = pert_site if pert_site is not None else W // 2
        _probe_site = probe_site if probe_site is not None else 0
        raw: dict[float, dict[int, list]] = {
            p: {L: [] for L in L_values} for p in p_values
        }

        for p in p_values:
            for c_idx in range(n_circuits):
                circuit_seed = base_seed * 10_000 + c_idx
                for i_idx in range(n_init_states):
                    init_seed_val = base_seed * 10_000_000 + c_idx * 1_000 + i_idx
                    meas_seed_u_val  = _meas_base * 10_000_000 + c_idx * 1_000 + i_idx
                    meas_seed_ud_val = (_meas_base + 1) * 10_000_000 + c_idx * 1_000 + i_idx

                    for L in L_values:
                        n_meas_total = min(round(p * W * L), W * max(L - 1, 0))
                        shared_kwargs = dict(
                            L=L,
                            W=W,
                            pert_site=_pert_site,
                            pert_op=pert_op,
                            probe_site=_probe_site,
                            probe_angle=probe_angle,
                            n_meas_total=n_meas_total,
                            seed=circuit_seed,
                            init_seed=init_seed_val,
                            meas_seed=meas_seed_u_val,
                            meas_seed_ud=meas_seed_ud_val,
                            add_barrier=_add_barrier,
                        )
                        qc_pert = generate_time_reversal_breaking_random_brick_wall(
                            **shared_kwargs, unperturbed=False
                        )
                        qc_unpert = generate_time_reversal_breaking_random_brick_wall(
                            **shared_kwargs, unperturbed=True
                        )
                        probe_bit = qc_pert.n_bits - 1

                        tag_base = f"N{W}_p{p:.2f}_L{L:03d}_c{c_idx}_i{i_idx}"
                        shots_p, shots_u = _run_circuits(
                            [qc_pert, qc_unpert],
                            backend,
                            n_shots,
                            tags=[f"{tag_base}_pert", f"{tag_base}_unpert"],
                        )

                        out_p = 1 - 2 * shots_p[:, probe_bit].astype(int)
                        out_u = 1 - 2 * shots_u[:, probe_bit].astype(int)
                        raw[p][L].append(out_u.mean() - out_p.mean())
                        print(
                            f"N={W:2d}  p={p:.2f}  L={L:3d}  "
                            f"c={c_idx}  i={i_idx}  C_t={raw[p][L][-1]:.4f}"
                        )

            for L in L_values:
                raw[p][L] = np.array(raw[p][L])

        all_stats[W] = compute_C_t(raw)

    return all_stats


def sweep_single_shot_disorder(
    p_values: list[float],
    L_max: int,
    N_values: list[int] | None = None,
    n_realizations: int = 400,
    base_seed: int = 42,
    record_every: int = 5,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    backend=None,
    device_name: str | None = None,
) -> dict[int, dict[str, dict[float, dict[int, float]]]]:
    """
    Disorder-averaged sweep where every sample is a fully independent circuit
    executed for exactly 1 shot.

    Each of the n_realizations samples draws fresh gate angles, a fresh initial
    Pauli word, fresh U measurement sites, and fresh U† measurement sites from
    four independent RNG streams keyed by the realization index r. The perturbed
    and unperturbed variants of each realization are batched together per (N, p, L)
    into a single _run_circuits call.

    Parameters:
    p_values (list[float]): Measurement rates to sweep over.
    L_max (int): Maximum number of layers (time steps).
    N_values (list[int]): System sizes (must be even). Defaults to [8..16 step 2].
    n_realizations (int): Number of independent circuit configurations per
    (N, p, L) grid point. Total circuits per point = 2 * n_realizations.
    base_seed (int): Root seed. Per-realization seeds:
    gate angles   → (base_seed + 0) * 10_000_000 + r
    initial state → (base_seed + 1) * 10_000_000 + r
    U meas sites  → (base_seed + 2) * 10_000_000 + r
    U† meas sites → (base_seed + 3) * 10_000_000 + r
    record_every (int): Only record/execute at L = record_every, 2*record_every, ...
    pert_site, pert_op, probe_site, probe_angle: forwarded to the generator.
    backend / device_name: same semantics as sweep_over_all_disorder_axes.

    Returns:
    all_stats[N] = compute_C_t output with keys "C", "var", "cv",
    each mapping p → L → scalar float.
    """
    if N_values is None:
        N_values = list(range(8, 17, 2))

    if backend is None:
        if device_name is not None:
            backend = QuantinuumBackend(device_name)
        else:
            from pytket.extensions.qiskit import AerBackend
            backend = AerBackend()

    _add_barrier = isinstance(backend, QuantinuumBackend)
    print(f"Backend: {backend}  |  add_barrier: {_add_barrier}")

    L_values = list(range(record_every, L_max + 1, record_every))
    all_stats: dict[int, dict] = {}

    for W in N_values:
        _pert_site = pert_site if pert_site is not None else W // 2
        _probe_site = probe_site if probe_site is not None else 0
        raw: dict[float, dict[int, np.ndarray]] = {}

        for p in p_values:
            raw[p] = {}
            for L in L_values:
                n_meas = min(round(p * W * L), W * max(L - 1, 0))

                circuits_pert, circuits_unpert = [], []
                for r in range(n_realizations):
                    # Include L in the seed so each (r, L) point draws a
                    # completely fresh disorder instance — circuits at different
                    # depths share no angles, initial state, or measurement sites.
                    def _s(axis):
                        return (base_seed + axis) * 1_000_000_000 + L * n_realizations + r
                    kwargs = dict(
                        L=L,
                        W=W,
                        pert_site=_pert_site,
                        pert_op=pert_op,
                        probe_site=_probe_site,
                        probe_angle=probe_angle,
                        n_meas_total=n_meas,
                        seed=_s(0),
                        init_seed=_s(1),
                        meas_seed=_s(2),
                        meas_seed_ud=_s(3),
                        add_barrier=_add_barrier,
                    )
                    circuits_pert.append(
                        generate_time_reversal_breaking_random_brick_wall(
                            **kwargs, unperturbed=False
                        )
                    )
                    circuits_unpert.append(
                        generate_time_reversal_breaking_random_brick_wall(
                            **kwargs, unperturbed=True
                        )
                    )

                probe_bit = circuits_pert[0].n_bits - 1
                tag = f"N{W}_p{p:.2f}_L{L:03d}"
                all_shots = _run_circuits(
                    circuits_pert + circuits_unpert,
                    backend,
                    1,
                    tags=[f"{tag}_r{r}_pert"   for r in range(n_realizations)]
                       + [f"{tag}_r{r}_unpert" for r in range(n_realizations)],
                )

                out_p = np.array([1 - 2 * int(s[0, probe_bit]) for s in all_shots[:n_realizations]])
                out_u = np.array([1 - 2 * int(s[0, probe_bit]) for s in all_shots[n_realizations:]])
                raw[p][L] = out_u - out_p
                print(f"N={W:2d}  p={p:.2f}  L={L:3d}  C_t={raw[p][L].mean():.4f}")

        all_stats[W] = compute_C_t(raw)

    return all_stats


def plot_results(
    all_stats: dict[int, dict],
    L_values: list[int],
    p_values: list[float],
    N_values: list[int],
    figure_path_C: str = "C_vs_t.png",
    figure_path_cv: str = "cv_vs_t.png",
    fixed_T: int | None = None,
    fixed_N: int | None = None,
) -> None:
    """
    Produces two figures, each with three subplots showing every cross-section
    of the (p, T, N) space.

    Figure 1  —  C = 1-F (the signal):
      subplot 1: C vs T,  one line per N  (at p = p_values[0])
      subplot 2: C vs p,  one line per N  (at T = fixed_T)
      subplot 3: C vs T,  one line per p  (at N = fixed_N)

    Figure 2  —  CV = sqrt(Var)/|C| (self-averaging):
      same three-panel layout as Figure 1.

    Parameters:
    all_stats : output of compute_F, keyed all_stats[N][stat][p][L].
    L_values  : recorded L values (time steps).
    p_values  : measurement rates swept.
    N_values  : system sizes swept.
    figure_path_C  : save path for the C figure.
    figure_path_cv : save path for the CV figure.
    fixed_T   : T used for the "vs p" subplot. Defaults to last L.
    fixed_N   : N used for the "vs T by p" subplot. Defaults to largest N.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if fixed_T is None:
        fixed_T = L_values[-1]
    if fixed_N is None:
        fixed_N = N_values[-1]

    N_colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    p_colors = plt.cm.plasma(np.linspace(0, 1, max(len(p_values), 2)))

    for stat_key, ylabel, fig_path in [
        ("C",  r"$C_t = \langle\sigma_z\rangle_\mathrm{unpert} - \langle\sigma_z\rangle_\mathrm{pert}$", figure_path_C),
        ("cv", r"CV $= \sqrt{\mathrm{Var}}\,/\,|C_t|$", figure_path_cv),
    ]:
        with_errorbars = stat_key == "C"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        def _plot(ax, xs, ys, yerr, label, color, marker):
            if with_errorbars:
                ax.errorbar(
                    xs, ys, yerr=yerr, label=label, color=color,
                    marker=marker, capsize=3, capthick=0.8, elinewidth=0.8,
                )
            else:
                ax.plot(xs, ys, marker=marker, label=label, color=color)

        # ── Subplot 1: stat vs T, one line per N (p fixed to p_values[0]) ──
        ax = axes[0]
        p0 = p_values[0]
        for n_idx, N in enumerate(N_values):
            vals = [all_stats[N][stat_key][p0][L] for L in L_values]
            errs = [all_stats[N]["se"][p0][L]      for L in L_values]
            _plot(ax, L_values, vals, errs, f"N={N}", N_colors[n_idx], "o")
        ax.set_xlabel("T  (layers)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"vs T   (p = {p0:.2f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Subplot 2: stat vs p, one line per N (T fixed to fixed_T) ──
        ax = axes[1]
        for n_idx, N in enumerate(N_values):
            vals = [all_stats[N][stat_key][p][fixed_T] for p in p_values]
            errs = [all_stats[N]["se"][p][fixed_T]      for p in p_values]
            _plot(ax, p_values, vals, errs, f"N={N}", N_colors[n_idx], "s")
        ax.set_xlabel("p")
        ax.set_ylabel(ylabel)
        ax.set_title(f"vs p   (T = {fixed_T})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Subplot 3: stat vs T, one line per p (N fixed to fixed_N) ──
        ax = axes[2]
        for p_idx, p in enumerate(p_values):
            vals = [all_stats[fixed_N][stat_key][p][L] for L in L_values]
            errs = [all_stats[fixed_N]["se"][p][L]      for L in L_values]
            _plot(ax, L_values, vals, errs, f"p={p:.2f}", p_colors[p_idx], "^")
        ax.set_xlabel("T  (layers)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"vs T   (N = {fixed_N})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if with_errorbars:
            fig.suptitle(ylabel + r"  (error bars = $\pm$SE)", fontsize=11)
        else:
            fig.suptitle(ylabel, fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved → {fig_path}")

# def sweep_x_and_t(
#     L_max: int,
#     W: int,
#     p_values: list[float] | None = None,
#     probe_sites: list[int] | None = None,
#     n_shots: int = 100,
#     n_circuits: int = 5,
#     n_init_states: int = 5,
#     base_seed: int = 42,
#     meas_seed: int | None = None,
#     record_every: int = 5,
#     pert_site: int | None = None,
#     pert_op: str = "measure",
#     probe_angle: float = 0.5,
#     backend=None,
#     device_name: str | None = None,
# ) -> dict[float, dict[int, dict[int, np.ndarray]]]:
#     """
#     Disorder-averaged sweep over probe site x, time T, and measurement rate p.
#     The perturbation is placed at pert_site (default W//2); x is measured as
#     signed distance from pert_site so the OTOC wavefront is centred at x=0.

#     For a given realization (c_idx, i_idx), the meas_seed is fixed, so
#     changing p only moves the threshold on pre-drawn uniforms — angles and
#     measurement site draws are shared across all p values.

#     Returns:
#     raw[p][probe_site][L] = 1-D array of per-realization (pert_mean - unpert_mean).
#     Pass to compute_F (keyed by p then probe_site), then plot_otoc.
#     """
#     if p_values is None:
#         p_values = [0.5]
#     if pert_site is None:
#         pert_site = W // 2
#     if probe_sites is None:
#         probe_sites = [q for q in range(W) if q != pert_site]

#     if backend is None:
#         if device_name is not None:
#             backend = QuantinuumBackend(device_name)
#         else:
#             from pytket.extensions.qiskit import AerBackend

#             backend = AerBackend()

#     _add_barrier = isinstance(backend, QuantinuumBackend)
#     L_values = list(range(record_every, L_max + 1, record_every))
#     _meas_base = meas_seed if meas_seed is not None else base_seed + 1

#     raw: dict[float, dict[int, dict[int, list]]] = {
#         p: {ps: {L: [] for L in L_values} for ps in probe_sites} for p in p_values
#     }

#     for ps in probe_sites:
#         for c_idx in range(n_circuits):
#             circuit_seed = base_seed * 10_000 + c_idx
#             for i_idx in range(n_init_states):
#                 init_seed_val = base_seed * 10_000_000 + c_idx * 1_000 + i_idx
#                 # meas_seed fixed per realization so p is a pure threshold
#                 meas_seed_val = _meas_base * 10_000_000 + c_idx * 1_000 + i_idx

#                 for p in p_values:
#                     for L in L_values:
#                         shared_kwargs = dict(
#                             L=L,
#                             W=W,
#                             pert_site=pert_site,
#                             pert_op=pert_op,
#                             probe_site=ps,
#                             probe_angle=probe_angle,
#                             p=p,
#                             seed=circuit_seed,
#                             init_seed=init_seed_val,
#                             meas_seed=meas_seed_val,
#                             add_barrier=_add_barrier,
#                         )
#                         qc_pert = generate_time_reversal_breaking_random_brick_wall(
#                             **shared_kwargs, unperturbed=False
#                         )
#                         qc_unpert = generate_time_reversal_breaking_random_brick_wall(
#                             **shared_kwargs, unperturbed=True
#                         )
#                         probe_bit = qc_pert.n_bits - 1

#                         shots_p = _run_circuit_backend(qc_pert, backend, n_shots)
#                         shots_u = _run_circuit_backend(qc_unpert, backend, n_shots)

#                         out_p = 1 - 2 * shots_p[:, probe_bit].astype(int)
#                         out_u = 1 - 2 * shots_u[:, probe_bit].astype(int)

#                         raw[p][ps][L].append(out_p.mean() - out_u.mean())
#                         print(
#                             f"p={p:.2f}  x={ps - pert_site:+d}  L={L:3d}  "
#                             f"c={c_idx}  i={i_idx}  dF={raw[p][ps][L][-1]:.4f}"
#                         )

#     for p in p_values:
#         for ps in probe_sites:
#             for L in L_values:
#                 raw[p][ps][L] = np.array(raw[p][ps][L])

#     return raw


def plot_otoc(
    raw_xtp: dict[float, dict[int, dict[int, np.ndarray]]],
    pert_site: int,
    L_values: list[int],
    probe_sites: list[int],
    p_values: list[float],
    topmost_qubit: int | None = None,
    figure_path_cv: str = "otoc_cv.png",
    figure_path_2d: str = "otoc_F_vs_t.png",
    figure_path_3d: str = "otoc_3d.png",
) -> None:
    """
    Produces three figures from the output of sweep_x_and_t.

    Graph 1 — Self-averaging (CV vs T, fixed p=p_values[0]):
      One line per probe site x; shows uncertainty evolution across sites.

    Graph 2 — C vs T at topmost qubit, one line per p:
      Fixed x = topmost_qubit − pert_site (the furthest probe from pert_site).
      Different p values shown as separate curves to reveal how measurement
      rate controls the butterfly spreading speed.

    Graph 3 — 3D surface (x, T) → C (fixed p=p_values[0]):
      x = probe_site − pert_site (signed). x=0 inserted as C=0 (pert site).
      The rising wavefront slope gives the butterfly velocity v_B.

    Parameters:
    raw_xtp       : output of sweep_x_and_t (keyed p → probe_site → L → array).
    pert_site     : qubit index of the perturbation (origin x=0).
    L_values      : recorded time steps.
    probe_sites   : qubit indices used as probe (excludes pert_site).
    p_values      : measurement rates that were swept.
    topmost_qubit : probe site for graph 2. Defaults to max(probe_sites)
                    (highest-index qubit, furthest from pert_site at W//2).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if topmost_qubit is None:
        topmost_qubit = max(probe_sites)

    # Compute stats for every (p, probe_site) combination.
    # compute_F expects dict[outer_key, dict[inner_key, array]]; we call it per p.
    stats: dict[float, dict] = {p: compute_F(raw_xtp[p]) for p in p_values}
    # stats[p][probe_site]["F"/"var"/"cv"][L]

    p0 = p_values[0]
    x_top = topmost_qubit - pert_site
    x_sorted = sorted(probe_sites, key=lambda ps: ps - pert_site)
    x_coords = [ps - pert_site for ps in x_sorted]
    x_colors = plt.cm.viridis(np.linspace(0, 1, len(x_sorted)))
    p_colors = plt.cm.plasma(np.linspace(0, 1, max(len(p_values), 2)))

    # ── Graph 1: CV vs T at topmost qubit, one line per p ──────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for p_idx, p in enumerate(p_values):
        cv_vals = [stats[p][topmost_qubit]["cv"][L] for L in L_values]
        ax.plot(
            L_values, cv_vals, marker="o", label=f"p={p:.2f}", color=p_colors[p_idx]
        )
    ax.set_xlabel("T  (layers)")
    ax.set_ylabel(r"CV $= \sqrt{\mathrm{Var}}\,/\,|C|$")
    ax.set_title(f"Self-averaging: CV vs T  (x = {x_top:+d}, topmost qubit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path_cv, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path_cv}")

    # ── Graph 2: C vs T at topmost qubit, one line per p ───────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for p_idx, p in enumerate(p_values):
        C_vals = [stats[p][topmost_qubit]["F"][L] for L in L_values]
        ax.plot(L_values, C_vals, marker="o", label=f"p={p:.2f}", color=p_colors[p_idx])
    ax.set_xlabel("T  (layers)")
    ax.set_ylabel(r"$C = 1 - \mathcal{F}$")
    ax.set_title(f"OTOC spreading: C vs T  (x = {x_top:+d}, topmost qubit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path_2d, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path_2d}")

    # ── Graph 3: 3D surface (x, T) → C (fixed p=p0) ────────────────────────
    all_x = sorted(set(x_coords + [0]))
    T_arr = np.array(L_values, dtype=float)
    X_arr = np.array(all_x, dtype=float)
    X_grid, T_grid = np.meshgrid(X_arr, T_arr)
    C_grid = np.zeros_like(X_grid)

    for j, x in enumerate(all_x):
        ps = x + pert_site
        if ps in stats[p0]:
            for i, L in enumerate(L_values):
                C_grid[i, j] = stats[p0][ps]["F"][L]

    fig = plt.figure(figsize=(9, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(
        X_grid,
        T_grid,
        C_grid,
        cmap="viridis",
        edgecolor="none",
        alpha=0.85,
    )
    fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1, label=r"$C = 1 - \mathcal{F}$")
    ax3d.set_xlabel("x  (probe − pert site)")
    ax3d.set_ylabel("T  (layers)")
    ax3d.set_zlabel(r"$C$")
    ax3d.set_title(rf"OTOC butterfly front: $C(x,T)$  (p = {p0:.2f})")
    fig.tight_layout()
    fig.savefig(figure_path_3d, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path_3d}")

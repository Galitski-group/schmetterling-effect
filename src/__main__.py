import datetime
from collections import Counter
import numpy as np
import qnexus as qnx
import qnexus.exceptions as qnx_exc
from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend


def build_aer_noise_model(
    p1q: float | None = None,
    p2q: float | None = None,
    pm:  float | None = None,
    T1:  float | None = None,
    T2:  float | None = None,
    t_gate_1q: float = 50e-9,
    t_gate_2q: float = 300e-9,
    n_qubits: int = 20,
):
    """
    Build a per-qubit noise model compatible with pytket-qiskit >= 0.60 / AerBackend.

    pytket's AerBackend (v0.60+) rejects add_all_qubit_quantum_error; errors must
    be registered per qubit.  n_qubits should be >= the widest circuit you run;
    extra qubits are harmless.

    Parameters
    ----------
    p1q      : depolarizing error rate on single-qubit gates (None = skip)
    p2q      : depolarizing error rate on two-qubit gates    (None = skip)
    pm       : symmetric readout bitflip probability          (None = skip)
    T1       : longitudinal relaxation time in seconds        (None = skip)
    T2       : transverse relaxation time in seconds (must be <= 2*T1, None = skip)
    t_gate_1q: single-qubit gate duration in seconds (default 50 ns)
    t_gate_2q: two-qubit gate duration in seconds    (default 300 ns)
    n_qubits : number of qubits to register errors for (default 20)

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error,
    )

    if T1 is not None and T2 is not None and T2 > 2 * T1:
        raise ValueError(f"T2 ({T2}) must be <= 2*T1 ({2*T1}) for a physical channel.")

    nm = NoiseModel()
    # Include both old ("u1","u2","u3") and new ("u") Qiskit single-qubit gate names
    # so that pytket TK1 compilation works regardless of Qiskit version.
    _1q_gates = ["u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z", "s", "t"]
    _2q_gates = ["cx", "cz", "ecr", "rzz"]

    def _compose(depol_err, relax_err):
        if depol_err is not None and relax_err is not None:
            return depol_err.compose(relax_err)
        return depol_err if depol_err is not None else relax_err

    relax_1q = (
        thermal_relaxation_error(T1, T2, t_gate_1q)
        if T1 is not None and T2 is not None else None
    )
    relax_2q = (
        thermal_relaxation_error(T1, T2, t_gate_2q).expand(
            thermal_relaxation_error(T1, T2, t_gate_2q)
        )
        if T1 is not None and T2 is not None else None
    )

    err_1q = _compose(depolarizing_error(p1q, 1) if p1q else None, relax_1q)
    err_2q = _compose(depolarizing_error(p2q, 2) if p2q else None, relax_2q)

    for q in range(n_qubits):
        if err_1q is not None:
            nm.add_quantum_error(err_1q, _1q_gates, [q])
        if pm:
            nm.add_readout_error(ReadoutError([[1 - pm, pm], [pm, 1 - pm]]), [q])

    if err_2q is not None:
        for q1 in range(n_qubits):
            for q2 in range(n_qubits):
                if q1 != q2:
                    nm.add_quantum_error(err_2q, _2q_gates, [q1, q2])

    return nm


def _sample_cartan(rng) -> tuple[float, float, float]:
    """
    Sample (cx, cy, cz) from the Haar-induced Weyl-chamber density via
    rejection sampling.  Uses `rng` exclusively for reproducibility.
    """
    half_pi = np.pi / 2
    while True:
        ux, uy, uz, ua = rng.uniform(0, 1, size=4)
        cx = half_pi * ux ** (1 / 3)
        cy = cx * np.sqrt(uy)
        cz = cy * (2 * uz - 1)
        w = abs(
            np.sin(cx + cy) * np.sin(cx - cy)
            * np.sin(cx + cz) * np.sin(cx - cz)
            * np.sin(cy + cz) * np.sin(cy - cz)
        )
        if ua < 4 * w:
            return cx, cy, cz


def _haar_su4_params(rng) -> tuple:
    """
    Sample parameters for a Haar-random SU(4) in native TK1+TK2 form.

    The KAK decomposition gives:
        U = (TK1_post_i ⊗ TK1_post_j) · TK2(t1,t2,t3) · (TK1_pre_i ⊗ TK1_pre_j)

    TK1(a,b,c) = Rz(aπ)·Rx(bπ)·Rz(cπ)  (pytket half-turns)
    TK2(t1,t2,t3) = exp(-iπ/2 (t1·XX + t2·YY + t3·ZZ))

    Inverse (for U†): reverse layer order and negate all angles:
        TK1(a,b,c)† = TK1(-c,-b,-a)
        TK2(t1,t2,t3)† = TK2(-t1,-t2,-t3)

    Returns (pre_i, pre_j, tk2, post_i, post_j) where each is a 3-tuple
    of half-turn angles.
    """
    def _su2_tk1(rng_):
        # Haar-random SU(2) as ZYZ, converted to TK1's ZXZ parameterisation.
        # Rz(φ)·Ry(θ)·Rz(λ) = Rz(φ+π/2)·Rx(θ)·Rz(λ-π/2)
        #                     = TK1(φ/π+½, θ/π, λ/π-½)
        phi   = rng_.uniform(0, 2 * np.pi)
        lam   = rng_.uniform(0, 2 * np.pi)
        theta = np.arccos(1 - 2 * rng_.uniform(0, 1))
        return phi / np.pi + 0.5, theta / np.pi, lam / np.pi - 0.5

    pre_i  = _su2_tk1(rng)
    pre_j  = _su2_tk1(rng)
    cx, cy, cz = _sample_cartan(rng)
    post_i = _su2_tk1(rng)
    post_j = _su2_tk1(rng)
    # TK2 angles in half-turns: TK2(cx/π,cy/π,cz/π) = Rxx(cx)·Ryy(cy)·Rzz(cz)
    return pre_i, pre_j, (cx / np.pi, cy / np.pi, cz / np.pi), post_i, post_j


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


def _build_otoc_circuit_impl(
    layer_pairs_fn,
    L: int,
    W: int,
    unperturbed: bool,
    pert_site: int,
    pert_op: str,
    probe_site: int,
    probe_angle: float,
    add_barrier: bool,
    do_reset: bool,
    seed,
    init_seed,
    meas_seed,
    meas_seed_ud,
    p: float,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    qc: Circuit | None = None,
    qubit_offset: int = 0,
    scratch_bit: int = 0,
    probe_bit: int = 1,
) -> Circuit:
    """Shared implementation for brickwall and bowtie OTOC circuits.

    layer_pairs_fn(l) -> list[(i,j)]  defines which qubit pairs gate layer l.
    L_back caps how many of the L reversed (U†) layers are actually applied
    after the perturbation (None = all L, the standard full echo). Forward
    U is always built to its full L layers regardless of L_back; only the
    backward evolution is truncated. Building the same forward U with
    L_back = 0..L and measuring after each traces the correlator's build-up
    during the unwinding half of the echo — before the first backward layer
    (L_back=0) the probe carries no perturbation signal by causality.

    qc, qubit_offset, scratch_bit, probe_bit: internal hooks used by
    build_parallel_otoc_circuit to append several independent instances of
    this circuit onto one larger Circuit, each on its own disjoint qubit
    range (qubit_offset) with its own probe_bit, while all instances share
    one scratch_bit — safe because scratch outcomes are always discarded,
    so concurrent writers don't need distinct bits. Leave these at their
    defaults (qc=None) to build a standalone 2-bit Circuit exactly as before.
    All other parameters are identical to the public build_* functions.
    """
    _init_ops = init_ops if init_ops is not None else ["I", "X", "Y", "Z"]
    rng = np.random.default_rng(seed=seed)
    init_rng = np.random.default_rng(seed=init_seed if init_seed is not None else seed)
    _ms_u = meas_seed if meas_seed is not None else seed
    _ms_ud = meas_seed_ud if meas_seed_ud is not None else (_ms_u + 1 if _ms_u is not None else 1)
    meas_rng_u  = np.random.default_rng(seed=_ms_u)
    meas_rng_ud = np.random.default_rng(seed=_ms_ud)

    # Stochastic measurements use Bernoulli(p) per eligible site per layer.
    # The final layer of U (index L-1, directly before the perturbation) is
    # never measured. U and U† draw independently via their own RNGs.
    forward = []
    for l in range(L):
        pairs_in_layer = layer_pairs_fn(l)
        paired = sorted({q for pair in pairs_in_layer for q in pair})

        if l < L - 1 and p > 0.0 and paired:
            chosen_u  = {q for q, hit in zip(paired, meas_rng_u.random(len(paired))  < p) if hit}
            chosen_ud = {q for q, hit in zip(paired, meas_rng_ud.random(len(paired)) < p) if hit}
        else:
            chosen_u  = set()
            chosen_ud = set()

        pairs_data = []
        for i, j in pairs_in_layer:
            pre_i, pre_j, tk2, post_i, post_j = _haar_su4_params(rng)
            pairs_data.append((pre_i, pre_j, tk2, post_i, post_j, i, j,
                                i in chosen_u,  j in chosen_u,
                                i in chosen_ud, j in chosen_ud))
        forward.append(pairs_data)

    # bit `scratch_bit` : receives every mid-circuit measurement (discarded)
    # bit `probe_bit`   : final readout only
    if qc is None:
        qc = Circuit(W, 2)
    o = qubit_offset
    barrier_sequence = [o + q for q in range(W)]
    all_bits = sorted({scratch_bit, probe_bit})
    pert_site_abs = o + pert_site
    probe_site_abs = o + probe_site

    # State preparation: probe_site → Ry(probe_angle·π); all others → uniform
    # sample from init_ops. "I" means leave the qubit in |0⟩; any other string
    # is applied as a pytket single-qubit gate.
    qc.Ry(probe_angle, probe_site_abs)
    for q in range(W):
        if q == probe_site:
            continue
        op = init_rng.choice(_init_ops)
        if op != "I":
            getattr(qc, op)(o + q)

    # Build U: Haar SU(4) decomposed as TK1·TK2·TK1, then stochastic measurements.
    for pairs_data in forward:
        for pre_i, pre_j, tk2, post_i, post_j, i, j, mi_u, mj_u, _, _ in pairs_data:
            ii, jj = o + i, o + j
            qc.TK1(*pre_i, ii); qc.TK1(*pre_j, jj)
            qc.TK2(*tk2, ii, jj)
            qc.TK1(*post_i, ii); qc.TK1(*post_j, jj)
            if mi_u: qc.Measure(ii, scratch_bit)
            if mj_u: qc.Measure(jj, scratch_bit)
        if add_barrier:
            qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Mid-circuit perturbation (suppressed when unperturbed=True).
    if not unperturbed:
        if pert_op == "measure":
            qc.Measure(pert_site_abs, scratch_bit)
            if do_reset:
                qc.Reset(pert_site_abs)
        else:
            getattr(qc, pert_op)(pert_site_abs)

    if add_barrier:
        qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Apply U†: reverse layer order, negate all TK1/TK2 angles.
    # TK1(a,b,c)† = TK1(-c,-b,-a);  TK2(t1,t2,t3)† = TK2(-t1,-t2,-t3)
    # L_back truncates how many of the L reversed layers actually run.
    _L_back = L if L_back is None else L_back
    for pairs_data in list(reversed(forward))[:_L_back]:
        for pre_i, pre_j, tk2, post_i, post_j, i, j, _, _, mi_ud, mj_ud in reversed(pairs_data):
            ii, jj = o + i, o + j
            qc.TK1(-post_i[2], -post_i[1], -post_i[0], ii)
            qc.TK1(-post_j[2], -post_j[1], -post_j[0], jj)
            qc.TK2(-tk2[0], -tk2[1], -tk2[2], ii, jj)
            qc.TK1(-pre_i[2], -pre_i[1], -pre_i[0], ii)
            qc.TK1(-pre_j[2], -pre_j[1], -pre_j[0], jj)
            if mj_ud: qc.Measure(jj, scratch_bit)
            if mi_ud: qc.Measure(ii, scratch_bit)
        if add_barrier:
            qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    qc.Measure(probe_site_abs, probe_bit)
    return qc


def _otoc_circuit_args(
    L, W, unperturbed, pert_site, pert_op, probe_site, probe_angle,
    add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
):
    """Validate and resolve defaults shared by all build_*_otoc_circuit functions."""
    if pert_op != "measure" and pert_op not in _PERT_UNITARY_OPS:
        raise ValueError(
            f"Unsupported pert_op {pert_op!r}. Use 'measure' or one of {_PERT_UNITARY_OPS}."
        )
    if W % 2 != 0:
        raise ValueError("Circuit only defined for even # of qubits.")
    if pert_site is None:
        pert_site = W // 2
    if probe_site is None:
        probe_site = 0
    if probe_site == pert_site:
        raise ValueError(f"probe_site and pert_site must differ (both are {probe_site}).")
    return pert_site, probe_site


def build_brickwall_otoc_circuit(
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
    p: float = 0.0,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
) -> Circuit:
    """Loschmidt-echo OTOC circuit with a standard alternating brick-wall layout.

    Even layers pair qubits (0,1),(2,3),(4,5),…
    Odd  layers pair qubits (1,2),(3,4),(5,6),…

    U (L layers) is followed by the perturbation at pert_site and then U†
    (same L layers in reverse with negated angles).  Mid-circuit measurements
    are placed independently in U and U† via Bernoulli(p) per eligible site per
    layer (the last layer before the perturbation is never measured).

    L_back : caps how many of the L reversed (U†) layers are actually built,
    for tracing the correlator through partial backward evolution (None =
    full L, the standard echo). See _build_otoc_circuit_impl.

    Parameters mirror build_bowtie_otoc_circuit — see that docstring for details.
    """
    pert_site, probe_site = _otoc_circuit_args(
        L, W, unperturbed, pert_site, pert_op, probe_site, probe_angle,
        add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
    )

    def layer_pairs(l):
        if l % 2 == 0:
            return [(2 * w, 2 * w + 1) for w in range(W // 2)]
        else:
            return [(2 * w + 1, 2 * w + 2) for w in range(W // 2 - 1)]

    return _build_otoc_circuit_impl(
        layer_pairs, L, W, unperturbed, pert_site, pert_op, probe_site,
        probe_angle, add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
        init_ops=init_ops, L_back=L_back,
    )


def build_bowtie_otoc_circuit(
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
    p: float = 0.0,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
) -> Circuit:
    """Loschmidt-echo OTOC circuit with a converging-diamond (bowtie) layout.

    Layer l of U activates only qubits in the range [l, W-1-l], so the active
    region shrinks by one qubit on each side every layer:

        l=0 (W=8):  (0,1),(2,3),(4,5),(6,7)   — full width
        l=1:        (1,2),(3,4),(5,6)
        l=2:        (2,3),(4,5)
        l=3:        (3,4)                       — single central pair
        l≥4:        (empty)

    U† applies the same layers in reverse order, so U converges toward the
    center while U† diverges back out — together they look like a bowtie or
    diamond when drawn as a spacetime diagram.

    For L > W//2 the extra layers simply have no gates (the diamond has closed).
    The maximum meaningful depth is W//2.

    Parameters:
    L (int): Number of layers per channel (U and U† each get L layers).
    W (int): Number of qubits (must be even).
    unperturbed (bool): Suppress the mid-circuit perturbation (keep measurements).
    pert_site (int): Qubit for the perturbation; defaults to W//2 (0-indexed).
    pert_op (str): "measure" or any pytket single-qubit gate ("X","Y","Z","H","S","T").
    probe_site (int): Qubit read out at the end; defaults to 0. Must ≠ pert_site.
    probe_angle (float): Ry half-turns applied to probe_site during state prep.
    add_barrier (bool): Insert barriers between layers (required for Quantinuum).
    do_reset (bool): Reset pert_site after a "measure" perturbation.
    seed (int): RNG seed for Haar-random gate angles.
    init_seed (int): RNG seed for the initial random Pauli word.
    meas_seed (int): RNG seed for U measurement site selection.
    meas_seed_ud (int): RNG seed for U† measurement site selection.
    p (float): Per-site Bernoulli measurement probability in layers 0..L-2.
    init_ops (list[str]): Gates to sample uniformly for state prep on each
    non-probe qubit. "I" means leave the qubit in |0⟩; any other string is
    applied as a pytket single-qubit gate. Defaults to ["I","X","Y","Z"].
    Example for bowtie: ["I","X"].
    L_back (int): Caps how many of the L reversed (U†) layers are actually
    built, for tracing the correlator through partial backward evolution
    (None = full L, the standard echo).

    Returns:
    Circuit: State-prep → U (bowtie, converging) → Perturbation →
             U† (bowtie, diverging) → probe measurement.
    """
    pert_site, probe_site = _otoc_circuit_args(
        L, W, unperturbed, pert_site, pert_op, probe_site, probe_angle,
        add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
    )

    def layer_pairs(l):
        # Active range [l, W-1-l] shrinks by 1 from each boundary every layer.
        # Pairs start at index l so offset naturally alternates even/odd with l.
        lo, hi = l, W - 1 - l
        if lo >= hi:
            return []
        return [(a, a + 1) for a in range(lo, hi, 2) if a + 1 <= hi]

    return _build_otoc_circuit_impl(
        layer_pairs, L, W, unperturbed, pert_site, pert_op, probe_site,
        probe_angle, add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
        init_ops=init_ops, L_back=L_back,
    )


# Backwards-compatible alias — sweep functions use this name.
build_otoc_circuit = build_brickwall_otoc_circuit


def _build_lightcone_probe_all_circuit_impl(
    layer_pairs_fn,
    L: int,
    W: int,
    unperturbed: bool,
    pert_site: int,
    pert_op: str,
    add_barrier: bool,
    do_reset: bool,
    seed,
    init_seed,
    meas_seed,
    meas_seed_ud,
    p: float,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    qc: Circuit | None = None,
    qubit_offset: int = 0,
    scratch_bit: int = 0,
    probe_bit_base: int = 1,
) -> Circuit:
    """Shared implementation for the "measure every site" light-cone circuit.

    Same Loschmidt-echo structure as _build_otoc_circuit_impl (forward U,
    mid-circuit perturbation, backward U†), but built for an all-sites
    read-out protocol instead of a single probe qubit:

      - State prep samples each qubit independently from init_ops (default
        ["I", "X"] — computational-basis-only, no Y/Z superposition prep)
        INCLUDING pert_site. There is no single distinguished probe qubit
        to exclude here, since every site is read out at the end.
      - Every qubit is measured into its own classical bit at the end
        (bits probe_bit_base .. probe_bit_base+W-1, one per qubit; bit
        scratch_bit is the shared scratch bit for discarded mid-circuit
        outcomes, matching _build_otoc_circuit_impl's scratch/probe split
        convention).
      - Immediately before that final measurement, any qubit whose state
        prep drew "X" gets one more X applied first. Applying X right
        before a Z-basis measurement is an exact, deterministic bit-flip
        of that qubit's own outcome (not a statistical approximation —
        X anticommutes with the Z measurement basis shot-by-shot), so the
        net effect is: a qubit prepped in I (|0>) reads out as usual, one
        prepped in X (|1>) has its readout flipped back, and either way
        the final bit means "matches vs deviates from that site's OWN
        initial condition" — uniform across every site regardless of
        which of I/X it individually started in.

    layer_pairs_fn, L_back, add_barrier, do_reset, seed, init_seed,
    meas_seed, meas_seed_ud, p: identical meaning to _build_otoc_circuit_impl.

    qc, qubit_offset, scratch_bit, probe_bit_base: internal hooks used by
    build_parallel_lightcone_probe_all_circuit to append several independent
    instances of this circuit onto one larger Circuit, each on its own
    disjoint qubit range (qubit_offset) with its own W-bit probe_bit_base
    range, while all instances in a width-block share one scratch_bit —
    safe because scratch outcomes are always discarded. Leave these at
    their defaults (qc=None) to build a standalone (W+1)-bit Circuit
    exactly as before.
    """
    _init_ops = init_ops if init_ops is not None else ["I", "X"]
    rng = np.random.default_rng(seed=seed)
    init_rng = np.random.default_rng(seed=init_seed if init_seed is not None else seed)
    _ms_u = meas_seed if meas_seed is not None else seed
    _ms_ud = meas_seed_ud if meas_seed_ud is not None else (_ms_u + 1 if _ms_u is not None else 1)
    meas_rng_u  = np.random.default_rng(seed=_ms_u)
    meas_rng_ud = np.random.default_rng(seed=_ms_ud)

    # Stochastic measurements use Bernoulli(p) per eligible site per layer,
    # identical to _build_otoc_circuit_impl — see its comment for details.
    forward = []
    for l in range(L):
        pairs_in_layer = layer_pairs_fn(l)
        paired = sorted({q for pair in pairs_in_layer for q in pair})

        if l < L - 1 and p > 0.0 and paired:
            chosen_u  = {q for q, hit in zip(paired, meas_rng_u.random(len(paired))  < p) if hit}
            chosen_ud = {q for q, hit in zip(paired, meas_rng_ud.random(len(paired)) < p) if hit}
        else:
            chosen_u  = set()
            chosen_ud = set()

        pairs_data = []
        for i, j in pairs_in_layer:
            pre_i, pre_j, tk2, post_i, post_j = _haar_su4_params(rng)
            pairs_data.append((pre_i, pre_j, tk2, post_i, post_j, i, j,
                                i in chosen_u,  j in chosen_u,
                                i in chosen_ud, j in chosen_ud))
        forward.append(pairs_data)

    # bit scratch_bit                        : receives every mid-circuit
    #                                           measurement (discarded)
    # bits probe_bit_base..probe_bit_base+W-1 : final per-site readout,
    #                                           bit (probe_bit_base+q) <-> qubit (o+q)
    if qc is None:
        qc = Circuit(W, W + 1)
    o = qubit_offset
    barrier_sequence = [o + q for q in range(W)]
    all_bits = sorted({scratch_bit} | {probe_bit_base + q for q in range(W)})
    pert_site_abs = o + pert_site

    # State preparation: every qubit (including pert_site) samples
    # independently from init_ops. Track which ones drew "X" so the final
    # readout can undo it (see docstring above).
    was_x = [False] * W
    for q in range(W):
        op = init_rng.choice(_init_ops)
        if op != "I":
            getattr(qc, op)(o + q)
            was_x[q] = (op == "X")

    if add_barrier:
        qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Build U: Haar SU(4) decomposed as TK1·TK2·TK1, then stochastic measurements.
    for pairs_data in forward:
        for pre_i, pre_j, tk2, post_i, post_j, i, j, mi_u, mj_u, _, _ in pairs_data:
            ii, jj = o + i, o + j
            qc.TK1(*pre_i, ii); qc.TK1(*pre_j, jj)
            qc.TK2(*tk2, ii, jj)
            qc.TK1(*post_i, ii); qc.TK1(*post_j, jj)
            if mi_u: qc.Measure(ii, scratch_bit)
            if mj_u: qc.Measure(jj, scratch_bit)
        if add_barrier:
            qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Mid-circuit perturbation (suppressed when unperturbed=True).
    if not unperturbed:
        if pert_op == "measure":
            qc.Measure(pert_site_abs, scratch_bit)
            if do_reset:
                qc.Reset(pert_site_abs)
        else:
            getattr(qc, pert_op)(pert_site_abs)

    if add_barrier:
        qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Apply U†: reverse layer order, negate all TK1/TK2 angles.
    # L_back truncates how many of the L reversed layers actually run.
    _L_back = L if L_back is None else L_back
    for pairs_data in list(reversed(forward))[:_L_back]:
        for pre_i, pre_j, tk2, post_i, post_j, i, j, _, _, mi_ud, mj_ud in reversed(pairs_data):
            ii, jj = o + i, o + j
            qc.TK1(-post_i[2], -post_i[1], -post_i[0], ii)
            qc.TK1(-post_j[2], -post_j[1], -post_j[0], jj)
            qc.TK2(-tk2[0], -tk2[1], -tk2[2], ii, jj)
            qc.TK1(-pre_i[2], -pre_i[1], -pre_i[0], ii)
            qc.TK1(-pre_j[2], -pre_j[1], -pre_j[0], jj)
            if mj_ud: qc.Measure(jj, scratch_bit)
            if mi_ud: qc.Measure(ii, scratch_bit)
        if add_barrier:
            qc.add_barrier(qubits=barrier_sequence, bits=all_bits)

    # Final readout: undo the "X" sites' preparation right before measuring
    # them, so every site's bit reads relative to ITS OWN initial condition.
    for q in range(W):
        if was_x[q]:
            qc.X(o + q)
        qc.Measure(o + q, probe_bit_base + q)

    return qc


def build_brickwall_lightcone_probe_all_circuit(
    L: int,
    W: int,
    unperturbed: bool = False,
    pert_site: int | None = None,
    pert_op: str = "measure",
    add_barrier: bool = True,
    do_reset: bool = False,
    seed: int | None = None,
    init_seed: int | None = None,
    meas_seed: int | None = None,
    meas_seed_ud: int | None = None,
    p: float = 0.0,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    anchor_parity_at_perturbation: str | None = None,
) -> Circuit:
    """Brickwall Loschmidt-echo circuit that measures EVERY site, for the
    all-sites light-cone sweep — see _build_lightcone_probe_all_circuit_impl
    for the exact state-prep/read-out protocol.

    Same brickwall layer geometry as build_brickwall_otoc_circuit (even
    layers pair (0,1),(2,3),...; odd layers pair (1,2),(3,4),...). Parameters
    mirror build_brickwall_otoc_circuit minus probe_site/probe_angle — there
    is no single distinguished probe here, every site is one, each read out
    relative to its own I/X initial condition. init_ops defaults to ["I","X"]
    (computational-basis-only state prep), not the 4-Pauli
    ["I","X","Y","Z"] the single-probe builders use. L_back caps how many
    of the L reversed (U†) layers are built (None = full L, standard echo).

    anchor_parity_at_perturbation : None (default), "even", or "odd".

        None: layer parity is indexed from state prep (forward layer l=0
        is always even-parity), so the layer immediately adjacent to the
        perturbation — forward's LAST layer, l=L-1, which backward's FIRST
        step always replays — has parity (L-1) % 2, flipping between
        even/odd depending on whether L itself is odd or even. Since
        pert_site sits at a fixed (even) qubit index, this means which
        neighbour it first couples to right after the perturbation isn't
        consistent across different L, producing a left/right imbalance in
        the light-cone growth that doesn't fully wash out even at larger L
        (confirmed empirically: e.g. L=2 showed x=+2 at 0.0 vs x=-2 at 0.88).

        "even" / "odd": instead index parity from the PERTURBATION
        outward, forcing the layer adjacent to it (forward l=L-1) to
        always be the CHOSEN parity, regardless of L — "even" couples
        pert_site to its right neighbour (x=+1) first every time, "odd" to
        its left neighbour (x=-1) first every time. Implemented as
        ((L - 1 - l) + (1 if odd else 0)) % 2 in place of the default l % 2.
        The far boundary (forward's first layer, next to state prep) is
        what varies with L instead — which doesn't matter for the
        light-cone geometry, since nothing about causal spreading from the
        perturbation depends on the state-prep end.

        Neither setting produces true x<->-x symmetry — a single perturbed
        qubit can only bond with one neighbour per layer, so one direction
        always gets a one-layer head start over the other. What this
        controls is WHICH direction gets that head start, consistently
        across every L, instead of it alternating unpredictably by L's
        own parity.
    """
    if pert_op != "measure" and pert_op not in _PERT_UNITARY_OPS:
        raise ValueError(
            f"Unsupported pert_op {pert_op!r}. Use 'measure' or one of {_PERT_UNITARY_OPS}."
        )
    if W % 2 != 0:
        raise ValueError("Circuit only defined for even # of qubits.")
    if anchor_parity_at_perturbation not in (None, "even", "odd"):
        raise ValueError(
            f"anchor_parity_at_perturbation must be None, 'even', or 'odd', "
            f"got {anchor_parity_at_perturbation!r}."
        )
    if pert_site is None:
        pert_site = W // 2

    def layer_pairs(l):
        if anchor_parity_at_perturbation is None:
            parity = l % 2
        else:
            target = 0 if anchor_parity_at_perturbation == "even" else 1
            parity = (L - 1 - l + target) % 2
        if parity == 0:
            return [(2 * w, 2 * w + 1) for w in range(W // 2)]
        else:
            return [(2 * w + 1, 2 * w + 2) for w in range(W // 2 - 1)]

    return _build_lightcone_probe_all_circuit_impl(
        layer_pairs, L, W, unperturbed, pert_site, pert_op,
        add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
        init_ops=init_ops, L_back=L_back,
    )


def build_bowtie_lightcone_probe_all_circuit(
    L: int,
    W: int,
    unperturbed: bool = False,
    pert_site: int | None = None,
    pert_op: str = "measure",
    add_barrier: bool = True,
    do_reset: bool = False,
    seed: int | None = None,
    init_seed: int | None = None,
    meas_seed: int | None = None,
    meas_seed_ud: int | None = None,
    p: float = 0.0,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
) -> Circuit:
    """Bowtie-topology counterpart to build_brickwall_lightcone_probe_all_circuit
    — see that function and build_bowtie_otoc_circuit for the converging/
    diverging layer geometry, and _build_lightcone_probe_all_circuit_impl
    for the all-sites state-prep/read-out protocol.
    """
    if pert_op != "measure" and pert_op not in _PERT_UNITARY_OPS:
        raise ValueError(
            f"Unsupported pert_op {pert_op!r}. Use 'measure' or one of {_PERT_UNITARY_OPS}."
        )
    if W % 2 != 0:
        raise ValueError("Circuit only defined for even # of qubits.")
    if pert_site is None:
        pert_site = W // 2

    def layer_pairs(l):
        lo, hi = l, W - 1 - l
        if lo >= hi:
            return []
        return [(a, a + 1) for a in range(lo, hi, 2) if a + 1 <= hi]

    return _build_lightcone_probe_all_circuit_impl(
        layer_pairs, L, W, unperturbed, pert_site, pert_op,
        add_barrier, do_reset, seed, init_seed, meas_seed, meas_seed_ud, p,
        init_ops=init_ops, L_back=L_back,
    )


def build_parallel_lightcone_probe_all_circuit(
    L: int,
    W: int,
    unperturbed: bool,
    n_qubits_budget: int,
    realizations: list[dict],
    pert_site: int | None = None,
    pert_op: str = "measure",
    add_barrier: bool = True,
    do_reset: bool = False,
    p: float = 0.0,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    topology: str = "brickwall",
    anchor_parity_at_perturbation: str | None = None,
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_gates: int | None = None,
) -> tuple[Circuit, int, int]:
    """
    Tiled analogue of build_parallel_otoc_circuit, but for the "measure
    every site" light-cone circuit (build_*_lightcone_probe_all_circuit)
    instead of the single-probe OTOC circuit — packs MANY independent
    disorder realizations onto one larger circuit so one hardware
    submission returns every site's outcome for every realization at
    once, instead of needing one circuit per realization. Same
    width/depth tiling scheme as build_parallel_otoc_circuit (spatial
    replication across n_qubits_budget // W disjoint blocks, temporal
    replication via Reset once width alone can't fit everything) — see
    that function's docstring for the full width/depth explanation, which
    applies unchanged here.

    The one structural difference: build_parallel_otoc_circuit needs only
    ONE probe bit per realization (a single distinguished probe_site), but
    this circuit measures every one of its W qubits, so each realization
    needs its OWN W-bit block of final-readout bits instead of a single
    bit. Classical bit layout: `width` shared-per-block scratch bits
    (index 0..width-1, one per width-block, same
    write-after-write-dependency rationale as build_parallel_otoc_circuit
    — see its own docstring), followed by `n_realizations` consecutive
    W-bit ranges, one per realization, in the same row-major fill order
    build_parallel_otoc_circuit uses (realization idx's bits start at
    width + idx*W). This means classical bits are consumed ~W times
    faster per realization than the single-probe tiled circuit — real
    device classical-register limits (e.g. 64 on H2-2) bind much sooner
    here, so expect noticeably smaller chunks out of
    build_parallel_lightcone_probe_all_circuits_batched than the
    single-probe equivalent at the same W.

    Parameters mirror build_brickwall_lightcone_probe_all_circuit /
    build_bowtie_lightcone_probe_all_circuit (seed/init_seed/meas_seed/
    meas_seed_ud now come from `realizations` instead of single scalars —
    pert_site/p/init_ops/topology/anchor_parity_at_perturbation are shared
    across all instances, since only the random draws should differ
    between realizations), plus n_qubits_budget/realizations/max_depth/
    max_classical_bits/max_su4_gates — identical meaning to
    build_parallel_otoc_circuit's own (same ValueError-on-exceeded
    behaviour, same "callers should split into multiple circuits instead
    of raising the cap" philosophy). anchor_parity_at_perturbation is
    brickwall-only (see build_brickwall_lightcone_probe_all_circuit) —
    ignored under topology="bowtie".

    Returns
    -------
    (circuit, width, n_realizations) — n_realizations = len(realizations);
    pass these to extract_parallel_lightcone_probe_all_means to read
    results back.
    """
    if topology == "brickwall":
        def layer_pairs(l, _L=L):
            if anchor_parity_at_perturbation is None:
                parity = l % 2
            else:
                target = 0 if anchor_parity_at_perturbation == "even" else 1
                parity = (_L - 1 - l + target) % 2
            if parity == 0:
                return [(2 * w, 2 * w + 1) for w in range(W // 2)]
            return [(2 * w + 1, 2 * w + 2) for w in range(W // 2 - 1)]
    elif topology == "bowtie":
        def layer_pairs(l):
            lo, hi = l, W - 1 - l
            if lo >= hi:
                return []
            return [(a, a + 1) for a in range(lo, hi, 2) if a + 1 <= hi]
    else:
        raise ValueError(f"Unknown topology {topology!r}. Choose 'brickwall' or 'bowtie'.")

    if pert_op != "measure" and pert_op not in _PERT_UNITARY_OPS:
        raise ValueError(
            f"Unsupported pert_op {pert_op!r}. Use 'measure' or one of {_PERT_UNITARY_OPS}."
        )
    if W % 2 != 0:
        raise ValueError("Circuit only defined for even # of qubits.")
    if pert_site is None:
        pert_site = W // 2

    n_realizations = len(realizations)
    width = max(1, n_qubits_budget // W)
    depth = -(-n_realizations // width)  # ceil(n_realizations / width)

    if max_depth is not None and depth > max_depth:
        raise ValueError(
            f"build_parallel_lightcone_probe_all_circuit: depth={depth} exceeds "
            f"max_depth={max_depth} for {n_realizations} realizations tiled at "
            f"width={width} (n_qubits_budget={n_qubits_budget} // W={W}). Fix by either "
            f"raising n_qubits_budget (more width, fewer rows needed), splitting "
            f"`realizations` across multiple build_parallel_lightcone_probe_all_circuit "
            f"calls, or raising max_depth explicitly if you've checked it's still within "
            f"your device's coherence budget."
        )

    total_qubits = width * W
    total_bits = width + n_realizations * W  # width scratch bits + W readout bits per realization

    if max_classical_bits is not None and total_bits > max_classical_bits:
        raise ValueError(
            f"build_parallel_lightcone_probe_all_circuit: total_bits={total_bits} "
            f"(width={width} scratch + {n_realizations} realizations x {W} readout bits "
            f"each) exceeds max_classical_bits={max_classical_bits} for {n_realizations} "
            f"realizations at width={width}. Fix by splitting `realizations` across "
            f"multiple build_parallel_lightcone_probe_all_circuit calls (fewer realizations "
            f"per circuit — note each one now costs W bits, not 1), or raising "
            f"max_classical_bits explicitly if you've confirmed the real device/backend "
            f"supports it."
        )

    total_su4 = _n_su4_per_realization(L, W) * n_realizations
    if max_su4_gates is not None and total_su4 > max_su4_gates:
        raise ValueError(
            f"build_parallel_lightcone_probe_all_circuit: total_su4={total_su4} "
            f"({_n_su4_per_realization(L, W)} per realization x {n_realizations} realizations) "
            f"exceeds max_su4_gates={max_su4_gates}. Fix by splitting `realizations` across "
            f"multiple build_parallel_lightcone_probe_all_circuit calls (fewer realizations "
            f"per circuit at this L), or raising max_su4_gates if you've confirmed larger "
            f"circuits compile within your backend's budget."
        )

    qc = Circuit(total_qubits, total_bits)

    idx = 0
    for d in range(depth):
        row_width = min(width, n_realizations - idx)
        for w in range(row_width):
            r = realizations[idx]
            _build_lightcone_probe_all_circuit_impl(
                layer_pairs, L, W, unperturbed, pert_site, pert_op,
                add_barrier, do_reset,
                r.get("seed"), r.get("init_seed"), r.get("meas_seed"), r.get("meas_seed_ud"),
                p, init_ops=init_ops, L_back=L_back,
                qc=qc, qubit_offset=w * W, scratch_bit=w, probe_bit_base=width + idx * W,
            )
            idx += 1
        if d < depth - 1:
            # Free every qubit for the next row's state prep. Skipped after
            # the last row — nothing follows it in the circuit.
            for q in range(total_qubits):
                qc.Reset(q)
            if add_barrier:
                qc.add_barrier(qubits=list(range(total_qubits)),
                                bits=list(range(total_bits)))

    return qc, width, n_realizations


def build_parallel_lightcone_probe_all_circuits_batched(
    L: int,
    W: int,
    unperturbed: bool,
    n_qubits_budget: int,
    realizations: list[dict],
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_per_chunk: int | None = None,
    pert_site: int | None = None,
    pert_op: str = "measure",
    add_barrier: bool = True,
    do_reset: bool = False,
    p: float = 0.0,
    init_ops: list[str] | None = None,
    L_back: int | None = None,
    topology: str = "brickwall",
    anchor_parity_at_perturbation: str | None = None,
) -> list[tuple[Circuit, int, int]]:
    """
    Batched analogue of build_parallel_lightcone_probe_all_circuit —
    mirrors build_parallel_otoc_circuits_batched exactly (splits
    `realizations` into as many circuits as needed to respect max_depth,
    max_classical_bits, AND max_su4_per_chunk, whichever is tightest; see
    that function's docstring for the full three-way chunking explanation).

    The only difference from build_parallel_otoc_circuits_batched's own
    chunk-size math: each realization here costs W classical bits, not 1,
    so bits_chunk_size = (max_classical_bits - width) // W instead of
    (max_classical_bits - width) — see build_parallel_lightcone_probe_all_
    circuit's docstring for why.

    Returns
    -------
    list of (circuit, width, n_realizations_in_this_chunk) tuples, in the
    same order as `realizations`. Submit every circuit in the list
    together, reduce each with extract_parallel_lightcone_probe_all_means
    using ITS OWN width/n_realizations, then np.concatenate the per-chunk
    results (axis=0) to get back one (n_realizations, W) array covering
    all of `realizations`.
    """
    if max_depth is not None and max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    width = max(1, n_qubits_budget // W)

    depth_chunk_size = len(realizations) if max_depth is None else width * max_depth
    if max_classical_bits is not None:
        bits_chunk_size = (max_classical_bits - width) // W
        if bits_chunk_size < 1:
            raise ValueError(
                f"max_classical_bits={max_classical_bits} leaves no room for even one "
                f"realization's {W} readout bits once width={width} scratch bits are "
                f"accounted for — lower n_qubits_budget (reduces width) or raise "
                f"max_classical_bits."
            )
    else:
        bits_chunk_size = len(realizations)
    if max_su4_per_chunk is not None:
        su4_per_realization = max(1, _n_su4_per_realization(L, W))
        gates_chunk_size = max(1, max_su4_per_chunk // su4_per_realization)
    else:
        gates_chunk_size = len(realizations)
    chunk_size = min(depth_chunk_size, bits_chunk_size, gates_chunk_size)

    circuits = []
    for start in range(0, len(realizations), chunk_size):
        chunk = realizations[start : start + chunk_size]
        qc, w, n_real = build_parallel_lightcone_probe_all_circuit(
            L=L, W=W, unperturbed=unperturbed, n_qubits_budget=n_qubits_budget,
            realizations=chunk, pert_site=pert_site, pert_op=pert_op,
            add_barrier=add_barrier, do_reset=do_reset, p=p, init_ops=init_ops,
            L_back=L_back, topology=topology,
            anchor_parity_at_perturbation=anchor_parity_at_perturbation,
            max_depth=max_depth, max_classical_bits=max_classical_bits,
            max_su4_gates=max_su4_per_chunk,
        )
        circuits.append((qc, w, n_real))
    return circuits


def extract_parallel_lightcone_probe_all_means(
    shots: np.ndarray, width: int, n_realizations: int, W: int
) -> np.ndarray:
    """
    Reduce the raw get_shots() array from a
    build_parallel_lightcone_probe_all_circuit run to one ±1-encoded
    per-SITE mean per realization, averaged over native hardware shots —
    the light-cone analogue of extract_parallel_probe_means, generalised
    from one probe bit per realization to W (one per qubit).

    shots : (n_native_shots, width + n_realizations*W) array from
            Backend.get_result().get_shots() — columns 0..width-1 are
            scratch bits, columns width..width+n_realizations*W-1 are the
            per-realization W-bit readout blocks in row-major fill order
            (realization idx's bits are width+idx*W .. width+idx*W+W-1).
    Returns a (n_realizations, W) array — row idx, column q is the
    ±1-encoded mean for qubit q of realization idx. Feed
    d_r_array = out_u_means - out_p_means (shape (n_realizations, W)) into
    the same reduction sweep_lightcone_probe_all uses per-site.
    """
    site_bits = shots[:, width : width + n_realizations * W].astype(int)
    site_bits = site_bits.reshape(shots.shape[0], n_realizations, W)
    return (1 - 2 * site_bits).mean(axis=0)


def _n_su4_per_realization(L: int, W: int) -> int:
    """
    Number of Haar-SU(4)/TK2 gate applications in one L-layer forward +
    backward brickwall OTOC circuit instance — same formula
    estimate_hqc_cost (run_disorder_avg.py) uses for its own gate-count
    estimate, kept in sync deliberately so both give consistent numbers.
    """
    L_even = (L + 1) // 2
    L_odd = L // 2
    n_su4_half = L_even * (W // 2) + L_odd * (W // 2 - 1)
    return 2 * n_su4_half  # forward + backward


def build_parallel_otoc_circuit(
    L: int,
    W: int,
    unperturbed: bool,
    n_qubits_budget: int,
    realizations: list[dict],
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    add_barrier: bool = True,
    do_reset: bool = False,
    p: float = 0.0,
    init_ops: list[str] | None = None,
    topology: str = "brickwall",
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_gates: int | None = None,
) -> tuple[Circuit, int, int]:
    """
    Tile MANY DIFFERENT disorder realizations of the W-qubit OTOC circuit
    (distinct gate-angle / init-state / measurement-site seeds — genuinely
    independent random circuits, not repeated shots of one fixed circuit)
    onto a single larger circuit, so one hardware shot returns one probe
    outcome PER REALIZATION instead of needing a separate circuit submission
    for each. This is the tiled analogue of sweep_single_shot_disorder
    (each realization is one independent 1-shot circuit) — it amortises the
    flat per-circuit submission overhead (e.g. Quantinuum's per-job HQC
    cost) across len(realizations) independent disorder samples.

    Two independent axes of replication:

    * Width — floor(n_qubits_budget / W) disjoint W-qubit blocks placed side
      by side on distinct qubits, all built within the same layers of gates
      (spatial parallelism; free in circuit depth).
    * Depth — ceil(len(realizations) / width) sequential rows: after a
      block's probe measurement, every qubit in that block is Reset and the
      next row of realizations is appended reusing the same physical qubits
      (temporal parallelism; costs extra circuit depth, needed once width
      alone can't fit all realizations on the available qubits).

    Each entry of `realizations` is a dict of per-instance seeds, e.g.
    {"seed": circuit_seed, "init_seed": ..., "meas_seed": ..., "meas_seed_ud": ...}
    — same seed semantics as build_brickwall_otoc_circuit's own arguments,
    just one full set per instance instead of one for the whole circuit.
    One realization is consumed per (width, depth) slot in row-major order;
    if len(realizations) isn't a multiple of width, the final row is
    simply shorter (unused trailing slots in that row are not built at all).

    Classical bit layout: each of the `width` blocks gets its OWN scratch
    bit (index 0..width-1), reused by every depth-row's mid-circuit
    stochastic-measurement outcomes for that block — safe, because those
    outcomes are never read back, only the collapse they cause matters, and
    a block's depth-rows are already forced strictly sequential by the
    qubit-reuse Reset between them, so sharing one bit there costs nothing.
    Keeping scratch bits separate PER BLOCK (rather than one global bit)
    avoids a classical write-after-write dependency between otherwise-
    independent width-blocks, which could give a compiler/scheduler an
    excuse to serialise physically-parallel blocks on real hardware. Each
    realization additionally gets its own dedicated probe bit (indices
    width .. width + len(realizations) - 1) so every final readout
    survives independently in the classical register.

    Parameters mirror build_brickwall_otoc_circuit / build_bowtie_otoc_circuit
    (seed/init_seed/meas_seed/meas_seed_ud now come from `realizations`
    instead of single scalars — pert_site/probe_site/p/init_ops/topology
    are shared across all instances, since only the random draws should
    differ between realizations), plus:
    n_qubits_budget : total physical qubits available to tile into.
    realizations    : list of per-instance seed dicts (see above); its
                      length is the number of independent samples produced.
    topology        : "brickwall" or "bowtie" — which layer_pairs layout to
                       replicate in every block.
    max_depth       : hard cap on the number of sequential depth-rows (None
                       = uncapped). Depth-rows are temporally sequential —
                       each one adds real circuit duration on top of the
                       previous rows' gates, even though the Reset between
                       them refreshes the QUANTUM STATE. On real hardware a
                       qubit's effective T1/T2 bounds how much total gate
                       time can elapse before errors dominate the signal;
                       packing too many rows into one circuit to chase a
                       smaller submission count can silently walk past that
                       budget and return noise-dominated garbage rather than
                       an error. Raise ValueError instead of building such a
                       circuit; callers should split into multiple circuits
                       (or multiple native shots of a shallower one) instead
                       of raising this cap arbitrarily.
    max_classical_bits : hard cap on total classical bits (width scratch
                       bits + one probe bit per realization). Real devices
                       impose a hard limit on the classical register size
                       per circuit (e.g. 64 on H2-2) independent of qubit
                       count or coherence time — a circuit tiling hundreds
                       of realizations can trivially blow past this even
                       when comfortably within max_depth, since bit count
                       scales with realizations directly, not with depth.
                       Raises ValueError rather than building an
                       unsubmittable circuit.
    max_su4_gates      : hard cap on total Haar-SU(4)/TK2 gates in this
                       circuit (_n_su4_per_realization(L, W) * n_realizations).
                       Real backend COMPILE services (e.g. Nexus compiling
                       for H2-2) can have their own server-side time budget
                       for turning generic two-qubit unitaries into native
                       gates via KAK decomposition — observed empirically:
                       1624 total SU4 gates compiled fine, 3248 got the
                       compile job cancelled server-side with a timeout.
                       Unlike max_depth/max_classical_bits this isn't a
                       documented hard limit, just a measured cliff — pick a
                       conservative value with margin below wherever your
                       own compiles start failing. Raises ValueError rather
                       than building a circuit that's likely uncompilable.

    Returns
    -------
    (circuit, width, n_realizations) — n_realizations = len(realizations);
    pass these to extract_parallel_probe_means to read results back.
    """
    if topology == "brickwall":
        def layer_pairs(l):
            if l % 2 == 0:
                return [(2 * w, 2 * w + 1) for w in range(W // 2)]
            return [(2 * w + 1, 2 * w + 2) for w in range(W // 2 - 1)]
    elif topology == "bowtie":
        def layer_pairs(l):
            lo, hi = l, W - 1 - l
            if lo >= hi:
                return []
            return [(a, a + 1) for a in range(lo, hi, 2) if a + 1 <= hi]
    else:
        raise ValueError(f"Unknown topology {topology!r}. Choose 'brickwall' or 'bowtie'.")

    pert_site, probe_site = _otoc_circuit_args(
        L, W, unperturbed, pert_site, pert_op, probe_site, probe_angle,
        add_barrier, do_reset, None, None, None, None, p,
    )

    n_realizations = len(realizations)
    width = max(1, n_qubits_budget // W)
    depth = -(-n_realizations // width)  # ceil(n_realizations / width)

    if max_depth is not None and depth > max_depth:
        raise ValueError(
            f"build_parallel_otoc_circuit: depth={depth} exceeds max_depth={max_depth} "
            f"for {n_realizations} realizations tiled at width={width} "
            f"(n_qubits_budget={n_qubits_budget} // W={W}). A deeper circuit means more "
            f"real elapsed gate time per qubit before its final measurement — past some "
            f"point that approaches the device's T1/T2 and the 'realizations' packed into "
            f"later rows stop being clean 1-shot samples and start being noise. Fix by "
            f"either raising n_qubits_budget (more width, fewer rows needed), splitting "
            f"`realizations` across multiple build_parallel_otoc_circuit calls, or raising "
            f"max_depth explicitly if you've checked it's still within your device's "
            f"coherence budget."
        )

    total_qubits = width * W
    total_bits = width + n_realizations  # width scratch bits + one probe bit per realization

    if max_classical_bits is not None and total_bits > max_classical_bits:
        raise ValueError(
            f"build_parallel_otoc_circuit: total_bits={total_bits} (width={width} scratch "
            f"+ {n_realizations} probe bits) exceeds max_classical_bits={max_classical_bits} "
            f"for {n_realizations} realizations at width={width}. Fix by splitting "
            f"`realizations` across multiple build_parallel_otoc_circuit calls (fewer "
            f"realizations per circuit), or raising max_classical_bits explicitly if you've "
            f"confirmed the real device/backend supports it."
        )

    total_su4 = _n_su4_per_realization(L, W) * n_realizations
    if max_su4_gates is not None and total_su4 > max_su4_gates:
        raise ValueError(
            f"build_parallel_otoc_circuit: total_su4={total_su4} "
            f"({_n_su4_per_realization(L, W)} per realization x {n_realizations} realizations) "
            f"exceeds max_su4_gates={max_su4_gates}. Larger L means more SU4/TK2 gates per "
            f"realization, so a chunk size that compiles fine at low L can become too complex "
            f"for the backend compiler at high L with the same realization count. Fix by "
            f"splitting `realizations` across multiple build_parallel_otoc_circuit calls "
            f"(fewer realizations per circuit at this L), or raising max_su4_gates if you've "
            f"confirmed larger circuits compile within your backend's budget."
        )

    qc = Circuit(total_qubits, total_bits)

    idx = 0
    for d in range(depth):
        row_width = min(width, n_realizations - idx)
        for w in range(row_width):
            r = realizations[idx]
            _build_otoc_circuit_impl(
                layer_pairs, L, W, unperturbed, pert_site, pert_op, probe_site,
                probe_angle, add_barrier, do_reset,
                r.get("seed"), r.get("init_seed"), r.get("meas_seed"), r.get("meas_seed_ud"),
                p, init_ops=init_ops,
                qc=qc, qubit_offset=w * W, scratch_bit=w, probe_bit=width + idx,
            )
            idx += 1
        if d < depth - 1:
            # Free every qubit for the next row's state prep. Skipped after
            # the last row — nothing follows it in the circuit.
            for q in range(total_qubits):
                qc.Reset(q)
            if add_barrier:
                qc.add_barrier(qubits=list(range(total_qubits)),
                                bits=list(range(total_bits)))

    return qc, width, n_realizations


def build_parallel_otoc_circuits_batched(
    L: int,
    W: int,
    unperturbed: bool,
    n_qubits_budget: int,
    realizations: list[dict],
    max_depth: int | None = None,
    max_classical_bits: int | None = None,
    max_su4_per_chunk: int | None = None,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    add_barrier: bool = True,
    do_reset: bool = False,
    p: float = 0.0,
    init_ops: list[str] | None = None,
    topology: str = "brickwall",
) -> list[tuple[Circuit, int, int]]:
    """
    Batched analogue of build_parallel_otoc_circuit: instead of raising when
    `realizations` would need more than max_depth sequential depth-rows, OR
    more than max_classical_bits total classical bits, OR more than
    max_su4_per_chunk total SU4/TK2 gates, on n_qubits_budget qubits, splits
    `realizations` into as many circuits as needed to respect ALL THREE
    limits ("what's left over" after the first circuit fills up goes into
    the next one, and so on) and builds every one of them, ready to be
    submitted together.

    Two independent, unrelated limits set how many realizations fit per
    circuit:
    * max_depth           bounds width * max_depth realizations (coherence
                           budget — see build_parallel_otoc_circuit).
    * max_classical_bits   bounds max_classical_bits - width realizations
                           (classical register size — real devices cap this
                           independent of qubit count or depth, e.g. 64 on
                           H2-2; width scratch bits are a fixed overhead
                           that eats into that budget before any probe bits
                           do).
    * max_su4_per_chunk    bounds max_su4_per_chunk // _n_su4_per_realization(L, W)
                           realizations (compile-server time budget — grows
                           with L, unlike the other two limits, so the same
                           realization count that compiles fine at low L can
                           get its compile job cancelled server-side at high
                           L; see build_parallel_otoc_circuit).
    Whichever is tightest determines the actual chunk size; every returned
    circuit individually satisfies all three by construction.

    width = n_qubits_budget // W realizations fit per depth-row, chunked in
    the same row-major order build_parallel_otoc_circuit itself fills a
    single circuit.

    max_depth=None, max_classical_bits=None, and max_su4_per_chunk=None
    reproduces build_parallel_otoc_circuit exactly (one chunk holding every
    realization, uncapped) — this function is a strict generalisation, not
    a different tool: callers can always use this one.

    Parameters mirror build_parallel_otoc_circuit's exactly.

    Returns
    -------
    list of (circuit, width, n_realizations_in_this_chunk) tuples, in the
    same order as `realizations`. Submit every circuit in the list together
    (e.g. one _run_circuits call, letting Nexus batch them into one queue
    slot) for the throughput benefit; reduce each with
    extract_parallel_probe_means using ITS OWN width/n_realizations, then
    np.concatenate the per-chunk results to get back one array covering all
    of `realizations`.
    """
    if max_depth is not None and max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    width = max(1, n_qubits_budget // W)

    depth_chunk_size = len(realizations) if max_depth is None else width * max_depth
    if max_classical_bits is not None:
        bits_chunk_size = max_classical_bits - width
        if bits_chunk_size < 1:
            raise ValueError(
                f"max_classical_bits={max_classical_bits} leaves no room for probe bits "
                f"once width={width} scratch bits are accounted for — lower n_qubits_budget "
                f"(reduces width) or raise max_classical_bits."
            )
    else:
        bits_chunk_size = len(realizations)
    if max_su4_per_chunk is not None:
        su4_per_realization = max(1, _n_su4_per_realization(L, W))
        gates_chunk_size = max(1, max_su4_per_chunk // su4_per_realization)
    else:
        gates_chunk_size = len(realizations)
    chunk_size = min(depth_chunk_size, bits_chunk_size, gates_chunk_size)

    circuits = []
    for start in range(0, len(realizations), chunk_size):
        chunk = realizations[start : start + chunk_size]
        qc, w, n_real = build_parallel_otoc_circuit(
            L=L, W=W, unperturbed=unperturbed, n_qubits_budget=n_qubits_budget,
            realizations=chunk, pert_site=pert_site, pert_op=pert_op,
            probe_site=probe_site, probe_angle=probe_angle, add_barrier=add_barrier,
            do_reset=do_reset, p=p, init_ops=init_ops, topology=topology,
            max_depth=max_depth, max_classical_bits=max_classical_bits,
            max_su4_gates=max_su4_per_chunk,
        )
        circuits.append((qc, w, n_real))
    return circuits


def extract_parallel_probe_means(shots: np.ndarray, width: int, n_realizations: int) -> np.ndarray:
    """
    Reduce the raw get_shots() array from a build_parallel_otoc_circuit run
    to one ±1-encoded probe MEAN per realization, averaged over native
    hardware shots if more than one was requested — the direct analogue of
    `out.mean()` in the standard single-circuit pipeline, but vectorised
    over every tiled realization at once. Feed the result straight into
    `d_r_array = out_u_means - out_p_means`, which is exactly the
    per-realization array compute_C_t expects for one (N, p, L) point.

    shots : (n_native_shots, width + n_realizations) array from
            Backend.get_result().get_shots().
    Returns a (n_realizations,) array.
    """
    probe_bits = shots[:, width : width + n_realizations]
    return (1 - 2 * probe_bits.astype(int)).mean(axis=0)


def _wait_for_job_resilient(
    job_ref,
    max_retries: int = 200,
    initial_delay: float = 15.0,
    max_delay: float = 300.0,
):
    """
    qnx.jobs.wait_for wrapper that retries on transient connection/auth
    failures instead of crashing the whole process. The underlying Nexus
    JOB keeps running/queueing server-side regardless of whether this
    local call can currently reach it — a websocket drop, an auth-token
    hiccup, or any other connectivity blip during a long wait (e.g. a
    device under maintenance for hours) is a failure of THIS STATUS CHECK,
    not of the job itself, so it's worth retrying rather than propagating
    as a fatal error that kills an otherwise-fine, still-queued job.

    Does NOT retry qnx_exc.JobError — that means Nexus told us the JOB
    ITSELF reached an error/cancelled terminal state (e.g. the compile
    timeout from genuine circuit complexity we hit earlier), and retrying
    the same wait_for call can't fix that; it's re-raised immediately.

    Exponential backoff from initial_delay up to max_delay, capped at
    max_retries attempts — defaults comfortably outlast a multi-hour
    device maintenance window when running unattended (e.g. via nohup).
    """
    import time

    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return qnx.jobs.wait_for(job_ref)
        except qnx_exc.JobError:
            raise  # the job itself errored — retrying this call won't help
        except Exception as exc:
            print(
                f"  [wait retry {attempt}/{max_retries}] {type(exc).__name__}: {exc} "
                f"— re-authenticating and retrying in {delay:.0f}s "
                f"(the job keeps queueing/running server-side regardless)"
            )
            try:
                qnx.login_with_credentials()
            except Exception as login_exc:
                print(f"    (re-auth attempt itself failed: {login_exc})")
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)
    raise TimeoutError(f"_wait_for_job_resilient: gave up after {max_retries} retries")


def _run_circuits(
    circuits: list[Circuit],
    backend,
    n_shots: int,
    tags: list[str] | None = None,
    optimisation_level: int = 0,
    nexus_project: str = "schmetterling-effect-test",
    use_batch: bool = True,
    job_info_out: dict | None = None,
    wait_for_execute: bool = True,
) -> list[np.ndarray] | None:
    """
    Compile and execute circuits on `backend`.

    For local AerBackend: compiles and runs directly via the pytket backend API.

    For QuantinuumBackend: goes through Nexus — uploads circuits, compiles as
    a single compile job, then executes via qnx.start_execute_job.
    When use_batch=True, attempt_batching=True is set so all circuits share one
    hardware queue slot.  If your organisation does not have batching enabled,
    set use_batch=False and the job is submitted without the batch flag.

    Parameters:
    circuits (list[Circuit]): Circuits to run; output order matches input order.
    backend: AerBackend for local simulation or QuantinuumBackend for hardware.
    n_shots (int): Shots per circuit.
    tags (list[str]): Optional names for each circuit. Used as Nexus circuit
    names and local log labels.
    optimisation_level (int): pytket compilation level 0 / 1 / 2. Default 0.
    nexus_project (str): Nexus project name (created if absent). Only used for
    QuantinuumBackend runs.
    use_batch (bool): Enable Nexus batching (attempt_batching=True). Requires
    the feature to be enabled for your organisation. Set False if you receive
    a 403 "Batching needs to be enabled" error. Default True.
    job_info_out (dict): if given, populated (in place) with traceability
    info for this call — on the QuantinuumBackend path: compile_job_id,
    execute_job_id (both str(UUID)), circuit_ids (one per input circuit, in
    order), and tags. On the local AerBackend path: just {"backend": "local",
    "tags": tags} (no real job to reference). Lets a caller record exactly
    which Nexus job/circuits produced a given result, recoverable from Nexus
    directly later even without keeping the downloaded shots around.
    wait_for_execute (bool): QuantinuumBackend path ONLY (ignored for local
    AerBackend, which always runs synchronously). When True (default): waits
    for the execute job to finish and downloads/returns results, as before.
    When False: still compiles and WAITS for the compile job (needed to get
    compiled circuit refs to submit for execution), submits the execute job,
    then returns immediately WITHOUT waiting for it to finish — for
    submit-many-jobs-and-check-back-later workflows where per-job queue time
    on the real device (observed: hours to multiple DAYS) makes waiting
    sequentially for each one impractical. job_info_out (REQUIRED when this
    is False — raises ValueError otherwise, since it's the only way to find
    the execute job again) still gets populated with the execute_job_id/tags/
    etc. needed to fetch results later via get_results_by_job_id. Returns
    None instead of shot arrays in this case — there IS nothing to return
    yet, the caller must harvest results separately once the job completes.

    Returns:
    list of np.ndarray of shape (n_shots, n_bits), one per input circuit, in
    the same order as the input list — or None if wait_for_execute=False (see
    above; only possible on the QuantinuumBackend path).
    """
    import uuid

    if tags is not None:
        if len(tags) != len(circuits):
            raise ValueError(f"len(tags)={len(tags)} must equal len(circuits)={len(circuits)}")
        for qc, name in zip(circuits, tags):
            qc.name = name

    if not wait_for_execute and job_info_out is None:
        raise ValueError(
            "wait_for_execute=False requires job_info_out (a dict to populate with "
            "the execute_job_id) — otherwise there would be no way to find this job "
            "again to fetch its results later."
        )

    _tags = tags or [f"circuit_{i}" for i in range(len(circuits))]
    backend_label = getattr(backend, "_device_name", type(backend).__name__)
    print(f"[{backend_label}] {len(circuits)} circuit(s)  n_shots={n_shots}  "
          f"opt={optimisation_level}  batch={use_batch}")

    # ── Quantinuum path: compile + execute via Nexus ───────────────────────────
    if isinstance(backend, QuantinuumBackend):
        device_name = backend._device_name

        print(f"[{backend_label}] logging into Nexus …")
        qnx.login_with_credentials()
        project_ref = qnx.projects.get_or_create(name=nexus_project)

        print(f"[{backend_label}] uploading {len(circuits)} circuit(s) …")
        circuit_refs = [
            qnx.circuits.upload(circuit=qc, name=tag, project=project_ref)
            for qc, tag in zip(circuits, _tags)
        ]

        print(f"[{backend_label}] compiling (opt={optimisation_level}) …")
        compile_job = qnx.start_compile_job(
            programs=circuit_refs,
            name=f"compile_{_tags[0]}",
            optimisation_level=optimisation_level,
            backend_config=qnx.QuantinuumConfig(device_name=device_name),
            project=project_ref,
        )
        _wait_for_job_resilient(compile_job)
        compiled_refs = [item.get_output() for item in qnx.jobs.results(compile_job)]

        if use_batch:
            backend_config = qnx.QuantinuumConfig(
                device_name=device_name,
                attempt_batching=True,
                batch_id=uuid.uuid4(),
            )
            print(f"[{backend_label}] submitting batched execute job …")
        else:
            backend_config = qnx.QuantinuumConfig(device_name=device_name)
            print(f"[{backend_label}] submitting execute job (no batching) …")

        try:
            execute_job = qnx.start_execute_job(
                programs=compiled_refs,
                n_shots=[n_shots] * len(compiled_refs),
                backend_config=backend_config,
                name=f"execute_{_tags[0]}",
                project=project_ref,
            )
        except Exception as exc:
            msg = str(exc)
            if use_batch and ("403" in msg or "Batching needs to be enabled" in msg):
                raise RuntimeError(
                    "Batching is not enabled for your organisation (HTTP 403).\n"
                    "Set USE_BATCH = False in the config and rerun."
                ) from exc
            raise

        if job_info_out is not None:
            job_info_out["compile_job_id"] = str(compile_job.id)
            job_info_out["execute_job_id"] = str(execute_job.id)
            job_info_out["circuit_ids"] = [str(ref.id) for ref in circuit_refs]
            job_info_out["tags"] = list(_tags)
            job_info_out["nexus_project"] = nexus_project
            job_info_out["device_name"] = device_name

        if not wait_for_execute:
            print(f"[{backend_label}] execute job submitted, NOT waiting "
                  f"(execute_job_id={execute_job.id}) — fetch results later via "
                  f"get_results_by_job_id.")
            return None

        print(f"[{backend_label}] waiting for results …")
        _wait_for_job_resilient(execute_job)
        results = [
            item.download_result().get_shots()
            for item in qnx.jobs.results(execute_job)
        ]

    # ── Local path: direct pytket backend API (AerBackend etc.) ──────────────
    else:
        compiled = [
            backend.get_compiled_circuit(qc, optimisation_level=optimisation_level)
            for qc in circuits
        ]
        handles = backend.process_circuits(compiled, n_shots=[n_shots] * len(compiled))
        print(f"[{backend_label}] waiting for results …")
        results = [backend.get_result(h).get_shots() for h in handles]

        if job_info_out is not None:
            job_info_out["backend"] = "local"
            job_info_out["tags"] = list(_tags)

    print(f"[{backend_label}] done — {len(results)} result(s) received")
    return results


def get_results_by_job_id(
    job_id: str,
    tags: list[str] | None = None,
    wait: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Retrieve raw shot results directly from Nexus given an EXECUTE job ID —
    no local process or checkpoint file needed, no re-submission. Recovers
    the execute_job_id values _run_circuits records in job_info_out (also
    what sweep_single_shot_disorder_tiled's raw_results_path stores per
    point, or what's visible in the Nexus UI/qnx.jobs.get for any past job).

    IMPORTANT — per-item identity: ExecutionResultRef.annotations is set to
    execute_job.annotations for EVERY item (see qnexus's own
    client/jobs/_execute.py) — i.e. every item in a multi-circuit job
    shares the literal same annotations object as the JOB itself, not its
    own uploaded circuit's name. Using item.annotations.name to tell items
    apart silently collapses every result onto one identical name.

    job_item_integer_id is NOT a per-job 0-indexed position either (an
    earlier version of this function assumed it was — verified wrong
    against a real job, where it came back as e.g. 4673313/4673314): per
    qnexus's own _execute.py it's `item.get("item_id")` straight off the
    raw API response, which is a global/database-wide sequence, not a
    local index. It's kept below only as a diagnostic in error messages,
    never for indexing.

    The reliable per-item identifier is POSITION in the list returned by
    qnx.jobs.results(job_ref): _execute._results builds that list with a
    plain top-to-bottom `for item in resp_data[...]["items"]: ...append(...)`
    over the job's own definition — no sorting, no reindexing — so list
    order mirrors the job definition's item order, which is the original
    submission order. That position is matched here against `tags` in the
    exact order they were originally passed to _run_circuits' own `tags`
    argument (that's also exactly what job_info_out["tags"] records, or
    what a caller reconstructs itself when resubmission order is
    deterministic, as with this codebase's chunk builders). allow_incomplete
    =True is passed through so no item is silently dropped from that list
    (which would desync every position after it), even ones stuck in an
    unexpected status.

    Use case: if a local process dies AFTER a job was submitted to Nexus
    but BEFORE it could download+checkpoint the results itself (compile/
    execute already ran and were BILLED on the real device either way),
    this recovers the results that already exist server-side instead of
    re-submitting (and re-paying for) the same circuits.

    A multi-circuit execute job can PARTIALLY fail even after reaching an
    overall terminal status: wait_for() targets the JOB's status, but
    individual circuits within it have their own per-item status, and one
    can come back CANCELLED (e.g. a device maintenance window interrupting
    it mid-batch) while the rest genuinely completed. Downloading is
    therefore done per item with its own try/except — one bad circuit
    doesn't take down every already-good result alongside it, which a
    naive loop-and-crash would do.

    Parameters
    ----------
    job_id : str(UUID) of the EXECUTE job (not the compile job — compile
        jobs produce compiled circuits, not shot results).
    tags   : the circuit tags in EXACT original submission order (e.g.
        [f"{tag_base}_pert_chunk{i}" for i in range(n_pert_chunks)] +
        [f"{tag_base}_unpert_chunk{i}" for i in range(n_unpert_chunks)]).
        Required to recover meaningful per-circuit names — each result
        item's POSITION in qnx.jobs.results(job_ref) (not any ID field on
        the item itself) is used to index into this list. If omitted, or
        the job's item count doesn't match len(tags) (a loud WARNING is
        printed in that case — positions can't be trusted to line up),
        items fall back to being keyed by their own result id, which won't
        match any tag a caller might be looking for.
    wait   : if True (default), block until the job reaches a terminal
        state before fetching results (qnx.jobs.wait_for); set False only
        if you already know the job finished and want to skip that check.

    Returns
    -------
    (results, failures)
    results  : dict mapping each circuit's name/tag (exactly the string
        passed as `tags` to _run_circuits at submission time, e.g.
        "N8_p0.00_L008_s1962_tiled_pert_chunk3") to its raw get_shots()
        array, for every item that downloaded successfully. Keyed by name
        rather than list position — Nexus's result ordering isn't
        something to rely on, and the tag is what ties a result back to a
        specific (N, p, L, seed, chunk, pert/unpert) instance.
    failures : dict mapping tag -> error message, for every item that
        failed to download. ALWAYS CHECK THIS — a non-empty `results` does
        not mean everything came back; a chunk whose pert circuit
        succeeded but whose unpert circuit is in `failures` (or vice
        versa) can't produce a valid d_r on its own and should be dropped
        from both sides, not just silently under-counted.

    To turn `results` into C(t) and its variance: slice each array's
    probe-bit columns (shots[:, width:width+n_realizations] — see
    build_parallel_otoc_circuit's bit layout, or extract_parallel_probe_means
    which does exactly this slice), reduce to a ±1-encoded mean per
    realization, concatenate across a point's SURVIVING chunks, difference
    perturbed vs unperturbed, then feed to compute_C_t — the same pipeline
    run_disorder_avg.rebuild_stats_from_raw_results implements against a
    raw_results_path file; reuse that pipeline rather than re-deriving it.
    """
    qnx.login_with_credentials()
    job_ref = qnx.jobs.get(id=job_id)

    if wait:
        _wait_for_job_resilient(job_ref)

    items = list(qnx.jobs.results(job_ref, allow_incomplete=True))

    if tags is not None and len(items) != len(tags):
        print(
            f"  [get_results_by_job_id] WARNING: job has {len(items)} item(s) but "
            f"{len(tags)} tag(s) were provided — positional alignment is almost "
            f"certainly wrong. Verify the reconstructed circuit/chunk count matches "
            f"the original submission before trusting anything returned below."
        )

    results: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for idx, item in enumerate(items):
        if tags is not None and idx < len(tags):
            name = tags[idx]
        else:
            name = str(item.id)
        try:
            results[name] = item.download_result().get_shots()
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(
                f"  [get_results_by_job_id] FAILED to download {name!r} "
                f"(position {idx}, job_item_integer_id="
                f"{getattr(item, 'job_item_integer_id', None)}): {failures[name]}"
            )

    if failures:
        print(
            f"  [get_results_by_job_id] {len(failures)}/{len(results) + len(failures)} "
            f"item(s) failed to download; {len(results)} succeeded."
        )
    return results, failures


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
    base_seeds: list[int] | None = None,
    meas_seed: int | None = None,
    record_every: int = 5,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    backend=None,
    device_name: str | None = None,
    optimisation_level: int = 0,
    use_batch: bool = True,
    circuit_builder=None,
    init_ops: list[str] | None = None,
    nexus_project: str = "schmetterling-effect-test",
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
    base_seed (int): Root seed used when base_seeds is None.
    base_seeds (list[int]): List of root seeds to average over. Each seed
    contributes n_circuits × n_init_states independent realizations; results
    are concatenated before computing statistics. Overrides base_seed when set.
    meas_seed (int): Root seed for measurement site selection. Defaults to
    base_seed + 1 (or _seed + 1 for each seed in base_seeds).
    record_every (int): Only record/execute at L = record_every, 2*record_every, ...
    Ignored when L_values is provided.
    L_values (list[int]): Explicit list of layer counts to evaluate (e.g. [1,2,5,10,25]).
    When given, overrides L_max and record_every.
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

    if L_values is None:
        L_values = list(range(record_every, L_max + 1, record_every))

    _build_circuit = circuit_builder if circuit_builder is not None else build_brickwall_otoc_circuit
    _seeds = base_seeds if base_seeds is not None else [base_seed]
    print(f"Seed average over {len(_seeds)} seed(s): {_seeds}")

    all_stats: dict[int, dict] = {}

    for W in N_values:
        _pert_site = pert_site if pert_site is not None else W // 2
        _probe_site = probe_site if probe_site is not None else 0
        raw: dict[float, dict[int, list]] = {
            p: {L: [] for L in L_values} for p in p_values
        }

        for _seed in _seeds:
            _meas_base = meas_seed if meas_seed is not None else _seed + 1

            for p in p_values:
                for c_idx in range(n_circuits):
                    circuit_seed = _seed * 10_000 + c_idx
                    for i_idx in range(n_init_states):
                        init_seed_val    = _seed * 10_000_000 + c_idx * 1_000 + i_idx
                        meas_seed_u_val  = _meas_base * 10_000_000 + c_idx * 1_000 + i_idx
                        meas_seed_ud_val = (_meas_base + 1) * 10_000_000 + c_idx * 1_000 + i_idx

                        for L in L_values:
                            shared_kwargs = dict(
                                L=L, W=W,
                                pert_site=_pert_site, pert_op=pert_op,
                                probe_site=_probe_site, probe_angle=probe_angle,
                                p=p,
                                seed=circuit_seed, init_seed=init_seed_val,
                                meas_seed=meas_seed_u_val, meas_seed_ud=meas_seed_ud_val,
                                add_barrier=_add_barrier,
                                init_ops=init_ops,
                            )
                            qc_pert   = _build_circuit(**shared_kwargs, unperturbed=False)
                            qc_unpert = _build_circuit(**shared_kwargs, unperturbed=True)
                            probe_bit = qc_pert.n_bits - 1

                            tag_base = f"N{W}_p{p:.2f}_L{L:03d}_s{_seed}_c{c_idx}_i{i_idx}"
                            shots_p, shots_u = _run_circuits(
                                [qc_pert, qc_unpert], backend, n_shots,
                                tags=[f"{tag_base}_pert", f"{tag_base}_unpert"],
                                nexus_project=nexus_project,
                                optimisation_level=optimisation_level,
                                use_batch=use_batch,
                            )

                            out_p = 1 - 2 * shots_p[:, probe_bit].astype(int)
                            out_u = 1 - 2 * shots_u[:, probe_bit].astype(int)
                            raw[p][L].append(out_u.mean() - out_p.mean())
                            print(
                                f"N={W:2d}  p={p:.2f}  L={L:3d}  "
                                f"seed={_seed}  c={c_idx}  i={i_idx}  "
                                f"C_t={raw[p][L][-1]:.4f}"
                            )

        for p in p_values:
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
    base_seeds: list[int] | None = None,
    record_every: int = 5,
    L_values: list[int] | None = None,
    pert_site: int | None = None,
    pert_op: str = "measure",
    probe_site: int | None = None,
    probe_angle: float = 0.5,
    backend=None,
    device_name: str | None = None,
    optimisation_level: int = 0,
    use_batch: bool = True,
    circuit_builder=None,
    init_ops: list[str] | None = None,
    nexus_project: str = "schmetterling-effect-test",
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
    (N, p, L) grid point, per seed. Total realizations = n_realizations × len(base_seeds).
    base_seed (int): Root seed used when base_seeds is None.
    base_seeds (list[int]): List of root seeds to average over. Each seed
    contributes n_realizations independent (out_u − out_p) samples; arrays are
    concatenated before computing statistics. Overrides base_seed when set.
    Per-realization seeds for each _seed in base_seeds:
    gate angles   → (_seed + 0) * 10^9 + L * n_realizations + r
    initial state → (_seed + 1) * 10^9 + L * n_realizations + r
    U meas sites  → (_seed + 2) * 10^9 + L * n_realizations + r
    U† meas sites → (_seed + 3) * 10^9 + L * n_realizations + r
    record_every (int): Only record/execute at L = record_every, 2*record_every, ...
    Ignored when L_values is provided.
    L_values (list[int]): Explicit list of layer counts to evaluate (e.g. [1,2,5,10,25]).
    When given, overrides L_max and record_every.
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

    if L_values is None:
        L_values = list(range(record_every, L_max + 1, record_every))

    _build_circuit = circuit_builder if circuit_builder is not None else build_brickwall_otoc_circuit
    _seeds = base_seeds if base_seeds is not None else [base_seed]
    print(f"Seed average over {len(_seeds)} seed(s): {_seeds}")

    all_stats: dict[int, dict] = {}

    for W in N_values:
        _pert_site = pert_site if pert_site is not None else W // 2
        _probe_site = probe_site if probe_site is not None else 0
        raw: dict[float, dict[int, list]] = {p: {L: [] for L in L_values} for p in p_values}

        for _seed in _seeds:
            for p in p_values:
                for L in L_values:
                    circuits_pert, circuits_unpert = [], []
                    for r in range(n_realizations):
                        # Each (seed, r, L) triple draws a completely fresh
                        # disorder instance — no overlap across seeds or depths.
                        def _s(axis, s=_seed):
                            return (s + axis) * 1_000_000_000 + L * n_realizations + r
                        kwargs = dict(
                            L=L, W=W,
                            pert_site=_pert_site, pert_op=pert_op,
                            probe_site=_probe_site, probe_angle=probe_angle,
                            p=p,
                            seed=_s(0), init_seed=_s(1),
                            meas_seed=_s(2), meas_seed_ud=_s(3),
                            add_barrier=_add_barrier,
                            init_ops=init_ops,
                        )
                        circuits_pert.append(_build_circuit(**kwargs, unperturbed=False))
                        circuits_unpert.append(_build_circuit(**kwargs, unperturbed=True))

                    probe_bit = circuits_pert[0].n_bits - 1
                    tag = f"N{W}_p{p:.2f}_L{L:03d}_s{_seed}"
                    all_shots = _run_circuits(
                        circuits_pert + circuits_unpert,
                        backend, 1,
                        tags=[f"{tag}_r{r}_pert"   for r in range(n_realizations)]
                           + [f"{tag}_r{r}_unpert" for r in range(n_realizations)],
                        nexus_project=nexus_project,
                        optimisation_level=optimisation_level,
                        use_batch=use_batch,
                    )

                    out_p = np.array([1 - 2 * int(s[0, probe_bit]) for s in all_shots[:n_realizations]])
                    out_u = np.array([1 - 2 * int(s[0, probe_bit]) for s in all_shots[n_realizations:]])
                    raw[p][L].extend((out_u - out_p).tolist())
                    print(
                        f"N={W:2d}  p={p:.2f}  L={L:3d}  seed={_seed}  "
                        f"C_t={(out_u - out_p).mean():.4f}  "
                        f"n_total={len(raw[p][L])}"
                    )

        for p in p_values:
            for L in L_values:
                raw[p][L] = np.array(raw[p][L])

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
#                         qc_pert = build_otoc_circuit(
#                             **shared_kwargs, unperturbed=False
#                         )
#                         qc_unpert = build_otoc_circuit(
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

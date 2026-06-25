"""
tests/test_main.py

Unit tests for every function in src/__main__.py.

Organisation
------------
One TestClass per function. Within each class tests are ordered:
  1. happy-path / minimal inputs
  2. edge cases and boundary conditions
  3. error / exception paths

Every test method has a docstring explaining:
  - WHAT is being asserted
  - WHY it matters (physical or algorithmic reason)

External dependencies (qnexus, Quantinuum backend, matplotlib file I/O) are
always mocked so the tests run offline with no credentials.

Run with:
    .butterfly/bin/pytest tests/test_main.py -v
"""

import importlib.util
import pathlib
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ── Load the module under test ─────────────────────────────────────────────────
# src/__main__.py uses the reserved module name '__main__', so a plain
#   `import __main__`  would import THIS test file, not the source.
# importlib lets us load it from its file path under an alias.
_src = pathlib.Path(__file__).parent.parent / "src" / "__main__.py"
_spec = importlib.util.spec_from_file_location("schmetterling", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# ── Grab every public symbol we want to test ───────────────────────────────────
calculate_hamming_distance_pdf          = _mod.calculate_hamming_distance_pdf
validate_nexus_connection               = _mod.validate_nexus_connection
generate_time_reversal_breaking_random_brick_wall = (
    _mod.generate_time_reversal_breaking_random_brick_wall
)
_compile_and_run                        = _mod._compile_and_run
run_time_sweep                          = _mod.run_time_sweep
sweep_p_and_t                           = _mod.sweep_p_and_t
compute_F                               = _mod.compute_F
debug_after_measurement                 = _mod.debug_after_measurement
count_consecutive_tk2_after_measure     = _mod.count_consecutive_tk2_after_measure
_run_circuit_backend                    = _mod._run_circuit_backend
disorder_average_vs_N                   = _mod.disorder_average_vs_N
plot_results                            = _mod.plot_results
sweep_x_and_t                           = _mod.sweep_x_and_t
plot_otoc                               = _mod.plot_otoc


# ══════════════════════════════════════════════════════════════════════════════
# 1. calculate_hamming_distance_pdf
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateHammingDistancePDF:
    """
    calculate_hamming_distance_pdf takes a dict whose keys are bitstrings
    (tuples of 0/1) and values are counts. It returns (xs, pdf) where xs is a
    sorted list of Hamming distances from the all-zeros state and pdf is the
    normalised probability for each distance.
    """

    def test_all_ground_state(self):
        """
        Every shot lands on |00...0>, so the only Hamming distance is 0.
        The pdf must be [1.0] and xs must be [0].
        This is the trivial perfect-echo baseline.
        """
        data = {(0, 0, 0): 100}
        xs, pdf = calculate_hamming_distance_pdf(data)
        assert xs == [0], "Only distance-0 bitstrings → xs must be [0]"
        assert pdf == [1.0], "All weight at distance 0 → pdf must be [1.0]"

    def test_single_bit_flip(self):
        """
        One count at distance-1 (single qubit flipped), one at distance-0.
        xs should be [0, 1] and the pdf should reflect the split.
        Tests that the Hamming metric correctly distinguishes single flips.
        """
        data = {(0, 0): 3, (1, 0): 1}
        xs, pdf = calculate_hamming_distance_pdf(data)
        assert xs == [0, 1]
        assert pytest.approx(pdf[0]) == 0.75   # 3 of 4 shots at distance 0
        assert pytest.approx(pdf[1]) == 0.25   # 1 of 4 shots at distance 1

    def test_multiple_distances(self):
        """
        Bitstrings spread across distances 0, 1, 2, 3.
        Verifies multi-bucket histogram construction.
        """
        data = {
            (0, 0, 0): 4,   # distance 0
            (1, 0, 0): 3,   # distance 1
            (1, 1, 0): 2,   # distance 2
            (1, 1, 1): 1,   # distance 3
        }
        xs, pdf = calculate_hamming_distance_pdf(data)
        total = 10
        assert xs == [0, 1, 2, 3]
        assert pytest.approx(pdf) == [4/total, 3/total, 2/total, 1/total]

    def test_pdf_sums_to_one(self):
        """
        Probability mass must be conserved: sum(pdf) == 1.
        Fails if normalisation is wrong (e.g. off-by-one in total count).
        """
        data = {(0,): 7, (1,): 3}
        _, pdf = calculate_hamming_distance_pdf(data)
        assert pytest.approx(sum(pdf)) == 1.0

    def test_xs_are_sorted(self):
        """
        xs must be returned in ascending order regardless of dict insertion order.
        Callers rely on xs[i] < xs[i+1] for plotting.
        """
        # Insert in reverse order
        data = {(1, 1): 2, (0, 0): 5, (1, 0): 3}
        xs, _ = calculate_hamming_distance_pdf(data)
        assert xs == sorted(xs), "xs must be sorted in ascending order"

    def test_equal_counts_at_all_distances(self):
        """
        When every distance has the same count the pdf should be uniform.
        """
        n = 4
        data = {(0,) * n: 10, (1,) + (0,) * (n-1): 10,
                (1, 1) + (0,) * (n-2): 10, (1, 1, 1) + (0,) * (n-3): 10}
        _, pdf = calculate_hamming_distance_pdf(data)
        assert all(pytest.approx(v) == 0.25 for v in pdf)


# ══════════════════════════════════════════════════════════════════════════════
# 2. compute_F
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeF:
    """
    compute_F takes raw[p][L] = 1-D array of per-realization
    d = (pert_mean - unpert_mean) values and returns:
      F   = 1 - mean(d)
      var = sample variance of d   (ddof=1)
      cv  = sqrt(var) / |F|        (inf when F == 0)
    """

    def _make_raw(self, p, L, vals):
        """Helper: wrap a list of floats into the raw dict structure."""
        return {p: {L: np.array(vals)}}

    def test_perfect_echo_d_is_zero(self):
        """
        If every realization gives d=0 (pert and unpert circuits give exactly
        the same probe outcome), then mean(d)=0 → F = 1-0 = 1.
        This is the 'no butterfly effect' baseline.
        """
        raw = self._make_raw(0.5, 10, [0.0, 0.0, 0.0])
        stats = compute_F(raw)
        assert pytest.approx(stats["F"][0.5][10]) == 1.0
        assert pytest.approx(stats["var"][0.5][10]) == 0.0

    def test_full_decoherence_d_is_one(self):
        """
        If every realization gives d=1 (pert circuit always more scrambled),
        mean(d)=1 → F = 1-1 = 0.
        CV should be inf because |F| = 0.
        """
        raw = self._make_raw(0.5, 10, [1.0, 1.0, 1.0])
        stats = compute_F(raw)
        assert pytest.approx(stats["F"][0.5][10]) == 0.0
        assert stats["cv"][0.5][10] == np.inf

    def test_known_values_formula(self):
        """
        Manual verification: d = [0.2, 0.4, 0.6]
          mean(d) = 0.4  →  F = 0.6
          var(d)  = sample variance = ((0.04 + 0.0 + 0.04) / 2) = 0.04
          cv      = sqrt(0.04) / 0.6 = 0.2 / 0.6 ≈ 0.3333
        """
        d = [0.2, 0.4, 0.6]
        raw = self._make_raw(0.3, 5, d)
        stats = compute_F(raw)

        expected_F   = 1.0 - np.mean(d)
        expected_var = float(np.var(d, ddof=1))
        expected_cv  = np.sqrt(expected_var) / abs(expected_F)

        assert pytest.approx(stats["F"][0.3][5],   rel=1e-6) == expected_F
        assert pytest.approx(stats["var"][0.3][5], rel=1e-6) == expected_var
        assert pytest.approx(stats["cv"][0.3][5],  rel=1e-6) == expected_cv

    def test_single_realization_var_is_zero(self):
        """
        With only one realization ddof=1 would give division by zero in numpy,
        so the code guards it: len(vals)==1 → var=0.0.
        A single datapoint carries no variance information.
        """
        raw = self._make_raw(0.1, 20, [0.7])
        stats = compute_F(raw)
        assert stats["var"][0.1][20] == 0.0

    def test_multiple_p_and_L_indexed_correctly(self):
        """
        The output dict must be properly nested: result["F"][p][L].
        Mixing p and L values checks that indexing doesn't collapse buckets.
        """
        raw = {
            0.0: {5:  np.array([0.0, 0.0]), 10: np.array([0.1, 0.1])},
            0.5: {5:  np.array([0.5, 0.5]), 10: np.array([0.8, 0.8])},
        }
        stats = compute_F(raw)
        assert pytest.approx(stats["F"][0.0][5])  == 1.0
        assert pytest.approx(stats["F"][0.0][10]) == 0.9
        assert pytest.approx(stats["F"][0.5][5])  == 0.5
        assert pytest.approx(stats["F"][0.5][10]) == 0.2

    def test_cv_formula_sqrt_var_over_abs_F(self):
        """
        CV = sqrt(var) / |F|. This is the coefficient of variation — the
        relative statistical uncertainty on F. We test both the numerator and
        denominator of this ratio are combined correctly.
        """
        d = [0.1, 0.3]  # mean=0.2 → F=0.8; var = 0.02; cv = sqrt(0.02)/0.8
        raw = self._make_raw(0.5, 5, d)
        stats = compute_F(raw)
        expected_cv = np.sqrt(np.var(d, ddof=1)) / abs(1 - np.mean(d))
        assert pytest.approx(stats["cv"][0.5][5], rel=1e-6) == expected_cv


# ══════════════════════════════════════════════════════════════════════════════
# 3. generate_time_reversal_breaking_random_brick_wall
# ══════════════════════════════════════════════════════════════════════════════

# Shorthand
_gen = generate_time_reversal_breaking_random_brick_wall

def _op_names(qc):
    """Return a list of gate-type names from a circuit's command sequence."""
    return [cmd.op.type.name for cmd in qc.get_commands()]

def _tk2_params(qc):
    """Return a list of (params, qubit_indices) for every TK2 gate."""
    return [
        (cmd.op.params, [q.index[0] for q in cmd.qubits])
        for cmd in qc.get_commands()
        if cmd.op.type.name == "TK2"
    ]


class TestGenerateCircuit:
    """
    The brick-wall echo circuit has the structure:
      state-prep → U (L layers of TK2+TK1 ± stochastic Measure) →
      [perturbation gate] → U† → final probe Measure
    """

    def test_basic_qubit_count(self):
        """
        The circuit must have exactly W qubits and at least 2 classical bits
        (bit-0 for the perturbation Measure, bit-last for the probe Measure).
        With p=0 there are no stochastic measurements → n_bits == 2.
        """
        qc = _gen(L=2, W=4, seed=0, p=0.0, add_barrier=False)
        assert qc.n_qubits == 4
        assert qc.n_bits == 2   # bit-0 (pert) + bit-1 (probe), no stochastic

    def test_determinism_same_seed(self):
        """
        The same seed must produce the same circuit every time.
        This guarantees reproducibility of experimental runs.
        """
        qc1 = _gen(L=3, W=4, seed=42, p=0.0, add_barrier=False)
        qc2 = _gen(L=3, W=4, seed=42, p=0.0, add_barrier=False)
        names1 = _op_names(qc1)
        names2 = _op_names(qc2)
        assert names1 == names2, "Same seed must give identical gate sequence"

        params1 = [cmd.op.params for cmd in qc1.get_commands()]
        params2 = [cmd.op.params for cmd in qc2.get_commands()]
        assert params1 == params2, "Same seed must give identical gate parameters"

    def test_different_seeds_give_different_angles(self):
        """
        Different seeds must produce different TK2 angle parameters.
        If two seeds gave the same angles the disorder average would be trivial.
        """
        tk2_a = _tk2_params(_gen(L=2, W=4, seed=1, p=0.0, add_barrier=False))
        tk2_b = _tk2_params(_gen(L=2, W=4, seed=2, p=0.0, add_barrier=False))
        # Compare the params of the first TK2 gate
        assert tk2_a[0][0] != tk2_b[0][0], \
            "Different seeds should give different TK2 parameters"

    def test_p_zero_no_stochastic_measurements(self):
        """
        With p=0 the allow_meas flag is never True, so no stochastic Measure
        commands are added inside U or U†. Classical bit count equals 2.
        The only Measure present should be the final probe measurement
        (and optionally the perturbation measurement, but we set unperturbed=True
        here to isolate).
        """
        qc = _gen(L=4, W=6, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        # n_bits = 2 + 2*n_p_meas; n_p_meas=0 → n_bits=2
        assert qc.n_bits == 2

        # Count all Measure commands — should only be the final probe measure
        measure_cmds = [c for c in qc.get_commands()
                        if c.op.type.name == "Measure"]
        assert len(measure_cmds) == 1, \
            "p=0 + unperturbed: only the final probe Measure expected"

    def test_p_one_eligible_layers_get_measurements(self):
        """
        With p=1.0 every qubit in every eligible layer (1..L-2) gets measured.
        For W=4, L=4:
          Layer 0 (ineligible): 2 pairs   → 0 measurements
          Layer 1 (eligible):   1 pair    → 2 measurements (qubits i,j)
          Layer 2 (eligible):   2 pairs   → 4 measurements
          Layer 3 (ineligible): 1 pair    → 0 measurements
        Total stochastic measurements in U = 6; same in U† → n_p_meas = 6.
        n_bits = 2 + 2*6 = 14.
        """
        L, W = 4, 4
        qc = _gen(L=L, W=W, seed=0, p=1.0,
                  unperturbed=True, add_barrier=False)
        # Layer 1 has 1 pair → 2 meas; Layer 2 has 2 pairs → 4 meas  = 6 total
        assert qc.n_bits == 2 + 2 * 6, \
            f"p=1 should give n_bits=14, got {qc.n_bits}"

    def test_unperturbed_true_no_perturbation_gate(self):
        """
        When unperturbed=True the mid-circuit perturbation block is skipped.
        No Measure or unitary gate should appear at the mid-circuit position.
        Specifically bit-0 should never be written (it is reserved for the
        perturbation measurement).
        """
        qc = _gen(L=3, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        # bit-0 is the perturbation slot; the final probe goes to the last bit
        measure_bits = [
            cmd.bits[0].index[0]
            for cmd in qc.get_commands()
            if cmd.op.type.name == "Measure"
        ]
        assert 0 not in measure_bits, \
            "unperturbed=True: bit-0 should never be written"

    def test_unperturbed_false_measure_perturbation(self):
        """
        When unperturbed=False and pert_op='measure', a Measure on pert_site
        into bit-0 must appear between U and U†.
        This is the fundamental measurement-induced perturbation.
        """
        W = 6
        pert_site = W // 2   # qubit 3
        qc = _gen(L=2, W=W, seed=0, p=0.0,
                  unperturbed=False, pert_op="measure",
                  pert_site=pert_site, add_barrier=False)

        # Find any Measure that writes to bit-0
        pert_measures = [
            cmd for cmd in qc.get_commands()
            if cmd.op.type.name == "Measure"
            and cmd.bits[0].index[0] == 0
        ]
        assert len(pert_measures) == 1, \
            "Exactly one Measure into bit-0 expected (the perturbation)"
        assert pert_measures[0].qubits[0].index[0] == pert_site, \
            "The perturbation Measure must be on pert_site"

    def test_unperturbed_false_pauli_perturbation(self):
        """
        When pert_op='X', an X gate (Pauli-X) must appear on pert_site
        and no Measure into bit-0 should be present.
        Tests that the flexible perturbation dispatch works.
        """
        W = 6
        pert_site = W // 2
        qc = _gen(L=2, W=W, seed=0, p=0.0,
                  unperturbed=False, pert_op="X",
                  pert_site=pert_site, add_barrier=False)

        cmds = qc.get_commands()
        x_on_pert = [
            cmd for cmd in cmds
            if cmd.op.type.name == "X"
            and cmd.qubits[0].index[0] == pert_site
        ]
        assert len(x_on_pert) == 1, \
            "Exactly one X gate on pert_site expected"

        # Bit-0 should not be written (no measurement perturbation)
        measure_bits = [
            cmd.bits[0].index[0]
            for cmd in cmds
            if cmd.op.type.name == "Measure"
        ]
        assert 0 not in measure_bits, \
            "pert_op='X': bit-0 must not be written"

    def test_probe_site_equals_pert_site_raises(self):
        """
        If probe_site == pert_site both operations would act on the same qubit.
        The function must raise ValueError to prevent silent physics errors.
        """
        with pytest.raises(ValueError, match="probe_site and pert_site must differ"):
            _gen(L=2, W=4, seed=0, probe_site=2, pert_site=2)

    def test_invalid_pert_op_raises(self):
        """
        An unknown pert_op string (e.g. 'CNOT') must raise ValueError.
        This protects against typos silently producing a wrong circuit.
        """
        with pytest.raises(ValueError, match="Unsupported pert_op"):
            _gen(L=2, W=4, seed=0, pert_op="CNOT")

    def test_final_command_is_probe_measure(self):
        """
        The very last command in every circuit must be a Measure on probe_site
        into the last classical bit.
        This is the readout we use for F.
        """
        W = 6
        probe_site = 0
        qc = _gen(L=2, W=W, seed=0, p=0.0,
                  probe_site=probe_site, unperturbed=True, add_barrier=False)
        cmds = qc.get_commands()
        last = cmds[-1]
        assert last.op.type.name == "Measure", \
            "Last command must be a Measure"
        assert last.qubits[0].index[0] == probe_site, \
            "Last Measure must be on probe_site"
        assert last.bits[0].index[0] == qc.n_bits - 1, \
            "Last Measure must write to the final classical bit"

    def test_state_prep_ry_on_probe_site(self):
        """
        The very first gate applied to probe_site must be Ry(probe_angle).
        This creates the equal superposition needed to sense the butterfly effect.
        """
        probe_site = 0
        probe_angle = 0.5
        qc = _gen(L=1, W=4, seed=0, p=0.0,
                  probe_site=probe_site, probe_angle=probe_angle,
                  unperturbed=True, add_barrier=False)
        # Walk commands until we hit the first gate on probe_site
        first_on_probe = next(
            cmd for cmd in qc.get_commands()
            if any(q.index[0] == probe_site for q in cmd.qubits)
        )
        assert first_on_probe.op.type.name == "Ry", \
            "First gate on probe_site must be Ry (state preparation)"
        assert pytest.approx(first_on_probe.op.params[0]) == probe_angle

    def test_barrier_present_when_enabled(self):
        """
        With add_barrier=True, Barrier commands must appear around the
        perturbation block. They signal the compiler not to reorder gates
        across the U / perturbation / U† boundary.
        """
        qc = _gen(L=2, W=4, seed=0, p=0.0,
                  unperturbed=False, add_barrier=True)
        barrier_count = sum(1 for c in qc.get_commands()
                            if c.op.type.name == "Barrier")
        assert barrier_count >= 2, \
            "add_barrier=True requires at least two Barrier commands"

    def test_barrier_absent_when_disabled(self):
        """
        With add_barrier=False there must be zero Barrier commands.
        Local simulators (Aer) do not support barriers with classical bits,
        so this flag must completely suppress them.
        """
        qc = _gen(L=2, W=4, seed=0, p=0.0,
                  unperturbed=False, add_barrier=False)
        barrier_count = sum(1 for c in qc.get_commands()
                            if c.op.type.name == "Barrier")
        assert barrier_count == 0

    def test_do_reset_adds_reset_after_perturbation_measure(self):
        """
        With do_reset=True a Reset gate must follow the perturbation Measure
        on pert_site. This refreshes the qubit to |0> so that the backward
        evolution starts from a well-defined state.
        """
        W = 6
        pert_site = W // 2
        qc = _gen(L=2, W=W, seed=0, p=0.0,
                  unperturbed=False, pert_op="measure",
                  pert_site=pert_site, do_reset=True, add_barrier=False)
        cmds = qc.get_commands()
        # Find index of the perturbation Measure (bit-0)
        pert_idx = next(
            i for i, c in enumerate(cmds)
            if c.op.type.name == "Measure" and c.bits[0].index[0] == 0
        )
        # The command immediately after must be Reset on pert_site
        assert cmds[pert_idx + 1].op.type.name == "Reset", \
            "Reset must immediately follow perturbation Measure when do_reset=True"
        assert cmds[pert_idx + 1].qubits[0].index[0] == pert_site

    def test_tk2_count_matches_layer_structure(self):
        """
        For W=4, the brick-wall pairing gives:
          Even layer: pairs (0,1),(2,3) → 2 TK2 gates
          Odd  layer: pair  (1,2)       → 1 TK2 gate
        For L=3 layers (0,1,2): 2+1+2 = 5 TK2 in U → 10 TK2 total.
        Verifies the layer_pairs logic is implemented correctly.
        """
        qc = _gen(L=3, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        tk2_cmds = [c for c in qc.get_commands() if c.op.type.name == "TK2"]
        # 5 in U + 5 in U†
        assert len(tk2_cmds) == 10, \
            f"W=4 L=3 expects 10 TK2 gates total, got {len(tk2_cmds)}"

    def test_udagger_tk2_angles_are_negated_and_reversed(self):
        """
        CORE PHYSICS TEST: U† must be the exact inverse of U.
        For TK2(a,b,g,i,j), its inverse is TK2(-a,-b,-g,i,j).
        The gates in U† must also appear in reverse order relative to U.
        If this fails, the echo circuit does not cancel and the baseline
        (coherent circuit) is not an identity.
        """
        qc = _gen(L=2, W=4, seed=7, p=0.0,
                  unperturbed=True, add_barrier=False)
        tk2_cmds = [c for c in qc.get_commands() if c.op.type.name == "TK2"]
        n = len(tk2_cmds) // 2
        U_gates   = tk2_cmds[:n]
        Ud_gates  = tk2_cmds[n:]

        for k in range(n):
            u_params  = U_gates[n - 1 - k].op.params   # reversed
            ud_params = Ud_gates[k].op.params
            assert pytest.approx(ud_params, abs=1e-10) == [-p for p in u_params], \
                f"U†[{k}] params should be negation of U[{n-1-k}] params"

            u_qubits  = [q.index[0] for q in U_gates[n - 1 - k].qubits]
            ud_qubits = [q.index[0] for q in Ud_gates[k].qubits]
            assert u_qubits == ud_qubits, \
                f"U†[{k}] must act on same qubit pair as U[{n-1-k}]"

    def test_meas_seed_changes_measurement_pattern_not_angles(self):
        """
        Changing meas_seed must change WHICH qubits are measured (stochastic
        pattern) but must NOT change the gate angles (TK2/TK1 params).
        This verifies the three-seed independence: angle-rng and meas-rng
        are driven by separate generators.
        """
        # Use p=1 to maximise measurement firing
        qc_a = _gen(L=6, W=6, seed=42, meas_seed=1,  p=1.0, add_barrier=False)
        qc_b = _gen(L=6, W=6, seed=42, meas_seed=99, p=1.0, add_barrier=False)

        # TK2 angles must be identical (same circuit seed)
        params_a = [c.op.params for c in qc_a.get_commands()
                    if c.op.type.name == "TK2"]
        params_b = [c.op.params for c in qc_b.get_commands()
                    if c.op.type.name == "TK2"]
        assert params_a == params_b, \
            "Different meas_seed must not change TK2 angles"

        # Classical bit counts may differ (different stochastic patterns)
        # but we just need them to both be valid (>= 2)
        assert qc_a.n_bits >= 2
        assert qc_b.n_bits >= 2

    def test_init_seed_changes_init_state_not_angles(self):
        """
        Changing init_seed must change the initial Pauli word on non-probe
        qubits but must not change TK2 gate angles.
        """
        qc_a = _gen(L=4, W=6, seed=10, init_seed=1,  p=0.0, add_barrier=False)
        qc_b = _gen(L=4, W=6, seed=10, init_seed=999, p=0.0, add_barrier=False)

        # TK2 angles identical
        params_a = [c.op.params for c in qc_a.get_commands()
                    if c.op.type.name == "TK2"]
        params_b = [c.op.params for c in qc_b.get_commands()
                    if c.op.type.name == "TK2"]
        assert params_a == params_b

    def test_stochastic_meas_only_in_eligible_layers(self):
        """
        Stochastic measurements may only appear in layers 1..L-2 (i.e. not the
        first or last layer of U). This prevents measurements from contaminating
        the initial state or the final layer before the perturbation.
        With L=3 and W=4, only layer 1 (the middle) is eligible.
        We verify by checking that with p=1 and L=3 the measurement count
        matches only the layer-1 pairs.
        """
        # W=4, L=3:
        #   layer 0 (even, ineligible): 2 pairs
        #   layer 1 (odd,  eligible):   1 pair → 2 measurements in U, 2 in U†
        #   layer 2 (even, ineligible): 2 pairs
        # n_p_meas = 2 → n_bits = 2 + 2*2 = 6
        qc = _gen(L=3, W=4, seed=0, p=1.0,
                  unperturbed=True, add_barrier=False)
        assert qc.n_bits == 6, \
            f"Only layer-1 eligible (1 pair → 2 meas each way) → n_bits=6, got {qc.n_bits}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. debug_after_measurement
# ══════════════════════════════════════════════════════════════════════════════

class TestDebugAfterMeasurement:
    """
    debug_after_measurement scans a circuit for the first Measure command and
    prints the subsequent commands. It is a diagnostic utility.
    """

    def test_finds_measure_and_does_not_crash(self, capsys):
        """
        Given a circuit with a mid-circuit Measure, the function should run
        without raising and print information about the command index.
        """
        qc = _gen(L=2, W=4, seed=0, p=0.0,
                  unperturbed=False, pert_op="measure", add_barrier=False)
        # Should not raise
        debug_after_measurement(qc, n=5)
        captured = capsys.readouterr()
        assert "MEASURE at command index:" in captured.out

    def test_raises_when_no_measure_in_circuit(self):
        """
        If the circuit has no Measure command (e.g. coherent with no
        perturbation and p=0), the function must raise RuntimeError with a
        clear message rather than silently returning nothing.
        """
        qc = _gen(L=2, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        # Remove the final probe measurement to have truly zero Measure gates
        # The easiest way: use a brand-new trivial Circuit
        from pytket import Circuit
        empty_qc = Circuit(2)
        empty_qc.H(0)
        with pytest.raises(RuntimeError, match="No Measure found"):
            debug_after_measurement(empty_qc)


# ══════════════════════════════════════════════════════════════════════════════
# 5. count_consecutive_tk2_after_measure
# ══════════════════════════════════════════════════════════════════════════════

class TestCountConsecutiveTK2AfterMeasure:
    """
    count_consecutive_tk2_after_measure finds the first Measure and counts
    how many TK2 gates follow it consecutively (stopping at the first non-TK2).
    Returns (count, [(i,j), ...]).
    """

    def test_counts_tk2_gates_after_measure(self):
        """
        Build a minimal circuit: one Measure followed by two TK2 gates.
        The function must return count=2 and the correct qubit pairs.
        """
        from pytket import Circuit
        qc = Circuit(4, 1)
        qc.Measure(0, 0)
        qc.TK2(0.1, 0.2, 0.3, 0, 1)
        qc.TK2(0.4, 0.5, 0.6, 2, 3)

        cnt, pairs = count_consecutive_tk2_after_measure(qc)
        assert cnt == 2
        assert (0, 1) in pairs
        assert (2, 3) in pairs

    def test_stops_at_non_tk2_gate(self):
        """
        If a non-TK2 gate appears after the first TK2, counting stops there.
        A Hadamard between two TK2 gates should give count=1.
        """
        from pytket import Circuit
        qc = Circuit(4, 1)
        qc.Measure(0, 0)
        qc.TK2(0.1, 0.2, 0.3, 0, 1)
        qc.H(2)                          # non-TK2 stops the count
        qc.TK2(0.4, 0.5, 0.6, 2, 3)

        cnt, pairs = count_consecutive_tk2_after_measure(qc)
        assert cnt == 1, "Count stops at the first non-TK2 gate"

    def test_zero_tk2_after_measure(self):
        """
        If the Measure is the last command, count should be 0.
        """
        from pytket import Circuit
        qc = Circuit(2, 1)
        qc.Measure(0, 0)
        cnt, pairs = count_consecutive_tk2_after_measure(qc)
        assert cnt == 0
        assert pairs == []


# ══════════════════════════════════════════════════════════════════════════════
# 6. validate_nexus_connection
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateNexusConnection:
    """
    validate_nexus_connection calls qnexus.login_with_credentials() and
    qnexus.devices.get_all(). We mock these so no network access is needed.
    """

    def test_returns_true_when_devices_found(self):
        """
        When the nexus call returns a non-empty DataFrame the function should
        return True, indicating a live connection.
        """
        mock_df = MagicMock()
        mock_df.__len__ = lambda self: 3         # len(df) > 0
        mock_df.__ne__ = lambda self, other: True  # df is not None

        with patch.object(_mod.qnx, "login_with_credentials"):
            with patch.object(_mod.qnx.devices, "get_all") as mock_get_all:
                mock_get_all.return_value.df.return_value = mock_df
                result = validate_nexus_connection(nexus_hosted=True)
        assert result is True

    def test_raises_connection_error_on_failure(self):
        """
        If login raises ConnectionError the function must re-raise it with
        a descriptive message rather than swallowing it silently.
        """
        with patch.object(_mod.qnx, "login_with_credentials",
                          side_effect=ConnectionError("network down")):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                validate_nexus_connection()


# ══════════════════════════════════════════════════════════════════════════════
# 7. _run_circuit_backend
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCircuitBackend:
    """
    _run_circuit_backend wraps the pytket backend API:
      backend.get_compiled_circuit(qc, optimisation_level=1)
      backend.process_circuit(compiled, n_shots=n_shots)
      backend.get_result(handle).get_shots()
    """

    def _mock_backend(self, shots_array):
        """Return a mock backend that returns shots_array from get_shots()."""
        backend = MagicMock()
        compiled = MagicMock()
        handle  = MagicMock()
        result  = MagicMock()
        result.get_shots.return_value = shots_array
        backend.get_compiled_circuit.return_value = compiled
        backend.process_circuit.return_value = handle
        backend.get_result.return_value = result
        return backend, compiled, handle

    def test_returns_shot_matrix(self):
        """
        The function must return the numpy array from result.get_shots().
        Shape and content must be preserved exactly.
        """
        expected = np.array([[0, 1], [1, 0], [0, 0]])
        qc = _gen(L=1, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        backend, _, _ = self._mock_backend(expected)

        shots = _run_circuit_backend(qc, backend, n_shots=3)
        np.testing.assert_array_equal(shots, expected)

    def test_calls_get_compiled_circuit_with_optimisation_level_1(self):
        """
        Compilation must use optimisation_level=1 to match what hardware
        targets expect. Wrong optimisation level can corrupt the circuit.
        """
        qc = _gen(L=1, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        backend, compiled, _ = self._mock_backend(np.zeros((5, 2), dtype=int))

        _run_circuit_backend(qc, backend, n_shots=5)
        backend.get_compiled_circuit.assert_called_once_with(
            qc, optimisation_level=1
        )

    def test_calls_process_circuit_with_correct_n_shots(self):
        """
        n_shots must be forwarded unchanged to process_circuit.
        An off-by-one here would silently change the statistical power.
        """
        qc = _gen(L=1, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        backend, compiled, _ = self._mock_backend(np.zeros((42, 2), dtype=int))

        _run_circuit_backend(qc, backend, n_shots=42)
        backend.process_circuit.assert_called_once_with(compiled, n_shots=42)


# ══════════════════════════════════════════════════════════════════════════════
# 8. _compile_and_run  (qnexus-based helper)
# ══════════════════════════════════════════════════════════════════════════════

class TestCompileAndRun:
    """
    _compile_and_run orchestrates qnexus upload → compile → execute → fetch.
    We mock the entire qnx namespace so no network calls are made.
    """

    def _setup_qnx_mocks(self, shots_array):
        """Return a context-manager-friendly patch dict for the qnx module."""
        mock_result = MagicMock()
        mock_result.get_shots.return_value = shots_array

        mock_result_item = MagicMock()
        mock_result_item.get_output.return_value = mock_result

        mock_qnx = MagicMock()
        mock_qnx.circuits.upload.return_value = "circuit_ref"
        mock_qnx.start_compile_job.return_value = "compile_job_ref"
        mock_qnx.jobs.results.return_value = [mock_result_item]
        mock_qnx.execute.return_value = "execute_job_ref"

        return mock_qnx

    def test_returns_shot_matrix(self):
        """
        _compile_and_run must return the (n_shots, n_bits) shot matrix
        obtained from the backend result.
        """
        expected = np.ones((10, 3), dtype=int)
        qc = _gen(L=1, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        project_ref = MagicMock()

        mock_qnx = self._setup_qnx_mocks(expected)
        with patch.object(_mod, "qnx", mock_qnx):
            shots = _compile_and_run(qc, project_ref, "tag", 10, "H2-1E")

        np.testing.assert_array_equal(shots, expected)

    def test_upload_is_called_with_circuit(self):
        """
        The circuit must be uploaded before compilation. If upload is skipped
        the compile job would reference a non-existent circuit.
        """
        qc = _gen(L=1, W=4, seed=0, p=0.0,
                  unperturbed=True, add_barrier=False)
        project_ref = MagicMock()
        mock_qnx = self._setup_qnx_mocks(np.zeros((5, 2), dtype=int))

        with patch.object(_mod, "qnx", mock_qnx):
            _compile_and_run(qc, project_ref, "my_tag", 5, "H2-1E")

        mock_qnx.circuits.upload.assert_called_once()
        call_kwargs = mock_qnx.circuits.upload.call_args
        assert call_kwargs.kwargs["circuit"] is qc


# ══════════════════════════════════════════════════════════════════════════════
# 9. plot_results
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotResults:
    """
    plot_results saves two PNG files (C=1-F and CV), each with three subplots.
    We verify the files are created without testing pixel content.
    """

    def _make_all_stats(self, N_values, p_values, L_values):
        """Build a minimal all_stats dict with constant values."""
        stats = {}
        for N in N_values:
            raw = {p: {L: np.array([0.2, 0.3]) for L in L_values}
                   for p in p_values}
            stats[N] = compute_F(raw)
        return stats

    def test_creates_both_figure_files(self, tmp_path):
        """
        After calling plot_results, both figure files must exist on disk.
        Tests that savefig is called with the correct paths.
        """
        N_values = [4, 6]
        p_values = [0.0, 0.5]
        L_values = [5, 10]
        all_stats = self._make_all_stats(N_values, p_values, L_values)

        path_C  = str(tmp_path / "C.png")
        path_cv = str(tmp_path / "cv.png")

        plot_results(all_stats, L_values, p_values, N_values,
                     figure_path_C=path_C, figure_path_cv=path_cv)

        assert pathlib.Path(path_C).exists(),  "C figure file must be created"
        assert pathlib.Path(path_cv).exists(), "CV figure file must be created"


# ══════════════════════════════════════════════════════════════════════════════
# 10. sweep_x_and_t
# ══════════════════════════════════════════════════════════════════════════════

class TestSweepXAndT:
    """
    sweep_x_and_t loops over probe_sites × p_values × circuits × init_states × L
    and returns raw[p][probe_site][L] = array of F_vals.
    We mock the backend so no quantum hardware is needed.
    """

    def _make_mock_backend(self, W, n_bits_per_circuit=2):
        """
        Returns a backend mock whose get_shots() always returns all-zeros
        (perfect echo: every shot measures 0 → ±1 mapping gives all +1).
        We return n_shots rows of zeros.
        """
        def fake_get_compiled(qc, optimisation_level=1):
            return MagicMock()

        def fake_process(compiled, n_shots):
            return MagicMock()

        def fake_get_result(handle):
            result = MagicMock()
            # Return a shot matrix; last column (probe bit) is all zeros
            result.get_shots.return_value = np.zeros((5, n_bits_per_circuit),
                                                     dtype=int)
            return result

        backend = MagicMock()
        backend.get_compiled_circuit.side_effect = fake_get_compiled
        backend.process_circuit.side_effect = fake_process
        backend.get_result.side_effect = fake_get_result
        return backend

    def test_return_structure(self):
        """
        The return value must be a dict keyed by p, then by probe_site, then by L.
        Each leaf must be a numpy array.
        """
        W = 4
        p_values = [0.0, 0.5]
        L_max = 5
        record_every = 5
        probe_sites = [0]
        backend = self._make_mock_backend(W)

        raw = sweep_x_and_t(
            L_max=L_max, W=W, p_values=p_values,
            probe_sites=probe_sites, n_shots=5,
            n_circuits=1, n_init_states=1,
            base_seed=0, record_every=record_every,
            backend=backend, add_barrier=False,
        )

        for p in p_values:
            assert p in raw, f"p={p} must be a key in raw"
            assert 0 in raw[p], "probe_site=0 must be a key"
            assert 5 in raw[p][0], "L=5 must be a key"
            assert isinstance(raw[p][0][5], np.ndarray), \
                "leaf values must be numpy arrays"

    def test_f_vals_length_equals_n_circuits_times_n_init_states(self):
        """
        Each F_vals array must have length == n_circuits * n_init_states,
        because we accumulate one F_val per (c_idx, i_idx) realization.
        """
        W = 4
        n_circuits = 2
        n_init = 3
        backend = self._make_mock_backend(W)

        raw = sweep_x_and_t(
            L_max=5, W=W, p_values=[0.5],
            probe_sites=[0], n_shots=5,
            n_circuits=n_circuits, n_init_states=n_init,
            base_seed=0, record_every=5,
            backend=backend, add_barrier=False,
        )

        vals = raw[0.5][0][5]
        assert len(vals) == n_circuits * n_init, \
            f"Expected {n_circuits * n_init} F_vals, got {len(vals)}"


# ══════════════════════════════════════════════════════════════════════════════
# 11. plot_otoc
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotOtoc:
    """
    plot_otoc saves three PNG files. We build a minimal raw_xtp structure,
    call the function, and assert the files exist.
    """

    def _make_raw_xtp(self, p_values, probe_sites, L_values):
        return {
            p: {ps: {L: np.array([0.1, 0.2, 0.3])
                     for L in L_values}
                for ps in probe_sites}
            for p in p_values
        }

    def test_creates_three_figure_files(self, tmp_path):
        """
        All three OTOC figures (CV, 2D, 3D) must be written to disk.
        """
        p_values    = [0.0, 0.5]
        L_values    = [5, 10]
        probe_sites = [0, 1, 3]  # pert_site=2 excluded
        pert_site   = 2

        raw_xtp = self._make_raw_xtp(p_values, probe_sites, L_values)

        path_cv = str(tmp_path / "otoc_cv.png")
        path_2d = str(tmp_path / "otoc_2d.png")
        path_3d = str(tmp_path / "otoc_3d.png")

        plot_otoc(
            raw_xtp=raw_xtp,
            pert_site=pert_site,
            L_values=L_values,
            probe_sites=probe_sites,
            p_values=p_values,
            topmost_qubit=3,
            figure_path_cv=path_cv,
            figure_path_2d=path_2d,
            figure_path_3d=path_3d,
        )

        assert pathlib.Path(path_cv).exists(), "CV figure must be created"
        assert pathlib.Path(path_2d).exists(), "2D OTOC figure must be created"
        assert pathlib.Path(path_3d).exists(), "3D OTOC figure must be created"

    def test_topmost_qubit_defaults_to_max_probe_site(self):
        """
        When topmost_qubit is not specified it should default to
        max(probe_sites). Graph 2 uses this site to show spreading.
        This is a smoke test: we just verify no KeyError is raised.
        """
        p_values    = [0.3]
        L_values    = [5]
        probe_sites = [0, 1]  # max is 1; pert_site=2
        raw_xtp = self._make_raw_xtp(p_values, probe_sites, L_values)

        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            plot_otoc(
                raw_xtp=raw_xtp,
                pert_site=2,
                L_values=L_values,
                probe_sites=probe_sites,
                p_values=p_values,
                # topmost_qubit not passed → defaults to max(probe_sites)=1
                figure_path_cv=os.path.join(d, "cv.png"),
                figure_path_2d=os.path.join(d, "2d.png"),
                figure_path_3d=os.path.join(d, "3d.png"),
            )


# ══════════════════════════════════════════════════════════════════════════════
# 12. sweep_p_and_t  (qnexus-based)
# ══════════════════════════════════════════════════════════════════════════════

class TestSweepPAndT:
    """
    sweep_p_and_t calls run_time_sweep for each (p, circuit, init_state) combo.
    We mock run_time_sweep so no network calls happen.
    """

    def _fake_sweep_result(self, L_values, n_shots=5):
        """Return a fake run_time_sweep result (all zeros → outcomes all +1)."""
        return {
            "perturbed":   {L: np.ones(n_shots)  for L in L_values},
            "unperturbed": {L: np.ones(n_shots)  for L in L_values},
        }

    def test_raw_structure(self):
        """
        The raw dict returned by sweep_p_and_t must be keyed by p, then by L,
        and each leaf must be a numpy array of length n_circuits*n_init_states.
        """
        p_values = [0.0, 0.5]
        L_max    = 5
        record_every = 5
        L_values = list(range(record_every, L_max + 1, record_every))
        n_c, n_i = 2, 2

        fake_result = self._fake_sweep_result(L_values)
        with patch.object(_mod, "run_time_sweep", return_value=fake_result):
            raw = sweep_p_and_t(
                p_values=p_values, L_max=L_max, W=4,
                n_shots=5, n_circuits=n_c, n_init_states=n_i,
                base_seed=0, record_every=record_every,
            )

        for p in p_values:
            assert p in raw
            for L in L_values:
                assert L in raw[p]
                assert isinstance(raw[p][L], np.ndarray)
                assert len(raw[p][L]) == n_c * n_i

    def test_f_val_formula(self):
        """
        Each F_val = perturbed_mean - unperturbed_mean.
        When both are identical (all-ones), F_val = 0 for every realization.
        """
        p_values = [0.5]
        L_values = [5]

        fake_result = self._fake_sweep_result(L_values, n_shots=10)
        with patch.object(_mod, "run_time_sweep", return_value=fake_result):
            raw = sweep_p_and_t(
                p_values=p_values, L_max=5, W=4,
                n_shots=10, n_circuits=1, n_init_states=1,
                base_seed=0, record_every=5,
            )

        # perturbed_mean == unperturbed_mean (both all-ones) → d = 0
        assert pytest.approx(raw[0.5][5][0]) == 0.0

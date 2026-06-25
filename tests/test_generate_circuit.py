"""
tests/test_generate_circuit.py

Tests for generate_time_reversal_breaking_random_brick_wall.

Circuit structure:
  state-prep  →  U (L brick-wall layers of TK2+TK1 ± stochastic Measure)
              →  [perturbation gate]
              →  U†
              →  final probe Measure

The most important test is test_udagger_tk2_angles_are_negated_and_reversed,
which verifies the echo structure: U† is the exact inverse of U.
"""

import pytest
import numpy as np
from tests.helpers import generate_time_reversal_breaking_random_brick_wall as gen


# ── Small helpers ──────────────────────────────────────────────────────────────

def _op_names(qc):
    """List of gate-type name strings in command order."""
    return [cmd.op.type.name for cmd in qc.get_commands()]


def _tk2_cmds(qc):
    """All TK2 Command objects in order."""
    return [cmd for cmd in qc.get_commands() if cmd.op.type.name == "TK2"]


# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateCircuit:

    # ── Dimensions ──────────────────────────────────────────────────────────

    def test_qubit_count_equals_W(self):
        """Circuit must contain exactly W qubits."""
        qc = gen(L=2, W=6, seed=0, p=0.0, add_barrier=False)
        assert qc.n_qubits == 6

    def test_n_bits_equals_two_when_p_is_zero(self):
        """
        With p=0 and unperturbed=True: no stochastic Measure commands.
        n_bits = 2 + 2*n_p_meas = 2 + 0 = 2
          bit 0 → perturbation slot (unused but allocated)
          bit 1 → final probe measurement
        """
        qc = gen(L=2, W=4, seed=0, p=0.0, unperturbed=True, add_barrier=False)
        assert qc.n_bits == 2

    def test_n_bits_grows_with_p_one(self):
        """
        With p=1 all eligible layer pairs produce stochastic measurements.
        W=4, L=4: layer 1 (1 pair) and layer 2 (2 pairs) are eligible.
        Each pair contributes 2 measurements in U and 2 in U†.
        n_p_meas = (1+2)*2 = 6  →  n_bits = 2 + 2*6 = 14.
        """
        qc = gen(L=4, W=4, seed=0, p=1.0, unperturbed=True, add_barrier=False)
        assert qc.n_bits == 14

    # ── Reproducibility ─────────────────────────────────────────────────────

    def test_same_seed_gives_identical_circuit(self):
        """Same seed must produce bit-for-bit identical gate sequences."""
        qc1 = gen(L=3, W=4, seed=42, p=0.0, add_barrier=False)
        qc2 = gen(L=3, W=4, seed=42, p=0.0, add_barrier=False)
        assert _op_names(qc1) == _op_names(qc2)
        params1 = [cmd.op.params for cmd in qc1.get_commands()]
        params2 = [cmd.op.params for cmd in qc2.get_commands()]
        assert params1 == params2

    def test_different_seeds_give_different_tk2_angles(self):
        """
        Different seeds must give different TK2 parameters.
        If they were the same the disorder average would be trivial.
        """
        tk2_a = _tk2_cmds(gen(L=2, W=4, seed=1, p=0.0, add_barrier=False))
        tk2_b = _tk2_cmds(gen(L=2, W=4, seed=2, p=0.0, add_barrier=False))
        assert tk2_a[0].op.params != tk2_b[0].op.params

    # ── Perturbation gate ────────────────────────────────────────────────────

    def test_unperturbed_true_bit0_never_written(self):
        """
        unperturbed=True skips the perturbation block.
        Bit 0 (the perturbation slot) must never be written.
        """
        qc = gen(L=3, W=4, seed=0, p=0.0, unperturbed=True, add_barrier=False)
        measure_bits = [
            cmd.bits[0].index[0]
            for cmd in qc.get_commands()
            if cmd.op.type.name == "Measure"
        ]
        assert 0 not in measure_bits

    def test_unperturbed_false_measure_on_pert_site_at_bit0(self):
        """
        unperturbed=False with pert_op='measure' must insert a Measure on
        pert_site into bit 0 between U and U†.
        """
        W, pert = 6, 3
        qc = gen(L=2, W=W, seed=0, p=0.0,
                 unperturbed=False, pert_op="measure",
                 pert_site=pert, add_barrier=False)
        pert_meas = [
            cmd for cmd in qc.get_commands()
            if cmd.op.type.name == "Measure" and cmd.bits[0].index[0] == 0
        ]
        assert len(pert_meas) == 1
        assert pert_meas[0].qubits[0].index[0] == pert

    def test_unperturbed_false_pauli_x_on_pert_site(self):
        """
        pert_op='X' must insert exactly one X gate on pert_site.
        Bit 0 must not be written (no measurement perturbation).
        """
        W, pert = 6, 3
        qc = gen(L=2, W=W, seed=0, p=0.0,
                 unperturbed=False, pert_op="X",
                 pert_site=pert, add_barrier=False)
        x_gates = [
            cmd for cmd in qc.get_commands()
            if cmd.op.type.name == "X"
            and cmd.qubits[0].index[0] == pert
        ]
        assert len(x_gates) == 1
        measure_bits = [
            cmd.bits[0].index[0]
            for cmd in qc.get_commands()
            if cmd.op.type.name == "Measure"
        ]
        assert 0 not in measure_bits

    def test_do_reset_adds_reset_after_perturbation_measure(self):
        """
        do_reset=True must insert a Reset on pert_site immediately after
        the perturbation Measure, refreshing the qubit to |0> for U†.
        """
        W, pert = 6, 3
        qc = gen(L=2, W=W, seed=0, p=0.0,
                 unperturbed=False, pert_op="measure",
                 pert_site=pert, do_reset=True, add_barrier=False)
        cmds = qc.get_commands()
        pert_idx = next(
            i for i, c in enumerate(cmds)
            if c.op.type.name == "Measure" and c.bits[0].index[0] == 0
        )
        assert cmds[pert_idx + 1].op.type.name == "Reset"
        assert cmds[pert_idx + 1].qubits[0].index[0] == pert

    # ── Validation ───────────────────────────────────────────────────────────

    def test_probe_equals_pert_raises_value_error(self):
        """probe_site == pert_site must raise ValueError immediately."""
        with pytest.raises(ValueError, match="probe_site and pert_site must differ"):
            gen(L=2, W=4, seed=0, probe_site=2, pert_site=2)

    def test_invalid_pert_op_raises_value_error(self):
        """An unrecognised pert_op string must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported pert_op"):
            gen(L=2, W=4, seed=0, pert_op="CNOT")

    # ── State preparation ────────────────────────────────────────────────────

    def test_first_gate_on_probe_site_is_ry(self):
        """
        The probe site must be initialised with Ry(probe_angle) as its very
        first gate.  This creates the equal superposition needed to sense
        the butterfly effect.
        """
        probe, angle = 0, 0.5
        qc = gen(L=1, W=4, seed=0, p=0.0,
                 probe_site=probe, probe_angle=angle,
                 unperturbed=True, add_barrier=False)
        first_on_probe = next(
            cmd for cmd in qc.get_commands()
            if any(q.index[0] == probe for q in cmd.qubits)
        )
        assert first_on_probe.op.type.name == "Ry"
        assert pytest.approx(first_on_probe.op.params[0]) == angle

    def test_final_command_is_probe_measure(self):
        """
        The very last command must be Measure on probe_site into the last bit.
        This is the readout we use to compute F.
        """
        probe = 0
        qc = gen(L=2, W=6, seed=0, p=0.0,
                 probe_site=probe, unperturbed=True, add_barrier=False)
        last = qc.get_commands()[-1]
        assert last.op.type.name == "Measure"
        assert last.qubits[0].index[0] == probe
        assert last.bits[0].index[0] == qc.n_bits - 1

    # ── Barriers ─────────────────────────────────────────────────────────────

    def test_barriers_present_when_add_barrier_true(self):
        """add_barrier=True must produce at least two Barrier commands."""
        qc = gen(L=2, W=4, seed=0, p=0.0,
                 unperturbed=False, add_barrier=True)
        count = sum(1 for c in qc.get_commands() if c.op.type.name == "Barrier")
        assert count >= 2

    def test_no_barriers_when_add_barrier_false(self):
        """
        add_barrier=False must produce zero Barrier commands.
        Local simulators (Aer) reject barriers with classical bits, so
        this flag must completely suppress them.
        """
        qc = gen(L=2, W=4, seed=0, p=0.0,
                 unperturbed=False, add_barrier=False)
        count = sum(1 for c in qc.get_commands() if c.op.type.name == "Barrier")
        assert count == 0

    # ── Gate structure ───────────────────────────────────────────────────────

    def test_tk2_count_matches_brick_wall_pairing(self):
        """
        W=4 brick-wall pairing:
          Even layer: (0,1),(2,3) → 2 pairs
          Odd  layer: (1,2)       → 1 pair
        L=3 layers (0,1,2): 2+1+2 = 5 TK2 gates in U → 10 total.
        """
        qc = gen(L=3, W=4, seed=0, p=0.0,
                 unperturbed=True, add_barrier=False)
        tk2 = _tk2_cmds(qc)
        assert len(tk2) == 10

    def test_stochastic_meas_only_in_eligible_layers(self):
        """
        Stochastic measurements are forbidden in the first (l=0) and last
        (l=L-1) layers.  With W=4, L=3 and p=1.0 only layer 1 (1 pair) is
        eligible → 2 meas in U and 2 in U† → n_p_meas=2 → n_bits=6.
        """
        qc = gen(L=3, W=4, seed=0, p=1.0,
                 unperturbed=True, add_barrier=False)
        assert qc.n_bits == 6

    # ── CORE PHYSICS: U† is the exact inverse of U ──────────────────────────

    def test_udagger_tk2_angles_are_negated_and_reversed(self):
        """
        For TK2(a,b,g, i,j) in U, the corresponding gate in U† must be
        TK2(-a,-b,-g, i,j) applied in reversed layer order.

        If this fails, U†U ≠ I and the echo circuit is broken — the
        coherent baseline would not return to the initial state.
        """
        qc = gen(L=2, W=4, seed=7, p=0.0,
                 unperturbed=True, add_barrier=False)
        tk2 = _tk2_cmds(qc)
        n = len(tk2) // 2
        U_gates  = tk2[:n]
        Ud_gates = tk2[n:]

        for k in range(n):
            u_params  = U_gates[n - 1 - k].op.params    # reversed
            ud_params = Ud_gates[k].op.params
            assert pytest.approx(ud_params, abs=1e-10) == [-p for p in u_params], \
                f"U†[{k}] params must negate U[{n-1-k}] params"

            u_qubs  = [q.index[0] for q in U_gates[n - 1 - k].qubits]
            ud_qubs = [q.index[0] for q in Ud_gates[k].qubits]
            assert u_qubs == ud_qubs, \
                f"U†[{k}] must act on same qubit pair as U[{n-1-k}]"

    # ── Seed independence ────────────────────────────────────────────────────

    def test_meas_seed_changes_pattern_not_angles(self):
        """
        Different meas_seed → different measurement sites (stochastic pattern),
        but TK2 gate angles must stay identical (same circuit seed).
        """
        qc_a = gen(L=6, W=6, seed=42, meas_seed=1,  p=1.0, add_barrier=False)
        qc_b = gen(L=6, W=6, seed=42, meas_seed=99, p=1.0, add_barrier=False)
        params_a = [c.op.params for c in qc_a.get_commands()
                    if c.op.type.name == "TK2"]
        params_b = [c.op.params for c in qc_b.get_commands()
                    if c.op.type.name == "TK2"]
        assert params_a == params_b

    def test_init_seed_changes_init_state_not_angles(self):
        """
        Different init_seed → different initial Pauli word, but TK2 angles
        must remain identical (same circuit seed).
        """
        qc_a = gen(L=4, W=6, seed=10, init_seed=1,   p=0.0, add_barrier=False)
        qc_b = gen(L=4, W=6, seed=10, init_seed=999,  p=0.0, add_barrier=False)
        params_a = [c.op.params for c in qc_a.get_commands()
                    if c.op.type.name == "TK2"]
        params_b = [c.op.params for c in qc_b.get_commands()
                    if c.op.type.name == "TK2"]
        assert params_a == params_b

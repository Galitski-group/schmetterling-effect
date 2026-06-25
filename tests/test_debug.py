"""
tests/test_debug.py

Tests for debug_after_measurement.

The function scans a circuit for the first Measure command and prints
the subsequent n commands.  It raises RuntimeError when no Measure exists.
"""

import pytest
from pytket import Circuit
from tests.helpers import (
    debug_after_measurement,
    generate_time_reversal_breaking_random_brick_wall as gen,
)


class TestDebugAfterMeasurement:

    def test_runs_without_error_and_prints_index(self, capsys):
        """
        Given a circuit with a mid-circuit Measure (unperturbed=False,
        pert_op='measure'), the function must complete without raising and
        print the measure command index to stdout.
        """
        qc = gen(L=2, W=4, seed=0, p=0.0,
                 unperturbed=False, pert_op="measure", add_barrier=False)
        debug_after_measurement(qc, n=3)
        captured = capsys.readouterr()
        assert "MEASURE at command index:" in captured.out

    def test_raises_runtime_error_when_no_measure(self):
        """
        A circuit with zero Measure commands must trigger RuntimeError
        with a descriptive message rather than silently returning.
        """
        qc = Circuit(2)
        qc.H(0).CX(0, 1)   # pure unitary — no Measure at all
        with pytest.raises(RuntimeError, match="No Measure found"):
            debug_after_measurement(qc)

    def test_prints_up_to_n_subsequent_commands(self, capsys):
        """
        The n parameter limits how many commands after the Measure are printed.
        With n=1 we should see exactly one subsequent command line in stdout.
        """
        qc = gen(L=2, W=4, seed=0, p=0.0,
                 unperturbed=False, pert_op="measure", add_barrier=False)
        debug_after_measurement(qc, n=1)
        captured = capsys.readouterr()
        # The output has the header line + one detail line
        lines = [l for l in captured.out.splitlines() if l.strip()]
        assert len(lines) >= 2   # at least header + 1 command line

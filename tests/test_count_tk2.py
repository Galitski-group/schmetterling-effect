"""
tests/test_count_tk2.py

Tests for count_consecutive_tk2_after_measure.

The function finds the first Measure command and counts how many TK2
gates follow it consecutively (stopping at the first non-TK2 gate).
Returns (count, [(i,j), ...]) of qubit-index pairs.
"""

import pytest
from pytket import Circuit
from tests.helpers import count_consecutive_tk2_after_measure


class TestCountConsecutiveTK2AfterMeasure:

    def test_counts_two_consecutive_tk2(self):
        """
        Circuit: Measure → TK2(0,1) → TK2(2,3).
        Expected count = 2 and both qubit pairs returned.
        """
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
        Circuit: Measure → TK2 → H → TK2.
        The H breaks the consecutive run so count must be 1.
        """
        qc = Circuit(4, 1)
        qc.Measure(0, 0)
        qc.TK2(0.1, 0.2, 0.3, 0, 1)
        qc.H(2)                           # non-TK2 stops counting
        qc.TK2(0.4, 0.5, 0.6, 2, 3)

        cnt, pairs = count_consecutive_tk2_after_measure(qc)
        assert cnt == 1

    def test_zero_tk2_after_measure(self):
        """
        Measure is the last command → count = 0, pairs = [].
        """
        qc = Circuit(2, 1)
        qc.Measure(0, 0)

        cnt, pairs = count_consecutive_tk2_after_measure(qc)
        assert cnt == 0
        assert pairs == []

    def test_returns_correct_qubit_pair_indices(self):
        """
        Returned qubit pairs must match the actual qubit indices of each TK2,
        not their position in the circuit.
        """
        qc = Circuit(6, 1)
        qc.Measure(0, 0)
        qc.TK2(0.1, 0.2, 0.3, 3, 4)   # qubits 3 and 4

        _, pairs = count_consecutive_tk2_after_measure(qc)
        assert pairs == [(3, 4)]

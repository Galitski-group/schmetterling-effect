"""
tests/test_run_circuit_backend.py

Tests for _run_circuit_backend.

The function wraps the pytket backend API:
  1. backend.get_compiled_circuit(qc, optimisation_level=1)
  2. backend.process_circuit(compiled, n_shots=n_shots)
  3. backend.get_result(handle).get_shots()

No real backend is needed; a MagicMock is used throughout.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from tests.helpers import (
    _run_circuit_backend,
    generate_time_reversal_breaking_random_brick_wall as gen,
)


def _make_backend(shots_array: np.ndarray):
    """
    Return a fully configured mock backend whose get_shots() returns
    shots_array, plus the intermediate mock objects for assertion.
    """
    backend  = MagicMock()
    compiled = MagicMock()
    handle   = MagicMock()
    result   = MagicMock()
    result.get_shots.return_value = shots_array
    backend.get_compiled_circuit.return_value = compiled
    backend.process_circuit.return_value      = handle
    backend.get_result.return_value           = result
    return backend, compiled, handle


class TestRunCircuitBackend:

    def _simple_circuit(self):
        return gen(L=1, W=4, seed=0, p=0.0, unperturbed=True, add_barrier=False)

    def test_returns_shot_matrix_unchanged(self):
        """
        The function must return exactly the numpy array from result.get_shots().
        Shape and content must be preserved.
        """
        expected = np.array([[0, 1], [1, 0], [0, 0]])
        backend, _, _ = _make_backend(expected)
        shots = _run_circuit_backend(self._simple_circuit(), backend, n_shots=3)
        np.testing.assert_array_equal(shots, expected)

    def test_calls_get_compiled_circuit_with_optimisation_level_1(self):
        """
        Compilation must use optimisation_level=1.  A wrong level can
        produce an incorrect or unrunnable circuit on hardware.
        """
        qc = self._simple_circuit()
        backend, compiled, _ = _make_backend(np.zeros((5, 2), dtype=int))
        _run_circuit_backend(qc, backend, n_shots=5)
        backend.get_compiled_circuit.assert_called_once_with(
            qc, optimisation_level=1
        )

    def test_calls_process_circuit_with_exact_n_shots(self):
        """
        n_shots must be forwarded unchanged to process_circuit.
        An off-by-one error here silently changes the statistical power.
        """
        backend, compiled, _ = _make_backend(np.zeros((42, 2), dtype=int))
        _run_circuit_backend(self._simple_circuit(), backend, n_shots=42)
        backend.process_circuit.assert_called_once_with(compiled, n_shots=42)

    def test_calls_get_result_with_process_handle(self):
        """
        get_result must receive the handle returned by process_circuit.
        Passing the wrong object here would return results for a different job.
        """
        backend, _, handle = _make_backend(np.zeros((10, 2), dtype=int))
        _run_circuit_backend(self._simple_circuit(), backend, n_shots=10)
        backend.get_result.assert_called_once_with(handle)

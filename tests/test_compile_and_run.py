"""
tests/test_compile_and_run.py

Tests for _compile_and_run (qnexus-based pipeline helper).

The function orchestrates:
  1. qnx.circuits.upload(circuit=qc, ...)
  2. qnx.start_compile_job(programs=[circuit_ref], ...)
  3. qnx.jobs.wait_for(compile_job_ref)
  4. [item.get_output() for item in qnx.jobs.results(compile_job_ref)]
  5. qnx.execute(programs=compiled, ...)
  6. qnx.jobs.wait_for(execute_job_ref)
  7. qnx.jobs.results(execute_job_ref)[0].get_output().get_shots()

All qnexus calls are mocked.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from tests.helpers import (
    _compile_and_run,
    generate_time_reversal_breaking_random_brick_wall as gen,
    mod,
)


def _setup_qnx_mocks(shots_array: np.ndarray):
    """
    Builds a mock qnx namespace wired to return shots_array from get_shots().
    Returns (mock_qnx, mock_result_item) for assertion inspection.
    """
    mock_result      = MagicMock()
    mock_result.get_shots.return_value = shots_array
    mock_result_item = MagicMock()
    mock_result_item.get_output.return_value = mock_result

    mock_qnx = MagicMock()
    mock_qnx.circuits.upload.return_value   = "circuit_ref"
    mock_qnx.start_compile_job.return_value = "compile_job_ref"
    mock_qnx.jobs.results.return_value      = [mock_result_item]
    mock_qnx.execute.return_value           = "execute_job_ref"

    return mock_qnx, mock_result_item


class TestCompileAndRun:

    def _simple_circuit(self):
        return gen(L=1, W=4, seed=0, p=0.0, unperturbed=True, add_barrier=False)

    def test_returns_shot_matrix(self):
        """
        The function must return exactly the shot matrix from get_shots().
        """
        expected = np.ones((10, 3), dtype=int)
        mock_qnx, _ = _setup_qnx_mocks(expected)
        project_ref  = MagicMock()

        with patch.object(mod, "qnx", mock_qnx):
            shots = _compile_and_run(
                self._simple_circuit(), project_ref, "tag", 10, "H2-1E"
            )

        np.testing.assert_array_equal(shots, expected)

    def test_upload_called_with_circuit(self):
        """
        The circuit object must be passed to qnx.circuits.upload as the
        'circuit' keyword argument.  Uploading the wrong object would
        compile and run a different circuit.
        """
        mock_qnx, _ = _setup_qnx_mocks(np.zeros((5, 2), dtype=int))
        qc = self._simple_circuit()

        with patch.object(mod, "qnx", mock_qnx):
            _compile_and_run(qc, MagicMock(), "tag", 5, "H2-1E")

        mock_qnx.circuits.upload.assert_called_once()
        assert mock_qnx.circuits.upload.call_args.kwargs["circuit"] is qc

    def test_compile_job_waits_before_execute(self):
        """
        qnx.jobs.wait_for must be called on the compile job ref before
        qnx.execute is called.  Out-of-order calls would try to execute
        a circuit that hasn't been compiled yet.
        """
        mock_qnx, _ = _setup_qnx_mocks(np.zeros((5, 2), dtype=int))

        call_order = []
        mock_qnx.jobs.wait_for.side_effect = lambda ref: call_order.append(
            ("wait", ref)
        )
        mock_qnx.execute.side_effect = lambda **kw: call_order.append(
            ("execute",)
        ) or "execute_job_ref"

        with patch.object(mod, "qnx", mock_qnx):
            _compile_and_run(self._simple_circuit(), MagicMock(), "t", 5, "H2-1E")

        first_wait = next(x for x in call_order if x[0] == "wait")
        first_exec = next(x for x in call_order if x[0] == "execute")
        assert call_order.index(first_wait) < call_order.index(first_exec), \
            "Compile job must complete before execute is called"

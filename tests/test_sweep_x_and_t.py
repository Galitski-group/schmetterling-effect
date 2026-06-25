"""
tests/test_sweep_x_and_t.py

Tests for sweep_x_and_t.

The function loops over probe_sites × p_values × (c_idx, i_idx) ×
L_values, submitting perturbed and unperturbed circuits via a backend,
and returns raw[p][probe_site][L] = np.ndarray of F_vals.

A mock backend is used so no quantum hardware is needed.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from tests.helpers import sweep_x_and_t


def _make_backend(n_bits: int = 2, n_shots: int = 5):
    """
    Return a backend mock whose get_shots() returns all-zeros
    (probe bit = 0 → outcome +1 after the ±1 mapping).
    """
    backend = MagicMock()

    def _get_result(_handle):
        result = MagicMock()
        result.get_shots.return_value = np.zeros((n_shots, n_bits), dtype=int)
        return result

    backend.get_compiled_circuit.side_effect = lambda qc, **kw: MagicMock()
    backend.process_circuit.side_effect      = lambda compiled, **kw: MagicMock()
    backend.get_result.side_effect           = _get_result
    return backend


class TestSweepXAndT:

    def test_return_structure_has_correct_keys(self):
        """
        raw must be keyed raw[p][probe_site][L], all as expected types.
        """
        W            = 4
        p_values     = [0.0, 0.5]
        probe_sites  = [0]
        record_every = 5
        L_max        = 5

        backend = _make_backend()
        raw = sweep_x_and_t(
            L_max=L_max, W=W, p_values=p_values,
            probe_sites=probe_sites, n_shots=5,
            n_circuits=1, n_init_states=1,
            base_seed=0, record_every=record_every,
            backend=backend, add_barrier=False,
        )

        for p in p_values:
            assert p in raw
            assert 0 in raw[p]
            assert 5 in raw[p][0]
            assert isinstance(raw[p][0][5], np.ndarray)

    def test_f_vals_length_equals_n_circuits_times_n_init(self):
        """
        Each F_vals array must have exactly n_circuits * n_init_states entries,
        one per disorder realization.
        """
        n_c, n_i = 2, 3
        backend = _make_backend()

        raw = sweep_x_and_t(
            L_max=5, W=4, p_values=[0.5],
            probe_sites=[0], n_shots=5,
            n_circuits=n_c, n_init_states=n_i,
            base_seed=0, record_every=5,
            backend=backend, add_barrier=False,
        )

        assert len(raw[0.5][0][5]) == n_c * n_i

    def test_all_zeros_shots_give_zero_f_val(self):
        """
        When both perturbed and unperturbed circuits return all-zero shots,
        out_p.mean() == out_u.mean() == 1.0 → F_val = 0.0 for every realization.
        """
        backend = _make_backend(n_bits=2, n_shots=10)
        raw = sweep_x_and_t(
            L_max=5, W=4, p_values=[0.5],
            probe_sites=[0], n_shots=10,
            n_circuits=1, n_init_states=1,
            base_seed=0, record_every=5,
            backend=backend, add_barrier=False,
        )
        assert pytest.approx(raw[0.5][0][5][0]) == 0.0

    def test_default_probe_sites_excludes_pert_site(self):
        """
        When probe_sites is not specified it must default to all qubits
        except pert_site, so probe and pert never coincide.
        """
        W        = 4
        backend  = _make_backend()
        pert_site = W // 2   # qubit 2

        raw = sweep_x_and_t(
            L_max=5, W=W, p_values=[0.5],
            n_shots=5, n_circuits=1, n_init_states=1,
            base_seed=0, record_every=5,
            pert_site=pert_site,
            backend=backend, add_barrier=False,
        )

        for p in raw:
            assert pert_site not in raw[p], \
                "pert_site must not appear as a probe site"

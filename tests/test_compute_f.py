"""
tests/test_compute_f.py

Tests for compute_F.

compute_F takes raw[p][L] = 1-D array of per-realization values
    d = pert_mean - unpert_mean
and returns a dict with keys "F", "var", "cv" where:
    F   = 1 - mean(d)
    var = sample variance of d   (ddof=1)
    cv  = sqrt(var) / |F|        (inf when F == 0)
"""

import numpy as np
import pytest
from tests.helpers import compute_F


def _raw(p, L, vals):
    """Wrap a list into the raw dict format expected by compute_F."""
    return {p: {L: np.array(vals)}}


class TestComputeF:

    def test_perfect_echo_d_is_zero(self):
        """
        d = 0 for every realization → mean(d) = 0 → F = 1 - 0 = 1.
        This is the 'no butterfly effect' baseline: pert and unpert
        circuits produce identical probe outcomes.
        """
        stats = compute_F(_raw(0.5, 10, [0.0, 0.0, 0.0]))
        assert pytest.approx(stats["F"][0.5][10])   == 1.0
        assert pytest.approx(stats["var"][0.5][10]) == 0.0

    def test_full_decoherence_d_is_one(self):
        """
        d = 1 for every realization → F = 0 → CV = inf.
        The perturbation has maximally scrambled the probe.
        """
        stats = compute_F(_raw(0.5, 10, [1.0, 1.0, 1.0]))
        assert pytest.approx(stats["F"][0.5][10]) == 0.0
        assert stats["cv"][0.5][10] == np.inf

    def test_known_values_formula(self):
        """
        Manual check with d = [0.2, 0.4, 0.6]:
          mean(d) = 0.4  →  F = 0.6
          var(d)  = sample variance = 0.04
          cv      = sqrt(0.04) / 0.6 ≈ 0.3333
        """
        d = [0.2, 0.4, 0.6]
        stats = compute_F(_raw(0.3, 5, d))

        expected_F   = 1.0 - np.mean(d)
        expected_var = float(np.var(d, ddof=1))
        expected_cv  = np.sqrt(expected_var) / abs(expected_F)

        assert pytest.approx(stats["F"][0.3][5],   rel=1e-6) == expected_F
        assert pytest.approx(stats["var"][0.3][5], rel=1e-6) == expected_var
        assert pytest.approx(stats["cv"][0.3][5],  rel=1e-6) == expected_cv

    def test_single_realization_var_is_zero(self):
        """
        With only one data point, ddof=1 would give division-by-zero in numpy.
        The code guards against this: len(vals) == 1 → var = 0.0.
        """
        stats = compute_F(_raw(0.1, 20, [0.7]))
        assert stats["var"][0.1][20] == 0.0

    def test_multiple_p_and_L_indexed_correctly(self):
        """
        Output must be nested result["F"][p][L].
        Mixing p and L values checks that keys don't collapse.
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

    def test_cv_is_sqrt_var_over_abs_F(self):
        """
        CV = sqrt(var) / |F|.  Verifies both numerator and denominator
        are combined correctly.
        """
        d = [0.1, 0.3]
        stats = compute_F(_raw(0.5, 5, d))
        expected_cv = np.sqrt(np.var(d, ddof=1)) / abs(1 - np.mean(d))
        assert pytest.approx(stats["cv"][0.5][5], rel=1e-6) == expected_cv

    def test_negative_mean_d_gives_F_greater_than_one(self):
        """
        If d is negative on average (unperturbed more scrambled than perturbed),
        F = 1 - mean(d) > 1.  compute_F must not clamp or reject this.
        """
        d = [-0.3, -0.3]
        stats = compute_F(_raw(0.5, 5, d))
        assert pytest.approx(stats["F"][0.5][5]) == 1.3

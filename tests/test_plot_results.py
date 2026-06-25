"""
tests/test_plot_results.py

Tests for plot_results.

plot_results saves two PNG files (C=1-F and CV), each with three
subplots.  We verify the files are created without inspecting pixels.
"""

import pathlib
import numpy as np
import pytest
from tests.helpers import plot_results, compute_F


def _make_all_stats(N_values, p_values, L_values):
    """
    Build a minimal all_stats structure with constant d=0.2 values
    so F, var, cv are all well-defined non-zero numbers.
    """
    stats = {}
    for N in N_values:
        raw = {p: {L: np.array([0.2, 0.3]) for L in L_values}
               for p in p_values}
        stats[N] = compute_F(raw)
    return stats


class TestPlotResults:

    def test_creates_both_output_files(self, tmp_path):
        """
        After calling plot_results both figure files must exist on disk.
        Tests that matplotlib savefig is reached for each figure.
        """
        N_values = [4, 6]
        p_values = [0.0, 0.5]
        L_values = [5, 10]
        all_stats = _make_all_stats(N_values, p_values, L_values)

        path_C  = str(tmp_path / "C.png")
        path_cv = str(tmp_path / "cv.png")

        plot_results(all_stats, L_values, p_values, N_values,
                     figure_path_C=path_C, figure_path_cv=path_cv)

        assert pathlib.Path(path_C).exists(),  "C figure file must be created"
        assert pathlib.Path(path_cv).exists(), "CV figure file must be created"

    def test_single_p_single_N_does_not_crash(self, tmp_path):
        """
        Minimal input (one p, one N, one L) must not raise.
        Guards against index-out-of-range in the subplot loops.
        """
        all_stats = _make_all_stats([4], [0.5], [5])
        path_C  = str(tmp_path / "C.png")
        path_cv = str(tmp_path / "cv.png")

        plot_results(all_stats, [5], [0.5], [4],
                     figure_path_C=path_C, figure_path_cv=path_cv)

        assert pathlib.Path(path_C).exists()
        assert pathlib.Path(path_cv).exists()

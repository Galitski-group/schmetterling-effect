"""
tests/test_plot_otoc.py

Tests for plot_otoc.

plot_otoc saves three PNG files:
  otoc_cv.png  — CV vs T at topmost qubit (one line per p)
  otoc_2d.png  — C vs T at topmost qubit (one line per p)
  otoc_3d.png  — 3D surface (x, T) → C at p = p_values[0]

We build a minimal raw_xtp structure and assert the files are created.
"""

import os
import pathlib
import numpy as np
import pytest
from tests.helpers import plot_otoc


def _make_raw_xtp(p_values, probe_sites, L_values):
    """Minimal raw_xtp with constant d=0.2 values."""
    return {
        p: {
            ps: {L: np.array([0.1, 0.2, 0.3]) for L in L_values}
            for ps in probe_sites
        }
        for p in p_values
    }


class TestPlotOtoc:

    def test_creates_all_three_figure_files(self, tmp_path):
        """
        All three OTOC figures must be written to disk after calling plot_otoc.
        """
        p_values    = [0.0, 0.5]
        L_values    = [5, 10]
        probe_sites = [0, 1, 3]   # pert_site = 2
        pert_site   = 2

        raw_xtp = _make_raw_xtp(p_values, probe_sites, L_values)

        path_cv = str(tmp_path / "cv.png")
        path_2d = str(tmp_path / "2d.png")
        path_3d = str(tmp_path / "3d.png")

        plot_otoc(
            raw_xtp=raw_xtp, pert_site=pert_site,
            L_values=L_values, probe_sites=probe_sites,
            p_values=p_values, topmost_qubit=3,
            figure_path_cv=path_cv,
            figure_path_2d=path_2d,
            figure_path_3d=path_3d,
        )

        assert pathlib.Path(path_cv).exists(), "CV figure must be created"
        assert pathlib.Path(path_2d).exists(), "2D OTOC figure must be created"
        assert pathlib.Path(path_3d).exists(), "3D OTOC figure must be created"

    def test_topmost_qubit_defaults_to_max_probe_site(self, tmp_path):
        """
        When topmost_qubit is omitted it must default to max(probe_sites).
        The function must not raise KeyError accessing that site's data.
        """
        p_values    = [0.3]
        L_values    = [5]
        probe_sites = [0, 1]   # max = 1; pert_site = 2
        raw_xtp = _make_raw_xtp(p_values, probe_sites, L_values)

        plot_otoc(
            raw_xtp=raw_xtp, pert_site=2,
            L_values=L_values, probe_sites=probe_sites,
            p_values=p_values,
            # topmost_qubit not passed → defaults to max(probe_sites) = 1
            figure_path_cv=str(tmp_path / "cv.png"),
            figure_path_2d=str(tmp_path / "2d.png"),
            figure_path_3d=str(tmp_path / "3d.png"),
        )

    def test_single_p_single_L_does_not_crash(self, tmp_path):
        """
        Minimal input (one p, one L, two probe sites) must not raise.
        Guards against index errors in the single-element subplot loops.
        """
        raw_xtp = _make_raw_xtp([0.5], [0, 1], [5])
        plot_otoc(
            raw_xtp=raw_xtp, pert_site=2,
            L_values=[5], probe_sites=[0, 1],
            p_values=[0.5], topmost_qubit=1,
            figure_path_cv=str(tmp_path / "cv.png"),
            figure_path_2d=str(tmp_path / "2d.png"),
            figure_path_3d=str(tmp_path / "3d.png"),
        )

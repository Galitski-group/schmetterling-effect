"""
tests/test_hamming.py

Tests for calculate_hamming_distance_pdf.

The function takes a dict whose keys are bitstrings (tuples of 0/1)
and values are shot counts. It returns (xs, pdf) where xs is the
sorted list of Hamming distances from |00...0> and pdf is the
normalised probability mass at each distance.
"""

import pytest
from tests.helpers import calculate_hamming_distance_pdf


class TestCalculateHammingDistancePDF:

    def test_all_ground_state(self):
        """
        Every shot lands on |00...0> → only distance 0 exists.
        xs must be [0] and pdf must be [1.0].
        This is the trivial perfect-echo baseline.
        """
        data = {(0, 0, 0): 100}
        xs, pdf = calculate_hamming_distance_pdf(data)
        assert xs == [0]
        assert pdf == [1.0]

    def test_single_bit_flip(self):
        """
        Three counts at distance-0, one at distance-1 (single flip).
        xs = [0, 1]; pdf reflects the 3:1 split.
        Verifies the Hamming metric distinguishes single-qubit errors.
        """
        data = {(0, 0): 3, (1, 0): 1}
        xs, pdf = calculate_hamming_distance_pdf(data)
        assert xs == [0, 1]
        assert pytest.approx(pdf[0]) == 0.75
        assert pytest.approx(pdf[1]) == 0.25

    def test_multiple_distances(self):
        """
        Bitstrings spread across distances 0, 1, 2, 3.
        Verifies multi-bucket histogram construction and correct weighting.
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
        Fails if the normalisation divides by the wrong total.
        """
        data = {(0,): 7, (1,): 3}
        _, pdf = calculate_hamming_distance_pdf(data)
        assert pytest.approx(sum(pdf)) == 1.0

    def test_xs_are_sorted(self):
        """
        xs must be in ascending order regardless of dict insertion order.
        Callers rely on xs[i] < xs[i+1] for plotting.
        """
        data = {(1, 1): 2, (0, 0): 5, (1, 0): 3}
        xs, _ = calculate_hamming_distance_pdf(data)
        assert xs == sorted(xs)

    def test_uniform_counts_give_uniform_pdf(self):
        """
        Equal counts at every distance → uniform pdf.
        """
        data = {
            (0, 0, 0, 0): 10,
            (1, 0, 0, 0): 10,
            (1, 1, 0, 0): 10,
            (1, 1, 1, 0): 10,
        }
        _, pdf = calculate_hamming_distance_pdf(data)
        assert all(pytest.approx(v) == 0.25 for v in pdf)

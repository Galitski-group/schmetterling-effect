"""
tests/conftest.py

Shared pytest fixtures available to every test file automatically.
No imports needed in individual test files to use these.
"""
import pytest


@pytest.fixture
def small_circuit():
    """
    A minimal W=4, L=2, p=0 circuit with no barriers.
    Used by multiple test files as a cheap circuit to pass to functions
    that only care about having a valid Circuit object.
    """
    from tests.helpers import generate_time_reversal_breaking_random_brick_wall as gen
    return gen(L=2, W=4, seed=0, p=0.0, unperturbed=True, add_barrier=False)

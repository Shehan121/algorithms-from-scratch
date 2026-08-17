"""Search correctness, with emphasis on duplicates and boundaries."""

import random

import pytest

from algokit.searching import (
    binary_search,
    binary_search_leftmost,
    binary_search_recursive,
    exponential_search,
    interpolation_search,
    linear_search,
)

ALL = [linear_search, binary_search, binary_search_recursive, interpolation_search, exponential_search]


@pytest.mark.parametrize("fn", ALL)
def test_finds_every_element(fn):
    data = [1, 3, 5, 7, 9, 11, 13]
    for i, value in enumerate(data):
        assert fn(data, value) == i


@pytest.mark.parametrize("fn", ALL)
def test_missing_returns_minus_one(fn):
    data = [1, 3, 5, 7, 9]
    for missing in (0, 2, 4, 10, 100, -7):
        assert fn(data, missing) == -1


@pytest.mark.parametrize("fn", ALL)
def test_empty_and_single(fn):
    assert fn([], 1) == -1
    assert fn([5], 5) == 0
    assert fn([5], 4) == -1


@pytest.mark.parametrize("fn", ALL)
def test_boundaries(fn):
    """First and last positions are where off-by-one errors surface."""
    data = list(range(0, 200, 2))
    assert fn(data, data[0]) == 0
    assert fn(data, data[-1]) == len(data) - 1


def test_leftmost_finds_first_duplicate():
    """Plain binary search may land on any duplicate; leftmost must not."""
    data = [1, 2, 2, 2, 2, 3]
    assert binary_search_leftmost(data, 2) == 1
    assert binary_search(data, 2) in {1, 2, 3, 4}  # any match is acceptable


def test_leftmost_on_all_equal():
    assert binary_search_leftmost([7] * 10, 7) == 0
    assert binary_search_leftmost([7] * 10, 8) == -1


@pytest.mark.parametrize("fn", ALL)
def test_against_reference_random(fn):
    rng = random.Random(3)
    for _ in range(40):
        data = sorted(rng.randrange(100) for _ in range(rng.randrange(1, 30)))
        target = rng.randrange(100)
        found = fn(data, target)
        if target in data:
            assert found != -1 and data[found] == target
        else:
            assert found == -1


def test_interpolation_handles_constant_range():
    """A constant array makes the interpolation denominator zero."""
    assert interpolation_search([4] * 8, 4) == 0
    assert interpolation_search([4] * 8, 5) == -1

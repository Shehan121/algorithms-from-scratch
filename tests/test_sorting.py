"""Every sort must agree with sorted() on every input shape."""

import random

import pytest
from conftest import EDGE_CASES

from algokit.sorting import COMPARISON_SORTS, SORTS, insertion_sort, merge_sort


@pytest.mark.parametrize("name,fn", SORTS.items())
@pytest.mark.parametrize("data", EDGE_CASES)
def test_matches_reference(name, fn, data):
    """Correctness is defined as agreeing with the built-in sort."""
    assert fn(data) == sorted(data), f"{name} failed on {data}"


@pytest.mark.parametrize("name,fn", SORTS.items())
def test_input_not_mutated(name, fn):
    """Sorts return a new list; the caller's data is untouched.

    Benchmarks reuse one generated input across algorithms, so a sort that
    mutated in place would silently hand the next algorithm pre-sorted data.
    """
    data = [5, 3, 9, 1, 7]
    original = list(data)
    fn(data)
    assert data == original, f"{name} mutated its input"


@pytest.mark.parametrize("name,fn", SORTS.items())
def test_random_inputs(name, fn):
    rng = random.Random(7)
    for _ in range(25):
        data = [rng.randrange(50) for _ in range(rng.randrange(40))]
        assert fn(data) == sorted(data)


@pytest.mark.parametrize("name,fn", COMPARISON_SORTS.items())
def test_handles_negative_and_duplicate_values(name, fn):
    data = [0, -5, 3, -5, 12, 0, -100]
    assert fn(data) == sorted(data)


def test_counting_and_radix_reject_unsuitable_input():
    """The non-comparison sorts have preconditions; they should say so."""
    from algokit.sorting import radix_sort

    with pytest.raises(ValueError):
        radix_sort([3, -1, 2])


@pytest.mark.parametrize("fn", [insertion_sort, merge_sort])
def test_stability(fn):
    """Stable sorts preserve the input order of equal keys.

    Records are (key, tag); after sorting by key alone the tags for a repeated
    key must stay in their original sequence.
    """
    records = [(1, "a"), (0, "b"), (1, "c"), (0, "d"), (1, "e")]

    class ByKey:
        def __init__(self, item):
            self.item = item

        def __lt__(self, other):
            return self.item[0] < other.item[0]

        def __eq__(self, other):
            return self.item[0] == other.item[0]

    result = [w.item for w in fn([ByKey(r) for r in records])]
    assert result == [(0, "b"), (0, "d"), (1, "a"), (1, "c"), (1, "e")]


def test_selection_sort_is_not_stable():
    """Documenting a real limitation rather than glossing over it."""
    from algokit.sorting import selection_sort

    class ByKey:
        def __init__(self, item):
            self.item = item

        def __lt__(self, other):
            return self.item[0] < other.item[0]

        def __eq__(self, other):
            return self.item[0] == other.item[0]

    records = [(1, "a"), (1, "b"), (0, "c")]
    result = [w.item for w in selection_sort([ByKey(r) for r in records])]
    # (0,'c') swaps with (1,'a'), pushing 'a' after 'b'.
    assert result == [(0, "c"), (1, "b"), (1, "a")]

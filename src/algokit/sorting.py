"""Sorting algorithms, implemented from scratch.

Every function takes a sequence and returns a **new** sorted list, leaving the
input untouched. That costs one copy, but it makes the functions safe to call
repeatedly on the same benchmark input, which matters when comparing timings.

No comparison counters appear in this file — see :mod:`algokit.instrument` for
why. The comparisons are written as plain ``a[i] < a[j]``, and the measurement
happens in the values.

Complexity summary, verified empirically in ``reports/``:

===============  ==========  ==========  ==========  ======  ======
algorithm        best        average     worst       stable  extra
===============  ==========  ==========  ==========  ======  ======
bubble           O(n)        O(n^2)      O(n^2)      yes     O(1)
insertion        O(n)        O(n^2)      O(n^2)      yes     O(1)
selection        O(n^2)      O(n^2)      O(n^2)      no      O(1)
merge            O(n log n)  O(n log n)  O(n log n)  yes     O(n)
quick (random)   O(n log n)  O(n log n)  O(n^2)      no      O(log n)
quick (lomuto)   O(n log n)  O(n log n)  O(n^2)      no      O(log n)
heap             O(n log n)  O(n log n)  O(n log n)  no      O(1)
counting         O(n + k)    O(n + k)    O(n + k)    yes     O(n + k)
radix (LSD)      O(d(n + k)) O(d(n + k)) O(d(n + k)) yes     O(n + k)
===============  ==========  ==========  ==========  ======  ======
"""

from __future__ import annotations

import random
from typing import Any, Sequence

__all__ = [
    "bubble_sort",
    "insertion_sort",
    "selection_sort",
    "merge_sort",
    "quick_sort",
    "quick_sort_lomuto",
    "heap_sort",
    "counting_sort",
    "radix_sort",
    "SORTS",
    "COMPARISON_SORTS",
]


# --------------------------------------------------------------------------
# Quadratic sorts
# --------------------------------------------------------------------------


def bubble_sort(seq: Sequence[Any]) -> list[Any]:
    """Repeatedly swap adjacent out-of-order pairs until a pass makes none.

    The ``swapped`` flag is what gives bubble sort its O(n) best case: on an
    already-sorted input the first pass makes no swaps and the function returns
    after a single scan. Without that early exit it would be O(n^2)
    unconditionally, which is how it is often written — and why bubble sort is
    sometimes wrongly described as having no good case.

    After pass ``i`` the largest ``i`` elements are in final position, so the
    inner loop shrinks: hence ``n - 1 - i``.
    """
    a = list(seq)
    n = len(a)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if a[j + 1] < a[j]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def insertion_sort(seq: Sequence[Any]) -> list[Any]:
    """Grow a sorted prefix, shifting each new element back into place.

    The workhorse for small inputs. The inner ``while`` stops as soon as it
    finds a smaller element, so on nearly-sorted data it barely moves anything —
    the number of shifts equals the number of inversions in the input. That is
    why real library sorts (Timsort, introsort) fall back to insertion sort
    below a size threshold instead of recursing further.
    """
    a = list(seq)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and key < a[j]:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def selection_sort(seq: Sequence[Any]) -> list[Any]:
    """Repeatedly select the minimum of the unsorted suffix.

    Unlike bubble and insertion sort it has **no** best case: finding each
    minimum always scans the whole remaining suffix, so it performs exactly
    n(n-1)/2 comparisons on every input regardless of order. It compensates with
    at most n-1 swaps, which used to matter when writes were far more expensive
    than reads.
    """
    a = list(seq)
    n = len(a)
    for i in range(n - 1):
        smallest = i
        for j in range(i + 1, n):
            if a[j] < a[smallest]:
                smallest = j
        if smallest != i:
            a[i], a[smallest] = a[smallest], a[i]
    return a


# --------------------------------------------------------------------------
# Divide and conquer
# --------------------------------------------------------------------------


def merge_sort(seq: Sequence[Any]) -> list[Any]:
    """Split in half, sort each half, merge.

    Guaranteed O(n log n) — the recursion depth is log n regardless of input,
    because the split is positional rather than value-dependent. That is the
    structural difference from quicksort, which splits by value and can
    therefore be unbalanced.

    Stable, because ``left[i] <= right[j]`` takes from the left half on ties.
    Writing that comparison as ``<`` instead would silently break stability.
    """
    a = list(seq)
    if len(a) <= 1:
        return a

    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return _merge(left, right)


def _merge(left: list[Any], right: list[Any]) -> list[Any]:
    merged: list[Any] = []
    i = j = 0
    while i < len(left) and j < len(right):
        # `not (right[j] < left[i])` is `left[i] <= right[j]` using only `<`,
        # which keeps the sort stable while needing just one comparison.
        if not (right[j] < left[i]):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(seq: Sequence[Any], rng: random.Random | None = None) -> list[Any]:
    """Quicksort with a **randomly chosen** pivot.

    Randomising the pivot is what makes the O(n^2) worst case improbable rather
    than merely unlucky: no fixed input can reliably trigger it, because the
    adversary cannot predict the pivot. Compare ``quick_sort_lomuto``, which
    takes the last element and degrades to quadratic on sorted input — a
    difference the benchmarks in this repo measure directly.

    This is the three-way partition form, so duplicate keys land in ``mid`` and
    are never recursed into. On input with few distinct values that turns the
    usual O(n^2) duplicate-key blowup into near-linear behaviour.
    """
    a = list(seq)
    if len(a) <= 1:
        return a
    rng = rng or random.Random(0)

    pivot = a[rng.randrange(len(a))]
    smaller = [x for x in a if x < pivot]
    equal = [x for x in a if not (x < pivot) and not (pivot < x)]
    larger = [x for x in a if pivot < x]
    return quick_sort(smaller, rng) + equal + quick_sort(larger, rng)


def quick_sort_lomuto(seq: Sequence[Any]) -> list[Any]:
    """In-place quicksort, Lomuto partition, **last element as pivot**.

    Included deliberately as the naive version. Choosing a fixed position for
    the pivot means an already-sorted input produces maximally unbalanced
    partitions of size n-1 and 0 at every level, giving O(n^2) time and O(n)
    recursion depth. The benchmarks show it hitting Python's recursion limit on
    sorted input at a size the randomised version handles without noticing.
    """
    a = list(seq)
    _lomuto_sort(a, 0, len(a) - 1)
    return a


def _lomuto_sort(a: list[Any], lo: int, hi: int) -> None:
    if lo >= hi:
        return
    p = _lomuto_partition(a, lo, hi)
    _lomuto_sort(a, lo, p - 1)
    _lomuto_sort(a, p + 1, hi)


def _lomuto_partition(a: list[Any], lo: int, hi: int) -> int:
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i


def heap_sort(seq: Sequence[Any]) -> list[Any]:
    """Build a max-heap in place, then repeatedly move the root to the back.

    A max-heap is used rather than a min-heap precisely so the sort can be done
    in place ascending: the largest element sits at index 0 and is swapped to
    the end, which is exactly where it belongs.

    Heapify runs from ``n // 2 - 1`` down to 0 because every index above
    ``n // 2 - 1`` is a leaf and already a valid heap. Building bottom-up this
    way is O(n), not O(n log n) — most nodes are near the bottom and sift down
    only a short distance.
    """
    a = list(seq)
    n = len(a)

    for start in range(n // 2 - 1, -1, -1):
        _sift_down(a, start, n)

    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        _sift_down(a, 0, end)
    return a


def _sift_down(a: list[Any], root: int, size: int) -> None:
    while True:
        child = 2 * root + 1
        if child >= size:
            return
        # Pick the larger child, guarding against a missing right child.
        if child + 1 < size and a[child] < a[child + 1]:
            child += 1
        if not (a[root] < a[child]):
            return
        a[root], a[child] = a[child], a[root]
        root = child


# --------------------------------------------------------------------------
# Non-comparison sorts
# --------------------------------------------------------------------------


def counting_sort(seq: Sequence[Any]) -> list[Any]:
    """Sort integers by tallying occurrences. O(n + k) for key range k.

    Beats the O(n log n) comparison lower bound by never comparing elements —
    it indexes them. The catch is the ``k`` term: sorting three integers spread
    across a range of a million allocates a million counters, so this is only
    sensible when k is comparable to n.

    Written as a stable sort (walking the input backwards while placing from
    the cumulative counts) so it can serve as the inner pass of ``radix_sort``,
    which requires stability to work at all.
    """
    a = list(seq)
    if not a:
        return a

    lo = int(min(a))
    hi = int(max(a))
    counts = [0] * (hi - lo + 1)
    for value in a:
        counts[int(value) - lo] += 1

    # Cumulative counts give each key its end position in the output.
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]

    out: list[Any] = [None] * len(a)
    for value in reversed(a):
        counts[int(value) - lo] -= 1
        out[counts[int(value) - lo]] = value
    return out


def radix_sort(seq: Sequence[Any], base: int = 10) -> list[Any]:
    """Least-significant-digit radix sort for non-negative integers.

    Runs one stable counting pass per digit, so d passes over n elements gives
    O(d(n + k)). It works *only* because each pass is stable: the ordering
    established by lower digits must survive the pass over higher digits. Swap
    in an unstable inner sort and the result is simply wrong, which is the most
    instructive way to discover what stability is for.
    """
    a = list(seq)
    if not a:
        return a
    if any(int(v) < 0 for v in a):
        raise ValueError("radix_sort requires non-negative integers")

    largest = int(max(a))
    place = 1
    while place <= largest:
        buckets: list[list[Any]] = [[] for _ in range(base)]
        for value in a:
            digit = (int(value) // place) % base
            buckets[digit].append(value)
        a = [value for bucket in buckets for value in bucket]
        place *= base
    return a


# --------------------------------------------------------------------------

#: Every sort, for the test suite and benchmarks.
SORTS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "selection": selection_sort,
    "merge": merge_sort,
    "quick (random pivot)": quick_sort,
    "quick (last pivot)": quick_sort_lomuto,
    "heap": heap_sort,
    "counting": counting_sort,
    "radix": radix_sort,
}

#: The subset bound by the O(n log n) comparison lower bound.
COMPARISON_SORTS = {
    name: fn
    for name, fn in SORTS.items()
    if name not in {"counting", "radix"}
}

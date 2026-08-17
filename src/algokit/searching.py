"""Search algorithms, and the boundary conditions that make them subtle.

Binary search is famously easy to get almost right. The three classic mistakes
are all avoided explicitly below and explained where they occur:

1. ``(lo + hi) // 2`` overflowing — harmless in Python, fatal in C/Java
2. an off-by-one in the loop bound causing an infinite loop or a missed element
3. returning *any* match rather than the first, when duplicates exist
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "linear_search",
    "binary_search",
    "binary_search_recursive",
    "binary_search_leftmost",
    "interpolation_search",
    "exponential_search",
    "SEARCHES",
]


def linear_search(seq: Sequence[Any], target: Any) -> int:
    """Scan from the start. O(n), and the only option on unsorted data.

    Worth keeping in the comparison: for small n it beats binary search in
    practice, because it is branch-predictable and cache-friendly while binary
    search jumps around memory.
    """
    for i in range(len(seq)):
        if seq[i] == target:
            return i
    return -1


def binary_search(seq: Sequence[Any], target: Any) -> int:
    """Iterative binary search on a sorted sequence. O(log n).

    ``lo + (hi - lo) // 2`` rather than ``(lo + hi) // 2``: identical in Python,
    where integers are arbitrary precision, but the latter is the overflow bug
    that sat in the JDK's own binary search for nine years. Written the safe way
    here because the point is to learn the correct form.

    The loop is ``while lo <= hi`` with an inclusive ``hi``. The pairing matters:
    inclusive bounds need ``<=``, and ``hi = mid - 1`` / ``lo = mid + 1`` always
    shrink the interval, so the loop cannot spin.
    """
    lo, hi = 0, len(seq) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if seq[mid] == target:
            return mid
        if seq[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def binary_search_recursive(seq: Sequence[Any], target: Any, lo: int = 0, hi: int | None = None) -> int:
    """The same algorithm expressed recursively.

    Included to make the space difference concrete: this costs O(log n) stack
    frames where the iterative form costs O(1). The recursion is tail-recursive,
    but Python does not eliminate tail calls, so the frames are really allocated.
    """
    if hi is None:
        hi = len(seq) - 1
    if lo > hi:
        return -1

    mid = lo + (hi - lo) // 2
    if seq[mid] == target:
        return mid
    if seq[mid] < target:
        return binary_search_recursive(seq, target, mid + 1, hi)
    return binary_search_recursive(seq, target, lo, mid - 1)


def binary_search_leftmost(seq: Sequence[Any], target: Any) -> int:
    """Find the **first** occurrence of ``target``. O(log n).

    Plain binary search returns whichever duplicate it happens to land on, which
    is fine for membership and wrong for "where does this run begin?". The fix is
    to stop testing for equality inside the loop and instead converge on a
    boundary: keep moving ``hi`` down whenever ``seq[mid] >= target``, and the
    invariant leaves ``lo`` at the leftmost candidate.

    Note the loop is ``while lo < hi`` with an *exclusive* upper bound here —
    a different bound convention from :func:`binary_search`, and mixing the two
    is how the off-by-one bugs happen.
    """
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if seq[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(seq) and seq[lo] == target else -1


def interpolation_search(seq: Sequence[Any], target: Any) -> int:
    """Guess the position by linear interpolation rather than halving.

    On **uniformly distributed** sorted numeric data this is O(log log n), which
    is dramatically better than binary search. On skewed data it degrades to
    O(n), which is dramatically worse. It is the clearest example in this
    repository of an algorithm whose complexity depends on the *distribution* of
    the input rather than just its size.

    The guard ``seq[hi] == seq[lo]`` prevents a division by zero when the range
    is constant.
    """
    lo, hi = 0, len(seq) - 1
    while lo <= hi and seq[lo] <= target <= seq[hi]:
        if seq[hi] == seq[lo]:
            return lo if seq[lo] == target else -1

        span = seq[hi] - seq[lo]
        offset = int((hi - lo) * (target - seq[lo]) / span)
        guess = lo + offset
        guess = max(lo, min(hi, guess))  # clamp against a bad estimate

        if seq[guess] == target:
            return guess
        if seq[guess] < target:
            lo = guess + 1
        else:
            hi = guess - 1
    return -1


def exponential_search(seq: Sequence[Any], target: Any) -> int:
    """Double a bound until it passes the target, then binary search inside it.

    Useful when the target is expected near the front, or when the sequence has
    no cheap length — the first phase never looks beyond index 2i. Cost is
    O(log i) for a match at index i, independent of total length.
    """
    if not seq:
        return -1
    if seq[0] == target:
        return 0

    bound = 1
    while bound < len(seq) and seq[bound] < target:
        bound *= 2

    lo = bound // 2
    hi = min(bound, len(seq) - 1)
    window = seq[lo : hi + 1]
    found = binary_search(window, target)
    return lo + found if found != -1 else -1


SEARCHES = {
    "linear": linear_search,
    "binary (iterative)": binary_search,
    "binary (recursive)": binary_search_recursive,
    "binary (leftmost)": binary_search_leftmost,
    "interpolation": interpolation_search,
    "exponential": exponential_search,
}

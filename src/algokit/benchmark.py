"""Timing and operation-counting harness.

Two kinds of measurement, used for different purposes:

``time_call``   wall-clock time. What users feel, but noisy — affected by CPU
                frequency scaling, garbage collection and cache state.
``count_calls`` primitive operations via :mod:`algokit.instrument`. Perfectly
                reproducible and machine-independent, which makes it the right
                input to complexity fitting.

Timing takes the **minimum** of several repeats rather than the mean. Noise on a
loaded machine is one-sided: interference can only make a run slower, never
faster, so the minimum is the closest estimate of the true cost. Averaging just
folds in whatever else the machine was doing.
"""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from algokit.instrument import Ops, instrument


@dataclass
class Timing:
    name: str
    n: int
    seconds: float
    repeats: int


@dataclass
class Counting:
    name: str
    n: int
    ops: Ops
    field_totals: dict[str, int] = field(default_factory=dict)


def time_call(func: Callable[..., Any], *args: Any, repeats: int = 3, **kwargs: Any) -> float:
    """Fastest of ``repeats`` runs, in seconds.

    Garbage collection is disabled during measurement so a collection triggered
    by one algorithm is not charged to another. ``perf_counter`` is used rather
    than ``time`` because it is monotonic and has the highest resolution
    available.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
        return best
    finally:
        if was_enabled:
            gc.enable()


def count_calls(func: Callable[..., Any], values: Iterable[Any], *args: Any, **kwargs: Any) -> Ops:
    """Run ``func`` over instrumented ``values`` and return the operation tally."""
    tracked, ops = instrument(values)
    func(tracked, *args, **kwargs)
    return ops


# --------------------------------------------------------------------------
# Input generators
# --------------------------------------------------------------------------


def random_list(n: int, seed: int = 0, spread: int | None = None) -> list[int]:
    """Uniform random integers. ``spread`` bounds the key range."""
    rng = random.Random(seed)
    hi = spread if spread is not None else max(n * 10, 10)
    return [rng.randrange(hi) for _ in range(n)]


def sorted_list(n: int) -> list[int]:
    """Ascending — the best case for insertion sort, worst for a fixed-pivot quicksort."""
    return list(range(n))


def reversed_list(n: int) -> list[int]:
    """Descending — the worst case for insertion and bubble sort."""
    return list(range(n, 0, -1))


def nearly_sorted_list(n: int, window: int = 8, fraction: float = 0.1, seed: int = 0) -> list[int]:
    """Sorted, then perturbed by *local* swaps within a bounded window.

    The most realistic input shape here, and the one adaptive sorts are designed
    for — a log file with a few late arrivals, a mostly-ordered index.

    The displacement is deliberately **bounded** rather than global. An earlier
    version of this generator swapped ``n // 100`` pairs of *arbitrary* indices,
    which looks reasonable and is useless for complexity measurement: both the
    number of swaps and the distance each element travels grow with n, so
    inversions grow quadratically (7 at n=64, but 24,838 at n=2048). The input
    family got harder faster than it got bigger, and bubble sort duly measured a
    doubling ratio of 7.17 — above the 4.0 that quadratic growth allows, which is
    the giveaway that the *input*, not the algorithm, was the variable.

    With each element displaced by at most ``window`` positions, inversions grow
    as O(n * window), i.e. linearly, so disorder per element is constant and the
    sweep varies only n.
    """
    rng = random.Random(seed)
    a = list(range(n))
    for _ in range(int(n * fraction)):
        i = rng.randrange(n)
        j = min(n - 1, max(0, i + rng.randint(-window, window)))
        a[i], a[j] = a[j], a[i]
    return a


def few_unique_list(n: int, distinct: int = 5, seed: int = 0) -> list[int]:
    """Many duplicates — the case that breaks two-way partition quicksort."""
    rng = random.Random(seed)
    return [rng.randrange(distinct) for _ in range(n)]


#: Named input shapes, so benchmarks can report per-case behaviour.
INPUT_SHAPES: dict[str, Callable[[int], list[int]]] = {
    "random": random_list,
    "sorted": sorted_list,
    "reversed": reversed_list,
    "nearly sorted": nearly_sorted_list,
    "few unique": few_unique_list,
}


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def sweep_time(
    func: Callable[[Sequence[Any]], Any],
    sizes: Sequence[int],
    generator: Callable[[int], list[Any]] = random_list,
    repeats: int = 3,
    name: str | None = None,
    limit_seconds: float | None = None,
) -> list[Timing]:
    """Time ``func`` across ``sizes``.

    ``limit_seconds`` stops the sweep once a single measurement exceeds the
    budget — without it, adding bubble sort at n = 100,000 to a benchmark makes
    the whole suite unrunnable.
    """
    label = name or getattr(func, "__name__", str(func))
    out: list[Timing] = []

    for n in sizes:
        data = generator(n)
        seconds = time_call(func, data, repeats=repeats)
        out.append(Timing(name=label, n=n, seconds=seconds, repeats=repeats))
        if limit_seconds is not None and seconds > limit_seconds:
            break
    return out


def sweep_ops(
    func: Callable[[Sequence[Any]], Any],
    sizes: Sequence[int],
    generator: Callable[[int], list[Any]] = random_list,
    name: str | None = None,
) -> list[Counting]:
    """Count operations for ``func`` across ``sizes``."""
    label = name or getattr(func, "__name__", str(func))
    out: list[Counting] = []

    for n in sizes:
        ops = count_calls(func, generator(n))
        out.append(Counting(name=label, n=n, ops=ops, field_totals=ops.as_dict()))
    return out

"""Counting operations without touching the algorithms.

The obvious way to count comparisons is to thread a counter through every
function::

    def insertion_sort(a, counter):
        counter.comparisons += 1          # noise in every branch
        if a[j] > key: ...

That buries the algorithm in bookkeeping, and the instrumented version is no
longer the code you would actually write. Worse, the counter argument slows down
the timing benchmarks too, so you cannot use the same implementation for both.

Instead the instrumentation lives in the *data*:

``Probe``        wraps a value and counts every comparison it takes part in
``TrackedList``  a list that counts reads and writes

Because both cooperate with the normal Python protocols (``__lt__``,
``__getitem__``, ``__setitem__``), the algorithms are written with plain
``a[i] < a[j]`` and need no knowledge that they are being measured. The same
unmodified function is used for timing (on plain ints) and for op counting
(on probes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Iterable, Iterator


@dataclass
class Ops:
    """A tally of the primitive operations an algorithm performed."""

    comparisons: int = 0
    reads: int = 0
    writes: int = 0
    calls: int = 0

    def reset(self) -> None:
        self.comparisons = self.reads = self.writes = self.calls = 0

    @property
    def total(self) -> int:
        return self.comparisons + self.reads + self.writes

    def as_dict(self) -> dict[str, int]:
        return {
            "comparisons": self.comparisons,
            "reads": self.reads,
            "writes": self.writes,
            "calls": self.calls,
        }


@total_ordering
class Probe:
    """A value that reports every comparison to a shared :class:`Ops` tally.

    ``@total_ordering`` fills in ``>``, ``<=`` and ``>=`` from ``__lt__`` and
    ``__eq__``, so only two methods have to count. Note the consequence: a
    single ``a > b`` in an algorithm may register as one comparison here even
    though the derived operator calls ``__lt__`` underneath — the count is of
    *comparison operations performed by the algorithm*, which is the quantity
    textbook complexity analysis actually talks about.
    """

    __slots__ = ("value", "ops")

    def __init__(self, value: Any, ops: Ops) -> None:
        self.value = value
        self.ops = ops

    def _other(self, other: Any) -> Any:
        return other.value if isinstance(other, Probe) else other

    def __lt__(self, other: Any) -> bool:
        self.ops.comparisons += 1
        return self.value < self._other(other)

    def __eq__(self, other: Any) -> bool:
        self.ops.comparisons += 1
        return self.value == self._other(other)

    def __hash__(self) -> int:
        return hash(self.value)

    # Arithmetic is needed by the non-comparison sorts (counting, radix), which
    # index buckets by key value rather than comparing elements.
    def __index__(self) -> int:
        return int(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __sub__(self, other: Any) -> Any:
        return self.value - self._other(other)

    def __floordiv__(self, other: Any) -> Any:
        return self.value // self._other(other)

    def __mod__(self, other: Any) -> Any:
        return self.value % self._other(other)

    def __repr__(self) -> str:
        return f"Probe({self.value!r})"


class TrackedList(list):
    """A list that counts element reads and writes.

    Slice access is counted as a single read rather than one per element. That
    is a deliberate simplification: it keeps ``a[lo:mid]`` in merge sort from
    dominating the tally, and the interesting quantity for merge sort is its
    comparisons.
    """

    __slots__ = ("ops",)

    def __init__(self, iterable: Iterable[Any] = (), ops: Ops | None = None) -> None:
        super().__init__(iterable)
        self.ops = ops if ops is not None else Ops()

    def __getitem__(self, index: Any) -> Any:
        self.ops.reads += 1
        return super().__getitem__(index)

    def __setitem__(self, index: Any, value: Any) -> None:
        self.ops.writes += 1
        super().__setitem__(index, value)

    def append(self, value: Any) -> None:
        self.ops.writes += 1
        super().append(value)

    def __iter__(self) -> Iterator[Any]:
        # Counting one read per element keeps `for x in a` comparable to
        # indexed access.
        for i in range(len(self)):
            self.ops.reads += 1
            yield super().__getitem__(i)


@dataclass
class Measured:
    """The result of running an algorithm under instrumentation."""

    name: str
    n: int
    ops: Ops
    result: Any = field(repr=False, default=None)


def instrument(values: Iterable[Any]) -> tuple[TrackedList, Ops]:
    """Wrap ``values`` so that comparisons, reads and writes are all counted.

    Returns the tracked list and the shared tally. Every element shares one
    :class:`Ops`, so the counts are aggregate rather than per element.

    .. warning::

       **Which counts are meaningful depends on whether the algorithm works in
       place.** The sorts in :mod:`algokit.sorting` all begin with
       ``a = list(seq)`` so they never mutate the caller's data. That copy is a
       plain ``list``, so ``reads`` and ``writes`` only capture the copy itself
       (``reads == n``, ``writes == 0``) and say nothing about the sort's
       internal array traffic.

       ``comparisons`` remains correct in that case, because the :class:`Probe`
       elements are copied *by reference* into the new list and keep reporting to
       the same tally wherever they travel. Every complexity claim in this
       repository is therefore built on comparison counts, and the sort benchmarks
       deliberately record nothing else.

       ``reads`` and ``writes`` are meaningful for algorithms that operate on the
       tracked list directly rather than on a copy.
    """
    ops = Ops()
    tracked = TrackedList((Probe(v, ops) for v in values), ops=ops)
    # Building the list went through append/__init__ rather than the algorithm,
    # so discard whatever setup cost was recorded.
    ops.reset()
    return tracked, ops


def unwrap(values: Iterable[Any]) -> list[Any]:
    """Strip :class:`Probe` wrappers, for comparing against a reference result."""
    return [v.value if isinstance(v, Probe) else v for v in values]


def count_ops(func, values: Iterable[Any], *args: Any, **kwargs: Any) -> Measured:
    """Run ``func`` over ``values`` under instrumentation and return the tally."""
    tracked, ops = instrument(values)
    result = func(tracked, *args, **kwargs)
    return Measured(name=getattr(func, "__name__", str(func)), n=len(tracked), ops=ops, result=result)

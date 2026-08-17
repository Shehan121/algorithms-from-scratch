"""A binary min-heap in a flat array."""

from __future__ import annotations

from typing import Any, Iterable


class MinHeap:
    """Binary min-heap backed by a list. push/pop O(log n), peek O(1).

    The whole structure rests on one idea: a complete binary tree needs no
    pointers, because the shape is implied by the indices.

        parent(i) = (i - 1) // 2
        left(i)   = 2i + 1
        right(i)  = 2i + 2

    That is why a heap is faster in practice than a pointer-based tree of the
    same asymptotic cost — the nodes are contiguous, so traversal is
    cache-friendly and there is no allocation per element.

    The invariant is only ``parent <= children``. Siblings are unordered, which
    is exactly why a heap is cheaper to maintain than a sorted array: it is the
    minimum structure that makes "give me the smallest" O(1).
    """

    def __init__(self, values: Iterable[Any] = ()) -> None:
        self._items: list[Any] = list(values)
        if self._items:
            self._heapify()

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"MinHeap({self._items!r})"

    def _heapify(self) -> None:
        """Build a heap from an arbitrary list in O(n), not O(n log n).

        Pushing n elements one at a time costs O(n log n). Sifting down from the
        last internal node is O(n), because half the nodes are leaves that need
        no work at all and only the few nodes near the root can sift far. The
        proof is a sum that converges; the intuition is that cost is
        concentrated where nodes are rare.
        """
        for i in range(len(self._items) // 2 - 1, -1, -1):
            self._sift_down(i)

    def push(self, value: Any) -> None:
        """O(log n) — append at the bottom, then sift up."""
        self._items.append(value)
        self._sift_up(len(self._items) - 1)

    def pop(self) -> Any:
        """Remove and return the minimum. O(log n).

        The last element is moved to the root before sifting down, because
        removing the root leaves a hole that must be filled by a leaf to keep
        the tree complete. Filling it with a child instead would break the
        implicit-index layout.
        """
        if not self._items:
            raise IndexError("pop from empty heap")
        smallest = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return smallest

    def peek(self) -> Any:
        """O(1) — the reason to use a heap."""
        if not self._items:
            raise IndexError("peek at empty heap")
        return self._items[0]

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            if not self._items[i] < self._items[parent]:
                return
            self._items[i], self._items[parent] = self._items[parent], self._items[i]
            i = parent

    def _sift_down(self, i: int) -> None:
        size = len(self._items)
        while True:
            smallest = i
            for child in (2 * i + 1, 2 * i + 2):
                if child < size and self._items[child] < self._items[smallest]:
                    smallest = child
            if smallest == i:
                return
            self._items[i], self._items[smallest] = self._items[smallest], self._items[i]
            i = smallest

    def sorted_drain(self) -> list[Any]:
        """Pop everything, yielding sorted order — this *is* heapsort."""
        return [self.pop() for _ in range(len(self._items))]

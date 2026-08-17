"""Stack and queue, plus why a naive list-backed queue is a trap."""

from __future__ import annotations

from typing import Any


class Stack:
    """LIFO stack over a Python list. All operations O(1) amortised.

    A list is the right backing store here because every operation happens at
    the end, which is exactly where a dynamic array is cheap. ``push`` is
    *amortised* O(1) rather than strictly O(1): occasionally the list grows and
    copies, but the growth is geometric so the cost spreads out to a constant.
    """

    def __init__(self, values: Any = ()) -> None:
        self._items: list[Any] = list(values)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"Stack({self._items!r})"

    def push(self, value: Any) -> None:
        self._items.append(value)

    def pop(self) -> Any:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> Any:
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]


class Queue:
    """FIFO queue built from two stacks, giving amortised O(1) dequeue.

    The obvious implementation — a list with ``pop(0)`` — is **O(n) per
    dequeue**, because every remaining element shifts down one slot. Draining a
    queue of n items that way costs O(n^2), which is a genuinely common
    performance bug.

    The two-stack construction fixes it without a linked list. Pushes go to
    ``_in``; pops come from ``_out``. When ``_out`` empties, ``_in`` is poured
    into it, reversing the order once. Each element is moved between stacks
    exactly once in its lifetime, so although a single ``dequeue`` may cost
    O(n), the *amortised* cost is O(1). This is the clearest example here of
    amortised analysis being the only honest way to describe a cost.
    """

    def __init__(self, values: Any = ()) -> None:
        self._in: list[Any] = list(values)
        self._out: list[Any] = []

    def __len__(self) -> int:
        return len(self._in) + len(self._out)

    def __bool__(self) -> bool:
        return bool(self._in or self._out)

    def __repr__(self) -> str:
        return f"Queue({list(reversed(self._out)) + self._in!r})"

    def enqueue(self, value: Any) -> None:
        """Always O(1)."""
        self._in.append(value)

    def dequeue(self) -> Any:
        """Amortised O(1); O(n) on the transfer step."""
        if not self._out:
            if not self._in:
                raise IndexError("dequeue from empty queue")
            while self._in:
                self._out.append(self._in.pop())
        return self._out.pop()

    def peek(self) -> Any:
        if not self._out:
            if not self._in:
                raise IndexError("peek at empty queue")
            while self._in:
                self._out.append(self._in.pop())
        return self._out[-1]

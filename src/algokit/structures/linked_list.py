"""A singly linked list, and an honest account of when it is worth using."""

from __future__ import annotations

from typing import Any, Iterator, Optional


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value: Any, nxt: Optional["_Node"] = None) -> None:
        self.value = value
        self.next = nxt


class LinkedList:
    """Singly linked list with O(1) push_front and O(n) indexing.

    The trade-off against a dynamic array, stated plainly: insertion at the head
    is O(1) here and O(n) in an array, but random access is O(n) here and O(1)
    in an array. In practice arrays win far more often than the asymptotics
    suggest, because sequential array access is cache-friendly while every
    ``node.next`` is a potential cache miss.

    A ``_size`` counter is maintained so ``len()`` is O(1). Without it, length
    would require a full traversal — a small design decision that turns a
    constant-time query into a linear one.
    """

    def __init__(self, values: Any = ()) -> None:
        self._head: Optional[_Node] = None
        self._size = 0
        for value in reversed(list(values)):
            self.push_front(value)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        node = self._head
        while node is not None:
            yield node.value
            node = node.next

    def __repr__(self) -> str:
        return f"LinkedList({list(self)!r})"

    def push_front(self, value: Any) -> None:
        """O(1) — the operation linked lists exist for."""
        self._head = _Node(value, self._head)
        self._size += 1

    def pop_front(self) -> Any:
        """O(1)."""
        if self._head is None:
            raise IndexError("pop from empty list")
        node = self._head
        self._head = node.next
        self._size -= 1
        return node.value

    def find(self, value: Any) -> int:
        """O(n). Returns the index, or -1."""
        for i, item in enumerate(self):
            if item == value:
                return i
        return -1

    def remove(self, value: Any) -> bool:
        """O(n). Removes the first match.

        The ``prev`` pointer is what makes single-link removal possible: you
        cannot delete a node you are standing on without knowing its
        predecessor, which is the whole reason doubly linked lists exist.
        """
        prev: Optional[_Node] = None
        node = self._head
        while node is not None:
            if node.value == value:
                if prev is None:
                    self._head = node.next
                else:
                    prev.next = node.next
                self._size -= 1
                return True
            prev, node = node, node.next
        return False

    def reverse(self) -> None:
        """Reverse in place, O(n) time and O(1) space.

        The three-pointer walk is the canonical linked-list exercise. Each step
        rewires exactly one ``next`` pointer; the ``nxt`` temporary exists only
        because overwriting ``node.next`` would otherwise lose the rest of the
        list.
        """
        prev: Optional[_Node] = None
        node = self._head
        while node is not None:
            nxt = node.next
            node.next = prev
            prev, node = node, nxt
        self._head = prev

    def middle(self) -> Any:
        """The middle element in one pass, using the tortoise and hare.

        ``fast`` moves two steps for every one of ``slow``, so when ``fast``
        reaches the end ``slow`` is halfway. The naive approach needs two passes
        (one to find the length); this needs one and no length counter — the
        same trick underlies cycle detection.
        """
        if self._head is None:
            raise IndexError("empty list")
        slow = fast = self._head
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next          # type: ignore[assignment]
            fast = fast.next.next
        return slow.value

    def has_cycle(self) -> bool:
        """Floyd's cycle detection, O(n) time and O(1) space.

        If a cycle exists the hare eventually laps the tortoise inside it. The
        alternative — a set of visited nodes — is also O(n) time but costs O(n)
        space, which is the point of the comparison.
        """
        slow = fast = self._head
        while fast is not None and fast.next is not None:
            slow = slow.next          # type: ignore[assignment]
            fast = fast.next.next
            if slow is fast:
                return True
        return False

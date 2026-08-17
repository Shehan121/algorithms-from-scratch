"""An unbalanced binary search tree, including its failure mode."""

from __future__ import annotations

from typing import Any, Iterator, Optional


class _Node:
    __slots__ = ("key", "left", "right")

    def __init__(self, key: Any) -> None:
        self.key = key
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None


class BST:
    """Binary search tree. O(log n) average, **O(n) worst case**.

    Deliberately left unbalanced, because the degenerate case is the lesson.
    Insert 1..n in ascending order and every node becomes a right child: the
    tree is a linked list, height n, and every operation is O(n). The benchmarks
    in this repo measure that collapse directly — average height on random input
    against height on sorted input.

    That failure is the entire motivation for AVL and red-black trees, which pay
    a rotation cost on write to guarantee O(log n) height.
    """

    def __init__(self, keys: Any = ()) -> None:
        self._root: Optional[_Node] = None
        self._size = 0
        for key in keys:
            self.insert(key)

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.search(key)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.in_order())

    def insert(self, key: Any) -> bool:
        """O(height). Duplicates are ignored; returns whether it was added."""
        if self._root is None:
            self._root = _Node(key)
            self._size += 1
            return True

        node = self._root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = _Node(key)
                    self._size += 1
                    return True
                node = node.left
            elif node.key < key:
                if node.right is None:
                    node.right = _Node(key)
                    self._size += 1
                    return True
                node = node.right
            else:
                return False

    def search(self, key: Any) -> bool:
        """O(height). Iterative, so it costs no stack."""
        node = self._root
        while node is not None:
            if key < node.key:
                node = node.left
            elif node.key < key:
                node = node.right
            else:
                return True
        return False

    def delete(self, key: Any) -> bool:
        """O(height). The three-case deletion, which is where BSTs get fiddly."""
        self._root, removed = self._delete(self._root, key)
        if removed:
            self._size -= 1
        return removed

    def _delete(self, node: Optional[_Node], key: Any) -> tuple[Optional[_Node], bool]:
        if node is None:
            return None, False

        if key < node.key:
            node.left, removed = self._delete(node.left, key)
            return node, removed
        if node.key < key:
            node.right, removed = self._delete(node.right, key)
            return node, removed

        # Found it. Case 1 and 2: zero or one child - splice the child up.
        if node.left is None:
            return node.right, True
        if node.right is None:
            return node.left, True

        # Case 3: two children. Replace the key with its in-order successor
        # (smallest key in the right subtree), then delete that successor.
        # Using the successor rather than an arbitrary descendant is what keeps
        # the search property intact.
        successor = node.right
        while successor.left is not None:
            successor = successor.left
        node.key = successor.key
        node.right, _ = self._delete(node.right, successor.key)
        return node, True

    def in_order(self) -> list[Any]:
        """Sorted order, iteratively with an explicit stack.

        In-order traversal of a BST yields sorted keys - that equivalence is the
        definition of the structure. Done with a stack rather than recursion so a
        degenerate tree of 10,000 nodes does not blow the interpreter's limit,
        which is precisely the input the benchmarks use.
        """
        out: list[Any] = []
        stack: list[_Node] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            out.append(node.key)
            node = node.right
        return out

    def height(self) -> int:
        """Height in edges; -1 for an empty tree. O(n), iterative."""
        if self._root is None:
            return -1
        best = 0
        stack = [(self._root, 0)]
        while stack:
            node, depth = stack.pop()
            best = max(best, depth)
            if node.left is not None:
                stack.append((node.left, depth + 1))
            if node.right is not None:
                stack.append((node.right, depth + 1))
        return best

    def min(self) -> Any:
        if self._root is None:
            raise IndexError("empty tree")
        node = self._root
        while node.left is not None:
            node = node.left
        return node.key

    def max(self) -> Any:
        if self._root is None:
            raise IndexError("empty tree")
        node = self._root
        while node.right is not None:
            node = node.right
        return node.key

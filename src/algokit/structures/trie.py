"""A prefix tree, and the trade it makes against a hash set."""

from __future__ import annotations

from typing import Iterator


class _Node:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: dict[str, "_Node"] = {}
        self.is_word = False


class Trie:
    """Prefix tree over strings. Operations cost O(L) in the key length.

    The point of a trie is what a hash set cannot do. Both give you membership -
    a hash set in O(L) to hash the string, a trie in O(L) to walk it, so lookup
    is a wash. The difference is that a trie answers **prefix** questions:
    "every word starting with 'pre'" is a subtree walk here and a full scan in a
    hash set.

    The cost is memory: one node per distinct prefix, each with a dict. A trie
    over random strings is far larger than the equivalent set. It pays off when
    keys share prefixes, which is why it backs autocomplete and IP routing
    tables rather than general-purpose maps.

    ``is_word`` is separate from "has no children" because one word can be a
    prefix of another - without the flag, inserting "car" then "carpet" would
    lose "car".
    """

    def __init__(self, words: Iterator[str] | list[str] = ()) -> None:
        self._root = _Node()
        self._size = 0
        for word in words:
            self.insert(word)

    def __len__(self) -> int:
        return self._size

    def __contains__(self, word: str) -> bool:
        return self.search(word)

    def insert(self, word: str) -> None:
        """O(L)."""
        node = self._root
        for char in word:
            node = node.children.setdefault(char, _Node())
        if not node.is_word:
            node.is_word = True
            self._size += 1

    def _walk(self, prefix: str) -> _Node | None:
        node = self._root
        for char in prefix:
            node = node.children.get(char)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        """Exact match. O(L)."""
        node = self._walk(word)
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        """Is any word prefixed by this? O(L) - the operation a hash set cannot do."""
        return self._walk(prefix) is not None

    def with_prefix(self, prefix: str) -> list[str]:
        """Every word under ``prefix``, O(size of that subtree).

        Crucially independent of how many words the trie holds overall - the
        cost scales with the number of answers, not the size of the dictionary.
        """
        node = self._walk(prefix)
        if node is None:
            return []
        out: list[str] = []
        self._collect(node, prefix, out)
        return out

    def _collect(self, node: _Node, path: str, out: list[str]) -> None:
        if node.is_word:
            out.append(path)
        for char, child in sorted(node.children.items()):
            self._collect(child, path + char, out)

    def longest_common_prefix(self) -> str:
        """The longest prefix shared by every word. O(L) in that prefix.

        Walk down while exactly one child exists and no word ends. A trie makes
        this almost free; comparing strings pairwise would be O(n*L).
        """
        node = self._root
        parts: list[str] = []
        while len(node.children) == 1 and not node.is_word:
            char, node = next(iter(node.children.items()))
            parts.append(char)
        return "".join(parts)

    def node_count(self) -> int:
        """Total nodes - the memory cost the trie pays for prefix queries."""
        total = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.children.values())
        return total

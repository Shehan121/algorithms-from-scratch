"""Disjoint-set union, and the two optimisations that make it nearly O(1)."""

from __future__ import annotations


class UnionFind:
    """Disjoint-set forest with union by rank and path compression.

    Amortised O(alpha(n)) per operation, where alpha is the inverse Ackermann
    function - below 5 for any n that fits in the universe, so effectively
    constant. Getting there needs *both* optimisations; either alone leaves it
    at O(log n).

    **Union by rank** attaches the shorter tree under the taller one. Attaching
    arbitrarily instead lets a chain of n unions build a path of length n,
    making find O(n).

    **Path compression** re-points every node visited by ``find`` directly at
    the root, so the next query on that branch is O(1). The two together make
    the trees so flat that the amortised bound collapses to near-constant.

    ``rank`` is an upper bound on height, not the exact height - path
    compression flattens trees without updating ranks. That is fine: it is only
    ever used to decide which way round to attach, and a slightly stale bound
    still makes a good choice.
    """

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._parent = list(range(size))
        self._rank = [0] * size
        self._components = size

    def __len__(self) -> int:
        return len(self._parent)

    @property
    def components(self) -> int:
        """Number of disjoint sets, maintained in O(1) rather than recomputed."""
        return self._components

    def find(self, x: int) -> int:
        """Representative of x's set, compressing the path on the way.

        Written iteratively in two passes rather than recursively: the recursive
        one-liner is elegant but costs stack depth proportional to the path
        length, which defeats the point on the very inputs that need compressing.
        """
        root = x
        while self._parent[root] != root:
            root = self._parent[root]

        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge two sets. Returns False if they were already joined."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def max_depth(self) -> int:
        """Deepest path in the forest - how flat compression has kept it."""
        best = 0
        for i in range(len(self._parent)):
            depth = 0
            node = i
            while self._parent[node] != node:
                node = self._parent[node]
                depth += 1
            best = max(best, depth)
        return best

"""A hash table with separate chaining, and a demonstration of why load factor matters."""

from __future__ import annotations

from typing import Any, Iterator


class HashTable:
    """Hash map using separate chaining. O(1) average, O(n) worst case.

    Collisions are resolved by chaining: each bucket holds a list of
    ``(key, value)`` pairs. The alternative, open addressing, probes for the next
    free slot - faster when the table is sparse, but it degrades sharply as the
    table fills and complicates deletion (a removed entry must leave a tombstone
    or later probes terminate early).

    The **load factor** is the whole story of performance here. With n entries in
    m buckets the average chain length is n/m, so lookup is O(1 + n/m). Keep n/m
    bounded by resizing and you get amortised O(1); let the table fill and every
    operation drifts toward O(n). This class resizes at 0.75, the same threshold
    Java's HashMap uses, and ``load_factor`` and ``collision_count`` are exposed
    so the benchmarks can show the effect rather than assert it.

    Worst case is genuinely O(n): if every key hashes to one bucket the table is
    a linked list. That is not merely theoretical - it is the shape of a hash
    collision denial-of-service attack.
    """

    _MAX_LOAD = 0.75

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._buckets: list[list[tuple[Any, Any]]] = [[] for _ in range(capacity)]
        self._size = 0
        self.resizes = 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"HashTable({dict(self.items())!r})"

    @property
    def capacity(self) -> int:
        return len(self._buckets)

    @property
    def load_factor(self) -> float:
        return self._size / len(self._buckets)

    def collision_count(self) -> int:
        """Entries sharing a bucket with at least one other entry."""
        return sum(len(b) - 1 for b in self._buckets if len(b) > 1)

    def longest_chain(self) -> int:
        """The worst-case lookup length in the current table."""
        return max((len(b) for b in self._buckets), default=0)

    def _index(self, key: Any) -> int:
        # Python's hash() can be negative, so mask to a non-negative value
        # before taking the modulus. Using abs() would be subtly worse: it maps
        # h and -h to the same bucket, doubling collisions for no reason.
        return (hash(key) & 0x7FFFFFFF) % len(self._buckets)

    def put(self, key: Any, value: Any) -> None:
        """Insert or overwrite. Amortised O(1)."""
        bucket = self._buckets[self._index(key)]
        for i, (existing, _) in enumerate(bucket):
            if existing == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self.load_factor > self._MAX_LOAD:
            self._resize()

    def get(self, key: Any, default: Any = None) -> Any:
        """O(1) average, O(chain length) actual."""
        for existing, value in self._buckets[self._index(key)]:
            if existing == key:
                return value
        return default

    def __contains__(self, key: Any) -> bool:
        return any(existing == key for existing, _ in self._buckets[self._index(key)])

    def delete(self, key: Any) -> bool:
        """O(1) average. Chaining makes deletion trivial - just drop the pair."""
        bucket = self._buckets[self._index(key)]
        for i, (existing, _) in enumerate(bucket):
            if existing == key:
                del bucket[i]
                self._size -= 1
                return True
        return False

    def _resize(self) -> None:
        """Double the bucket count and rehash every entry.

        Every key must be rehashed because the bucket index depends on the
        table size. Doubling keeps this amortised: the table grows geometrically,
        so the total rehash work across n inserts is O(n).
        """
        old = self._buckets
        self._buckets = [[] for _ in range(len(old) * 2)]
        self._size = 0
        self.resizes += 1
        for bucket in old:
            for key, value in bucket:
                self.put(key, value)

    def items(self) -> Iterator[tuple[Any, Any]]:
        for bucket in self._buckets:
            yield from bucket

    def keys(self) -> Iterator[Any]:
        for key, _ in self.items():
            yield key

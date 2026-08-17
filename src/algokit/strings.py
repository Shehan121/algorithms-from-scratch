"""String matching: three algorithms with the same job and different failure modes.

All three find every occurrence of ``pattern`` in ``text`` and return the start
indices. What separates them is what happens on adversarial input, which is why
the benchmarks include a worst case built specifically for the naive version.
"""

from __future__ import annotations

__all__ = ["naive_search", "kmp_search", "kmp_table", "rabin_karp_search", "MATCHERS"]


def naive_search(text: str, pattern: str) -> list[int]:
    """Check every alignment. O(n * m) worst case, O(n) typical.

    In practice this is often fine — the inner loop usually breaks on the first
    character. It collapses only when text and pattern share long repeated
    prefixes, e.g. ``"aaaa...a"`` searched for ``"aaa...ab"``, where every
    alignment matches m-1 characters before failing. The benchmarks use exactly
    that input, because on random text the naive version is competitive and the
    problem looks solved.
    """
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    hits: list[int] = []
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            hits.append(i)
    return hits


def kmp_table(pattern: str) -> list[int]:
    """Longest proper prefix which is also a suffix, for each prefix. O(m).

    This is the whole insight of KMP. ``table[i]`` says: having matched ``i``
    characters and then failed, how many of them are still usable because they
    form a prefix of the pattern? That is how KMP never re-reads a character of
    the text.

    The construction is itself KMP matching the pattern against itself, which is
    why the ``while`` loop falls back through ``table[length - 1]`` rather than
    restarting.
    """
    table = [0] * len(pattern)
    length = 0

    for i in range(1, len(pattern)):
        while length > 0 and pattern[i] != pattern[length]:
            length = table[length - 1]
        if pattern[i] == pattern[length]:
            length += 1
        table[i] = length
    return table


def kmp_search(text: str, pattern: str) -> list[int]:
    """Knuth-Morris-Pratt. **O(n + m) guaranteed**, no worst case.

    The text index never moves backwards — that is the guarantee. On failure the
    pattern slides forward by as much as the prefix table permits, so each text
    character is examined a bounded number of times. Total work is linear
    regardless of input, which is what the naive version cannot promise.

    The cost is O(m) preprocessing and the conceptual difficulty of the table.
    On random text the naive search is frequently *faster* in wall-clock terms,
    because its inner loop is trivial and KMP's bookkeeping is not — a real
    example of asymptotic superiority not implying practical superiority.
    """
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    table = kmp_table(pattern)
    hits: list[int] = []
    matched = 0

    for i in range(n):
        while matched > 0 and text[i] != pattern[matched]:
            matched = table[matched - 1]
        if text[i] == pattern[matched]:
            matched += 1
        if matched == m:
            hits.append(i - m + 1)
            matched = table[matched - 1]  # keep going, allowing overlaps
    return hits


def rabin_karp_search(text: str, pattern: str, base: int = 256, modulus: int = 1_000_000_007) -> list[int]:
    """Rolling-hash search. O(n + m) expected, O(n * m) worst case.

    A window's hash is derived from the previous one in constant time by
    subtracting the outgoing character and adding the incoming one, so hashing
    every alignment costs O(n) rather than O(n * m).

    The hash is a filter, not a decision: matching hashes trigger an explicit
    character comparison, because different strings can share a hash. Skipping
    that verification is the classic Rabin-Karp bug — it produces false
    positives that only appear on inputs you did not test.

    Worst case is O(n * m) when every window collides. A large prime modulus
    makes that vanishingly unlikely for non-adversarial input, but an attacker
    who knows the modulus can construct it. Its real strength is multi-pattern
    search: several patterns of equal length can be checked against one rolling
    hash in a single pass.
    """
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    high_order = pow(base, m - 1, modulus)
    pattern_hash = 0
    window_hash = 0
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % modulus
        window_hash = (window_hash * base + ord(text[i])) % modulus

    hits: list[int] = []
    for i in range(n - m + 1):
        # Verify on hash match - a hash hit is only a candidate.
        if window_hash == pattern_hash and text[i : i + m] == pattern:
            hits.append(i)

        if i < n - m:
            outgoing = ord(text[i]) * high_order
            window_hash = ((window_hash - outgoing) * base + ord(text[i + m])) % modulus
            # Python's % already returns non-negative, but being explicit
            # documents the step that C-family languages must handle manually.
            window_hash %= modulus
    return hits


MATCHERS = {
    "naive": naive_search,
    "kmp": kmp_search,
    "rabin-karp": rabin_karp_search,
}

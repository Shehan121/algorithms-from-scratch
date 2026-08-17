"""Dynamic programming, built up from the naive version each time.

Every problem here is implemented more than once — naive, memoised, tabulated,
space-optimised — because the *progression* is the lesson. Dynamic programming
is not a family of algorithms so much as a single observation: if a recursive
solution recomputes subproblems, store them.

``reports/`` measures the difference rather than asserting it, and the gap is
larger than most people expect.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = [
    "fib_naive",
    "fib_memo",
    "fib_table",
    "fib_constant_space",
    "grid_paths_naive",
    "grid_paths_table",
    "lcs_length",
    "lcs_string",
    "edit_distance",
    "edit_distance_rolling",
    "knapsack_01",
    "knapsack_items",
    "coin_change",
    "longest_increasing_subsequence",
    "lis_binary_search",
]


# --------------------------------------------------------------------------
# Fibonacci — the canonical demonstration
# --------------------------------------------------------------------------


def fib_naive(n: int) -> int:
    """Exponential Fibonacci. O(phi^n) time, O(n) space.

    Included because it is the clearest illustration of overlapping subproblems.
    ``fib(5)`` calls ``fib(3)`` twice, ``fib(2)`` three times, ``fib(1)`` five
    times — the call counts are themselves Fibonacci numbers. Total calls for
    ``fib(n)`` is ``2*fib(n+1) - 1``.

    The benchmarks stop at n=30 for this one; n=50 would take hours.
    """
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n: int) -> int:
    """Top-down with memoisation. O(n) time, O(n) space.

    Same recursion, one cache. Every distinct argument is computed once, so the
    call tree collapses from exponential to linear — an exponential speedup from
    a single decorator, which is the most dramatic one-line change in this
    repository.

    What the cache does *not* buy is stack depth: the first descent still goes n
    frames deep, so this raises ``RecursionError`` for large n where
    :func:`fib_table` is fine. Where exactly it breaks is version-dependent —
    CPython 3.12 stopped charging the C-level ``lru_cache`` wrapper against the
    Python recursion limit, so n=500 raises on 3.10 and returns on 3.13.
    """

    @lru_cache(maxsize=None)
    def go(k: int) -> int:
        if k < 2:
            return k
        return go(k - 1) + go(k - 2)

    return go(n)


def fib_table(n: int) -> int:
    """Bottom-up tabulation. O(n) time, O(n) space.

    Same complexity as memoisation but no recursion, so no stack limit and no
    per-call overhead. The trade is that tabulation computes *every* subproblem
    whereas memoisation computes only the ones actually reached — which matters
    when the reachable subset is sparse, as in :func:`coin_change`.
    """
    if n < 2:
        return n
    table = [0] * (n + 1)
    table[1] = 1
    for i in range(2, n + 1):
        table[i] = table[i - 1] + table[i - 2]
    return table[n]


def fib_constant_space(n: int) -> int:
    """O(n) time, **O(1)** space.

    The table only ever reads the last two entries, so the rest is dead weight.
    Recognising that a DP recurrence has a bounded window is the standard route
    from O(n) to O(1) space, and the same idea powers
    :func:`edit_distance_rolling`.
    """
    if n < 2:
        return n
    previous, current = 0, 1
    for _ in range(n - 1):
        previous, current = current, previous + current
    return current


# --------------------------------------------------------------------------
# Grid paths — counting, and why the naive form explodes
# --------------------------------------------------------------------------


def grid_paths_naive(rows: int, cols: int) -> int:
    """Count monotone lattice paths recursively. O(2^(rows+cols))."""
    if rows == 1 or cols == 1:
        return 1
    return grid_paths_naive(rows - 1, cols) + grid_paths_naive(rows, cols - 1)


def grid_paths_table(rows: int, cols: int) -> int:
    """The same count in O(rows * cols).

    Each cell is the sum of the cell above and the cell to the left. The naive
    version recomputes overlapping sub-grids exponentially often; the table
    visits each cell once.
    """
    table = [[1] * cols for _ in range(rows)]
    for r in range(1, rows):
        for c in range(1, cols):
            table[r][c] = table[r - 1][c] + table[r][c - 1]
    return table[rows - 1][cols - 1]


# --------------------------------------------------------------------------
# Sequence problems
# --------------------------------------------------------------------------


def lcs_length(a: str, b: str) -> int:
    """Longest common subsequence length. O(len(a) * len(b)).

    Subsequence, not substring — the characters need not be adjacent. The
    recurrence is the classic two-case split: if the last characters match, they
    are both in the LCS; otherwise drop one or the other and take the better.

    The table is padded with a zero row and column so the base case needs no
    special-casing inside the loop, a small trick that removes several
    off-by-one opportunities.
    """
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[n][m]


def lcs_string(a: str, b: str) -> str:
    """Reconstruct the LCS itself by walking the table backwards.

    Note that the *length* can be found in O(min(n, m)) space with a rolling
    row, but recovering the actual subsequence needs the full O(n*m) table to
    backtrack through. That tension — cheap answer versus recoverable answer — is
    a recurring DP trade-off.
    """
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    out: list[str] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. O(len(a) * len(b)) time and space.

    Three edits — insert, delete, substitute — each costing 1, so every cell is
    ``1 + min`` of three neighbours. The first row and column are seeded with
    ``0..n``: turning a prefix into the empty string costs one deletion per
    character.
    """
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                table[i][j] = 1 + min(
                    table[i - 1][j],      # delete from a
                    table[i][j - 1],      # insert into a
                    table[i - 1][j - 1],  # substitute
                )
    return table[n][m]


def edit_distance_rolling(a: str, b: str) -> int:
    """Levenshtein in O(min(len(a), len(b))) space.

    Each row depends only on the row above, so two rows suffice. Swapping the
    arguments so ``b`` is the shorter string bounds the memory by the smaller
    input — a free improvement that costs one comparison.
    """
    if len(b) > len(a):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                current[j] = previous[j - 1]
            else:
                current[j] = 1 + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[len(b)]


# --------------------------------------------------------------------------
# Choice problems
# --------------------------------------------------------------------------


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 knapsack: maximum value within a weight budget. O(n * capacity).

    "0/1" means each item is taken whole or not at all, which is what makes it
    hard — the fractional version is solved greedily by value density in
    O(n log n), and that greedy approach is simply **wrong** here.

    Worth being precise about the complexity: O(n * capacity) is *pseudo*
    polynomial. Capacity is a numeric value, not an input length, so the runtime
    is exponential in the number of bits used to write the capacity down. 0/1
    knapsack is NP-hard, and this table does not contradict that.
    """
    table = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]

    for i in range(1, len(weights) + 1):
        weight, value = weights[i - 1], values[i - 1]
        for cap in range(capacity + 1):
            table[i][cap] = table[i - 1][cap]                      # skip item
            if weight <= cap:
                take = table[i - 1][cap - weight] + value          # take item
                if take > table[i][cap]:
                    table[i][cap] = take
    return table[len(weights)][capacity]


def knapsack_items(weights: list[int], values: list[int], capacity: int) -> list[int]:
    """Which items an optimal knapsack contains, by backtracking the table.

    A cell differing from the one above means item ``i`` was taken — the same
    backtracking idea as :func:`lcs_string`.
    """
    n = len(weights)
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        weight, value = weights[i - 1], values[i - 1]
        for cap in range(capacity + 1):
            table[i][cap] = table[i - 1][cap]
            if weight <= cap:
                take = table[i - 1][cap - weight] + value
                if take > table[i][cap]:
                    table[i][cap] = take

    chosen: list[int] = []
    cap = capacity
    for i in range(n, 0, -1):
        if table[i][cap] != table[i - 1][cap]:
            chosen.append(i - 1)
            cap -= weights[i - 1]
    return sorted(chosen)


def coin_change(coins: list[int], amount: int) -> int:
    """Fewest coins making ``amount``, or -1. O(len(coins) * amount).

    The natural greedy approach — always take the largest coin that fits — is
    optimal for currency systems like EUR but wrong in general. With coins
    ``[1, 3, 4]`` and amount 6, greedy takes 4+1+1 = three coins while the
    optimum is 3+3 = two. The tests assert exactly this case, because a greedy
    solution that works on every example you happen to try is a trap.
    """
    INF = float("inf")
    best = [0] + [INF] * amount

    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < best[value]:
                best[value] = best[value - coin] + 1
    return -1 if best[amount] == INF else int(best[amount])


def longest_increasing_subsequence(seq: list[int]) -> int:
    """LIS length, the O(n^2) DP.

    ``best[i]`` is the longest increasing subsequence ending at ``i``. Simple,
    and the obvious first solution — kept so the O(n log n) version below has
    something to be measured against.
    """
    if not seq:
        return 0
    best = [1] * len(seq)
    for i in range(1, len(seq)):
        for j in range(i):
            if seq[j] < seq[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best)


def lis_binary_search(seq: list[int]) -> int:
    """LIS length in O(n log n), via patience sorting.

    ``tails[k]`` holds the smallest possible tail of an increasing subsequence
    of length k+1. That array is always sorted, so the position for each new
    element can be found by binary search instead of a linear scan — which is
    where the O(n) inner loop becomes O(log n).

    Note ``tails`` is *not* itself an LIS; only its length is meaningful.
    Recovering the actual subsequence needs extra parent bookkeeping.
    """
    tails: list[int] = []
    for value in seq:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if tails[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(value)
        else:
            tails[lo] = value
    return len(tails)

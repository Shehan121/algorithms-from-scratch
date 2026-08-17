"""Dynamic programming: every variant must agree, and greedy must be shown wrong."""

import random

import pytest

from algokit.dynamic import (
    coin_change,
    edit_distance,
    edit_distance_rolling,
    fib_constant_space,
    fib_memo,
    fib_naive,
    fib_table,
    grid_paths_naive,
    grid_paths_table,
    knapsack_01,
    knapsack_items,
    lcs_length,
    lcs_string,
    lis_binary_search,
    longest_increasing_subsequence,
)

FIB_20 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]


class TestFibonacci:
    @pytest.mark.parametrize("fn", [fib_naive, fib_memo, fib_table, fib_constant_space])
    def test_matches_known_values(self, fn):
        assert [fn(i) for i in range(21)] == FIB_20

    def test_all_implementations_agree_at_scale(self):
        """The four differ enormously in cost and not at all in result."""
        for n in (50, 200, 500):
            assert fib_memo(n) == fib_table(n) == fib_constant_space(n)


class TestGridPaths:
    @pytest.mark.parametrize("rows,cols,expected", [(1, 1, 1), (2, 2, 2), (3, 3, 6), (3, 7, 28)])
    def test_known_values(self, rows, cols, expected):
        assert grid_paths_table(rows, cols) == expected

    def test_naive_and_table_agree(self):
        for r in range(1, 8):
            for c in range(1, 8):
                assert grid_paths_naive(r, c) == grid_paths_table(r, c)


class TestSequences:
    @pytest.mark.parametrize(
        "a,b,expected",
        [("ABCBDAB", "BDCABA", 4), ("", "abc", 0), ("abc", "abc", 3), ("abc", "xyz", 0)],
    )
    def test_lcs_length(self, a, b, expected):
        assert lcs_length(a, b) == expected

    def test_lcs_string_is_a_subsequence_of_both(self):
        a, b = "ABCBDAB", "BDCABA"
        result = lcs_string(a, b)
        assert len(result) == lcs_length(a, b)
        for source in (a, b):
            it = iter(source)
            assert all(ch in it for ch in result)

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("", "abc", 3),
            ("abc", "", 3),
            ("same", "same", 0),
        ],
    )
    def test_edit_distance(self, a, b, expected):
        assert edit_distance(a, b) == expected

    def test_rolling_matches_full_table(self):
        rng = random.Random(4)
        letters = "abcd"
        for _ in range(40):
            a = "".join(rng.choice(letters) for _ in range(rng.randrange(12)))
            b = "".join(rng.choice(letters) for _ in range(rng.randrange(12)))
            assert edit_distance(a, b) == edit_distance_rolling(a, b)


class TestKnapsack:
    def test_known_optimum(self):
        weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
        assert knapsack_01(weights, values, 7) == 9      # items of weight 3 and 4

    def test_items_match_the_reported_value(self):
        weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
        chosen = knapsack_items(weights, values, 7)
        assert sum(weights[i] for i in chosen) <= 7
        assert sum(values[i] for i in chosen) == knapsack_01(weights, values, 7)

    def test_zero_capacity_and_no_items(self):
        assert knapsack_01([1, 2], [10, 20], 0) == 0
        assert knapsack_01([], [], 10) == 0

    def test_greedy_by_density_is_not_optimal(self):
        """Why 0/1 knapsack needs DP: the best density-first pick loses.

        Densities: 6/3 = 2.0, 10/5 = 2.0, 12/6 = 2.0 - all equal, but with
        capacity 8 the optimum is items 0+1 (weight 8, value 16), while a greedy
        pass that grabs item 2 first is stuck at 12.
        """
        weights, values = [3, 5, 6], [6, 10, 12]
        assert knapsack_01(weights, values, 8) == 16


class TestCoinChange:
    @pytest.mark.parametrize(
        "coins,amount,expected",
        [([1, 2, 5], 11, 3), ([2], 3, -1), ([1], 0, 0), ([5, 10, 25], 30, 2)],
    )
    def test_known_values(self, coins, amount, expected):
        assert coin_change(coins, amount) == expected

    def test_greedy_would_be_wrong(self):
        """[1,3,4] making 6: greedy gives 4+1+1 = 3 coins; optimum is 3+3 = 2."""
        assert coin_change([1, 3, 4], 6) == 2


class TestLIS:
    @pytest.mark.parametrize(
        "seq,expected",
        [
            ([10, 9, 2, 5, 3, 7, 101, 18], 4),
            ([], 0),
            ([5], 1),
            ([3, 3, 3], 1),
            ([1, 2, 3, 4], 4),
            ([4, 3, 2, 1], 1),
        ],
    )
    def test_known_values(self, seq, expected):
        assert longest_increasing_subsequence(seq) == expected

    def test_quadratic_and_log_versions_agree(self):
        """The O(n log n) version is only trustworthy if it matches the simple one."""
        rng = random.Random(6)
        for _ in range(60):
            seq = [rng.randrange(40) for _ in range(rng.randrange(30))]
            assert longest_increasing_subsequence(seq) == lis_binary_search(seq)

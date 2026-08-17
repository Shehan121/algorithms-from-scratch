"""String matching: all three algorithms must return identical results."""

import random

import pytest

from algokit.strings import MATCHERS, kmp_table

ALL = list(MATCHERS.values())


@pytest.mark.parametrize("fn", ALL)
@pytest.mark.parametrize(
    "text,pattern,expected",
    [
        ("abracadabra", "abra", [0, 7]),
        ("aaaa", "aa", [0, 1, 2]),          # overlapping matches must all be found
        ("hello", "world", []),
        ("hello", "", []),
        ("", "a", []),
        ("a", "a", [0]),
        ("abc", "abcd", []),                 # pattern longer than text
        ("mississippi", "issi", [1, 4]),
    ],
)
def test_known_matches(fn, text, pattern, expected):
    assert fn(text, pattern) == expected


@pytest.mark.parametrize("fn", ALL)
def test_agrees_with_reference_on_random_input(fn):
    """Cross-checked against a straightforward str.find loop."""
    rng = random.Random(8)
    for _ in range(120):
        text = "".join(rng.choice("abc") for _ in range(rng.randrange(40)))
        pattern = "".join(rng.choice("abc") for _ in range(rng.randrange(1, 4)))

        expected, start = [], 0
        while (found := text.find(pattern, start)) != -1:
            expected.append(found)
            start = found + 1
        assert fn(text, pattern) == expected


def test_all_matchers_agree_on_the_naive_worst_case():
    """The input built to make naive search quadratic."""
    text = "a" * 500 + "b"
    pattern = "a" * 100 + "b"
    results = [fn(text, pattern) for fn in ALL]
    assert all(r == results[0] for r in results)
    assert results[0] == [400]


class TestKmpTable:
    @pytest.mark.parametrize(
        "pattern,expected",
        [
            ("aaaa", [0, 1, 2, 3]),
            ("abcd", [0, 0, 0, 0]),
            ("ababaca", [0, 0, 1, 2, 3, 0, 1]),
            ("aabaaa", [0, 1, 0, 1, 2, 2]),
        ],
    )
    def test_known_tables(self, pattern, expected):
        assert kmp_table(pattern) == expected

    def test_table_entries_are_proper_prefixes(self):
        """Each entry must be a real prefix-suffix, and strictly shorter."""
        rng = random.Random(12)
        for _ in range(50):
            pattern = "".join(rng.choice("ab") for _ in range(rng.randrange(1, 15)))
            table = kmp_table(pattern)
            for i, length in enumerate(table):
                assert length <= i
                assert pattern[:length] == pattern[i - length + 1 : i + 1]

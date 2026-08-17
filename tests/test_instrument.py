"""The measurement machinery itself needs testing - a wrong counter is worse than none."""

import math

from algokit.complexity import MODELS, best_fit, fit_all, growth_ratios
from algokit.instrument import Ops, Probe, TrackedList, instrument, unwrap
from algokit.sorting import insertion_sort, merge_sort, selection_sort


class TestProbe:
    def test_counts_comparisons(self):
        ops = Ops()
        a, b = Probe(1, ops), Probe(2, ops)
        assert a < b
        assert ops.comparisons == 1

    def test_comparisons_are_correct_not_just_counted(self):
        ops = Ops()
        assert Probe(3, ops) > Probe(2, ops)
        assert Probe(2, ops) <= Probe(2, ops)
        assert Probe(2, ops) == Probe(2, ops)

    def test_tracked_list_counts_reads_and_writes(self):
        ops = Ops()
        a = TrackedList([1, 2, 3], ops=ops)
        _ = a[0]
        a[1] = 9
        assert ops.reads == 1 and ops.writes == 1

    def test_instrument_starts_from_zero(self):
        """Construction cost must not be charged to the algorithm."""
        _, ops = instrument(range(100))
        assert ops.comparisons == ops.reads == ops.writes == 0

    def test_unwrap_recovers_values(self):
        tracked, _ = instrument([3, 1, 2])
        assert unwrap(merge_sort(tracked)) == [1, 2, 3]


class TestKnownCounts:
    def test_selection_sort_comparison_count_is_exact(self):
        """Selection sort always performs exactly n(n-1)/2 comparisons.

        A closed form makes this the ideal check that the counter is accurate
        rather than merely plausible.
        """
        for n in (10, 25, 50):
            tracked, ops = instrument(range(n, 0, -1))
            selection_sort(tracked)
            assert ops.comparisons == n * (n - 1) // 2

    def test_selection_sort_count_is_input_independent(self):
        """It has no best case - sorted and reversed input cost the same."""
        counts = []
        for data in (range(40), range(40, 0, -1)):
            tracked, ops = instrument(data)
            selection_sort(tracked)
            counts.append(ops.comparisons)
        assert counts[0] == counts[1]

    def test_insertion_sort_best_case_is_linear(self):
        """Sorted input costs exactly n-1 comparisons."""
        n = 50
        tracked, ops = instrument(range(n))
        insertion_sort(tracked)
        assert ops.comparisons == n - 1

    def test_merge_sort_comparisons_are_near_the_lower_bound(self):
        """Should sit close to n log2(n), and far below n^2 / 2."""
        n = 512
        tracked, ops = instrument(range(n, 0, -1))
        merge_sort(tracked)
        assert ops.comparisons < 2 * n * math.log2(n)
        assert ops.comparisons < n * n / 10


class TestComplexityFitting:
    def test_recovers_linear_growth(self):
        sizes = [100, 200, 400, 800, 1600]
        assert best_fit(sizes, [3.0 * n for n in sizes]).model == "O(n)"

    def test_recovers_quadratic_growth(self):
        sizes = [10, 20, 40, 80, 160]
        assert best_fit(sizes, [float(n * n) for n in sizes]).model == "O(n^2)"

    def test_recovers_n_log_n(self):
        sizes = [128, 256, 512, 1024, 2048, 4096]
        costs = [n * math.log2(n) for n in sizes]
        assert best_fit(sizes, costs).model == "O(n log n)"

    def test_perfect_fit_scores_one(self):
        sizes = [10, 20, 30, 40]
        assert best_fit(sizes, [float(n) for n in sizes]).r_squared > 0.9999

    def test_doubling_ratios_reveal_the_class(self):
        sizes = [10, 20, 40, 80]
        quadratic = growth_ratios(sizes, [float(n * n) for n in sizes])
        assert all(abs(r - 4.0) < 1e-9 for r in quadratic)
        linear = growth_ratios(sizes, [float(n) for n in sizes])
        assert all(abs(r - 2.0) < 1e-9 for r in linear)

    def test_every_model_is_fitted(self):
        sizes = [10, 20, 40, 80]
        assert {f.model for f in fit_all(sizes, [float(n) for n in sizes])} == set(MODELS)

"""Measure every algorithm and write the results to ``reports/``.

Run with::

    python scripts/run_benchmarks.py

Produces CSV files that ``scripts/make_figures.py`` turns into charts, and the
numbers quoted in the README. Nothing here asserts a complexity — every claim is
derived from the measurements.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algokit import benchmark as bm  # noqa: E402
from algokit.complexity import best_fit, growth_ratios  # noqa: E402
from algokit.dynamic import (  # noqa: E402
    fib_constant_space,
    fib_memo,
    fib_naive,
    fib_table,
    lis_binary_search,
    longest_increasing_subsequence,
)
from algokit.instrument import instrument  # noqa: E402
from algokit.sorting import COMPARISON_SORTS, SORTS  # noqa: E402
from algokit.strings import kmp_search, naive_search, rabin_karp_search  # noqa: E402
from algokit.structures import BST, HashTable  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Quadratic sorts become unusable long before the O(n log n) ones do, so each
# gets its own size ladder. Powers of two make the doubling ratios directly
# readable.
SMALL_SIZES = [64, 128, 256, 512, 1024, 2048]
LARGE_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]

QUADRATIC = {"bubble", "insertion", "selection", "quick (last pivot)"}


def write_csv(name: str, rows: list[dict], fields: list[str]) -> Path:
    path = REPORTS / name
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")
    return path


# --------------------------------------------------------------------------


def bench_sort_operations() -> list[dict]:
    """Comparison counts per sort per input shape. Deterministic and exact.

    Only comparisons are recorded. The sorts copy their input (``a = list(seq)``)
    so they never mutate the caller's data, which means the tracked list sees the
    copy and nothing after it — read and write counts would report the copy, not
    the sort. See the warning on ``algokit.instrument.instrument``.
    """
    rows: list[dict] = []
    for shape, generator in bm.INPUT_SHAPES.items():
        for name, fn in COMPARISON_SORTS.items():
            sizes = SMALL_SIZES if name in QUADRATIC else LARGE_SIZES
            # A fixed-pivot quicksort recurses n deep on sorted input, so cap it.
            if name == "quick (last pivot)" and shape in {"sorted", "nearly sorted"}:
                sizes = [s for s in sizes if s <= 512]

            for n in sizes:
                tracked, ops = instrument(generator(n))
                try:
                    fn(tracked)
                except RecursionError:
                    rows.append({
                        "algorithm": name, "shape": shape, "n": n,
                        "comparisons": -1, "note": "RecursionError",
                    })
                    break
                rows.append({
                    "algorithm": name, "shape": shape, "n": n,
                    "comparisons": ops.comparisons, "note": "",
                })
    return rows


def bench_sort_timing() -> list[dict]:
    """Wall-clock timing per sort on random input."""
    rows: list[dict] = []
    for name, fn in SORTS.items():
        sizes = SMALL_SIZES if name in QUADRATIC else LARGE_SIZES
        for timing in bm.sweep_time(fn, sizes, repeats=3, name=name, limit_seconds=2.0):
            rows.append({"algorithm": name, "n": timing.n, "seconds": f"{timing.seconds:.8f}"})
    return rows


def bench_crossover() -> list[dict]:
    """Where does insertion sort stop beating merge sort?

    The question behind every real library sort's small-array cutoff. Measured on
    two input shapes, because the answer differs by more than an order of
    magnitude between them — which is the point. On random input insertion sort
    is genuinely O(n^2); on nearly-sorted input it is O(n*k) for bounded
    displacement k, so it stays competitive far longer than the usual "switch
    below 32 elements" folklore suggests.

    The range runs to 8192 so the crossover is actually located rather than
    assumed. An earlier version stopped at 256 and insertion sort won at every
    single size, which tells you nothing except that the range was too narrow.
    """
    from algokit.sorting import insertion_sort, merge_sort

    rows: list[dict] = []
    shapes = {"random": bm.random_list, "nearly sorted": bm.nearly_sorted_list}

    for shape, generator in shapes.items():
        for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            data = generator(n)
            # Fewer repeats at large n, where a single run is already slow.
            repeats = 200 if n <= 256 else 20 if n <= 2048 else 5
            insertion = bm.time_call(insertion_sort, data, repeats=repeats)
            merge = bm.time_call(merge_sort, data, repeats=repeats)
            rows.append({
                "shape": shape,
                "n": n,
                "insertion_seconds": f"{insertion:.9f}",
                "merge_seconds": f"{merge:.9f}",
                "winner": "insertion" if insertion < merge else "merge",
            })
    return rows


def bench_fibonacci() -> list[dict]:
    """The DP progression, measured. Naive stops early by necessity."""
    rows: list[dict] = []
    for n in range(5, 36):
        row = {"n": n}
        if n <= 30:
            row["naive_seconds"] = f"{bm.time_call(fib_naive, n, repeats=1):.9f}"
        else:
            row["naive_seconds"] = ""
        row["memo_seconds"] = f"{bm.time_call(fib_memo, n, repeats=5):.9f}"
        row["table_seconds"] = f"{bm.time_call(fib_table, n, repeats=5):.9f}"
        row["constant_seconds"] = f"{bm.time_call(fib_constant_space, n, repeats=5):.9f}"
        rows.append(row)
    return rows


def bench_bst_height() -> list[dict]:
    """Random versus sorted insertion - the degenerate tree, measured.

    Theory says random insertion gives height ~4.31*log2(n) and sorted
    insertion gives exactly n-1.
    """
    rows: list[dict] = []
    for n in [100, 200, 400, 800, 1600, 3200]:
        random_tree = BST(bm.random_list(n, seed=1, spread=n * 100))
        sorted_tree = BST(range(n))
        rows.append({
            "n": n,
            "random_height": random_tree.height(),
            "sorted_height": sorted_tree.height(),
            "log2_n": f"{math.log2(n):.2f}",
        })
    return rows


def bench_hash_load_factor() -> list[dict]:
    """Chain length against load factor, with resizing disabled.

    Forcing a fixed capacity shows what the resize policy is protecting against.
    """
    rows: list[dict] = []
    capacity = 64
    for n in [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        table = HashTable(capacity=capacity)
        table._MAX_LOAD = float("inf")  # suppress resizing for the experiment
        for i in range(n):
            table.put(f"key-{i}", i)
        rows.append({
            "n": n,
            "capacity": capacity,
            "load_factor": f"{n / capacity:.3f}",
            "longest_chain": table.longest_chain(),
            "collisions": table.collision_count(),
        })
    return rows


def bench_string_matching() -> list[dict]:
    """Naive versus KMP versus Rabin-Karp on random and adversarial text."""
    rows: list[dict] = []
    matchers = {"naive": naive_search, "kmp": kmp_search, "rabin-karp": rabin_karp_search}

    for n in [1000, 2000, 4000, 8000, 16000]:
        # Adversarial: every alignment matches m-1 characters before failing.
        adversarial_text = "a" * n
        adversarial_pattern = "a" * 50 + "b"
        # Benign: random text where mismatches happen on the first character.
        random_text = "".join(bm.random.Random(3).choice("abcdefgh") for _ in range(n))
        random_pattern = "hgfe"

        for label, text, pattern in [
            ("adversarial", adversarial_text, adversarial_pattern),
            ("random", random_text, random_pattern),
        ]:
            for name, fn in matchers.items():
                seconds = bm.time_call(fn, text, pattern, repeats=3)
                rows.append({
                    "algorithm": name, "shape": label, "n": n,
                    "seconds": f"{seconds:.8f}",
                })
    return rows


def bench_lis() -> list[dict]:
    """O(n^2) LIS against the O(n log n) patience-sorting version."""
    rows: list[dict] = []
    for n in [100, 200, 400, 800, 1600, 3200]:
        data = bm.random_list(n, seed=2)
        rows.append({
            "n": n,
            "quadratic_seconds": f"{bm.time_call(longest_increasing_subsequence, data, repeats=3):.8f}",
            "nlogn_seconds": f"{bm.time_call(lis_binary_search, data, repeats=3):.8f}",
        })
    return rows


# --------------------------------------------------------------------------


def summarise_complexity(op_rows: list[dict]) -> list[dict]:
    """Fit each sort's measured comparison counts to a growth class."""
    out: list[dict] = []
    for shape in bm.INPUT_SHAPES:
        for name in COMPARISON_SORTS:
            points = [
                (r["n"], float(r["comparisons"]))
                for r in op_rows
                if r["algorithm"] == name and r["shape"] == shape and r["comparisons"] > 0
            ]
            if len(points) < 4:
                continue  # too few points to distinguish adjacent classes
            sizes = [p[0] for p in points]
            costs = [p[1] for p in points]
            fit = best_fit(sizes, costs)
            ratios = growth_ratios(sizes, costs)
            out.append({
                "algorithm": name,
                "shape": shape,
                "measured_class": fit.model,
                "r_squared": f"{fit.r_squared:.5f}",
                "mean_doubling_ratio": f"{sum(ratios) / len(ratios):.2f}" if ratios else "",
                "points": len(points),
            })
    return out


def main() -> None:
    print("Benchmarking (this takes a minute or two)\n")

    print("operation counts for sorts")
    op_rows = bench_sort_operations()
    write_csv("sort_operations.csv", op_rows, ["algorithm", "shape", "n", "comparisons", "note"])

    print("complexity fits")
    fits = summarise_complexity(op_rows)
    write_csv("sort_complexity_fits.csv", fits,
              ["algorithm", "shape", "measured_class", "r_squared", "mean_doubling_ratio", "points"])

    print("sort timing")
    write_csv("sort_timing.csv", bench_sort_timing(), ["algorithm", "n", "seconds"])

    print("insertion/merge crossover")
    write_csv("crossover.csv", bench_crossover(),
              ["shape", "n", "insertion_seconds", "merge_seconds", "winner"])

    print("fibonacci progression")
    write_csv("fibonacci.csv", bench_fibonacci(),
              ["n", "naive_seconds", "memo_seconds", "table_seconds", "constant_seconds"])

    print("BST height")
    write_csv("bst_height.csv", bench_bst_height(), ["n", "random_height", "sorted_height", "log2_n"])

    print("hash table load factor")
    write_csv("hash_load_factor.csv", bench_hash_load_factor(),
              ["n", "capacity", "load_factor", "longest_chain", "collisions"])

    print("string matching")
    write_csv("string_matching.csv", bench_string_matching(), ["algorithm", "shape", "n", "seconds"])

    print("LIS variants")
    write_csv("lis.csv", bench_lis(), ["n", "quadratic_seconds", "nlogn_seconds"])

    # A machine-readable summary for the README to quote.
    summary = {
        "sizes": {"quadratic_sorts": SMALL_SIZES, "nlogn_sorts": LARGE_SIZES},
        "measured_classes": {f"{r['algorithm']} / {r['shape']}": r["measured_class"] for r in fits},
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  wrote reports/summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()

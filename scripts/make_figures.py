"""Turn the benchmark CSVs into the figures used in the README.

Run with::

    python scripts/make_figures.py

Charting decisions worth stating, since they affect what the reader concludes:

* **Log-log axes** for the growth plots. On linear axes a quadratic curve is a
  wall and everything else is flat against the floor; on log-log each complexity
  class becomes a straight line whose *slope* is the exponent, so O(n) and
  O(n^2) are distinguishable by eye.
* **Reference curves** (dashed) for n, n log n and n^2, scaled to pass through
  the first data point. A measured line running parallel to a reference is the
  visual form of the same claim the R^2 fit makes numerically.
* **Four series maximum** per panel, with the line labelled at its right end.
  Beyond four, colour alone stops being readable and the chart is split into
  small multiples instead.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Categorical slots in fixed order, validated for colour-vision deficiency
# separation against the light chart surface. Never cycled: a chart needing a
# fifth series becomes small multiples.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e6e5e1"

# Single hue, light to dark, for magnitude.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
    })


def load(name: str) -> list[dict]:
    with (REPORTS / name).open() as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  {name}")


def thousands(value, _pos) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def label_end(ax, xs, ys, text, colour) -> None:
    """Label a line at its right-hand end — identity without a legend lookup."""
    ax.annotate(text, xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=colour, va="center", fontweight="semibold")


def reference_curve(ax, xs, first_y, f, label, y_offset=1.0):
    """A dashed guide for a complexity class, scaled through the first point."""
    scale = first_y / f(xs[0])
    ys = [scale * f(x) * y_offset for x in xs]
    ax.plot(xs, ys, ls=(0, (4, 3)), lw=1.1, color=MUTED, zorder=1)
    ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=MUTED, va="center", style="italic")


# --------------------------------------------------------------------------


def fig_sort_comparisons() -> None:
    """Comparison counts on random input, split into the two complexity families."""
    rows = [r for r in load("sort_operations.csv")
            if r["shape"] == "random" and int(r["comparisons"]) > 0]

    families = [
        ("Quadratic family", ["selection", "bubble", "insertion"]),
        ("Linearithmic family", ["merge", "heap", "quick (random pivot)"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (title, names) in zip(axes, families):
        for i, name in enumerate(names):
            points = sorted(((int(r["n"]), int(r["comparisons"])) for r in rows
                             if r["algorithm"] == name), key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=name, zorder=3)
            label_end(ax, xs, ys, name.replace(" (random pivot)", ""), SERIES[i])

        first = sorted(((int(r["n"]), int(r["comparisons"])) for r in rows
                        if r["algorithm"] == names[0]), key=lambda p: p[0])
        xs = [p[0] for p in first]
        if "Quadratic" in title:
            reference_curve(ax, xs, first[0][1], lambda n: n * n, "n²")
        else:
            reference_curve(ax, xs, first[0][1], lambda n: n * math.log2(n), "n log n")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("input size n")
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax.set_xlim(right=xs[-1] * 2.9)
    axes[0].set_ylabel("comparisons")

    fig.suptitle("Measured comparison counts on random input (log–log: slope is the exponent)",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "01_sort_comparisons.png")


def fig_complexity_heatmap() -> None:
    """Doubling ratio per algorithm per input shape — the class, read directly."""
    rows = load("sort_complexity_fits.csv")
    shapes = ["random", "sorted", "reversed", "nearly sorted", "few unique"]
    algos = ["selection", "bubble", "insertion", "quick (last pivot)",
             "quick (random pivot)", "merge", "heap"]

    grid: list[list[float | None]] = []
    labels: list[list[str]] = []
    for algo in algos:
        row_v: list[float | None] = []
        row_l: list[str] = []
        for shape in shapes:
            match = [r for r in rows if r["algorithm"] == algo and r["shape"] == shape]
            if match and match[0]["mean_doubling_ratio"]:
                ratio = float(match[0]["mean_doubling_ratio"])
                row_v.append(ratio)
                cls = match[0]["measured_class"][2:-1].replace("^2", "\u00b2").replace("^3", "\u00b3")
                row_l.append(f"{ratio:.2f}\n{cls}")
            else:
                row_v.append(None)
                row_l.append("n/a")
        grid.append(row_v)
        labels.append(row_l)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    norm = Normalize(vmin=1.8, vmax=4.2)

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value is None:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#f2f1ee",
                                           edgecolor=SURFACE, linewidth=2))
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            # 2px surface gap between cells, per the mark spec.
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=cmap(norm(value)),
                                       edgecolor=SURFACE, linewidth=2))
            dark = norm(value) > 0.62
            ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=8.2,
                    color="#ffffff" if dark else INK, linespacing=1.35)

    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_xlim(-0.5, len(shapes) - 0.5)
    ax.set_ylim(len(algos) - 0.5, -0.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                       shrink=0.72, pad=0.02)
    bar.set_label("cost ratio per doubling of n", color=INK_2, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, colors=MUTED)

    ax.set_title("Measured growth per doubling of n  ·  2.0 = linear, ~2.2 = n log n, 4.0 = quadratic",
                 pad=14)
    save(fig, "02_complexity_heatmap.png")


def fig_crossover() -> None:
    """Insertion versus merge on two input shapes — where the crossover really is."""
    rows = load("crossover.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)

    for ax, shape in zip(axes, ["random", "nearly sorted"]):
        subset = sorted((r for r in rows if r["shape"] == shape), key=lambda r: int(r["n"]))
        xs = [int(r["n"]) for r in subset]
        insertion = [float(r["insertion_seconds"]) * 1e6 for r in subset]
        merge = [float(r["merge_seconds"]) * 1e6 for r in subset]

        ax.plot(xs, insertion, marker="o", color=SERIES[0], label="insertion", zorder=3)
        ax.plot(xs, merge, marker="s", color=SERIES[1], label="merge", zorder=3)
        label_end(ax, xs, insertion, "insertion", SERIES[0])
        label_end(ax, xs, merge, "merge", SERIES[1])

        crossing = next((int(r["n"]) for r in subset if r["winner"] == "merge"), None)
        if crossing is not None:
            ax.axvline(crossing, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f"crossover\nn = {crossing}", xy=(crossing, max(insertion) * 0.55),
                        xytext=(8, 0), textcoords="offset points", fontsize=8.5,
                        color=INK_2, va="center")
        else:
            ax.annotate("insertion wins at every size tested", xy=(0.04, 0.93),
                        xycoords="axes fraction", fontsize=8.8, color=INK_2)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{shape} input")
        ax.set_xlabel("input size n")
        ax.set_xlim(right=xs[-1] * 3.4)
    axes[0].set_ylabel("time (µs)")

    fig.suptitle("Insertion sort versus merge sort: the crossover depends on the input shape",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.04)
    save(fig, "03_crossover.png")


def fig_fibonacci() -> None:
    """The DP progression: one decorator, an exponential speedup."""
    rows = load("fibonacci.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    series = [
        ("naive_seconds", "naive recursion", SERIES[1], "o"),
        ("memo_seconds", "memoised", SERIES[0], "s"),
        ("table_seconds", "tabulated", SERIES[2], "^"),
        ("constant_seconds", "constant space", SERIES[3], "D"),
    ]
    for field, label, colour, marker in series:
        points = [(int(r["n"]), float(r[field])) for r in rows if r[field]]
        xs = [p[0] for p in points]
        ys = [p[1] * 1e6 for p in points]
        ax.plot(xs, ys, marker=marker, color=colour, label=label, zorder=3, markevery=3)
        label_end(ax, xs, ys, label, colour)

    naive = [(int(r["n"]), float(r["naive_seconds"])) for r in rows if r["naive_seconds"]]
    memo = {int(r["n"]): float(r["memo_seconds"]) for r in rows}
    ratio = naive[-1][1] / memo[naive[-1][0]]
    ax.annotate(f"at n = {naive[-1][0]}, naive is\n{ratio:,.0f}× slower",
                xy=(naive[-1][0], naive[-1][1] * 1e6), xytext=(-104, -6),
                textcoords="offset points", fontsize=9, color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.9))

    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("time (µs, log scale)")
    ax.set_xlim(right=40)
    ax.set_title("Fibonacci: exponential, linear, and constant-space, measured")
    save(fig, "04_fibonacci.png")


def fig_bst_height() -> None:
    """Random versus sorted insertion — the degenerate tree."""
    rows = load("bst_height.csv")
    xs = [int(r["n"]) for r in rows]
    random_h = [int(r["random_height"]) for r in rows]
    sorted_h = [int(r["sorted_height"]) for r in rows]
    log2 = [float(r["log2_n"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(xs, sorted_h, marker="s", color=SERIES[1], label="sorted insertion", zorder=3)
    ax.plot(xs, random_h, marker="o", color=SERIES[0], label="random insertion", zorder=3)
    ax.plot(xs, log2, ls=(0, (4, 3)), lw=1.2, color=MUTED, zorder=1)

    label_end(ax, xs, sorted_h, "sorted", SERIES[1])
    label_end(ax, xs, random_h, "random", SERIES[0])
    ax.annotate("log₂ n", xy=(xs[-1], log2[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=MUTED, va="center", style="italic")

    ax.annotate(f"{sorted_h[-1] / random_h[-1]:.0f}× taller\nat n = {xs[-1]:,}",
                xy=(xs[-1], sorted_h[-1]), xytext=(-96, -14), textcoords="offset points",
                fontsize=9, color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.9))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("keys inserted")
    ax.set_ylabel("tree height (edges)")
    ax.set_xlim(right=xs[-1] * 2.4)
    ax.set_title("An unbalanced BST collapses to a linked list on sorted input")
    save(fig, "05_bst_height.png")


def fig_hash_load_factor() -> None:
    """Longest chain against load factor, with the resize threshold marked."""
    rows = load("hash_load_factor.csv")
    loads = [float(r["load_factor"]) for r in rows]
    chains = [int(r["longest_chain"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(loads, chains, marker="o", color=SERIES[0], zorder=3)

    ax.axvline(0.75, color=SERIES[1], lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate("resize threshold (0.75)\nthe table never goes right of here",
                xy=(0.75, max(chains) * 0.82), xytext=(12, 0), textcoords="offset points",
                fontsize=8.8, color=SERIES[1], va="center")

    for factor, chain in zip(loads, chains):
        if factor in (0.75, 4.0, 8.0):
            ax.annotate(f"{chain}", xy=(factor, chain), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)

    ax.set_xlabel("load factor  (entries ÷ buckets)")
    ax.set_ylabel("longest chain (worst-case lookup)")
    ax.set_title("Hash table degradation with resizing disabled")
    save(fig, "06_hash_load_factor.png")


def fig_string_matching() -> None:
    """Naive versus KMP versus Rabin-Karp, on benign and adversarial text."""
    rows = load("string_matching.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    for ax, shape in zip(axes, ["adversarial", "random"]):
        for i, name in enumerate(["naive", "kmp", "rabin-karp"]):
            points = sorted(((int(r["n"]), float(r["seconds"]) * 1000) for r in rows
                             if r["algorithm"] == name and r["shape"] == shape),
                            key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=name, zorder=3)
            label_end(ax, xs, ys, name, SERIES[i])

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("text length n")
        ax.set_xlim(right=xs[-1] * 3.2)
        subtitle = ("aaaa… searched for aaa…b" if shape == "adversarial"
                    else "random text, 8-letter alphabet")
        ax.set_title(f"{shape}\n{subtitle}", fontsize=10.5)
    axes[0].set_ylabel("time (ms)")

    fig.suptitle("String matching: KMP's guarantee only pays off on adversarial input",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.06)
    save(fig, "07_string_matching.png")


def fig_lis() -> None:
    """The O(n^2) → O(n log n) improvement, and how the gap compounds."""
    rows = load("lis.csv")
    xs = [int(r["n"]) for r in rows]
    quad = [float(r["quadratic_seconds"]) * 1000 for r in rows]
    fast = [float(r["nlogn_seconds"]) * 1000 for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(xs, quad, marker="o", color=SERIES[1], label="O(n²) dynamic programming", zorder=3)
    ax.plot(xs, fast, marker="s", color=SERIES[0], label="O(n log n) patience sorting", zorder=3)
    label_end(ax, xs, quad, "O(n²)", SERIES[1])
    label_end(ax, xs, fast, "O(n log n)", SERIES[0])

    ax.annotate(f"{quad[-1] / fast[-1]:.0f}× gap at n = {xs[-1]:,}\n"
                f"(only {quad[0] / fast[0]:.0f}× at n = {xs[0]})",
                xy=(xs[-1], quad[-1]), xytext=(-118, -18), textcoords="offset points",
                fontsize=9, color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.9))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length n")
    ax.set_ylabel("time (ms)")
    ax.set_xlim(right=xs[-1] * 2.2)
    ax.set_title("Longest increasing subsequence: the asymptotic gap compounds")
    save(fig, "08_lis.png")


def main() -> None:
    style()
    print(f"Writing figures to {FIGURES.relative_to(ROOT)}")
    fig_sort_comparisons()
    fig_complexity_heatmap()
    fig_crossover()
    fig_fibonacci()
    fig_bst_height()
    fig_hash_load_factor()
    fig_string_matching()
    fig_lis()
    print("Done.")


if __name__ == "__main__":
    main()

"""Infer an algorithm's complexity class from measurements.

The idea: given (n, cost) pairs, fit each candidate growth curve
``cost ≈ a * f(n) + b`` by least squares and report which ``f`` explains the data
best. If an implementation is genuinely O(n log n), the ``n log n`` model should
fit noticeably better than ``n`` or ``n^2``.

This is how the claims in the README are checked rather than asserted. It has
real limits, stated here because pretending otherwise would defeat the purpose:

* Adjacent classes are hard to separate. ``n`` and ``n log n`` differ by a factor
  of ~13 across n = 100..10,000, so noise or a constant-factor offset can flip
  the verdict.
* Only the classes listed are considered. A genuinely O(n^1.5) algorithm will be
  reported as whichever neighbour fits least badly.
* Timing noise, garbage collection and cache effects all show up as curvature.
  Operation counts are deterministic and therefore far more reliable here —
  which is why the instrumentation exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

#: Candidate growth functions, ordered from cheapest to most expensive.
MODELS: dict[str, Callable[[float], float]] = {
    "O(1)": lambda n: 1.0,
    "O(log n)": lambda n: math.log2(n) if n > 1 else 1.0,
    "O(n)": lambda n: n,
    "O(n log n)": lambda n: n * math.log2(n) if n > 1 else 1.0,
    "O(n^2)": lambda n: n * n,
    "O(n^3)": lambda n: n * n * n,
}


@dataclass
class Fit:
    """How well one growth model explains a set of measurements."""

    model: str
    r_squared: float
    scale: float
    offset: float

    def predict(self, n: float) -> float:
        return self.scale * MODELS[self.model](n) + self.offset


def _least_squares(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Fit ``y = a*x + b``; return (a, b, r_squared)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    if sxx == 0:
        return 0.0, mean_y, 0.0

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def fit_all(sizes: Sequence[int], costs: Sequence[float]) -> list[Fit]:
    """Fit every candidate model, best first.

    Each model is fitted by transforming n through ``f`` and then doing an
    ordinary linear regression — a curve that is nonlinear in n is still linear
    in its coefficients, so no iterative solver is needed.
    """
    if len(sizes) < 3:
        raise ValueError("need at least three measurements to fit a curve")

    fits: list[Fit] = []
    for name, f in MODELS.items():
        xs = [f(float(n)) for n in sizes]
        slope, intercept, r2 = _least_squares(xs, list(costs))
        fits.append(Fit(model=name, r_squared=r2, scale=slope, offset=intercept))

    return sorted(fits, key=lambda f: f.r_squared, reverse=True)


def best_fit(sizes: Sequence[int], costs: Sequence[float]) -> Fit:
    """The single best-explaining growth class."""
    return fit_all(sizes, costs)[0]


def growth_ratios(sizes: Sequence[int], costs: Sequence[float]) -> list[float]:
    """Cost ratio for each doubling of n — the quickest sanity check there is.

    For a doubling of input size the ratio approaches:

    ===============  =====
    complexity       ratio
    ===============  =====
    O(1)             1
    O(log n)         ~1
    O(n)             2
    O(n log n)       ~2.1-2.3
    O(n^2)           4
    O(n^3)           8
    ===============  =====

    Reading these ratios off a table is often more convincing than any R^2,
    because they need no model at all — a column of 4s *is* quadratic behaviour.
    """
    ratios: list[float] = []
    for i in range(1, len(costs)):
        previous = costs[i - 1]
        ratios.append(float("nan") if previous == 0 else costs[i] / previous)
    return ratios


def describe(sizes: Sequence[int], costs: Sequence[float], top: int = 3) -> str:
    """A short human-readable verdict, for printing in reports."""
    fits = fit_all(sizes, costs)
    lines = [f"best fit: {fits[0].model}  (R^2 = {fits[0].r_squared:.4f})"]
    for fit in fits[1:top]:
        lines.append(f"  next:   {fit.model:<12} R^2 = {fit.r_squared:.4f}")

    ratios = growth_ratios(sizes, costs)
    if ratios:
        formatted = ", ".join("n/a" if r != r else f"{r:.2f}" for r in ratios)
        lines.append(f"  ratios per step: {formatted}")
    return "\n".join(lines)

"""Forecast accuracy metrics.

Pure functions — no side effects, no I/O.
"""

from __future__ import annotations

import numpy as np


def rmse(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Root mean squared error."""
    valid = ~(np.isnan(actuals) | np.isnan(forecasts))
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actuals[valid] - forecasts[valid]) ** 2)))


def mae(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Mean absolute error."""
    valid = ~(np.isnan(actuals) | np.isnan(forecasts))
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actuals[valid] - forecasts[valid])))


def directional_accuracy(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Fraction of steps where forecast correctly predicted the direction of change.

    Direction is the sign of the step-over-step CHANGE (acceleration vs
    deceleration), not the sign of the level. A flat forecast against
    positive-growth actuals should score near 0.5 (random), not 1.0.

    Returns value in [0, 1], or NaN if fewer than 2 valid pairs.
    """
    valid = ~(np.isnan(actuals) | np.isnan(forecasts))
    a = actuals[valid]
    f = forecasts[valid]
    if len(a) < 2:
        return float("nan")
    a_diff = np.diff(a)
    f_diff = np.diff(f)
    # Treat exact ties on the forecast side as incorrect — a flat forecast
    # made no directional call.
    correct = (np.sign(a_diff) == np.sign(f_diff)) & (f_diff != 0)
    return float(correct.mean())


def ci_coverage(
    actuals: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
) -> float:
    """Fraction of actuals that fall within the confidence interval.

    Returns value in [0, 1]. A well-calibrated 80% CI should have
    coverage near 0.80.
    """
    valid = ~(np.isnan(actuals) | np.isnan(ci_lo) | np.isnan(ci_hi))
    if valid.sum() == 0:
        return float("nan")
    a = actuals[valid]
    lo = ci_lo[valid]
    hi = ci_hi[valid]
    inside = (a >= lo) & (a <= hi)
    return float(inside.mean())

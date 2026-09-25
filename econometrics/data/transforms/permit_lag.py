"""Calibrate the permit-to-delivery lag for a specific metro.

Estimates how many months it takes for building permits to convert to
actual unit deliveries by finding the lag that maximizes correlation
between lagged permits and observed deliveries.

Usage:
    econ calibrate-permit-lag --metro dallas_tx
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_permit_lag(
    permits: pd.Series,
    deliveries: pd.Series,
    candidate_lags_months: list[int] | None = None,
    min_overlap: int = 12,
) -> dict:
    """Find the optimal permit-to-delivery lag for a metro.

    Args:
        permits: Quarterly building permit counts (PeriodIndex).
        deliveries: Quarterly delivery counts from CoStar (PeriodIndex).
        candidate_lags_months: Lags to test (in months). Default: 12-30 in 2-month steps.
        min_overlap: Minimum overlapping quarters for a valid correlation.

    Returns:
        Dict with:
        - optimal_lag_months: int — best lag in months
        - optimal_lag_quarters: int — best lag in quarters
        - correlation: float — Pearson r at optimal lag
        - mse: float — MSE at optimal lag
        - all_results: list[dict] — results for every candidate lag
        - valid: bool — True if correlation >= 0.5
    """
    if candidate_lags_months is None:
        candidate_lags_months = list(range(12, 31, 2))

    results = []
    for lag_months in candidate_lags_months:
        lag_quarters = lag_months // 3

        # Shift permits backward by lag: permits[t - lag] should predict deliveries[t]
        # A positive shift aligns past permits with current deliveries.
        lagged = permits.shift(lag_quarters)

        # Align and drop NaN
        combined = pd.DataFrame({
            "permits_lagged": lagged,
            "deliveries": deliveries,
        }).dropna()

        if len(combined) < min_overlap:
            continue

        corr = combined["permits_lagged"].corr(combined["deliveries"])
        mse = ((combined["permits_lagged"] - combined["deliveries"]) ** 2).mean()

        results.append({
            "lag_months": lag_months,
            "lag_quarters": lag_quarters,
            "correlation": corr,
            "mse": mse,
            "n_overlap": len(combined),
        })

    if not results:
        return {
            "optimal_lag_months": 18,  # Default fallback
            "optimal_lag_quarters": 6,
            "correlation": 0.0,
            "mse": float("nan"),
            "all_results": [],
            "valid": False,
        }

    # Select lag with highest correlation
    best = max(results, key=lambda r: r["correlation"])

    return {
        "optimal_lag_months": best["lag_months"],
        "optimal_lag_quarters": best["lag_quarters"],
        "correlation": best["correlation"],
        "mse": best["mse"],
        "all_results": results,
        "valid": best["correlation"] >= 0.5,
    }

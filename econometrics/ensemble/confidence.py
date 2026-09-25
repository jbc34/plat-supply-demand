"""Empirical quantile confidence intervals from ensemble residuals.

Does NOT assume normality. Aggregates weighted residual distributions
from all ensemble members and computes percentile-based bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from econometrics.models.base import ForecastResult


def empirical_ci(
    results: list[ForecastResult],
    weights: dict[str, float],
    levels: tuple[float, float] = (0.80, 0.95),
) -> dict[str, np.ndarray]:
    """Compute empirical confidence intervals from weighted model residuals.

    Args:
        results: List of ForecastResult objects (one per model).
        weights: Dict mapping model_name → weight (should sum to ~1.0).
        levels: CI levels to compute (e.g., 0.80 and 0.95).

    Returns:
        Dict with keys: ci_80_lo, ci_80_hi, ci_95_lo, ci_95_hi
        (arrays with same length as forecast horizon).
    """
    # Collect weighted residuals from all models
    all_resid = []
    all_weights = []

    for result in results:
        if result.residuals is None:
            continue
        w = weights.get(result.model_name, 0.0)
        if w <= 0:
            continue

        resid = result.residuals
        if isinstance(resid, pd.Series):
            resid = resid.values

        resid = resid[~np.isnan(resid)]
        all_resid.append(resid)
        all_weights.append(np.full(len(resid), w))

    if not all_resid:
        # Fallback: zero CI
        horizon = results[0].horizon_quarters if results else 1
        zeros = np.zeros(horizon)
        return {"ci_80_lo": zeros, "ci_80_hi": zeros,
                "ci_95_lo": zeros, "ci_95_hi": zeros}

    resid_pool = np.concatenate(all_resid)
    weight_pool = np.concatenate(all_weights)

    # Normalize weights
    weight_pool = weight_pool / weight_pool.sum()

    # Compute weighted percentiles
    horizon = results[0].horizon_quarters

    output = {}
    for level in levels:
        alpha = 1 - level
        lo_pct = alpha / 2 * 100
        hi_pct = (1 - alpha / 2) * 100

        lo_val = _weighted_percentile(resid_pool, weight_pool, lo_pct)
        hi_val = _weighted_percentile(resid_pool, weight_pool, hi_pct)

        # Scale with horizon (wider CI for longer forecasts)
        horizon_scale = np.sqrt(np.arange(1, horizon + 1))

        level_key = f"{int(level * 100)}"
        output[f"ci_{level_key}_lo"] = lo_val * horizon_scale
        output[f"ci_{level_key}_hi"] = hi_val * horizon_scale

    return output


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    """Compute weighted percentile using linear interpolation."""
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    cumulative = np.cumsum(sorted_weights)
    # Normalize to [0, 100]
    cumulative = cumulative / cumulative[-1] * 100

    return float(np.interp(percentile, cumulative, sorted_values))

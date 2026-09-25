"""Combine multiple model forecasts into a single ensemble forecast.

Produces a weighted point forecast and empirical confidence intervals
from the individual model outputs.
"""

from __future__ import annotations

import json

import numpy as np

from econometrics.ensemble.confidence import empirical_ci
from econometrics.models.base import ForecastResult


def combine(
    results: list[ForecastResult],
    weights: dict[str, float],
) -> ForecastResult:
    """Combine model forecasts into an ensemble.

    Args:
        results: List of ForecastResult objects (one per model).
        weights: Dict mapping model_name → weight.

    Returns:
        Single ForecastResult with weighted point estimate and empirical CIs.
    """
    if not results:
        raise ValueError("No results to combine")

    horizon = results[0].horizon_quarters

    # Weighted point forecast
    point = np.zeros(horizon)
    total_weight = 0.0

    for result in results:
        w = weights.get(result.model_name, 0.0)
        if w > 0 and len(result.point) == horizon:
            point += w * result.point
            total_weight += w

    if total_weight > 0:
        point = point / total_weight

    # Empirical confidence intervals
    ci = empirical_ci(results, weights)

    # Apply CI offsets to point forecast
    ci_80_lo = point + ci.get("ci_80_lo", np.zeros(horizon))
    ci_80_hi = point + ci.get("ci_80_hi", np.zeros(horizon))
    ci_95_lo = point + ci.get("ci_95_lo", np.zeros(horizon))
    ci_95_hi = point + ci.get("ci_95_hi", np.zeros(horizon))

    # Collect all residuals (weighted) for the ensemble
    all_resid = []
    for result in results:
        if result.residuals is not None:
            w = weights.get(result.model_name, 0.0)
            if w > 0:
                all_resid.extend((result.residuals * w).tolist())

    # Use first result's metadata as template
    template = results[0]

    return ForecastResult(
        point=point,
        ci_80_lo=ci_80_lo,
        ci_80_hi=ci_80_hi,
        ci_95_lo=ci_95_lo,
        ci_95_hi=ci_95_hi,
        model_name=f"Ensemble({json.dumps({r.model_name: round(weights.get(r.model_name, 0), 3) for r in results})})",
        metro_id=template.metro_id,
        vintage_date=template.vintage_date,
        horizon_quarters=horizon,
        forecast_periods=template.forecast_periods,
        residuals=np.array(all_resid) if all_resid else None,
    )

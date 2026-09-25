"""Apply shock scenarios to panels and produce stressed forecasts.

Takes a baseline panel, applies the shock overrides from shock_library,
then re-runs the ensemble to produce an alternative forecast path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from econometrics.models.base import ForecastResult
from econometrics.scenarios.shock_library import ScenarioShock


@dataclass
class StressTestResult:
    """Result of a stress test."""

    scenario_name: str
    baseline: ForecastResult
    stressed: ForecastResult
    delta_point: np.ndarray     # stressed.point - baseline.point
    delta_pct: np.ndarray       # percentage change


def apply_shock(
    panel: pd.DataFrame,
    shock: ScenarioShock,
    start_quarter: int | None = None,
) -> pd.DataFrame:
    """Apply a shock scenario to a panel's last N quarters.

    Modifies the panel in-place (copy first if needed) by overriding
    variable values in the shock window.

    Args:
        panel: Quarterly panel to shock.
        shock: ScenarioShock definition.
        start_quarter: Index position to start the shock. If None, appends
            to the end of the panel.

    Returns:
        Shocked panel (copy — original is not modified).
    """
    shocked = panel.copy()
    n = len(shocked)

    if start_quarter is None:
        start_quarter = n - shock.duration_quarters

    start_quarter = max(0, min(start_quarter, n - 1))

    for var, shift in shock.variable_overrides.items():
        if var not in shocked.columns:
            continue

        for q in range(shock.duration_quarters):
            idx = start_quarter + q
            if idx >= n:
                break

            # Ramp-in: linearly scale from 0 to full shock
            if shock.ramp_in_quarters > 0 and q < shock.ramp_in_quarters:
                scale = (q + 1) / shock.ramp_in_quarters
            else:
                scale = 1.0

            shocked.iloc[idx, shocked.columns.get_loc(var)] += shift * scale

    return shocked


def run_stress_test(
    panel: pd.DataFrame,
    shock: ScenarioShock,
    model_cls: type,
    horizon: int = 8,
    metro_id: str = "",
    vintage_date: str = "",
    **model_kwargs,
) -> StressTestResult:
    """Run a single-model stress test.

    1. Fit model on unshocked panel → baseline forecast
    2. Apply shock to panel
    3. Fit model on shocked panel → stressed forecast
    4. Compare

    Args:
        panel: Quarterly panel.
        shock: Scenario shock to apply.
        model_cls: Model class to use.
        horizon: Forecast horizon.
        metro_id: Metro identifier.
        vintage_date: Vintage date.
        **model_kwargs: Passed to model constructor.

    Returns:
        StressTestResult with baseline, stressed, and delta.
    """
    # Baseline
    baseline_model = model_cls(**model_kwargs)
    baseline_model.fit(panel)
    baseline = baseline_model.predict(
        horizon=horizon, metro_id=metro_id, vintage_date=vintage_date,
    )

    # Shocked
    shocked_panel = apply_shock(panel, shock)
    stressed_model = model_cls(**model_kwargs)
    stressed_model.fit(shocked_panel)
    stressed = stressed_model.predict(
        horizon=horizon, metro_id=metro_id, vintage_date=vintage_date,
    )

    delta = stressed.point - baseline.point
    delta_pct = np.where(
        np.abs(baseline.point) > 1e-6,
        delta / np.abs(baseline.point) * 100,
        0.0,
    )

    return StressTestResult(
        scenario_name=shock.name,
        baseline=baseline,
        stressed=stressed,
        delta_point=delta,
        delta_pct=delta_pct,
    )

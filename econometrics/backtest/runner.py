"""Expanding-window backtest runner.

Fits a model on progressively larger training sets and evaluates
forecast accuracy at each step. Results are appended to the registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from econometrics.backtest import metrics, registry
from econometrics.models.base import ForecastModel, ForecastResult


def run_backtest(
    model_cls: type,
    panel: pd.DataFrame,
    metro_id: str,
    start_quarter: str | None = None,
    step_quarters: int = 1,
    horizon: int = 4,
    min_train: int = 40,
    register: bool = True,
    **model_kwargs,
) -> dict:
    """Run expanding-window backtest for a single model.

    At each step:
    1. Train on panel[:t]
    2. Forecast horizon quarters ahead
    3. Compare to actuals in panel[t:t+horizon]
    4. Optionally append to registry

    Args:
        model_cls: Model class to instantiate (must implement ForecastModel).
        panel: Full quarterly panel with actual values.
        metro_id: Metro identifier for registry.
        start_quarter: First quarter to use as forecast origin (e.g., "2018Q1").
            If None, starts at min_train quarters in.
        step_quarters: How many quarters to step forward between backtests.
        horizon: Forecast horizon in quarters.
        min_train: Minimum training set size.
        register: If True, append each forecast to the backtest registry.
        **model_kwargs: Passed to model_cls constructor.

    Returns:
        Dict with:
        - results: list of per-step dicts {vintage, rmse, mae, dir_acc}
        - overall_rmse: float
        - overall_mae: float
        - n_steps: int
    """
    n = len(panel)
    if n < min_train + horizon:
        return {"results": [], "overall_rmse": float("nan"),
                "overall_mae": float("nan"), "n_steps": 0}

    # Determine start index
    if start_quarter:
        start_period = pd.Period(start_quarter, freq="Q")
        start_idx = panel.index.get_loc(start_period)
        if isinstance(start_idx, slice):
            start_idx = start_idx.start
    else:
        start_idx = min_train

    start_idx = max(start_idx, min_train)
    target_col = model_kwargs.pop("target_col", "rent_growth_yoy")

    step_results = []
    all_actuals = []
    all_forecasts = []

    for t in range(start_idx, n - horizon + 1, step_quarters):
        train = panel.iloc[:t]
        test = panel.iloc[t:t + horizon]

        vintage_date = str(panel.index[t - 1])

        try:
            model = model_cls(**model_kwargs)
            model.fit(train)
            result = model.predict(
                horizon=horizon,
                metro_id=metro_id,
                vintage_date=vintage_date,
            )
        except Exception as e:
            # Skip this step if model fails (e.g., too few observations)
            continue

        # Get actuals
        if target_col in test.columns:
            actuals = test[target_col].values[:horizon]
        else:
            continue

        # Trim to available actuals
        n_actual = min(len(actuals), len(result.point))
        a = actuals[:n_actual]
        f = result.point[:n_actual]

        step_rmse = metrics.rmse(a, f)
        step_mae = metrics.mae(a, f)
        step_dir = metrics.directional_accuracy(a, f)

        step_results.append({
            "vintage": vintage_date,
            "rmse": step_rmse,
            "mae": step_mae,
            "directional_accuracy": step_dir,
        })

        all_actuals.extend(a.tolist())
        all_forecasts.extend(f.tolist())

        # Register forecast
        if register:
            run_id = registry.append(result)
            # Immediately fill actuals for this backtest
            actuals_dict = {
                result.forecast_periods[i]: float(a[i])
                for i in range(n_actual)
                if i < len(result.forecast_periods)
            }
            registry.fill_actuals(metro_id, actuals_dict)

    # Overall metrics
    all_a = np.array(all_actuals)
    all_f = np.array(all_forecasts)

    return {
        "results": step_results,
        "overall_rmse": metrics.rmse(all_a, all_f) if len(all_a) > 0 else float("nan"),
        "overall_mae": metrics.mae(all_a, all_f) if len(all_a) > 0 else float("nan"),
        "n_steps": len(step_results),
    }

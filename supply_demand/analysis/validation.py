"""Model validation and backtesting."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from supply_demand.models.state_space.kalman import KalmanFilter, FilterResults


@dataclass
class ValidationResult:
    """Results of model validation."""

    rmse: float
    mae: float
    mape: float
    correlation: float
    direction_accuracy: float
    hit_rate: float  # % of predictions within confidence interval


def backtest_model(
    kf: KalmanFilter,
    Y: np.ndarray,
    train_end: int,
    forecast_horizon: int = 12,
    rolling: bool = True,
    step_size: int = 1,
) -> pd.DataFrame:
    """Backtest the model using rolling or expanding window.

    Args:
        kf: Configured Kalman filter
        Y: Full observation matrix
        train_end: Initial training period end index
        forecast_horizon: Forecast horizon to evaluate
        rolling: If True, use rolling window; else expanding
        step_size: Steps between forecast origins

    Returns:
        DataFrame with actual vs predicted values
    """
    T = len(Y)
    results = []

    for t in range(train_end, T - forecast_horizon, step_size):
        # Training data
        if rolling:
            train_start = max(0, t - train_end)
            Y_train = Y[train_start:t]
        else:
            Y_train = Y[:t]

        # Reset and filter on training data
        kf.reset()
        filter_results = kf.filter(Y_train)

        # Forecast
        forecasts, forecast_covs = kf.forecast(forecast_horizon)

        # Get actuals
        actuals = Y[t:t + forecast_horizon]

        # Store results for each horizon
        for h in range(min(forecast_horizon, len(actuals))):
            results.append({
                "origin_date": t,
                "horizon": h + 1,
                "forecast": forecasts[h, 0],  # First state variable
                "forecast_std": np.sqrt(forecast_covs[h, 0, 0]),
                "actual": actuals[h, 0] if actuals.ndim > 1 else actuals[h],
            })

    df = pd.DataFrame(results)
    return df


def compute_forecast_metrics(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    forecast_stds: Optional[np.ndarray] = None,
) -> ValidationResult:
    """Compute forecast accuracy metrics.

    Args:
        forecasts: Predicted values
        actuals: Actual values
        forecast_stds: Forecast standard deviations (optional)

    Returns:
        ValidationResult with various metrics
    """
    errors = actuals - forecasts

    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAE
    mae = np.mean(np.abs(errors))

    # MAPE (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs(errors / actuals) * 100
        ape = ape[np.isfinite(ape)]
        mape = np.mean(ape) if len(ape) > 0 else np.nan

    # Correlation
    correlation = np.corrcoef(forecasts, actuals)[0, 1]

    # Direction accuracy
    if len(forecasts) > 1:
        forecast_direction = np.sign(np.diff(forecasts))
        actual_direction = np.sign(np.diff(actuals))
        direction_accuracy = np.mean(forecast_direction == actual_direction) * 100
    else:
        direction_accuracy = np.nan

    # Hit rate (predictions within 2 std)
    if forecast_stds is not None:
        within_interval = np.abs(errors) < 2 * forecast_stds
        hit_rate = np.mean(within_interval) * 100
    else:
        hit_rate = np.nan

    return ValidationResult(
        rmse=rmse,
        mae=mae,
        mape=mape,
        correlation=correlation,
        direction_accuracy=direction_accuracy,
        hit_rate=hit_rate,
    )


def cross_validate(
    kf: KalmanFilter,
    Y: np.ndarray,
    n_folds: int = 5,
    forecast_horizon: int = 12,
) -> pd.DataFrame:
    """Time-series cross-validation.

    Args:
        kf: Kalman filter
        Y: Observations
        n_folds: Number of CV folds
        forecast_horizon: Forecast horizon

    Returns:
        DataFrame with metrics for each fold
    """
    T = len(Y)
    fold_size = T // (n_folds + 1)

    results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + forecast_horizon, T)

        if test_end <= test_start:
            continue

        # Train
        kf.reset()
        kf.filter(Y[:train_end])

        # Forecast
        h = test_end - test_start
        forecasts, forecast_covs = kf.forecast(h)

        # Evaluate
        actuals = Y[test_start:test_end]

        metrics = compute_forecast_metrics(
            forecasts[:, 0],
            actuals[:, 0] if actuals.ndim > 1 else actuals,
            np.sqrt(forecast_covs[:, 0, 0]),
        )

        results.append({
            "fold": fold,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "correlation": metrics.correlation,
            "hit_rate": metrics.hit_rate,
        })

    return pd.DataFrame(results)

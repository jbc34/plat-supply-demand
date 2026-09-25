"""STL seasonality decomposition for quarterly rent data.

Separates trend, seasonal, and residual components before feeding
into VAR/ECM models. This is a pre-processing transform, not a
ForecastModel — it doesn't produce forecasts directly.

CoStar quarterly data has seasonal patterns (Q2/Q3 lease-up season).
Without decomposition, the VAR attributes seasonal variance to
structural dynamics, inflating lag coefficients and producing
noisy forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import STL


@dataclass
class STLComponents:
    """Result of STL decomposition."""

    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    original: pd.Series
    period: int


def decompose(
    series: pd.Series,
    period: int = 4,
    robust: bool = True,
) -> STLComponents:
    """STL decomposition of a quarterly series.

    Args:
        series: Quarterly time series (PeriodIndex or DatetimeIndex).
        period: Seasonal period (4 for quarterly data).
        robust: If True, use robust fitting (down-weights outliers).

    Returns:
        STLComponents with trend, seasonal, and residual.
    """
    clean = series.dropna()
    if len(clean) < 2 * period:
        # Too short for STL — return original as trend, zero seasonal
        return STLComponents(
            trend=clean,
            seasonal=pd.Series(0.0, index=clean.index),
            residual=pd.Series(0.0, index=clean.index),
            original=clean,
            period=period,
        )

    stl = STL(clean, period=period, robust=robust)
    result = stl.fit()

    return STLComponents(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        original=clean,
        period=period,
    )


def decompose_panel(
    panel: pd.DataFrame,
    columns: list[str] | None = None,
    period: int = 4,
) -> tuple[pd.DataFrame, dict[str, STLComponents]]:
    """Apply STL decomposition to multiple panel columns.

    Replaces each column with its trend component. Seasonal components
    are stored separately and can be added back to forecasts.

    Args:
        panel: Quarterly panel DataFrame.
        columns: Columns to decompose. If None, uses all endogenous vars.
        period: Seasonal period.

    Returns:
        Tuple of (deseasonalized panel, dict of STLComponents by column name).
    """
    if columns is None:
        columns = [
            "rent_growth_yoy", "vacancy_rate",
            "deliveries_pct_inventory", "absorption_pct_inventory",
        ]

    result = panel.copy()
    components = {}

    for col in columns:
        if col not in result.columns:
            continue

        stl = decompose(result[col], period=period)
        components[col] = stl
        # Replace with trend (deseasonalized)
        result.loc[stl.trend.index, col] = stl.trend

    return result, components


def reseasonalize_forecast(
    forecast: pd.Series,
    seasonal: STLComponents,
) -> pd.Series:
    """Add seasonal pattern back to a forecast.

    The seasonal component repeats with `period` frequency. For forecast
    periods beyond the observed data, the seasonal pattern is tiled.

    Args:
        forecast: Point forecast series (PeriodIndex).
        seasonal: STLComponents from the original decomposition.

    Returns:
        Forecast with seasonal component added back.
    """
    seasonal_values = seasonal.seasonal.values
    period = seasonal.period

    # Tile the seasonal pattern to cover the forecast horizon
    n = len(forecast)
    # Determine which quarter each forecast period falls in
    tiled = []
    for i in range(n):
        idx = i % period
        tiled.append(seasonal_values[-(period - idx)] if idx < period else 0.0)

    return forecast + pd.Series(tiled, index=forecast.index)

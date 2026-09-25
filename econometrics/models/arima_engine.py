"""ARIMA/SARIMA model for rent growth forecasting.

Univariate benchmark — if the ensemble can't beat ARIMA, the exogenous
variables aren't adding value. Also handles non-stationarity via
integrated differencing that Ridge/GBM ignore.

Implements the ForecastModel protocol for ensemble integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from econometrics.models.base import ForecastResult


class ARIMAEngine:
    """ARIMA with automatic order selection via AIC grid search."""

    def __init__(
        self,
        target_col: str = "rent_growth_yoy",
        max_p: int = 4,
        max_d: int = 2,
        max_q: int = 4,
        seasonal_period: int = 4,
        seasonal: bool = True,
    ):
        """
        Args:
            target_col: Target variable column name.
            max_p: Maximum AR order to search.
            max_d: Maximum differencing order to search.
            max_q: Maximum MA order to search.
            seasonal_period: Seasonal period (4 for quarterly).
            seasonal: If True, fit SARIMA with seasonal terms.
        """
        self.target_col = target_col
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal_period = seasonal_period
        self.seasonal = seasonal

        self._result = None
        self._order = None
        self._seasonal_order = None
        self._series = None
        self._resid = None

    def _select_d(self, series: pd.Series) -> int:
        """Select differencing order using ADF test.

        Applies Δ iteratively (d-th order), NOT series.diff(d) which is the
        lag-d gap y_t - y_{t-d} and gives wrong answers for d>=2.
        """
        s = series
        for d in range(self.max_d + 1):
            clean = s.dropna()
            if len(clean) < 10:
                return d
            adf_stat, p_value, *_ = adfuller(clean, maxlag=8)
            if p_value < 0.05:
                return d
            s = s.diff(1)
        # If nothing became stationary within max_d, return the highest order tried.
        return self.max_d

    def _grid_search(self, series: pd.Series, d: int) -> tuple:
        """Find best (p, d, q) and seasonal order by AIC."""
        best_aic = float("inf")
        best_order = (1, d, 1)
        best_seasonal = (0, 0, 0, 0)

        # Determine seasonal candidates
        if self.seasonal:
            seasonal_candidates = [
                (0, 0, 0, self.seasonal_period),
                (1, 0, 0, self.seasonal_period),
                (1, 0, 1, self.seasonal_period),
                (0, 0, 1, self.seasonal_period),
            ]
        else:
            seasonal_candidates = [(0, 0, 0, 0)]

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip trivial model
                for seasonal_order in seasonal_candidates:
                    try:
                        model = ARIMA(
                            series,
                            order=(p, d, q),
                            seasonal_order=seasonal_order if seasonal_order != (0, 0, 0, 0) else None,
                        )
                        result = model.fit(method_kwargs={"maxiter": 200})
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)
                            best_seasonal = seasonal_order
                    except Exception:
                        continue

        return best_order, best_seasonal, best_aic

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit ARIMA with automatic order selection.

        Args:
            panel: Quarterly DataFrame. Only uses target_col.
        """
        self._series = panel[self.target_col].dropna()

        if len(self._series) < 16:
            raise ValueError(f"ARIMA needs at least 16 observations, got {len(self._series)}")

        # Select differencing order
        d = self._select_d(self._series)

        # Grid search for best (p, d, q) + seasonal
        self._order, self._seasonal_order, aic = self._grid_search(self._series, d)

        # Fit final model
        model = ARIMA(
            self._series,
            order=self._order,
            seasonal_order=self._seasonal_order if self._seasonal_order != (0, 0, 0, 0) else None,
        )
        self._result = model.fit(method_kwargs={"maxiter": 500})
        self._resid = self._result.resid

    def predict(self, horizon: int = 8, metro_id: str = "", vintage_date: str = "") -> ForecastResult:
        """Forecast with confidence intervals from the fitted ARIMA."""
        if self._result is None:
            raise RuntimeError("Must call fit() before predict()")

        forecast = self._result.get_forecast(steps=horizon)
        point = forecast.predicted_mean.values

        ci_80 = forecast.conf_int(alpha=0.20)
        ci_95 = forecast.conf_int(alpha=0.05)

        # Build period labels
        last_period = self._series.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=point,
            ci_80_lo=ci_80.iloc[:, 0].values,
            ci_80_hi=ci_80.iloc[:, 1].values,
            ci_95_lo=ci_95.iloc[:, 0].values,
            ci_95_hi=ci_95.iloc[:, 1].values,
            model_name="ARIMAEngine",
            metro_id=metro_id,
            vintage_date=vintage_date,
            horizon_quarters=horizon,
            forecast_periods=periods,
            residuals=self._resid.values,
        )

    def residuals(self) -> pd.Series:
        if self._resid is None:
            raise RuntimeError("Must call fit() before residuals()")
        return self._resid

    def model_name(self) -> str:
        return "ARIMAEngine"

    @property
    def order(self) -> tuple | None:
        return self._order

    @property
    def seasonal_order(self) -> tuple | None:
        return self._seasonal_order

    @property
    def aic(self) -> float | None:
        return self._result.aic if self._result else None

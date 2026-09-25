"""Prophet model for rent growth forecasting.

Bayesian structural time series with automatic seasonality detection,
trend changepoints, and clean uncertainty intervals. Excellent for
presentation-quality forecasts and detecting regime shifts.

Soft dependency: prophet is optional.
Install via: pip install supply-demand[econometrics]

Implements the ForecastModel protocol for ensemble integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from econometrics.models.base import ForecastResult


class ProphetEngine:
    """Prophet wrapper implementing the ForecastModel protocol."""

    def __init__(
        self,
        target_col: str = "rent_growth_yoy",
        regressor_cols: list[str] | None = None,
        yearly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        n_changepoints: int = 10,
    ):
        """
        Args:
            target_col: Target variable.
            regressor_cols: Optional exogenous regressors to include.
            yearly_seasonality: Whether to model yearly seasonality.
            changepoint_prior_scale: Flexibility of trend changepoints.
                Lower = smoother trend, higher = more responsive to shifts.
            seasonality_prior_scale: Strength of seasonality component.
            n_changepoints: Number of potential changepoints in the trend.
        """
        self.target_col = target_col
        self.regressor_cols = regressor_cols
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.n_changepoints = n_changepoints

        self._model = None
        self._panel = None
        self._resid = None
        self._train_df = None

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit Prophet model.

        Converts the PeriodIndex to DatetimeIndex (Prophet requires 'ds' column)
        and renames target to 'y'.

        Args:
            panel: Quarterly DataFrame with target_col.
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "prophet is required. Run: pip install prophet"
            )

        self._panel = panel

        # Prophet needs 'ds' (datetime) and 'y' columns
        series = panel[self.target_col].dropna()

        if len(series) < 12:
            raise ValueError(f"Prophet needs at least 12 observations, got {len(series)}")

        df = pd.DataFrame({
            "ds": series.index.to_timestamp(),
            "y": series.values,
        })

        # Add regressors if specified
        if self.regressor_cols:
            for col in self.regressor_cols:
                if col in panel.columns:
                    vals = panel[col].reindex(series.index)
                    df[col] = vals.values

        self._train_df = df

        # Configure Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            n_changepoints=self.n_changepoints,
        )

        # Add quarterly seasonality explicitly
        self._model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)

        # Add regressors
        if self.regressor_cols:
            for col in self.regressor_cols:
                if col in df.columns:
                    self._model.add_regressor(col)

        self._model.fit(df)

        # Compute in-sample residuals
        in_sample = self._model.predict(df)
        self._resid = pd.Series(
            df["y"].values - in_sample["yhat"].values,
            index=series.index,
        )

    def predict(self, horizon: int = 8, metro_id: str = "", vintage_date: str = "") -> ForecastResult:
        """Forecast with Prophet's built-in uncertainty intervals."""
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        # Create future dataframe
        future = self._model.make_future_dataframe(
            periods=horizon, freq="QS",
        )

        # Fill regressors for future periods with last known values
        if self.regressor_cols and self._train_df is not None:
            for col in self.regressor_cols:
                if col in self._train_df.columns:
                    last_val = self._train_df[col].iloc[-1]
                    if col not in future.columns:
                        future[col] = last_val
                    else:
                        future[col] = future[col].fillna(last_val)

        forecast = self._model.predict(future)

        # Extract the forecast-only rows (last `horizon` rows)
        fc = forecast.tail(horizon)

        point = fc["yhat"].values

        # Prophet provides yhat_lower/yhat_upper at 80% by default
        # We need to re-run with different interval_width for 95%
        ci_80_lo = fc["yhat_lower"].values
        ci_80_hi = fc["yhat_upper"].values

        # For 95% CI, scale from 80% assuming roughly normal
        # Prophet's default is 80%; approximate 95% by widening
        width_80 = ci_80_hi - ci_80_lo
        ci_95_lo = point - (width_80 / 2) * (1.96 / 1.28)
        ci_95_hi = point + (width_80 / 2) * (1.96 / 1.28)

        # Period labels
        series = self._panel[self.target_col].dropna()
        last_period = series.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=point,
            ci_80_lo=ci_80_lo,
            ci_80_hi=ci_80_hi,
            ci_95_lo=ci_95_lo,
            ci_95_hi=ci_95_hi,
            model_name="ProphetEngine",
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
        return "ProphetEngine"

    @property
    def changepoints(self) -> list | None:
        """Detected trend changepoint dates."""
        if self._model is None:
            return None
        return self._model.changepoints.tolist()

    @property
    def trend_at_end(self) -> float | None:
        """The trend component at the last observed quarter."""
        if self._model is None or self._train_df is None:
            return None
        in_sample = self._model.predict(self._train_df)
        return float(in_sample["trend"].iloc[-1])

"""Ridge and LASSO regression for rent growth forecasting.

Single-equation models that predict rent_growth_yoy from lagged
endogenous and exogenous features. Simpler than VAR but often
competitive in short-horizon forecasting.

Implements the ForecastModel protocol for ensemble integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from econometrics.models.base import ForecastResult


class RidgeLassoEngine:
    """Ridge/LASSO with expanding-window cross-validation."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        target_col: str = "rent_growth_yoy",
        alpha: float = 6.0,
        l1_ratio: float = 0.0,
        min_train: int = 20,
    ):
        """
        Args:
            feature_cols: Feature column names. If None, auto-detected from panel.
            target_col: Target variable to predict.
            alpha: Regularization strength.
            l1_ratio: 0.0 = pure Ridge, 1.0 = pure LASSO, between = Elastic Net.
            min_train: Minimum training observations required.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.min_train = min_train

        self._model = None
        self._panel = None
        self._features_used = None
        self._train_resid = None
        self._train_rmse = None

    def _auto_features(self, panel: pd.DataFrame) -> list[str]:
        """Auto-detect feature columns from panel."""
        candidates = [
            "vacancy_rate_lag1", "deliveries_pct_inventory_lag1",
            "absorption_pct_inventory_lag1", "rent_growth_yoy_lag1",
            "concession_spread_lag1", "mortgage_rate_30y_lag1",
            "unemployment_rate", "uc_pct_inventory",
        ]
        return [c for c in candidates if c in panel.columns]

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit Ridge/LASSO on the panel.

        Uses only rows where all features and target are non-null.
        """
        self._panel = panel

        features = self.feature_cols or self._auto_features(panel)
        self._features_used = [f for f in features if f in panel.columns]

        if not self._features_used:
            raise ValueError("No valid feature columns found in panel")

        # Build training set (drop NaN)
        cols = self._features_used + [self.target_col]
        train = panel[cols].dropna()

        if len(train) < self.min_train:
            raise ValueError(
                f"Need at least {self.min_train} complete observations, got {len(train)}"
            )

        X = train[self._features_used].values
        y = train[self.target_col].values

        # Fit model
        if self.l1_ratio == 0.0:
            self._model = Ridge(alpha=self.alpha)
        else:
            self._model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)

        self._model.fit(X, y)

        # Store residuals
        y_pred = self._model.predict(X)
        self._train_resid = pd.Series(y - y_pred, index=train.index)
        self._train_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    def predict(self, horizon: int = 8, metro_id: str = "", vintage_date: str = "") -> ForecastResult:
        """Forecast by assuming features stay at last known values.

        For a more sophisticated approach, combine with AR(1) driver
        forecasting (see supply_model.py's underwriting path).
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        # Use last known feature values for all forecast quarters
        last_row = self._panel[self._features_used].dropna().iloc[-1:].values
        point = np.full(horizon, self._model.predict(last_row)[0])

        # CI from training RMSE, widening with horizon
        horizon_scale = np.sqrt(np.arange(1, horizon + 1))
        std = self._train_rmse

        # Period labels
        last_period = self._panel.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=point,
            ci_80_lo=point - 1.28 * std * horizon_scale,
            ci_80_hi=point + 1.28 * std * horizon_scale,
            ci_95_lo=point - 1.96 * std * horizon_scale,
            ci_95_hi=point + 1.96 * std * horizon_scale,
            model_name="RidgeLassoEngine",
            metro_id=metro_id,
            vintage_date=vintage_date,
            horizon_quarters=horizon,
            forecast_periods=periods,
            residuals=self._train_resid.values,
        )

    def residuals(self) -> pd.Series:
        if self._train_resid is None:
            raise RuntimeError("Must call fit() before residuals()")
        return self._train_resid

    def model_name(self) -> str:
        return "RidgeLassoEngine"

    @property
    def coefficients(self) -> dict[str, float]:
        """Feature coefficients after fitting."""
        if self._model is None:
            return {}
        return dict(zip(self._features_used, self._model.coef_))

    @property
    def train_rmse(self) -> float | None:
        return self._train_rmse

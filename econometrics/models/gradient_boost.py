"""Gradient boosting model for rent growth forecasting.

Uses LightGBM if available, falls back to sklearn GradientBoostingRegressor.
Implements the ForecastModel protocol for ensemble integration.

Soft dependency: lightgbm is optional. Install via pip install supply-demand[econometrics].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from econometrics.models.base import ForecastResult


class GBMEngine:
    """Gradient boosting with soft LightGBM dependency."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        target_col: str = "rent_growth_yoy",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_train: int = 20,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_train = min_train

        self._model = None
        self._panel = None
        self._features_used = None
        self._train_resid = None
        self._train_rmse = None

    def _auto_features(self, panel: pd.DataFrame) -> list[str]:
        candidates = [
            "vacancy_rate_lag1", "deliveries_pct_inventory_lag1",
            "absorption_pct_inventory_lag1", "rent_growth_yoy_lag1",
            "concession_spread_lag1", "mortgage_rate_30y_lag1",
            "unemployment_rate", "uc_pct_inventory",
        ]
        return [c for c in candidates if c in panel.columns]

    def fit(self, panel: pd.DataFrame) -> None:
        self._panel = panel
        features = self.feature_cols or self._auto_features(panel)
        self._features_used = [f for f in features if f in panel.columns]

        if not self._features_used:
            raise ValueError("No valid feature columns found in panel")

        cols = self._features_used + [self.target_col]
        train = panel[cols].dropna()

        if len(train) < self.min_train:
            raise ValueError(f"Need at least {self.min_train} observations, got {len(train)}")

        X = train[self._features_used].values
        y = train[self.target_col].values

        # Try LightGBM first, fall back to sklearn
        try:
            import lightgbm as lgb

            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                verbose=-1,
            )
            self._model.fit(X, y)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            self._model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
            )
            self._model.fit(X, y)

        y_pred = self._model.predict(X)
        self._train_resid = pd.Series(y - y_pred, index=train.index)
        self._train_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    def predict(self, horizon: int = 8, metro_id: str = "", vintage_date: str = "") -> ForecastResult:
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        last_row = self._panel[self._features_used].dropna().iloc[-1:].values
        point = np.full(horizon, self._model.predict(last_row)[0])

        horizon_scale = np.sqrt(np.arange(1, horizon + 1))
        std = self._train_rmse

        last_period = self._panel.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=point,
            ci_80_lo=point - 1.28 * std * horizon_scale,
            ci_80_hi=point + 1.28 * std * horizon_scale,
            ci_95_lo=point - 1.96 * std * horizon_scale,
            ci_95_hi=point + 1.96 * std * horizon_scale,
            model_name="GBMEngine",
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
        return "GBMEngine"

    @property
    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(self._features_used, importances))

    @property
    def train_rmse(self) -> float | None:
        return self._train_rmse

"""Base protocol and result types for forecast models.

All econometric models implement the ForecastModel protocol. The ensemble
combiner only depends on this protocol, never on concrete model classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Standard forecast output from any model.

    All arrays are indexed by forecast horizon (0 = first forecast quarter).
    """

    point: np.ndarray               # Point forecast (rent_growth_yoy %)
    ci_80_lo: np.ndarray            # 80% CI lower bound
    ci_80_hi: np.ndarray            # 80% CI upper bound
    ci_95_lo: np.ndarray            # 95% CI lower bound
    ci_95_hi: np.ndarray            # 95% CI upper bound
    model_name: str                 # e.g., "VAREngine"
    metro_id: str                   # e.g., "dallas_tx"
    vintage_date: str               # e.g., "2026-Q1"
    horizon_quarters: int           # Number of forecast periods
    forecast_periods: list[str] = field(default_factory=list)  # ["2026-Q2", "2026-Q3", ...]
    residuals: np.ndarray | None = None  # In-sample residuals (for ensemble CI)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a DataFrame for reporting/export."""
        return pd.DataFrame({
            "period": self.forecast_periods,
            "rent_growth_yoy_point": self.point,
            "rent_growth_yoy_ci80_lo": self.ci_80_lo,
            "rent_growth_yoy_ci80_hi": self.ci_80_hi,
            "rent_growth_yoy_ci95_lo": self.ci_95_lo,
            "rent_growth_yoy_ci95_hi": self.ci_95_hi,
            "model_name": self.model_name,
            "vintage_date": self.vintage_date,
        })


@dataclass
class IRFResult:
    """Impulse response function output from VAR models."""

    periods: int                    # Number of IRF periods
    impulse_var: str                # Variable that was shocked
    response_var: str               # Variable whose response is measured
    irf_values: np.ndarray          # Response values per period
    ci_lo: np.ndarray | None = None
    ci_hi: np.ndarray | None = None


@runtime_checkable
class ForecastModel(Protocol):
    """Protocol that all econometric models must implement.

    The ensemble combiner depends only on this interface.
    """

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit the model on a quarterly panel DataFrame.

        The panel has a DatetimeIndex or PeriodIndex at quarterly frequency.
        Columns include endogenous vars (rent_growth_yoy, vacancy_rate, etc.)
        and optionally exogenous vars (mortgage_rate_30y, etc.).
        """
        ...

    def predict(self, horizon: int) -> ForecastResult:
        """Produce a forecast for `horizon` quarters ahead.

        Returns a ForecastResult with point estimate and confidence intervals.
        """
        ...

    def residuals(self) -> pd.Series:
        """Return in-sample residuals for the rent_growth_yoy equation.

        Used by the ensemble confidence module for empirical CI construction.
        """
        ...

    def model_name(self) -> str:
        """Return the model's display name (e.g., 'VAREngine')."""
        ...

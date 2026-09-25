"""Unit tests for ARIMA differencing order selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from econometrics.models.arima_engine import ARIMAEngine


def test_select_d_applies_iterative_differencing_not_lag_gap():
    """_select_d must apply Δ iteratively, not compute y_t - y_{t-d}."""
    rng = np.random.default_rng(7)
    n = 120
    eps = rng.normal(0, 1.0, n)
    y1 = np.cumsum(eps)
    y2 = np.cumsum(y1)
    series = pd.Series(y2, index=pd.period_range("2000Q1", periods=n, freq="Q"))

    engine = ARIMAEngine(max_d=2)
    d = engine._select_d(series)

    assert d == 2


def test_select_d_equals_zero_for_stationary_series():
    rng = np.random.default_rng(3)
    n = 120
    series = pd.Series(rng.normal(0, 1.0, n), index=pd.period_range("2000Q1", periods=n, freq="Q"))
    engine = ARIMAEngine(max_d=2)
    assert engine._select_d(series) == 0

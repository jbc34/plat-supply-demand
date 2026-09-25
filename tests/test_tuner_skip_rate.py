"""Unit test: tuner must penalize configs that fail most backtest folds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from econometrics.tuning import bvar_tuner


class _AlwaysFailEngine:
    def __init__(self, **kwargs): pass
    def fit(self, panel): raise RuntimeError("intentional fail")
    def predict(self, **kwargs): raise RuntimeError("intentional fail")


def _panel() -> pd.DataFrame:
    idx = pd.period_range("2000Q1", periods=80, freq="Q")
    return pd.DataFrame(
        {
            "rent_growth_yoy": np.linspace(0.01, 0.04, 80),
            "vacancy_rate": np.linspace(0.05, 0.08, 80),
            "deliveries_pct_inventory": np.linspace(0.005, 0.015, 80),
            "absorption_pct_inventory": np.linspace(0.006, 0.014, 80),
        },
        index=idx,
    )


def test_backtest_returns_inf_when_all_folds_fail(monkeypatch):
    from econometrics.models import var_engine as var_engine_mod
    monkeypatch.setattr(var_engine_mod, "VAREngine", _AlwaysFailEngine)

    rmse = bvar_tuner._backtest_bvar(
        panel=_panel(),
        horizon=4,
        lam=0.2,
        cross_shrink=0.5,
        ic="aic",
        exog_cols=[],
        min_train=40,
        step_quarters=2,
    )
    assert rmse == float("inf")


def test_backtest_returns_inf_when_skip_rate_exceeds_half(monkeypatch):
    """Even if a few folds succeed, >50% fold failure = infinite loss."""
    from econometrics.models import var_engine as var_engine_mod
    real_engine_cls = var_engine_mod.VAREngine

    class _MostlyFailEngine(real_engine_cls):
        _call_count = [0]

        def fit(self, panel):
            _MostlyFailEngine._call_count[0] += 1
            if _MostlyFailEngine._call_count[0] % 4 != 0:
                raise RuntimeError("induced skip")
            super().fit(panel)

    monkeypatch.setattr(var_engine_mod, "VAREngine", _MostlyFailEngine)

    rmse = bvar_tuner._backtest_bvar(
        panel=_panel(),
        horizon=4,
        lam=0.2,
        cross_shrink=0.5,
        ic="aic",
        exog_cols=[],
        min_train=40,
        step_quarters=2,
    )
    assert rmse == float("inf"), f"Expected inf for >50% skip, got {rmse}"

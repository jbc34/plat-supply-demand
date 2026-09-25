"""Tests for Phases 4-7: backtest, ensemble, scenarios, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ─── Backtest Metrics ────────────────────────────────────────────────


class TestMetrics:

    def test_rmse(self):
        from econometrics.backtest.metrics import rmse
        a = np.array([1.0, 2.0, 3.0])
        f = np.array([1.1, 2.2, 2.8])
        assert 0 < rmse(a, f) < 0.25

    def test_mae(self):
        from econometrics.backtest.metrics import mae
        a = np.array([1.0, 2.0, 3.0])
        f = np.array([1.1, 2.2, 2.8])
        assert abs(mae(a, f) - 0.1667) < 0.01

    def test_directional_accuracy_perfect(self):
        from econometrics.backtest.metrics import directional_accuracy
        a = np.array([0.01, -0.02, 0.03])
        f = np.array([0.005, -0.01, 0.02])
        assert directional_accuracy(a, f) == 1.0

    def test_ci_coverage(self):
        from econometrics.backtest.metrics import ci_coverage
        a = np.array([1.0, 2.0, 3.0, 10.0])
        lo = np.array([0.5, 1.5, 2.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5, 3.5])
        assert ci_coverage(a, lo, hi) == 0.75  # 3 out of 4 inside

    def test_nan_handling(self):
        from econometrics.backtest.metrics import rmse
        a = np.array([1.0, np.nan, 3.0])
        f = np.array([1.1, 2.0, 2.8])
        result = rmse(a, f)
        assert not np.isnan(result)  # Should skip NaN pair


# ─── Backtest Registry ───────────────────────────────────────────────


class TestRegistry:

    def test_append_and_load(self, synthetic_panel, tmp_path, monkeypatch):
        from econometrics.backtest import registry
        from econometrics.models.base import ForecastResult

        # Redirect registry to tmp dir
        monkeypatch.setattr("econometrics.backtest.registry._registry_path",
                            lambda mid: tmp_path / f"{mid}_backtest.json")

        result = ForecastResult(
            point=np.array([0.02, 0.015]),
            ci_80_lo=np.array([0.01, 0.005]),
            ci_80_hi=np.array([0.03, 0.025]),
            ci_95_lo=np.array([0.0, -0.005]),
            ci_95_hi=np.array([0.04, 0.035]),
            model_name="VAREngine",
            metro_id="test_metro",
            vintage_date="2024Q4",
            horizon_quarters=2,
            forecast_periods=["2025Q1", "2025Q2"],
        )

        run_id = registry.append(result)
        assert run_id is not None

        entries = registry.load_all("test_metro")
        assert len(entries) == 1
        assert entries[0]["model_name"] == "VAREngine"
        assert entries[0]["actuals"] == [None, None]

    def test_fill_actuals(self, tmp_path, monkeypatch):
        from econometrics.backtest import registry
        from econometrics.models.base import ForecastResult

        monkeypatch.setattr("econometrics.backtest.registry._registry_path",
                            lambda mid: tmp_path / f"{mid}_backtest.json")

        result = ForecastResult(
            point=np.array([0.02, 0.015]),
            ci_80_lo=np.array([0.01, 0.005]),
            ci_80_hi=np.array([0.03, 0.025]),
            ci_95_lo=np.array([0.0, -0.005]),
            ci_95_hi=np.array([0.04, 0.035]),
            model_name="TestModel",
            metro_id="test_metro",
            vintage_date="2024Q4",
            horizon_quarters=2,
            forecast_periods=["2025Q1", "2025Q2"],
        )
        registry.append(result)

        n = registry.fill_actuals("test_metro", {"2025Q1": 0.018, "2025Q2": 0.012})
        assert n == 1

        entries = registry.load_all("test_metro")
        assert entries[0]["actuals"] == [0.018, 0.012]
        assert entries[0]["rmse"] is not None


# ─── Ensemble ────────────────────────────────────────────────────────


class TestEnsemble:

    def test_equal_weights_cold_start(self):
        from econometrics.ensemble.weighter import compute_weights
        weights = compute_weights(
            accuracy_by_model={},
            enabled_models=["VAREngine", "ECMEngine", "RidgeLassoEngine"],
        )
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert all(abs(w - 1 / 3) < 1e-10 for w in weights.values())

    def test_inverse_rmse_weighting(self):
        from econometrics.ensemble.weighter import compute_weights
        accuracy = {
            "ModelA": {"mean_rmse": 0.01, "n_completed": 5},
            "ModelB": {"mean_rmse": 0.02, "n_completed": 5},
        }
        weights = compute_weights(accuracy, enabled_models=["ModelA", "ModelB"])
        assert weights["ModelA"] > weights["ModelB"]  # Lower RMSE → higher weight
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_combiner_produces_forecast_result(self, synthetic_panel):
        from econometrics.ensemble.combiner import combine
        from econometrics.models.base import ForecastResult
        from econometrics.models.ridge_lasso import RidgeLassoEngine
        from econometrics.models.var_engine import VAREngine

        var = VAREngine()
        var.fit(synthetic_panel)
        r1 = var.predict(horizon=4, metro_id="test", vintage_date="2024Q4")

        ridge = RidgeLassoEngine()
        ridge.fit(synthetic_panel)
        r2 = ridge.predict(horizon=4, metro_id="test", vintage_date="2024Q4")

        weights = {"VAREngine": 0.6, "RidgeLassoEngine": 0.4}
        ensemble = combine([r1, r2], weights)

        assert isinstance(ensemble, ForecastResult)
        assert len(ensemble.point) == 4
        assert "Ensemble" in ensemble.model_name

    def test_empirical_ci(self, synthetic_panel):
        from econometrics.ensemble.confidence import empirical_ci
        from econometrics.models.ridge_lasso import RidgeLassoEngine
        from econometrics.models.var_engine import VAREngine

        var = VAREngine()
        var.fit(synthetic_panel)
        r1 = var.predict(horizon=4, metro_id="test", vintage_date="2024Q4")

        ridge = RidgeLassoEngine()
        ridge.fit(synthetic_panel)
        r2 = ridge.predict(horizon=4, metro_id="test", vintage_date="2024Q4")

        ci = empirical_ci([r1, r2], {"VAREngine": 0.5, "RidgeLassoEngine": 0.5})
        assert "ci_80_lo" in ci
        assert "ci_95_lo" in ci
        # 95% band should be wider than 80%
        width_95 = ci["ci_95_hi"] - ci["ci_95_lo"]
        width_80 = ci["ci_80_hi"] - ci["ci_80_lo"]
        assert (width_95 >= width_80 - 1e-6).all()


# ─── Scenarios ───────────────────────────────────────────────────────


class TestScenarios:

    def test_shock_library_has_entries(self):
        from econometrics.scenarios.shock_library import SHOCK_REGISTRY
        assert len(SHOCK_REGISTRY) >= 4
        assert "GFC_2008" in SHOCK_REGISTRY
        assert "COVID_2020" in SHOCK_REGISTRY

    def test_apply_shock_modifies_panel(self, synthetic_panel):
        from econometrics.scenarios.shock_library import GFC_2008
        from econometrics.scenarios.stress_tester import apply_shock

        shocked = apply_shock(synthetic_panel, GFC_2008)
        # Unemployment should be higher in shocked version
        orig_unemp = synthetic_panel["unemployment_rate"].iloc[-3:]
        shock_unemp = shocked["unemployment_rate"].iloc[-3:]
        assert (shock_unemp >= orig_unemp).all()

    def test_stress_test_produces_result(self, synthetic_panel):
        from econometrics.models.ridge_lasso import RidgeLassoEngine
        from econometrics.scenarios.shock_library import GFC_2008
        from econometrics.scenarios.stress_tester import StressTestResult, run_stress_test

        result = run_stress_test(
            synthetic_panel, GFC_2008, RidgeLassoEngine,
            horizon=4, metro_id="test",
        )
        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "GFC_2008"
        assert len(result.delta_point) == 4


# ─── Reports ─────────────────────────────────────────────────────────


class TestReports:

    def test_markdown_writer(self, synthetic_panel):
        from econometrics.models.ridge_lasso import RidgeLassoEngine
        from econometrics.models.var_engine import VAREngine
        from econometrics.reports.markdown_writer import render_forecast_report

        var = VAREngine()
        var.fit(synthetic_panel)
        r1 = var.predict(horizon=4, metro_id="test", vintage_date="2024Q4")

        ridge = RidgeLassoEngine()
        ridge.fit(synthetic_panel)
        r2 = ridge.predict(horizon=4, metro_id="test", vintage_date="2024Q4")

        from econometrics.ensemble.combiner import combine
        weights = {"VAREngine": 0.6, "RidgeLassoEngine": 0.4}
        ensemble = combine([r1, r2], weights)

        config = {"display_name": "Test Metro", "metro_id": "test"}
        md = render_forecast_report(ensemble, [r1, r2], weights, config)

        assert "## Econometric Rent Growth Forecast" in md
        assert "VAREngine" in md
        assert "RidgeLassoEngine" in md
        assert "| Period |" in md

    def test_fan_chart(self, synthetic_panel, tmp_path):
        from econometrics.models.var_engine import VAREngine
        from econometrics.reports.chart_builder import fan_chart

        engine = VAREngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        history = synthetic_panel["rent_growth_yoy"].tail(12)
        path = tmp_path / "test_fan.png"
        fig, ax = fan_chart(result, history=history, title="Test Fan Chart", save_path=path)
        assert path.exists()
        assert path.stat().st_size > 1000
        import matplotlib.pyplot as plt
        plt.close(fig)

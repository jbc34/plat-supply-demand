"""Tests for Phase 3 econometric models.

All tests use the synthetic_panel fixture from conftest.py (60 quarters,
deterministic seed, known VAR(1) DGP).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── STL Decomposition ──────────────────────────────────────────────


class TestSTLDecomp:

    def test_decompose_returns_components(self, synthetic_panel):
        from econometrics.models.stl_decomp import decompose

        result = decompose(synthetic_panel["rent_growth_yoy"])
        assert result.trend is not None
        assert result.seasonal is not None
        assert result.residual is not None
        assert len(result.trend) == len(result.original)

    def test_components_sum_to_original(self, synthetic_panel):
        from econometrics.models.stl_decomp import decompose

        result = decompose(synthetic_panel["rent_growth_yoy"])
        reconstructed = result.trend + result.seasonal + result.residual
        np.testing.assert_allclose(
            reconstructed.values, result.original.values, atol=1e-10,
        )

    def test_short_series_returns_original_as_trend(self):
        from econometrics.models.stl_decomp import decompose

        short = pd.Series([0.03, 0.02, 0.04], index=pd.period_range("2020Q1", periods=3, freq="Q"))
        result = decompose(short, period=4)
        pd.testing.assert_series_equal(result.trend, short)
        assert (result.seasonal == 0).all()

    def test_decompose_panel(self, synthetic_panel):
        from econometrics.models.stl_decomp import decompose_panel

        deseasonalized, components = decompose_panel(synthetic_panel)
        assert "rent_growth_yoy" in components
        assert "vacancy_rate" in components
        # Deseasonalized should be trend (different from original)
        assert not deseasonalized["rent_growth_yoy"].equals(synthetic_panel["rent_growth_yoy"])


# ─── VAR Engine ──────────────────────────────────────────────────────


class TestVAREngine:

    def test_fit_selects_lag(self, synthetic_panel):
        from econometrics.models.var_engine import VAREngine

        engine = VAREngine()
        engine.fit(synthetic_panel)
        assert engine.selected_lag is not None
        assert engine.selected_lag >= 1

    def test_predict_returns_forecast_result(self, synthetic_panel):
        from econometrics.models.var_engine import VAREngine
        from econometrics.models.base import ForecastResult

        engine = VAREngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4, metro_id="test", vintage_date="2024-Q4")

        assert isinstance(result, ForecastResult)
        assert len(result.point) == 4
        assert len(result.ci_80_lo) == 4
        assert len(result.ci_95_lo) == 4
        assert len(result.forecast_periods) == 4
        assert result.model_name == "VAREngine"

    def test_ci_bounds_bracket_point(self, synthetic_panel):
        from econometrics.models.var_engine import VAREngine

        engine = VAREngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        assert (result.ci_95_lo <= result.point).all()
        assert (result.ci_95_hi >= result.point).all()
        assert (result.ci_80_lo <= result.point).all()
        assert (result.ci_80_hi >= result.point).all()

    def test_95_ci_wider_than_80(self, synthetic_panel):
        from econometrics.models.var_engine import VAREngine

        engine = VAREngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        width_95 = result.ci_95_hi - result.ci_95_lo
        width_80 = result.ci_80_hi - result.ci_80_lo
        assert (width_95 >= width_80 - 1e-10).all()

    def test_residuals_returned(self, synthetic_panel):
        from econometrics.models.var_engine import VAREngine

        engine = VAREngine()
        engine.fit(synthetic_panel)
        resid = engine.residuals()
        assert len(resid) > 0

    def test_irf_returns_results(self, synthetic_panel):
        from econometrics.models.var_engine import VAREngine

        engine = VAREngine()
        engine.fit(synthetic_panel)
        irfs = engine.irf(periods=12)

        assert len(irfs) == 4  # One per endogenous variable
        for irf in irfs:
            assert irf.response_var == "rent_growth_yoy"
            assert len(irf.irf_values) == 13  # periods + 1 (includes t=0)

    def test_too_few_observations_raises(self, synthetic_panel_short):
        from econometrics.models.var_engine import VAREngine

        # 20 observations is the minimum; this fixture has exactly 20
        engine = VAREngine()
        engine.fit(synthetic_panel_short)  # Should just barely work

    def test_fewer_than_20_raises(self):
        from econometrics.models.var_engine import VAREngine

        tiny = pd.DataFrame({
            "rent_growth_yoy": [0.03] * 10,
            "vacancy_rate": [0.06] * 10,
            "deliveries_pct_inventory": [0.02] * 10,
            "absorption_pct_inventory": [0.025] * 10,
        }, index=pd.period_range("2020Q1", periods=10, freq="Q"))

        engine = VAREngine()
        with pytest.raises(ValueError, match="at least 20"):
            engine.fit(tiny)


# ─── ECM Engine ──────────────────────────────────────────────────────


class TestECMEngine:

    def test_fit_runs_johansen_test(self, synthetic_panel):
        from econometrics.models.ecm_engine import ECMEngine

        engine = ECMEngine()
        engine.fit(synthetic_panel)
        assert engine.cointegration_test is not None
        assert isinstance(engine.cointegration_test.trace_stat, list)

    def test_predict_returns_forecast_result(self, synthetic_panel):
        from econometrics.models.ecm_engine import ECMEngine
        from econometrics.models.base import ForecastResult

        engine = ECMEngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        assert isinstance(result, ForecastResult)
        assert len(result.point) == 4
        assert result.model_name == "ECMEngine"

    def test_ci_bounds_bracket_point(self, synthetic_panel):
        from econometrics.models.ecm_engine import ECMEngine

        engine = ECMEngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        assert (result.ci_95_lo <= result.point + 1e-10).all()
        assert (result.ci_95_hi >= result.point - 1e-10).all()

    def test_residuals_returned(self, synthetic_panel):
        from econometrics.models.ecm_engine import ECMEngine

        engine = ECMEngine()
        engine.fit(synthetic_panel)
        resid = engine.residuals()
        assert len(resid) > 0

    def test_fallback_on_short_series(self, synthetic_panel_short):
        """Short series likely won't cointegrate — should fall back to VAR-in-diff."""
        from econometrics.models.ecm_engine import ECMEngine

        engine = ECMEngine()
        engine.fit(synthetic_panel_short)
        # Should still produce a forecast regardless of cointegration
        result = engine.predict(horizon=4)
        assert len(result.point) == 4

    def test_missing_columns_raises(self):
        from econometrics.models.ecm_engine import ECMEngine

        panel = pd.DataFrame({"vacancy_rate": [0.06] * 30},
                             index=pd.period_range("2017Q1", periods=30, freq="Q"))
        engine = ECMEngine()
        with pytest.raises(ValueError, match="effective_rent_unit"):
            engine.fit(panel)


# ─── Ridge/LASSO Engine ─────────────────────────────────────────────


class TestRidgeLassoEngine:

    def test_fit_and_predict(self, synthetic_panel):
        from econometrics.models.ridge_lasso import RidgeLassoEngine
        from econometrics.models.base import ForecastResult

        engine = RidgeLassoEngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        assert isinstance(result, ForecastResult)
        assert len(result.point) == 4
        assert result.model_name == "RidgeLassoEngine"

    def test_coefficients_exist(self, synthetic_panel):
        from econometrics.models.ridge_lasso import RidgeLassoEngine

        engine = RidgeLassoEngine()
        engine.fit(synthetic_panel)
        coef = engine.coefficients
        assert len(coef) > 0
        # Vacancy lag should have a negative coefficient (higher vacancy → lower rent growth)
        if "vacancy_rate_lag1" in coef:
            assert coef["vacancy_rate_lag1"] < 0

    def test_lasso_mode(self, synthetic_panel):
        from econometrics.models.ridge_lasso import RidgeLassoEngine

        engine = RidgeLassoEngine(alpha=0.001, l1_ratio=1.0)
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)
        assert len(result.point) == 4

    def test_residuals(self, synthetic_panel):
        from econometrics.models.ridge_lasso import RidgeLassoEngine

        engine = RidgeLassoEngine()
        engine.fit(synthetic_panel)
        resid = engine.residuals()
        assert len(resid) > 0
        assert engine.train_rmse > 0


# ─── GBM Engine ──────────────────────────────────────────────────────


class TestGBMEngine:

    def test_fit_and_predict(self, synthetic_panel):
        from econometrics.models.gradient_boost import GBMEngine
        from econometrics.models.base import ForecastResult

        engine = GBMEngine()
        engine.fit(synthetic_panel)
        result = engine.predict(horizon=4)

        assert isinstance(result, ForecastResult)
        assert len(result.point) == 4
        assert result.model_name == "GBMEngine"

    def test_feature_importance(self, synthetic_panel):
        from econometrics.models.gradient_boost import GBMEngine

        engine = GBMEngine()
        engine.fit(synthetic_panel)
        importance = engine.feature_importance
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

    def test_residuals(self, synthetic_panel):
        from econometrics.models.gradient_boost import GBMEngine

        engine = GBMEngine()
        engine.fit(synthetic_panel)
        resid = engine.residuals()
        assert len(resid) > 0
        assert engine.train_rmse > 0

    def test_gbm_overfits_less_than_zero_rmse(self, synthetic_panel):
        """GBM shouldn't have zero RMSE on training data (regularization)."""
        from econometrics.models.gradient_boost import GBMEngine

        engine = GBMEngine(n_estimators=10, max_depth=2, learning_rate=0.05)
        engine.fit(synthetic_panel)
        assert engine.train_rmse > 0.001


# ─── Protocol Conformance ────────────────────────────────────────────


class TestProtocolConformance:
    """All models should implement the ForecastModel protocol."""

    @pytest.mark.parametrize("engine_cls_path", [
        "econometrics.models.var_engine.VAREngine",
        "econometrics.models.ecm_engine.ECMEngine",
        "econometrics.models.ridge_lasso.RidgeLassoEngine",
        "econometrics.models.gradient_boost.GBMEngine",
    ])
    def test_implements_protocol(self, engine_cls_path, synthetic_panel):
        from econometrics.models.base import ForecastModel

        parts = engine_cls_path.rsplit(".", 1)
        module = __import__(parts[0], fromlist=[parts[1]])
        cls = getattr(module, parts[1])

        engine = cls()
        engine.fit(synthetic_panel)

        # Protocol checks
        assert hasattr(engine, "fit")
        assert hasattr(engine, "predict")
        assert hasattr(engine, "residuals")
        assert hasattr(engine, "model_name")

        result = engine.predict(horizon=4)
        assert len(result.point) == 4

        resid = engine.residuals()
        assert len(resid) > 0

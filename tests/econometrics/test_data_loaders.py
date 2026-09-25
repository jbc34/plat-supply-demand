"""Tests for Phase 2 data loaders and transforms.

All tests use local data or synthetic fixtures — no network calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestCoStarLoader:
    """Test CoStar CSV and parquet loaders."""

    @pytest.fixture
    def dallas_csv_path(self):
        path = PROJECT_ROOT / "data" / "costar" / "processed" / "dallas_multifamily.csv"
        if not path.exists():
            pytest.skip(f"CoStar CSV not found: {path}")
        return path

    def test_load_csv_returns_period_index(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path, start_date="2000-01-01")
        assert isinstance(panel.index, pd.PeriodIndex)
        assert panel.index.freq == "Q-DEC" or str(panel.index.freqstr).startswith("Q")

    def test_load_csv_canonical_columns(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path, start_date="2000-01-01")
        # Must have the 4 endogenous vars
        assert "rent_growth_yoy" in panel.columns
        assert "vacancy_rate" in panel.columns
        assert "deliveries_pct_inventory" in panel.columns
        assert "absorption_pct_inventory" in panel.columns

    def test_load_csv_derived_columns(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path, start_date="2000-01-01")
        assert "concession_spread" in panel.columns
        assert "uc_pct_inventory" in panel.columns

    def test_load_csv_date_filter(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path, start_date="2015-01-01")
        assert panel.index.min() >= pd.Period("2015Q1")

    def test_load_csv_no_duplicates(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path)
        assert not panel.index.duplicated().any()

    def test_vacancy_rate_range(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path, start_date="2000-01-01")
        vac = panel["vacancy_rate"].dropna()
        assert vac.min() >= 0
        assert vac.max() <= 0.30  # No DFW submarket has ever hit 30%

    def test_rent_growth_range(self, dallas_csv_path):
        from econometrics.data.loaders.costar_panel import load_costar_csv

        panel = load_costar_csv(dallas_csv_path, start_date="2000-01-01")
        rg = panel["rent_growth_yoy"].dropna()
        assert rg.min() > -0.20  # No 20%+ rent decline in DFW history
        assert rg.max() < 0.25   # No 25%+ rent spike


class TestConcessions:
    """Test concession spread computation."""

    def test_spread_computed_correctly(self):
        from econometrics.data.loaders.concessions import compute_concession_features

        panel = pd.DataFrame({
            "asking_rent_unit": [1200.0, 1250.0, 1300.0, 1350.0, 1400.0],
            "effective_rent_unit": [1150.0, 1200.0, 1250.0, 1250.0, 1300.0],
        })
        result = compute_concession_features(panel)
        spread = result["concession_spread"]
        # (1200-1150)/1200 = 0.04167
        assert abs(spread.iloc[0] - 0.04167) < 0.001

    def test_yoy_change_requires_4_periods(self):
        from econometrics.data.loaders.concessions import compute_concession_features

        panel = pd.DataFrame({
            "asking_rent_unit": [1200.0] * 6,
            "effective_rent_unit": [1150.0, 1150.0, 1150.0, 1150.0, 1100.0, 1100.0],
        })
        result = compute_concession_features(panel)
        # First 4 should be NaN (not enough history for YoY)
        assert result["concession_spread_chg_yoy"].iloc[:4].isna().all()

    def test_missing_columns_returns_unchanged(self):
        from econometrics.data.loaders.concessions import compute_concession_features

        panel = pd.DataFrame({"vacancy_rate": [0.05, 0.06]})
        result = compute_concession_features(panel)
        assert "concession_spread" not in result.columns


class TestSpatialWeights:
    """Test spatial weights matrix construction."""

    def test_row_normalization(self, dallas_config):
        from econometrics.data.transforms.spatial_weights import build_inverse_distance_weights

        W, ids = build_inverse_distance_weights(dallas_config["submarkets"])
        row_sums = W.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_zero_diagonal(self, dallas_config):
        from econometrics.data.transforms.spatial_weights import build_inverse_distance_weights

        W, ids = build_inverse_distance_weights(dallas_config["submarkets"])
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_correct_dimensions(self, dallas_config):
        from econometrics.data.transforms.spatial_weights import build_inverse_distance_weights

        n = len(dallas_config["submarkets"])
        W, ids = build_inverse_distance_weights(dallas_config["submarkets"])
        assert W.shape == (n, n)
        assert len(ids) == n

    def test_fewer_than_2_submarkets_raises(self):
        from econometrics.data.transforms.spatial_weights import build_inverse_distance_weights

        with pytest.raises(ValueError, match="at least 2"):
            build_inverse_distance_weights([{"id": "a", "lat": 32.0, "lon": -96.0}])

    def test_spatial_lag_computation(self, dallas_config):
        from econometrics.data.transforms.spatial_weights import (
            build_inverse_distance_weights,
            spatial_lag,
        )

        W, ids = build_inverse_distance_weights(dallas_config["submarkets"])
        y = np.array([3.0, 2.0, 1.0, 4.0])
        Wy = spatial_lag(W, y)
        assert Wy.shape == (4,)
        # Spatial lag should be a weighted average of neighbors
        # For each i, Wy[i] != y[i] (since W has zero diagonal)
        assert not np.allclose(Wy, y)


class TestPermitLag:
    """Test permit-to-delivery lag calibration."""

    def test_perfect_lag_detected(self):
        from econometrics.data.transforms.permit_lag import calibrate_permit_lag

        rng = np.random.default_rng(42)
        idx = pd.period_range("2005Q1", periods=80, freq="Q")
        permits = pd.Series(500 + 100 * np.sin(np.arange(80) * 2 * np.pi / 24), index=idx)
        permits += rng.normal(0, 10, 80)
        # Deliveries at time t = permits at time t-6 (i.e., 18 months earlier) + noise
        deliveries = permits.shift(6) + rng.normal(0, 10, 80)
        # Drop NaN from the shift
        mask = deliveries.notna()
        permits = permits[mask]
        deliveries = deliveries[mask]

        result = calibrate_permit_lag(permits, deliveries)
        assert result["optimal_lag_months"] == 18
        assert result["correlation"] > 0.7
        assert result["valid"]

    def test_no_signal_returns_invalid(self):
        from econometrics.data.transforms.permit_lag import calibrate_permit_lag

        rng = np.random.default_rng(42)
        idx = pd.period_range("2010Q1", periods=60, freq="Q")
        permits = pd.Series(rng.normal(500, 50, 60), index=idx)
        deliveries = pd.Series(rng.normal(400, 40, 60), index=idx)  # Unrelated

        result = calibrate_permit_lag(permits, deliveries)
        # Correlation should be low for unrelated series
        assert result["correlation"] < 0.5 or not result["valid"]

    def test_short_series_returns_default(self):
        from econometrics.data.transforms.permit_lag import calibrate_permit_lag

        idx = pd.period_range("2020Q1", periods=5, freq="Q")
        permits = pd.Series([100, 110, 120, 115, 125], index=idx)
        deliveries = pd.Series([90, 95, 100, 105, 110], index=idx)

        result = calibrate_permit_lag(permits, deliveries, min_overlap=12)
        # Not enough overlap → fallback
        assert result["optimal_lag_months"] == 18
        assert result["valid"] is False


class TestSyntheticPanel:
    """Test that the conftest synthetic panel has expected properties."""

    def test_panel_shape(self, synthetic_panel):
        assert len(synthetic_panel) == 60
        assert isinstance(synthetic_panel.index, pd.PeriodIndex)

    def test_endogenous_vars_present(self, synthetic_panel):
        for col in ["rent_growth_yoy", "vacancy_rate",
                     "deliveries_pct_inventory", "absorption_pct_inventory"]:
            assert col in synthetic_panel.columns
            assert synthetic_panel[col].notna().all()

    def test_exogenous_vars_present(self, synthetic_panel):
        for col in ["mortgage_rate_30y", "unemployment_rate", "concession_spread"]:
            assert col in synthetic_panel.columns

    def test_lagged_vars_present(self, synthetic_panel):
        assert "rent_growth_yoy_lag1" in synthetic_panel.columns
        assert "vacancy_rate_lag1" in synthetic_panel.columns
        # Lag1 should have one NaN (first row)
        assert synthetic_panel["rent_growth_yoy_lag1"].isna().sum() == 1

    def test_vacancy_mean_reverts(self, synthetic_panel):
        """Vacancy should oscillate around 6% (the DGP target)."""
        vac = synthetic_panel["vacancy_rate"]
        assert abs(vac.mean() - 0.06) < 0.02

    def test_rent_income_for_ecm(self, synthetic_panel):
        """Panel should have rent and income levels for cointegration testing."""
        assert "effective_rent_unit" in synthetic_panel.columns
        assert "median_hh_income" in synthetic_panel.columns
        assert synthetic_panel["effective_rent_unit"].iloc[-1] > synthetic_panel["effective_rent_unit"].iloc[0]

    def test_short_panel_is_subset(self, synthetic_panel, synthetic_panel_short):
        assert len(synthetic_panel_short) == 20
        pd.testing.assert_frame_equal(
            synthetic_panel_short, synthetic_panel.iloc[:20]
        )

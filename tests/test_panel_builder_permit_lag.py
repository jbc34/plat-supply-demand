"""Integration test: permit lag must actually apply to the panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econometrics.data.transforms.panel_builder import build_panel


def _fake_costar() -> pd.DataFrame:
    idx = pd.period_range("2015Q1", periods=40, freq="Q")
    return pd.DataFrame(
        {
            "rent_growth_yoy": np.linspace(0.02, 0.04, 40),
            "vacancy_rate": np.linspace(0.05, 0.08, 40),
            "deliveries_pct_inventory": np.linspace(0.005, 0.015, 40),
            "absorption_pct_inventory": np.linspace(0.006, 0.014, 40),
        },
        index=idx,
    )


def _patch_loaders(monkeypatch, permits: pd.DataFrame) -> None:
    from econometrics.data.loaders import costar_panel, census_permits, concessions

    monkeypatch.setattr(costar_panel, "load_costar", lambda config, source="csv": _fake_costar())
    monkeypatch.setattr(
        concessions, "compute_concession_features",
        lambda panel: panel.assign(concession_spread=0.0),
    )
    monkeypatch.setattr(census_permits, "load_permits", lambda config: permits.copy())


def test_permit_pipeline_is_actually_shifted(monkeypatch):
    """permit_pipeline must be shifted, not silently NaN'd by index mismatch."""
    # Fake permits with DatetimeIndex (triggers the alignment bug surface)
    permits = pd.DataFrame(
        {"permits_5plus": np.arange(40, dtype=float) * 100},
        index=pd.period_range("2015Q1", periods=40, freq="Q").to_timestamp(),
    )
    _patch_loaders(monkeypatch, permits)

    config = {
        "metro_id": "test_metro",
        "display_name": "Test",
        "endogenous_vars": ["rent_growth_yoy", "vacancy_rate"],
        "permit_lag_months": 18,
    }
    panel = build_panel(config, fetch_macro=False, fetch_migration=False)

    assert "permit_pipeline" in panel.columns
    assert panel["permit_pipeline"].notna().sum() > 0, (
        "permit_pipeline is entirely NaN — index alignment is broken"
    )
    # 18mo = 6 quarters. permits[t=6] = 600, so panel[2016Q3].permit_pipeline should be 0
    # (value from permits[2015Q1] = index 0 * 100 = 0)
    # More robustly: check that Q2016Q3 equals permits_5plus value at 2015Q1 (idx 0)
    assert panel.loc[pd.Period("2016Q3", freq="Q"), "permit_pipeline"] == pytest.approx(0.0)
    # And Q2020Q1 (idx 20) should equal permits[2018Q3] = idx 14 * 100 = 1400
    assert panel.loc[pd.Period("2020Q1", freq="Q"), "permit_pipeline"] == pytest.approx(1400.0)


def test_all_nan_permit_pipeline_raises_and_does_not_swallow(monkeypatch):
    """If shift/reindex yields all-NaN pipeline, build_panel MUST raise ValueError.

    Regression guard: the original fix wrapped the ValueError in an outer
    'except Exception' that silently printed a warning, defeating the
    assertion. This test forces all-NaN via a non-overlapping permits index
    and verifies the exception propagates.
    """
    idx_mismatched = pd.period_range("1980Q1", periods=40, freq="Q")
    permits = pd.DataFrame(
        {"permits_5plus": np.arange(40, dtype=float) * 100},
        index=idx_mismatched,
    )
    _patch_loaders(monkeypatch, permits)

    config = {
        "metro_id": "test_metro",
        "display_name": "Test",
        "endogenous_vars": ["rent_growth_yoy", "vacancy_rate"],
        "permit_lag_months": 18,
    }
    with pytest.raises(ValueError, match="permit_pipeline is entirely NaN"):
        build_panel(config, fetch_macro=False, fetch_migration=False)


def test_raw_unlagged_permit_columns_not_exposed_for_model_selection(monkeypatch):
    """_raw columns must not leak into the panel where exog selection would pick them up."""
    idx = pd.period_range("2015Q1", periods=40, freq="Q")
    permits = pd.DataFrame({"permits_5plus": np.arange(40, dtype=float) * 100}, index=idx)
    _patch_loaders(monkeypatch, permits)

    config = {
        "metro_id": "test_metro",
        "display_name": "Test",
        "endogenous_vars": ["rent_growth_yoy", "vacancy_rate"],
        "permit_lag_months": 18,
    }
    panel = build_panel(config, fetch_macro=False, fetch_migration=False)

    raw_leaks = [c for c in panel.columns if c.endswith("_raw") and "permit" in c.lower()]
    assert raw_leaks == [], f"Raw unlagged permit columns leaked: {raw_leaks}"

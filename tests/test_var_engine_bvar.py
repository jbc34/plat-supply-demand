"""Unit tests for Bayesian VAR Minnesota prior math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econometrics.models.var_engine import VAREngine


def _make_panel(seed: int = 0, n: int = 80) -> pd.DataFrame:
    """Build a synthetic panel with known AR(1) structure for each variable."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005Q1", periods=n, freq="Q")

    rent = np.zeros(n)
    for t in range(1, n):
        rent[t] = 0.7 * rent[t - 1] + rng.normal(0, 0.01)
    rent += 0.03

    vac = np.zeros(n)
    for t in range(1, n):
        vac[t] = 0.9 * vac[t - 1] + rng.normal(0, 0.005)
    vac += 0.08

    dlv = np.zeros(n)
    for t in range(1, n):
        dlv[t] = 0.5 * dlv[t - 1] + rng.normal(0, 0.002)
    dlv += 0.01

    absb = np.zeros(n)
    for t in range(1, n):
        absb[t] = 0.6 * absb[t - 1] + rng.normal(0, 0.002)
    absb += 0.01

    return pd.DataFrame(
        {
            "rent_growth_yoy": rent,
            "vacancy_rate": vac,
            "deliveries_pct_inventory": dlv,
            "absorption_pct_inventory": absb,
        },
        index=idx,
    )


def test_sigma_sq_matches_ols_ar1_residual_variance():
    """_fit_bayesian must estimate sigma_sq via OLS slope, not corrcoef."""
    panel = _make_panel(seed=1)
    engine = VAREngine(method="bayesian", minnesota_lambda=0.2)
    engine.fit(panel)

    Y = engine._endog.values
    expected = np.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        y = Y[:, i]
        b = np.cov(y[1:], y[:-1], ddof=0)[0, 1] / np.var(y[:-1], ddof=0)
        resid = y[1:] - b * y[:-1]
        expected[i] = np.var(resid, ddof=0)

    assert engine._bvar_sigma_sq is not None
    np.testing.assert_allclose(engine._bvar_sigma_sq, expected, rtol=1e-10)


def test_sigma_sq_scale_invariance_to_variable_units():
    """Scaling one variable by a constant must scale its sigma_sq by that constant squared."""
    panel = _make_panel(seed=2)
    engine_a = VAREngine(method="bayesian")
    engine_a.fit(panel)

    scaled = panel.copy()
    scaled["rent_growth_yoy"] = scaled["rent_growth_yoy"] * 100
    engine_b = VAREngine(method="bayesian")
    engine_b.fit(scaled)

    ratio = engine_b._bvar_sigma_sq[0] / engine_a._bvar_sigma_sq[0]
    assert ratio == pytest.approx(100 ** 2, rel=1e-6)


def test_endog_ordering_is_stable_regardless_of_panel_column_order():
    """Fitted model's endog ordering must follow endog_cols, not panel.columns."""
    panel = _make_panel(seed=4)
    panel_reversed = panel[list(reversed(panel.columns.tolist()))]

    engine_a = VAREngine(
        method="bayesian",
        endog_cols=[
            "rent_growth_yoy",
            "vacancy_rate",
            "deliveries_pct_inventory",
            "absorption_pct_inventory",
        ],
    )
    engine_b = VAREngine(
        method="bayesian",
        endog_cols=[
            "rent_growth_yoy",
            "vacancy_rate",
            "deliveries_pct_inventory",
            "absorption_pct_inventory",
        ],
    )

    engine_a.fit(panel)
    engine_b.fit(panel_reversed)

    assert list(engine_a._endog.columns) == list(engine_b._endog.columns), (
        f"Column order differs: {list(engine_a._endog.columns)} vs "
        f"{list(engine_b._endog.columns)}"
    )
    f_a = engine_a.predict(horizon=4)
    f_b = engine_b.predict(horizon=4)
    np.testing.assert_allclose(f_a.point, f_b.point, rtol=1e-10)

"""Shared test fixtures for econometrics module.

Provides deterministic synthetic quarterly panels with known statistical
properties for testing all model, ensemble, and backtest components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def quarterly_index():
    """60-quarter PeriodIndex (2010-Q1 through 2024-Q4)."""
    return pd.period_range("2010Q1", periods=60, freq="Q")


@pytest.fixture
def synthetic_panel(rng, quarterly_index):
    """Synthetic quarterly panel with realistic CoStar-like dynamics.

    Generates 60 quarters of data with known VAR(1) structure:
    - rent_growth_yoy: AR(1) with vacancy feedback
    - vacancy_rate: mean-reverting around 6%
    - deliveries_pct_inventory: cyclical pattern
    - absorption_pct_inventory: tracks deliveries with noise

    Plus exogenous variables:
    - mortgage_rate_30y: trending up then down
    - unemployment_rate: counter-cyclical to absorption
    - concession_spread: correlated with vacancy
    """
    n = len(quarterly_index)

    # Endogenous system with feedback
    vacancy = np.zeros(n)
    rent_growth = np.zeros(n)
    deliveries = np.zeros(n)
    absorption = np.zeros(n)

    vacancy[0] = 0.06
    rent_growth[0] = 0.03
    deliveries[0] = 0.02
    absorption[0] = 0.025

    for t in range(1, n):
        # Vacancy mean-reverts to 6%, affected by deliveries-absorption gap
        vacancy[t] = (
            0.06
            + 0.7 * (vacancy[t - 1] - 0.06)
            + 0.3 * (deliveries[t - 1] - absorption[t - 1])
            + rng.normal(0, 0.003)
        )
        vacancy[t] = np.clip(vacancy[t], 0.02, 0.15)

        # Rent growth inversely related to vacancy, autoregressive
        rent_growth[t] = (
            0.03
            + 0.5 * (rent_growth[t - 1] - 0.03)
            - 2.0 * (vacancy[t - 1] - 0.06)
            + rng.normal(0, 0.008)
        )

        # Deliveries: cyclical (construction cycle ~6 years)
        cycle = 0.005 * np.sin(2 * np.pi * t / 24)
        deliveries[t] = 0.02 + cycle + rng.normal(0, 0.003)
        deliveries[t] = max(deliveries[t], 0)

        # Absorption: tracks deliveries + demand signal
        absorption[t] = (
            0.6 * deliveries[t]
            + 0.3 * max(rent_growth[t - 1], 0)
            + rng.normal(0, 0.004)
        )
        absorption[t] = max(absorption[t], -0.01)

    # Exogenous variables
    mortgage = 4.0 + np.cumsum(rng.normal(0, 0.1, n))
    mortgage = np.clip(mortgage, 2.5, 8.0)

    unemployment = 4.5 - 10 * (absorption - 0.02) + rng.normal(0, 0.3, n)
    unemployment = np.clip(unemployment, 2.5, 10.0)

    concession = 0.02 + 0.5 * (vacancy - 0.06) + rng.normal(0, 0.005, n)
    concession = np.clip(concession, 0, 0.15)

    # Also generate rent levels for ECM testing
    effective_rent = np.zeros(n)
    effective_rent[0] = 1200
    for t in range(1, n):
        effective_rent[t] = effective_rent[t - 1] * (1 + rent_growth[t] / 4)

    median_income = np.zeros(n)
    median_income[0] = 55000
    for t in range(1, n):
        median_income[t] = median_income[t - 1] * (1 + rng.normal(0.008, 0.003))

    panel = pd.DataFrame(
        {
            "rent_growth_yoy": rent_growth,
            "vacancy_rate": vacancy,
            "deliveries_pct_inventory": deliveries,
            "absorption_pct_inventory": absorption,
            "mortgage_rate_30y": mortgage,
            "unemployment_rate": unemployment,
            "concession_spread": concession,
            "effective_rent_unit": effective_rent,
            "asking_rent_unit": effective_rent * (1 + concession),
            "median_hh_income": median_income,
            "inventory_units": np.linspace(50000, 65000, n),
        },
        index=quarterly_index,
    )

    # Add lagged versions
    for col in ["rent_growth_yoy", "vacancy_rate", "deliveries_pct_inventory",
                "absorption_pct_inventory", "concession_spread", "mortgage_rate_30y"]:
        panel[f"{col}_lag1"] = panel[col].shift(1)

    return panel


@pytest.fixture
def synthetic_panel_short(synthetic_panel):
    """Short panel (20 quarters) for testing low-data edge cases."""
    return synthetic_panel.iloc[:20]


@pytest.fixture
def dallas_config():
    """Dallas metro config dict (matches dallas_tx.yaml)."""
    return {
        "metro_id": "dallas_tx",
        "display_name": "Dallas-Fort Worth Multifamily",
        "msa_code": "19100",
        "state_fips": "48",
        "counties": ["085", "113"],
        "submarkets": [
            {"id": "far_north_dallas", "name": "Far North Dallas", "lat": 32.988, "lon": -96.770},
            {"id": "addison_galleria", "name": "Addison/Galleria", "lat": 32.954, "lon": -96.837},
            {"id": "richardson_utd", "name": "Richardson/UTD", "lat": 32.950, "lon": -96.730},
            {"id": "plano", "name": "Plano", "lat": 33.020, "lon": -96.699},
        ],
        "endogenous_vars": [
            "rent_growth_yoy", "vacancy_rate",
            "deliveries_pct_inventory", "absorption_pct_inventory",
        ],
        "exogenous_vars": [
            "mortgage_rate_30y", "unemployment_rate", "concession_spread",
        ],
        "enabled_models": ["VAREngine", "ECMEngine", "RidgeLassoEngine"],
        "ensemble_decay": 0.9,
        "min_backtest_quarters": 4,
        "default_horizon_quarters": 8,
        "decompose_seasonal": True,
    }


@pytest.fixture
def forecast_periods():
    """8-quarter forecast period labels."""
    return [f"2025-Q{q}" for q in range(1, 5)] + [f"2026-Q{q}" for q in range(1, 5)]

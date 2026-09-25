"""Named macroeconomic shock scenarios for stress testing.

Each shock defines variable overrides (additive shifts), duration,
and ramp-in behavior. Applied by stress_tester.py to produce
alternative forecast paths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioShock:
    """A named shock scenario."""

    name: str
    description: str
    variable_overrides: dict[str, float]  # Variable name → additive shift
    duration_quarters: int
    ramp_in_quarters: int = 0             # Linear ramp from 0 to full shock


# ─── Pre-defined shocks ─────────────────────────────────────────────


GFC_2008 = ScenarioShock(
    name="GFC_2008",
    description="2008-2009 financial crisis: credit freeze, job losses, demand collapse",
    variable_overrides={
        "unemployment_rate": 0.04,        # +4pp peak
        "mortgage_rate_30y": 0.012,       # +120bps (initially, then drops)
        "absorption_pct_inventory": -0.015,  # Absorption drops sharply
        "vacancy_rate": 0.03,             # +300bps vacancy spike
    },
    duration_quarters=6,
    ramp_in_quarters=2,
)

COVID_2020 = ScenarioShock(
    name="COVID_2020",
    description="2020 pandemic: sharp V-shaped demand shock, rapid recovery",
    variable_overrides={
        "absorption_pct_inventory": -0.025,  # -60% absorption for 2Q
        "vacancy_rate": 0.03,                # +300bps peak
        "rent_growth_yoy": -0.05,            # -5pp rent growth trough
        "unemployment_rate": 0.06,           # +6pp spike
    },
    duration_quarters=4,
    ramp_in_quarters=0,  # Immediate shock
)

RATE_SHOCK_400BPS = ScenarioShock(
    name="RATE_SHOCK_400BPS",
    description="Rapid 400bps mortgage rate increase (2022-style)",
    variable_overrides={
        "mortgage_rate_30y": 0.04,        # +400bps step
        "vacancy_rate": 0.01,             # Modest vacancy increase
        # Note: higher rates = fewer home purchases = more renters (mixed effect)
    },
    duration_quarters=4,
    ramp_in_quarters=2,
)

SUPPLY_GLUT = ScenarioShock(
    name="SUPPLY_GLUT",
    description="Sustained overbuilding (2015-2017 DFW pattern)",
    variable_overrides={
        "deliveries_pct_inventory": 0.015,    # +1.5pp above trend
        "vacancy_rate": 0.02,                 # +200bps from oversupply
        "absorption_pct_inventory": -0.005,   # Absorption can't keep up
        "concession_spread": 0.03,            # Concessions widen
    },
    duration_quarters=8,
    ramp_in_quarters=2,
)

# Registry of all shocks
SHOCK_REGISTRY: dict[str, ScenarioShock] = {
    "GFC_2008": GFC_2008,
    "COVID_2020": COVID_2020,
    "RATE_SHOCK_400BPS": RATE_SHOCK_400BPS,
    "SUPPLY_GLUT": SUPPLY_GLUT,
}

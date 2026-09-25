"""Structural equilibrium models.

Long-run equilibrium derived from economic fundamentals
and structural relationships.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StructuralEquilibriumResult:
    """Result of structural equilibrium estimation."""

    equilibrium_rent: pd.Series
    equilibrium_occupancy: pd.Series
    equilibrium_price: pd.Series
    natural_vacancy_rate: float
    cap_rate: pd.Series


class StructuralEquilibrium:
    """Structural model of market equilibrium.

    Models long-run equilibrium as the intersection of:
    - Demand: D = f(income, population, rents, preferences)
    - Supply: S = g(construction costs, interest rates, land availability)
    - Price formation: P adjusts to clear market

    At equilibrium:
    - Supply = Demand
    - Price = Replacement cost
    - Occupancy = Natural occupancy rate
    """

    def __init__(
        self,
        natural_vacancy_rate: float = 0.05,
        cap_rate_spread: float = 0.025,
        income_elasticity: float = 1.0,
        price_elasticity: float = -0.5,
        construction_lag: int = 18,
    ):
        """Initialize structural equilibrium model.

        Args:
            natural_vacancy_rate: Long-run vacancy rate at equilibrium
            cap_rate_spread: Cap rate spread over risk-free rate
            income_elasticity: Elasticity of demand to income
            price_elasticity: Elasticity of demand to price
            construction_lag: Months from start to delivery
        """
        self.natural_vacancy_rate = natural_vacancy_rate
        self.cap_rate_spread = cap_rate_spread
        self.income_elasticity = income_elasticity
        self.price_elasticity = price_elasticity
        self.construction_lag = construction_lag

    def estimate_equilibrium(
        self,
        data: pd.DataFrame,
        income_col: str = "median_income",
        construction_cost_col: str = "construction_cost",
        interest_rate_col: str = "DGS10",
        population_col: Optional[str] = None,
    ) -> StructuralEquilibriumResult:
        """Estimate equilibrium values from fundamentals.

        Args:
            data: DataFrame with economic indicators
            income_col: Column for income data
            construction_cost_col: Column for construction costs
            interest_rate_col: Column for risk-free rate
            population_col: Column for population (optional)

        Returns:
            StructuralEquilibriumResult with equilibrium series
        """
        # Get required series, using defaults if missing
        if income_col in data.columns:
            income = data[income_col]
        else:
            # Default income growth assumption
            income = pd.Series(
                60000 * (1.03 ** np.arange(len(data))),
                index=data.index
            )

        if construction_cost_col in data.columns:
            construction_cost = data[construction_cost_col]
        else:
            # Default construction cost per unit
            construction_cost = pd.Series(150000, index=data.index)

        if interest_rate_col in data.columns:
            risk_free_rate = data[interest_rate_col] / 100
        else:
            risk_free_rate = pd.Series(0.04, index=data.index)

        # Equilibrium cap rate
        cap_rate = risk_free_rate + self.cap_rate_spread

        # Equilibrium price (replacement cost)
        # P = Construction Cost / (1 - land_fraction)
        land_fraction = 0.2
        equilibrium_price = construction_cost / (1 - land_fraction)

        # Equilibrium NOI
        equilibrium_noi = equilibrium_price * cap_rate

        # Equilibrium rent (from NOI)
        operating_expense_ratio = 0.35
        equilibrium_rent = equilibrium_noi / (1 - operating_expense_ratio) / 12

        # Equilibrium occupancy
        equilibrium_occupancy = pd.Series(
            1 - self.natural_vacancy_rate,
            index=data.index
        )

        return StructuralEquilibriumResult(
            equilibrium_rent=equilibrium_rent,
            equilibrium_occupancy=equilibrium_occupancy,
            equilibrium_price=equilibrium_price,
            natural_vacancy_rate=self.natural_vacancy_rate,
            cap_rate=cap_rate,
        )

    def compute_deviations(
        self,
        actual: pd.DataFrame,
        equilibrium: StructuralEquilibriumResult,
        rent_col: str = "effective_rent",
        occupancy_col: str = "occupancy_rate",
        price_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute deviations of actual values from equilibrium.

        Args:
            actual: DataFrame with actual market values
            equilibrium: StructuralEquilibriumResult
            rent_col: Column for actual rents
            occupancy_col: Column for actual occupancy
            price_col: Column for actual prices (optional)

        Returns:
            DataFrame with deviation measures
        """
        deviations = pd.DataFrame(index=actual.index)

        # Rent deviation
        if rent_col in actual.columns:
            deviations["rent_deviation"] = (
                actual[rent_col] - equilibrium.equilibrium_rent
            )
            deviations["rent_deviation_pct"] = (
                deviations["rent_deviation"] / equilibrium.equilibrium_rent * 100
            )

        # Occupancy deviation
        if occupancy_col in actual.columns:
            # Convert percentage to decimal if needed
            actual_occ = actual[occupancy_col]
            if actual_occ.mean() > 1:
                actual_occ = actual_occ / 100

            deviations["occupancy_deviation"] = (
                actual_occ - equilibrium.equilibrium_occupancy
            )
            deviations["occupancy_deviation_pct"] = (
                deviations["occupancy_deviation"] /
                equilibrium.equilibrium_occupancy * 100
            )

        # Price deviation (if available)
        if price_col and price_col in actual.columns:
            deviations["price_deviation"] = (
                actual[price_col] - equilibrium.equilibrium_price
            )
            deviations["price_deviation_pct"] = (
                deviations["price_deviation"] /
                equilibrium.equilibrium_price * 100
            )

        return deviations

    def estimate_supply_demand_gap(
        self,
        data: pd.DataFrame,
        absorption_col: str = "net_absorption",
        deliveries_col: str = "deliveries",
        inventory_col: str = "inventory",
    ) -> pd.DataFrame:
        """Estimate supply-demand gap from flow data.

        Args:
            data: DataFrame with market flows
            absorption_col: Column for net absorption
            deliveries_col: Column for deliveries
            inventory_col: Column for inventory

        Returns:
            DataFrame with gap measures
        """
        result = pd.DataFrame(index=data.index)

        if absorption_col in data.columns and deliveries_col in data.columns:
            result["flow_gap"] = data[absorption_col] - data[deliveries_col]

            if inventory_col in data.columns:
                result["flow_gap_pct"] = (
                    result["flow_gap"] / data[inventory_col] * 100
                )

        # Cumulative gap
        if "flow_gap" in result.columns:
            result["cumulative_gap"] = result["flow_gap"].cumsum()

        return result

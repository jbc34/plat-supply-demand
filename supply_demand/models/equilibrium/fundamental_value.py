"""Fundamental value and long-term equilibrium estimation.

Estimates the equilibrium level around which market dynamics oscillate.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EquilibriumEstimate:
    """Result of equilibrium estimation."""

    value: float
    confidence_interval: tuple[float, float]
    method: str
    inputs: dict


class FundamentalValueEstimator(ABC):
    """Abstract base class for equilibrium estimators."""

    @abstractmethod
    def estimate(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> EquilibriumEstimate:
        """Estimate fundamental/equilibrium value.

        Args:
            data: DataFrame with relevant market data
            **kwargs: Method-specific parameters

        Returns:
            EquilibriumEstimate with value and metadata
        """
        pass


class ReplacementCostModel(FundamentalValueEstimator):
    """Estimates equilibrium based on replacement cost.

    Equilibrium rent/price is determined by the cost to build
    new supply plus required return to capital.

    Equilibrium Rent = (Land Cost + Construction Cost) * (Required Return + Depreciation)
    """

    def __init__(
        self,
        required_return: float = 0.06,
        depreciation_rate: float = 0.02,
        operating_expense_ratio: float = 0.35,
    ):
        """Initialize replacement cost model.

        Args:
            required_return: Required unlevered return (e.g., 6%)
            depreciation_rate: Annual depreciation rate (e.g., 2%)
            operating_expense_ratio: Operating expenses as % of revenue
        """
        self.required_return = required_return
        self.depreciation_rate = depreciation_rate
        self.operating_expense_ratio = operating_expense_ratio

    def estimate(
        self,
        data: pd.DataFrame,
        land_cost_per_unit: Optional[float] = None,
        construction_cost_per_unit: Optional[float] = None,
        **kwargs
    ) -> EquilibriumEstimate:
        """Estimate equilibrium rent from replacement cost.

        Args:
            data: DataFrame with cost data
            land_cost_per_unit: Land cost per unit (or column name in data)
            construction_cost_per_unit: Construction cost per unit

        Returns:
            EquilibriumEstimate for rent per unit
        """
        # Get cost inputs
        if land_cost_per_unit is None:
            if "land_cost" in data.columns:
                land_cost_per_unit = data["land_cost"].iloc[-1]
            else:
                land_cost_per_unit = 30000  # Default assumption

        if construction_cost_per_unit is None:
            if "construction_cost" in data.columns:
                construction_cost_per_unit = data["construction_cost"].iloc[-1]
            else:
                construction_cost_per_unit = 150000  # Default assumption

        # Total asset value per unit
        total_cost = land_cost_per_unit + construction_cost_per_unit

        # Required NOI per unit
        cap_rate = self.required_return + self.depreciation_rate
        required_noi = total_cost * cap_rate

        # Gross rent needed (backing out from NOI)
        gross_rent_annual = required_noi / (1 - self.operating_expense_ratio)
        equilibrium_rent_monthly = gross_rent_annual / 12

        # Confidence interval (rough estimate based on cost uncertainty)
        ci_lower = equilibrium_rent_monthly * 0.85
        ci_upper = equilibrium_rent_monthly * 1.15

        return EquilibriumEstimate(
            value=equilibrium_rent_monthly,
            confidence_interval=(ci_lower, ci_upper),
            method="replacement_cost",
            inputs={
                "land_cost": land_cost_per_unit,
                "construction_cost": construction_cost_per_unit,
                "required_return": self.required_return,
                "depreciation_rate": self.depreciation_rate,
                "operating_expense_ratio": self.operating_expense_ratio,
            }
        )

    def estimate_series(
        self,
        construction_costs: pd.Series,
        land_costs: Optional[pd.Series] = None,
        land_to_building_ratio: float = 0.2,
    ) -> pd.Series:
        """Estimate equilibrium rent time series.

        Args:
            construction_costs: Time series of construction costs
            land_costs: Time series of land costs (optional)
            land_to_building_ratio: Land as fraction of total if not provided

        Returns:
            Time series of equilibrium rents
        """
        if land_costs is None:
            land_costs = construction_costs * land_to_building_ratio

        total_cost = land_costs + construction_costs
        cap_rate = self.required_return + self.depreciation_rate
        required_noi = total_cost * cap_rate
        gross_rent_annual = required_noi / (1 - self.operating_expense_ratio)

        return gross_rent_annual / 12


class AffordabilityModel(FundamentalValueEstimator):
    """Estimates equilibrium based on affordability constraints.

    Equilibrium rent is determined by household income and
    standard affordability ratios.

    Equilibrium Rent = Median Income * Affordability Ratio / 12
    """

    def __init__(
        self,
        affordability_ratio: float = 0.30,
        income_percentile: float = 0.50,
    ):
        """Initialize affordability model.

        Args:
            affordability_ratio: Max rent as % of income (e.g., 30%)
            income_percentile: Income percentile to use (e.g., 50th = median)
        """
        self.affordability_ratio = affordability_ratio
        self.income_percentile = income_percentile

    def estimate(
        self,
        data: pd.DataFrame,
        income_col: str = "median_income",
        **kwargs
    ) -> EquilibriumEstimate:
        """Estimate equilibrium rent from affordability.

        Args:
            data: DataFrame with income data
            income_col: Column name for income data

        Returns:
            EquilibriumEstimate for rent per unit
        """
        if income_col not in data.columns:
            raise ValueError(f"Income column '{income_col}' not found in data")

        income = data[income_col].iloc[-1]

        # Monthly affordable rent
        equilibrium_rent = income * self.affordability_ratio / 12

        # Confidence interval based on affordability range
        ci_lower = income * 0.25 / 12
        ci_upper = income * 0.35 / 12

        return EquilibriumEstimate(
            value=equilibrium_rent,
            confidence_interval=(ci_lower, ci_upper),
            method="affordability",
            inputs={
                "income": income,
                "affordability_ratio": self.affordability_ratio,
                "income_percentile": self.income_percentile,
            }
        )

    def estimate_series(
        self,
        income_series: pd.Series,
    ) -> pd.Series:
        """Estimate equilibrium rent time series from income.

        Args:
            income_series: Time series of income levels

        Returns:
            Time series of equilibrium rents
        """
        return income_series * self.affordability_ratio / 12


class BlendedEquilibriumModel(FundamentalValueEstimator):
    """Blends multiple equilibrium estimation methods.

    Combines replacement cost and affordability approaches
    with configurable weights.
    """

    def __init__(
        self,
        replacement_cost_weight: float = 0.6,
        affordability_weight: float = 0.4,
        replacement_model: Optional[ReplacementCostModel] = None,
        affordability_model: Optional[AffordabilityModel] = None,
    ):
        """Initialize blended model.

        Args:
            replacement_cost_weight: Weight on replacement cost estimate
            affordability_weight: Weight on affordability estimate
            replacement_model: Custom replacement cost model
            affordability_model: Custom affordability model
        """
        self.replacement_weight = replacement_cost_weight
        self.affordability_weight = affordability_weight
        self.replacement_model = replacement_model or ReplacementCostModel()
        self.affordability_model = affordability_model or AffordabilityModel()

        # Normalize weights
        total = self.replacement_weight + self.affordability_weight
        self.replacement_weight /= total
        self.affordability_weight /= total

    def estimate(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> EquilibriumEstimate:
        """Estimate blended equilibrium.

        Args:
            data: DataFrame with required data for both models

        Returns:
            EquilibriumEstimate with blended value
        """
        estimates = []
        weights = []

        # Try replacement cost
        try:
            rc_estimate = self.replacement_model.estimate(data, **kwargs)
            estimates.append(rc_estimate.value)
            weights.append(self.replacement_weight)
        except Exception as e:
            print(f"Replacement cost estimation failed: {e}")

        # Try affordability
        try:
            aff_estimate = self.affordability_model.estimate(data, **kwargs)
            estimates.append(aff_estimate.value)
            weights.append(self.affordability_weight)
        except Exception as e:
            print(f"Affordability estimation failed: {e}")

        if not estimates:
            raise ValueError("All equilibrium estimation methods failed")

        # Normalize weights and compute blended estimate
        weights = np.array(weights) / sum(weights)
        blended_value = sum(e * w for e, w in zip(estimates, weights))

        # Confidence interval from component estimates
        all_bounds = []
        if len(estimates) > 0:
            all_bounds.extend([e * 0.9 for e in estimates])
            all_bounds.extend([e * 1.1 for e in estimates])

        ci_lower = min(all_bounds) if all_bounds else blended_value * 0.85
        ci_upper = max(all_bounds) if all_bounds else blended_value * 1.15

        return EquilibriumEstimate(
            value=blended_value,
            confidence_interval=(ci_lower, ci_upper),
            method="blended",
            inputs={
                "component_estimates": estimates,
                "component_weights": weights.tolist(),
            }
        )

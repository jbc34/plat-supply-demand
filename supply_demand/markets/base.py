"""Base class for market implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from supply_demand.models.state_space import StateSpaceParams
from supply_demand.models.coordinates import SpiralCoordinates


@dataclass
class MarketConfig:
    """Configuration for a market."""

    name: str
    msa_code: str
    state: str
    asset_class: str
    params: StateSpaceParams


class MarketBase(ABC):
    """Abstract base class for market-specific implementations.

    Subclasses implement market-specific data loading, proxy definitions,
    and equilibrium estimation.
    """

    def __init__(self, config: MarketConfig):
        """Initialize market.

        Args:
            config: Market configuration
        """
        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._equilibrium: Optional[pd.Series] = None

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def load_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load market data.

        Args:
            start_date: Start date for data
            end_date: End date (defaults to present)

        Returns:
            DataFrame with market data
        """
        pass

    @abstractmethod
    def get_proxy_config(self) -> dict:
        """Get proxy configuration for this market.

        Returns:
            Dictionary mapping proxy names to configurations
        """
        pass

    @abstractmethod
    def estimate_equilibrium(self) -> pd.Series:
        """Estimate equilibrium values over time.

        Returns:
            Series of equilibrium values
        """
        pass

    def get_current_position(self) -> SpiralCoordinates:
        """Get current spiral coordinates.

        Returns:
            SpiralCoordinates for current market state
        """
        from supply_demand.models.coordinates import compute_spiral_coordinates

        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Extract state variables from data
        # Subclasses should override this for market-specific logic
        D = self._data.get("demand_deviation", pd.Series([0])).values
        S = self._data.get("supply_deviation", pd.Series([0])).values
        K = self._data.get("capital_deviation", pd.Series([0])).values

        return compute_spiral_coordinates(D, S, K)

    def run_estimation(self) -> pd.DataFrame:
        """Run full estimation pipeline.

        Returns:
            DataFrame with estimated states
        """
        from supply_demand.models.state_space import (
            KalmanFilter,
            build_transition_matrix,
            build_measurement_matrix,
            build_process_noise_matrix,
            build_measurement_noise_matrix,
        )

        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Build matrices
        params = self.config.params
        proxy_config = self.get_proxy_config()

        F = build_transition_matrix(params)
        H = build_measurement_matrix(proxy_config)
        Q = build_process_noise_matrix(params)
        R = build_measurement_noise_matrix(len(proxy_config))

        # Create filter
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R)

        # Prepare observations
        # This should be customized per market
        Y = self._prepare_observations()

        # Run filter
        results = kf.filter(Y)

        return results.to_dataframe(
            index=self._data.index,
            state_names=["D", "S", "K", "dD", "dS", "dK"]
        )

    def _prepare_observations(self) -> pd.DataFrame:
        """Prepare observation matrix from data.

        Returns:
            Observation matrix aligned with proxy config
        """
        proxy_config = self.get_proxy_config()
        obs_data = []

        for proxy_name, config in proxy_config.items():
            col = config.get("column", proxy_name)
            if col in self._data.columns:
                obs_data.append(self._data[col])
            else:
                obs_data.append(pd.Series(index=self._data.index, dtype=float))

        return pd.concat(obs_data, axis=1).values

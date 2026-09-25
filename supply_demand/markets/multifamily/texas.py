"""Texas multifamily market implementations."""

from typing import Optional

import pandas as pd

from supply_demand.markets.base import MarketBase, MarketConfig
from supply_demand.models.state_space import StateSpaceParams
from supply_demand.data.sources import FREDClient, BLSClient


class TexasMultifamilyMarket(MarketBase):
    """Texas multifamily market implementation.

    Supports Dallas, Houston, Austin, and San Antonio metros.
    """

    MSA_PERMIT_SERIES = {
        "dallas": "DALL148BPPRIV",
        "houston": "HOUS448BPPRIV",
        "austin": "AUST448BPPRIV",
        "san_antonio": "SANA748BPPRIV",
    }

    MSA_CONFIGS = {
        "dallas": {
            "name": "Dallas-Fort Worth Multifamily",
            "msa_code": "19100",
            "bls_series": "SMU48191000000000001",  # Correct: area 19100
        },
        "houston": {
            "name": "Houston Multifamily",
            "msa_code": "26420",
            "bls_series": "SMU48264200000000001",
        },
        "austin": {
            "name": "Austin Multifamily",
            "msa_code": "12420",
            "bls_series": "SMU48124200000000001",  # Correct: area 12420
        },
        "san_antonio": {
            "name": "San Antonio Multifamily",
            "msa_code": "41700",
            "bls_series": "SMU48416600000000001",
        },
    }

    def __init__(
        self,
        msa: str = "dallas",
        params: Optional[StateSpaceParams] = None,
    ):
        """Initialize Texas multifamily market.

        Args:
            msa: MSA name (dallas, houston, austin, san_antonio)
            params: Model parameters (uses defaults if None)
        """
        if msa not in self.MSA_CONFIGS:
            raise ValueError(f"Unknown MSA: {msa}. Options: {list(self.MSA_CONFIGS.keys())}")

        msa_config = self.MSA_CONFIGS[msa]

        config = MarketConfig(
            name=msa_config["name"],
            msa_code=msa_config["msa_code"],
            state="TX",
            asset_class="multifamily",
            params=params or StateSpaceParams(),
        )

        super().__init__(config)

        self.msa = msa
        self.bls_series = msa_config["bls_series"]

    def load_data(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load market data from various sources.

        Args:
            start_date: Start date
            end_date: End date (defaults to present)

        Returns:
            DataFrame with aligned market data
        """
        data_frames = []

        # Load FRED data
        try:
            fred = FREDClient()
            fred_data = fred.get_real_estate_indicators(
                start_date=start_date,
                end_date=end_date,
            )
            data_frames.append(fred_data)
        except Exception as e:
            print(f"Warning: Failed to load FRED data: {e}")

        # MSA-level building permits (per-market Supply signal)
        try:
            fred_msa = FREDClient()
            msa_id = self.MSA_PERMIT_SERIES.get(self.msa)
            if msa_id:
                permits = fred_msa.get_series(
                    msa_id, start_date=start_date, end_date=end_date,
                )
                permits.name = "msa_permits"
                data_frames.append(permits.to_frame())
        except Exception as e:
            print(f"Warning: Failed to load MSA permits: {e}")

        # Load BLS employment
        try:
            bls = BLSClient()
            emp = bls.get_series(
                self.bls_series,
                start_date=start_date,
                end_date=end_date,
            )
            emp.name = "employment"
            emp_growth = emp.pct_change(12) * 100
            emp_growth.name = "employment_growth"
            data_frames.append(emp.to_frame())
            data_frames.append(emp_growth.to_frame())
        except Exception as e:
            print(f"Warning: Failed to load BLS data: {e}")

        if not data_frames:
            self._data = pd.DataFrame()
            return self._data

        # Combine all data
        self._data = pd.concat(data_frames, axis=1)

        # Ensure DatetimeIndex for resampling
        if not isinstance(self._data.index, pd.DatetimeIndex):
            self._data.index = pd.to_datetime(self._data.index)

        self._data = self._data.resample("MS").last()

        return self._data

    def get_proxy_config(self) -> dict:
        """Get proxy configuration for Texas multifamily.

        Returns:
            Proxy configuration dictionary
        """
        return {
            # Demand proxies
            "employment_growth": {
                "state": "D",
                "loading": 1.0,
                "column": "employment_growth",
            },
            # Supply proxies
            "permits": {
                "state": "S",
                "loading": 1.0,
                "column": (
                    "msa_permits"
                    if (self._data is not None and "msa_permits" in self._data.columns)
                    else "PERMIT"
                ),
            },
            "housing_starts": {
                "state": "S",
                "loading": 0.5,
                "column": "HOUST",
            },
            # Capital proxies
            "mortgage_rate": {
                "state": "K",
                "loading": -1.0,  # Higher rates = less capital
                "column": "MORTGAGE30US",
            },
            "treasury_10y": {
                "state": "K",
                "loading": -0.5,
                "column": "DGS10",
            },
        }

    def _prepare_observations(self):
        """Z-score each proxy observation so states are comparable deviations.

        Overrides the base (raw-scale) version: without standardizing, the
        large-magnitude national permit series dominates the spiral geometry
        and washes out per-MSA differences.
        """
        proxy_config = self.get_proxy_config()
        obs_data = []
        for proxy_name, config in proxy_config.items():
            col = config.get("column", proxy_name)
            if col in self._data.columns:
                s = self._data[col].astype(float)
                sd = s.std()
                s = (s - s.mean()) / sd if sd and sd > 0 else s - s.mean()
                obs_data.append(s)
            else:
                obs_data.append(pd.Series(index=self._data.index, dtype=float))
        return pd.concat(obs_data, axis=1).values

    def estimate_equilibrium(self) -> pd.Series:
        """Estimate equilibrium for Texas multifamily.

        Uses a combination of replacement cost and income growth.

        Returns:
            Series of equilibrium rent estimates
        """
        from supply_demand.models.equilibrium import ReplacementCostModel

        if self._data is None:
            raise ValueError("Data not loaded")

        model = ReplacementCostModel(
            required_return=0.06,
            depreciation_rate=0.02,
        )

        # Use construction PPI as cost proxy
        if "WPUSI012011" in self._data.columns:
            # Normalize to per-unit basis
            cost_index = self._data["WPUSI012011"]
            base_cost = 150000
            construction_cost = base_cost * (cost_index / cost_index.iloc[0])

            equilibrium = model.estimate_series(construction_cost)
        else:
            # Default trend
            equilibrium = pd.Series(
                1500 * (1.03 ** (range(len(self._data)) / 12)),
                index=self._data.index,
            )

        self._equilibrium = equilibrium
        return equilibrium

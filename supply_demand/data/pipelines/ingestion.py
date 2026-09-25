"""Data ingestion orchestration.

Coordinates fetching data from multiple sources and preparing
it for the estimation pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from supply_demand.config.settings import settings
from supply_demand.data.sources import (
    FREDClient,
    BLSClient,
    CensusClient,
    CoStarClient,
    LocalDataLoader,
)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""

    start_date: str
    end_date: Optional[str] = None
    markets: Optional[list[str]] = None
    frequency: str = "M"  # Monthly default
    include_fred: bool = True
    include_bls: bool = True
    include_census: bool = True
    include_costar: bool = True


class DataIngestionPipeline:
    """Orchestrates data ingestion from multiple sources.

    Fetches and combines data from FRED, BLS, Census, CoStar,
    and local sources into a unified dataset for analysis.

    Example:
        >>> pipeline = DataIngestionPipeline()
        >>> data = pipeline.run(
        ...     start_date="2010-01-01",
        ...     markets=["dallas_multifamily"]
        ... )
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        costar_data_dir: Optional[Path] = None,
        local_data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize ingestion pipeline.

        Args:
            config_path: Path to markets.yaml configuration
            costar_data_dir: Directory containing CoStar CSV exports
            local_data_dir: Directory containing local data files
            cache_dir: Directory for caching fetched data
        """
        self.config_path = config_path or settings.get_config_path("markets.yaml")
        self.cache_dir = cache_dir or settings.ensure_data_dir()

        # Initialize clients
        self._fred: Optional[FREDClient] = None
        self._bls: Optional[BLSClient] = None
        self._census: Optional[CensusClient] = None
        self._costar: Optional[CoStarClient] = None
        self._local: Optional[LocalDataLoader] = None

        self.costar_data_dir = costar_data_dir
        self.local_data_dir = local_data_dir

        # Load market configurations
        self.market_configs = self._load_market_configs()

    def _load_market_configs(self) -> dict:
        """Load market configurations from YAML."""
        if not self.config_path.exists():
            return {}

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        return config.get("markets", {})

    @property
    def fred(self) -> FREDClient:
        """Lazy initialization of FRED client."""
        if self._fred is None:
            self._fred = FREDClient(cache_dir=self.cache_dir / "fred")
        return self._fred

    @property
    def bls(self) -> BLSClient:
        """Lazy initialization of BLS client."""
        if self._bls is None:
            self._bls = BLSClient(cache_dir=self.cache_dir / "bls")
        return self._bls

    @property
    def census(self) -> CensusClient:
        """Lazy initialization of Census client."""
        if self._census is None:
            self._census = CensusClient(cache_dir=self.cache_dir / "census")
        return self._census

    @property
    def costar(self) -> CoStarClient:
        """Lazy initialization of CoStar client."""
        if self._costar is None:
            self._costar = CoStarClient(
                data_dir=self.costar_data_dir,
                cache_dir=self.cache_dir / "costar"
            )
        return self._costar

    @property
    def local(self) -> LocalDataLoader:
        """Lazy initialization of local data loader."""
        if self._local is None:
            self._local = LocalDataLoader(
                data_dir=self.local_data_dir,
                cache_dir=self.cache_dir / "local"
            )
        return self._local

    def run(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        markets: Optional[list[str]] = None,
        frequency: str = "M",
    ) -> dict[str, pd.DataFrame]:
        """Run the ingestion pipeline.

        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date (defaults to today)
            markets: List of markets to fetch (defaults to all configured)
            frequency: Target frequency for resampling

        Returns:
            Dictionary mapping market names to DataFrames
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if markets is None:
            markets = list(self.market_configs.keys())

        results = {}

        # Fetch macroeconomic data (shared across all markets)
        macro_data = self._fetch_macro_data(start_date, end_date, frequency)

        # Fetch market-specific data
        for market in markets:
            print(f"Ingesting data for {market}...")

            market_config = self.market_configs.get(market, {})
            market_data = self._fetch_market_data(
                market,
                market_config,
                start_date,
                end_date,
                frequency
            )

            # Combine macro and market data
            if not market_data.empty:
                combined = pd.concat([macro_data, market_data], axis=1)
            else:
                combined = macro_data.copy()

            results[market] = combined

        return results

    def _fetch_macro_data(
        self,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> pd.DataFrame:
        """Fetch macroeconomic indicators from FRED."""
        try:
            return self.fred.get_real_estate_indicators(
                start_date=start_date,
                end_date=end_date,
                frequency=frequency.lower()
            )
        except Exception as e:
            print(f"Warning: Failed to fetch FRED data: {e}")
            return pd.DataFrame()

    def _fetch_market_data(
        self,
        market: str,
        config: dict,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> pd.DataFrame:
        """Fetch market-specific data from various sources."""
        data_frames = []

        # Extract years from dates
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year

        # Fetch BLS employment data
        data_sources = config.get("data_sources", {})
        bls_config = data_sources.get("bls", {})
        if bls_config:
            try:
                emp_series_id = bls_config.get("employment_series")
                if emp_series_id:
                    emp = self.bls.get_series(
                        emp_series_id,
                        start_year=start_year,
                        end_year=end_year
                    )
                    if not emp.empty:
                        emp.name = "employment"
                        # Compute YoY growth
                        emp_growth = emp.pct_change(12) * 100
                        emp_growth.name = "employment_growth"
                        data_frames.append(emp.to_frame())
                        data_frames.append(emp_growth.to_frame())
            except Exception as e:
                print(f"Warning: Failed to fetch BLS data for {market}: {e}")

        # Fetch Census permits data
        census_config = data_sources.get("census", {})
        if census_config:
            try:
                # Get MSA name from market (e.g., "dallas_multifamily" -> "dallas")
                msa_name = market.split("_")[0]
                permits = self.census.get_building_permits(
                    msa=msa_name,
                    start_year=start_year,
                    end_year=end_year
                )
                if not permits.empty:
                    permits.name = "permits"
                    data_frames.append(permits.to_frame())
            except Exception as e:
                print(f"Warning: Failed to fetch Census data for {market}: {e}")

        # Fetch CoStar data if available
        costar_config = data_sources.get("costar", {})
        if costar_config:
            try:
                costar_market_id = costar_config.get("market_id", market)
                costar_data = self.costar.get_market_data(costar_market_id)
                if not costar_data.empty:
                    data_frames.append(costar_data)
            except Exception as e:
                print(f"Warning: Failed to fetch CoStar data for {market}: {e}")

        # Combine all market data
        if not data_frames:
            return pd.DataFrame()

        combined = pd.concat(data_frames, axis=1)

        # Resample to target frequency
        if frequency.upper() == "M":
            combined = combined.resample("MS").last()
        elif frequency.upper() == "Q":
            combined = combined.resample("QS").last()

        return combined

    def get_available_series(self) -> dict[str, list[str]]:
        """List all available series by source.

        Returns:
            Dictionary mapping source names to available series
        """
        return {
            "fred": list(FREDClient.SERIES_CATALOG.keys()),
            "bls": list(BLSClient.TEXAS_MSA_SERIES.keys()),
            "costar": self.costar.get_available_markets(),
            "local": self.local.list_available_files() if self.local_data_dir else [],
        }

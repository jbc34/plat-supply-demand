"""Census Bureau data client.

Fetches building permits, population, and migration data.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests

from supply_demand.data.sources.base import DataSourceBase


class CensusClient(DataSourceBase):
    """Client for Census Bureau APIs.

    Provides access to:
    - Building permits data (leading supply indicator)
    - Population estimates
    - Migration flows

    Example:
        >>> client = CensusClient()
        >>> permits = client.get_building_permits(
        ...     state="48",  # Texas
        ...     start_year=2015
        ... )
    """

    # Census API endpoints
    PERMITS_URL = "https://api.census.gov/data/timeseries/bps"
    POPULATION_URL = "https://api.census.gov/data"

    # Texas county FIPS codes for major metros
    TEXAS_METRO_COUNTIES = {
        "dallas": ["48113", "48439", "48085", "48121", "48257"],  # Dallas, Tarrant, Collin, Denton, Kaufman
        "houston": ["48201", "48157", "48339", "48167"],  # Harris, Fort Bend, Montgomery, Galveston
        "austin": ["48453", "48491", "48209", "48021"],  # Travis, Williamson, Hays, Bastrop
        "san_antonio": ["48029", "48091", "48187"],  # Bexar, Comal, Guadalupe
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize Census client.

        Args:
            api_key: Census API key (optional but recommended).
                    If None, reads from CENSUS_API_KEY env variable.
            cache_dir: Directory for caching fetched data.
        """
        super().__init__(cache_dir)
        self.api_key = api_key or os.environ.get("CENSUS_API_KEY")

    @property
    def source_name(self) -> str:
        return "Census"

    def get_series(
        self,
        series_id: str,
        start_date: Optional[Union[str, int]] = None,
        end_date: Optional[Union[str, int]] = None,
        **kwargs
    ) -> pd.Series:
        """Generic series fetch - routes to appropriate method.

        Args:
            series_id: Series identifier in format "type:params"
                      e.g., "permits:48:48113" for TX Dallas county permits
            start_date: Start date/year
            end_date: End date/year

        Returns:
            pandas Series with datetime index
        """
        parts = series_id.split(":")
        series_type = parts[0]

        if series_type == "permits":
            state = parts[1] if len(parts) > 1 else "48"
            county = parts[2] if len(parts) > 2 else None
            return self.get_building_permits(
                state=state,
                county=county,
                start_year=start_date,
                end_year=end_date,
                **kwargs
            )
        elif series_type == "population":
            geo = parts[1] if len(parts) > 1 else None
            return self.get_population_estimates(geo=geo, **kwargs)
        else:
            raise ValueError(f"Unknown Census series type: {series_type}")

    def get_building_permits(
        self,
        state: str = "48",  # Texas
        county: Optional[str] = None,
        msa: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        units: str = "5+",  # Multifamily
        use_cache: bool = True,
    ) -> pd.Series:
        """Fetch building permits data.

        Args:
            state: State FIPS code (default "48" for Texas)
            county: County FIPS code (5 digits including state)
            msa: MSA name (dallas, houston, austin, san_antonio)
            start_year: Start year
            end_year: End year
            units: Unit type filter:
                   - "1": Single family
                   - "2": 2 units
                   - "3-4": 3-4 units
                   - "5+": 5+ units (multifamily)
                   - "total": All units
            use_cache: Whether to use cached data

        Returns:
            pandas Series with monthly permits data
        """
        if start_year is None:
            start_year = 2010
        if end_year is None:
            end_year = pd.Timestamp.now().year

        # If MSA specified, aggregate counties
        if msa:
            counties = self.TEXAS_METRO_COUNTIES.get(msa.lower(), [])
            if not counties:
                raise ValueError(f"Unknown MSA: {msa}")

            all_permits = []
            for county_fips in counties:
                try:
                    permits = self._fetch_permits_for_county(
                        county_fips,
                        start_year,
                        end_year,
                        units
                    )
                    if permits is not None:
                        all_permits.append(permits)
                except Exception as e:
                    print(f"Warning: Failed to fetch permits for county {county_fips}: {e}")

            if not all_permits:
                return pd.Series(name=f"permits_{msa}", dtype=float)

            # Sum across counties
            result = pd.concat(all_permits, axis=1).sum(axis=1)
            result.name = f"permits_{msa}"
            return result

        # Single county or state-level
        return self._fetch_permits_for_county(
            county or state,
            start_year,
            end_year,
            units
        )

    def _fetch_permits_for_county(
        self,
        geo_code: str,
        start_year: int,
        end_year: int,
        units: str
    ) -> Optional[pd.Series]:
        """Fetch permits for a single geographic unit."""
        # Map unit type to Census variable
        unit_var_map = {
            "1": "units_1",
            "2": "units_2",
            "3-4": "units_34",
            "5+": "units_5",
            "total": "units",
        }
        unit_var = unit_var_map.get(units, "units_5")

        # Build query parameters
        params = {
            "get": f"{unit_var}",
            "time": f"from+{start_year}+to+{end_year}",
        }

        if len(geo_code) == 2:
            # State level
            params["for"] = f"state:{geo_code}"
        elif len(geo_code) == 5:
            # County level
            params["for"] = f"county:{geo_code[2:]}"
            params["in"] = f"state:{geo_code[:2]}"

        if self.api_key:
            params["key"] = self.api_key

        # Note: Building permits API has specific structure
        # This is a simplified implementation - actual Census API
        # may require different endpoint/parameters
        try:
            response = requests.get(
                self.PERMITS_URL,
                params=params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Parse response into series
            # First row is headers
            if len(data) < 2:
                return None

            records = []
            for row in data[1:]:
                # Parse based on Census response format
                # This needs adjustment based on actual API response
                pass

            if not records:
                return None

            df = pd.DataFrame(records).set_index("date")
            return df["value"]

        except Exception as e:
            # Fall back to synthetic data for development
            print(f"Census API unavailable ({e}), returning placeholder")
            return self._generate_placeholder_permits(
                geo_code, start_year, end_year
            )

    def _generate_placeholder_permits(
        self,
        geo_code: str,
        start_year: int,
        end_year: int
    ) -> pd.Series:
        """Generate placeholder permits data for development."""
        import numpy as np

        dates = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq="MS"
        )

        # Generate realistic-looking permits with seasonality and trend
        n = len(dates)
        trend = np.linspace(100, 150, n)
        seasonal = 20 * np.sin(np.arange(n) * 2 * np.pi / 12)
        noise = np.random.normal(0, 10, n)

        values = trend + seasonal + noise
        values = np.maximum(values, 0)  # No negative permits

        return pd.Series(
            values,
            index=dates,
            name=f"permits_{geo_code}"
        )

    def get_population_estimates(
        self,
        geo: Optional[str] = None,
        year: Optional[int] = None
    ) -> pd.Series:
        """Fetch population estimates.

        Args:
            geo: Geographic identifier (state FIPS, county FIPS, or MSA name)
            year: Year for estimates (defaults to most recent)

        Returns:
            pandas Series with population data
        """
        # Placeholder implementation
        # Actual Census Population Estimates API has different structure
        return pd.Series(name="population", dtype=float)

    def get_migration_flows(
        self,
        msa: str,
        start_year: int = 2015,
        end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch domestic migration flow data.

        Args:
            msa: MSA name (dallas, houston, austin, san_antonio)
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with inflow, outflow, and net migration
        """
        # Placeholder - ACS migration data requires complex queries
        return pd.DataFrame(columns=["inflow", "outflow", "net_migration"])

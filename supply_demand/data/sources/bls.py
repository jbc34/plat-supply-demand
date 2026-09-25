"""Bureau of Labor Statistics API client.

Fetches employment data at MSA level for demand estimation.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests

from supply_demand.data.sources.base import DataSourceBase


class BLSClient(DataSourceBase):
    """Client for BLS Public Data API.

    Provides access to employment data from the Bureau of Labor Statistics,
    including MSA-level employment for Texas markets.

    Example:
        >>> client = BLSClient()
        >>> dallas_employment = client.get_series(
        ...     "SMU48192000000000001",
        ...     start_year=2015,
        ...     end_year=2024
        ... )
    """

    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    # Texas MSA employment series (Current Employment Statistics)
    # Format: SMU + State FIPS (48) + Area Code (5 digits) + SuperSector (00000000) + DataType (01)
    TEXAS_MSA_SERIES = {
        "dallas": "SMU48191000000000001",      # Dallas-Fort Worth-Arlington (area 19100)
        "houston": "SMU48264200000000001",     # Houston-The Woodlands-Sugar Land (area 26420)
        "austin": "SMU48124200000000001",      # Austin-Round Rock-Georgetown (area 12420)
        "san_antonio": "SMU48416600000000001", # San Antonio-New Braunfels (area 41660)
    }

    # Series name descriptions
    SERIES_DESCRIPTIONS = {
        "SMU48191000000000001": "Dallas-Fort Worth-Arlington Total Nonfarm",
        "SMU48264200000000001": "Houston-The Woodlands-Sugar Land Total Nonfarm",
        "SMU48124200000000001": "Austin-Round Rock-Georgetown Total Nonfarm",
        "SMU48416600000000001": "San Antonio-New Braunfels Total Nonfarm",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize BLS client.

        Args:
            api_key: BLS API key (optional but recommended for higher rate limits).
                    If None, reads from BLS_API_KEY env variable.
            cache_dir: Directory for caching fetched data.
        """
        super().__init__(cache_dir)
        self.api_key = api_key or os.environ.get("BLS_API_KEY")
        # Note: BLS API works without key but has lower rate limits

    @property
    def source_name(self) -> str:
        return "BLS"

    def get_series(
        self,
        series_id: str,
        start_date: Optional[Union[str, int]] = None,
        end_date: Optional[Union[str, int]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> pd.Series:
        """Fetch a single time series from BLS.

        Args:
            series_id: BLS series identifier (e.g., "SMU48192000000000001")
            start_date: Start date (will extract year) or use start_year
            end_date: End date (will extract year) or use end_year
            start_year: Start year (takes precedence over start_date)
            end_year: End year (takes precedence over end_date)
            use_cache: Whether to use cached data if available

        Returns:
            pandas Series with datetime index and employment values

        Raises:
            ValueError: If API returns error status
            requests.HTTPError: If API request fails
        """
        # Parse years
        if start_year is None:
            if start_date:
                if isinstance(start_date, int):
                    start_year = start_date
                else:
                    start_year = pd.to_datetime(start_date).year
            else:
                start_year = 2010  # Default

        if end_year is None:
            if end_date:
                if isinstance(end_date, int):
                    end_year = end_date
                else:
                    end_year = pd.to_datetime(end_date).year
            else:
                end_year = pd.Timestamp.now().year

        # Try cache first
        cache_key = f"{series_id}_{start_year}_{end_year}"
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # BLS API limits to 20 years per request
        all_data = []
        current_start = start_year

        while current_start <= end_year:
            current_end = min(current_start + 19, end_year)

            chunk = self._fetch_series_chunk(
                series_id,
                current_start,
                current_end
            )
            if chunk is not None and len(chunk) > 0:
                all_data.append(chunk)

            current_start = current_end + 1

        if not all_data:
            return pd.Series(name=series_id, dtype=float)

        # Combine all chunks
        series = pd.concat(all_data).sort_index()
        series = series[~series.index.duplicated(keep="first")]
        series.name = series_id

        # Cache result
        if use_cache and len(series) > 0:
            self._save_to_cache(cache_key, series)

        return series

    def _fetch_series_chunk(
        self,
        series_id: str,
        start_year: int,
        end_year: int
    ) -> Optional[pd.Series]:
        """Fetch a chunk of data (max 20 years)."""
        headers = {"Content-type": "application/json"}
        data = {
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }

        if self.api_key:
            data["registrationkey"] = self.api_key

        response = requests.post(
            self.BASE_URL,
            json=data,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        if result.get("status") != "REQUEST_SUCCEEDED":
            error_msg = result.get("message", ["Unknown error"])
            raise ValueError(f"BLS API error: {error_msg}")

        # Parse results
        series_data = result.get("Results", {}).get("series", [])
        if not series_data:
            return None

        records = []
        for obs in series_data[0].get("data", []):
            year = int(obs["year"])
            period = obs["period"]

            # Handle monthly data (M01-M12)
            if period.startswith("M"):
                month = int(period[1:])
                if month > 12:  # Skip annual averages (M13)
                    continue
                date = pd.Timestamp(year=year, month=month, day=1)
                records.append({
                    "date": date,
                    "value": float(obs["value"])
                })
            # Handle quarterly data (Q01-Q04)
            elif period.startswith("Q"):
                quarter = int(period[1:])
                month = (quarter - 1) * 3 + 1
                date = pd.Timestamp(year=year, month=month, day=1)
                records.append({
                    "date": date,
                    "value": float(obs["value"])
                })

        if not records:
            return None

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df["value"]

    def get_texas_employment(
        self,
        start_year: int = 2010,
        end_year: Optional[int] = None,
        compute_growth: bool = True
    ) -> pd.DataFrame:
        """Fetch employment data for all Texas MSAs.

        Convenience method to get employment for Dallas, Houston,
        Austin, and San Antonio markets.

        Args:
            start_year: Start year for data
            end_year: End year (defaults to current year)
            compute_growth: If True, add year-over-year growth columns

        Returns:
            DataFrame with employment levels (and growth if requested)
        """
        if end_year is None:
            end_year = pd.Timestamp.now().year

        data = {}
        for name, series_id in self.TEXAS_MSA_SERIES.items():
            try:
                series = self.get_series(
                    series_id,
                    start_year=start_year,
                    end_year=end_year
                )
                data[name] = series
            except Exception as e:
                print(f"Warning: Failed to fetch {name} employment: {e}")

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if compute_growth:
            # Add year-over-year growth rates
            for col in df.columns:
                df[f"{col}_yoy"] = df[col].pct_change(periods=12, fill_method=None) * 100

        return df

    def get_msa_series_id(self, msa_name: str) -> Optional[str]:
        """Get the BLS series ID for a Texas MSA.

        Args:
            msa_name: MSA name (dallas, houston, austin, san_antonio)

        Returns:
            Series ID string or None if not found
        """
        return self.TEXAS_MSA_SERIES.get(msa_name.lower())

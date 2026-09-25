"""FRED (Federal Reserve Economic Data) API client.

Fetches macroeconomic time series for the estimation system.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests

from supply_demand.data.sources.base import DataSourceBase


class FREDClient(DataSourceBase):
    """Client for FRED API.

    Provides access to Federal Reserve Economic Data including interest rates,
    employment, construction, and credit condition indicators.

    Example:
        >>> client = FREDClient()
        >>> mortgage_rates = client.get_series("MORTGAGE30US", start_date="2010-01-01")
        >>> print(mortgage_rates.head())
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Throttle between live FRED API requests to avoid 429 rate-limit skips.
    # Each series that 429-skips would leave that market with a different
    # column set, breaking cross-market comparability. Sleep AFTER each live
    # network fetch (cache hits are not throttled).
    REQUEST_DELAY_SECONDS = 0.6
    MAX_RETRIES = 4

    # Commonly used series for real estate analysis
    SERIES_CATALOG = {
        # Interest rates
        "FEDFUNDS": "Federal Funds Rate",
        "DGS10": "10-Year Treasury Constant Maturity Rate",
        "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average",
        "DPRIME": "Bank Prime Loan Rate",
        # Employment
        "PAYEMS": "Total Nonfarm Payrolls",
        "UNRATE": "Unemployment Rate",
        "ICSA": "Initial Jobless Claims",
        # Construction
        "PERMIT": "New Private Housing Units Authorized by Building Permits",
        "HOUST": "Housing Starts",
        "WPUSI012011": "PPI: Construction Materials",
        "TLRESCONS": "Total Construction Spending: Residential",
        # Income/Spending
        "DSPIC96": "Real Disposable Personal Income",
        "PCE": "Personal Consumption Expenditures",
        "PSAVERT": "Personal Saving Rate",
        # Credit conditions
        "DRTSCILM": "Net Percentage of Banks Tightening Standards for C&I Loans",
        "DRTSCLCC": "Net Percentage of Banks Tightening Standards for Consumer Loans",
        "TOTCI": "Commercial and Industrial Loans",
        # Financial conditions (K proxy expansion)
        "NFCI": "Chicago Fed National Financial Conditions Index",
        "NFCICREDIT": "NFCI Credit Subindex",
        "NFCILEVERAGE": "NFCI Leverage Subindex",
        "NFCINONFINLEVERAGE": "NFCI Nonfinancial Leverage Subindex",
        # Credit spreads (K proxy expansion)
        "BAMLC0A4CBBB": "ICE BofA BBB US Corporate Index Option-Adjusted Spread",
        "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
        "TEDRATE": "TED Spread (LIBOR-Treasury)",
        # Inflation
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
        "CPILFESL": "CPI Less Food and Energy",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize FRED client.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env variable.
            cache_dir: Directory for caching fetched data.

        Raises:
            ValueError: If no API key provided or found in environment.
        """
        super().__init__(cache_dir)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. Get a free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    @property
    def source_name(self) -> str:
        return "FRED"

    def get_series(
        self,
        series_id: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        frequency: Optional[str] = None,
        aggregation_method: str = "avg",
        use_cache: bool = True,
    ) -> pd.Series:
        """Fetch a single time series from FRED.

        Args:
            series_id: FRED series identifier (e.g., "MORTGAGE30US")
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            frequency: Frequency aggregation:
                - 'd': Daily
                - 'w': Weekly
                - 'bw': Bi-weekly
                - 'm': Monthly
                - 'q': Quarterly
                - 'sa': Semi-annual
                - 'a': Annual
            aggregation_method: Method for frequency conversion:
                - 'avg': Average
                - 'sum': Sum
                - 'eop': End of period
            use_cache: Whether to use cached data if available

        Returns:
            pandas Series with datetime index and series values

        Raises:
            requests.HTTPError: If API request fails
        """
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(series_id)
            if cached is not None:
                # Filter by date range if specified
                if start_date:
                    cached = cached[cached.index >= pd.to_datetime(start_date)]
                if end_date:
                    cached = cached[cached.index <= pd.to_datetime(end_date)]
                return cached

        # Build API request
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        if start_date:
            params["observation_start"] = self._parse_date(start_date)
        if end_date:
            params["observation_end"] = self._parse_date(end_date)
        if frequency:
            params["frequency"] = frequency
            params["aggregation_method"] = aggregation_method

        # Make request with 429-aware retry/backoff
        response = None
        for attempt in range(self.MAX_RETRIES):
            response = requests.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
                timeout=30
            )
            if response.status_code == 429:
                # Exponential backoff on rate limit, then retry the same series.
                backoff = self.REQUEST_DELAY_SECONDS * (2 ** attempt)
                time.sleep(backoff)
                continue
            break
        response.raise_for_status()

        # Inter-request throttle so the NEXT live fetch doesn't trip 429.
        time.sleep(self.REQUEST_DELAY_SECONDS)

        # Parse response
        data = response.json()
        observations = data.get("observations", [])

        dates = []
        values = []
        for obs in observations:
            dates.append(obs["date"])
            # Handle missing values (FRED uses "." for missing)
            val = obs["value"]
            if val == "." or val == "":
                values.append(None)
            else:
                values.append(float(val))

        series = pd.Series(
            values,
            index=pd.to_datetime(dates),
            name=series_id
        )

        # Remove missing values
        series = series.dropna()

        # Cache the result
        if use_cache and len(series) > 0:
            self._save_to_cache(series_id, series)

        return series

    def get_series_info(self, series_id: str) -> dict:
        """Get metadata about a FRED series.

        Args:
            series_id: FRED series identifier

        Returns:
            Dictionary with series metadata
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        response = requests.get(
            f"{self.BASE_URL}/series",
            params=params,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        if data.get("seriess"):
            return data["seriess"][0]
        return {}

    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """Search for FRED series by text.

        Args:
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            DataFrame with matching series info
        """
        params = {
            "search_text": search_text,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
        }

        response = requests.get(
            f"{self.BASE_URL}/series/search",
            params=params,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        series_list = data.get("seriess", [])

        return pd.DataFrame(series_list)

    def get_real_estate_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch common real estate-related indicators.

        Convenience method to get a standard set of indicators used
        for real estate market analysis.

        Args:
            start_date: Start date
            end_date: End date
            frequency: Optional frequency aggregation (not all series support this)

        Returns:
            DataFrame with all indicators (at their native frequencies)
        """
        indicators = [
            "MORTGAGE30US",  # Mortgage rate (weekly)
            "DGS10",         # 10Y Treasury (daily)
            "FEDFUNDS",      # Fed funds rate (monthly)
            "PERMIT",        # Building permits (monthly)
            "HOUST",         # Housing starts (monthly)
            "WPUSI012011",   # Construction PPI (monthly)
            "UNRATE",        # Unemployment (monthly)
            "CPIAUCSL",      # CPI (monthly)
        ]

        return self.get_multiple_series(
            indicators,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )

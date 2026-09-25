"""CoStar data integration.

Loads CoStar market data from local CSV exports or API.
CoStar provides the richest source of commercial real estate fundamentals.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from supply_demand.data.sources.base import DataSourceBase


class CoStarClient(DataSourceBase):
    """Client for CoStar data.

    CoStar data can be loaded from:
    1. CSV exports from CoStar Analytics
    2. CoStar API (requires enterprise subscription)

    This implementation focuses on CSV loading as that's more commonly available.

    Example:
        >>> client = CoStarClient(data_dir=Path("./costar_exports"))
        >>> dallas_data = client.get_market_data("dallas_multifamily")
    """

    # Expected columns in CoStar exports
    EXPECTED_COLUMNS = {
        "occupancy": ["occupancy", "occupancy_rate", "occ_rate"],
        "effective_rent": ["effective_rent", "eff_rent", "effective_rent_psf"],
        "asking_rent": ["asking_rent", "ask_rent", "asking_rent_psf"],
        "absorption": ["net_absorption", "absorption", "net_abs"],
        "deliveries": ["deliveries", "delivered_units", "completions"],
        "under_construction": ["under_construction", "uc_units", "construction_pipeline"],
        "inventory": ["inventory", "total_units", "stock"],
        "cap_rate": ["cap_rate", "market_cap_rate"],
        "sales_volume": ["sales_volume", "transaction_volume", "volume"],
    }

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize CoStar client.

        Args:
            data_dir: Directory containing CoStar CSV exports
            api_key: CoStar API key (if using API access)
            cache_dir: Directory for caching processed data
        """
        super().__init__(cache_dir)
        self.data_dir = data_dir
        self.api_key = api_key
        self._loaded_data: dict[str, pd.DataFrame] = {}

    @property
    def source_name(self) -> str:
        return "CoStar"

    def get_series(
        self,
        series_id: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs
    ) -> pd.Series:
        """Fetch a single series from CoStar data.

        Args:
            series_id: Series identifier in format "market:metric"
                      e.g., "dallas_multifamily:occupancy"
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            pandas Series with datetime index
        """
        parts = series_id.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid series_id format: {series_id}. "
                "Expected 'market:metric' (e.g., 'dallas_multifamily:occupancy')"
            )

        market, metric = parts
        df = self.get_market_data(market)

        if df.empty:
            return pd.Series(name=series_id, dtype=float)

        # Find the metric column
        metric_col = self._find_column(df, metric)
        if metric_col is None:
            raise ValueError(
                f"Metric '{metric}' not found in {market} data. "
                f"Available columns: {list(df.columns)}"
            )

        series = df[metric_col]
        series.name = series_id

        # Apply date filters
        if start_date:
            series = series[series.index >= pd.to_datetime(start_date)]
        if end_date:
            series = series[series.index <= pd.to_datetime(end_date)]

        return series

    def get_market_data(
        self,
        market: str,
        metrics: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Load all available data for a market.

        Args:
            market: Market identifier (e.g., "dallas_multifamily")
            metrics: Optional list of metrics to include

        Returns:
            DataFrame with datetime index and metric columns
        """
        # Check cache
        if market in self._loaded_data:
            df = self._loaded_data[market]
            if metrics:
                cols = [self._find_column(df, m) for m in metrics]
                cols = [c for c in cols if c is not None]
                return df[cols] if cols else pd.DataFrame()
            return df

        # Try loading from CSV
        df = self._load_from_csv(market)

        if df is not None and not df.empty:
            self._loaded_data[market] = df
            if metrics:
                cols = [self._find_column(df, m) for m in metrics]
                cols = [c for c in cols if c is not None]
                return df[cols] if cols else pd.DataFrame()
            return df

        # Return empty DataFrame if no data found
        return pd.DataFrame()

    def _load_from_csv(self, market: str) -> Optional[pd.DataFrame]:
        """Load market data from CSV export."""
        if self.data_dir is None:
            return None

        # Look for CSV files matching market name
        possible_files = [
            self.data_dir / f"{market}.csv",
            self.data_dir / f"{market}_data.csv",
            self.data_dir / f"{market}_export.csv",
            self.data_dir / market / "market_data.csv",
        ]

        for filepath in possible_files:
            if filepath.exists():
                return self._parse_costar_csv(filepath)

        return None

    def _parse_costar_csv(self, filepath: Path) -> pd.DataFrame:
        """Parse CoStar CSV export into DataFrame.

        Handles various date formats and column naming conventions
        used in CoStar exports.
        """
        # Try common date columns
        date_cols = ["date", "period", "quarter", "month", "Date", "Period"]

        df = pd.read_csv(filepath)

        # Find date column
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            # Try first column
            date_col = df.columns[0]

        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        return df

    def _find_column(self, df: pd.DataFrame, metric: str) -> Optional[str]:
        """Find the actual column name for a metric."""
        metric_lower = metric.lower()

        # Direct match
        if metric_lower in df.columns:
            return metric_lower

        # Check aliases
        if metric_lower in self.EXPECTED_COLUMNS:
            for alias in self.EXPECTED_COLUMNS[metric_lower]:
                if alias in df.columns:
                    return alias

        # Partial match
        for col in df.columns:
            if metric_lower in col:
                return col

        return None

    def load_csv_export(
        self,
        filepath: Union[str, Path],
        market: str
    ) -> pd.DataFrame:
        """Load a CoStar CSV export and register it for the market.

        Args:
            filepath: Path to CSV file
            market: Market identifier to register data under

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        df = self._parse_costar_csv(filepath)
        self._loaded_data[market] = df
        return df

    def compute_derived_metrics(
        self,
        df: pd.DataFrame,
        treasury_10y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Compute derived metrics from raw CoStar data.

        Args:
            df: Raw CoStar data
            treasury_10y: 10-year Treasury rates for cap rate spread

        Returns:
            DataFrame with additional computed columns
        """
        result = df.copy()

        # Rent growth
        rent_col = self._find_column(df, "effective_rent")
        if rent_col:
            result["rent_growth_yoy"] = df[rent_col].pct_change(4) * 100  # Quarterly
            result["rent_growth_mom"] = df[rent_col].pct_change(1) * 100

        # Absorption as % of inventory
        abs_col = self._find_column(df, "absorption")
        inv_col = self._find_column(df, "inventory")
        if abs_col and inv_col:
            result["absorption_pct"] = df[abs_col] / df[inv_col] * 100

        # Supply pipeline as % of inventory
        uc_col = self._find_column(df, "under_construction")
        if uc_col and inv_col:
            result["pipeline_pct"] = df[uc_col] / df[inv_col] * 100

        # Cap rate spread
        cap_col = self._find_column(df, "cap_rate")
        if cap_col and treasury_10y is not None:
            # Align treasury data to CoStar dates
            treasury_aligned = treasury_10y.reindex(df.index, method="ffill")
            result["cap_rate_spread"] = df[cap_col] - treasury_aligned

        return result

    def get_available_markets(self) -> list[str]:
        """List available markets in the data directory."""
        if self.data_dir is None:
            return list(self._loaded_data.keys())

        markets = []
        for f in self.data_dir.glob("*.csv"):
            market = f.stem.replace("_data", "").replace("_export", "")
            markets.append(market)

        # Add any manually loaded markets
        markets.extend(self._loaded_data.keys())

        return list(set(markets))

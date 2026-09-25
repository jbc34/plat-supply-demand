"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class DataSourceBase(ABC):
    """Abstract base class for all data source clients.

    Provides common interface for fetching time series data from various
    APIs and data providers. Subclasses implement source-specific logic.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data source.

        Args:
            cache_dir: Directory for caching fetched data. If None, no caching.
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_series(
        self,
        series_id: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        **kwargs
    ) -> pd.Series:
        """Fetch a single time series.

        Args:
            series_id: Identifier for the series (source-specific format)
            start_date: Start date for data range
            end_date: End date for data range
            **kwargs: Additional source-specific parameters

        Returns:
            pandas Series with datetime index
        """
        pass

    def get_multiple_series(
        self,
        series_ids: list[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch multiple time series and align to common dates.

        Args:
            series_ids: List of series identifiers
            start_date: Start date for data range
            end_date: End date for data range
            **kwargs: Additional source-specific parameters

        Returns:
            DataFrame with series as columns, datetime index
        """
        series_list = []
        for sid in series_ids:
            try:
                s = self.get_series(sid, start_date, end_date, **kwargs)
                series_list.append(s)
            except Exception as e:
                print(f"Warning: Failed to fetch {sid}: {e}")
                continue

        if not series_list:
            return pd.DataFrame()

        return pd.concat(series_list, axis=1)

    def _parse_date(self, date: Optional[Union[str, datetime]]) -> Optional[str]:
        """Parse date to string format YYYY-MM-DD."""
        if date is None:
            return None
        if isinstance(date, str):
            return date
        return date.strftime("%Y-%m-%d")

    def _get_cache_path(self, series_id: str) -> Optional[Path]:
        """Get cache file path for a series."""
        if self.cache_dir is None:
            return None
        safe_id = series_id.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_id}.parquet"

    def _load_from_cache(self, series_id: str) -> Optional[pd.Series]:
        """Load series from cache if available."""
        cache_path = self._get_cache_path(series_id)
        if cache_path and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]
        return None

    def _save_to_cache(self, series_id: str, data: pd.Series) -> None:
        """Save series to cache."""
        cache_path = self._get_cache_path(series_id)
        if cache_path:
            df = data.to_frame()
            df.to_parquet(cache_path)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source."""
        pass

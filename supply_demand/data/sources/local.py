"""Local/proprietary data loader.

Loads data from local files including CSV, Excel, and Parquet formats.
Useful for proprietary data or data not available via public APIs.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from supply_demand.data.sources.base import DataSourceBase


class LocalDataLoader(DataSourceBase):
    """Loader for local data files.

    Supports multiple file formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Parquet (.parquet)
    - JSON (.json)

    Example:
        >>> loader = LocalDataLoader(data_dir=Path("./data"))
        >>> df = loader.load_file("my_data.csv")
        >>> series = loader.get_series("my_data:column_name")
    """

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".parquet": "parquet",
        ".json": "json",
    }

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize local data loader.

        Args:
            data_dir: Base directory for data files
            cache_dir: Directory for caching processed data
        """
        super().__init__(cache_dir)
        self.data_dir = Path(data_dir) if data_dir else Path(".")
        self._loaded_files: dict[str, pd.DataFrame] = {}

    @property
    def source_name(self) -> str:
        return "Local"

    def get_series(
        self,
        series_id: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs
    ) -> pd.Series:
        """Fetch a series from local data.

        Args:
            series_id: Series identifier in format "filename:column"
                      e.g., "rent_data:effective_rent"
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            pandas Series with datetime index
        """
        parts = series_id.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid series_id format: {series_id}. "
                "Expected 'filename:column' (e.g., 'rent_data:effective_rent')"
            )

        filename = parts[0]
        column = ":".join(parts[1:])  # Handle colons in column names

        df = self.load_file(filename)

        if df.empty:
            return pd.Series(name=series_id, dtype=float)

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in {filename}. "
                f"Available columns: {list(df.columns)}"
            )

        series = df[column]
        series.name = series_id

        # Apply date filters if index is datetime
        if isinstance(series.index, pd.DatetimeIndex):
            if start_date:
                series = series[series.index >= pd.to_datetime(start_date)]
            if end_date:
                series = series[series.index <= pd.to_datetime(end_date)]

        return series

    def load_file(
        self,
        filename: str,
        date_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load a data file.

        Args:
            filename: Name of file (with or without extension)
            date_column: Column to use as datetime index
            **kwargs: Additional arguments passed to pandas reader

        Returns:
            DataFrame with data
        """
        # Check cache
        if filename in self._loaded_files:
            return self._loaded_files[filename]

        # Find the file
        filepath = self._find_file(filename)
        if filepath is None:
            print(f"Warning: File not found: {filename}")
            return pd.DataFrame()

        # Load based on format
        suffix = filepath.suffix.lower()
        format_type = self.SUPPORTED_FORMATS.get(suffix)

        if format_type == "csv":
            df = pd.read_csv(filepath, **kwargs)
        elif format_type == "excel":
            df = pd.read_excel(filepath, **kwargs)
        elif format_type == "parquet":
            df = pd.read_parquet(filepath, **kwargs)
        elif format_type == "json":
            df = pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Set datetime index
        df = self._set_datetime_index(df, date_column)

        # Cache the result
        self._loaded_files[filename] = df

        return df

    def _find_file(self, filename: str) -> Optional[Path]:
        """Find a file in the data directory."""
        # Check exact path first
        filepath = self.data_dir / filename
        if filepath.exists():
            return filepath

        # Try adding extensions
        for ext in self.SUPPORTED_FORMATS.keys():
            filepath = self.data_dir / f"{filename}{ext}"
            if filepath.exists():
                return filepath

        # Search recursively
        for ext in self.SUPPORTED_FORMATS.keys():
            matches = list(self.data_dir.rglob(f"{filename}{ext}"))
            if matches:
                return matches[0]
            matches = list(self.data_dir.rglob(f"*{filename}*{ext}"))
            if matches:
                return matches[0]

        return None

    def _set_datetime_index(
        self,
        df: pd.DataFrame,
        date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Set datetime index on DataFrame."""
        # Try specified column first
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.dropna(subset=[date_column])
            df = df.set_index(date_column).sort_index()
            return df

        # Try common date column names
        date_cols = ["date", "Date", "DATE", "period", "Period", "timestamp", "time"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().any():
                    df = df.dropna(subset=[col])
                    df = df.set_index(col).sort_index()
                    return df

        # Try first column if it looks like dates
        first_col = df.columns[0]
        try:
            dates = pd.to_datetime(df[first_col], errors="coerce")
            if dates.notna().sum() > len(df) * 0.5:  # >50% valid dates
                df[first_col] = dates
                df = df.dropna(subset=[first_col])
                df = df.set_index(first_col).sort_index()
        except Exception:
            pass

        return df

    def register_dataframe(
        self,
        name: str,
        df: pd.DataFrame
    ) -> None:
        """Register a DataFrame directly.

        Useful for programmatically created data.

        Args:
            name: Name to register under
            df: DataFrame to register
        """
        self._loaded_files[name] = df

    def list_available_files(self) -> list[str]:
        """List available data files."""
        files = []
        for ext in self.SUPPORTED_FORMATS.keys():
            files.extend(self.data_dir.glob(f"*{ext}"))
        return [f.stem for f in files]

    def get_file_info(self, filename: str) -> dict:
        """Get information about a data file.

        Args:
            filename: Name of file

        Returns:
            Dictionary with file metadata
        """
        filepath = self._find_file(filename)
        if filepath is None:
            return {"error": f"File not found: {filename}"}

        df = self.load_file(filename)

        return {
            "path": str(filepath),
            "format": filepath.suffix,
            "rows": len(df),
            "columns": list(df.columns),
            "index_type": str(type(df.index).__name__),
            "date_range": (
                str(df.index.min()),
                str(df.index.max())
            ) if isinstance(df.index, pd.DatetimeIndex) else None,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

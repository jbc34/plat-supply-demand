"""Temporal alignment of data across sources.

Handles frequency conversion, lead/lag alignment, and date synchronization.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class AlignmentConfig:
    """Configuration for temporal alignment."""

    target_frequency: str = "M"  # M=Monthly, Q=Quarterly, D=Daily
    aggregation_method: str = "last"  # last, first, mean, sum
    fill_method: str = "ffill"  # ffill, bfill, interpolate
    max_gap: int = 3  # Maximum periods to fill


class TemporalAligner:
    """Aligns time series data to common frequency and dates.

    Handles:
    - Frequency conversion (daily to monthly, quarterly to monthly, etc.)
    - Lead/lag adjustments for proxies
    - Date synchronization across sources
    - Missing value interpolation

    Example:
        >>> aligner = TemporalAligner()
        >>> aligned_df = aligner.align(
        ...     data_dict={"fred": fred_df, "bls": bls_df},
        ...     target_frequency="M"
        ... )
    """

    FREQUENCY_MAP = {
        "D": "D",
        "W": "W-FRI",
        "M": "MS",
        "Q": "QS",
        "A": "AS",
    }

    def __init__(self, config: Optional[AlignmentConfig] = None):
        """Initialize temporal aligner.

        Args:
            config: Alignment configuration (uses defaults if None)
        """
        self.config = config or AlignmentConfig()

    def align(
        self,
        data_dict: dict[str, pd.DataFrame],
        target_frequency: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Align multiple DataFrames to common frequency and dates.

        Args:
            data_dict: Dictionary of source name to DataFrame
            target_frequency: Target frequency (overrides config)
            start_date: Common start date
            end_date: Common end date

        Returns:
            Single aligned DataFrame
        """
        freq = target_frequency or self.config.target_frequency
        freq_str = self.FREQUENCY_MAP.get(freq, "MS")

        aligned_dfs = []

        for source_name, df in data_dict.items():
            if df.empty:
                continue

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    print(f"Warning: Cannot convert index to datetime for {source_name}")
                    continue

            # Resample to target frequency
            aligned = self._resample(df, freq_str)

            # Add source prefix to column names
            aligned.columns = [f"{source_name}_{c}" for c in aligned.columns]

            aligned_dfs.append(aligned)

        if not aligned_dfs:
            return pd.DataFrame()

        # Combine all sources
        result = pd.concat(aligned_dfs, axis=1)

        # Filter to date range
        if start_date:
            result = result[result.index >= pd.to_datetime(start_date)]
        if end_date:
            result = result[result.index <= pd.to_datetime(end_date)]

        # Handle remaining missing values
        result = self._fill_gaps(result)

        return result

    def _resample(
        self,
        df: pd.DataFrame,
        freq: str
    ) -> pd.DataFrame:
        """Resample DataFrame to target frequency."""
        method = self.config.aggregation_method

        if method == "last":
            return df.resample(freq).last()
        elif method == "first":
            return df.resample(freq).first()
        elif method == "mean":
            return df.resample(freq).mean()
        elif method == "sum":
            return df.resample(freq).sum()
        else:
            return df.resample(freq).last()

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in aligned data."""
        result = df.copy()
        method = self.config.fill_method
        max_gap = self.config.max_gap

        for col in result.columns:
            if method == "ffill":
                result[col] = result[col].ffill(limit=max_gap)
            elif method == "bfill":
                result[col] = result[col].bfill(limit=max_gap)
            elif method == "interpolate":
                result[col] = result[col].interpolate(
                    method="time",
                    limit=max_gap
                )

        return result

    def apply_lead_lag(
        self,
        df: pd.DataFrame,
        lead_lag_config: dict[str, int]
    ) -> pd.DataFrame:
        """Apply lead/lag adjustments to columns.

        Args:
            df: Input DataFrame
            lead_lag_config: Dictionary mapping column names to lag values
                            Positive = lag (historical), Negative = lead (future)

        Returns:
            DataFrame with shifted columns

        Example:
            >>> config = {"permits": -18, "employment_growth": -2}
            >>> shifted = aligner.apply_lead_lag(df, config)
        """
        result = df.copy()

        for col, lag in lead_lag_config.items():
            if col in result.columns:
                # Positive lag = shift forward (use past values)
                # Negative lag = shift backward (leading indicator)
                result[col] = result[col].shift(-lag)

        return result

    def create_proxy_matrix(
        self,
        df: pd.DataFrame,
        proxy_config: dict,
        target_dates: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """Create observation matrix from proxies with lead/lag adjustments.

        Args:
            df: Raw data DataFrame
            proxy_config: Dictionary with proxy configurations including lags
            target_dates: Target date index (uses df.index if None)

        Returns:
            Aligned proxy matrix ready for Kalman filter
        """
        if target_dates is None:
            target_dates = df.index

        proxy_data = {}

        for proxy_name, config in proxy_config.items():
            col_name = config.get("column", proxy_name)

            if col_name not in df.columns:
                print(f"Warning: Column '{col_name}' not found for proxy '{proxy_name}'")
                continue

            series = df[col_name].copy()

            # Apply lead/lag
            lag = config.get("lag", 0)
            if lag != 0:
                series = series.shift(-lag)

            # Apply weight
            weight = config.get("weight", 1.0)
            series = series * weight

            # Reindex to target dates
            series = series.reindex(target_dates)

            proxy_data[proxy_name] = series

        return pd.DataFrame(proxy_data, index=target_dates)

    def compute_cross_correlations(
        self,
        df: pd.DataFrame,
        max_lag: int = 24,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Compute cross-correlations to identify lead/lag relationships.

        Args:
            df: Input DataFrame
            max_lag: Maximum lag to compute (both directions)
            target_col: Target column (computes all pairwise if None)

        Returns:
            DataFrame with correlation values at each lag
        """
        results = []

        cols = df.columns.tolist()
        if target_col:
            pairs = [(c, target_col) for c in cols if c != target_col]
        else:
            pairs = [(c1, c2) for i, c1 in enumerate(cols)
                     for c2 in cols[i+1:]]

        for col1, col2 in pairs:
            s1 = df[col1].dropna()
            s2 = df[col2].dropna()

            # Align series
            common_idx = s1.index.intersection(s2.index)
            s1 = s1.loc[common_idx]
            s2 = s2.loc[common_idx]

            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = s1.corr(s2)
                elif lag > 0:
                    corr = s1.iloc[lag:].reset_index(drop=True).corr(
                        s2.iloc[:-lag].reset_index(drop=True)
                    )
                else:
                    corr = s1.iloc[:lag].reset_index(drop=True).corr(
                        s2.iloc[-lag:].reset_index(drop=True)
                    )

                results.append({
                    "series1": col1,
                    "series2": col2,
                    "lag": lag,
                    "correlation": corr
                })

        return pd.DataFrame(results)

    def find_optimal_lags(
        self,
        df: pd.DataFrame,
        target_col: str,
        max_lag: int = 24
    ) -> dict[str, int]:
        """Find optimal lead/lag for each column relative to target.

        Args:
            df: Input DataFrame
            target_col: Target column to align others to
            max_lag: Maximum lag to search

        Returns:
            Dictionary mapping column names to optimal lag values
        """
        correlations = self.compute_cross_correlations(
            df, max_lag=max_lag, target_col=target_col
        )

        optimal_lags = {}

        for col in df.columns:
            if col == target_col:
                continue

            col_corrs = correlations[correlations["series1"] == col]
            if col_corrs.empty:
                col_corrs = correlations[correlations["series2"] == col]

            if not col_corrs.empty:
                # Find lag with maximum absolute correlation
                idx = col_corrs["correlation"].abs().idxmax()
                optimal_lags[col] = col_corrs.loc[idx, "lag"]

        return optimal_lags

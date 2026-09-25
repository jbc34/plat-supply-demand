"""Data cleaning and validation.

Handles missing values, outliers, and data quality issues.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""

    # Missing value handling
    max_missing_pct: float = 0.2  # Max 20% missing
    fill_method: str = "interpolate"  # interpolate, ffill, bfill, drop

    # Outlier handling
    outlier_method: str = "iqr"  # iqr, zscore, mad
    outlier_threshold: float = 3.0  # IQR multiplier or z-score

    # Stationarity
    detrend: bool = False
    difference: bool = False

    # Transformations
    log_transform: bool = False
    standardize: bool = True


class DataCleaner:
    """Cleans and validates time series data.

    Handles common data quality issues including:
    - Missing values (interpolation, forward/backward fill)
    - Outliers (IQR, z-score, MAD methods)
    - Transformations (log, differencing, standardization)

    Example:
        >>> cleaner = DataCleaner()
        >>> clean_df = cleaner.clean(raw_df)
        >>> report = cleaner.generate_quality_report(clean_df)
    """

    def __init__(self, config: Optional[CleaningConfig] = None):
        """Initialize data cleaner.

        Args:
            config: Cleaning configuration (uses defaults if None)
        """
        self.config = config or CleaningConfig()

    def clean(
        self,
        df: pd.DataFrame,
        config: Optional[CleaningConfig] = None
    ) -> pd.DataFrame:
        """Clean a DataFrame.

        Args:
            df: Input DataFrame with datetime index
            config: Override default configuration

        Returns:
            Cleaned DataFrame
        """
        cfg = config or self.config
        result = df.copy()

        # Handle missing values
        result = self._handle_missing(result, cfg)

        # Handle outliers
        result = self._handle_outliers(result, cfg)

        # Apply transformations
        if cfg.log_transform:
            result = self._log_transform(result)

        if cfg.detrend:
            result = self._detrend(result)

        if cfg.difference:
            result = self._difference(result)

        if cfg.standardize:
            result = self._standardize(result)

        return result

    def _handle_missing(
        self,
        df: pd.DataFrame,
        cfg: CleaningConfig
    ) -> pd.DataFrame:
        """Handle missing values."""
        result = df.copy()

        for col in result.columns:
            missing_pct = result[col].isna().mean()

            # Drop columns with too many missing values
            if missing_pct > cfg.max_missing_pct:
                print(
                    f"Warning: Column '{col}' has {missing_pct:.1%} missing "
                    f"values (threshold: {cfg.max_missing_pct:.1%})"
                )

            # Fill remaining missing values
            if cfg.fill_method == "interpolate":
                result[col] = result[col].interpolate(method="time")
            elif cfg.fill_method == "ffill":
                result[col] = result[col].ffill()
            elif cfg.fill_method == "bfill":
                result[col] = result[col].bfill()
            elif cfg.fill_method == "drop":
                pass  # Leave NaNs for later handling

        return result

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        cfg: CleaningConfig
    ) -> pd.DataFrame:
        """Handle outliers."""
        result = df.copy()

        for col in result.columns:
            if cfg.outlier_method == "iqr":
                result[col] = self._clip_iqr(
                    result[col],
                    multiplier=cfg.outlier_threshold
                )
            elif cfg.outlier_method == "zscore":
                result[col] = self._clip_zscore(
                    result[col],
                    threshold=cfg.outlier_threshold
                )
            elif cfg.outlier_method == "mad":
                result[col] = self._clip_mad(
                    result[col],
                    threshold=cfg.outlier_threshold
                )

        return result

    def _clip_iqr(
        self,
        series: pd.Series,
        multiplier: float = 1.5
    ) -> pd.Series:
        """Clip outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        return series.clip(lower, upper)

    def _clip_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """Clip outliers using z-score method."""
        mean = series.mean()
        std = series.std()

        if std == 0:
            return series

        lower = mean - threshold * std
        upper = mean + threshold * std

        return series.clip(lower, upper)

    def _clip_mad(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """Clip outliers using Median Absolute Deviation."""
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return series

        # MAD to standard deviation conversion
        mad_std = mad * 1.4826

        lower = median - threshold * mad_std
        upper = median + threshold * mad_std

        return series.clip(lower, upper)

    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to positive columns."""
        result = df.copy()

        for col in result.columns:
            if (result[col] > 0).all():
                result[col] = np.log(result[col])
            elif (result[col] >= 0).all():
                # Add small constant for zeros
                result[col] = np.log(result[col] + 1e-10)

        return result

    def _detrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linear trend from each column."""
        result = df.copy()

        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) < 2:
                continue

            x = np.arange(len(valid))
            coeffs = np.polyfit(x, valid.values, 1)
            trend = np.polyval(coeffs, x)

            result.loc[valid.index, col] = valid.values - trend

        return result

    def _difference(self, df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Apply differencing."""
        return df.diff(periods).dropna()

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize to zero mean and unit variance."""
        return (df - df.mean()) / df.std()

    def generate_quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with quality metrics per column
        """
        report = []

        for col in df.columns:
            series = df[col]

            report.append({
                "column": col,
                "count": len(series),
                "missing": series.isna().sum(),
                "missing_pct": series.isna().mean() * 100,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "skew": series.skew(),
                "kurtosis": series.kurtosis(),
                "zeros": (series == 0).sum(),
                "negative": (series < 0).sum(),
            })

        return pd.DataFrame(report).set_index("column")

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Detect anomalous values.

        Args:
            df: DataFrame to analyze
            method: Detection method (zscore, iqr, mad)
            threshold: Detection threshold

        Returns:
            Boolean DataFrame indicating anomalies
        """
        anomalies = pd.DataFrame(
            index=df.index,
            columns=df.columns,
            dtype=bool
        )

        for col in df.columns:
            series = df[col]

            if method == "zscore":
                z = (series - series.mean()) / series.std()
                anomalies[col] = np.abs(z) > threshold

            elif method == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                anomalies[col] = (
                    (series < q1 - threshold * iqr) |
                    (series > q3 + threshold * iqr)
                )

            elif method == "mad":
                median = series.median()
                mad = np.median(np.abs(series - median)) * 1.4826
                anomalies[col] = np.abs(series - median) > threshold * mad

        return anomalies

"""Filtering methods for trend and cycle extraction.

Provides HP filter, bandpass filters, and other smoothing methods
for separating trend from cycle in time series.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass
class FilterResult:
    """Result of filtering operation."""

    trend: pd.Series
    cycle: pd.Series
    original: pd.Series


class HPFilter:
    """Hodrick-Prescott filter for trend-cycle decomposition.

    Separates a time series into trend and cyclical components
    by minimizing:
        sum((y_t - tau_t)^2) + lambda * sum((tau_{t+1} - 2*tau_t + tau_{t-1})^2)

    where tau is the trend and lambda controls smoothness.
    """

    # Standard lambda values by frequency
    LAMBDA_VALUES = {
        "D": 10_000_000,   # Daily
        "W": 270_400,      # Weekly
        "M": 129_600,      # Monthly
        "Q": 1_600,        # Quarterly
        "A": 6.25,         # Annual
    }

    def __init__(self, lambda_param: Optional[float] = None):
        """Initialize HP filter.

        Args:
            lambda_param: Smoothing parameter. If None, determined by frequency.
        """
        self.lambda_param = lambda_param

    def filter(
        self,
        series: pd.Series,
        frequency: Optional[str] = None,
    ) -> FilterResult:
        """Apply HP filter to time series.

        Args:
            series: Input time series
            frequency: Data frequency for lambda selection

        Returns:
            FilterResult with trend and cycle components
        """
        y = series.dropna().values
        n = len(y)

        if n < 4:
            raise ValueError("Series must have at least 4 observations")

        # Determine lambda
        if self.lambda_param is not None:
            lam = self.lambda_param
        elif frequency:
            lam = self.LAMBDA_VALUES.get(frequency.upper(), 129_600)
        else:
            # Try to infer from index
            if isinstance(series.index, pd.DatetimeIndex):
                freq = pd.infer_freq(series.index)
                if freq:
                    freq_letter = freq[0].upper()
                    lam = self.LAMBDA_VALUES.get(freq_letter, 129_600)
                else:
                    lam = 129_600
            else:
                lam = 129_600

        # Construct the second-difference matrix
        diag_data = np.ones((3, n))
        diag_data[0, 2:] = 1
        diag_data[1, 1:] = -2
        diag_data[2, :] = 1

        D = diags(diag_data, [0, 1, 2], shape=(n-2, n))

        # Solve for trend
        # (I + lambda * D'D) * tau = y
        A = diags(np.ones(n), 0) + lam * (D.T @ D)
        trend = spsolve(A.tocsc(), y)

        # Cycle is residual
        cycle = y - trend

        # Create output series
        trend_series = pd.Series(trend, index=series.dropna().index, name="trend")
        cycle_series = pd.Series(cycle, index=series.dropna().index, name="cycle")

        return FilterResult(
            trend=trend_series,
            cycle=cycle_series,
            original=series.dropna()
        )


class BandpassFilter:
    """Bandpass filter for extracting cycles of specific frequencies.

    Extracts cyclical component within a specified frequency band,
    useful for isolating business cycle frequencies.
    """

    def __init__(
        self,
        low_period: int = 6,
        high_period: int = 32,
        order: int = 3,
    ):
        """Initialize bandpass filter.

        Args:
            low_period: Minimum cycle period (periods)
            high_period: Maximum cycle period (periods)
            order: Filter order
        """
        self.low_period = low_period
        self.high_period = high_period
        self.order = order

    def filter(
        self,
        series: pd.Series,
        periods_per_year: int = 12,
    ) -> FilterResult:
        """Apply bandpass filter.

        Args:
            series: Input time series
            periods_per_year: Number of observations per year

        Returns:
            FilterResult with trend and cycle components
        """
        y = series.dropna().values

        # Convert periods to frequencies
        low_freq = 2.0 / self.high_period
        high_freq = 2.0 / self.low_period

        # Design Butterworth bandpass filter
        b, a = signal.butter(
            self.order,
            [low_freq, high_freq],
            btype="band"
        )

        # Apply filter (forward-backward to avoid phase shift)
        cycle = signal.filtfilt(b, a, y)

        # Trend is original minus cycle
        trend = y - cycle

        trend_series = pd.Series(trend, index=series.dropna().index, name="trend")
        cycle_series = pd.Series(cycle, index=series.dropna().index, name="cycle")

        return FilterResult(
            trend=trend_series,
            cycle=cycle_series,
            original=series.dropna()
        )


class TrendExtractor:
    """Flexible trend extraction using multiple methods."""

    def __init__(self, method: str = "hp"):
        """Initialize trend extractor.

        Args:
            method: Extraction method:
                - "hp": Hodrick-Prescott filter
                - "bandpass": Bandpass filter
                - "lowess": LOWESS smoother
                - "moving_average": Simple moving average
                - "exponential": Exponential smoothing
        """
        self.method = method

    def extract(
        self,
        series: pd.Series,
        **kwargs
    ) -> FilterResult:
        """Extract trend from series.

        Args:
            series: Input time series
            **kwargs: Method-specific parameters

        Returns:
            FilterResult with trend and cycle
        """
        if self.method == "hp":
            return self._hp_filter(series, **kwargs)
        elif self.method == "bandpass":
            return self._bandpass_filter(series, **kwargs)
        elif self.method == "lowess":
            return self._lowess_smooth(series, **kwargs)
        elif self.method == "moving_average":
            return self._moving_average(series, **kwargs)
        elif self.method == "exponential":
            return self._exponential_smooth(series, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _hp_filter(self, series: pd.Series, **kwargs) -> FilterResult:
        """Apply HP filter."""
        hp = HPFilter(lambda_param=kwargs.get("lambda_param"))
        return hp.filter(series, frequency=kwargs.get("frequency"))

    def _bandpass_filter(self, series: pd.Series, **kwargs) -> FilterResult:
        """Apply bandpass filter."""
        bp = BandpassFilter(
            low_period=kwargs.get("low_period", 6),
            high_period=kwargs.get("high_period", 32),
            order=kwargs.get("order", 3),
        )
        return bp.filter(series, periods_per_year=kwargs.get("periods_per_year", 12))

    def _lowess_smooth(self, series: pd.Series, **kwargs) -> FilterResult:
        """Apply LOWESS smoother."""
        from statsmodels.nonparametric.smoothers_lowess import lowess

        y = series.dropna().values
        x = np.arange(len(y))

        frac = kwargs.get("frac", 0.3)
        smoothed = lowess(y, x, frac=frac, return_sorted=False)

        trend = pd.Series(smoothed, index=series.dropna().index, name="trend")
        cycle = series.dropna() - trend
        cycle.name = "cycle"

        return FilterResult(
            trend=trend,
            cycle=cycle,
            original=series.dropna()
        )

    def _moving_average(self, series: pd.Series, **kwargs) -> FilterResult:
        """Apply centered moving average."""
        window = kwargs.get("window", 12)

        trend = series.rolling(window=window, center=True).mean()
        cycle = series - trend

        trend.name = "trend"
        cycle.name = "cycle"

        return FilterResult(
            trend=trend.dropna(),
            cycle=cycle.dropna(),
            original=series
        )

    def _exponential_smooth(self, series: pd.Series, **kwargs) -> FilterResult:
        """Apply exponential smoothing."""
        alpha = kwargs.get("alpha", 0.3)

        trend = series.ewm(alpha=alpha).mean()
        cycle = series - trend

        trend.name = "trend"
        cycle.name = "cycle"

        return FilterResult(
            trend=trend,
            cycle=cycle,
            original=series
        )

    def compute_deviation_from_trend(
        self,
        series: pd.Series,
        **kwargs
    ) -> pd.Series:
        """Compute percentage deviation from trend.

        Args:
            series: Input time series
            **kwargs: Method parameters

        Returns:
            Series of percentage deviations
        """
        result = self.extract(series, **kwargs)
        deviation = (result.original - result.trend) / result.trend * 100
        deviation.name = "deviation_pct"
        return deviation

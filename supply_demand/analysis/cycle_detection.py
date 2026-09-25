"""Cycle detection and turning point identification."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class CycleInfo:
    """Information about a detected cycle."""

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    duration_months: int
    expansion_months: int
    contraction_months: int
    peak_amplitude: float
    trough_amplitude: float


def detect_cycles(
    series: pd.Series,
    min_cycle_length: int = 12,
    min_amplitude: float = 0.1,
    smooth_window: int = 3,
) -> List[CycleInfo]:
    """Detect cycles in a time series.

    Uses a combination of local extrema detection and
    amplitude thresholding.

    Args:
        series: Time series with datetime index
        min_cycle_length: Minimum months for a valid cycle
        min_amplitude: Minimum peak-to-trough amplitude
        smooth_window: Smoothing window for noise reduction

    Returns:
        List of CycleInfo objects
    """
    # Smooth the series
    if smooth_window > 1:
        smoothed = series.rolling(window=smooth_window, center=True).mean()
    else:
        smoothed = series

    smoothed = smoothed.dropna()
    values = smoothed.values
    dates = smoothed.index

    # Find peaks and troughs
    peaks, peak_props = signal.find_peaks(values, distance=min_cycle_length // 2)
    troughs, trough_props = signal.find_peaks(-values, distance=min_cycle_length // 2)

    # Match peaks with subsequent troughs
    cycles = []

    for peak_idx in peaks:
        # Find next trough after this peak
        next_troughs = troughs[troughs > peak_idx]
        if len(next_troughs) == 0:
            continue

        trough_idx = next_troughs[0]

        # Check minimum amplitude
        amplitude = values[peak_idx] - values[trough_idx]
        if amplitude < min_amplitude:
            continue

        # Check minimum duration
        duration = trough_idx - peak_idx
        if duration < min_cycle_length // 2:
            continue

        # Find previous trough for expansion measurement
        prev_troughs = troughs[troughs < peak_idx]
        if len(prev_troughs) > 0:
            prev_trough_idx = prev_troughs[-1]
            expansion_months = peak_idx - prev_trough_idx
        else:
            expansion_months = peak_idx

        contraction_months = trough_idx - peak_idx

        cycles.append(CycleInfo(
            peak_date=dates[peak_idx],
            trough_date=dates[trough_idx],
            duration_months=expansion_months + contraction_months,
            expansion_months=expansion_months,
            contraction_months=contraction_months,
            peak_amplitude=values[peak_idx],
            trough_amplitude=values[trough_idx],
        ))

    return cycles


def detect_turning_points(
    series: pd.Series,
    threshold: float = 0.0,
    min_duration: int = 6,
) -> pd.DataFrame:
    """Detect peaks and troughs in a series.

    Args:
        series: Input time series
        threshold: Minimum change from peak to qualify
        min_duration: Minimum periods between turning points

    Returns:
        DataFrame with turning point information
    """
    values = series.dropna().values
    dates = series.dropna().index

    # Find local maxima and minima
    peaks, _ = signal.find_peaks(values, distance=min_duration)
    troughs, _ = signal.find_peaks(-values, distance=min_duration)

    turning_points = []

    for idx in peaks:
        turning_points.append({
            "date": dates[idx],
            "type": "peak",
            "value": values[idx],
            "index": idx,
        })

    for idx in troughs:
        turning_points.append({
            "date": dates[idx],
            "type": "trough",
            "value": values[idx],
            "index": idx,
        })

    df = pd.DataFrame(turning_points)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def estimate_cycle_position(
    series: pd.Series,
    lookback: int = 60,
) -> dict:
    """Estimate current position in the cycle.

    Args:
        series: Time series ending at current date
        lookback: Months to look back for cycle analysis

    Returns:
        Dictionary with position estimates
    """
    recent = series.tail(lookback)

    # Find recent peak and trough
    peaks, _ = signal.find_peaks(recent.values, distance=6)
    troughs, _ = signal.find_peaks(-recent.values, distance=6)

    current_value = recent.iloc[-1]
    current_date = recent.index[-1]

    result = {
        "current_value": current_value,
        "current_date": current_date,
    }

    if len(peaks) > 0:
        last_peak_idx = peaks[-1]
        result["last_peak_date"] = recent.index[last_peak_idx]
        result["last_peak_value"] = recent.iloc[last_peak_idx]
        result["months_since_peak"] = len(recent) - 1 - last_peak_idx
        result["pct_from_peak"] = (current_value / recent.iloc[last_peak_idx] - 1) * 100

    if len(troughs) > 0:
        last_trough_idx = troughs[-1]
        result["last_trough_date"] = recent.index[last_trough_idx]
        result["last_trough_value"] = recent.iloc[last_trough_idx]
        result["months_since_trough"] = len(recent) - 1 - last_trough_idx
        result["pct_from_trough"] = (current_value / recent.iloc[last_trough_idx] - 1) * 100

    # Estimate phase
    if len(peaks) > 0 and len(troughs) > 0:
        if peaks[-1] > troughs[-1]:
            # Most recent turning point was a peak
            result["phase"] = "contraction"
        else:
            # Most recent turning point was a trough
            result["phase"] = "expansion"
    else:
        result["phase"] = "unknown"

    return result

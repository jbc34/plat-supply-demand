"""Spiral-based cycle metrics and analysis.

Computes various metrics from the spiral representation including
cycle duration, amplitude, and turning point detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CycleMetrics:
    """Metrics for a single cycle."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_months: int
    peak_amplitude: float
    trough_amplitude: float
    mean_angular_velocity: float
    expansion_duration: int
    contraction_duration: int


@dataclass
class TurningPoint:
    """A cycle turning point."""

    date: pd.Timestamp
    type: str  # "peak" or "trough"
    amplitude: float
    phase: float
    confidence: float


def compute_cycle_metrics(
    r: np.ndarray,
    theta: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> List[CycleMetrics]:
    """Compute metrics for each complete cycle.

    A complete cycle is defined as phase going from 0 to 2π.

    Args:
        r: Amplitude time series
        theta: Phase time series (in radians, possibly wrapped)
        dates: Optional datetime index

    Returns:
        List of CycleMetrics for each complete cycle
    """
    if dates is None:
        dates = pd.date_range(start="2000-01-01", periods=len(r), freq="MS")

    # Unwrap phase to find cycle boundaries
    theta_unwrapped = np.unwrap(theta)

    # Find complete cycles (2π crossings)
    cycle_count = theta_unwrapped // (2 * np.pi)
    cycle_changes = np.where(np.diff(cycle_count) > 0)[0]

    cycles = []

    for i in range(len(cycle_changes) - 1):
        start_idx = cycle_changes[i]
        end_idx = cycle_changes[i + 1]

        cycle_r = r[start_idx:end_idx + 1]
        cycle_theta = theta[start_idx:end_idx + 1]

        # Expansion: phase 0 to π
        # Contraction: phase π to 2π
        cycle_theta_wrapped = cycle_theta % (2 * np.pi)
        expansion_mask = cycle_theta_wrapped < np.pi
        contraction_mask = ~expansion_mask

        metrics = CycleMetrics(
            start_date=dates[start_idx],
            end_date=dates[end_idx],
            duration_months=end_idx - start_idx,
            peak_amplitude=np.max(cycle_r),
            trough_amplitude=np.min(cycle_r),
            mean_angular_velocity=2 * np.pi / (end_idx - start_idx),
            expansion_duration=expansion_mask.sum(),
            contraction_duration=contraction_mask.sum(),
        )
        cycles.append(metrics)

    return cycles


def compute_turning_points(
    r: np.ndarray,
    theta: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    min_prominence: float = 0.1,
) -> List[TurningPoint]:
    """Identify cycle turning points (peaks and troughs).

    Turning points are identified based on:
    - Phase crossing π (peak) and 0/2π (trough)
    - Local amplitude extrema

    Args:
        r: Amplitude time series
        theta: Phase time series
        dates: Optional datetime index
        min_prominence: Minimum prominence for peak detection

    Returns:
        List of TurningPoint objects
    """
    from scipy.signal import find_peaks

    if dates is None:
        dates = pd.date_range(start="2000-01-01", periods=len(r), freq="MS")

    turning_points = []
    theta_wrapped = theta % (2 * np.pi)

    # Find peaks in amplitude (cycle highs)
    peaks, properties = find_peaks(r, prominence=min_prominence)

    for idx in peaks:
        # Check if this is near the cycle peak (phase ≈ π/2)
        phase_at_peak = theta_wrapped[idx]
        is_expansion_peak = np.pi / 4 < phase_at_peak < 3 * np.pi / 4

        if is_expansion_peak:
            confidence = properties["prominences"][list(peaks).index(idx)]
            confidence = min(1.0, confidence / np.std(r))

            turning_points.append(TurningPoint(
                date=dates[idx],
                type="peak",
                amplitude=r[idx],
                phase=phase_at_peak,
                confidence=confidence,
            ))

    # Find troughs in amplitude (cycle lows)
    troughs, properties = find_peaks(-r, prominence=min_prominence)

    for idx in troughs:
        # Check if this is near the cycle trough (phase ≈ 3π/2)
        phase_at_trough = theta_wrapped[idx]
        is_contraction_trough = 5 * np.pi / 4 < phase_at_trough < 7 * np.pi / 4

        if is_contraction_trough:
            confidence = properties["prominences"][list(troughs).index(idx)]
            confidence = min(1.0, confidence / np.std(r))

            turning_points.append(TurningPoint(
                date=dates[idx],
                type="trough",
                amplitude=r[idx],
                phase=phase_at_trough,
                confidence=confidence,
            ))

    # Sort by date
    turning_points.sort(key=lambda x: x.date)

    return turning_points


def compute_current_position_metrics(
    r: float,
    theta: float,
    r_dot: float,
    theta_dot: float,
    historical_r: Optional[np.ndarray] = None,
) -> dict:
    """Compute interpretable metrics for current cycle position.

    Args:
        r: Current amplitude
        theta: Current phase
        r_dot: Amplitude velocity
        theta_dot: Phase velocity
        historical_r: Historical amplitude for percentile calc

    Returns:
        Dictionary of interpretable metrics
    """
    theta_deg = np.degrees(theta) % 360

    # Determine cycle stage
    if 0 <= theta_deg < 45:
        stage = "Early Recovery"
        stage_progress = theta_deg / 45
    elif 45 <= theta_deg < 90:
        stage = "Late Recovery"
        stage_progress = (theta_deg - 45) / 45
    elif 90 <= theta_deg < 135:
        stage = "Early Expansion"
        stage_progress = (theta_deg - 90) / 45
    elif 135 <= theta_deg < 180:
        stage = "Late Expansion"
        stage_progress = (theta_deg - 135) / 45
    elif 180 <= theta_deg < 225:
        stage = "Peak / Early Decline"
        stage_progress = (theta_deg - 180) / 45
    elif 225 <= theta_deg < 270:
        stage = "Mid Contraction"
        stage_progress = (theta_deg - 225) / 45
    elif 270 <= theta_deg < 315:
        stage = "Late Contraction"
        stage_progress = (theta_deg - 270) / 45
    else:
        stage = "Trough / Early Recovery"
        stage_progress = (theta_deg - 315) / 45

    # Amplitude interpretation
    if historical_r is not None:
        amplitude_percentile = (historical_r < r).mean() * 100
    else:
        amplitude_percentile = None

    # Velocity interpretation
    if r_dot > 0:
        amplitude_trend = "Increasing"
    elif r_dot < 0:
        amplitude_trend = "Decreasing"
    else:
        amplitude_trend = "Stable"

    # Time to next phase
    theta_dot_monthly = theta_dot  # Assuming monthly data
    if theta_dot_monthly > 0:
        months_to_next_quadrant = (90 - (theta_deg % 90)) / np.degrees(theta_dot_monthly)
    else:
        months_to_next_quadrant = float('inf')

    return {
        "stage": stage,
        "stage_progress_pct": stage_progress * 100,
        "phase_degrees": theta_deg,
        "amplitude": r,
        "amplitude_percentile": amplitude_percentile,
        "amplitude_trend": amplitude_trend,
        "amplitude_change_rate": r_dot,
        "cycle_velocity": np.degrees(theta_dot),  # degrees per period
        "months_to_next_quadrant": months_to_next_quadrant,
    }


def compute_distance_to_equilibrium(
    r: float,
    historical_r: Optional[np.ndarray] = None,
) -> dict:
    """Compute distance from equilibrium metrics.

    Args:
        r: Current amplitude (distance from equilibrium)
        historical_r: Historical amplitudes for context

    Returns:
        Dictionary of distance metrics
    """
    if historical_r is not None:
        percentile = (historical_r < r).mean() * 100
        z_score = (r - np.mean(historical_r)) / (np.std(historical_r) + 1e-10)
        max_historical = np.max(historical_r)
        pct_of_max = r / max_historical * 100 if max_historical > 0 else 0
    else:
        percentile = None
        z_score = None
        pct_of_max = None

    return {
        "amplitude": r,
        "percentile": percentile,
        "z_score": z_score,
        "pct_of_historical_max": pct_of_max,
    }

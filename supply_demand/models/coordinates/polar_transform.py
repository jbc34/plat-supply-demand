"""Polar coordinate transformation for market state.

Transform estimated state from Cartesian (D, S, K) space
to polar/cylindrical coordinates (r, θ) representing
position in the market cycle spiral.

The spiral representation captures:
- r (amplitude): How far from equilibrium
- θ (phase): Where in the cycle (expansion, peak, contraction, recovery)
- ṙ (radial velocity): Is the cycle intensifying or moderating?
- θ̇ (angular velocity): How fast is the cycle progressing?

Recommended approach (from cycle extraction study):
- Use D-S spread as primary cycle signal (economically grounded)
- Apply bandpass filter (48-84 months) to isolate business cycle
- See research/01_cycle_extraction_study.md for methodology details
"""

from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt


# Default bandpass filter parameters (from EXP_005 sensitivity analysis)
DEFAULT_BANDPASS_LOW_PERIOD = 84   # months (7 years)
DEFAULT_BANDPASS_HIGH_PERIOD = 48  # months (4 years)
DEFAULT_BANDPASS_ORDER = 2


class SpiralCoordinates(NamedTuple):
    """Spiral coordinate representation of market state."""

    r: float           # Radial distance from equilibrium (amplitude)
    theta: float       # Angular position in cycle (phase, 0-2π)
    r_dot: float       # Radial velocity (amplitude change rate)
    theta_dot: float   # Angular velocity (cycle speed)
    quadrant: str      # Current cycle quadrant name
    confidence: float  # Confidence in phase estimate (0-1)


# Cycle quadrant definitions
CYCLE_QUADRANTS = {
    (0, 90): "Early Expansion",
    (90, 180): "Peak / Late Expansion",
    (180, 270): "Contraction",
    (270, 360): "Recovery / Early Cycle",
}


def phase_to_quadrant(theta: float) -> str:
    """Map phase angle to cycle quadrant name.

    Args:
        theta: Phase angle in radians (0 to 2π)

    Returns:
        Quadrant name string
    """
    theta_deg = np.degrees(theta) % 360

    for (low, high), name in CYCLE_QUADRANTS.items():
        if low <= theta_deg < high:
            return name

    return "Recovery / Early Cycle"


def cartesian_to_polar(
    x: float,
    y: float,
) -> Tuple[float, float]:
    """Convert Cartesian coordinates to polar.

    Args:
        x: x-coordinate
        y: y-coordinate

    Returns:
        Tuple of (r, theta) where theta is in [0, 2π)
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    # Normalize to [0, 2π)
    if theta < 0:
        theta += 2 * np.pi

    return r, theta


def polar_to_cartesian(
    r: float,
    theta: float,
) -> Tuple[float, float]:
    """Convert polar coordinates to Cartesian.

    Args:
        r: Radius
        theta: Angle in radians

    Returns:
        Tuple of (x, y)
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def bandpass_filter(
    x: np.ndarray,
    low_period: float = DEFAULT_BANDPASS_LOW_PERIOD,
    high_period: float = DEFAULT_BANDPASS_HIGH_PERIOD,
    fs: float = 12.0,
    order: int = DEFAULT_BANDPASS_ORDER,
) -> np.ndarray:
    """Apply bandpass filter to isolate business cycle frequency band.

    Filters the input signal to retain only frequencies corresponding
    to cycles between high_period and low_period months.

    Args:
        x: Input time series
        low_period: Longest cycle period to retain (months), default 84 (7 years)
        high_period: Shortest cycle period to retain (months), default 48 (4 years)
        fs: Sampling frequency (12 = monthly data)
        order: Butterworth filter order

    Returns:
        Bandpass-filtered time series

    Note:
        Uses zero-phase filtering (filtfilt) to avoid phase distortion.
        Parameters validated in EXP_005 sensitivity analysis.
    """
    # Convert periods to frequencies
    low_freq = 1 / low_period   # Lowest frequency (longest cycle)
    high_freq = 1 / high_period  # Highest frequency (shortest cycle)

    # Normalize to Nyquist frequency
    nyq = fs / 2
    low = low_freq / nyq
    high = high_freq / nyq

    # Ensure valid frequencies (avoid edge cases)
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    # Design and apply filter
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def extract_dominant_cycle(
    D: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    weights: Optional[Tuple[float, float, float]] = None,
    method: str = "ds_spread",
    apply_bandpass: bool = True,
    k_amplitude_weight: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract amplitude and phase of the dominant cycle from state estimates.

    Combines the three state variables into a composite signal and
    extracts instantaneous amplitude and phase.

    Args:
        D: Demand deviation time series
        S: Supply deviation time series
        K: Capital flow deviation time series
        weights: Weights for combining (D, S, K), used by "hilbert" method only.
            Defaults to (0.4, 0.35, 0.25)
        method: Extraction method:
            - "ds_spread" (RECOMMENDED): D-S imbalance as cycle signal, bandpass filtered.
                Economically grounded: excess demand drives expansion, excess supply
                drives contraction. See EXP_003 for validation.
            - "hilbert": Hilbert transform on weighted composite (legacy)
            - "pca": PCA on the state space to extract common cycle
        apply_bandpass: Whether to apply bandpass filter (48-84 months).
            Recommended True to remove seasonality. See EXP_005.
        k_amplitude_weight: How much K modulates amplitude in ds_spread method.
            Default 0.2 means ±20% amplitude modulation based on capital flow.

    Returns:
        Tuple of (amplitude, phase) arrays
        - amplitude: Cycle amplitude over time (r)
        - phase: Cycle phase (0 to 2π) over time (θ)

    Note:
        The "ds_spread" method is recommended based on the cycle extraction study
        (research/01_cycle_extraction_study.md). It provides:
        - Theoretical grounding (demand-supply imbalance mechanism)
        - No arbitrary weight parameters
        - Clear economic interpretation
        - Robustness to parameter variations (EXP_005)
    """
    if method == "ds_spread":
        # RECOMMENDED: D-S spread as primary cycle signal
        # Economic interpretation: D > S = expansion, S > D = contraction

        if apply_bandpass:
            D_filt = bandpass_filter(D)
            S_filt = bandpass_filter(S)
            K_filt = bandpass_filter(K)
        else:
            D_filt = D - np.mean(D)
            S_filt = S - np.mean(S)
            K_filt = K - np.mean(K)

        # Cycle signal is demand-supply imbalance
        cycle_signal = D_filt - S_filt

        # Hilbert transform for analytic signal
        analytic = signal.hilbert(cycle_signal)
        amplitude = np.abs(analytic)
        phase = np.angle(analytic)

        # Unwrap and normalize phase
        phase = np.unwrap(phase)
        phase = phase % (2 * np.pi)

        # K modulates amplitude (capital flow affects cycle intensity)
        if k_amplitude_weight > 0 and np.std(K_filt) > 1e-10:
            k_normalized = K_filt / np.std(K_filt)
            amplitude = amplitude * (1 + k_amplitude_weight * k_normalized)
            amplitude = np.maximum(amplitude, 0)  # Ensure non-negative

    elif method == "hilbert":
        # Legacy method: weighted composite
        if weights is None:
            weights = (0.4, 0.35, 0.25)

        w_d, w_s, w_k = weights

        if apply_bandpass:
            D_filt = bandpass_filter(D)
            S_filt = bandpass_filter(S)
            K_filt = bandpass_filter(K)
        else:
            D_filt = D
            S_filt = S
            K_filt = K

        # Create composite signal
        composite = w_d * D_filt + w_s * S_filt + w_k * K_filt

        # Remove mean
        composite = composite - np.mean(composite)

        # Hilbert transform for analytic signal
        analytic = signal.hilbert(composite)

        amplitude = np.abs(analytic)
        phase = np.angle(analytic)

        # Unwrap phase for continuity
        phase = np.unwrap(phase)

        # Normalize to [0, 2π)
        phase = phase % (2 * np.pi)

    elif method == "pca":
        # PCA method: data-driven weights
        if apply_bandpass:
            D_filt = bandpass_filter(D)
            S_filt = bandpass_filter(S)
            K_filt = bandpass_filter(K)
        else:
            D_filt = D
            S_filt = S
            K_filt = K

        # Stack state variables
        X = np.column_stack([D_filt, S_filt, K_filt])

        # Center
        X_centered = X - X.mean(axis=0)

        # PCA
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # First PC captures dominant cycle
        pc1 = U[:, 0] * s[0]

        # Extract amplitude and phase from PC1
        analytic = signal.hilbert(pc1)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic)) % (2 * np.pi)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ds_spread' (recommended), 'hilbert', or 'pca'.")

    return amplitude, phase


def compute_spiral_coordinates(
    D: Union[float, np.ndarray],
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    D_eq: float = 0.0,
    S_eq: float = 0.0,
    K_eq: float = 0.0,
    method: str = "ds_spread",
    apply_bandpass: bool = True,
) -> SpiralCoordinates:
    """Compute current spiral coordinates from state estimates.

    Args:
        D: Demand estimate (scalar for point, array for time series)
        S: Supply estimate
        K: Capital flow estimate
        D_eq: Demand equilibrium value
        S_eq: Supply equilibrium value
        K_eq: Capital flow equilibrium value
        method: Phase extraction method ("ds_spread" recommended, "hilbert", or "pca")
        apply_bandpass: Whether to apply bandpass filter (recommended True)

    Returns:
        SpiralCoordinates with current position and velocity
    """
    # Deviations from equilibrium
    d = np.asarray(D) - D_eq
    s = np.asarray(S) - S_eq
    k = np.asarray(K) - K_eq

    # Handle time series case
    if isinstance(d, np.ndarray) and d.ndim > 0 and len(d) > 1:
        amplitude, phase = extract_dominant_cycle(d, s, k, method=method, apply_bandpass=apply_bandpass)

        r = amplitude[-1]
        theta = phase[-1]

        # Numerical derivatives for velocities
        if len(amplitude) > 1:
            r_dot = amplitude[-1] - amplitude[-2]

            # Handle phase wraparound
            theta_diff = phase[-1] - phase[-2]
            if theta_diff > np.pi:
                theta_diff -= 2 * np.pi
            elif theta_diff < -np.pi:
                theta_diff += 2 * np.pi
            theta_dot = theta_diff
        else:
            r_dot = 0.0
            theta_dot = 0.0

        # Confidence based on signal clarity
        # Higher amplitude relative to noise = higher confidence
        amplitude_std = np.std(amplitude)
        noise_estimate = np.std(np.diff(amplitude))
        snr = amplitude_std / (noise_estimate + 1e-10)
        confidence = min(1.0, snr / 5.0)

    else:
        # Point estimate: can only compute r, approximate theta
        d = float(d) if isinstance(d, np.ndarray) else d
        s = float(s) if isinstance(s, np.ndarray) else s
        k = float(k) if isinstance(k, np.ndarray) else k

        r = np.sqrt(d ** 2 + s ** 2 + k ** 2)

        # Simplified phase estimate based on S-D relationship
        # When D > S: expansion (0-180°)
        # When S > D: contraction (180-360°)
        excess_demand = d - s
        theta = np.arctan2(excess_demand, k) + np.pi  # Shift to [0, 2π]
        theta = theta % (2 * np.pi)

        r_dot = 0.0
        theta_dot = 0.0
        confidence = 0.3  # Low confidence for point estimate

    # Determine quadrant
    quadrant = phase_to_quadrant(theta)

    return SpiralCoordinates(
        r=float(r),
        theta=float(theta),
        r_dot=float(r_dot),
        theta_dot=float(theta_dot),
        quadrant=quadrant,
        confidence=float(confidence),
    )


def compute_spiral_trajectory(
    D: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    D_eq: float = 0.0,
    S_eq: float = 0.0,
    K_eq: float = 0.0,
    smooth_window: int = 3,
    method: str = "ds_spread",
    apply_bandpass: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute full spiral coordinate trajectory over time.

    Args:
        D, S, K: State time series
        D_eq, S_eq, K_eq: Equilibrium values
        smooth_window: Smoothing window for derivatives
        method: Phase extraction method ("ds_spread" recommended)
        apply_bandpass: Whether to apply bandpass filter (recommended True)

    Returns:
        Tuple of (r, theta, r_dot, theta_dot) arrays
    """
    # Deviations
    d = D - D_eq
    s = S - S_eq
    k = K - K_eq

    # Extract cycle using recommended method
    r, theta = extract_dominant_cycle(d, s, k, method=method, apply_bandpass=apply_bandpass)

    # Smooth amplitude
    if smooth_window > 1:
        r_smooth = np.convolve(
            r,
            np.ones(smooth_window) / smooth_window,
            mode='same'
        )
    else:
        r_smooth = r

    # Compute derivatives
    r_dot = np.gradient(r_smooth)

    # Phase velocity with wraparound handling
    theta_diff = np.diff(theta)
    theta_diff = np.where(theta_diff > np.pi, theta_diff - 2 * np.pi, theta_diff)
    theta_diff = np.where(theta_diff < -np.pi, theta_diff + 2 * np.pi, theta_diff)
    theta_dot = np.concatenate([[0], theta_diff])

    return r, theta, r_dot, theta_dot

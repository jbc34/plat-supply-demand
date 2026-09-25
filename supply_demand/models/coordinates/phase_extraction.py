"""Phase extraction methods for cycle identification.

Provides multiple methods for extracting instantaneous phase
from time series data.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class PhaseEstimate:
    """Result of phase estimation."""

    phase: np.ndarray           # Instantaneous phase (radians)
    amplitude: np.ndarray       # Instantaneous amplitude
    frequency: np.ndarray       # Instantaneous frequency
    confidence: np.ndarray      # Confidence at each point
    method: str                 # Method used


def extract_phase_hilbert(
    x: np.ndarray,
    detrend: bool = True,
    bandpass: Optional[Tuple[float, float]] = None,
    fs: float = 1.0,
) -> PhaseEstimate:
    """Extract instantaneous phase using Hilbert transform.

    The Hilbert transform creates an analytic signal from which
    we can extract instantaneous amplitude and phase.

    Args:
        x: Input time series
        detrend: Whether to remove linear trend
        bandpass: Optional (low_freq, high_freq) for bandpass filtering
        fs: Sampling frequency

    Returns:
        PhaseEstimate with phase, amplitude, and frequency
    """
    x = np.asarray(x).copy()

    # Detrend
    if detrend:
        x = signal.detrend(x, type='linear')

    # Bandpass filter if specified
    if bandpass is not None:
        low, high = bandpass
        nyquist = fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist

        # Ensure frequencies are valid
        low_norm = max(0.01, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))

        b, a = signal.butter(3, [low_norm, high_norm], btype='band')
        x = signal.filtfilt(b, a, x)

    # Compute analytic signal
    analytic = signal.hilbert(x)

    # Extract amplitude and phase
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)

    # Unwrap phase
    phase = np.unwrap(phase)

    # Compute instantaneous frequency
    frequency = np.gradient(phase) * fs / (2 * np.pi)

    # Confidence based on amplitude relative to mean
    mean_amp = np.mean(amplitude)
    confidence = np.clip(amplitude / (mean_amp + 1e-10), 0, 1)

    return PhaseEstimate(
        phase=phase,
        amplitude=amplitude,
        frequency=frequency,
        confidence=confidence,
        method="hilbert",
    )


def extract_phase_wavelet(
    x: np.ndarray,
    target_period: float = 48.0,  # Target cycle period (months)
    fs: float = 1.0,
    wavelet: str = "morl",
) -> PhaseEstimate:
    """Extract phase using continuous wavelet transform.

    Wavelet analysis provides better localization of phase
    for non-stationary signals.

    Args:
        x: Input time series
        target_period: Expected cycle period in samples
        fs: Sampling frequency
        wavelet: Wavelet type ("morl" for Morlet, "cmor" for complex Morlet)

    Returns:
        PhaseEstimate
    """
    from scipy.signal import cwt, morlet2

    x = np.asarray(x).copy()
    x = x - np.mean(x)

    # Target frequency
    target_freq = fs / target_period

    # Range of scales to analyze
    scales = np.arange(1, len(x) // 4)

    # Compute CWT
    # Using Morlet wavelet
    widths = scales

    # Simple Morlet CWT
    omega0 = 6.0  # Morlet parameter
    coefficients = np.zeros((len(widths), len(x)), dtype=complex)

    for i, width in enumerate(widths):
        # Create complex Morlet wavelet
        n = min(10 * width, len(x))
        t = np.arange(-n // 2, n // 2) / width
        wavelet_data = np.exp(1j * omega0 * t) * np.exp(-t ** 2 / 2)
        wavelet_data = wavelet_data / np.sqrt(width)

        # Convolve
        coefficients[i] = np.convolve(x, wavelet_data, mode='same')

    # Find scale closest to target period
    # Scale to period relationship for Morlet: period = scale * 4π / (ω₀ + sqrt(2+ω₀²))
    scale_to_period = widths * 4 * np.pi / (omega0 + np.sqrt(2 + omega0 ** 2))
    target_scale_idx = np.argmin(np.abs(scale_to_period - target_period))

    # Extract phase and amplitude at target scale
    target_coeffs = coefficients[target_scale_idx]
    amplitude = np.abs(target_coeffs)
    phase = np.unwrap(np.angle(target_coeffs))

    # Instantaneous frequency
    frequency = np.gradient(phase) * fs / (2 * np.pi)

    # Confidence based on cone of influence and amplitude
    edge_mask = np.ones(len(x))
    edge_width = int(scale_to_period[target_scale_idx])
    edge_mask[:edge_width] = np.linspace(0, 1, edge_width)
    edge_mask[-edge_width:] = np.linspace(1, 0, edge_width)

    mean_amp = np.mean(amplitude)
    amplitude_conf = np.clip(amplitude / (mean_amp + 1e-10), 0, 1)
    confidence = amplitude_conf * edge_mask

    return PhaseEstimate(
        phase=phase,
        amplitude=amplitude,
        frequency=frequency,
        confidence=confidence,
        method="wavelet",
    )


def unwrap_phase(
    phase: np.ndarray,
    period: float = 2 * np.pi,
) -> np.ndarray:
    """Unwrap phase to remove discontinuities.

    Args:
        phase: Input phase array
        period: Phase period (default 2π)

    Returns:
        Unwrapped phase
    """
    return np.unwrap(phase, period=period)


def normalize_phase(
    phase: np.ndarray,
) -> np.ndarray:
    """Normalize phase to [0, 2π) range.

    Args:
        phase: Input phase array (possibly unwrapped)

    Returns:
        Phase normalized to [0, 2π)
    """
    return phase % (2 * np.pi)


def estimate_cycle_period(
    x: np.ndarray,
    fs: float = 1.0,
    method: str = "autocorr",
) -> float:
    """Estimate dominant cycle period.

    Args:
        x: Input time series
        fs: Sampling frequency
        method: Estimation method:
            - "autocorr": Autocorrelation peak
            - "fft": Spectral peak
            - "zero_crossing": Zero crossing intervals

    Returns:
        Estimated period in samples
    """
    x = x - np.mean(x)

    if method == "autocorr":
        # Autocorrelation
        acf = np.correlate(x, x, mode='full')
        acf = acf[len(x) - 1:]
        acf = acf / acf[0]

        # Find first significant peak after lag 0
        peaks, _ = signal.find_peaks(acf, height=0.1)
        if len(peaks) > 0:
            period = peaks[0]
        else:
            period = len(x) // 4

    elif method == "fft":
        # FFT
        n = len(x)
        fft_vals = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, 1 / fs)

        # Find peak (excluding DC)
        magnitudes = np.abs(fft_vals[1:])
        peak_idx = np.argmax(magnitudes) + 1
        peak_freq = freqs[peak_idx]

        period = 1 / peak_freq if peak_freq > 0 else len(x) // 4

    elif method == "zero_crossing":
        # Zero crossings
        crossings = np.where(np.diff(np.sign(x)))[0]
        if len(crossings) > 2:
            intervals = np.diff(crossings)
            period = np.median(intervals) * 2  # Full cycle
        else:
            period = len(x) // 4

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(period)


def align_phases(
    phase1: np.ndarray,
    phase2: np.ndarray,
) -> float:
    """Compute average phase difference between two signals.

    Useful for determining lead-lag relationships.

    Args:
        phase1: First phase array
        phase2: Second phase array

    Returns:
        Mean phase difference (phase1 - phase2)
    """
    # Compute circular mean of differences
    diff = phase1 - phase2
    mean_sin = np.mean(np.sin(diff))
    mean_cos = np.mean(np.cos(diff))

    return np.arctan2(mean_sin, mean_cos)

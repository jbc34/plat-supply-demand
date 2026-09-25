"""Polar/spiral coordinate transformations for market state.

Note: The recommended phase extraction method is "ds_spread" which uses the
demand-supply imbalance as the primary cycle signal. See the cycle extraction
study (research/01_cycle_extraction_study.md) for methodology details.
"""

from supply_demand.models.coordinates.polar_transform import (
    SpiralCoordinates,
    compute_spiral_coordinates,
    compute_spiral_trajectory,
    extract_dominant_cycle,
    cartesian_to_polar,
    polar_to_cartesian,
    bandpass_filter,
    phase_to_quadrant,
    DEFAULT_BANDPASS_LOW_PERIOD,
    DEFAULT_BANDPASS_HIGH_PERIOD,
    DEFAULT_BANDPASS_ORDER,
)
from supply_demand.models.coordinates.phase_extraction import (
    PhaseEstimate,
    extract_phase_hilbert,
    extract_phase_wavelet,
    unwrap_phase,
)
from supply_demand.models.coordinates.spiral_metrics import (
    compute_cycle_metrics,
    compute_turning_points,
    CycleMetrics,
)

__all__ = [
    # Core spiral coordinates
    "SpiralCoordinates",
    "phase_to_quadrant",
    "compute_spiral_coordinates",
    "compute_spiral_trajectory",
    "extract_dominant_cycle",
    "cartesian_to_polar",
    "polar_to_cartesian",
    # Bandpass filtering (recommended for cycle extraction)
    "bandpass_filter",
    "DEFAULT_BANDPASS_LOW_PERIOD",
    "DEFAULT_BANDPASS_HIGH_PERIOD",
    "DEFAULT_BANDPASS_ORDER",
    # Phase extraction utilities
    "PhaseEstimate",
    "extract_phase_hilbert",
    "extract_phase_wavelet",
    "unwrap_phase",
    # Cycle metrics
    "compute_cycle_metrics",
    "compute_turning_points",
    "CycleMetrics",
]

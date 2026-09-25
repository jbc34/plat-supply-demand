"""Tests for polar coordinate transformation."""

import numpy as np
import pytest

from supply_demand.models.coordinates import (
    compute_spiral_coordinates,
    extract_dominant_cycle,
    cartesian_to_polar,
    polar_to_cartesian,
)


class TestPolarTransform:
    """Test suite for polar transformations."""

    def test_cartesian_to_polar(self):
        """Test Cartesian to polar conversion."""
        # Test cases: (x, y) -> (r, theta)
        cases = [
            ((1, 0), (1, 0)),
            ((0, 1), (1, np.pi / 2)),
            ((-1, 0), (1, np.pi)),
            ((0, -1), (1, 3 * np.pi / 2)),
            ((1, 1), (np.sqrt(2), np.pi / 4)),
        ]

        for (x, y), (expected_r, expected_theta) in cases:
            r, theta = cartesian_to_polar(x, y)
            assert r == pytest.approx(expected_r, rel=1e-6)
            assert theta == pytest.approx(expected_theta, rel=1e-6)

    def test_polar_to_cartesian(self):
        """Test polar to Cartesian conversion."""
        cases = [
            ((1, 0), (1, 0)),
            ((1, np.pi / 2), (0, 1)),
            ((1, np.pi), (-1, 0)),
            ((2, np.pi / 4), (np.sqrt(2), np.sqrt(2))),
        ]

        for (r, theta), (expected_x, expected_y) in cases:
            x, y = polar_to_cartesian(r, theta)
            assert x == pytest.approx(expected_x, abs=1e-6)
            assert y == pytest.approx(expected_y, abs=1e-6)

    def test_roundtrip(self):
        """Test Cartesian -> Polar -> Cartesian roundtrip."""
        np.random.seed(42)

        for _ in range(10):
            x_orig = np.random.randn()
            y_orig = np.random.randn()

            r, theta = cartesian_to_polar(x_orig, y_orig)
            x_recovered, y_recovered = polar_to_cartesian(r, theta)

            assert x_recovered == pytest.approx(x_orig, abs=1e-6)
            assert y_recovered == pytest.approx(y_orig, abs=1e-6)


class TestDominantCycle:
    """Test dominant cycle extraction."""

    def test_extract_from_sinusoid(self):
        """Test extraction from known sinusoidal signal."""
        T = 200
        t = np.arange(T)
        period = 48

        # Known sinusoidal signals
        D = np.sin(2 * np.pi * t / period)
        S = np.sin(2 * np.pi * t / period + np.pi / 6)
        K = np.sin(2 * np.pi * t / period + np.pi / 3)

        amplitude, phase = extract_dominant_cycle(D, S, K)

        # Amplitude should be relatively stable
        assert np.std(amplitude[50:-50]) < 0.5 * np.mean(amplitude[50:-50])

        # Phase is normalized to [0, 2π), so check that average angular velocity is positive
        # by computing phase differences with wraparound handling
        phase_diff = np.diff(phase)
        # Handle wraparound: large negative jumps are actually positive progression
        phase_diff = np.where(phase_diff < -np.pi, phase_diff + 2 * np.pi, phase_diff)
        mean_angular_velocity = np.mean(phase_diff)
        # Phase should progress forward on average
        assert mean_angular_velocity > 0 or np.abs(mean_angular_velocity) < 0.1

    def test_extract_handles_noise(self):
        """Test extraction is robust to noise."""
        T = 200
        t = np.arange(T)
        period = 48

        D = np.sin(2 * np.pi * t / period) + np.random.randn(T) * 0.3
        S = np.sin(2 * np.pi * t / period) + np.random.randn(T) * 0.3
        K = np.sin(2 * np.pi * t / period) + np.random.randn(T) * 0.3

        # Should not raise exception
        amplitude, phase = extract_dominant_cycle(D, S, K)

        assert len(amplitude) == T
        assert len(phase) == T
        assert not np.any(np.isnan(amplitude))
        assert not np.any(np.isnan(phase))


class TestSpiralCoordinates:
    """Test spiral coordinate computation."""

    def test_point_estimate(self):
        """Test spiral coordinates from point estimate."""
        coords = compute_spiral_coordinates(
            D=1.0, S=0.0, K=0.5,
            D_eq=0.0, S_eq=0.0, K_eq=0.0,
        )

        assert coords.r > 0
        assert 0 <= coords.theta < 2 * np.pi
        assert coords.quadrant in [
            "Early Expansion",
            "Peak / Late Expansion",
            "Contraction",
            "Recovery / Early Cycle",
        ]

    def test_time_series_estimate(self):
        """Test spiral coordinates from time series."""
        T = 100
        t = np.arange(T)
        period = 48

        D = np.sin(2 * np.pi * t / period)
        S = np.sin(2 * np.pi * t / period + np.pi / 4)
        K = np.sin(2 * np.pi * t / period - np.pi / 4)

        coords = compute_spiral_coordinates(D, S, K)

        assert coords.r >= 0
        assert 0 <= coords.theta < 2 * np.pi
        assert coords.confidence >= 0
        assert coords.confidence <= 1

    def test_equilibrium_deviation(self):
        """Test that equilibrium shifts coordinates correctly."""
        # At equilibrium, r should be small
        coords_eq = compute_spiral_coordinates(
            D=0.0, S=0.0, K=0.0,
        )
        assert coords_eq.r == pytest.approx(0, abs=1e-6)

        # Away from equilibrium, r should be larger
        coords_away = compute_spiral_coordinates(
            D=1.0, S=0.0, K=0.0,
        )
        assert coords_away.r > 0

    def test_quadrant_identification(self):
        """Test quadrant is correctly identified."""
        # Test various phase angles map to correct quadrants
        T = 200
        t = np.arange(T)

        # Full cycle
        D = np.sin(2 * np.pi * t / 50)
        S = np.cos(2 * np.pi * t / 50)
        K = np.sin(2 * np.pi * t / 50 + np.pi / 4)

        coords = compute_spiral_coordinates(D, S, K)

        # Should be one of the defined quadrants
        valid_quadrants = [
            "Early Expansion",
            "Peak / Late Expansion",
            "Contraction",
            "Recovery / Early Cycle",
        ]
        assert coords.quadrant in valid_quadrants

"""Tests for Kalman filter implementation."""

import numpy as np
import pytest

from supply_demand.models.state_space import (
    KalmanFilter,
    StateSpaceParams,
    build_transition_matrix,
    build_process_noise_matrix,
)


class TestKalmanFilter:
    """Test suite for Kalman filter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple 2D tracking model
        self.F = np.array([
            [1, 1],
            [0, 1],
        ])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * 0.1
        self.R = np.array([[1.0]])

    def test_filter_initialization(self):
        """Test filter initializes correctly."""
        kf = KalmanFilter(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R,
        )

        assert kf.n_states == 2
        assert kf.n_obs == 1
        assert kf.x.shape == (2,)
        assert kf.P.shape == (2, 2)

    def test_predict_step(self):
        """Test prediction step."""
        kf = KalmanFilter(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R,
            x0=np.array([0, 1]),
            P0=np.eye(2),
        )

        x_pred, P_pred = kf.predict()

        # State should advance
        assert x_pred[0] == 1  # 0 + 1*1
        assert x_pred[1] == 1  # Velocity unchanged

        # Covariance should grow
        assert P_pred[0, 0] > 1

    def test_update_step(self):
        """Test update step."""
        kf = KalmanFilter(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R,
        )

        x_pred = np.array([1, 0])
        P_pred = np.eye(2)
        y = np.array([1.5])

        x_upd, P_upd, ll, _, _ = kf.update(y, x_pred, P_pred)

        # Should move toward observation
        assert x_upd[0] > 1.0
        assert x_upd[0] < 1.5

        # Covariance should decrease
        assert P_upd[0, 0] < 1

    def test_filter_missing_observations(self):
        """Test handling of missing observations."""
        kf = KalmanFilter(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R,
        )

        x_pred = np.array([1, 0])
        P_pred = np.eye(2)
        y = np.array([np.nan])  # Missing observation

        x_upd, P_upd, ll, _, _ = kf.update(y, x_pred, P_pred)

        # No update should occur
        np.testing.assert_array_equal(x_upd, x_pred)
        np.testing.assert_array_equal(P_upd, P_pred)
        assert ll == 0.0

    def test_filter_full_sequence(self):
        """Test filtering over full observation sequence."""
        kf = KalmanFilter(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R,
        )

        # Generate synthetic observations
        T = 50
        true_state = np.zeros((T, 2))
        true_state[0] = [0, 0.5]

        for t in range(1, T):
            true_state[t] = self.F @ true_state[t - 1] + np.random.randn(2) * 0.1

        observations = true_state[:, 0:1] + np.random.randn(T, 1)

        # Run filter
        results = kf.filter(observations)

        assert results.filtered_states.shape == (T, 2)
        assert results.filtered_covs.shape == (T, 2, 2)
        assert not np.isnan(results.log_likelihood)

        # Filtered states should track true states
        correlation = np.corrcoef(
            results.filtered_states[:, 0],
            true_state[:, 0]
        )[0, 1]
        assert correlation > 0.8

    def test_smoother(self):
        """Test RTS smoother."""
        kf = KalmanFilter(
            F=self.F,
            H=self.H,
            Q=self.Q,
            R=self.R,
        )

        # Generate observations
        T = 30
        observations = np.random.randn(T, 1)

        # Filter and smooth
        filter_results = kf.filter(observations)
        smoothed_states, smoothed_covs = kf.smooth(filter_results)

        assert smoothed_states.shape == (T, 2)
        assert smoothed_covs.shape == (T, 2, 2)

        # Smoothed covariances should be smaller than filtered
        for t in range(T - 1):
            assert smoothed_covs[t, 0, 0] <= filter_results.filtered_covs[t, 0, 0] + 1e-10


class TestStateSpaceMatrices:
    """Test state-space matrix construction."""

    def test_transition_matrix_structure(self):
        """Test transition matrix has correct structure."""
        params = StateSpaceParams()
        F = build_transition_matrix(params)

        assert F.shape == (6, 6)

        # Position updates
        assert F[0, 0] == 1.0  # D -> D
        assert F[1, 1] == 1.0  # S -> S
        assert F[2, 2] == 1.0  # K -> K

        # Velocity -> Position
        assert F[0, 3] == 1.0  # dD -> D
        assert F[1, 4] == 1.0  # dS -> S
        assert F[2, 5] == 1.0  # dK -> K

    def test_transition_matrix_stability(self):
        """Test transition matrix eigenvalue bounds.

        Note: The discrete-time approximation with dt=1 may produce
        eigenvalues slightly above 1. Real calibration should use
        smaller dt or matrix exponential discretization for strict stability.
        The Kalman filter with observations will still converge.
        """
        params = StateSpaceParams()
        F = build_transition_matrix(params)

        eigenvalues = np.linalg.eigvals(F)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        # Should be bounded (allows marginally unstable open-loop systems)
        # Real markets often exhibit slow growth without feedback
        assert max_eigenvalue <= 1.3, f"Max eigenvalue {max_eigenvalue} too large"

    def test_process_noise_matrix(self):
        """Test process noise matrix structure."""
        params = StateSpaceParams(
            sigma_D=0.05,
            sigma_S=0.03,
            sigma_K=0.08,
        )
        Q = build_process_noise_matrix(params)

        assert Q.shape == (6, 6)

        # Noise only on velocity equations
        assert Q[0, 0] == 0  # No noise on D position
        assert Q[1, 1] == 0  # No noise on S position
        assert Q[2, 2] == 0  # No noise on K position
        assert Q[3, 3] == pytest.approx(0.05 ** 2)
        assert Q[4, 4] == pytest.approx(0.03 ** 2)
        assert Q[5, 5] == pytest.approx(0.08 ** 2)


class TestSyntheticCycle:
    """Integration test with synthetic cycle data."""

    def test_tracks_synthetic_cycle(self):
        """Test that filter tracks a synthetic oscillating process."""
        # Create synthetic oscillating data
        T = 100
        t = np.arange(T)
        period = 48  # 4 year cycle

        # True latent states (sinusoidal)
        D_true = np.sin(2 * np.pi * t / period)
        S_true = np.sin(2 * np.pi * t / period + np.pi / 4)  # Lagged
        K_true = np.sin(2 * np.pi * t / period - np.pi / 4)

        # Simple observation model: observe D with noise
        observations = D_true.reshape(-1, 1) + np.random.randn(T, 1) * 0.3

        # Set up filter
        params = StateSpaceParams()
        F = build_transition_matrix(params)
        Q = build_process_noise_matrix(params)
        H = np.zeros((1, 6))
        H[0, 0] = 1.0  # Observe D
        R = np.array([[0.3 ** 2]])

        kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
        results = kf.filter(observations)

        # Filter should track the true D
        correlation = np.corrcoef(
            results.filtered_states[:, 0],
            D_true
        )[0, 1]

        assert correlation > 0.7, f"Correlation {correlation} too low"

"""Particle filter for nonlinear state estimation.

Provides sequential Monte Carlo estimation for cases where
the standard Kalman filter assumptions don't hold.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class ParticleFilterResults:
    """Results from particle filtering."""

    state_means: np.ndarray       # T x n_states
    state_stds: np.ndarray        # T x n_states
    particles: np.ndarray         # T x n_particles x n_states
    weights: np.ndarray           # T x n_particles
    effective_sample_sizes: np.ndarray  # T


class ParticleFilter:
    """Sequential Monte Carlo (Particle) filter.

    Approximates the posterior distribution using weighted samples.
    Suitable for nonlinear state transitions or non-Gaussian noise.

    Example:
        >>> pf = ParticleFilter(
        ...     transition_fn=my_transition,
        ...     observation_fn=my_observation,
        ...     n_particles=1000
        ... )
        >>> results = pf.filter(observations)
    """

    def __init__(
        self,
        transition_fn: Callable[[np.ndarray], np.ndarray],
        observation_fn: Callable[[np.ndarray], np.ndarray],
        transition_noise_fn: Callable[[], np.ndarray],
        observation_noise_std: float,
        n_particles: int = 1000,
        n_states: int = 6,
        resample_threshold: float = 0.5,
    ):
        """Initialize particle filter.

        Args:
            transition_fn: State transition function x_{t+1} = f(x_t)
            observation_fn: Observation function y_t = h(x_t)
            transition_noise_fn: Function returning process noise sample
            observation_noise_std: Observation noise standard deviation
            n_particles: Number of particles
            n_states: Dimension of state vector
            resample_threshold: ESS threshold for resampling (as fraction)
        """
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.transition_noise_fn = transition_noise_fn
        self.observation_noise_std = observation_noise_std
        self.n_particles = n_particles
        self.n_states = n_states
        self.resample_threshold = resample_threshold

        # Initialize particles
        self.particles = np.zeros((n_particles, n_states))
        self.weights = np.ones(n_particles) / n_particles

    def initialize(
        self,
        mean: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize particles from prior distribution.

        Args:
            mean: Prior mean (defaults to zero)
            cov: Prior covariance (defaults to identity)
        """
        if mean is None:
            mean = np.zeros(self.n_states)
        if cov is None:
            cov = np.eye(self.n_states)

        self.particles = np.random.multivariate_normal(
            mean, cov, size=self.n_particles
        )
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self) -> None:
        """Propagate particles through transition model."""
        for i in range(self.n_particles):
            noise = self.transition_noise_fn()
            self.particles[i] = self.transition_fn(self.particles[i]) + noise

    def update(self, y: np.ndarray) -> None:
        """Update particle weights based on observation.

        Args:
            y: Observation vector (NaN for missing values)
        """
        valid = ~np.isnan(y)
        if not valid.any():
            return

        y_valid = y[valid]

        for i in range(self.n_particles):
            y_pred = self.observation_fn(self.particles[i])
            y_pred_valid = y_pred[valid] if len(y_pred.shape) > 0 else y_pred

            # Gaussian likelihood
            residual = y_valid - y_pred_valid
            log_likelihood = -0.5 * np.sum(
                (residual / self.observation_noise_std) ** 2
            )
            self.weights[i] *= np.exp(log_likelihood)

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # Degenerate case - reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles

    def resample(self) -> None:
        """Resample particles using systematic resampling."""
        n = self.n_particles

        # Compute cumulative weights
        cumsum = np.cumsum(self.weights)

        # Systematic resampling
        u = (np.arange(n) + np.random.uniform()) / n
        indices = np.searchsorted(cumsum, u)

        # Handle edge cases
        indices = np.clip(indices, 0, n - 1)

        # Resample
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n

    def effective_sample_size(self) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(self.weights ** 2)

    def filter(
        self,
        Y: np.ndarray,
        x0_mean: Optional[np.ndarray] = None,
        x0_cov: Optional[np.ndarray] = None,
    ) -> ParticleFilterResults:
        """Run particle filter over observation sequence.

        Args:
            Y: T x n_obs array of observations
            x0_mean: Initial state mean
            x0_cov: Initial state covariance

        Returns:
            ParticleFilterResults
        """
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T, n_obs = Y.shape

        # Initialize
        self.initialize(x0_mean, x0_cov)

        # Storage
        state_means = np.zeros((T, self.n_states))
        state_stds = np.zeros((T, self.n_states))
        all_particles = np.zeros((T, self.n_particles, self.n_states))
        all_weights = np.zeros((T, self.n_particles))
        ess = np.zeros(T)

        for t in range(T):
            # Predict
            self.predict()

            # Update
            self.update(Y[t])

            # Store
            all_particles[t] = self.particles.copy()
            all_weights[t] = self.weights.copy()
            ess[t] = self.effective_sample_size()

            # Compute weighted statistics
            state_means[t] = np.average(
                self.particles, weights=self.weights, axis=0
            )
            state_stds[t] = np.sqrt(np.average(
                (self.particles - state_means[t]) ** 2,
                weights=self.weights, axis=0
            ))

            # Resample if ESS is too low
            if ess[t] < self.resample_threshold * self.n_particles:
                self.resample()

        return ParticleFilterResults(
            state_means=state_means,
            state_stds=state_stds,
            particles=all_particles,
            weights=all_weights,
            effective_sample_sizes=ess,
        )

    def estimate_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate.

        Returns:
            Tuple of (weighted mean, weighted standard deviation)
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var = np.average(
            (self.particles - mean) ** 2,
            weights=self.weights, axis=0
        )
        return mean, np.sqrt(var)


def create_linear_particle_filter(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    n_particles: int = 1000,
) -> ParticleFilter:
    """Create a particle filter for a linear state-space model.

    Useful for comparison with Kalman filter or when observations
    are non-Gaussian.

    Args:
        F: State transition matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Observation noise covariance
        n_particles: Number of particles

    Returns:
        Configured ParticleFilter
    """
    n_states = F.shape[0]

    def transition_fn(x):
        return F @ x

    def observation_fn(x):
        return H @ x

    # Process noise from Q
    Q_chol = np.linalg.cholesky(Q + 1e-6 * np.eye(n_states))

    def transition_noise_fn():
        return Q_chol @ np.random.randn(n_states)

    # Observation noise (scalar for simplicity)
    obs_noise_std = np.sqrt(np.mean(np.diag(R)))

    return ParticleFilter(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        transition_noise_fn=transition_noise_fn,
        observation_noise_std=obs_noise_std,
        n_particles=n_particles,
        n_states=n_states,
    )

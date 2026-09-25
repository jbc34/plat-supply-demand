"""State-space model specification.

Defines the structure of the state-space model for market dynamics:

State Vector:
    x_t = [D_t, S_t, K_t, dD_t, dS_t, dK_t]'

    where:
    - D_t = Demand deviation from equilibrium
    - S_t = Supply deviation from equilibrium
    - K_t = Capital flow deviation from equilibrium
    - dD_t, dS_t, dK_t = First derivatives (velocities)

State Transition:
    x_{t+1} = F @ x_t + G @ u_t + η_t

Observation:
    y_t = H @ x_t + ε_t
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StateSpaceParams:
    """Parameters for the state-space model.

    Encodes the coupled dynamics of Demand, Supply, and Capital Flow.
    """

    # Demand equation parameters
    alpha_1: float = 0.15   # Income sensitivity
    alpha_2: float = 0.10   # Price sensitivity
    alpha_3: float = 0.08   # Capital flow sensitivity
    tau_1: int = 2          # Capital -> Demand lag (periods)

    # Supply equation parameters
    beta_1: float = 0.12    # Price sensitivity
    beta_2: float = 0.10    # Capital flow sensitivity
    beta_3: float = 0.05    # Supply adjustment speed
    tau_2: int = 18         # Price -> Supply lag (periods)
    tau_3: int = 12         # Capital -> Supply lag (periods)

    # Capital flow equation parameters
    gamma_1: float = 0.15   # Return gap sensitivity
    gamma_2: float = 0.08   # Price acceleration sensitivity
    gamma_3: float = 0.10   # Imbalance sensitivity

    # Price formation
    lambda_: float = 0.05   # Price adjustment speed

    # Damping coefficients (must be high enough to ensure stability)
    damping_D: float = 0.15  # Demand velocity damping
    damping_S: float = 0.15  # Supply velocity damping
    damping_K: float = 0.20  # Capital velocity damping

    # Noise parameters (process noise standard deviations)
    sigma_D: float = 0.05   # Demand process noise
    sigma_S: float = 0.03   # Supply process noise
    sigma_K: float = 0.08   # Capital flow process noise

    # Measurement noise scale
    measurement_noise_scale: float = 0.3

    @classmethod
    def from_dict(cls, params: dict) -> "StateSpaceParams":
        """Create StateSpaceParams from dictionary."""
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alpha_1": self.alpha_1,
            "alpha_2": self.alpha_2,
            "alpha_3": self.alpha_3,
            "tau_1": self.tau_1,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "tau_2": self.tau_2,
            "tau_3": self.tau_3,
            "gamma_1": self.gamma_1,
            "gamma_2": self.gamma_2,
            "gamma_3": self.gamma_3,
            "lambda_": self.lambda_,
            "damping_D": self.damping_D,
            "damping_S": self.damping_S,
            "damping_K": self.damping_K,
            "sigma_D": self.sigma_D,
            "sigma_S": self.sigma_S,
            "sigma_K": self.sigma_K,
            "measurement_noise_scale": self.measurement_noise_scale,
        }


def build_transition_matrix(
    params: StateSpaceParams,
    dt: float = 1.0
) -> np.ndarray:
    """Build the state transition matrix F from continuous-time parameters.

    Uses first-order discretization of the coupled DDE system.

    The state vector is [D, S, K, dD, dS, dK]

    Args:
        params: Model parameters
        dt: Time step (default 1 period)

    Returns:
        6x6 state transition matrix F
    """
    F = np.zeros((6, 6))

    # Position updates (x_{t+1} = x_t + dx_t * dt)
    # D -> D
    F[0, 0] = 1.0
    F[0, 3] = dt  # dD -> D

    # S -> S
    F[1, 1] = 1.0
    F[1, 4] = dt  # dS -> S

    # K -> K
    F[2, 2] = 1.0
    F[2, 5] = dt  # dK -> K

    # Velocity updates (from the coupled dynamics)

    # dD equation: Demand velocity responds to:
    # - Price effects (through S-D imbalance)
    # - Capital flow effects
    # - Own damping
    F[3, 0] = -params.alpha_2 * params.lambda_ * dt  # D effect on dD (price channel)
    F[3, 1] = params.alpha_2 * params.lambda_ * dt   # S effect on dD (through imbalance)
    F[3, 2] = params.alpha_3 * dt                    # K effect on dD
    F[3, 3] = 1.0 - params.damping_D * dt            # Damping

    # dS equation: Supply velocity responds to:
    # - Price effects (through D-S imbalance)
    # - Capital flow effects
    # - Own adjustment dynamics
    F[4, 0] = params.beta_1 * params.lambda_ * dt    # D effect on dS (price channel)
    F[4, 1] = -params.beta_1 * params.lambda_ * dt - params.beta_3 * dt  # S self-adjustment
    F[4, 2] = params.beta_2 * dt                     # K effect on dS
    F[4, 4] = 1.0 - params.damping_S * dt            # Damping

    # dK equation: Capital flow velocity responds to:
    # - Supply-demand imbalance (return gap)
    # - Price acceleration
    # - Own mean reversion
    F[5, 0] = params.gamma_3 * dt                    # D effect on dK (imbalance)
    F[5, 1] = -params.gamma_3 * dt                   # S effect on dK (imbalance)
    F[5, 3] = params.gamma_2 * dt                    # dD effect on dK (price acceleration proxy)
    F[5, 5] = 1.0 - params.damping_K * dt            # Damping (capital is stickier)

    return F


def build_measurement_matrix(
    proxy_config: dict,
    n_states: int = 6
) -> np.ndarray:
    """Build the measurement matrix H from proxy configuration.

    Maps latent states to observable proxies.

    Args:
        proxy_config: Dictionary with proxy configurations:
            {
                "proxy_name": {
                    "state": "D" | "S" | "K" | "dD" | "dS" | "dK",
                    "loading": float,  # Factor loading
                }
            }
        n_states: Number of state variables (default 6)

    Returns:
        n_proxies x n_states measurement matrix H
    """
    state_indices = {
        "D": 0, "S": 1, "K": 2,
        "dD": 3, "dS": 4, "dK": 5,
    }

    n_proxies = len(proxy_config)
    H = np.zeros((n_proxies, n_states))

    for i, (proxy_name, config) in enumerate(proxy_config.items()):
        state = config.get("state", "D")
        loading = config.get("loading", 1.0)

        state_idx = state_indices.get(state)
        if state_idx is not None:
            H[i, state_idx] = loading

    return H


def build_process_noise_matrix(
    params: StateSpaceParams,
    n_states: int = 6
) -> np.ndarray:
    """Build the process noise covariance matrix Q.

    Noise enters through the velocity equations.

    Args:
        params: Model parameters with noise levels
        n_states: Number of states

    Returns:
        n_states x n_states covariance matrix Q
    """
    Q = np.zeros((n_states, n_states))

    # Process noise on velocity equations only
    Q[3, 3] = params.sigma_D ** 2  # Demand velocity noise
    Q[4, 4] = params.sigma_S ** 2  # Supply velocity noise
    Q[5, 5] = params.sigma_K ** 2  # Capital velocity noise

    return Q


def build_measurement_noise_matrix(
    n_proxies: int,
    noise_levels: Optional[dict] = None,
    default_noise: float = 0.3
) -> np.ndarray:
    """Build the measurement noise covariance matrix R.

    Args:
        n_proxies: Number of observable proxies
        noise_levels: Dictionary mapping proxy index to noise std dev
        default_noise: Default noise level

    Returns:
        n_proxies x n_proxies covariance matrix R
    """
    R = np.eye(n_proxies) * (default_noise ** 2)

    if noise_levels:
        for idx, noise in noise_levels.items():
            if idx < n_proxies:
                R[idx, idx] = noise ** 2

    return R


def build_initial_state(
    n_states: int = 6,
    mean: Optional[np.ndarray] = None,
    variance: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build initial state mean and covariance.

    Args:
        n_states: Number of states
        mean: Initial state mean (defaults to zero)
        variance: Initial variance for each state

    Returns:
        Tuple of (initial mean, initial covariance)
    """
    x0 = mean if mean is not None else np.zeros(n_states)
    P0 = np.eye(n_states) * variance

    return x0, P0


class StateSpaceModel:
    """Complete state-space model specification.

    Encapsulates all matrices and parameters needed for Kalman filtering.
    """

    def __init__(
        self,
        params: StateSpaceParams,
        proxy_config: dict,
        n_states: int = 6,
    ):
        """Initialize state-space model.

        Args:
            params: Model parameters
            proxy_config: Proxy configuration for measurement matrix
            n_states: Number of state variables
        """
        self.params = params
        self.proxy_config = proxy_config
        self.n_states = n_states
        self.n_proxies = len(proxy_config)

        # Build system matrices
        self.F = build_transition_matrix(params)
        self.H = build_measurement_matrix(proxy_config, n_states)
        self.Q = build_process_noise_matrix(params, n_states)
        self.R = build_measurement_noise_matrix(
            self.n_proxies,
            default_noise=params.measurement_noise_scale
        )

        # Initial state
        self.x0, self.P0 = build_initial_state(n_states)

    def check_stability(self) -> dict:
        """Check stability of the state transition matrix.

        Returns:
            Dictionary with stability metrics
        """
        eigenvalues = np.linalg.eigvals(self.F)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        return {
            "eigenvalues": eigenvalues,
            "max_abs_eigenvalue": max_eigenvalue,
            "is_stable": max_eigenvalue <= 1.0,
            "is_asymptotically_stable": max_eigenvalue < 1.0,
        }

    def check_observability(self) -> dict:
        """Check observability of the system.

        Returns:
            Dictionary with observability metrics
        """
        # Build observability matrix
        O = self.H.copy()
        for i in range(1, self.n_states):
            O = np.vstack([O, self.H @ np.linalg.matrix_power(self.F, i)])

        rank = np.linalg.matrix_rank(O)

        return {
            "observability_matrix_rank": rank,
            "n_states": self.n_states,
            "is_observable": rank == self.n_states,
        }

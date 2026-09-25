"""Discretization of continuous-time systems.

Converts continuous-time dynamics to discrete-time state-space form
suitable for Kalman filtering.
"""

from typing import Tuple

import numpy as np
from scipy.linalg import expm


def euler_discretization(
    A: np.ndarray,
    B: np.ndarray,
    Q_c: np.ndarray,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-order Euler discretization.

    Converts continuous-time system:
        dx/dt = A @ x + B @ u + w, w ~ N(0, Q_c)

    To discrete-time:
        x_{k+1} = F @ x_k + G @ u_k + v, v ~ N(0, Q_d)

    Args:
        A: Continuous-time state matrix
        B: Continuous-time input matrix
        Q_c: Continuous-time process noise intensity
        dt: Time step

    Returns:
        Tuple of (F, G, Q_d) discrete-time matrices
    """
    n = A.shape[0]

    # State transition: F = I + A * dt
    F = np.eye(n) + A * dt

    # Input matrix: G = B * dt
    G = B * dt if B is not None else None

    # Process noise: Q_d = Q_c * dt
    Q_d = Q_c * dt

    return F, G, Q_d


def matrix_exponential_discretization(
    A: np.ndarray,
    B: np.ndarray,
    Q_c: np.ndarray,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact discretization using matrix exponential.

    More accurate than Euler for larger time steps.

    Args:
        A: Continuous-time state matrix
        B: Continuous-time input matrix (can be None)
        Q_c: Continuous-time process noise intensity
        dt: Time step

    Returns:
        Tuple of (F, G, Q_d) discrete-time matrices
    """
    n = A.shape[0]

    # State transition: F = exp(A * dt)
    F = expm(A * dt)

    # Input matrix: G = integral_0^dt exp(A*s) ds @ B
    if B is not None:
        # Approximate integral
        G = np.linalg.solve(A, (F - np.eye(n))) @ B
    else:
        G = None

    # Process noise covariance (Van Loan's method)
    # Build augmented matrix
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Q_c
    M[n:, n:] = A.T

    # Matrix exponential
    exp_M = expm(M * dt)

    # Extract Q_d
    F_inv_Q_d = exp_M[:n, n:]
    Q_d = F.T @ F_inv_Q_d

    # Ensure symmetry
    Q_d = (Q_d + Q_d.T) / 2

    return F, G, Q_d


def discretize_continuous_system(
    params: dict,
    dt: float = 1.0,
    method: str = "euler",
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize the market dynamics system.

    Builds continuous-time A matrix from parameters and discretizes.

    Args:
        params: Dictionary of system parameters
        dt: Time step (default 1 month)
        method: Discretization method ("euler" or "exact")

    Returns:
        Tuple of (F, Q) discrete-time matrices
    """
    # Build continuous-time state matrix A
    # State: [D, S, K, dD, dS, dK]
    A = np.zeros((6, 6))

    # Position equations: dx/dt = velocity
    A[0, 3] = 1.0  # dD/dt = dD
    A[1, 4] = 1.0  # dS/dt = dS
    A[2, 5] = 1.0  # dK/dt = dK

    # Velocity equations (accelerations)
    alpha_2 = params.get("alpha_2", 0.2)
    alpha_3 = params.get("alpha_3", 0.15)
    beta_1 = params.get("beta_1", 0.25)
    beta_2 = params.get("beta_2", 0.2)
    beta_3 = params.get("beta_3", 0.1)
    gamma_1 = params.get("gamma_1", 0.3)
    gamma_2 = params.get("gamma_2", 0.15)
    gamma_3 = params.get("gamma_3", 0.2)
    lambda_ = params.get("lambda_", 0.1)
    damping = params.get("damping", 0.1)

    # d(dD)/dt equation
    A[3, 0] = -alpha_2 * lambda_  # Price effect via D
    A[3, 1] = alpha_2 * lambda_   # Price effect via S
    A[3, 2] = alpha_3             # Capital effect
    A[3, 3] = -damping            # Damping

    # d(dS)/dt equation
    A[4, 0] = beta_1 * lambda_    # Price effect via D
    A[4, 1] = -beta_1 * lambda_ - beta_3  # Price + adjustment
    A[4, 2] = beta_2             # Capital effect
    A[4, 4] = -damping           # Damping

    # d(dK)/dt equation
    A[5, 0] = gamma_1 + gamma_3  # Return gap + imbalance via D
    A[5, 1] = -gamma_1 - gamma_3 # Return gap + imbalance via S
    A[5, 3] = gamma_2            # Price acceleration
    A[5, 5] = -damping * 1.5     # Damping (capital stickier)

    # Process noise intensity (continuous time)
    sigma_D = params.get("sigma_D", 0.05)
    sigma_S = params.get("sigma_S", 0.03)
    sigma_K = params.get("sigma_K", 0.08)

    Q_c = np.zeros((6, 6))
    Q_c[3, 3] = sigma_D ** 2
    Q_c[4, 4] = sigma_S ** 2
    Q_c[5, 5] = sigma_K ** 2

    # Discretize
    if method == "euler":
        F, _, Q = euler_discretization(A, None, Q_c, dt)
    elif method == "exact":
        F, _, Q = matrix_exponential_discretization(A, None, Q_c, dt)
    else:
        raise ValueError(f"Unknown method: {method}")

    return F, Q


def compute_steady_state_covariance(
    F: np.ndarray,
    Q: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute steady-state covariance by iteration.

    Solves the discrete Lyapunov equation:
        P = F @ P @ F' + Q

    Args:
        F: State transition matrix
        Q: Process noise covariance
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Steady-state covariance matrix P
    """
    n = F.shape[0]
    P = np.eye(n)

    for _ in range(max_iter):
        P_new = F @ P @ F.T + Q
        P_new = (P_new + P_new.T) / 2  # Ensure symmetry

        if np.max(np.abs(P_new - P)) < tol:
            return P_new

        P = P_new

    return P

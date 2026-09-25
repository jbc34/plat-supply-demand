"""Stability analysis for dynamic systems.

Analyzes eigenvalues, stability margins, and system behavior.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class StabilityResult:
    """Result of stability analysis."""

    eigenvalues: np.ndarray
    max_eigenvalue_modulus: float
    is_stable: bool
    is_asymptotically_stable: bool
    damping_ratios: np.ndarray
    natural_frequencies: np.ndarray
    stability_margin: float


def compute_eigenvalues(F: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of state transition matrix.

    Args:
        F: State transition matrix

    Returns:
        Array of eigenvalues (may be complex)
    """
    return np.linalg.eigvals(F)


def analyze_stability(F: np.ndarray) -> StabilityResult:
    """Comprehensive stability analysis of discrete-time system.

    For a discrete-time system x_{t+1} = F @ x_t:
    - Stable if all eigenvalues have |λ| ≤ 1
    - Asymptotically stable if all |λ| < 1

    Args:
        F: State transition matrix

    Returns:
        StabilityResult with detailed stability metrics
    """
    eigenvalues = compute_eigenvalues(F)
    moduli = np.abs(eigenvalues)
    max_modulus = np.max(moduli)

    is_stable = max_modulus <= 1.0 + 1e-10
    is_asymptotically_stable = max_modulus < 1.0 - 1e-10

    # Compute damping ratios and natural frequencies
    # For complex eigenvalue λ = r * exp(i*θ)
    # Damping ratio: ζ = -ln(r) / sqrt(ln²(r) + θ²)
    # Natural frequency: ω_n = sqrt(ln²(r) + θ²)

    damping_ratios = []
    natural_frequencies = []

    for lam in eigenvalues:
        r = np.abs(lam)
        theta = np.angle(lam)

        if r > 0:
            ln_r = np.log(r)
            omega_n = np.sqrt(ln_r ** 2 + theta ** 2)
            if omega_n > 0:
                zeta = -ln_r / omega_n
            else:
                zeta = 1.0
        else:
            omega_n = 0.0
            zeta = 1.0

        damping_ratios.append(zeta)
        natural_frequencies.append(omega_n)

    # Stability margin: how far the largest eigenvalue is from unit circle
    stability_margin = 1.0 - max_modulus

    return StabilityResult(
        eigenvalues=eigenvalues,
        max_eigenvalue_modulus=max_modulus,
        is_stable=is_stable,
        is_asymptotically_stable=is_asymptotically_stable,
        damping_ratios=np.array(damping_ratios),
        natural_frequencies=np.array(natural_frequencies),
        stability_margin=stability_margin,
    )


def compute_controllability_matrix(
    F: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    """Compute controllability matrix.

    C = [G, FG, F²G, ..., F^(n-1)G]

    Args:
        F: State transition matrix (n x n)
        G: Input matrix (n x m)

    Returns:
        Controllability matrix (n x nm)
    """
    n = F.shape[0]
    m = G.shape[1] if G.ndim > 1 else 1

    if G.ndim == 1:
        G = G.reshape(-1, 1)

    C = np.zeros((n, n * m))

    F_power = np.eye(n)
    for i in range(n):
        C[:, i * m:(i + 1) * m] = F_power @ G
        F_power = F_power @ F

    return C


def compute_observability_matrix(
    F: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Compute observability matrix.

    O = [H; HF; HF²; ...; HF^(n-1)]

    Args:
        F: State transition matrix (n x n)
        H: Observation matrix (p x n)

    Returns:
        Observability matrix (pn x n)
    """
    n = F.shape[0]
    p = H.shape[0]

    O = np.zeros((p * n, n))

    F_power = np.eye(n)
    for i in range(n):
        O[i * p:(i + 1) * p, :] = H @ F_power
        F_power = F_power @ F

    return O


def is_controllable(F: np.ndarray, G: np.ndarray) -> bool:
    """Check if system is controllable.

    Args:
        F: State transition matrix
        G: Input matrix

    Returns:
        True if system is controllable
    """
    C = compute_controllability_matrix(F, G)
    rank = np.linalg.matrix_rank(C)
    return rank == F.shape[0]


def is_observable(F: np.ndarray, H: np.ndarray) -> bool:
    """Check if system is observable.

    Args:
        F: State transition matrix
        H: Observation matrix

    Returns:
        True if system is observable
    """
    O = compute_observability_matrix(F, H)
    rank = np.linalg.matrix_rank(O)
    return rank == F.shape[0]


def find_dominant_modes(
    F: np.ndarray,
    n_modes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find dominant (slowest decaying) modes of the system.

    Args:
        F: State transition matrix
        n_modes: Number of modes to return

    Returns:
        Tuple of (eigenvalues, eigenvectors, time constants)
        for the n_modes slowest modes
    """
    eigenvalues, eigenvectors = np.linalg.eig(F)

    # Sort by modulus (descending)
    idx = np.argsort(np.abs(eigenvalues))[::-1]

    top_eigenvalues = eigenvalues[idx[:n_modes]]
    top_eigenvectors = eigenvectors[:, idx[:n_modes]]

    # Time constants: τ = -1 / ln|λ|
    with np.errstate(divide='ignore'):
        time_constants = -1.0 / np.log(np.abs(top_eigenvalues))
        time_constants = np.where(
            np.isinf(time_constants),
            np.inf,
            time_constants
        )

    return top_eigenvalues, top_eigenvectors, time_constants


def compute_impulse_response(
    F: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    n_steps: int = 100,
) -> np.ndarray:
    """Compute impulse response of the system.

    y_t = H @ F^t @ G

    Args:
        F: State transition matrix
        G: Input matrix
        H: Observation matrix
        n_steps: Number of time steps

    Returns:
        n_steps x n_outputs array of impulse responses
    """
    n_inputs = G.shape[1] if G.ndim > 1 else 1
    n_outputs = H.shape[0]

    if G.ndim == 1:
        G = G.reshape(-1, 1)

    response = np.zeros((n_steps, n_outputs, n_inputs))

    F_power = np.eye(F.shape[0])
    for t in range(n_steps):
        response[t] = H @ F_power @ G
        F_power = F_power @ F

    # Return squeezed if single input/output
    if n_inputs == 1 and n_outputs == 1:
        return response.squeeze()

    return response

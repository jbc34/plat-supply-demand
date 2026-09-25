"""Parameter estimation for state-space models.

Provides Maximum Likelihood Estimation (MLE) and related methods
for estimating model parameters from observed data.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from supply_demand.models.state_space.kalman import KalmanFilter, FilterResults
from supply_demand.models.state_space.specification import (
    StateSpaceParams,
    build_transition_matrix,
    build_process_noise_matrix,
)


@dataclass
class EstimationResult:
    """Result of parameter estimation."""

    params: StateSpaceParams
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_parameters: int
    convergence: bool
    message: str


class StateSpaceEstimator:
    """Base class for state-space model estimation."""

    def __init__(
        self,
        H: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        """Initialize estimator.

        Args:
            H: Observation matrix (fixed during estimation)
            R: Observation noise covariance (fixed during estimation)
            x0: Initial state mean
            P0: Initial state covariance
        """
        self.H = H
        self.R = R
        self.x0 = x0
        self.P0 = P0

        self.n_states = 6
        self.n_obs = H.shape[0]

    def log_likelihood(
        self,
        params: StateSpaceParams,
        Y: np.ndarray,
    ) -> float:
        """Compute log-likelihood for given parameters.

        Args:
            params: Model parameters
            Y: Observation matrix T x n_obs

        Returns:
            Log-likelihood value
        """
        F = build_transition_matrix(params)
        Q = build_process_noise_matrix(params)

        kf = KalmanFilter(
            F=F,
            H=self.H,
            Q=Q,
            R=self.R,
            x0=self.x0,
            P0=self.P0,
        )

        results = kf.filter(Y)
        return results.log_likelihood


class MLEEstimator(StateSpaceEstimator):
    """Maximum Likelihood Estimation for state-space models.

    Estimates model parameters by maximizing the likelihood
    of observed data using the Kalman filter.
    """

    # Parameter bounds (min, max)
    PARAM_BOUNDS = {
        "alpha_1": (0.01, 1.0),
        "alpha_2": (0.01, 1.0),
        "alpha_3": (0.01, 1.0),
        "beta_1": (0.01, 1.0),
        "beta_2": (0.01, 1.0),
        "beta_3": (0.01, 0.5),
        "gamma_1": (0.01, 1.0),
        "gamma_2": (0.01, 1.0),
        "gamma_3": (0.01, 1.0),
        "lambda_": (0.01, 0.5),
        "damping_D": (0.01, 0.5),
        "damping_S": (0.01, 0.5),
        "damping_K": (0.01, 0.5),
        "sigma_D": (0.001, 0.5),
        "sigma_S": (0.001, 0.5),
        "sigma_K": (0.001, 0.5),
    }

    def __init__(
        self,
        H: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        params_to_estimate: Optional[list[str]] = None,
    ):
        """Initialize MLE estimator.

        Args:
            H: Observation matrix
            R: Observation noise covariance
            x0: Initial state mean
            P0: Initial state covariance
            params_to_estimate: List of parameter names to estimate
                               (others fixed at initial values)
        """
        super().__init__(H, R, x0, P0)

        # Default parameters to estimate
        self.params_to_estimate = params_to_estimate or [
            "alpha_1", "alpha_2", "alpha_3",
            "beta_1", "beta_2", "beta_3",
            "gamma_1", "gamma_2", "gamma_3",
            "lambda_",
            "sigma_D", "sigma_S", "sigma_K",
        ]

    def _params_to_vector(self, params: StateSpaceParams) -> np.ndarray:
        """Convert parameters to optimization vector."""
        return np.array([
            getattr(params, name) for name in self.params_to_estimate
        ])

    def _vector_to_params(
        self,
        x: np.ndarray,
        base_params: StateSpaceParams
    ) -> StateSpaceParams:
        """Convert optimization vector to parameters."""
        params_dict = base_params.to_dict()
        for i, name in enumerate(self.params_to_estimate):
            params_dict[name] = x[i]
        return StateSpaceParams.from_dict(params_dict)

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization."""
        return [
            self.PARAM_BOUNDS.get(name, (0.001, 1.0))
            for name in self.params_to_estimate
        ]

    def _objective(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        base_params: StateSpaceParams,
    ) -> float:
        """Negative log-likelihood objective function."""
        params = self._vector_to_params(x, base_params)

        try:
            ll = self.log_likelihood(params, Y)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll  # Minimize negative log-likelihood
        except Exception:
            return 1e10

    def estimate(
        self,
        Y: np.ndarray,
        initial_params: Optional[StateSpaceParams] = None,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        verbose: bool = False,
    ) -> EstimationResult:
        """Estimate parameters via MLE.

        Args:
            Y: Observation matrix T x n_obs
            initial_params: Starting parameter values
            method: Optimization method
            maxiter: Maximum iterations
            verbose: Print progress

        Returns:
            EstimationResult with optimal parameters
        """
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T = Y.shape[0]

        if initial_params is None:
            initial_params = StateSpaceParams()

        x0 = self._params_to_vector(initial_params)
        bounds = self._get_bounds()

        # Optimization
        result = minimize(
            self._objective,
            x0,
            args=(Y, initial_params),
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter, "disp": verbose},
        )

        # Extract optimal parameters
        optimal_params = self._vector_to_params(result.x, initial_params)
        log_likelihood = -result.fun

        # Information criteria
        n_params = len(self.params_to_estimate)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(T) * n_params - 2 * log_likelihood

        return EstimationResult(
            params=optimal_params,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=T,
            n_parameters=n_params,
            convergence=result.success,
            message=result.message,
        )

    def estimate_with_restarts(
        self,
        Y: np.ndarray,
        n_restarts: int = 5,
        **kwargs,
    ) -> EstimationResult:
        """Estimate with multiple random restarts.

        Args:
            Y: Observation matrix
            n_restarts: Number of random restarts
            **kwargs: Additional arguments to estimate()

        Returns:
            Best EstimationResult across restarts
        """
        best_result = None
        best_ll = -np.inf

        for i in range(n_restarts):
            # Random initial parameters
            initial_dict = {}
            for name in self.params_to_estimate:
                low, high = self.PARAM_BOUNDS.get(name, (0.01, 0.5))
                initial_dict[name] = np.random.uniform(low, high)

            initial_params = StateSpaceParams.from_dict(initial_dict)

            try:
                result = self.estimate(Y, initial_params, **kwargs)

                if result.log_likelihood > best_ll:
                    best_ll = result.log_likelihood
                    best_result = result
            except Exception as e:
                print(f"Restart {i} failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("All estimation attempts failed")

        return best_result


def compute_standard_errors(
    estimator: MLEEstimator,
    params: StateSpaceParams,
    Y: np.ndarray,
    eps: float = 1e-5,
) -> dict[str, float]:
    """Compute parameter standard errors via numerical Hessian.

    Args:
        estimator: MLEEstimator instance
        params: Estimated parameters
        Y: Observation matrix
        eps: Finite difference step size

    Returns:
        Dictionary mapping parameter names to standard errors
    """
    x = estimator._params_to_vector(params)
    n = len(x)

    # Compute Hessian numerically
    H = np.zeros((n, n))

    f0 = -estimator._objective(x, Y, StateSpaceParams())

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps

            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps

            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps

            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps

            f_pp = -estimator._objective(x_pp, Y, StateSpaceParams())
            f_pm = -estimator._objective(x_pm, Y, StateSpaceParams())
            f_mp = -estimator._objective(x_mp, Y, StateSpaceParams())
            f_mm = -estimator._objective(x_mm, Y, StateSpaceParams())

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
            H[j, i] = H[i, j]

    # Standard errors from inverse Hessian
    try:
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)

    return {
        name: se[i] for i, name in enumerate(estimator.params_to_estimate)
    }

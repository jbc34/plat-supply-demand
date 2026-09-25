"""Kalman filter for state estimation.

Provides:
- Filtering: P(x_t | y_1:t)
- Smoothing: P(x_t | y_1:T)
- Prediction: P(x_{t+h} | y_1:t)

Handles missing observations by marginalizing over missing values.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class KalmanState:
    """Current state of the Kalman filter."""

    x: np.ndarray          # State mean
    P: np.ndarray          # State covariance
    log_likelihood: float  # Running log-likelihood


@dataclass
class FilterResults:
    """Complete results from Kalman filtering."""

    filtered_states: np.ndarray    # T x n_states
    filtered_covs: np.ndarray      # T x n_states x n_states
    predicted_states: np.ndarray   # T x n_states
    predicted_covs: np.ndarray     # T x n_states x n_states
    log_likelihood: float          # Total log-likelihood
    innovations: np.ndarray        # T x n_obs (innovation residuals)
    innovation_covs: np.ndarray    # T x n_obs x n_obs

    def to_dataframe(
        self,
        index: Optional[pd.DatetimeIndex] = None,
        state_names: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Convert filtered states to DataFrame.

        Args:
            index: DateTime index for the results
            state_names: Names for state variables

        Returns:
            DataFrame with filtered state estimates
        """
        if state_names is None:
            state_names = ["D", "S", "K", "dD", "dS", "dK"]

        n_states = self.filtered_states.shape[1]
        state_names = state_names[:n_states]

        df = pd.DataFrame(
            self.filtered_states,
            columns=state_names,
            index=index
        )

        # Add standard errors
        for i, name in enumerate(state_names):
            df[f"{name}_se"] = np.sqrt(self.filtered_covs[:, i, i])

        return df


class KalmanFilter:
    """Standard Kalman filter with support for missing observations.

    Implements the standard Kalman filter algorithm:
    - Predict step: x_pred = F @ x, P_pred = F @ P @ F' + Q
    - Update step: x_upd = x_pred + K @ (y - H @ x_pred)

    Handles missing values (NaN) by marginalizing over missing observations.

    Example:
        >>> kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
        >>> results = kf.filter(observations)
        >>> smoothed = kf.smooth(results)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        """Initialize Kalman filter.

        Args:
            F: State transition matrix (n_states x n_states)
            H: Observation matrix (n_obs x n_states)
            Q: Process noise covariance (n_states x n_states)
            R: Observation noise covariance (n_obs x n_obs)
            x0: Initial state mean (n_states,)
            P0: Initial state covariance (n_states x n_states)
        """
        self.F = np.asarray(F)
        self.H = np.asarray(H)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

        self.n_states = F.shape[0]
        self.n_obs = H.shape[0]

        # Initialize state
        self.x = x0 if x0 is not None else np.zeros(self.n_states)
        self.P = P0 if P0 is not None else np.eye(self.n_states)

    def predict(
        self,
        x: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step: propagate state forward.

        Args:
            x: Current state mean (uses internal state if None)
            P: Current state covariance (uses internal if None)

        Returns:
            Tuple of (predicted mean, predicted covariance)
        """
        if x is None:
            x = self.x
        if P is None:
            P = self.P

        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q

        return x_pred, P_pred

    def update(
        self,
        y: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """Update step: incorporate observation.

        Handles missing values (NaN) by marginalizing over missing observations.

        Args:
            y: Observation vector (n_obs,) - NaN for missing
            x_pred: Predicted state mean
            P_pred: Predicted state covariance

        Returns:
            Tuple of (updated mean, updated cov, log-likelihood,
                     innovation, innovation covariance)
        """
        # Handle missing observations
        valid = ~np.isnan(y)

        if not valid.any():
            # All missing: no update
            innov = np.full(self.n_obs, np.nan)
            S = np.full((self.n_obs, self.n_obs), np.nan)
            return x_pred, P_pred, 0.0, innov, S

        # Extract valid observations
        y_valid = y[valid]
        H_valid = self.H[valid, :]
        R_valid = self.R[np.ix_(valid, valid)]

        # Predicted observation
        y_pred = H_valid @ x_pred

        # Innovation (prediction error)
        innov_valid = y_valid - y_pred

        # Innovation covariance
        S_valid = H_valid @ P_pred @ H_valid.T + R_valid

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S_valid)
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudoinverse
            S_inv = np.linalg.pinv(S_valid)

        K = P_pred @ H_valid.T @ S_inv

        # State update
        x_upd = x_pred + K @ innov_valid

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ H_valid
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R_valid @ K.T

        # Ensure symmetry
        P_upd = (P_upd + P_upd.T) / 2

        # Log-likelihood contribution
        sign, logdet = np.linalg.slogdet(S_valid)
        if sign <= 0:
            ll = -1e10  # Very negative for numerical issues
        else:
            ll = -0.5 * (
                len(y_valid) * np.log(2 * np.pi) +
                logdet +
                innov_valid @ S_inv @ innov_valid
            )

        # Full innovation vector (NaN for missing)
        innov = np.full(self.n_obs, np.nan)
        innov[valid] = innov_valid

        # Full innovation covariance
        S = np.full((self.n_obs, self.n_obs), np.nan)
        S[np.ix_(valid, valid)] = S_valid

        return x_upd, P_upd, ll, innov, S

    def filter(
        self,
        Y: np.ndarray,
        return_predictions: bool = True,
    ) -> FilterResults:
        """Run filter over full observation sequence.

        Args:
            Y: T x n_obs array of observations (NaN for missing)
            return_predictions: Whether to store one-step predictions

        Returns:
            FilterResults with full state trajectory
        """
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T = Y.shape[0]

        # Storage arrays
        filtered_states = np.zeros((T, self.n_states))
        filtered_covs = np.zeros((T, self.n_states, self.n_states))
        predicted_states = np.zeros((T, self.n_states))
        predicted_covs = np.zeros((T, self.n_states, self.n_states))
        innovations = np.zeros((T, self.n_obs))
        innovation_covs = np.zeros((T, self.n_obs, self.n_obs))

        total_ll = 0.0

        # Initialize
        x = self.x.copy()
        P = self.P.copy()

        for t in range(T):
            # Predict
            x_pred, P_pred = self.predict(x, P)
            predicted_states[t] = x_pred
            predicted_covs[t] = P_pred

            # Update
            x_upd, P_upd, ll, innov, S = self.update(Y[t], x_pred, P_pred)
            filtered_states[t] = x_upd
            filtered_covs[t] = P_upd
            innovations[t] = innov
            innovation_covs[t] = S
            total_ll += ll

            # Update internal state for next iteration
            x = x_upd
            P = P_upd

        # Update internal state
        self.x = x
        self.P = P

        return FilterResults(
            filtered_states=filtered_states,
            filtered_covs=filtered_covs,
            predicted_states=predicted_states,
            predicted_covs=predicted_covs,
            log_likelihood=total_ll,
            innovations=innovations,
            innovation_covs=innovation_covs,
        )

    def smooth(
        self,
        filter_results: FilterResults,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rauch-Tung-Striebel smoother for retrospective estimation.

        Uses future information to improve past state estimates.

        Args:
            filter_results: Results from forward filter pass

        Returns:
            Tuple of (smoothed states T x n_states, smoothed covs T x n x n)
        """
        T = filter_results.filtered_states.shape[0]

        smoothed_states = np.zeros_like(filter_results.filtered_states)
        smoothed_covs = np.zeros_like(filter_results.filtered_covs)

        # Initialize with final filtered values
        smoothed_states[-1] = filter_results.filtered_states[-1]
        smoothed_covs[-1] = filter_results.filtered_covs[-1]

        # Backward pass
        for t in range(T - 2, -1, -1):
            P_filt = filter_results.filtered_covs[t]
            P_pred = filter_results.predicted_covs[t + 1]

            # Smoother gain
            try:
                P_pred_inv = np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_pred)

            J = P_filt @ self.F.T @ P_pred_inv

            # Smoothed estimates
            smoothed_states[t] = (
                filter_results.filtered_states[t] +
                J @ (smoothed_states[t + 1] - filter_results.predicted_states[t + 1])
            )

            smoothed_covs[t] = (
                P_filt +
                J @ (smoothed_covs[t + 1] - P_pred) @ J.T
            )

            # Ensure symmetry
            smoothed_covs[t] = (smoothed_covs[t] + smoothed_covs[t].T) / 2

        return smoothed_states, smoothed_covs

    def forecast(
        self,
        h: int,
        x: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-step ahead forecast.

        Args:
            h: Forecast horizon (number of steps)
            x: Initial state (uses internal if None)
            P: Initial covariance (uses internal if None)

        Returns:
            Tuple of (forecast means h x n_states, forecast covs h x n x n)
        """
        if x is None:
            x = self.x
        if P is None:
            P = self.P

        forecast_means = np.zeros((h, self.n_states))
        forecast_covs = np.zeros((h, self.n_states, self.n_states))

        x_curr = x.copy()
        P_curr = P.copy()

        for i in range(h):
            x_curr, P_curr = self.predict(x_curr, P_curr)
            forecast_means[i] = x_curr
            forecast_covs[i] = P_curr

        return forecast_means, forecast_covs

    def reset(
        self,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """Reset filter to initial state.

        Args:
            x0: New initial state mean
            P0: New initial state covariance
        """
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n_states)

        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.eye(self.n_states)

"""VAR(p) model for the endogenous rent-vacancy-supply system.

Models rent growth, vacancy, deliveries, and absorption as a jointly
determined system where each variable depends on its own lags and
the lags of all other variables. Captures feedback loops that
single-equation models miss.

Supports two estimation modes:
- "ols" (default): standard frequentist VAR via statsmodels
- "bayesian": Minnesota prior shrinkage — penalizes cross-variable
  lags harder than own lags, preventing over-parameterization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from econometrics.models.base import ForecastResult, IRFResult


class VAREngine:
    """VAR(p) with automatic lag selection, optional Minnesota prior, and exog support."""

    def __init__(
        self,
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
        max_lags: int = 8,
        ic: str = "aic",
        target_col: str = "rent_growth_yoy",
        method: str = "bayesian",
        minnesota_lambda: float = 0.2,
        minnesota_cross_shrink: float = 0.5,
    ):
        """
        Args:
            endog_cols: Endogenous variable column names. Default: standard 4-var system.
            exog_cols: Exogenous variable column names (optional).
            max_lags: Maximum lag order to consider.
            ic: Information criterion for lag selection: "aic", "bic", "hqic", "fpe".
            target_col: Which endogenous variable is the primary forecast target.
            method: "ols" for frequentist, "bayesian" for Minnesota prior.
            minnesota_lambda: Overall tightness of the Minnesota prior.
                Lower = more shrinkage toward zero. 0.2 is standard.
            minnesota_cross_shrink: Extra shrinkage for cross-variable lags (0-1).
                0.5 means cross lags get 50% of the variance of own lags.
        """
        self.endog_cols = endog_cols or [
            "rent_growth_yoy", "vacancy_rate",
            "deliveries_pct_inventory", "absorption_pct_inventory",
        ]
        self.exog_cols = exog_cols
        self.max_lags = max_lags
        self.ic = ic
        self.target_col = target_col
        self.method = method
        self.minnesota_lambda = minnesota_lambda
        self.minnesota_cross_shrink = minnesota_cross_shrink

        self._result = None       # statsmodels VARResults (OLS path)
        self._bvar_coefs = None   # dict of per-equation coef arrays (Bayesian path)
        self._bvar_intercepts = None
        self._bvar_sigma_sq = None
        self._panel = None
        self._endog = None
        self._exog = None
        self._selected_lag = None
        self._resid_df = None
        self._locf_logged = False

    def _prepare_data(self, panel: pd.DataFrame) -> None:
        """Extract and clean endogenous + exogenous data."""
        self._panel = panel

        # Preserve self.endog_cols declaration order exactly — the Minnesota
        # prior penalty matrix and coefficient vector are positionally indexed,
        # so any reordering would silently misassign coefficients to equations.
        available_endog = [c for c in self.endog_cols if c in panel.columns]
        if len(available_endog) < 2:
            raise ValueError(
                f"VAR requires at least 2 endogenous variables, found: {available_endog}"
            )

        # Explicit column selection with ordered list — never rely on panel col order.
        self._endog = panel.loc[:, available_endog].dropna().copy()
        # Hard raise (not assert) — this guards coefficient alignment in the
        # Minnesota prior. Assertions are stripped under `python -O`, and a
        # silent ordering drift would misassign every coefficient.
        if list(self._endog.columns) != available_endog:
            raise RuntimeError(
                "Endogenous column ordering was altered during selection. "
                "Minnesota prior math requires stable ordering — refusing to fit."
            )

        # Extract exogenous — forward-fill NaN, then drop any remaining NaN rows
        if self.exog_cols:
            available_exog = [c for c in self.exog_cols if c in panel.columns]
            if available_exog:
                exog = panel[available_exog].reindex(self._endog.index).ffill().bfill()
                # Drop rows where exog still has NaN
                valid = exog.notna().all(axis=1)
                self._endog = self._endog.loc[valid]
                self._exog = exog.loc[valid]
            else:
                self._exog = None
        else:
            self._exog = None

        if len(self._endog) < 20:
            raise ValueError(
                f"VAR needs at least 20 observations, got {len(self._endog)}"
            )

    def _select_lag_order(self) -> int:
        """Select lag order via information criterion."""
        model = VAR(self._endog, exog=self._exog)
        try:
            max_lag = min(self.max_lags, len(self._endog) // 5)
            lag_order = model.select_order(maxlags=max(max_lag, 1))
            p = lag_order.selected_orders.get(self.ic, 2)
            return max(p, 1)
        except Exception:
            return 2

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit VAR model.

        If method="bayesian", uses Minnesota prior shrinkage (Ridge per equation
        with structured penalty). If method="ols", uses standard statsmodels VAR.
        """
        self._prepare_data(panel)
        self._selected_lag = self._select_lag_order()

        if self.method == "bayesian":
            self._fit_bayesian()
        else:
            self._fit_ols()

    def _fit_ols(self) -> None:
        """Standard OLS VAR via statsmodels."""
        model = VAR(self._endog, exog=self._exog)
        self._result = model.fit(self._selected_lag)
        self._resid_df = self._result.resid

    def _fit_bayesian(self) -> None:
        """Minnesota prior VAR: Ridge regression per equation with structured penalty.

        The Minnesota prior says:
        - Own lags get penalty lambda^2 / lag^2 (decay with lag distance)
        - Cross-variable lags get penalty lambda^2 * cross_shrink / lag^2
          (extra shrinkage for other variables' lags)
        - This prevents the over-parameterization that kills frequentist VAR
          with small samples.
        """
        p = self._selected_lag
        k = len(self._endog.columns)
        T = len(self._endog)

        Y = self._endog.values  # (T, k)

        # Build lagged regressor matrix: [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}, 1, exog]
        X_parts = []
        for lag in range(1, p + 1):
            X_parts.append(Y[p - lag:T - lag])

        X = np.column_stack(X_parts)  # (T-p, k*p)

        # Add intercept
        X = np.column_stack([X, np.ones(T - p)])

        # Add exogenous if present
        n_exog = 0
        if self._exog is not None:
            exog_vals = self._exog.values[p:]
            X = np.column_stack([X, exog_vals])
            n_exog = exog_vals.shape[1]

        Y_dep = Y[p:]  # (T-p, k)

        # Build Minnesota prior penalty matrix (diagonal)
        n_params = X.shape[1]  # k*p + 1 + n_exog
        lam = self.minnesota_lambda
        cross = self.minnesota_cross_shrink

        # Estimate per-variable variance from OLS AR(1) residuals (Litterman 1986).
        # Uses cov/var slope, NOT np.corrcoef, which would bound the slope to [-1,1]
        # and corrupt sigma_sq for variables with scales far from unit.
        sigma_sq = np.zeros(k)
        for i in range(k):
            y_i = Y[:, i]
            if len(y_i) > 2:
                y_lag = y_i[:-1]
                y_cur = y_i[1:]
                var_lag = float(np.var(y_lag, ddof=0))
                if var_lag > 0:
                    b = float(np.cov(y_cur, y_lag, ddof=0)[0, 1] / var_lag)
                    resid = y_cur - b * y_lag
                    sigma_sq[i] = float(np.var(resid, ddof=0))
                else:
                    sigma_sq[i] = float(np.var(y_i, ddof=0))
            else:
                sigma_sq[i] = float(np.var(y_i, ddof=0))
        sigma_sq[sigma_sq == 0] = 1e-10
        # Expose for testability
        self._bvar_sigma_sq = sigma_sq

        # Fit each equation with structured Ridge
        self._bvar_coefs = {}
        self._bvar_intercepts = {}
        resid_all = np.zeros_like(Y_dep)

        for eq_idx, eq_name in enumerate(self._endog.columns):
            # Build penalty vector for this equation
            penalty = np.zeros(n_params)

            for lag in range(1, p + 1):
                for var_idx in range(k):
                    param_idx = (lag - 1) * k + var_idx
                    if var_idx == eq_idx:
                        # Own lag: penalty = (sigma_i / lambda)^2 * lag^2
                        penalty[param_idx] = (lag / lam) ** 2
                    else:
                        # Cross lag: extra shrinkage
                        penalty[param_idx] = (lag / (lam * cross)) ** 2 * (sigma_sq[eq_idx] / sigma_sq[var_idx])

            # No penalty on intercept and exog
            # (penalty for intercept and exog stays at 0 = no regularization)

            # Ridge regression: (X'X + diag(penalty))^{-1} X'y
            y_eq = Y_dep[:, eq_idx]
            XtX = X.T @ X
            Xty = X.T @ y_eq
            ridge_mat = XtX + np.diag(penalty)

            try:
                coef = np.linalg.solve(ridge_mat, Xty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(ridge_mat, Xty, rcond=None)[0]

            self._bvar_coefs[eq_name] = coef
            self._bvar_intercepts[eq_name] = coef[k * p]  # intercept position

            # Residuals
            resid_all[:, eq_idx] = y_eq - X @ coef

        self._resid_df = pd.DataFrame(
            resid_all,
            columns=self._endog.columns,
            index=self._endog.index[p:],
        )

        # Also fit OLS VAR for IRF / forecast_interval (used as fallback)
        try:
            model = VAR(self._endog, exog=self._exog)
            self._result = model.fit(self._selected_lag)
        except Exception:
            self._result = None

    def predict(self, horizon: int = 8, metro_id: str = "", vintage_date: str = "") -> ForecastResult:
        """Forecast with confidence intervals."""
        if self._resid_df is None:
            raise RuntimeError("Must call fit() before predict()")

        if self.method == "bayesian" and self._bvar_coefs:
            return self._predict_bayesian(horizon, metro_id, vintage_date)
        elif self._result is not None:
            return self._predict_ols(horizon, metro_id, vintage_date)
        else:
            raise RuntimeError("No fitted model available")

    def _predict_bayesian(self, horizon: int, metro_id: str, vintage_date: str) -> ForecastResult:
        """Recursive multi-step forecast from Bayesian VAR coefficients."""
        p = self._selected_lag
        k = len(self._endog.columns)
        Y = self._endog.values

        # Initialize with last p observations
        y_history = Y[-p:].copy()  # (p, k)

        # Exog future values — held constant at last observed value (LOCF).
        # Silent modeling assumption: rising mortgage rates or worsening
        # unemployment will NOT be reflected in the forecast unless the caller
        # passes explicit future exog values. Logged once per fit/predict cycle.
        if self._exog is not None:
            exog_last = self._exog.values[-1:]
            if horizon > 1 and not self._locf_logged:
                print(
                    f"   [VAREngine] NOTE: {self._exog.shape[1]} exog variable(s) held "
                    f"constant at last observed values for all {horizon} forecast periods."
                )
                self._locf_logged = True
        else:
            exog_last = None

        forecasts = np.zeros((horizon, k))

        for h in range(horizon):
            # Build regressor row: [y_{t-1}, ..., y_{t-p}, 1, exog]
            x_row = []
            for lag in range(1, p + 1):
                x_row.append(y_history[-(lag)])
            x_row = np.concatenate(x_row)
            x_row = np.append(x_row, 1.0)  # intercept
            if exog_last is not None:
                x_row = np.append(x_row, exog_last.flatten())

            # Predict each equation
            y_new = np.zeros(k)
            for eq_idx, eq_name in enumerate(self._endog.columns):
                y_new[eq_idx] = x_row @ self._bvar_coefs[eq_name]

            forecasts[h] = y_new
            # Update history
            y_history = np.vstack([y_history, y_new])

        # CI from residual std, widening with horizon
        resid_std = self._resid_df.std().values
        target_idx = list(self._endog.columns).index(self.target_col)
        std = resid_std[target_idx]
        horizon_scale = np.sqrt(np.arange(1, horizon + 1))

        point = forecasts[:, target_idx]

        last_period = self._endog.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=point,
            ci_80_lo=point - 1.28 * std * horizon_scale,
            ci_80_hi=point + 1.28 * std * horizon_scale,
            ci_95_lo=point - 1.96 * std * horizon_scale,
            ci_95_hi=point + 1.96 * std * horizon_scale,
            model_name="VAREngine",
            metro_id=metro_id,
            vintage_date=vintage_date,
            horizon_quarters=horizon,
            forecast_periods=periods,
            residuals=self._resid_df[self.target_col].values,
        )

    def _predict_ols(self, horizon: int, metro_id: str, vintage_date: str) -> ForecastResult:
        """Forecast from OLS VAR via statsmodels."""
        k_ar = self._result.k_ar
        y_last = self._endog.values[-k_ar:]

        if self._exog is not None:
            exog_future = np.tile(self._exog.values[-1:], (horizon, 1))
            if horizon > 1 and not self._locf_logged:
                print(
                    f"   [VAREngine] NOTE: {self._exog.shape[1]} exog variable(s) held "
                    f"constant at last observed values for all {horizon} forecast periods."
                )
                self._locf_logged = True
        else:
            exog_future = None

        point_all = self._result.forecast(y_last, steps=horizon, exog_future=exog_future)

        try:
            _, lo_80, hi_80 = self._result.forecast_interval(
                y_last, steps=horizon, alpha=0.20, exog_future=exog_future,
            )
            _, lo_95, hi_95 = self._result.forecast_interval(
                y_last, steps=horizon, alpha=0.05, exog_future=exog_future,
            )
        except Exception:
            resid_std = self._result.resid.std(axis=0)
            target_idx = list(self._endog.columns).index(self.target_col)
            std = resid_std[target_idx]
            lo_80 = point_all - 1.28 * std
            hi_80 = point_all + 1.28 * std
            lo_95 = point_all - 1.96 * std
            hi_95 = point_all + 1.96 * std

        target_idx = list(self._endog.columns).index(self.target_col)
        last_period = self._endog.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=point_all[:, target_idx],
            ci_80_lo=lo_80[:, target_idx],
            ci_80_hi=hi_80[:, target_idx],
            ci_95_lo=lo_95[:, target_idx],
            ci_95_hi=hi_95[:, target_idx],
            model_name="VAREngine",
            metro_id=metro_id,
            vintage_date=vintage_date,
            horizon_quarters=horizon,
            forecast_periods=periods,
            residuals=self._resid_df[self.target_col].values if self._resid_df is not None else self._result.resid[self.target_col].values,
        )

    def residuals(self) -> pd.Series:
        """Return in-sample residuals for the target equation."""
        if self._resid_df is None:
            raise RuntimeError("Must call fit() before residuals()")
        return self._resid_df[self.target_col]

    def model_name(self) -> str:
        return "VAREngine"

    @property
    def selected_lag(self) -> int | None:
        return self._selected_lag

    @property
    def summary(self) -> str:
        if self._result is None:
            return f"Bayesian VAR(p={self._selected_lag}), Minnesota prior lambda={self.minnesota_lambda}"
        return str(self._result.summary())

    def irf(self, periods: int = 20) -> list[IRFResult]:
        """Compute impulse response functions.

        Uses the OLS VAR result (fitted as fallback even in Bayesian mode)
        since IRF computation requires the full VARResults object.
        """
        if self._result is None:
            raise RuntimeError("IRF requires OLS VAR result (not available)")

        irf_result = self._result.irf(periods=periods)
        target_idx = list(self._endog.columns).index(self.target_col)

        results = []
        for i, impulse_var in enumerate(self._endog.columns):
            values = irf_result.irfs[:, i, target_idx]

            try:
                ci = irf_result.ci(alpha=0.05)
                ci_lo = ci[:, i, target_idx, 0] if ci is not None else None
                ci_hi = ci[:, i, target_idx, 1] if ci is not None else None
            except Exception:
                ci_lo = None
                ci_hi = None

            results.append(IRFResult(
                periods=periods,
                impulse_var=impulse_var,
                response_var=self.target_col,
                irf_values=values,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            ))

        return results

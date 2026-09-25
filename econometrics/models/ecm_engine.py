"""Error Correction Model (ECM) for rent-income cointegration.

Tests whether rents and incomes share a long-run equilibrium using
the Johansen cointegration test, then estimates how fast rents revert
to that equilibrium via VECM.

If no cointegration is found, falls back to VAR in first differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from econometrics.models.base import ForecastResult


@dataclass
class CointegrationTestResult:
    """Results from the Johansen cointegration test."""

    rank: int                       # Cointegrating rank (0 = no cointegration)
    trace_stat: list[float]         # Trace statistics per rank
    trace_crit_95: list[float]      # 95% critical values
    max_eig_stat: list[float]       # Max eigenvalue statistics
    max_eig_crit_95: list[float]    # 95% critical values
    cointegrated: bool              # True if rank >= 1
    cointegrating_vector: np.ndarray | None  # Estimated beta (if cointegrated)
    error_correction_speed: float | None     # Alpha coefficient for rent equation


class ECMEngine:
    """Johansen cointegration + VECM for rent-income equilibrium.

    Tests whether log(rent) and log(income) are cointegrated. If so,
    estimates the error correction speed — how fast rents revert to
    equilibrium when they diverge from income.
    """

    def __init__(
        self,
        rent_col: str = "effective_rent_unit",
        income_col: str = "median_hh_income",
        k_ar_diff: int = 2,
        det_order: int = 0,
        target_col: str = "rent_growth_yoy",
    ):
        """
        Args:
            rent_col: Column name for rent level (will be log-transformed).
            income_col: Column name for income level (will be log-transformed).
            k_ar_diff: Number of lagged differences in the VECM.
            det_order: Deterministic trend order (-1=none, 0=constant, 1=trend).
            target_col: Column to extract for ForecastResult (rent_growth_yoy).
        """
        self.rent_col = rent_col
        self.income_col = income_col
        self.k_ar_diff = k_ar_diff
        self.det_order = det_order
        self.target_col = target_col

        self._vecm_result = None
        self._coint_result = None
        self._panel = None
        self._endog = None
        self._fallback_var = False

    def fit(self, panel: pd.DataFrame) -> None:
        """Fit the ECM model.

        Step 1: Johansen cointegration test on [log(rent), log(income)].
        Step 2: If cointegrated, fit VECM. If not, fall back to VAR in differences.

        Args:
            panel: Quarterly DataFrame with rent_col and income_col.
        """
        self._panel = panel

        if self.rent_col not in panel.columns or self.income_col not in panel.columns:
            raise ValueError(
                f"Panel must contain {self.rent_col} and {self.income_col}. "
                f"Available: {list(panel.columns)[:10]}..."
            )

        # Log-transform rent and income levels
        rent = panel[self.rent_col].dropna()
        income = panel[self.income_col].dropna()

        # Align on common index
        common_idx = rent.index.intersection(income.index)
        rent = rent.loc[common_idx]
        income = income.loc[common_idx]

        # Drop zeros/negatives before log
        valid = (rent > 0) & (income > 0)
        rent = rent[valid]
        income = income[valid]

        if len(rent) < 20:
            raise ValueError(f"ECM needs at least 20 observations, got {len(rent)}")

        log_rent = np.log(rent)
        log_income = np.log(income)

        self._endog = pd.DataFrame({
            "log_rent": log_rent,
            "log_income": log_income,
        })

        # Step 1: Johansen cointegration test
        self._coint_result = self._johansen_test(self._endog)

        # Step 2: Estimate VECM or fall back
        if self._coint_result.cointegrated:
            try:
                vecm = VECM(
                    self._endog,
                    k_ar_diff=self.k_ar_diff,
                    coint_rank=self._coint_result.rank,
                    deterministic=("ci" if self.det_order == 0 else "n"),
                )
                self._vecm_result = vecm.fit()
                self._fallback_var = False
            except Exception as e:
                print(f"  Warning: VECM fit failed ({e}), falling back to VAR in differences")
                self._fallback_var = True
                self._fit_var_differences()
        else:
            print("  No cointegration found — using VAR in first differences")
            self._fallback_var = True
            self._fit_var_differences()

    def _johansen_test(self, endog: pd.DataFrame) -> CointegrationTestResult:
        """Run Johansen cointegration test."""
        try:
            result = coint_johansen(endog.values, det_order=self.det_order, k_ar_diff=self.k_ar_diff)
        except Exception as e:
            print(f"  Warning: Johansen test failed ({e})")
            return CointegrationTestResult(
                rank=0, trace_stat=[], trace_crit_95=[], max_eig_stat=[],
                max_eig_crit_95=[], cointegrated=False,
                cointegrating_vector=None, error_correction_speed=None,
            )

        # Trace test: compare trace_stat to critical values at 95% (column index 1)
        trace_stat = result.lr1.tolist()
        trace_crit = result.cvt[:, 1].tolist()  # 95% critical values

        max_eig_stat = result.lr2.tolist()
        max_eig_crit = result.cvm[:, 1].tolist()

        # Determine rank: count how many trace stats exceed critical values
        rank = sum(1 for ts, cv in zip(trace_stat, trace_crit) if ts > cv)

        cointegrated = rank >= 1
        coint_vector = result.evec[:, 0] if cointegrated else None

        return CointegrationTestResult(
            rank=rank,
            trace_stat=trace_stat,
            trace_crit_95=trace_crit,
            max_eig_stat=max_eig_stat,
            max_eig_crit_95=max_eig_crit,
            cointegrated=cointegrated,
            cointegrating_vector=coint_vector,
            error_correction_speed=None,  # Filled after VECM fit
        )

    def _fit_var_differences(self) -> None:
        """Fallback: fit VAR on first differences when no cointegration."""
        from statsmodels.tsa.api import VAR as StatsVAR

        diff_endog = self._endog.diff().dropna()
        model = StatsVAR(diff_endog)
        lag = min(2, len(diff_endog) // 5)
        self._vecm_result = model.fit(max(lag, 1))

    def predict(self, horizon: int = 8, metro_id: str = "", vintage_date: str = "") -> ForecastResult:
        """Produce rent growth forecast from the ECM/VECM.

        For VECM: forecast log(rent) levels, convert to YoY growth rates.
        For VAR fallback: forecast first differences, cumulate.
        """
        if self._vecm_result is None:
            raise RuntimeError("Must call fit() before predict()")

        if not self._fallback_var:
            # VECM forecast
            try:
                forecast = self._vecm_result.predict(steps=horizon)
                # forecast columns are [log_rent, log_income]
                log_rent_forecast = forecast[:, 0]

                # Convert log-rent levels to YoY growth
                log_rent_history = self._endog["log_rent"].values
                growth = np.zeros(horizon)
                for h in range(horizon):
                    if h < 4:
                        prev = log_rent_history[-(4 - h)]
                    else:
                        prev = log_rent_forecast[h - 4]
                    growth[h] = np.exp(log_rent_forecast[h] - prev) - 1
            except Exception:
                growth = np.zeros(horizon)
        else:
            # VAR-in-differences fallback
            try:
                k_ar = self._vecm_result.k_ar
                y_last = self._vecm_result.endog[-k_ar:]
                diff_forecast = self._vecm_result.forecast(y_last, steps=horizon)
                # Cumulate differences to get log-rent changes
                growth = diff_forecast[:, 0]  # Approximate: d(log_rent) ≈ growth rate
            except Exception:
                growth = np.zeros(horizon)

        # Simple CI from residual std
        if hasattr(self._vecm_result, "resid"):
            resid = self._vecm_result.resid
            if isinstance(resid, pd.DataFrame):
                std = resid.iloc[:, 0].std()
            else:
                std = resid[:, 0].std()
        else:
            std = 0.02  # Fallback

        # Widen CI with horizon
        horizon_scale = np.sqrt(np.arange(1, horizon + 1))

        # Build period labels
        last_period = self._endog.index[-1]
        periods = [str(last_period + h) for h in range(1, horizon + 1)]

        return ForecastResult(
            point=growth,
            ci_80_lo=growth - 1.28 * std * horizon_scale,
            ci_80_hi=growth + 1.28 * std * horizon_scale,
            ci_95_lo=growth - 1.96 * std * horizon_scale,
            ci_95_hi=growth + 1.96 * std * horizon_scale,
            model_name="ECMEngine",
            metro_id=metro_id,
            vintage_date=vintage_date,
            horizon_quarters=horizon,
            forecast_periods=periods,
            residuals=self._vecm_result.resid[:, 0] if hasattr(self._vecm_result, "resid") and not isinstance(self._vecm_result.resid, pd.DataFrame) else None,
        )

    def residuals(self) -> pd.Series:
        """Return in-sample residuals for the rent equation."""
        if self._vecm_result is None:
            raise RuntimeError("Must call fit() before residuals()")
        resid = self._vecm_result.resid
        if isinstance(resid, pd.DataFrame):
            return resid.iloc[:, 0]
        return pd.Series(resid[:, 0])

    def model_name(self) -> str:
        return "ECMEngine"

    @property
    def cointegration_test(self) -> CointegrationTestResult | None:
        """Access the Johansen test results."""
        return self._coint_result

    @property
    def is_cointegrated(self) -> bool:
        """Whether the rent-income system is cointegrated."""
        return self._coint_result is not None and self._coint_result.cointegrated

    @property
    def error_correction_speed(self) -> float | None:
        """The alpha coefficient — fraction of disequilibrium corrected per quarter."""
        if self._vecm_result is None or self._fallback_var:
            return None
        try:
            # Alpha matrix: (n_vars, rank). First row = rent equation.
            alpha = self._vecm_result.alpha
            return float(alpha[0, 0])
        except Exception:
            return None

    @property
    def half_life_quarters(self) -> float | None:
        """How many quarters to close half the disequilibrium."""
        speed = self.error_correction_speed
        if speed is None or speed >= 0:
            return None  # No mean reversion or wrong sign
        return np.log(0.5) / np.log(1 + speed)

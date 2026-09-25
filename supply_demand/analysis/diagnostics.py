"""Model diagnostics and residual analysis."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf as sm_acf


@dataclass
class DiagnosticsResult:
    """Results of diagnostic tests."""

    ljung_box_stat: float
    ljung_box_pvalue: float
    normality_stat: float
    normality_pvalue: float
    heteroskedasticity_stat: float
    heteroskedasticity_pvalue: float
    mean_innovation: float
    std_innovation: float
    is_white_noise: bool


def compute_residual_diagnostics(
    innovations: np.ndarray,
    lags: int = 10,
    significance: float = 0.05,
) -> DiagnosticsResult:
    """Compute diagnostic statistics for filter innovations.

    Well-specified model should have white noise innovations.

    Args:
        innovations: Filter innovation (prediction error) series
        lags: Number of lags for autocorrelation tests
        significance: Significance level for tests

    Returns:
        DiagnosticsResult with test statistics
    """
    # Remove NaN
    innovations = innovations[~np.isnan(innovations)]
    n = len(innovations)

    if n < lags + 1:
        return DiagnosticsResult(
            ljung_box_stat=np.nan,
            ljung_box_pvalue=np.nan,
            normality_stat=np.nan,
            normality_pvalue=np.nan,
            heteroskedasticity_stat=np.nan,
            heteroskedasticity_pvalue=np.nan,
            mean_innovation=np.mean(innovations),
            std_innovation=np.std(innovations),
            is_white_noise=False,
        )

    # Ljung-Box test for autocorrelation
    lb_stat, lb_pvalue = ljung_box_test(innovations, lags)

    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = stats.jarque_bera(innovations)

    # ARCH test for heteroskedasticity
    arch_stat, arch_pvalue = arch_test(innovations, lags=5)

    # Overall white noise assessment
    is_white_noise = (
        lb_pvalue > significance and
        jb_pvalue > significance and
        arch_pvalue > significance
    )

    return DiagnosticsResult(
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_pvalue,
        normality_stat=jb_stat,
        normality_pvalue=jb_pvalue,
        heteroskedasticity_stat=arch_stat,
        heteroskedasticity_pvalue=arch_pvalue,
        mean_innovation=np.mean(innovations),
        std_innovation=np.std(innovations),
        is_white_noise=is_white_noise,
    )


def ljung_box_test(
    x: np.ndarray,
    lags: int = 10,
) -> Tuple[float, float]:
    """Ljung-Box test for autocorrelation."""
    result = acorr_ljungbox(x, lags=[lags], return_df=False)
    return float(result[0][-1]), float(result[1][-1])


def arch_test(
    x: np.ndarray,
    lags: int = 5,
) -> Tuple[float, float]:
    """ARCH test for heteroskedasticity."""
    try:
        lm_stat, lm_pvalue, _, _ = het_arch(x, nlags=lags)
        return float(lm_stat), float(lm_pvalue)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    except Exception:
        return np.nan, np.nan


def compute_acf(
    x: np.ndarray,
    nlags: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation function with confidence bands."""
    acf_values = sm_acf(x, nlags=nlags, fft=True)
    conf_int = 1.96 / np.sqrt(len(x))
    return acf_values, conf_int


def compute_information_criteria(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
) -> Tuple[float, float, float]:
    """Compute AIC, BIC, and HQIC.

    Args:
        log_likelihood: Model log-likelihood
        n_params: Number of estimated parameters
        n_obs: Number of observations

    Returns:
        Tuple of (AIC, BIC, HQIC)
    """
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_obs) * n_params - 2 * log_likelihood
    hqic = 2 * np.log(np.log(n_obs)) * n_params - 2 * log_likelihood

    return aic, bic, hqic

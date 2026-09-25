"""Optuna-based hyperparameter tuner for Bayesian VAR (Minnesota prior).

Finds optimal lambda, cross_shrink, lag-order IC, and exogenous variable
selection that minimizes RMSE on an expanding-window backtest at a given
forecast horizon.

Run via CLI:
    econ tune-bvar --metro dallas_tx --horizons 4 8 --n-trials 300

Results saved to data/econometrics/tuned/{metro_id}_bvar_params.yaml
and loaded automatically by the forecast pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# All candidate exogenous variables (must match panel_builder output names)
ALL_EXOG = [
    "mortgage_rate_30y",
    "unemployment_rate",
    "net_migration_returns",
    "income_delta_pct",
    "permit_pipeline",
    "concession_spread",
    "hh_growth_yoy",
    "renter_affordability",
]


def _backtest_bvar(
    panel: pd.DataFrame,
    horizon: int,
    lam: float,
    cross_shrink: float,
    ic: str,
    exog_cols: list[str],
    min_train: int = 40,
    step_quarters: int = 2,
) -> float:
    """Run expanding-window backtest for one BVAR config. Returns mean RMSE.

    Returns float("inf") if:
      - panel is too short,
      - >50% of folds fail to fit/predict (penalizes fail-fast configs),
      - fewer than 5 valid (actual, forecast) pairs survive.
    """
    from econometrics.models.var_engine import VAREngine

    n = len(panel)
    if n < min_train + horizon:
        return float("inf")

    all_actuals: list[float] = []
    all_forecasts: list[float] = []
    n_attempted = 0
    n_failed = 0

    for t in range(min_train, n - horizon + 1, step_quarters):
        n_attempted += 1
        train = panel.iloc[:t]
        test = panel.iloc[t: t + horizon]

        try:
            engine = VAREngine(
                exog_cols=exog_cols or None,
                method="bayesian",
                minnesota_lambda=lam,
                minnesota_cross_shrink=cross_shrink,
                ic=ic,
            )
            engine.fit(train)
            result = engine.predict(
                horizon=horizon,
                metro_id="",
                vintage_date=str(panel.index[t - 1]),
            )
        except Exception:
            n_failed += 1
            continue

        actuals = test["rent_growth_yoy"].values[:horizon]
        n_actual = min(len(actuals), len(result.point))
        if n_actual == 0:
            n_failed += 1
            continue

        all_actuals.extend(actuals[:n_actual].tolist())
        all_forecasts.extend(result.point[:n_actual].tolist())

    # Penalize configs that fail more than half the folds.
    if n_attempted > 0 and n_failed / n_attempted > 0.5:
        return float("inf")

    if len(all_actuals) < 5:
        return float("inf")

    a = np.array(all_actuals)
    f = np.array(all_forecasts)
    mask = ~(np.isnan(a) | np.isnan(f) | np.isinf(a) | np.isinf(f))
    if mask.sum() < 5:
        return float("inf")
    return float(np.sqrt(np.mean((a[mask] - f[mask]) ** 2)))


def tune_bvar(
    panel: pd.DataFrame,
    metro_id: str,
    horizons: list[int] | None = None,
    n_trials: int = 200,
    min_train: int = 40,
    step_quarters: int = 2,
    available_exog: list[str] | None = None,
    verbose: bool = True,
) -> dict[int, dict[str, Any]]:
    """Tune BVAR hyperparameters via Optuna TPE for each horizon.

    Args:
        panel: Quarterly panel DataFrame with rent_growth_yoy and exog columns.
        metro_id: Metro identifier (for logging).
        horizons: Forecast horizons to tune for (default: [4, 8] = 12mo, 24mo).
        n_trials: Number of Optuna trials per horizon.
        min_train: Minimum training set size for backtest folds.
        step_quarters: Step size between backtest origins.
        available_exog: Exog variables actually present in panel. Defaults to
            ALL_EXOG filtered to panel columns.
        verbose: Print Optuna progress.

    Returns:
        Dict mapping horizon → best params dict with keys:
            lambda, cross_shrink, ic, exog_cols, rmse
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna is required for BVAR tuning. "
            "Run: pip install optuna"
        )

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if horizons is None:
        horizons = [4, 8]

    if available_exog is None:
        available_exog = [c for c in ALL_EXOG if c in panel.columns]

    results: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        h_label = f"{horizon * 3}mo"
        if verbose:
            print(f"\n  Tuning BVAR for {metro_id} at h={horizon} ({h_label}), "
                  f"{n_trials} trials, {len(available_exog)} exog candidates...")

        def objective(trial: "optuna.Trial") -> float:
            # With corrected sigma_sq, optimizer may prefer minimal shrinkage
            # (lambda → ∞ approaches OLS). Range of 0.03-20.0 covers everything
            # from heavy shrinkage to near-OLS. If the optimum parks above 10.0,
            # the model gains essentially nothing from the Minnesota prior and
            # a pure OLS VAR would be equivalent.
            lam = trial.suggest_float("lambda", 0.03, 100.0, log=True)
            cross_shrink = trial.suggest_float("cross_shrink", 0.1, 1.0)
            ic = trial.suggest_categorical("ic", ["aic", "bic", "hqic"])

            # Each exog variable is independently included/excluded
            exog_cols = [
                v for v in available_exog
                if trial.suggest_categorical(f"use_{v}", [True, False])
            ]

            return _backtest_bvar(
                panel=panel,
                horizon=horizon,
                lam=lam,
                cross_shrink=cross_shrink,
                ic=ic,
                exog_cols=exog_cols,
                min_train=min_train,
                step_quarters=step_quarters,
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        best = study.best_params
        best_rmse = study.best_value

        best_exog = [v for v in available_exog if best.get(f"use_{v}", False)]

        results[horizon] = {
            "lambda": round(best["lambda"], 4),
            "cross_shrink": round(best["cross_shrink"], 4),
            "ic": best["ic"],
            "exog_cols": best_exog,
            "rmse": round(best_rmse, 5),
            "n_trials": n_trials,
            "h_quarters": horizon,
        }

        if verbose:
            print(f"  Best h={horizon}: RMSE={best_rmse:.4f}  "
                  f"lambda={best['lambda']:.3f}  "
                  f"cross_shrink={best['cross_shrink']:.3f}  "
                  f"ic={best['ic']}  "
                  f"exog={best_exog}")

    return results


def save_tuned_params(
    metro_id: str,
    tuned: dict[int, dict[str, Any]],
    tuned_dir: Path | None = None,
) -> Path:
    """Save tuned BVAR params to YAML.

    Args:
        metro_id: Metro identifier.
        tuned: Output of tune_bvar().
        tuned_dir: Directory to save to. Defaults to
            data/econometrics/tuned/ relative to project root.

    Returns:
        Path of the saved file.
    """
    import yaml
    from econometrics.config.settings import DATA_DIR

    if tuned_dir is None:
        tuned_dir = DATA_DIR / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)

    out_path = tuned_dir / f"{metro_id}_bvar_params.yaml"

    # Convert int keys to strings for YAML compatibility
    payload = {f"h{k}": v for k, v in tuned.items()}
    payload["metro_id"] = metro_id

    with open(out_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    return out_path


def load_tuned_params(metro_id: str, horizon: int) -> dict[str, Any] | None:
    """Load previously tuned BVAR params for a metro/horizon.

    Returns None if no tuned params exist yet (caller falls back to defaults).
    """
    import yaml
    from econometrics.config.settings import DATA_DIR

    path = DATA_DIR / "tuned" / f"{metro_id}_bvar_params.yaml"
    if not path.exists():
        return None

    with open(path) as f:
        data = yaml.safe_load(f)

    key = f"h{horizon}"
    return data.get(key)


def load_tuned_params_or_warn(metro_id: str, horizon: int) -> dict[str, Any] | None:
    """Like load_tuned_params, but prints a loud warning when the requested
    horizon has no tuned params saved for the metro.

    Forecast pipelines should prefer this over raw load_tuned_params to make
    silent fallback to default BVAR visible in CLI output.
    """
    params = load_tuned_params(metro_id, horizon)
    if params is None:
        import yaml
        from econometrics.config.settings import DATA_DIR

        path = DATA_DIR / "tuned" / f"{metro_id}_bvar_params.yaml"
        available_horizons: list[int] = []
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            available_horizons = sorted(
                int(k[1:]) for k in data.keys() if k.startswith("h") and k[1:].isdigit()
            )

        print(
            f"   ⚠️  WARNING: {metro_id} has NO TUNED BVAR params for horizon {horizon}. "
            f"Available tuned horizons: {available_horizons or 'none'}. "
            f"Falling back to DEFAULT BVAR (lambda=0.2) — forecast is NOT tuned."
        )
    return params

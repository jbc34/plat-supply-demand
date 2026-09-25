"""econ forecast — run full ensemble forecast pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def run_forecast(metro_id: str, horizon: int = 8, fetch: bool = True) -> None:
    """Full pipeline: load → build panel → fit all models → ensemble → report.

    Args:
        metro_id: Metro config ID (e.g., "dallas_tx").
        horizon: Forecast horizon in quarters.
        fetch: If True, fetch FRED/BLS/migration data from APIs.
    """
    from econometrics.backtest import registry
    from econometrics.config.settings import FORECASTS_DIR, REPORTS_DIR, load_metro_config
    from econometrics.data.transforms.panel_builder import build_panel
    from econometrics.ensemble.combiner import combine
    from econometrics.ensemble.weighter import compute_weights
    from econometrics.models.base import ForecastResult
    from econometrics.reports.chart_builder import fan_chart
    from econometrics.reports.markdown_writer import render_forecast_report

    print(f"═══ Econometric Forecast — {metro_id} ═══\n")

    # 1. Load config
    config = load_metro_config(metro_id)
    print(f"Metro: {config['display_name']}")

    # 2. Build panel
    print("\n1. Building quarterly panel...")
    panel = build_panel(config, fetch_macro=fetch, fetch_migration=fetch)
    print(f"   Panel: {len(panel)} quarters ({panel.index.min()} to {panel.index.max()})")

    # 3. Determine vintage date
    vintage_date = str(panel.index[-1])

    # 4. Fit all enabled models
    print(f"\n2. Fitting models (horizon={horizon}Q)...")
    enabled = config.get("enabled_models", [])
    model_results: list[ForecastResult] = []

    model_registry = {
        "VAREngine": ("econometrics.models.var_engine", "VAREngine"),
        "ECMEngine": ("econometrics.models.ecm_engine", "ECMEngine"),
        "RidgeLassoEngine": ("econometrics.models.ridge_lasso", "RidgeLassoEngine"),
        "GBMEngine": ("econometrics.models.gradient_boost", "GBMEngine"),
    }

    ecm_info = None

    # Build model kwargs from config
    exog_vars = config.get("exogenous_vars", [])
    # Use lagged versions for exogenous features (available in panel from panel_builder)
    exog_lagged = [f"{v}_lag1" for v in exog_vars if f"{v}_lag1" in panel.columns]
    # Also include raw exog that exist in panel
    exog_available = [v for v in exog_vars if v in panel.columns]

    # Load tuned BVAR params for the primary forecast horizon (if available)
    from econometrics.tuning.bvar_tuner import load_tuned_params_or_warn
    bvar_tuned = load_tuned_params_or_warn(metro_id, horizon)

    if bvar_tuned:
        print(f"   Using tuned BVAR params (h={horizon}): "
              f"lambda={bvar_tuned['lambda']}, "
              f"cross_shrink={bvar_tuned['cross_shrink']}, "
              f"ic={bvar_tuned['ic']}")
        var_exog = [c for c in (bvar_tuned.get("exog_cols") or []) if c in panel.columns]
        var_kwargs = {
            "exog_cols": var_exog or None,
            "method": "bayesian",
            "minnesota_lambda": bvar_tuned["lambda"],
            "minnesota_cross_shrink": bvar_tuned["cross_shrink"],
            "ic": bvar_tuned["ic"],
        }
    else:
        var_kwargs = {"exog_cols": exog_available or None}

    model_kwargs = {
        "VAREngine": var_kwargs,
        "ECMEngine": {},
        "RidgeLassoEngine": {"feature_cols": exog_lagged + [
            f"{v}_lag1" for v in config.get("endogenous_vars", [])
            if f"{v}_lag1" in panel.columns
        ]} if exog_lagged else None,
        "GBMEngine": {"feature_cols": exog_lagged + [
            f"{v}_lag1" for v in config.get("endogenous_vars", [])
            if f"{v}_lag1" in panel.columns
        ]} if exog_lagged else None,
        "ARIMAEngine": {},
        "ProphetEngine": {"regressor_cols": exog_available or None},
    }

    for model_name in enabled:
        if model_name not in model_registry:
            print(f"   {model_name}: unknown, skipping")
            continue

        module_path, cls_name = model_registry[model_name]
        try:
            module = __import__(module_path, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            kwargs = model_kwargs.get(model_name, {}) or {}
            engine = cls(**kwargs)
            engine.fit(panel)
            result = engine.predict(
                horizon=horizon, metro_id=metro_id, vintage_date=vintage_date,
            )
            model_results.append(result)
            print(f"   {model_name}: OK (Q1 forecast: {result.point[0] * 100:.1f}%)")

            # Capture ECM cointegration info
            if model_name == "ECMEngine":
                ecm_info = {
                    "cointegrated": engine.is_cointegrated,
                    "error_correction_speed": engine.error_correction_speed,
                    "half_life_quarters": engine.half_life_quarters,
                }
                ct = engine.cointegration_test
                if ct:
                    ecm_info["trace_stat"] = ct.trace_stat[0] if ct.trace_stat else None
                    ecm_info["trace_crit_95"] = ct.trace_crit_95[0] if ct.trace_crit_95 else None

        except Exception as e:
            print(f"   {model_name}: FAILED ({e})")

    if not model_results:
        print("\nNo models produced forecasts. Aborting.")
        return

    # 5. Ensemble weights
    print("\n3. Computing ensemble weights...")
    accuracy = registry.accuracy_summary(metro_id)
    weights = compute_weights(
        accuracy,
        enabled_models=[r.model_name for r in model_results],
        min_completed=config.get("min_backtest_quarters", 4),
        decay=config.get("ensemble_decay", 1.0),
    )
    for m, w in weights.items():
        print(f"   {m}: {w:.1%}")

    # 6. Combine
    print("\n4. Combining ensemble forecast...")
    ensemble = combine(model_results, weights)
    print(f"   Ensemble Q1: {ensemble.point[0] * 100:.1f}% "
          f"[{ensemble.ci_80_lo[0] * 100:.1f}, {ensemble.ci_80_hi[0] * 100:.1f}] 80% CI")

    # 7. Register forecasts
    print("\n5. Registering forecasts...")
    for result in model_results:
        registry.append(result)
    registry.append(ensemble)
    print(f"   {len(model_results) + 1} entries logged")

    # 8. Save outputs
    print("\n6. Saving outputs...")

    out_dir = FORECASTS_DIR / metro_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet
    pq_path = out_dir / f"{metro_id}_forecast_{vintage_date}.parquet"
    ensemble.to_dataframe().to_parquet(str(pq_path))
    print(f"   Parquet: {pq_path}")

    # Chart
    report_dir = REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    history = panel["rent_growth_yoy"].dropna().tail(20)
    chart_path = report_dir / f"forecast_{metro_id}_{vintage_date}_fan.png"
    fan_chart(
        ensemble, history=history,
        title=f"{config['display_name']} — Rent Growth Forecast ({vintage_date})",
        save_path=chart_path,
    )
    print(f"   Chart: {chart_path}")
    import matplotlib.pyplot as plt
    plt.close("all")

    # Markdown
    backtest_summary = registry.accuracy_summary(metro_id) or None
    md = render_forecast_report(
        ensemble, model_results, weights, config,
        cointegration_info=ecm_info,
        backtest_summary=backtest_summary if backtest_summary else None,
    )
    md_path = report_dir / f"forecast_{metro_id}_{vintage_date}.md"
    md_path.write_text(md)
    print(f"   Report: {md_path}")

    print(f"\n═══ Forecast complete ═══")

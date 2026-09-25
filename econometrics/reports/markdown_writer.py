"""Generate markdown forecast reports.

Produces structured markdown with standard sections matching
the market-study-agent report conventions.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from econometrics.models.base import ForecastResult
from econometrics.scenarios.stress_tester import StressTestResult


def render_forecast_report(
    ensemble_result: ForecastResult,
    model_results: list[ForecastResult],
    weights: dict[str, float],
    metro_config: dict,
    stress_results: list[StressTestResult] | None = None,
    cointegration_info: dict | None = None,
    backtest_summary: dict | None = None,
) -> str:
    """Render a complete forecast report as markdown.

    Args:
        ensemble_result: Combined ensemble forecast.
        model_results: Individual model forecasts.
        weights: Ensemble weights by model.
        metro_config: Metro YAML config dict.
        stress_results: Optional stress test results.
        cointegration_info: Optional ECM cointegration test results.
        backtest_summary: Optional accuracy summary by model.

    Returns:
        Markdown string.
    """
    metro = metro_config.get("display_name", metro_config.get("metro_id", "Unknown"))
    vintage = ensemble_result.vintage_date
    horizon = ensemble_result.horizon_quarters

    lines = []

    # --- Header ---
    lines.append(f"## Econometric Rent Growth Forecast — {metro}\n")
    lines.append(f"**Vintage**: {vintage}  ")
    lines.append(f"**Horizon**: {horizon} quarters  ")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Models**: {', '.join(weights.keys())}\n")

    # --- Forecast Table ---
    lines.append("### Forecast Summary\n")
    lines.append("| Period | Point (%) | 80% CI | 95% CI |")
    lines.append("| --- | ---: | --- | --- |")
    for i, period in enumerate(ensemble_result.forecast_periods):
        pt = f"{ensemble_result.point[i] * 100:.1f}"
        ci80 = f"[{ensemble_result.ci_80_lo[i] * 100:.1f}, {ensemble_result.ci_80_hi[i] * 100:.1f}]"
        ci95 = f"[{ensemble_result.ci_95_lo[i] * 100:.1f}, {ensemble_result.ci_95_hi[i] * 100:.1f}]"
        lines.append(f"| {period} | {pt} | {ci80} | {ci95} |")
    lines.append("")

    # --- Model Weights ---
    lines.append("### Ensemble Weights\n")
    lines.append("| Model | Weight | Point (Q1) |")
    lines.append("| --- | ---: | ---: |")
    for result in model_results:
        w = weights.get(result.model_name, 0)
        pt = f"{result.point[0] * 100:.1f}%" if len(result.point) > 0 else "N/A"
        lines.append(f"| {result.model_name} | {w:.1%} | {pt} |")
    lines.append("")

    # --- Cointegration ---
    if cointegration_info:
        lines.append("### Cointegration Analysis\n")
        if cointegration_info.get("cointegrated"):
            speed = cointegration_info.get("error_correction_speed")
            half_life = cointegration_info.get("half_life_quarters")
            lines.append(
                f"Rent and income are **cointegrated** "
                f"(Johansen trace stat {cointegration_info.get('trace_stat', 'N/A')} "
                f"> {cointegration_info.get('trace_crit_95', 'N/A')} critical at 95%).\n"
            )
            if speed is not None:
                lines.append(
                    f"Error correction speed: **{speed:.3f}** per quarter "
                    f"({abs(speed) * 100:.1f}% of disequilibrium corrected each quarter).\n"
                )
            if half_life is not None:
                lines.append(f"Half-life: **{half_life:.1f} quarters** ({half_life / 4:.1f} years).\n")
        else:
            lines.append("No cointegration detected between rent and income levels. "
                         "ECM fell back to VAR in first differences.\n")

    # --- Stress Tests ---
    if stress_results:
        lines.append("### Scenario Analysis\n")
        lines.append("| Scenario | Q1 Impact | Q4 Impact | Description |")
        lines.append("| --- | ---: | ---: | --- |")
        for sr in stress_results:
            q1_delta = f"{sr.delta_point[0] * 100:+.1f}pp" if len(sr.delta_point) > 0 else "N/A"
            q4_delta = f"{sr.delta_point[min(3, len(sr.delta_point) - 1)] * 100:+.1f}pp" if len(sr.delta_point) > 3 else "N/A"
            desc = sr.scenario_name.replace("_", " ")
            lines.append(f"| {sr.scenario_name} | {q1_delta} | {q4_delta} | {desc} |")
        lines.append("")

    # --- Backtest Accuracy ---
    if backtest_summary:
        lines.append("### Backtest Accuracy\n")
        lines.append("| Model | RMSE | MAE | Dir. Accuracy | CI80 Coverage | N |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for model, stats in backtest_summary.items():
            rmse_s = f"{stats['mean_rmse']:.4f}" if stats.get("mean_rmse") else "N/A"
            mae_s = f"{stats['mean_mae']:.4f}" if stats.get("mean_mae") else "N/A"
            dir_s = f"{stats['mean_directional_accuracy']:.0%}" if stats.get("mean_directional_accuracy") else "N/A"
            ci_s = f"{stats['mean_ci80_coverage']:.0%}" if stats.get("mean_ci80_coverage") else "N/A"
            n = stats.get("n_completed", 0)
            lines.append(f"| {model} | {rmse_s} | {mae_s} | {dir_s} | {ci_s} | {n} |")
        lines.append("")

    # --- Data Sources ---
    lines.append("### Data Sources\n")
    lines.append("- CoStar Submarket Time Series (quarterly, proprietary)")
    lines.append("- FRED: MORTGAGE30US, DGS10, FEDFUNDS, BAA10Y, T10Y2Y")
    lines.append("- BLS LAUS: MSA unemployment rate")
    lines.append("- Census BPS: Building permits (if configured)")
    lines.append("- IRS SOI: County-to-county migration (if configured)")
    lines.append("")

    return "\n".join(lines)

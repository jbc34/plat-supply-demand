"""Chart generation for econometric forecasts.

Produces 300 dpi PNGs for IC decks: fan charts, scenario overlays,
IRF plots, backtest accuracy, and ensemble weight evolution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from econometrics.models.base import ForecastResult, IRFResult
from econometrics.scenarios.stress_tester import StressTestResult

COLORS = {
    "primary": "#1B3A5C",
    "accent": "#2E86AB",
    "ci_80": "#6BAED6",
    "ci_95": "#C6DBEF",
    "baseline": "#1B3A5C",
    "stressed": "#D32F2F",
    "historical": "#333333",
}
FONT = "Helvetica Neue"


def fan_chart(
    result: ForecastResult,
    history: pd.Series | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple:
    """Fan chart: point forecast with shaded 80%/95% CI bands.

    Args:
        result: Ensemble or single-model ForecastResult.
        history: Historical rent_growth_yoy Series (plotted before forecast).
        title: Chart title.
        save_path: If given, save PNG to this path.
        dpi: Resolution.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")

    # Forecast periods as x-axis
    periods = result.forecast_periods
    x_forecast = np.arange(len(periods))

    # Plot history if provided
    if history is not None:
        x_history = np.arange(-len(history), 0)
        ax.plot(x_history, history.values * 100, color=COLORS["historical"],
                linewidth=1.5, label="Actual")

    # 95% CI band
    ax.fill_between(
        x_forecast, result.ci_95_lo * 100, result.ci_95_hi * 100,
        alpha=0.2, color=COLORS["ci_95"], label="95% CI",
    )
    # 80% CI band
    ax.fill_between(
        x_forecast, result.ci_80_lo * 100, result.ci_80_hi * 100,
        alpha=0.3, color=COLORS["ci_80"], label="80% CI",
    )
    # Point forecast
    ax.plot(x_forecast, result.point * 100, color=COLORS["primary"],
            linewidth=2.5, marker="o", markersize=5, label="Forecast")

    # Formatting
    ax.axhline(0, color="#999", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Rent Growth YoY (%)", fontsize=12, color="#333")
    ax.set_xlabel("Quarter", fontsize=12, color="#333")

    # X-axis labels
    all_labels = []
    if history is not None:
        all_labels.extend([str(p) for p in history.index[-min(8, len(history)):]])
    all_labels.extend(periods)
    # Simplify: just show forecast labels
    ax.set_xticks(x_forecast)
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=9)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold",
                      color=COLORS["primary"], loc="left", fontfamily=FONT)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig, ax


def scenario_overlay(
    stress_results: list[StressTestResult],
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple:
    """Overlay chart: baseline + multiple stress scenarios.

    Args:
        stress_results: List of StressTestResult objects.
        title: Chart title.
        save_path: Output path.
        dpi: Resolution.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")

    scenario_colors = ["#D32F2F", "#F57C00", "#1976D2", "#388E3C"]

    # Baseline from first result
    baseline = stress_results[0].baseline
    periods = baseline.forecast_periods
    x = np.arange(len(periods))

    ax.plot(x, baseline.point * 100, color=COLORS["baseline"],
            linewidth=2.5, marker="o", markersize=5, label="Baseline")

    for i, sr in enumerate(stress_results):
        color = scenario_colors[i % len(scenario_colors)]
        ax.plot(x, sr.stressed.point * 100, color=color,
                linewidth=2, linestyle="--", marker="s", markersize=4,
                label=sr.scenario_name)

    ax.axhline(0, color="#999", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Rent Growth YoY (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=9)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold",
                      color=COLORS["primary"], loc="left", fontfamily=FONT)

    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig, ax


def irf_chart(
    irf_results: list[IRFResult],
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple:
    """Plot impulse response functions from VAR.

    Args:
        irf_results: List of IRFResult (one per impulse variable).
        title: Chart title.
        save_path: Output path.
        dpi: Resolution.

    Returns:
        (fig, ax) tuple.
    """
    n = len(irf_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor="white", sharey=True)
    if n == 1:
        axes = [axes]

    impulse_colors = ["#D32F2F", "#F57C00", "#1976D2", "#388E3C"]

    for i, (irf, ax) in enumerate(zip(irf_results, axes)):
        x = np.arange(len(irf.irf_values))
        color = impulse_colors[i % len(impulse_colors)]

        ax.plot(x, irf.irf_values, color=color, linewidth=2)
        ax.axhline(0, color="#999", linewidth=0.5, linestyle="--")
        ax.fill_between(x, irf.irf_values, 0, alpha=0.1, color=color)

        if irf.ci_lo is not None and irf.ci_hi is not None:
            ax.fill_between(x, irf.ci_lo, irf.ci_hi, alpha=0.15, color=color)

        ax.set_title(f"Shock: {irf.impulse_var}", fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Quarters", fontsize=10)
        if i == 0:
            ax.set_ylabel(f"Response: {irf.response_var}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold",
                      color=COLORS["primary"], fontfamily=FONT, y=1.03)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig, axes

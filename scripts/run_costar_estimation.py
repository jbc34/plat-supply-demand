#!/usr/bin/env python3
"""End-to-end pipeline: CoStar DFW → Kalman Filter → Spiral Coordinates.

Demonstrates the full estimation pipeline using CoStar data as observations
instead of FRED/BLS proxies. This is the canonical example for the publication.

Usage:
    python3 scripts/run_costar_estimation.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from supply_demand.models.state_space.specification import (
    StateSpaceParams,
    StateSpaceModel,
)
from supply_demand.models.state_space.kalman import KalmanFilter
from supply_demand.models.coordinates.polar_transform import (
    extract_dominant_cycle,
    compute_spiral_trajectory,
    phase_to_quadrant,
    bandpass_filter,
)


def load_costar_dfw() -> pd.DataFrame:
    """Load and aggregate CoStar DFW data to MSA level."""
    df = pd.read_parquet("data/costar/submarket-clean/dfw_submarkets_actual.parquet")
    df = df[~df["is_forecast"]].copy()

    msa = (
        df.groupby("date")
        .agg(
            total_inventory=("inventory_units", "sum"),
            total_absorption=("absorption_units", "sum"),
            total_deliveries=("deliveries", "sum"),
            avg_vacancy_rate=("vacancy_rate", "mean"),
            cap_rate=("cap_rate", "mean"),
            asking_rent_growth_yoy=("asking_rent_growth_yoy", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    msa = msa[msa["date"] >= "2008-01-01"].reset_index(drop=True)
    return msa


def build_observations(msa: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Build observation matrix Y and proxy config from CoStar data.

    Maps CoStar fields to latent states:
        D ← absorption_rate (demand realized as fraction of stock)
        S ← delivery_rate (supply added as fraction of stock)
        K ← -cap_rate (inverted: lower cap rate = more capital inflow)
    """
    inv = msa["total_inventory"].replace(0, np.nan)

    D_obs = msa["total_absorption"] / inv
    S_obs = msa["total_deliveries"] / inv
    K_obs = -msa["cap_rate"]

    # Standardize
    D_std = (D_obs - D_obs.mean()) / (D_obs.std() + 1e-10)
    S_std = (S_obs - S_obs.mean()) / (S_obs.std() + 1e-10)
    K_std = (K_obs - K_obs.mean()) / (K_obs.std() + 1e-10)

    Y = np.column_stack([D_std.values, S_std.values, K_std.values])

    proxy_config = {
        "absorption_rate": {"state": "D", "loading": 1.0},
        "delivery_rate": {"state": "S", "loading": 1.0},
        "neg_cap_rate": {"state": "K", "loading": 1.0},
    }

    return Y, proxy_config


def run_kalman(Y: np.ndarray, proxy_config: dict) -> tuple:
    """Run Kalman filter on CoStar observations."""
    params = StateSpaceParams()
    model = StateSpaceModel(params, proxy_config)

    kf = KalmanFilter(
        F=model.F,
        H=model.H,
        Q=model.Q,
        R=model.R,
        x0=model.x0,
        P0=model.P0,
    )

    results = kf.filter(Y)
    return results, model


def compute_spiral(states: pd.DataFrame) -> tuple:
    """Compute spiral coordinates from filtered states."""
    D = states["D"].values
    S = states["S"].values
    K = states["K"].values

    # Use D-S spread method (recommended)
    amplitude, phase = extract_dominant_cycle(
        D, S, K,
        method="ds_spread",
        apply_bandpass=True,
    )

    # Full trajectory
    r, theta, r_dot, theta_dot = compute_spiral_trajectory(
        D, S, K,
        method="ds_spread",
        apply_bandpass=True,
    )

    quadrants = [phase_to_quadrant(t) for t in theta]

    return r, theta, r_dot, theta_dot, quadrants


def plot_pipeline(msa, states, r, theta, quadrants):
    """Plot the full pipeline output."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CoStar DFW → Kalman Filter → Spiral Coordinates", fontsize=14, fontweight="bold")
    dates = msa["date"]

    # Panel 1: Raw observations vs filtered states
    ax1 = axes[0, 0]
    for name, color in [("D", "#348ABD"), ("S", "#E24A33"), ("K", "#8EBA42")]:
        ax1.plot(dates, states[name], color=color, linewidth=1.5, label=f"Filtered {name}")
        ax1.fill_between(dates,
                         states[name] - 1.96 * states[f"{name}_se"],
                         states[name] + 1.96 * states[f"{name}_se"],
                         color=color, alpha=0.1)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_title("Kalman-Filtered State Estimates (D, S, K)")
    ax1.set_ylabel("State value")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: D-S spread
    ax2 = axes[0, 1]
    ds = states["D"] - states["S"]
    ax2.plot(dates, ds, color="#8B3A8B", linewidth=2)
    ax2.fill_between(dates, ds, 0, where=(ds > 0), alpha=0.2, color="#2ca02c", label="D > S (expansion)")
    ax2.fill_between(dates, ds, 0, where=(ds < 0), alpha=0.2, color="#d62728", label="S > D (contraction)")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_title("D-S Spread (Demand − Supply)")
    ax2.set_ylabel("Spread")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Spiral trajectory (polar plot)
    ax3 = axes[1, 0]
    ax3 = fig.add_subplot(2, 2, 3, projection="polar")
    # Color by time
    colors_time = plt.cm.viridis(np.linspace(0, 1, len(theta)))
    ax3.scatter(theta, r, c=range(len(theta)), cmap="viridis", s=15, alpha=0.8)
    ax3.set_title("Spiral Trajectory (θ = cycle phase, r = amplitude)", pad=20)
    # Mark current position
    ax3.scatter([theta[-1]], [r[-1]], color="red", s=100, zorder=5, marker="*")

    # Panel 4: Cycle phase over time
    ax4 = axes[1, 1]
    theta_deg = np.degrees(theta) % 360
    ax4.plot(dates, theta_deg, color="#8B3A8B", linewidth=1.5)
    ax4.set_ylabel("Phase (degrees)")
    ax4.set_title(f"Cycle Phase Over Time\nCurrent: {theta_deg[-1]:.0f}° ({quadrants[-1]})")

    # Add quadrant bands
    ax4.axhspan(0, 90, alpha=0.05, color="green", label="Early Expansion")
    ax4.axhspan(90, 180, alpha=0.05, color="orange", label="Peak / Late")
    ax4.axhspan(180, 270, alpha=0.05, color="red", label="Contraction")
    ax4.axhspan(270, 360, alpha=0.05, color="blue", label="Recovery")
    ax4.set_ylim(0, 360)
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = Path("research/figures/costar_pipeline_demo.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")


def main():
    print("=" * 60)
    print("CoStar DFW → Kalman Filter → Spiral Coordinates")
    print("=" * 60)

    print("\n[1/4] Loading CoStar DFW data...")
    msa = load_costar_dfw()
    print(f"  {len(msa)} quarters ({msa['date'].min().date()} to {msa['date'].max().date()})")

    print("[2/4] Building observations and running Kalman filter...")
    Y, proxy_config = build_observations(msa)
    results, model = run_kalman(Y, proxy_config)

    # Convert to DataFrame
    states = results.to_dataframe(
        index=pd.DatetimeIndex(msa["date"]),
        state_names=["D", "S", "K", "dD", "dS", "dK"],
    )
    print(f"  Log-likelihood: {results.log_likelihood:.2f}")
    print(f"  Stability: {model.check_stability()['is_stable']}")
    print(f"  Observability rank: {model.check_observability()['observability_matrix_rank']}/{model.n_states}")

    print("[3/4] Computing spiral coordinates...")
    D = states["D"].values
    S = states["S"].values
    K = states["K"].values

    r, theta, r_dot, theta_dot = compute_spiral_trajectory(
        D, S, K, method="ds_spread", apply_bandpass=True,
    )
    quadrants = [phase_to_quadrant(t) for t in theta]

    theta_deg = np.degrees(theta[-1]) % 360
    print(f"  Current position: r={r[-1]:.3f}, θ={theta_deg:.0f}° ({quadrants[-1]})")
    print(f"  Cycle velocity: ṙ={r_dot[-1]:.4f}, θ̇={np.degrees(theta_dot[-1]):.1f}°/quarter")

    print("[4/4] Generating pipeline demo figure...")
    plot_pipeline(msa, states, r, theta, quadrants)

    # Summary
    print("\n── PIPELINE SUMMARY ──────────────────────────────────")
    print(f"  Data: CoStar DFW, {len(msa)} quarters")
    print(f"  Model: 6-state Kalman filter (D, S, K + velocities)")
    print(f"  Observations: absorption_rate → D, delivery_rate → S, -cap_rate → K")
    print(f"  Phase extraction: D-S spread + bandpass (48-84 mo) + Hilbert")
    print(f"  Current cycle position: {quadrants[-1]} ({theta_deg:.0f}°)")
    print(f"  Current amplitude: {r[-1]:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

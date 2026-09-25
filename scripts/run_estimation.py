#!/usr/bin/env python
"""Run market estimation pipeline.

Usage:
    python scripts/run_estimation.py --market dallas --start 2010-01-01
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run market estimation")
    parser.add_argument(
        "--market",
        type=str,
        default="dallas",
        choices=["dallas", "houston", "austin", "san_antonio"],
        help="Market to analyze",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory",
    )

    args = parser.parse_args()

    print(f"Running estimation for {args.market} market...")
    print(f"Period: {args.start} to {args.end or 'present'}")

    # Import here to allow script to show help without dependencies
    from supply_demand.markets.multifamily.texas import TexasMultifamilyMarket
    from supply_demand.models.coordinates import compute_spiral_coordinates

    # Initialize market
    market = TexasMultifamilyMarket(msa=args.market)

    # Load data
    print("Loading data...")
    data = market.load_data(start_date=args.start, end_date=args.end)
    print(f"Loaded {len(data)} observations")

    if data.empty:
        print("Error: No data loaded. Check API keys in .env file.")
        return 1

    # Run estimation
    print("Running Kalman filter estimation...")
    try:
        states = market.run_estimation()
        print(f"Estimated {len(states)} state vectors")
    except Exception as e:
        print(f"Estimation failed: {e}")
        print("Using simplified estimation...")
        states = None

    # Get current position
    print("\nComputing spiral coordinates...")
    if states is not None:
        D = states["D"].values
        S = states["S"].values
        K = states["K"].values
    else:
        # Fall back to proxy-based estimation
        D = data.get("employment_growth", pd.Series([0])).fillna(0).values
        S = data.get("PERMIT", pd.Series([0])).fillna(0).values
        K = data.get("MORTGAGE30US", pd.Series([0])).fillna(0).values * -1

        # Standardize
        D = (D - D.mean()) / (D.std() + 1e-10)
        S = (S - S.mean()) / (S.std() + 1e-10)
        K = (K - K.mean()) / (K.std() + 1e-10)

    coords = compute_spiral_coordinates(D, S, K)

    # Print results
    print("\n" + "=" * 50)
    print(f"Market: {market.name}")
    print(f"Date: {data.index[-1].strftime('%Y-%m-%d')}")
    print("=" * 50)
    print(f"Cycle Phase: {coords.quadrant}")
    print(f"Amplitude (r): {coords.r:.3f}")
    print(f"Phase Angle: {coords.theta * 180 / 3.14159:.1f}°")
    print(f"Confidence: {coords.confidence:.1%}")
    print("=" * 50)

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if states is not None:
        states.to_csv(output_dir / f"{args.market}_states.csv")
        print(f"\nSaved states to {output_dir / f'{args.market}_states.csv'}")

    data.to_csv(output_dir / f"{args.market}_data.csv")
    print(f"Saved data to {output_dir / f'{args.market}_data.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())

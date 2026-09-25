"""CLI entrypoints for econometric forecasting.

Registered as 'econ' in pyproject.toml:
    econ forecast --metro dallas_tx
    econ backtest --metro dallas_tx
    econ stress --metro dallas_tx --scenario GFC_2008
"""

from __future__ import annotations

import argparse


def main():
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="econ",
        description="Econometric rent growth forecasting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Forecast
    fc = subparsers.add_parser("forecast", help="Run ensemble forecast")
    fc.add_argument("--metro", required=True, help="Metro config ID (e.g., dallas_tx)")
    fc.add_argument("--horizon", type=int, default=8, help="Forecast horizon in quarters")
    fc.add_argument("--no-fetch", action="store_true", help="Skip network data fetches")

    # Backtest
    bt = subparsers.add_parser("backtest", help="Run backtest or fill actuals")
    bt.add_argument("--metro", required=True)
    bt.add_argument("--fill-actuals", action="store_true", help="Fill actuals from latest panel")
    bt.add_argument("--summary", action="store_true", help="Print accuracy summary")

    # Stress test
    st = subparsers.add_parser("stress", help="Run stress test scenario")
    st.add_argument("--metro", required=True)
    st.add_argument("--scenario", required=True, help="Shock name from shock_library")
    st.add_argument("--horizon", type=int, default=8)

    # BVAR tuning
    tb = subparsers.add_parser("tune-bvar", help="Optuna hyperparameter search for BVAR")
    tb.add_argument("--metro", required=True)
    tb.add_argument(
        "--horizons", type=int, nargs="+", default=[4, 8],
        help="Forecast horizons in quarters to tune (default: 4 8 = 12mo 24mo)",
    )
    tb.add_argument(
        "--n-trials", type=int, default=200,
        help="Number of Optuna trials per horizon (default: 200)",
    )
    tb.add_argument("--no-fetch", action="store_true", help="Skip network data fetches")

    args = parser.parse_args()

    if args.command == "forecast":
        from econometrics.cli.forecast import run_forecast
        run_forecast(args.metro, args.horizon, fetch=not args.no_fetch)
    elif args.command == "backtest":
        from econometrics.cli.backtest import run_backtest_cli
        run_backtest_cli(args.metro, fill_actuals=args.fill_actuals, summary=args.summary)
    elif args.command == "stress":
        from econometrics.cli.stress import run_stress_cli
        run_stress_cli(args.metro, args.scenario, args.horizon)
    elif args.command == "tune-bvar":
        from econometrics.cli.tune import run_tune_bvar_cli
        run_tune_bvar_cli(
            args.metro,
            horizons=args.horizons,
            n_trials=args.n_trials,
            fetch=not args.no_fetch,
        )

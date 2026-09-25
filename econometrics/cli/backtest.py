"""econ backtest — run expanding-window backtest or fill actuals."""

from __future__ import annotations


def run_backtest_cli(metro_id: str, fill_actuals: bool = False, summary: bool = False) -> None:
    """CLI handler for backtest operations."""
    from econometrics.backtest import registry

    if summary:
        acc = registry.accuracy_summary(metro_id)
        if not acc:
            print(f"No completed backtests for {metro_id}")
            return
        print(f"\n═══ Backtest Accuracy — {metro_id} ═══\n")
        for model, stats in acc.items():
            print(f"  {model}:")
            print(f"    RMSE: {stats['mean_rmse']:.4f}")
            print(f"    MAE:  {stats['mean_mae']:.4f}")
            if stats.get("mean_directional_accuracy"):
                print(f"    Dir:  {stats['mean_directional_accuracy']:.0%}")
            print(f"    N:    {stats['n_completed']}")
        return

    if fill_actuals:
        from econometrics.config.settings import load_metro_config
        from econometrics.data.transforms.panel_builder import build_panel

        config = load_metro_config(metro_id)
        panel = build_panel(config, fetch_macro=False, fetch_migration=False)

        actuals = {}
        target = "rent_growth_yoy"
        if target in panel.columns:
            for period in panel.index:
                val = panel.loc[period, target]
                if not __import__("numpy").isnan(val):
                    actuals[str(period)] = float(val)

        n = registry.fill_actuals(metro_id, actuals)
        print(f"Updated {n} registry entries with actuals for {metro_id}")
        return

    print("Use --fill-actuals or --summary. Full backtest run coming in a future release.")

"""econ stress — run a stress test scenario."""

from __future__ import annotations


def run_stress_cli(metro_id: str, scenario_name: str, horizon: int = 8) -> None:
    """CLI handler for stress testing."""
    from econometrics.config.settings import REPORTS_DIR, load_metro_config
    from econometrics.data.transforms.panel_builder import build_panel
    from econometrics.models.var_engine import VAREngine
    from econometrics.reports.chart_builder import scenario_overlay
    from econometrics.scenarios.shock_library import SHOCK_REGISTRY
    from econometrics.scenarios.stress_tester import run_stress_test

    if scenario_name not in SHOCK_REGISTRY:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(SHOCK_REGISTRY.keys())}")
        return

    config = load_metro_config(metro_id)
    print(f"═══ Stress Test: {scenario_name} — {config['display_name']} ═══\n")

    panel = build_panel(config, fetch_macro=False, fetch_migration=False)
    vintage = str(panel.index[-1])

    shock = SHOCK_REGISTRY[scenario_name]
    print(f"Shock: {shock.description}")
    print(f"Duration: {shock.duration_quarters}Q, ramp-in: {shock.ramp_in_quarters}Q\n")

    result = run_stress_test(
        panel, shock, VAREngine,
        horizon=horizon, metro_id=metro_id, vintage_date=vintage,
    )

    print(f"Baseline Q1: {result.baseline.point[0] * 100:.1f}%")
    print(f"Stressed Q1: {result.stressed.point[0] * 100:.1f}%")
    print(f"Delta Q1:    {result.delta_point[0] * 100:+.1f}pp\n")

    # Save chart
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = REPORTS_DIR / f"stress_{metro_id}_{scenario_name}_{vintage}.png"
    scenario_overlay(
        [result],
        title=f"{config['display_name']} — {scenario_name} Stress Test",
        save_path=chart_path,
    )
    print(f"Chart: {chart_path}")
    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"\n═══ Stress test complete ═══")

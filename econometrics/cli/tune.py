"""econ tune-bvar — optimize BVAR Minnesota prior hyperparameters via Optuna."""

from __future__ import annotations


def run_tune_bvar_cli(
    metro_id: str,
    horizons: list[int],
    n_trials: int,
    fetch: bool,
) -> None:
    """Tune BVAR for a metro and save results."""
    from econometrics.config.settings import load_metro_config
    from econometrics.data.transforms.panel_builder import build_panel
    from econometrics.tuning.bvar_tuner import save_tuned_params, tune_bvar

    print(f"═══ BVAR Hyperparameter Tuning — {metro_id} ═══\n")

    config = load_metro_config(metro_id)
    print(f"Metro: {config['display_name']}")

    print("\n1. Building quarterly panel...")
    panel = build_panel(config, fetch_macro=fetch, fetch_migration=fetch)
    print(f"   Panel: {len(panel)} quarters ({panel.index.min()} to {panel.index.max()})")

    # Determine which exog are actually available in the panel
    from econometrics.tuning.bvar_tuner import ALL_EXOG
    available_exog = [c for c in ALL_EXOG if c in panel.columns]
    print(f"   Available exog: {available_exog}")

    print(f"\n2. Running Optuna (horizons={horizons}, n_trials={n_trials} each)...")
    tuned = tune_bvar(
        panel=panel,
        metro_id=metro_id,
        horizons=horizons,
        n_trials=n_trials,
        available_exog=available_exog,
        verbose=True,
    )

    print("\n3. Saving tuned params...")
    out_path = save_tuned_params(metro_id, tuned)
    print(f"   Saved: {out_path}")

    print("\n═══ Tuning complete ═══")
    print("\nSummary:")
    for h, params in tuned.items():
        print(f"  h={h} ({h * 3}mo): RMSE={params['rmse']:.4f}  "
              f"lambda={params['lambda']}  cross_shrink={params['cross_shrink']}  "
              f"ic={params['ic']}  exog={params['exog_cols']}")

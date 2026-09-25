"""Assemble aligned quarterly panel from all data sources.

Merges CoStar supply data, FRED macro series, BLS unemployment,
building permits, IRS migration, and concession spreads into a single
quarterly DataFrame ready for model fitting.

This is the central integration point — every model consumes its output.
"""

from __future__ import annotations

import pandas as pd


class DataQualityError(Exception):
    """Raised when the assembled panel fails quality checks."""


def build_panel(
    config: dict,
    costar_source: str = "csv",
    fetch_macro: bool = True,
    fetch_migration: bool = True,
) -> pd.DataFrame:
    """Build a complete quarterly panel from all configured data sources.

    Args:
        config: Metro YAML config dict.
        costar_source: "csv" (supply-demand data/) or "parquet" (market-study-agent output).
        fetch_macro: If True, fetch FRED + BLS data (requires internet).
        fetch_migration: If True, fetch IRS SOI data (requires internet).

    Returns:
        DataFrame with quarterly PeriodIndex and all endogenous + exogenous columns.

    Raises:
        DataQualityError: If the panel has gaps > 2 quarters or too few observations.
    """
    from econometrics.data.loaders.costar_panel import load_costar
    from econometrics.data.loaders.concessions import compute_concession_features

    # --- 1. Core CoStar panel (endogenous variables) ---
    panel = load_costar(config, source=costar_source)
    panel = compute_concession_features(panel)
    print(f"  CoStar: {len(panel)} quarters ({panel.index.min()} to {panel.index.max()})")

    # --- 2. FRED macro series (exogenous) ---
    if fetch_macro:
        try:
            from econometrics.data.loaders.fred_macro import load_fred_macro, load_bls_unemployment

            macro = load_fred_macro(config)
            if not macro.empty:
                panel = panel.join(macro, how="left")
                print(f"  FRED: {len(macro.columns)} series joined")

            unemployment = load_bls_unemployment(config)
            if not unemployment.empty:
                panel = panel.join(unemployment, how="left")
                print(f"  BLS: unemployment rate joined")
        except Exception as e:
            print(f"  Warning: macro fetch failed ({e}), continuing without")

    # --- 3. Building permits (exogenous, lagged) ---
    # Load-path failures are tolerated (warning + continue), but structural
    # data-quality failures (e.g. all-NaN pipeline from a silent index mismatch)
    # must propagate — they indicate a real bug, not a missing data source.
    try:
        from econometrics.data.loaders.census_permits import load_permits

        permits = load_permits(config)
    except Exception as e:
        print(f"  Warning: permits fetch failed ({e}), continuing without")
        permits = None

    if permits is not None and not permits.empty:
        lag_months = config.get("permit_lag_months", 18)
        lag_quarters = lag_months // 3

        # Align permits index to panel index type — avoid silent reindex-to-NaN
        # when permits has DatetimeIndex and panel has PeriodIndex (or vice versa).
        if isinstance(permits.index, pd.DatetimeIndex) and isinstance(
            panel.index, pd.PeriodIndex
        ):
            permits = permits.copy()
            try:
                permits.index = permits.index.to_period("Q")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot coerce permits DatetimeIndex to quarterly PeriodIndex: {e}. "
                    f"Permits loader must return a regular DatetimeIndex or PeriodIndex."
                ) from e
        elif isinstance(permits.index, pd.PeriodIndex) and isinstance(
            panel.index, pd.DatetimeIndex
        ):
            permits = permits.copy()
            permits.index = permits.index.to_timestamp()

        if isinstance(permits, pd.DataFrame):
            # Census BPS path: DataFrame with per-unit-type columns
            for col in permits.columns:
                # Only expose lagged values — unlagged {col}_raw is a leakage risk
                panel[f"{col}_lagged"] = (
                    permits[col].shift(lag_quarters).reindex(panel.index)
                )
            # Primary pipeline feature = 5+ unit (multifamily)
            if "permits_5plus" in permits.columns:
                panel["permit_pipeline"] = (
                    permits["permits_5plus"].shift(lag_quarters).reindex(panel.index)
                )
            elif "permits_total" in permits.columns:
                panel["permit_pipeline"] = (
                    permits["permits_total"].shift(lag_quarters).reindex(panel.index)
                )
            if panel["permit_pipeline"].notna().sum() == 0:
                raise ValueError(
                    "permit_pipeline is entirely NaN after shift/reindex. "
                    "Index alignment between permits and panel failed."
                )
            print(
                f"  Permits (BPS): {len(permits.columns)} series joined with {lag_months}mo lag"
            )
        else:
            # FRED fallback: single Series
            panel["permit_pipeline"] = (
                permits.shift(lag_quarters).reindex(panel.index)
            )
            if panel["permit_pipeline"].notna().sum() == 0:
                raise ValueError(
                    "permit_pipeline is entirely NaN after shift/reindex."
                )
            print(f"  Permits (FRED): joined with {lag_months}mo lag")

    # --- 3b. Renter median income + affordability ratio ---
    if fetch_macro:
        try:
            from econometrics.data.loaders.census_renter_income import load_renter_median_income

            costar_median = panel.get("median_hh_income")
            renter_income = load_renter_median_income(
                config,
                costar_median_hh=costar_median,
                backfill_ratio=config.get("renter_income_backfill_ratio", 0.67),
            )
            if not renter_income.empty:
                panel = panel.join(renter_income, how="left")
                # Forward-fill quarterly gaps from annual interpolation
                panel["renter_median_hh_income"] = (
                    panel["renter_median_hh_income"].interpolate().ffill().bfill()
                )
                # Derive renter affordability ratio: annual rent / renter income
                if "asking_rent_unit" in panel.columns:
                    annual_rent = panel["asking_rent_unit"] * 12
                    renter_inc = panel["renter_median_hh_income"].replace(0, float("nan"))
                    panel["renter_affordability"] = annual_rent / renter_inc
                    n_valid = panel["renter_affordability"].notna().sum()
                    latest = panel["renter_affordability"].dropna().iloc[-1] if n_valid > 0 else float("nan")
                    print(
                        f"  Renter affordability: {n_valid}Q "
                        f"(latest: {latest:.1%} of income)"
                    )
        except Exception as e:
            print(f"  Warning: renter income fetch failed ({e}), continuing without")

    # --- 4. IRS SOI migration (exogenous, annual → quarterly) ---
    if fetch_migration:
        try:
            from econometrics.data.loaders.irs_migration import load_migration

            migration = load_migration(config)
            if not migration.empty:
                panel = panel.join(migration, how="left")
                print(f"  Migration: {len(migration.columns)} series joined")
        except Exception as e:
            print(f"  Warning: migration fetch failed ({e}), continuing without")

    # --- 5. Derived demand-side variables ---
    if "total_employment" in panel.columns:
        panel["emp_growth_yoy"] = panel["total_employment"].pct_change(4)
    if "households" in panel.columns:
        panel["hh_growth_yoy"] = panel["households"].pct_change(4)
    if "population" in panel.columns:
        panel["pop_growth_yoy"] = panel["population"].pct_change(4)

    # --- 6. Add lagged versions of key features ---
    lag_cols = (
        config.get("endogenous_vars", [])
        + ["concession_spread", "mortgage_rate_30y", "unemployment_rate",
           "starts_pct_inventory", "uc_pct_inventory",
           "emp_growth_yoy", "hh_growth_yoy", "renter_affordability"]
    )
    for col in lag_cols:
        if col in panel.columns:
            panel[f"{col}_lag1"] = panel[col].shift(1)

    # --- 7. Quality checks ---
    _validate_panel(panel, config)

    return panel


def _validate_panel(panel: pd.DataFrame, config: dict) -> None:
    """Check panel quality — raise DataQualityError if critical issues found."""
    # Check for gaps > 2 quarters
    if isinstance(panel.index, pd.PeriodIndex):
        periods = panel.index.sort_values()
        if len(periods) > 1:
            diffs = [(periods[i] - periods[i - 1]).n for i in range(1, len(periods))]
            max_gap = max(diffs) if diffs else 0
            if max_gap > 2:
                raise DataQualityError(
                    f"Panel has a gap of {max_gap} quarters. "
                    f"Maximum allowed: 2. Check CoStar data for missing periods."
                )

    # Check minimum observations
    endogenous = config.get("endogenous_vars", [])
    target = "rent_growth_yoy"
    if target in panel.columns:
        n_valid = panel[target].dropna().shape[0]
        if n_valid < 20:
            print(
                f"  Warning: only {n_valid} non-null observations for {target}. "
                f"Models may have low confidence (minimum recommended: 40)."
            )

    # Report completeness
    total = len(panel)
    for col in endogenous:
        if col in panel.columns:
            n = panel[col].notna().sum()
            pct = n / total * 100
            if pct < 50:
                print(f"  Warning: {col} only {pct:.0f}% complete ({n}/{total} quarters)")

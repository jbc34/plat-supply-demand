"""Load CoStar submarket data into clean quarterly DataFrames.

Reads from the supply-demand project's own CSV files (data/costar/processed/)
or from market-study-agent's parquet output. Produces a standardized quarterly
panel with canonical column names matching the metro YAML endogenous_vars.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Canonical column mapping: CoStar CSV names → econometrics panel names
_COLUMN_MAP = {
    "effective_rent_growth_yoy": "rent_growth_yoy",
    "vacancy_rate": "vacancy_rate",
    "asking_rent_unit": "asking_rent_unit",
    "effective_rent_unit": "effective_rent_unit",
    "inventory_units": "inventory_units",
    "deliveries_12mo": "deliveries_12mo",
    "absorption_units_12mo": "absorption_12mo",
    "under_construction": "under_construction",
    "deliveries": "deliveries",
    "absorption_units": "absorption_units",
    "construction_starts": "construction_starts",
    "occupancy_rate": "occupancy_rate",
    "asking_rent_growth_yoy": "asking_rent_growth_yoy",
    "households": "households",
    "population": "population",
    "total_employment": "total_employment",
    "median_hh_income": "median_hh_income",
}


def load_costar_csv(
    path: str | Path,
    start_date: str | None = None,
) -> pd.DataFrame:
    """Load a CoStar metro CSV into a quarterly panel.

    Args:
        path: Path to the CoStar CSV (e.g., data/costar/processed/dallas_multifamily.csv).
        start_date: Optional start date filter (ISO format, e.g., "2000-01-01").
            Older data often has too many nulls. Default: keep everything.

    Returns:
        DataFrame with quarterly PeriodIndex, canonical column names, and
        derived columns (deliveries_pct_inventory, absorption_pct_inventory).
    """
    df = pd.read_csv(path, parse_dates=["date"])

    # Set quarterly index
    df["period"] = df["date"].dt.to_period("Q")
    df = df.sort_values("period").set_index("period")
    df = df[~df.index.duplicated(keep="last")]

    # Optional date filter
    if start_date:
        cutoff = pd.Period(start_date, freq="Q")
        df = df[df.index >= cutoff]

    # Rename to canonical names
    rename = {k: v for k, v in _COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Derive ratio columns (the endogenous vars for VAR)
    inv = df.get("inventory_units")
    if inv is not None and inv.notna().any():
        inv_safe = inv.replace(0, float("nan"))

        if "deliveries_12mo" in df.columns:
            df["deliveries_pct_inventory"] = df["deliveries_12mo"] / inv_safe

        if "absorption_12mo" in df.columns:
            df["absorption_pct_inventory"] = df["absorption_12mo"] / inv_safe

        if "under_construction" in df.columns:
            df["uc_pct_inventory"] = df["under_construction"] / inv_safe

        if "construction_starts" in df.columns:
            df["starts_pct_inventory"] = df["construction_starts"] / inv_safe

    # Concession spread (for concessions loader, but compute here if both columns exist)
    if "asking_rent_unit" in df.columns and "effective_rent_unit" in df.columns:
        asking = df["asking_rent_unit"].replace(0, float("nan"))
        df["concession_spread"] = (asking - df["effective_rent_unit"]) / asking

    return df


def load_costar_parquet(
    path: str | Path,
    start_date: str | None = None,
) -> pd.DataFrame:
    """Load a pre-built submarket panel parquet from market-study-agent.

    These are the output of etl/supply_model.py build_submarket_panel().
    Column names are already in the expected format.

    Args:
        path: Path to submarket_panel.parquet.
        start_date: Optional start date filter.

    Returns:
        DataFrame with quarterly PeriodIndex.
    """
    df = pd.read_parquet(path)

    # The parquet may have a PeriodIndex already or a date column
    if not isinstance(df.index, pd.PeriodIndex):
        if "date" in df.columns:
            df["period"] = pd.to_datetime(df["date"]).dt.to_period("Q")
            df = df.set_index("period")
        elif "period_q" in df.columns:
            df = df.set_index("period_q")

    df = df.sort_index()

    if start_date:
        cutoff = pd.Period(start_date, freq="Q")
        df = df[df.index >= cutoff]

    # Ensure canonical names exist
    rename = {k: v for k, v in _COLUMN_MAP.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=rename)

    return df


def load_costar(config: dict, source: str = "csv") -> pd.DataFrame:
    """Load CoStar data using metro config.

    Args:
        config: Metro YAML config dict.
        source: "csv" to read from supply-demand data/costar/processed/,
                "parquet" to read from market-study-agent output.

    Returns:
        Clean quarterly panel.
    """
    from econometrics.config.settings import COSTAR_PANEL_PATH, PROJECT_ROOT

    if source == "csv":
        market_key = config.get("costar_market_key", "")
        csv_path = PROJECT_ROOT / "data" / "costar" / "processed" / f"{market_key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CoStar CSV not found: {csv_path}")
        return load_costar_csv(csv_path, start_date="2000-01-01")

    elif source == "parquet":
        # Find the submarket panel parquet in market-study-agent output
        metro_slug = config.get("metro_id", "")
        parquet_path = COSTAR_PANEL_PATH / metro_slug / "submarket_panel.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"CoStar parquet not found: {parquet_path}")
        return load_costar_parquet(parquet_path)

    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'csv' or 'parquet'.")

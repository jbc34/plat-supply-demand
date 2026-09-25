"""Load IRS Statistics of Income (SOI) county-to-county migration data.

Produces net household migration and income quality features from the
IRS SOI county-level inflow/outflow files. Annual data interpolated
to quarterly for panel alignment.

Data source: https://www.irs.gov/statistics/soi-tax-stats-migration-data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests


# IRS SOI download URLs (YY1 = filing year start, YY2 = filing year end)
INFLOW_URL = "https://www.irs.gov/pub/irs-soi/countyinflow{yy1}{yy2}.csv"
OUTFLOW_URL = "https://www.irs.gov/pub/irs-soi/countyoutflow{yy1}{yy2}.csv"

# Cache directory
_CACHE_DIR = Path.home() / ".cache" / "econometrics" / "soi_migration"


def _download_soi(url: str, cache_path: Path) -> pd.DataFrame:
    """Download and cache an IRS SOI CSV."""
    if cache_path.exists():
        return pd.read_csv(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return pd.read_csv(cache_path)


def _county_fips_5digit(state_fips: str, county_fips: str) -> str:
    """Build 5-digit FIPS from state + county."""
    return f"{int(state_fips):02d}{int(county_fips):03d}"


def fetch_soi_year(
    state_fips: str,
    county_fips_list: list[str],
    year: int,
) -> dict:
    """Fetch SOI migration data for a set of counties in a given year.

    Args:
        state_fips: 2-digit state FIPS.
        county_fips_list: List of 3-digit county FIPS codes.
        year: Calendar year (e.g., 2022 → uses 2122 filing pair).

    Returns:
        Dict with:
          - net_returns: int (inflow - outflow tax returns)
          - in_avg_agi: float (avg AGI of arrivals, $thousands)
          - out_avg_agi: float (avg AGI of departures, $thousands)
          - income_delta: float (in_avg_agi - out_avg_agi)
          - income_delta_pct: float (income_delta / out_avg_agi)
    """
    # SOI naming: countyinflow2122.csv for 2021→2022 filing year
    yy1 = f"{(year - 1) % 100:02d}"
    yy2 = f"{year % 100:02d}"

    target_fips = {_county_fips_5digit(state_fips, c) for c in county_fips_list}

    # Inflow: people moving INTO target counties
    inflow_url = INFLOW_URL.format(yy1=yy1, yy2=yy2)
    cache_in = _CACHE_DIR / f"inflow_{yy1}{yy2}.csv"
    try:
        df_in = _download_soi(inflow_url, cache_in)
    except Exception:
        return {"net_returns": 0, "in_avg_agi": 0, "out_avg_agi": 0,
                "income_delta": 0, "income_delta_pct": 0}

    # Normalize column names (varies by year)
    df_in.columns = [c.strip().lower() for c in df_in.columns]

    # Filter to target counties as destination
    # SOI format: y2_statefips + y2_countyfips = destination
    if "y2_statefips" in df_in.columns and "y2_countyfips" in df_in.columns:
        df_in["dest_fips"] = (
            df_in["y2_statefips"].astype(str).str.zfill(2)
            + df_in["y2_countyfips"].astype(str).str.zfill(3)
        )
        inflow = df_in[df_in["dest_fips"].isin(target_fips)]
    else:
        inflow = pd.DataFrame()

    # Outflow: people moving OUT of target counties
    outflow_url = OUTFLOW_URL.format(yy1=yy1, yy2=yy2)
    cache_out = _CACHE_DIR / f"outflow_{yy1}{yy2}.csv"
    try:
        df_out = _download_soi(outflow_url, cache_out)
    except Exception:
        return {"net_returns": 0, "in_avg_agi": 0, "out_avg_agi": 0,
                "income_delta": 0, "income_delta_pct": 0}

    df_out.columns = [c.strip().lower() for c in df_out.columns]

    if "y1_statefips" in df_out.columns and "y1_countyfips" in df_out.columns:
        df_out["origin_fips"] = (
            df_out["y1_statefips"].astype(str).str.zfill(2)
            + df_out["y1_countyfips"].astype(str).str.zfill(3)
        )
        outflow = df_out[df_out["origin_fips"].isin(target_fips)]
    else:
        outflow = pd.DataFrame()

    # Aggregate
    n1_col = "n1" if "n1" in inflow.columns else "returns"
    agi_col = "agi" if "agi" in inflow.columns else "adjusted_gross_income"

    in_returns = inflow[n1_col].sum() if n1_col in inflow.columns else 0
    out_returns = outflow[n1_col].sum() if n1_col in outflow.columns else 0
    in_agi = inflow[agi_col].sum() if agi_col in inflow.columns else 0
    out_agi = outflow[agi_col].sum() if agi_col in outflow.columns else 0

    in_avg_agi = in_agi / in_returns if in_returns > 0 else 0
    out_avg_agi = out_agi / out_returns if out_returns > 0 else 0

    return {
        "net_returns": int(in_returns - out_returns),
        "in_avg_agi": float(in_avg_agi),
        "out_avg_agi": float(out_avg_agi),
        "income_delta": float(in_avg_agi - out_avg_agi),
        "income_delta_pct": float(
            (in_avg_agi - out_avg_agi) / out_avg_agi if out_avg_agi > 0 else 0
        ),
    }


def load_migration(
    config: dict,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Load IRS SOI migration data for the metro, interpolated to quarterly.

    Args:
        config: Metro YAML config dict.
        years: Years to fetch. Default: 2018-2023.

    Returns:
        DataFrame with quarterly PeriodIndex and columns:
        net_migration_returns, income_delta_pct.
    """
    state_fips = config.get("state_fips", "")
    counties = config.get("migration_counties", config.get("counties", []))
    interpolation = config.get("migration_interpolation", "linear")

    if years is None:
        years = list(range(2018, 2024))

    rows = []
    for year in years:
        try:
            result = fetch_soi_year(state_fips, counties, year)
            result["year"] = year
            rows.append(result)
        except Exception as e:
            print(f"  Warning: SOI {year} failed: {e}")

    if not rows:
        return pd.DataFrame()

    annual = pd.DataFrame(rows).set_index("year")

    # Convert annual to quarterly by assigning to Q1, then interpolating
    quarterly_index = pd.period_range(
        f"{annual.index.min()}Q1", f"{annual.index.max()}Q4", freq="Q"
    )
    quarterly = pd.DataFrame(index=quarterly_index)

    for col in ["net_migration_returns", "income_delta_pct"]:
        if col not in annual.columns:
            # Map from the dict keys
            if col == "net_migration_returns":
                annual[col] = annual["net_returns"]
            elif col == "income_delta_pct":
                pass  # Already exists

        # Place annual values at Q1 of each year
        for year in annual.index:
            period = pd.Period(f"{year}Q1")
            if period in quarterly.index:
                quarterly.loc[period, col] = annual.loc[year, col]

    # Interpolate
    if interpolation == "linear":
        quarterly = quarterly.astype(float).interpolate(method="linear")
    else:
        quarterly = quarterly.astype(float).ffill()

    return quarterly

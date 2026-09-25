"""Load FRED macroeconomic series for the interest rate channel.

Fetches mortgage rates, treasury yields, fed funds, and credit spreads
from the FRED API, resamples to quarterly frequency, and produces
derived features (YoY changes, spread differentials).

Uses the supply_demand FREDClient if available; falls back to direct
requests if not.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests


FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fetch_fred_series(series_id: str, api_key: str = "") -> pd.Series:
    """Fetch a single FRED series as a daily pandas Series.

    Uses the CSV export endpoint (no API key required for public series).
    """
    url = f"{FRED_BASE_URL}?id={series_id}"
    df = pd.read_csv(url, na_values=".")
    # FRED CSV uses either "DATE" or "observation_date" as the date column
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", series_id: "value"})
    df = df.dropna(subset=["value"]).set_index("date")["value"]
    df.name = series_id
    return df


def _resample_to_quarterly(series: pd.Series, method: str = "mean") -> pd.Series:
    """Resample a daily/monthly series to quarterly frequency."""
    quarterly = series.resample("QS").agg(method)
    quarterly.index = quarterly.index.to_period("Q")
    return quarterly


def load_fred_macro(config: dict) -> pd.DataFrame:
    """Load all FRED macro series specified in the metro config.

    Args:
        config: Metro YAML config dict with fred_series mapping.

    Returns:
        DataFrame with quarterly PeriodIndex and one column per series.
    """
    fred_series = config.get("fred_series", {})
    if not fred_series:
        return pd.DataFrame()

    frames = {}

    for name, series_id in fred_series.items():
        try:
            raw = _fetch_fred_series(series_id)
            quarterly = _resample_to_quarterly(raw)
            frames[name] = quarterly
        except Exception as e:
            print(f"  Warning: FRED {series_id} ({name}) failed: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)

    # Derive features
    if "mortgage_rate" in df.columns:
        df["mortgage_rate_30y"] = df["mortgage_rate"]
        df["mortgage_rate_chg_yoy"] = df["mortgage_rate"].diff(4)

    if "term_spread" in df.columns:
        df["term_spread_10y2y"] = df["term_spread"]

    if "baa_spread" in df.columns:
        df["credit_spread_baa"] = df["baa_spread"]

    return df


def load_bls_unemployment(config: dict) -> pd.Series:
    """Load MSA unemployment rate from BLS LAUS API.

    Args:
        config: Metro YAML config dict with bls_msa_code.

    Returns:
        Quarterly Series of unemployment rate (%).
    """
    msa_code = config.get("bls_msa_code", "")
    state_fips = config.get("state_fips", "")
    if not msa_code:
        return pd.Series(dtype=float, name="unemployment_rate")

    # BLS LAUS MSA series format: LAUMT + state(2) + cbsa(5) + 00000003
    # Total 20 chars: LAUMT(5) + state(2) + cbsa(5) + suffix(8)
    # Suffix = 7 zeros + measure digit (3 = unemployment rate)
    # Example: LAUMT481910000000003 for Dallas
    cbsa = msa_code.lstrip("0").zfill(5)
    series_id = f"LAUMT{state_fips}{cbsa}00000003"

    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    current_year = datetime.now().year
    # Fetch in a single request (BLS allows 20-year range)
    payload = {
        "seriesid": [series_id],
        "startyear": str(current_year - 19),
        "endyear": str(current_year),
    }

    bls_key = os.environ.get("BLS_API_KEY", "")
    if bls_key:
        payload["registrationkey"] = bls_key

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        print(f"  Warning: BLS API returned status {data.get('status')}")
        return pd.Series(dtype=float, name="unemployment_rate")

    series_data = data["Results"]["series"][0]["data"]

    rows = []
    for obs in series_data:
        year = int(obs["year"])
        period = obs["period"]  # "M01" through "M12"
        if not period.startswith("M"):
            continue
        month = int(period[1:])
        value = float(obs["value"])
        rows.append({"date": pd.Timestamp(year=year, month=month, day=1), "value": value})

    if not rows:
        return pd.Series(dtype=float, name="unemployment_rate")

    monthly = pd.DataFrame(rows).set_index("date")["value"].sort_index()
    quarterly = monthly.resample("QS").mean()
    quarterly.index = quarterly.index.to_period("Q")
    quarterly.name = "unemployment_rate"
    return quarterly

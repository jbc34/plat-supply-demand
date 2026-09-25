"""Load Census Bureau Building Permits Survey (BPS) data.

Downloads monthly XLS files from census.gov, parses permit counts by
structure type (1-unit, 2-unit, 3-4 unit, 5+ unit) for a specific
CBSA/MSA, and resamples to quarterly.

Data source: https://www.census.gov/construction/bps/
No API key required — public XLS downloads.

Falls back to FRED for metros that have pre-built permit series.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# BPS XLS download URLs (monthly, by CBSA)
# 2024+: cbsamonthly_YYYYMM.xls
# Pre-2024: msamonthly_YYYYMM.xls
BPS_BASE = "https://www.census.gov/construction/bps/xls"

# FRED fallback
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Cache directory
_CACHE_DIR = Path.home() / ".cache" / "econometrics" / "bps"

# Column indices in the BPS XLS (row 7 = header, data starts row 8)
# Current month columns:
_COL_CBSA = 1
_COL_NAME = 2
_COL_MONTH_TOTAL = 4
_COL_MONTH_1UNIT = 5
_COL_MONTH_2UNIT = 6
_COL_MONTH_34UNIT = 7
_COL_MONTH_5PLUS = 8
# YTD columns:
_COL_YTD_TOTAL = 11
_COL_YTD_1UNIT = 12
_COL_YTD_2UNIT = 13
_COL_YTD_34UNIT = 14
_COL_YTD_5PLUS = 15

_DATA_START_ROW = 8


def _download_bps_xls(year: int, month: int) -> bytes | None:
    """Download a monthly BPS XLS file, with caching."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"bps_{year}{month:02d}.xls"

    if cache_file.exists():
        return cache_file.read_bytes()

    # 2024+: cbsamonthly, pre-2024: msamonthly
    prefix = "cbsamonthly" if year >= 2024 else "msamonthly"
    url = f"{BPS_BASE}/{prefix}_{year}{month:02d}.xls"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            cache_file.write_bytes(resp.content)
            return resp.content
    except Exception:
        pass

    return None


def _parse_bps_xls(content: bytes, cbsa_code: str) -> dict | None:
    """Parse a BPS XLS file and extract data for a specific CBSA.

    Returns dict with permit counts by structure type, or None if CBSA not found.
    """
    try:
        import xlrd
    except ImportError:
        raise ImportError("xlrd is required for BPS XLS parsing. Run: pip install xlrd")

    wb = xlrd.open_workbook(file_contents=content)
    sheet = wb.sheet_by_index(0)  # "MSA Units" sheet

    target = str(int(float(cbsa_code)))  # Normalize "19100" format

    for row in range(_DATA_START_ROW, sheet.nrows):
        cell_cbsa = str(sheet.cell_value(row, _COL_CBSA)).strip()
        # CBSA comes as float in XLS (e.g., 19100.0)
        try:
            cell_cbsa = str(int(float(cell_cbsa)))
        except (ValueError, OverflowError):
            continue

        if cell_cbsa == target:
            def _val(col):
                try:
                    v = sheet.cell_value(row, col)
                    return int(float(v)) if v != "" else 0
                except (ValueError, IndexError):
                    return 0

            return {
                "name": str(sheet.cell_value(row, _COL_NAME)).strip(),
                "total": _val(_COL_MONTH_TOTAL),
                "units_1": _val(_COL_MONTH_1UNIT),
                "units_2": _val(_COL_MONTH_2UNIT),
                "units_3_4": _val(_COL_MONTH_34UNIT),
                "units_5_plus": _val(_COL_MONTH_5PLUS),
                "ytd_total": _val(_COL_YTD_TOTAL),
                "ytd_1": _val(_COL_YTD_1UNIT),
                "ytd_2": _val(_COL_YTD_2UNIT),
                "ytd_3_4": _val(_COL_YTD_34UNIT),
                "ytd_5_plus": _val(_COL_YTD_5PLUS),
            }

    return None


def fetch_bps_monthly(
    cbsa_code: str,
    start_year: int = 2004,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Fetch monthly BPS permit data for a CBSA across multiple years.

    Args:
        cbsa_code: 5-digit CBSA code (e.g., "19100" for Dallas).
        start_year: First year to fetch.
        end_year: Last year to fetch (inclusive).

    Returns:
        DataFrame with monthly DatetimeIndex and columns:
        permits_total, permits_1unit, permits_2unit, permits_3_4unit, permits_5plus
    """
    rows = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            content = _download_bps_xls(year, month)
            if content is None:
                continue

            data = _parse_bps_xls(content, cbsa_code)
            if data is None:
                continue

            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "permits_total": data["total"],
                "permits_1unit": data["units_1"],
                "permits_2unit": data["units_2"],
                "permits_3_4unit": data["units_3_4"],
                "permits_5plus": data["units_5_plus"],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def load_bps_quarterly(
    cbsa_code: str,
    start_year: int = 2004,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Fetch BPS data and resample to quarterly.

    Args:
        cbsa_code: 5-digit CBSA code.
        start_year: First year.
        end_year: Last year.

    Returns:
        DataFrame with quarterly PeriodIndex and all permit columns.
    """
    monthly = fetch_bps_monthly(cbsa_code, start_year, end_year)
    if monthly.empty:
        return pd.DataFrame()

    quarterly = monthly.resample("QS").sum()
    quarterly.index = quarterly.index.to_period("Q")
    return quarterly


def load_permits_from_fred(
    series_id: str,
    start_date: str = "2000-01-01",
) -> pd.Series:
    """Load building permit series from FRED (fallback).

    Args:
        series_id: FRED series ID (e.g., "DALL148BPPRIVSA").
        start_date: Start date filter.

    Returns:
        Quarterly Series of permit counts.
    """
    url = f"{FRED_URL}?id={series_id}"
    df = pd.read_csv(url, na_values=".")
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", series_id: "value"})
    df = df.dropna(subset=["value"]).set_index("date")

    if start_date:
        df = df[df.index >= start_date]

    quarterly = df["value"].resample("QS").mean()
    quarterly.index = quarterly.index.to_period("Q")
    quarterly.name = "permits_total"
    return quarterly


def load_permits(config: dict) -> pd.DataFrame | pd.Series:
    """Load building permit data using metro config.

    Tries Census BPS first (if permit_cbsa is configured), falls back
    to FRED if a fred_series.permits is set.

    Args:
        config: Metro YAML config dict.

    Returns:
        DataFrame with quarterly permit columns (BPS path) or
        Series with total permits (FRED fallback).
    """
    cbsa = config.get("permit_cbsa", "")

    # Try Census BPS first
    if cbsa:
        try:
            df = load_bps_quarterly(cbsa, start_year=2004, end_year=2025)
            if not df.empty:
                print(f"  Census BPS: {len(df)}Q of permits for CBSA {cbsa}")
                return df
        except Exception as e:
            print(f"  Warning: Census BPS failed ({e})")

    # FRED fallback
    fred_series = config.get("fred_series", {})
    permit_series = fred_series.get("permits")

    if permit_series:
        try:
            series = load_permits_from_fred(permit_series)
            print(f"  FRED permits: {len(series)}Q from {permit_series}")
            return series
        except Exception as e:
            print(f"  Warning: FRED permits failed ({e})")

    print("  Warning: No permit data available")
    return pd.Series(dtype=float, name="permits_total")

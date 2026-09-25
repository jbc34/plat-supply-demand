"""Load renter-specific median household income from Census ACS 1-Year.

Uses B25119_003E (median HH income for renter-occupied units) at the
MSA/CBSA level. Available annually for large MSAs from 2010 onward.

Pre-ACS years are backfilled from CoStar's blended median_hh_income
using a configurable renter-to-blended ratio (default 0.50), based on
the empirical observation that DFW renter median is ~47-53% of owner
median across the ACS era.

Interpolated from annual to quarterly via linear interpolation to match
the CoStar panel frequency.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def load_renter_median_income(
    config: dict,
    costar_median_hh: pd.Series | None = None,
    backfill_ratio: float = 0.67,
    start_year: int = 2010,
    end_year: int = 2023,
) -> pd.Series:
    """Load renter median HH income as a quarterly Series with PeriodIndex.

    Args:
        config: Metro YAML config dict. Must contain 'msa_code' (CBSA code).
        costar_median_hh: Optional quarterly Series of CoStar blended
            median_hh_income for backfilling pre-ACS years. If provided,
            years before start_year are filled with costar × backfill_ratio.
        backfill_ratio: Fraction of CoStar blended (all-household) median to
            use as renter proxy for pre-ACS years. Default 0.67 based on
            DFW empirical ratio of renter/overall median across 2010-2023
            (renter share stable at ~40%, ratio stable at 63-70%).
        start_year: First ACS 1-year vintage to pull.
        end_year: Last ACS 1-year vintage to pull.

    Returns:
        Quarterly Series named 'renter_median_hh_income' with PeriodIndex.
        NaN where neither ACS nor backfill is available.
    """
    cbsa = config.get("msa_code", "")
    if not cbsa:
        return pd.Series(dtype=float, name="renter_median_hh_income")

    # Normalize CBSA to 5-digit string
    cbsa = str(cbsa).lstrip("0").zfill(5)

    api_key = os.environ.get("CENSUS_API_KEY", "")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("CENSUS_API_KEY", "")

    # --- Pull ACS 1-year B25119_003E ---
    annual_data: dict[int, float] = {}

    if api_key:
        for year in range(start_year, end_year + 1):
            # 2020 ACS 1-year was not released (COVID)
            if year == 2020:
                continue
            url = f"https://api.census.gov/data/{year}/acs/acs1"
            params = {
                "get": "B25119_003E",
                "for": f"metropolitan statistical area/micropolitan statistical area:{cbsa}",
                "key": api_key,
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if len(data) > 1 and data[1][0] is not None:
                        val = float(data[1][0])
                        if val > 0:
                            annual_data[year] = val
            except Exception:
                continue

    if annual_data:
        print(
            f"  ACS renter income: {len(annual_data)} years "
            f"({min(annual_data)}-{max(annual_data)}) for CBSA {cbsa}"
        )
    else:
        print(f"  Warning: no ACS renter income data fetched for CBSA {cbsa}")

    # --- Build annual Series and interpolate to quarterly ---
    # Place annual values at Q3 (midpoint of the ACS reference year)
    annual_index = [pd.Period(f"{yr}Q3") for yr in sorted(annual_data.keys())]
    annual_series = pd.Series(
        [annual_data[yr] for yr in sorted(annual_data.keys())],
        index=annual_index,
        name="renter_median_hh_income",
    )

    # --- Backfill pre-ACS years from CoStar blended median ---
    if costar_median_hh is not None and not costar_median_hh.empty:
        # Compute actual renter-to-blended ratio from overlap period
        # (for informational logging only — we use the user-specified backfill_ratio)
        overlap_ratios = []
        for yr, renter_val in annual_data.items():
            q3 = pd.Period(f"{yr}Q3")
            if q3 in costar_median_hh.index:
                blended = costar_median_hh.loc[q3]
                if blended > 0:
                    overlap_ratios.append(renter_val / blended)
        if overlap_ratios:
            actual_ratio = np.mean(overlap_ratios)
            print(
                f"  Renter/blended ratio (ACS overlap): {actual_ratio:.3f} "
                f"(using {backfill_ratio:.2f} for backfill)"
            )

        # Build backfill for CoStar quarters before ACS coverage
        min_acs_period = min(annual_index) if annual_index else None
        if min_acs_period:
            pre_acs = costar_median_hh[costar_median_hh.index < min_acs_period]
        else:
            pre_acs = costar_median_hh

        backfill = (pre_acs * backfill_ratio).rename("renter_median_hh_income")

        # Combine: backfill + ACS actuals
        combined = pd.concat([backfill, annual_series]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = annual_series

    # Interpolate to quarterly
    if combined.empty:
        return pd.Series(dtype=float, name="renter_median_hh_income")

    # Create full quarterly index spanning the data range
    full_idx = pd.period_range(combined.index.min(), combined.index.max(), freq="Q")
    quarterly = combined.reindex(full_idx).interpolate(method="linear")
    quarterly.name = "renter_median_hh_income"

    return quarterly

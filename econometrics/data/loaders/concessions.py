"""Compute concession spread from CoStar panel data.

The concession spread (asking - effective rent gap) is a leading indicator
of market softening. When concessions widen, it signals demand weakness
1-2 quarters before vacancy rises or rent growth turns negative.

This is a derived transform, not a data fetch — it operates on columns
already present in the CoStar panel.
"""

from __future__ import annotations

import pandas as pd


def compute_concession_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add concession spread features to a CoStar panel.

    Args:
        panel: DataFrame with 'asking_rent_unit' and 'effective_rent_unit' columns.

    Returns:
        DataFrame with added columns:
        - concession_spread: (asking - effective) / asking (0 = no concessions)
        - concession_spread_chg_yoy: YoY change in spread (widening = positive)
    """
    result = panel.copy()

    if "asking_rent_unit" not in result.columns or "effective_rent_unit" not in result.columns:
        return result

    asking = result["asking_rent_unit"].replace(0, float("nan"))
    effective = result["effective_rent_unit"]

    result["concession_spread"] = (asking - effective) / asking
    result["concession_spread_chg_yoy"] = result["concession_spread"].diff(4)

    return result

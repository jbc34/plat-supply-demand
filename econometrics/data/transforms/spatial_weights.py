"""Build spatial weights matrix (W) from submarket coordinates.

Used by the spatial lag/error models to capture geographic spillover
effects between adjacent submarkets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_inverse_distance_weights(
    submarkets: list[dict],
    power: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Build a row-normalized inverse-distance spatial weights matrix.

    Args:
        submarkets: List of dicts with 'id', 'lat', 'lon' keys.
        power: Distance decay power (1.0 = linear, 2.0 = squared).

    Returns:
        Tuple of (W matrix as ndarray, list of submarket IDs in row/col order).
    """
    n = len(submarkets)
    if n < 2:
        raise ValueError(f"Need at least 2 submarkets for spatial weights, got {n}")

    ids = [s["id"] for s in submarkets]
    lats = np.array([s["lat"] for s in submarkets])
    lons = np.array([s["lon"] for s in submarkets])

    # Haversine distance in km (approximate, sufficient for submarket scale)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dlat = np.radians(lats[j] - lats[i])
            dlon = np.radians(lons[j] - lons[i])
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lats[i]))
                * np.cos(np.radians(lats[j]))
                * np.sin(dlon / 2) ** 2
            )
            dist_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            W[i, j] = 1.0 / max(dist_km, 0.1) ** power

    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Prevent division by zero
    W = W / row_sums

    return W, ids


def build_weights_from_config(config: dict) -> tuple[np.ndarray, list[str]]:
    """Build spatial weights matrix from metro YAML config.

    Args:
        config: Metro YAML config dict with 'submarkets' and 'spatial_contiguity'.

    Returns:
        Tuple of (W matrix, submarket IDs).
    """
    submarkets = config.get("submarkets", [])
    method = config.get("spatial_contiguity", "inverse_distance")

    if method == "inverse_distance":
        return build_inverse_distance_weights(submarkets)
    else:
        # Default to inverse distance; queen contiguity would require polygon data
        print(f"  Warning: spatial_contiguity='{method}' not implemented, using inverse_distance")
        return build_inverse_distance_weights(submarkets)


def spatial_lag(
    W: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute the spatial lag Wy.

    Args:
        W: Row-normalized spatial weights matrix (n x n).
        y: Variable values for each submarket (n,).

    Returns:
        Spatial lag array (n,) — the weighted average of neighbors' values.
    """
    return W @ y

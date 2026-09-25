"""Append-only JSON forecast registry.

Every forecast is logged with a UUID. Actuals are filled in retroactively
when new CoStar data arrives. Nothing is ever deleted.

Storage: one JSON file per metro at data/econometrics/registry/{metro_id}_backtest.json.
Each file contains a JSON array of forecast entries.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from econometrics.backtest.metrics import ci_coverage, directional_accuracy, mae, rmse
from econometrics.models.base import ForecastResult


def _registry_path(metro_id: str) -> Path:
    """Get the registry file path for a metro."""
    from econometrics.config.settings import REGISTRY_DIR

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    return REGISTRY_DIR / f"{metro_id}_backtest.json"


def _serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _load_entries(path: Path) -> list[dict]:
    """Load all entries from a registry file."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_entries(path: Path, entries: list[dict]) -> None:
    """Atomic write: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2, default=_serialize)
    os.replace(str(tmp), str(path))


def append(result: ForecastResult) -> str:
    """Append a new forecast to the registry.

    Args:
        result: ForecastResult from any model.

    Returns:
        The run_id (UUID) of the new entry.
    """
    run_id = str(uuid.uuid4())

    entry = {
        "run_id": run_id,
        "metro_id": result.metro_id,
        "vintage_date": result.vintage_date,
        "model_name": result.model_name,
        "horizon_quarters": result.horizon_quarters,
        "forecast_periods": result.forecast_periods,
        "point_forecast": result.point.tolist(),
        "ci_80_lo": result.ci_80_lo.tolist(),
        "ci_80_hi": result.ci_80_hi.tolist(),
        "ci_95_lo": result.ci_95_lo.tolist(),
        "ci_95_hi": result.ci_95_hi.tolist(),
        "actuals": [None] * result.horizon_quarters,
        "rmse": None,
        "mae": None,
        "directional_accuracy": None,
        "ci_80_coverage": None,
        "ci_95_coverage": None,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = _registry_path(result.metro_id)
    entries = _load_entries(path)
    entries.append(entry)
    _save_entries(path, entries)

    return run_id


def fill_actuals(metro_id: str, actuals_by_period: dict[str, float]) -> int:
    """Fill in actual values for past forecast periods.

    Args:
        metro_id: Metro identifier.
        actuals_by_period: Dict mapping period string (e.g., "2025Q1") to
            actual rent_growth_yoy value.

    Returns:
        Number of entries updated.
    """
    path = _registry_path(metro_id)
    entries = _load_entries(path)
    updated = 0

    for entry in entries:
        changed = False
        for i, period in enumerate(entry["forecast_periods"]):
            if entry["actuals"][i] is None and period in actuals_by_period:
                entry["actuals"][i] = actuals_by_period[period]
                changed = True

        if changed:
            # Recompute metrics if all actuals are filled
            actuals_arr = np.array(entry["actuals"], dtype=float)
            forecast_arr = np.array(entry["point_forecast"], dtype=float)

            if not np.any(np.isnan(actuals_arr)):
                entry["rmse"] = rmse(actuals_arr, forecast_arr)
                entry["mae"] = mae(actuals_arr, forecast_arr)
                entry["directional_accuracy"] = directional_accuracy(actuals_arr, forecast_arr)
                entry["ci_80_coverage"] = ci_coverage(
                    actuals_arr,
                    np.array(entry["ci_80_lo"]),
                    np.array(entry["ci_80_hi"]),
                )
                entry["ci_95_coverage"] = ci_coverage(
                    actuals_arr,
                    np.array(entry["ci_95_lo"]),
                    np.array(entry["ci_95_hi"]),
                )

            updated += 1

    if updated > 0:
        _save_entries(path, entries)

    return updated


def load_all(metro_id: str) -> list[dict]:
    """Load all entries for a metro."""
    return _load_entries(_registry_path(metro_id))


def load_completed(metro_id: str, min_quarters: int = 4) -> list[dict]:
    """Load entries where all actuals are filled and metrics computed.

    Args:
        metro_id: Metro identifier.
        min_quarters: Minimum forecast horizon to include.

    Returns:
        List of completed entries with RMSE/MAE populated.
    """
    entries = load_all(metro_id)
    return [
        e for e in entries
        if e.get("rmse") is not None
        and e.get("horizon_quarters", 0) >= min_quarters
    ]


def accuracy_summary(metro_id: str) -> dict[str, dict]:
    """Summary of accuracy by model.

    Returns:
        Dict mapping model_name → {mean_rmse, mean_mae, mean_dir_acc,
        mean_ci80_cov, mean_ci95_cov, n_completed}.
    """
    completed = load_completed(metro_id)
    by_model: dict[str, list[dict]] = {}

    for e in completed:
        model = e["model_name"]
        by_model.setdefault(model, []).append(e)

    summary = {}
    for model, entries in by_model.items():
        summary[model] = {
            "mean_rmse": float(np.mean([e["rmse"] for e in entries])),
            "mean_mae": float(np.mean([e["mae"] for e in entries])),
            "mean_directional_accuracy": float(
                np.mean([e["directional_accuracy"] for e in entries if e["directional_accuracy"] is not None])
            ) if any(e["directional_accuracy"] is not None for e in entries) else None,
            "mean_ci80_coverage": float(
                np.mean([e["ci_80_coverage"] for e in entries if e["ci_80_coverage"] is not None])
            ) if any(e["ci_80_coverage"] is not None for e in entries) else None,
            "mean_ci95_coverage": float(
                np.mean([e["ci_95_coverage"] for e in entries if e["ci_95_coverage"] is not None])
            ) if any(e["ci_95_coverage"] is not None for e in entries) else None,
            "n_completed": len(entries),
        }

    return summary

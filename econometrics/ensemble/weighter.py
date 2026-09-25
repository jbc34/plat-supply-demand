"""Inverse-RMSE ensemble weighting from backtest registry.

Models with better backtest accuracy get higher weight. Models with
fewer than min_backtest_quarters completed backtests get equal weight
(cold-start protection).
"""

from __future__ import annotations

import numpy as np


def compute_weights(
    accuracy_by_model: dict[str, dict],
    enabled_models: list[str],
    min_completed: int = 4,
    decay: float = 1.0,
) -> dict[str, float]:
    """Compute ensemble weights from backtest accuracy.

    Args:
        accuracy_by_model: From registry.accuracy_summary(). Dict mapping
            model_name → {mean_rmse, n_completed, ...}.
        enabled_models: List of model names enabled in the metro config.
        min_completed: Minimum completed backtests before using inverse-RMSE.
        decay: Exponential decay factor (1.0 = equal weight to all vintages).

    Returns:
        Dict mapping model_name → weight (sums to 1.0).
    """
    if not enabled_models:
        return {}

    n = len(enabled_models)

    # Check which models have enough backtest history
    ready_models = {}
    cold_models = []

    for model in enabled_models:
        info = accuracy_by_model.get(model, {})
        n_completed = info.get("n_completed", 0)
        mean_rmse = info.get("mean_rmse")

        if n_completed >= min_completed and mean_rmse is not None and mean_rmse > 0:
            ready_models[model] = mean_rmse
        else:
            cold_models.append(model)

    # If no model is ready, equal weights for all
    if not ready_models:
        return {m: 1.0 / n for m in enabled_models}

    # Inverse-RMSE weighting for ready models
    inv_rmse = {m: 1.0 / r for m, r in ready_models.items()}
    total_inv = sum(inv_rmse.values())

    # Cold-start models get the average weight of ready models
    ready_weight_share = len(ready_models) / n
    cold_weight_share = len(cold_models) / n if cold_models else 0

    weights = {}

    if cold_models:
        # Give cold models equal share, redistribute
        cold_each = (1.0 / n)  # Equal weight per cold model
        cold_total = cold_each * len(cold_models)
        ready_total = 1.0 - cold_total

        for m in cold_models:
            weights[m] = cold_each

        for m, inv in inv_rmse.items():
            weights[m] = ready_total * (inv / total_inv)
    else:
        for m, inv in inv_rmse.items():
            weights[m] = inv / total_inv

    return weights

"""Unit tests for change-in-level directional accuracy."""

from __future__ import annotations

import numpy as np

from econometrics.backtest.metrics import directional_accuracy


def test_directional_accuracy_rejects_flat_positive_forecast():
    """A flat positive forecast against positive-growth actuals must NOT score ~1.0."""
    actuals = np.array([0.02, 0.03, 0.025, 0.04, 0.035, 0.045, 0.05, 0.048])
    forecasts = np.full(8, 0.01)

    score = directional_accuracy(actuals, forecasts)
    assert score <= 0.6, f"Flat forecast scored {score}, should be near-random"


def test_directional_accuracy_perfect_on_matched_changes():
    actuals = np.array([0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03, 0.02, 0.01])
    forecasts = np.array([0.00, 0.015, 0.025, 0.015, 0.005, 0.015, 0.025, 0.015, 0.005])
    score = directional_accuracy(actuals, forecasts)
    assert score == 1.0


def test_directional_accuracy_zero_when_every_turn_wrong():
    actuals = np.array([0.01, 0.02, 0.03, 0.02])
    forecasts = np.array([0.03, 0.02, 0.01, 0.02])
    score = directional_accuracy(actuals, forecasts)
    assert score == 0.0

import numpy as np
import pandas as pd
from scripts.export_cycle_json import build_market_payload, build_cycle_json, MARKETS


def _fake_states(n=24):
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    rng = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "D": 0.01 * rng, "S": -0.01 * rng, "K": 0.02 * rng,
            "D_se": 0.04 + 0 * rng, "S_se": 0.03 + 0 * rng, "K_se": 0.05 + 0 * rng,
        },
        index=idx,
    )


def test_build_market_payload_shape():
    df = _fake_states()
    r = np.linspace(0, 0.2, len(df))
    theta = np.linspace(0, 6.0, len(df))
    payload = build_market_payload("DFW", "dallas", "#8b4c2a", df, r, theta)

    assert payload["key"] == "dallas"
    assert payload["color"] == "#8b4c2a"
    assert isinstance(payload["current_phase"], str) and payload["current_phase"]
    assert len(payload["series"]) == len(df)
    pt = payload["series"][0]
    assert set(pt) == {"date", "r", "theta", "D", "S", "K", "D_se", "S_se", "K_se", "quadrant"}
    assert pt["date"] == "2010-01"
    assert payload["current_phase"] == payload["series"][-1]["quadrant"]


def test_build_cycle_json_structure():
    df = _fake_states()
    r = np.linspace(0, 0.2, len(df)); theta = np.linspace(0, 6.0, len(df))
    per_market = {label: build_market_payload(label, key, "#000000", df, r, theta)
                  for label, key in MARKETS.items()}
    doc = build_cycle_json(per_market, as_of="2026-05")

    assert doc["meta"]["as_of"] == "2026-05"
    assert doc["meta"]["markets"] == list(MARKETS.keys())
    assert "disclaimer" in doc["meta"]
    assert set(doc["markets"]) == set(MARKETS.keys())

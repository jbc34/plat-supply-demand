#!/usr/bin/env python3
"""Export Texas multifamily cycle-spiral data to example.com as cycle.json.

Reads cached macro inputs (from scripts/refresh_macro_cache.py), runs the
state-space model for the four Texas MSAs, computes spiral coordinates, and
writes a single static JSON consumed by example.com /cycle. No network.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ordered: display label -> model msa key
MARKETS = {
    "DFW": "dallas",
    "Houston": "houston",
    "Austin": "austin",
    "San Antonio": "san_antonio",
}

COLORS = {  # Texas sports-team colors (instantly differentiable, locally resonant)
    "DFW": "#0053BC",          # Mavericks blue
    "Houston": "#CE1141",      # Rockets red
    "Austin": "#00B140",       # Austin FC verde
    "San Antonio": "#000000",  # Spurs black
}

OUT_PATH = Path("cycle.json")  # write to CWD by default; override via env CYCLE_JSON_OUT
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _clean(x):
    """JSON-safe float: NaN/inf -> None, else rounded float."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, 6)


def compute_phase(D, S, smooth=5):
    """Cycle phase from the demand-supply imbalance (D-S).

    Anchored so spread-maximum -> 135 deg (center of Peak/Late-Expansion), with
    time advancing counter-clockwise: Early Expansion -> Peak -> Contraction ->
    Recovery. spread and its velocity are z-scored so the phase portrait is
    circular rather than flattened onto the spread axis. Returns radians [0, 2pi).
    """
    spread = D - S
    if smooth > 1:
        spread = np.convolve(spread, np.ones(smooth) / smooth, mode="same")
    vel = np.gradient(spread)
    sp = (spread - np.nanmean(spread)) / (np.nanstd(spread) + 1e-9)
    vl = (vel - np.nanmean(vel)) / (np.nanstd(vel) + 1e-9)
    psi = np.degrees(np.arctan2(vl, sp))      # 0 deg at spread-max (vel~0)
    theta_deg = (135.0 - psi) % 360.0          # spread-max -> 135 deg (Peak center)
    return np.radians(theta_deg)


def build_market_payload(label, key, color, states_df, r, theta):
    """Serialize one market's states + spiral coords into the cycle.json shape."""
    from supply_demand.models.coordinates import phase_to_quadrant

    dates = [d.strftime("%Y-%m") for d in states_df.index]
    series = []
    for i, date in enumerate(dates):
        th = _clean(theta[i])
        quad = phase_to_quadrant(theta[i]) if th is not None else None
        series.append({
            "date": date,
            "r": _clean(r[i]),
            "theta": th,
            "D": _clean(states_df["D"].iloc[i]),
            "S": _clean(states_df["S"].iloc[i]),
            "K": _clean(states_df["K"].iloc[i]),
            "D_se": _clean(states_df["D_se"].iloc[i]),
            "S_se": _clean(states_df["S_se"].iloc[i]),
            "K_se": _clean(states_df["K_se"].iloc[i]),
            "quadrant": quad,
        })
    current_phase = next((p["quadrant"] for p in reversed(series) if p["quadrant"]), None)
    return {"key": key, "color": color, "current_phase": current_phase, "series": series}


def build_cycle_json(per_market, as_of):
    return {
        "meta": {
            "as_of": as_of,
            "markets": list(MARKETS.keys()),
            "sources": ["FRED", "BLS"],
            "method": ("State-space (Kalman) estimation of latent demand, supply, and "
                       "capital, mapped to cycle amplitude (r) and phase (theta). "
                       "Shown at quarterly resolution; a visual interpretation of the "
                       "underlying model."),
            "disclaimer": "Model estimate from public data. Not investment advice.",
        },
        "markets": per_market,
    }


def run():
    load_dotenv()  # not strictly needed offline, harmless
    from supply_demand.markets.multifamily.texas import TexasMultifamilyMarket
    from supply_demand.models.coordinates import compute_spiral_trajectory

    per_market = {}
    latest_date = None
    for label, key in MARKETS.items():
        print(f"  {label} ({key}) ...", flush=True)
        cache_file = CACHE_DIR / f"{key}_inputs.parquet"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Missing {cache_file}. Run scripts/refresh_macro_cache.py first (Task 1a)."
            )
        market = TexasMultifamilyMarket(msa=key)
        market._data = pd.read_parquet(cache_file)  # inject cache; no network
        states_df = market.run_estimation()
        # Resample monthly states to quarterly (mean). The real-estate cycle is a
        # quarterly phenomenon (CoStar fundamentals are quarterly); monthly carries
        # macro-proxy noise that is false precision for cycle position. Quarterly is
        # both calmer to render and the cadence reported in the methodology.
        states_df = states_df[["D", "S", "K", "D_se", "S_se", "K_se"]].resample("QE").mean()
        Dv, Sv, Kv = states_df["D"].values, states_df["S"].values, states_df["K"].values
        # Amplitude = true distance from equilibrium (magnitude of the standardized
        # D/S/K deviation). Surfaces the 2021-22 peak that the band-passed cycle
        # amplitude damped via filter edge effects, and matches the page's label
        # ("radius = distance from equilibrium").
        r = np.sqrt(Dv ** 2 + Sv ** 2 + Kv ** 2)
        # Phase anchored on the demand-supply imbalance (D-S): spread-maximum maps
        # to the center of the Peak/Late-Expansion quadrant, time runs
        # Early Expansion -> Peak -> Contraction -> Recovery. No bandpass/Hilbert,
        # so the recent cycle isn't edge-distorted.
        theta = compute_phase(Dv, Sv, smooth=1)  # quarterly already denoised
        per_market[label] = build_market_payload(label, key, COLORS[label], states_df, r, theta)
        last = states_df.index[-1]
        latest_date = max(latest_date, last) if latest_date is not None else last

    as_of = latest_date.strftime("%Y-%m") if latest_date is not None else ""
    doc = build_cycle_json(per_market, as_of=as_of)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, separators=(",", ":")))
    total = sum(len(m["series"]) for m in per_market.values())
    print(f"Wrote {OUT_PATH} ({total} points across {len(per_market)} markets, as_of={as_of})")


if __name__ == "__main__":
    sys.exit(run())

#!/usr/bin/env python3
"""Quarterly refresh: pull live FRED/BLS macro inputs for the 4 Texas MSAs and
cache them to disk. This is the ONLY step that touches the network / needs keys.

Run quarterly:  .venv/bin/python scripts/refresh_macro_cache.py
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

from dotenv import load_dotenv

# Hard timeout on every underlying HTTP call (requests/urllib honor this).
socket.setdefaulttimeout(30)

MARKETS = {"DFW": "dallas", "Houston": "houston", "Austin": "austin", "San Antonio": "san_antonio"}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def run():
    load_dotenv()  # supply-demand/.env (FRED_API_KEY)
    from supply_demand.markets.multifamily.texas import TexasMultifamilyMarket

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for label, key in MARKETS.items():
        print(f"  {label} ({key}) ...", flush=True)
        try:
            market = TexasMultifamilyMarket(msa=key)
            df = market.load_data(start_date="2010-01-01")
            if df is None or df.empty:
                raise RuntimeError("empty dataframe from load_data")
            out = CACHE_DIR / f"{key}_inputs.parquet"
            df.to_parquet(out)
            print(f"    cached {len(df)} rows -> {out.name}")
        except Exception as e:  # network/key failure: report, don't half-write
            print(f"    FAILED: {e}")
            failures.append((key, str(e)))

    if failures:
        print(f"\n{len(failures)} market(s) failed: {[f[0] for f in failures]}")
        return 1
    print(f"\nRefreshed {len(MARKETS)} markets -> {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

"""Econometrics module settings.

Separate from the paper's Settings class — owns its own paths and env vars.
Reads from the same .env file but does not import or extend supply_demand.config.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root: supply-demand/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data" / "econometrics"
FORECASTS_DIR = DATA_DIR / "forecasts"
REGISTRY_DIR = DATA_DIR / "registry"
SCENARIOS_DIR = DATA_DIR / "scenarios"
REPORTS_DIR = DATA_DIR / "reports"

# Metro config directory
METROS_DIR = Path(__file__).resolve().parent / "metros"

# CoStar panel path (reads from market-study-agent output)
# Set COSTAR_PANEL_PATH env var or defaults to sibling project
COSTAR_PANEL_PATH = Path(
    os.environ.get(
        "COSTAR_PANEL_PATH",
        str(PROJECT_ROOT.parent / "market-study-agent" / "data" / "public" / "processed" / "supply"),
    )
)

# API keys (reuse same env vars as other projects)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")


def load_metro_config(metro_id: str) -> dict:
    """Load a metro YAML config file.

    Args:
        metro_id: Metro identifier matching a YAML filename (e.g., "dallas_tx").

    Returns:
        Parsed YAML config dict.

    Raises:
        FileNotFoundError: If the metro config does not exist.
    """
    import yaml

    config_path = METROS_DIR / f"{metro_id}.yaml"
    if not config_path.exists():
        available = [f.stem for f in METROS_DIR.glob("*.yaml") if f.stem != "_template"]
        raise FileNotFoundError(
            f"Metro config not found: {config_path}\n"
            f"Available metros: {', '.join(available) or 'none'}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["_config_path"] = str(config_path)
    return config

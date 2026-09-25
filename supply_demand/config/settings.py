"""Application configuration management."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PACKAGE_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data" / "cache"
    CONFIG_DIR: Path = PACKAGE_ROOT / "config"

    # API Keys
    FRED_API_KEY: Optional[str] = os.environ.get("FRED_API_KEY")
    BLS_API_KEY: Optional[str] = os.environ.get("BLS_API_KEY")
    COSTAR_API_KEY: Optional[str] = os.environ.get("COSTAR_API_KEY")

    # Database
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL", f"sqlite:///{DATA_DIR}/market_data.db"
    )

    # Model defaults
    DEFAULT_ESTIMATION_WINDOW: int = int(
        os.environ.get("DEFAULT_ESTIMATION_WINDOW", "120")
    )  # months
    DEFAULT_FORECAST_HORIZON: int = int(
        os.environ.get("DEFAULT_FORECAST_HORIZON", "24")
    )  # months

    # Estimation settings
    KALMAN_INITIAL_VARIANCE: float = 1.0
    PROCESS_NOISE_SCALE: float = 0.1
    MEASUREMENT_NOISE_SCALE: float = 0.3

    # Data frequencies
    DEFAULT_FREQUENCY: str = "M"  # Monthly

    @classmethod
    def ensure_data_dir(cls) -> Path:
        """Ensure the data cache directory exists."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DATA_DIR

    @classmethod
    def get_config_path(cls, filename: str) -> Path:
        """Get path to a configuration file."""
        return cls.CONFIG_DIR / filename


# Singleton instance
settings = Settings()

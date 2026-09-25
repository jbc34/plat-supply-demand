"""
Configuration management for skills.

Provides centralized access to paths, parameters, and benchmarks.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@dataclass
class CoStarBenchmarks:
    """CoStar asking rent benchmarks by star rating."""
    star_12: Dict[str, float] = field(default_factory=dict)
    star_3: Dict[str, float] = field(default_factory=dict)
    star_45: Dict[str, float] = field(default_factory=dict)
    all_stars: Dict[str, float] = field(default_factory=dict)
    data_vintage: str = ""
    submarket: str = ""


@dataclass
class EquilibriumBenchmarks:
    """Equilibrium rent estimates from dislocation model."""
    hp_filter: float = 1516.0          # HP-filtered trend rent
    income_based: float = 1373.0       # 30% rent-to-income
    replacement_cost: float = 2142.0   # New construction economics
    data_vintage: str = ""


@dataclass
class ConvergenceParams:
    """Mean reversion parameters for rent projections."""
    theta: float = 0.307               # Convergence speed (completed cycles)
    theta_conservative: float = 0.141  # Full sample estimate
    theta_midpoint: float = 0.224      # Blended estimate
    organic_growth: float = 0.035      # 3.5% baseline
    half_life: float = 2.26            # Years for 50% gap closure


@dataclass
class SkillConfig:
    """Master configuration for all skills."""

    # Paths
    project_root: Path = field(default_factory=get_project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def costar_dir(self) -> Path:
        return self.data_dir / "costar"

    @property
    def costar_raw(self) -> Path:
        return self.costar_dir / "submarket-raw"

    @property
    def costar_clean(self) -> Path:
        return self.costar_dir / "submarket-clean"

    @property
    def costar_with_stars(self) -> Path:
        return self.costar_clean / "by_submarket_with_stars"

    @property
    def dislocation_dir(self) -> Path:
        return self.project_root / "dislocation_analysis"

    @property
    def property_analysis_dir(self) -> Path:
        return self.dislocation_dir / "property_analysis"

    @property
    def property_input_dir(self) -> Path:
        return self.property_analysis_dir / "input"

    @property
    def property_output_dir(self) -> Path:
        return self.property_analysis_dir / "output"

    @property
    def results_dir(self) -> Path:
        return self.dislocation_dir / "results"

    @property
    def skills_output_dir(self) -> Path:
        return self.project_root / "skills" / "output"

    # Default parameters
    convergence: ConvergenceParams = field(default_factory=ConvergenceParams)
    equilibrium: EquilibriumBenchmarks = field(default_factory=EquilibriumBenchmarks)

    # Submarket name mappings
    SUBMARKET_FILE_MAP: Dict[str, str] = field(default_factory=lambda: {
        "far_north_dallas": "dallasfort_worth_tx_usa_far_north_dallas",
        "plano": "dallasfort_worth_tx_usa_plano",
        "richardson": "dallasfort_worth_tx_usa_richardson",
        "frisco": "dallasfort_worth_tx_usa_friscoprosper",
        "allen_mckinney": "dallasfort_worth_tx_usa_allenmckinney",
        "north_dallas": "dallasfort_worth_tx_usa_north_dallas",
        "uptown": "dallasfort_worth_tx_usa_uptownpark_cities",
        "las_colinas": "dallasfort_worth_tx_usa_las_colinas",
        "irving": "dallasfort_worth_tx_usa_irving",
        "arlington": "dallasfort_worth_tx_usa_arlington",
    })

    def get_submarket_file(self, submarket: str) -> Path:
        """Get the parquet file path for a submarket."""
        clean_name = submarket.lower().replace(" ", "_").replace("-", "_")

        # Check direct mapping first
        if clean_name in self.SUBMARKET_FILE_MAP:
            filename = self.SUBMARKET_FILE_MAP[clean_name]
        else:
            # Try to find matching file
            filename = f"dallasfort_worth_tx_usa_{clean_name}"

        return self.costar_with_stars / f"{filename}.parquet"

    def load_costar_benchmarks(self, submarket: str, period: str = "2025 Q4") -> CoStarBenchmarks:
        """Load CoStar benchmarks for a submarket."""
        import pandas as pd

        file_path = self.get_submarket_file(submarket)
        if not file_path.exists():
            raise FileNotFoundError(f"Submarket file not found: {file_path}")

        df = pd.read_parquet(file_path)
        df_period = df[df['date'] == period]

        benchmarks = CoStarBenchmarks(
            submarket=submarket,
            data_vintage=period
        )

        for star_rating in ['1 & 2 Star', '3 Star', '4 & 5 Star', 'All']:
            row = df_period[df_period['star_rating'] == star_rating]
            if len(row) > 0:
                r = row.iloc[0]
                data = {
                    'overall': r.get('asking_rent_unit', 0),
                    'Studio': r.get('asking_rent_studio', 0),
                    '1BR': r.get('asking_rent_1br', 0),
                    '2BR': r.get('asking_rent_2br', 0),
                    '3BR': r.get('asking_rent_3br', 0),
                    'psf': r.get('asking_rent_sf', 0),
                }

                if star_rating == '1 & 2 Star':
                    benchmarks.star_12 = data
                elif star_rating == '3 Star':
                    benchmarks.star_3 = data
                elif star_rating == '4 & 5 Star':
                    benchmarks.star_45 = data
                else:
                    benchmarks.all_stars = data

        return benchmarks

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary for logging."""
        return {
            "project_root": str(self.project_root),
            "convergence": {
                "theta": self.convergence.theta,
                "organic_growth": self.convergence.organic_growth,
                "half_life": self.convergence.half_life
            },
            "equilibrium": {
                "hp_filter": self.equilibrium.hp_filter,
                "income_based": self.equilibrium.income_based,
                "replacement_cost": self.equilibrium.replacement_cost
            }
        }


# Global config instance
_config: Optional[SkillConfig] = None


def get_config() -> SkillConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = SkillConfig()
    return _config

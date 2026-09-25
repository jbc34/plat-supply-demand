"""
Rent Projection Skill

Projects rent growth using mean reversion convergence model.

Usage:
    from skills import RentProjection

    skill = RentProjection()
    result = skill.run(
        property_name="sample_property",
        current_rent=1278,
        submarket="far_north_dallas",
        projection_years=3
    )

Methodology:
    - Two-tier convergence: property → submarket → equilibrium
    - Mean reversion parameter θ = 0.307 (2.26-year half-life)
    - Organic growth: 3.5% annually
    - HP filter equilibrium as target

Author: Claude Code
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json

from skills.utils.config import get_config, SkillConfig, ConvergenceParams, EquilibriumBenchmarks
from skills.utils.logging import SkillLogger, create_session_id
from skills.utils.audit import AuditTrail


@dataclass
class ProjectionScenario:
    """A single rent projection scenario."""
    name: str
    description: str
    target: str
    target_value: float

    # Projected values (monthly)
    projections: List[float] = field(default_factory=list)

    # Summary metrics
    year1_growth: float = 0.0
    year2_growth: float = 0.0
    year3_growth: float = 0.0
    total_growth: float = 0.0
    cagr: float = 0.0
    final_rent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target": self.target,
            "target_value": self.target_value,
            "year1_growth": self.year1_growth,
            "year2_growth": self.year2_growth,
            "year3_growth": self.year3_growth,
            "total_growth": self.total_growth,
            "cagr": self.cagr,
            "final_rent": self.final_rent
        }


@dataclass
class RentProjectionResult:
    """Result of rent projection analysis."""
    property_name: str
    submarket: str
    session_id: str

    # Starting point
    current_rent: float = 0.0
    current_gap_to_equilibrium: float = 0.0

    # Scenarios
    scenarios: List[ProjectionScenario] = field(default_factory=list)

    # Convergence parameters used
    theta: float = 0.0
    organic_growth: float = 0.0
    half_life: float = 0.0

    # Output files
    report_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    figures: List[Path] = field(default_factory=list)
    audit_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "submarket": self.submarket,
            "session_id": self.session_id,
            "current_rent": self.current_rent,
            "current_gap_to_equilibrium": self.current_gap_to_equilibrium,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "theta": self.theta,
            "organic_growth": self.organic_growth,
            "half_life": self.half_life
        }


class RentProjection:
    """
    Rent Projection Skill

    Projects rent growth using convergence model with multiple scenarios:
    1. Market Drift - Maintain current premium/discount vs market
    2. Converge to 3-Star - Close gap to 3-Star benchmark
    3. Converge to 4/5-Star - Close gap to 4/5-Star (aggressive)
    4. Converge to Equilibrium - Close gap to HP fair value
    """

    SKILL_NAME = "rent-projection"
    VERSION = "1.0.0"

    # Color palette for scenarios
    COLORS = {
        'market_drift': '#3498db',
        'converge_3star': '#2ecc71',
        'converge_45star': '#9b59b6',
        'converge_equilibrium': '#e74c3c'
    }

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        property_name: str,
        current_rent: float,
        submarket: str,
        projection_years: int = 3,
        output_dir: Optional[Path] = None,
        costar_period: str = "2025 Q4",
        theta: Optional[float] = None,
        organic_growth: Optional[float] = None,
        include_45star_scenario: bool = True
    ) -> RentProjectionResult:
        """
        Run rent projection analysis.

        Args:
            property_name: Name of the property
            current_rent: Current average market rent
            submarket: Submarket name
            projection_years: Number of years to project
            output_dir: Output directory
            costar_period: CoStar data period
            theta: Override convergence speed (default: 0.307)
            organic_growth: Override organic growth rate (default: 0.035)
            include_45star_scenario: Include 4/5-Star convergence scenario

        Returns:
            RentProjectionResult with scenarios and outputs
        """
        # Initialize logging and audit
        session_id = create_session_id()
        logger = SkillLogger(self.SKILL_NAME, session_id)
        audit = AuditTrail(self.SKILL_NAME, session_id)

        logger.print_header(f"RENT PROJECTION: {property_name.upper()}")
        logger.info(f"Submarket: {submarket}")
        logger.info(f"Session: {session_id}")
        logger.print_separator()

        # Get convergence parameters
        conv = self.config.convergence
        theta_val = theta or conv.theta
        organic_val = organic_growth or conv.organic_growth
        half_life = np.log(2) / theta_val  # Years

        logger.info(f"Convergence parameters:")
        logger.info(f"  θ (convergence speed): {theta_val}")
        logger.info(f"  Half-life: {half_life:.2f} years")
        logger.info(f"  Organic growth: {organic_val:.1%}")

        # Set parameters for logging
        logger.set_parameters(
            property_name=property_name,
            current_rent=current_rent,
            submarket=submarket,
            projection_years=projection_years,
            theta=theta_val,
            organic_growth=organic_val,
            costar_period=costar_period
        )

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.property_output_dir / property_name
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # 1. Load benchmarks
            logger.print_separator()
            logger.info("Loading benchmarks...")

            benchmarks = self.config.load_costar_benchmarks(submarket, costar_period)
            equilibrium = self.config.equilibrium

            star_3 = benchmarks.star_3.get('overall', 0)
            star_45 = benchmarks.star_45.get('overall', 0)
            hp_eq = equilibrium.hp_filter

            logger.info(f"Current rent: ${current_rent:,.0f}")
            logger.info(f"3-Star benchmark: ${star_3:,.0f}")
            logger.info(f"4/5-Star benchmark: ${star_45:,.0f}")
            logger.info(f"HP Equilibrium: ${hp_eq:,.0f}")

            # Record model versions
            audit.add_model_version(
                "convergence_model", self.VERSION,
                parameters={
                    "theta": theta_val,
                    "organic_growth": organic_val,
                    "half_life": half_life,
                    "projection_years": projection_years
                },
                methodology_doc="dislocation_analysis/methodology/02_rent_convergence.md"
            )

            # Record assumptions
            audit.add_assumption(
                "Convergence Speed",
                f"θ = {theta_val}",
                "Based on completed cycle analysis (2010-2022)"
            )
            audit.add_assumption(
                "Organic Growth",
                f"{organic_val:.1%}",
                "Long-run nominal rent growth (inflation + real growth)"
            )

            # 2. Calculate current gaps
            gap_to_3star = (current_rent / star_3 - 1) * 100
            gap_to_45star = (current_rent / star_45 - 1) * 100
            gap_to_eq = (current_rent / hp_eq - 1) * 100

            logger.info(f"\nCurrent gaps:")
            logger.info(f"  vs 3-Star: {gap_to_3star:+.1f}%")
            logger.info(f"  vs 4/5-Star: {gap_to_45star:+.1f}%")
            logger.info(f"  vs HP Equilibrium: {gap_to_eq:+.1f}%")

            # 3. Generate scenarios
            logger.print_separator()
            logger.info("Generating projection scenarios...")

            months = projection_years * 12
            scenarios = []

            # Scenario 1: Market Drift
            # Maintain current gap while market grows at organic rate
            scenario_drift = self._project_scenario(
                name="Market Drift",
                description="Maintain current positioning vs 3-Star benchmark",
                current_rent=current_rent,
                target_rent=star_3 * (1 + gap_to_3star/100),  # Keep gap constant
                target_name="3-Star + Gap",
                target_growth_rate=organic_val,
                convergence_rate=0,  # No convergence, just follow market
                months=months,
                organic_growth=organic_val
            )
            scenarios.append(scenario_drift)

            # Scenario 2: Converge to 3-Star
            scenario_3star = self._project_scenario(
                name="Converge to 3-Star",
                description="Close gap to 3-Star benchmark over time",
                current_rent=current_rent,
                target_rent=star_3,
                target_name="3-Star",
                target_growth_rate=organic_val,
                convergence_rate=theta_val,
                months=months,
                organic_growth=organic_val
            )
            scenarios.append(scenario_3star)

            # Scenario 3: Converge to 4/5-Star (optional)
            if include_45star_scenario:
                scenario_45star = self._project_scenario(
                    name="Converge to 4/5-Star",
                    description="Close gap to 4/5-Star benchmark (aggressive)",
                    current_rent=current_rent,
                    target_rent=star_45,
                    target_name="4/5-Star",
                    target_growth_rate=organic_val,
                    convergence_rate=theta_val,
                    months=months,
                    organic_growth=organic_val
                )
                scenarios.append(scenario_45star)

            # Scenario 4: Converge to HP Equilibrium
            scenario_eq = self._project_scenario(
                name="Converge to Equilibrium",
                description="Close gap to HP fair value trend",
                current_rent=current_rent,
                target_rent=hp_eq,
                target_name="HP Equilibrium",
                target_growth_rate=organic_val,
                convergence_rate=theta_val,
                months=months,
                organic_growth=organic_val
            )
            scenarios.append(scenario_eq)

            # Log scenario results
            for s in scenarios:
                logger.info(f"\n{s.name}:")
                logger.info(f"  Year 1: {s.year1_growth:+.1f}%")
                logger.info(f"  Year 2: {s.year2_growth:+.1f}%")
                logger.info(f"  Year 3: {s.year3_growth:+.1f}%")
                logger.info(f"  Total: {s.total_growth:+.1f}% ({s.cagr:.1f}% CAGR)")
                logger.info(f"  Final Rent: ${s.final_rent:,.0f}")

            # 4. Build result
            result = RentProjectionResult(
                property_name=property_name,
                submarket=submarket,
                session_id=session_id,
                current_rent=current_rent,
                current_gap_to_equilibrium=gap_to_eq,
                scenarios=scenarios,
                theta=theta_val,
                organic_growth=organic_val,
                half_life=half_life
            )

            # 5. Generate outputs
            logger.print_separator()
            logger.info("Generating outputs...")

            # CSV with monthly projections
            csv_path = output_dir / "rent_growth_projections.csv"
            self._save_projections_csv(scenarios, months, csv_path)
            logger.add_output(str(csv_path))
            audit.add_output("projections_csv", csv_path, "Monthly rent projections")
            result.csv_path = csv_path
            logger.success(f"Saved: {csv_path.name}")

            # Figures
            fig_paths = self._generate_visualizations(
                scenarios, current_rent, star_3, star_45, hp_eq,
                property_name, projection_years, figures_dir
            )
            for fp in fig_paths:
                logger.add_output(str(fp))
                audit.add_output(fp.stem, fp, f"Visualization: {fp.stem}")
                result.figures.append(fp)
                logger.success(f"Saved: {fp.name}")

            # Markdown report
            report_path = output_dir / f"{property_name.upper()}_RENT_PROJECTIONS.md"
            report_content = self._generate_report(
                result, scenarios, benchmarks, equilibrium, audit
            )
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.add_output(str(report_path))
            audit.add_output("report", report_path, "Rent projection analysis report")
            result.report_path = report_path
            logger.success(f"Saved: {report_path.name}")

            # Audit trail
            audit_path = output_dir / f"audit_projection_{session_id}.json"
            audit.save(audit_path)
            result.audit_path = audit_path
            logger.success(f"Saved: {audit_path.name}")

            # Complete
            logger.complete("success")
            logger.save()

            logger.print_separator()
            logger.info(f"Analysis complete. Session: {session_id}")
            logger.info(f"Outputs: {output_dir}")

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.complete("failed")
            logger.save()
            raise

    def _project_scenario(
        self,
        name: str,
        description: str,
        current_rent: float,
        target_rent: float,
        target_name: str,
        target_growth_rate: float,
        convergence_rate: float,
        months: int,
        organic_growth: float
    ) -> ProjectionScenario:
        """
        Project rents under a convergence scenario.

        The model uses discrete monthly convergence:
        R(t+1) = R(t) * (1 + organic/12) + θ_monthly * (Target(t) - R(t))

        Where θ_monthly = 1 - (1-θ)^(1/12) for proper annual-to-monthly conversion.
        """
        scenario = ProjectionScenario(
            name=name,
            description=description,
            target=target_name,
            target_value=target_rent
        )

        projections = [current_rent]
        target_trajectory = [target_rent]

        # Convert annual theta to monthly
        theta_monthly = 1 - (1 - convergence_rate) ** (1/12)
        organic_monthly = (1 + organic_growth) ** (1/12) - 1

        rent = current_rent
        target = target_rent

        for m in range(1, months + 1):
            # Target grows at organic rate
            target = target * (1 + organic_monthly)
            target_trajectory.append(target)

            # Rent converges to target
            if convergence_rate > 0:
                rent = rent + (target - rent) * theta_monthly
            else:
                # Market drift - just organic growth
                rent = rent * (1 + organic_monthly)

            projections.append(rent)

        scenario.projections = projections

        # Calculate growth metrics
        year1_rent = projections[12]
        year2_rent = projections[24]
        year3_rent = projections[-1]

        scenario.year1_growth = (year1_rent / current_rent - 1) * 100
        scenario.year2_growth = (year2_rent / year1_rent - 1) * 100
        scenario.year3_growth = (year3_rent / year2_rent - 1) * 100
        scenario.total_growth = (year3_rent / current_rent - 1) * 100
        scenario.cagr = ((year3_rent / current_rent) ** (1/3) - 1) * 100
        scenario.final_rent = year3_rent

        return scenario

    def _save_projections_csv(
        self,
        scenarios: List[ProjectionScenario],
        months: int,
        output_path: Path
    ):
        """Save monthly projections to CSV."""
        data = {'month': list(range(months + 1))}

        for s in scenarios:
            col_name = s.name.lower().replace(' ', '_').replace('/', '_')
            data[col_name] = s.projections

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def _generate_visualizations(
        self,
        scenarios: List[ProjectionScenario],
        current_rent: float,
        star_3: float,
        star_45: float,
        hp_eq: float,
        property_name: str,
        projection_years: int,
        output_dir: Path
    ) -> List[Path]:
        """Generate visualization PNGs."""
        paths = []

        # 1. Rent trajectory chart
        fig, ax = plt.subplots(figsize=(14, 8))

        months = projection_years * 12
        x = np.arange(months + 1) / 12  # Years

        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

        for i, s in enumerate(scenarios):
            ax.plot(x, s.projections, label=s.name, color=colors[i % len(colors)],
                    linewidth=2.5)

        # Reference lines
        ax.axhline(y=star_3, color='green', linestyle='--', alpha=0.5,
                   label=f'3-Star: ${star_3:,.0f}')
        ax.axhline(y=star_45, color='purple', linestyle='--', alpha=0.5,
                   label=f'4/5-Star: ${star_45:,.0f}')
        ax.axhline(y=hp_eq, color='red', linestyle='--', alpha=0.5,
                   label=f'HP Equilibrium: ${hp_eq:,.0f}')

        ax.set_xlabel('Years', fontsize=12, fontweight='bold')
        ax.set_ylabel('Monthly Rent ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{property_name.title()}: {projection_years}-Year Rent Projections\n'
                     f'Starting Rent: ${current_rent:,.0f}',
                     fontsize=14, fontweight='bold')

        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, projection_years)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        path = output_dir / 'rent_projection_scenarios.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        # 2. Growth comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 7))

        n_scenarios = len(scenarios)
        x = np.arange(n_scenarios)
        width = 0.2

        year1 = [s.year1_growth for s in scenarios]
        year2 = [s.year2_growth for s in scenarios]
        year3 = [s.year3_growth for s in scenarios]

        ax.bar(x - width, year1, width, label='Year 1', color='#3498db')
        ax.bar(x, year2, width, label='Year 2', color='#2ecc71')
        ax.bar(x + width, year3, width, label='Year 3', color='#9b59b6')

        ax.set_xticks(x)
        ax.set_xticklabels([s.name for s in scenarios], rotation=15, ha='right')
        ax.set_ylabel('Annual Growth (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{property_name.title()}: Annual Rent Growth by Scenario',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)

        # Add value labels
        for i, s in enumerate(scenarios):
            ax.annotate(f'{s.total_growth:+.1f}%',
                       xy=(i, max(year1[i], year2[i], year3[i]) + 1),
                       ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        path = output_dir / 'annual_growth_comparison.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        return paths

    def _generate_report(
        self,
        result: RentProjectionResult,
        scenarios: List[ProjectionScenario],
        benchmarks,
        equilibrium,
        audit: AuditTrail
    ) -> str:
        """Generate markdown report."""
        report = f"""# {result.property_name.title()}: Rent Growth Projections

**Submarket**: {result.submarket.replace('_', ' ').title()}
**Analysis Date**: {audit.created_at[:10]}
**Session ID**: {result.session_id}
**Skill Version**: {self.VERSION}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Current Rent** | ${result.current_rent:,.0f} |
| **Gap to Equilibrium** | {result.current_gap_to_equilibrium:+.1f}% |
| **Convergence Speed (θ)** | {result.theta:.3f} |
| **Half-Life** | {result.half_life:.2f} years |
| **Organic Growth** | {result.organic_growth:.1%} |

---

## Scenario Comparison

| Scenario | Target | Year 1 | Year 2 | Year 3 | Total | CAGR | Final Rent |
|----------|--------|--------|--------|--------|-------|------|------------|
"""

        for s in scenarios:
            report += f"| {s.name} | {s.target} | {s.year1_growth:+.1f}% | {s.year2_growth:+.1f}% | {s.year3_growth:+.1f}% | **{s.total_growth:+.1f}%** | {s.cagr:.1f}% | ${s.final_rent:,.0f} |\n"

        report += f"""
---

## Scenario Details

"""

        for s in scenarios:
            report += f"""### {s.name}

**Description**: {s.description}

**Target**: {s.target} (${s.target_value:,.0f})

| Year | Starting Rent | Growth | Ending Rent |
|------|---------------|--------|-------------|
| Year 1 | ${result.current_rent:,.0f} | {s.year1_growth:+.1f}% | ${s.projections[12]:,.0f} |
| Year 2 | ${s.projections[12]:,.0f} | {s.year2_growth:+.1f}% | ${s.projections[24]:,.0f} |
| Year 3 | ${s.projections[24]:,.0f} | {s.year3_growth:+.1f}% | ${s.projections[-1]:,.0f} |

"""

        report += f"""---

## Methodology

### Mean Reversion Model

The projection uses a discrete monthly convergence model:

```
R(t+1) = R(t) × (1 + organic/12) + θ_monthly × (Target(t) - R(t))
```

Where:
- **R(t)** = Rent at time t
- **organic** = Annual organic growth rate ({result.organic_growth:.1%})
- **θ_monthly** = Monthly convergence rate derived from annual θ
- **Target(t)** = Benchmark rent at time t (growing at organic rate)

### Key Assumptions

| Assumption | Value | Rationale |
|------------|-------|-----------|
| Convergence Speed (θ) | {result.theta} | Based on completed cycle analysis (2010-2022) |
| Half-Life | {result.half_life:.2f} years | Time for 50% gap closure |
| Organic Growth | {result.organic_growth:.1%} | Long-run nominal rent growth |

---

## Visualizations

- `figures/rent_projection_scenarios.png` - Rent trajectory over time
- `figures/annual_growth_comparison.png` - Annual growth by scenario

---

{audit.generate_markdown_summary()}

---

*Generated by `/rent-projection` skill v{self.VERSION}*
"""

        return report

"""
Submarket Score Skill

Scores and ranks submarkets for investment potential.

Usage:
    from skills import SubmarketScore

    skill = SubmarketScore()
    result = skill.run(
        submarkets=["far_north_dallas", "plano", "richardson"],
        costar_period="2025 Q4"
    )

Methodology:
    - Multi-factor scoring across rent growth, vacancy, supply/demand, and dislocation
    - Weights calibrated from dislocation analysis framework
    - Score range: 0-100 (higher = more attractive)

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

from skills.utils.config import get_config, SkillConfig
from skills.utils.logging import SkillLogger, create_session_id
from skills.utils.audit import AuditTrail


@dataclass
class SubmarketMetrics:
    """Metrics for a single submarket."""
    name: str
    display_name: str

    # Raw metrics
    asking_rent: float = 0.0
    rent_3star: float = 0.0
    rent_45star: float = 0.0
    yoy_rent_growth: float = 0.0
    vacancy_rate: float = 0.0
    inventory: int = 0
    net_absorption: int = 0
    deliveries: int = 0

    # Dislocation metrics
    gap_to_equilibrium: float = 0.0  # % gap to HP equilibrium

    # Component scores (0-100)
    rent_growth_score: float = 0.0
    vacancy_score: float = 0.0
    supply_demand_score: float = 0.0
    dislocation_score: float = 0.0

    # Final composite score
    composite_score: float = 0.0
    rank: int = 0
    rating: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "asking_rent": self.asking_rent,
            "rent_3star": self.rent_3star,
            "yoy_rent_growth": self.yoy_rent_growth,
            "vacancy_rate": self.vacancy_rate,
            "gap_to_equilibrium": self.gap_to_equilibrium,
            "rent_growth_score": self.rent_growth_score,
            "vacancy_score": self.vacancy_score,
            "supply_demand_score": self.supply_demand_score,
            "dislocation_score": self.dislocation_score,
            "composite_score": self.composite_score,
            "rank": self.rank,
            "rating": self.rating
        }


@dataclass
class SubmarketScoreResult:
    """Result of submarket scoring analysis."""
    session_id: str
    analysis_period: str
    num_submarkets: int = 0

    # Ranked results
    rankings: List[SubmarketMetrics] = field(default_factory=list)

    # Score weights used
    weights: Dict[str, float] = field(default_factory=dict)

    # Output files
    report_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    figures: List[Path] = field(default_factory=list)
    audit_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "analysis_period": self.analysis_period,
            "num_submarkets": self.num_submarkets,
            "rankings": [r.to_dict() for r in self.rankings],
            "weights": self.weights
        }


class SubmarketScore:
    """
    Submarket Score Skill

    Scores submarkets on multiple investment criteria:
    - Rent Growth: YoY rent appreciation
    - Vacancy: Tightness of the market
    - Supply/Demand: Absorption vs deliveries balance
    - Dislocation: Gap to equilibrium (upside potential)
    """

    SKILL_NAME = "submarket-score"
    VERSION = "1.0.0"

    # Default score weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        'rent_growth': 0.25,
        'vacancy': 0.20,
        'supply_demand': 0.25,
        'dislocation': 0.30
    }

    # Rating thresholds
    RATING_THRESHOLDS = [
        (80, "A+ (Strong Buy)"),
        (70, "A (Buy)"),
        (60, "B+ (Accumulate)"),
        (50, "B (Hold)"),
        (40, "C (Underweight)"),
        (0, "D (Avoid)")
    ]

    # Color palette
    COLORS = {
        'A+ (Strong Buy)': '#27ae60',
        'A (Buy)': '#2ecc71',
        'B+ (Accumulate)': '#f39c12',
        'B (Hold)': '#e67e22',
        'C (Underweight)': '#e74c3c',
        'D (Avoid)': '#c0392b'
    }

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        submarkets: List[str],
        costar_period: str = "2025 Q4",
        output_dir: Optional[Path] = None,
        weights: Optional[Dict[str, float]] = None,
        equilibrium_rent: float = 1516.0
    ) -> SubmarketScoreResult:
        """
        Run submarket scoring analysis.

        Args:
            submarkets: List of submarket names to score
            costar_period: CoStar data period
            output_dir: Output directory
            weights: Custom score weights (default: balanced)
            equilibrium_rent: HP equilibrium rent for dislocation calc

        Returns:
            SubmarketScoreResult with rankings and outputs
        """
        # Initialize logging and audit
        session_id = create_session_id()
        logger = SkillLogger(self.SKILL_NAME, session_id)
        audit = AuditTrail(self.SKILL_NAME, session_id)

        logger.print_header(f"SUBMARKET SCORING: {len(submarkets)} SUBMARKETS")
        logger.info(f"Period: {costar_period}")
        logger.info(f"Session: {session_id}")
        logger.print_separator()

        # Set weights
        score_weights = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(score_weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"

        logger.info("Score Weights:")
        for k, v in score_weights.items():
            logger.info(f"  {k}: {v:.0%}")

        # Set parameters for logging
        logger.set_parameters(
            submarkets=submarkets,
            costar_period=costar_period,
            weights=score_weights,
            equilibrium_rent=equilibrium_rent
        )

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.skills_output_dir / "submarket_scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Record model version
        audit.add_model_version(
            "submarket_scoring", self.VERSION,
            parameters={
                "weights": score_weights,
                "equilibrium_rent": equilibrium_rent
            },
            methodology_doc="dislocation_analysis/methodology/04_submarket_scoring.md"
        )

        try:
            # 1. Load and score each submarket
            logger.print_separator()
            logger.info("Loading submarket data...")

            all_metrics = []
            year_ago = self._get_year_ago_period(costar_period)

            for submarket in submarkets:
                try:
                    metrics = self._load_submarket_metrics(
                        submarket, costar_period, year_ago, equilibrium_rent
                    )
                    all_metrics.append(metrics)
                    audit.add_data_source(
                        f"costar_{submarket}",
                        self.config.get_submarket_file(submarket),
                        description=f"CoStar {submarket}"
                    )
                    logger.info(f"  Loaded: {metrics.display_name}")
                except Exception as e:
                    logger.warning(f"  Failed to load {submarket}: {e}")

            if len(all_metrics) == 0:
                raise ValueError("No submarkets loaded successfully")

            # 2. Calculate component scores (normalize across submarkets)
            logger.print_separator()
            logger.info("Calculating component scores...")

            self._calculate_scores(all_metrics, score_weights)

            # 3. Rank and rate
            logger.info("Ranking submarkets...")

            all_metrics.sort(key=lambda x: x.composite_score, reverse=True)
            for i, m in enumerate(all_metrics, 1):
                m.rank = i
                m.rating = self._get_rating(m.composite_score)

            # Log top rankings
            logger.print_separator()
            logger.info("Top Rankings:")
            for m in all_metrics[:5]:
                logger.info(f"  #{m.rank} {m.display_name}: {m.composite_score:.1f} ({m.rating})")

            # 4. Build result
            result = SubmarketScoreResult(
                session_id=session_id,
                analysis_period=costar_period,
                num_submarkets=len(all_metrics),
                rankings=all_metrics,
                weights=score_weights
            )

            # 5. Generate outputs
            logger.print_separator()
            logger.info("Generating outputs...")

            # CSV
            csv_path = output_dir / "submarket_rankings.csv"
            self._save_rankings_csv(all_metrics, csv_path)
            logger.add_output(str(csv_path))
            audit.add_output("rankings_csv", csv_path, "Submarket rankings data")
            result.csv_path = csv_path
            logger.success(f"Saved: {csv_path.name}")

            # Figures
            fig_paths = self._generate_visualizations(
                all_metrics, score_weights, figures_dir
            )
            for fp in fig_paths:
                logger.add_output(str(fp))
                audit.add_output(fp.stem, fp, f"Visualization: {fp.stem}")
                result.figures.append(fp)
                logger.success(f"Saved: {fp.name}")

            # Markdown report
            report_path = output_dir / f"SUBMARKET_RANKINGS_{costar_period.replace(' ', '_')}.md"
            report_content = self._generate_report(result, all_metrics, audit)
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.add_output(str(report_path))
            audit.add_output("report", report_path, "Submarket scoring report")
            result.report_path = report_path
            logger.success(f"Saved: {report_path.name}")

            # Audit trail
            audit_path = output_dir / f"audit_{session_id}.json"
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

    def _get_year_ago_period(self, period: str) -> str:
        """Get the period from one year ago."""
        parts = period.split(' ')
        year = int(parts[0]) - 1
        return f"{year} {parts[1]}"

    def _load_submarket_metrics(
        self,
        submarket: str,
        period: str,
        year_ago: str,
        equilibrium_rent: float
    ) -> SubmarketMetrics:
        """Load metrics for a single submarket."""
        file_path = self.config.get_submarket_file(submarket)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_parquet(file_path)

        # Get current period - All star rating
        df_all = df[df['star_rating'] == 'All']
        df_current = df_all[df_all['date'] == period]

        if len(df_current) == 0:
            raise ValueError(f"Period {period} not found for {submarket}")

        current = df_current.iloc[0]

        # Get year-ago for YoY calc
        df_yago = df_all[df_all['date'] == year_ago]
        yoy_growth = 0.0
        if len(df_yago) > 0 and df_yago.iloc[0]['asking_rent_unit'] > 0:
            yoy_growth = (current['asking_rent_unit'] / df_yago.iloc[0]['asking_rent_unit'] - 1) * 100

        # Get star benchmarks
        benchmarks = self.config.load_costar_benchmarks(submarket, period)

        # Calculate dislocation
        gap_to_eq = ((current.get('asking_rent_unit', 0) / equilibrium_rent) - 1) * 100

        # Display name
        display_name = submarket.replace('_', ' ').title()

        return SubmarketMetrics(
            name=submarket,
            display_name=display_name,
            asking_rent=current.get('asking_rent_unit', 0),
            rent_3star=benchmarks.star_3.get('overall', 0),
            rent_45star=benchmarks.star_45.get('overall', 0),
            yoy_rent_growth=yoy_growth,
            vacancy_rate=current.get('vacancy_rate', 0),
            inventory=int(current.get('inventory_units', 0)),
            net_absorption=int(current.get('net_absorption_units', 0)),
            deliveries=int(current.get('deliveries_units', 0)),
            gap_to_equilibrium=gap_to_eq
        )

    def _calculate_scores(
        self,
        metrics: List[SubmarketMetrics],
        weights: Dict[str, float]
    ):
        """Calculate component and composite scores."""
        if len(metrics) == 0:
            return

        # Extract raw values
        rent_growths = [m.yoy_rent_growth for m in metrics]
        vacancies = [m.vacancy_rate for m in metrics]
        absorptions = [m.net_absorption for m in metrics]
        deliveries_list = [m.deliveries for m in metrics]
        dislocations = [m.gap_to_equilibrium for m in metrics]

        # Calculate supply/demand ratios (absorption / deliveries)
        sd_ratios = []
        for a, d in zip(absorptions, deliveries_list):
            if d > 0:
                sd_ratios.append(a / d)
            elif a > 0:
                sd_ratios.append(2.0)  # Absorption with no deliveries = strong
            else:
                sd_ratios.append(1.0)  # Neutral

        # Normalize each component to 0-100 scale
        for m in metrics:
            # Rent growth: higher is better (linear scale, -5% to +10%)
            m.rent_growth_score = self._normalize_score(
                m.yoy_rent_growth, -5, 10, higher_is_better=True
            )

            # Vacancy: lower is better (linear scale, 3% to 12%)
            m.vacancy_score = self._normalize_score(
                m.vacancy_rate, 3, 12, higher_is_better=False
            )

            # Supply/demand: higher ratio is better (0.5 to 2.0)
            if m.deliveries > 0:
                sd_ratio = m.net_absorption / m.deliveries
            elif m.net_absorption > 0:
                sd_ratio = 2.0
            else:
                sd_ratio = 1.0
            m.supply_demand_score = self._normalize_score(
                sd_ratio, 0.5, 2.0, higher_is_better=True
            )

            # Dislocation: negative gap (below equilibrium) is better (upside potential)
            # Range: -20% to +10%
            m.dislocation_score = self._normalize_score(
                m.gap_to_equilibrium, -20, 10, higher_is_better=False
            )

            # Composite score
            m.composite_score = (
                m.rent_growth_score * weights['rent_growth'] +
                m.vacancy_score * weights['vacancy'] +
                m.supply_demand_score * weights['supply_demand'] +
                m.dislocation_score * weights['dislocation']
            )

    def _normalize_score(
        self,
        value: float,
        min_val: float,
        max_val: float,
        higher_is_better: bool
    ) -> float:
        """Normalize a value to 0-100 scale."""
        if max_val == min_val:
            return 50.0

        # Clamp to range
        clamped = max(min_val, min(max_val, value))

        # Scale to 0-100
        if higher_is_better:
            score = (clamped - min_val) / (max_val - min_val) * 100
        else:
            score = (max_val - clamped) / (max_val - min_val) * 100

        return score

    def _get_rating(self, score: float) -> str:
        """Get letter rating from score."""
        for threshold, rating in self.RATING_THRESHOLDS:
            if score >= threshold:
                return rating
        return "D (Avoid)"

    def _save_rankings_csv(self, metrics: List[SubmarketMetrics], output_path: Path):
        """Save rankings to CSV."""
        data = []
        for m in metrics:
            data.append({
                'rank': m.rank,
                'submarket': m.display_name,
                'composite_score': round(m.composite_score, 1),
                'rating': m.rating,
                'rent_growth_score': round(m.rent_growth_score, 1),
                'vacancy_score': round(m.vacancy_score, 1),
                'supply_demand_score': round(m.supply_demand_score, 1),
                'dislocation_score': round(m.dislocation_score, 1),
                'asking_rent': round(m.asking_rent, 0),
                'yoy_rent_growth': round(m.yoy_rent_growth, 1),
                'vacancy_rate': round(m.vacancy_rate, 1),
                'gap_to_equilibrium': round(m.gap_to_equilibrium, 1)
            })
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def _generate_visualizations(
        self,
        metrics: List[SubmarketMetrics],
        weights: Dict[str, float],
        output_dir: Path
    ) -> List[Path]:
        """Generate visualization PNGs."""
        paths = []

        # 1. Rankings bar chart
        fig, ax = plt.subplots(figsize=(14, max(8, len(metrics) * 0.5)))

        names = [m.display_name for m in metrics]
        scores = [m.composite_score for m in metrics]
        colors = [self.COLORS.get(m.rating, '#95a5a6') for m in metrics]

        y = range(len(metrics))
        bars = ax.barh(y, scores, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel('Composite Score', fontsize=11, fontweight='bold')
        ax.set_title('Submarket Investment Rankings', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

        # Add score labels
        for i, (score, rating) in enumerate(zip(scores, [m.rating for m in metrics])):
            ax.text(score + 1, i, f'{score:.0f} ({rating.split()[0]})',
                    va='center', fontsize=9)

        plt.tight_layout()
        path = output_dir / 'submarket_rankings.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        # 2. Component scores radar chart (top 5)
        if len(metrics) >= 3:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            categories = ['Rent\nGrowth', 'Vacancy', 'Supply/\nDemand', 'Dislocation']
            n_cats = len(categories)
            angles = [n / n_cats * 2 * np.pi for n in range(n_cats)]
            angles += angles[:1]  # Close the loop

            top_n = min(5, len(metrics))
            colors = plt.cm.tab10(np.linspace(0, 1, top_n))

            for i, m in enumerate(metrics[:top_n]):
                values = [
                    m.rent_growth_score,
                    m.vacancy_score,
                    m.supply_demand_score,
                    m.dislocation_score
                ]
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=m.display_name,
                        color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_title('Top 5 Submarkets: Component Scores', fontsize=13, fontweight='bold',
                         y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

            plt.tight_layout()
            path = output_dir / 'component_scores_radar.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            paths.append(path)

        return paths

    def _generate_report(
        self,
        result: SubmarketScoreResult,
        metrics: List[SubmarketMetrics],
        audit: AuditTrail
    ) -> str:
        """Generate markdown report."""
        report = f"""# Submarket Investment Rankings

**Analysis Period**: {result.analysis_period}
**Submarkets Analyzed**: {result.num_submarkets}
**Session ID**: {result.session_id}
**Skill Version**: {self.VERSION}

---

## Methodology

The composite score weights multiple investment factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Rent Growth | {result.weights['rent_growth']:.0%} | YoY asking rent appreciation |
| Vacancy | {result.weights['vacancy']:.0%} | Market tightness (lower = better) |
| Supply/Demand | {result.weights['supply_demand']:.0%} | Absorption vs deliveries balance |
| Dislocation | {result.weights['dislocation']:.0%} | Gap to equilibrium (upside potential) |

---

## Overall Rankings

| Rank | Submarket | Score | Rating | Rent Growth | Vacancy | S/D | Dislocation |
|------|-----------|-------|--------|-------------|---------|-----|-------------|
"""

        for m in metrics:
            report += f"| {m.rank} | **{m.display_name}** | {m.composite_score:.0f} | {m.rating} | {m.rent_growth_score:.0f} | {m.vacancy_score:.0f} | {m.supply_demand_score:.0f} | {m.dislocation_score:.0f} |\n"

        report += f"""
---

## Top Picks

"""

        for m in metrics[:3]:
            report += f"""### #{m.rank} {m.display_name}

**Score**: {m.composite_score:.0f}/100 ({m.rating})

| Metric | Value | Score |
|--------|-------|-------|
| Asking Rent | ${m.asking_rent:,.0f} | - |
| YoY Rent Growth | {m.yoy_rent_growth:+.1f}% | {m.rent_growth_score:.0f} |
| Vacancy Rate | {m.vacancy_rate:.1f}% | {m.vacancy_score:.0f} |
| Gap to Equilibrium | {m.gap_to_equilibrium:+.1f}% | {m.dislocation_score:.0f} |

"""

        report += f"""---

## Detailed Metrics

| Submarket | Asking Rent | 3-Star | 4/5-Star | YoY Growth | Vacancy | Inventory |
|-----------|-------------|--------|----------|------------|---------|-----------|
"""

        for m in metrics:
            report += f"| {m.display_name} | ${m.asking_rent:,.0f} | ${m.rent_3star:,.0f} | ${m.rent_45star:,.0f} | {m.yoy_rent_growth:+.1f}% | {m.vacancy_rate:.1f}% | {m.inventory:,} |\n"

        report += f"""
---

## Visualizations

- `figures/submarket_rankings.png` - Composite scores by submarket
- `figures/component_scores_radar.png` - Top 5 component breakdown

---

{audit.generate_markdown_summary()}

---

*Generated by `/submarket-score` skill v{self.VERSION}*
"""

        return report

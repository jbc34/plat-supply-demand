"""
Market Pulse Skill

Quick market state assessment for a submarket.

Usage:
    from skills import MarketPulse

    skill = MarketPulse()
    result = skill.run(
        submarket="far_north_dallas",
        costar_period="2025 Q4"
    )

Provides:
    - Current rent levels by star rating
    - Recent rent growth trends
    - Supply pipeline (if available)
    - Vacancy trends
    - Market regime classification

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

from skills.utils.config import get_config, SkillConfig, CoStarBenchmarks
from skills.utils.logging import SkillLogger, create_session_id
from skills.utils.audit import AuditTrail


@dataclass
class MarketMetrics:
    """Market metrics for a period."""
    period: str
    asking_rent: float = 0.0
    effective_rent: float = 0.0
    vacancy_rate: float = 0.0
    inventory: int = 0
    absorption: int = 0
    deliveries: int = 0


@dataclass
class MarketPulseResult:
    """Result of market pulse analysis."""
    submarket: str
    session_id: str
    analysis_period: str

    # Current state
    current_asking_rent: float = 0.0
    current_vacancy: float = 0.0
    current_inventory: int = 0

    # By star rating
    rent_3star: float = 0.0
    rent_45star: float = 0.0
    rent_12star: float = 0.0

    # Trends (YoY)
    yoy_rent_growth: float = 0.0
    yoy_vacancy_change: float = 0.0
    yoy_absorption: int = 0
    yoy_deliveries: int = 0

    # Market regime
    regime: str = ""
    regime_description: str = ""

    # Historical data
    recent_history: List[MarketMetrics] = field(default_factory=list)

    # Output files
    report_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    figures: List[Path] = field(default_factory=list)
    audit_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submarket": self.submarket,
            "session_id": self.session_id,
            "analysis_period": self.analysis_period,
            "current_asking_rent": self.current_asking_rent,
            "current_vacancy": self.current_vacancy,
            "rent_3star": self.rent_3star,
            "rent_45star": self.rent_45star,
            "yoy_rent_growth": self.yoy_rent_growth,
            "yoy_vacancy_change": self.yoy_vacancy_change,
            "regime": self.regime,
            "regime_description": self.regime_description
        }


class MarketPulse:
    """
    Market Pulse Skill

    Provides a quick snapshot of market conditions:
    - Current rent levels by quality tier
    - Rent growth trends
    - Vacancy and absorption
    - Market regime (Expansion, Peak, Contraction, Trough)
    """

    SKILL_NAME = "market-pulse"
    VERSION = "1.0.0"

    # Regime classification thresholds
    REGIME_THRESHOLDS = {
        'rent_growth_strong': 5.0,  # >5% YoY = strong growth
        'rent_growth_weak': 0.0,    # <0% = declining
        'vacancy_high': 8.0,        # >8% = elevated vacancy
        'vacancy_low': 5.0,         # <5% = tight market
    }

    # Color palette
    COLORS = {
        'rent': '#3498db',
        'vacancy': '#e74c3c',
        'absorption': '#2ecc71',
        'deliveries': '#f39c12',
        '3_star': '#27ae60',
        '45_star': '#9b59b6',
        '12_star': '#e67e22'
    }

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        submarket: str,
        costar_period: str = "2025 Q4",
        output_dir: Optional[Path] = None,
        lookback_quarters: int = 8
    ) -> MarketPulseResult:
        """
        Run market pulse analysis.

        Args:
            submarket: Submarket name
            costar_period: Current analysis period
            output_dir: Output directory
            lookback_quarters: Number of quarters of history to show

        Returns:
            MarketPulseResult with market snapshot
        """
        # Initialize logging and audit
        session_id = create_session_id()
        logger = SkillLogger(self.SKILL_NAME, session_id)
        audit = AuditTrail(self.SKILL_NAME, session_id)

        logger.print_header(f"MARKET PULSE: {submarket.upper().replace('_', ' ')}")
        logger.info(f"Period: {costar_period}")
        logger.info(f"Session: {session_id}")
        logger.print_separator()

        # Set parameters for logging
        logger.set_parameters(
            submarket=submarket,
            costar_period=costar_period,
            lookback_quarters=lookback_quarters
        )

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.skills_output_dir / "market_pulse" / submarket
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # 1. Load submarket data
            logger.info("Loading CoStar data...")
            file_path = self.config.get_submarket_file(submarket)
            if not file_path.exists():
                raise FileNotFoundError(f"Submarket file not found: {file_path}")

            df = pd.read_parquet(file_path)
            audit.add_data_source("costar_submarket", file_path,
                                  description=f"CoStar {submarket}")

            # Record model version
            audit.add_model_version(
                "market_pulse", self.VERSION,
                parameters={
                    "lookback_quarters": lookback_quarters,
                    "regime_thresholds": self.REGIME_THRESHOLDS
                }
            )

            # 2. Get current period data
            logger.info(f"Analyzing {costar_period}...")

            # Get All ratings for overall metrics
            df_all = df[df['star_rating'] == 'All']
            df_current = df_all[df_all['date'] == costar_period]

            if len(df_current) == 0:
                available = df_all['date'].unique()[-5:]
                raise ValueError(f"Period {costar_period} not found. Available: {list(available)}")

            current = df_current.iloc[0]

            # Get star rating benchmarks
            benchmarks = self.config.load_costar_benchmarks(submarket, costar_period)

            # 3. Calculate trends
            logger.info("Calculating trends...")

            # Get year-ago period
            year_ago = self._get_year_ago_period(costar_period)
            df_yago = df_all[df_all['date'] == year_ago]

            yoy_rent_growth = 0.0
            yoy_vacancy_change = 0.0

            if len(df_yago) > 0:
                yago = df_yago.iloc[0]
                if yago['asking_rent_unit'] > 0:
                    yoy_rent_growth = (current['asking_rent_unit'] / yago['asking_rent_unit'] - 1) * 100
                if 'vacancy_rate' in yago and pd.notna(yago.get('vacancy_rate')):
                    yoy_vacancy_change = current.get('vacancy_rate', 0) - yago.get('vacancy_rate', 0)

            # 4. Build historical metrics
            recent_periods = self._get_recent_periods(df_all['date'].unique(), lookback_quarters)
            recent_history = []

            for period in recent_periods:
                row = df_all[df_all['date'] == period]
                if len(row) > 0:
                    r = row.iloc[0]
                    metrics = MarketMetrics(
                        period=period,
                        asking_rent=r.get('asking_rent_unit', 0),
                        effective_rent=r.get('effective_rent_unit', 0),
                        vacancy_rate=r.get('vacancy_rate', 0),
                        inventory=int(r.get('inventory_units', 0)),
                        absorption=int(r.get('net_absorption_units', 0)),
                        deliveries=int(r.get('deliveries_units', 0))
                    )
                    recent_history.append(metrics)

            # 5. Classify market regime
            regime, regime_desc = self._classify_regime(
                yoy_rent_growth,
                current.get('vacancy_rate', 0),
                int(current.get('net_absorption_units', 0)),
                int(current.get('deliveries_units', 0))
            )

            logger.info(f"\nMarket Regime: {regime}")
            logger.info(f"  {regime_desc}")

            # 6. Build result
            result = MarketPulseResult(
                submarket=submarket,
                session_id=session_id,
                analysis_period=costar_period,
                current_asking_rent=current.get('asking_rent_unit', 0),
                current_vacancy=current.get('vacancy_rate', 0),
                current_inventory=int(current.get('inventory_units', 0)),
                rent_3star=benchmarks.star_3.get('overall', 0),
                rent_45star=benchmarks.star_45.get('overall', 0),
                rent_12star=benchmarks.star_12.get('overall', 0),
                yoy_rent_growth=yoy_rent_growth,
                yoy_vacancy_change=yoy_vacancy_change,
                yoy_absorption=int(current.get('net_absorption_units', 0)),
                yoy_deliveries=int(current.get('deliveries_units', 0)),
                regime=regime,
                regime_description=regime_desc,
                recent_history=recent_history
            )

            # Log key metrics
            logger.print_separator()
            logger.info("Current Market State:")
            logger.info(f"  Asking Rent (All): ${result.current_asking_rent:,.0f}")
            logger.info(f"  3-Star Rent: ${result.rent_3star:,.0f}")
            logger.info(f"  4/5-Star Rent: ${result.rent_45star:,.0f}")
            logger.info(f"  Vacancy Rate: {result.current_vacancy:.1f}%")
            logger.info(f"  YoY Rent Growth: {result.yoy_rent_growth:+.1f}%")

            # 7. Generate outputs
            logger.print_separator()
            logger.info("Generating outputs...")

            # CSV with history
            csv_path = output_dir / f"{submarket}_market_history.csv"
            self._save_history_csv(recent_history, csv_path)
            logger.add_output(str(csv_path))
            audit.add_output("history_csv", csv_path, "Market history data")
            result.csv_path = csv_path
            logger.success(f"Saved: {csv_path.name}")

            # Figures
            fig_paths = self._generate_visualizations(
                result, recent_history, submarket, figures_dir
            )
            for fp in fig_paths:
                logger.add_output(str(fp))
                audit.add_output(fp.stem, fp, f"Visualization: {fp.stem}")
                result.figures.append(fp)
                logger.success(f"Saved: {fp.name}")

            # Markdown report
            report_path = output_dir / f"{submarket.upper()}_MARKET_PULSE.md"
            report_content = self._generate_report(result, recent_history, audit)
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.add_output(str(report_path))
            audit.add_output("report", report_path, "Market pulse report")
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
        # Format: "2025 Q4" -> "2024 Q4"
        parts = period.split(' ')
        year = int(parts[0]) - 1
        return f"{year} {parts[1]}"

    def _get_recent_periods(self, all_periods, n: int) -> List[str]:
        """Get the most recent n periods."""
        # Sort periods chronologically
        sorted_periods = sorted(all_periods, key=lambda x: (int(x.split()[0]), x.split()[1]))
        return sorted_periods[-n:]

    def _classify_regime(
        self,
        rent_growth: float,
        vacancy: float,
        absorption: int,
        deliveries: int
    ) -> Tuple[str, str]:
        """
        Classify market regime based on current conditions.

        Returns:
            Tuple of (regime_name, description)
        """
        t = self.REGIME_THRESHOLDS

        # Strong growth + low vacancy = EXPANSION
        if rent_growth > t['rent_growth_strong'] and vacancy < t['vacancy_high']:
            return "EXPANSION", "Strong rent growth with manageable vacancy - landlord-favorable conditions"

        # Positive growth + rising vacancy = PEAK/LATE CYCLE
        elif rent_growth > 0 and vacancy > t['vacancy_low'] and deliveries > absorption:
            return "LATE CYCLE", "Positive growth but supply outpacing demand - watch for softening"

        # Negative growth = CONTRACTION
        elif rent_growth < t['rent_growth_weak']:
            if vacancy > t['vacancy_high']:
                return "CONTRACTION", "Declining rents with elevated vacancy - tenant-favorable market"
            else:
                return "SOFTENING", "Rents declining despite moderate vacancy - early correction"

        # Low/moderate growth + low vacancy = RECOVERY/STABLE
        elif vacancy < t['vacancy_low']:
            return "TIGHTENING", "Low vacancy indicates supply constraints - potential rent acceleration ahead"

        # Default: STABLE
        else:
            return "STABLE", "Balanced market conditions with moderate growth and vacancy"

    def _save_history_csv(self, history: List[MarketMetrics], output_path: Path):
        """Save historical data to CSV."""
        data = []
        for m in history:
            data.append({
                'period': m.period,
                'asking_rent': m.asking_rent,
                'effective_rent': m.effective_rent,
                'vacancy_rate': m.vacancy_rate,
                'inventory': m.inventory,
                'absorption': m.absorption,
                'deliveries': m.deliveries
            })
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def _generate_visualizations(
        self,
        result: MarketPulseResult,
        history: List[MarketMetrics],
        submarket: str,
        output_dir: Path
    ) -> List[Path]:
        """Generate visualization PNGs."""
        paths = []

        if len(history) < 2:
            return paths

        # 1. Rent and Vacancy Trend
        fig, ax1 = plt.subplots(figsize=(14, 7))

        periods = [m.period for m in history]
        rents = [m.asking_rent for m in history]
        vacancies = [m.vacancy_rate for m in history]

        x = range(len(periods))

        # Rent on left axis
        color1 = self.COLORS['rent']
        ax1.set_xlabel('Period', fontsize=11)
        ax1.set_ylabel('Asking Rent ($)', color=color1, fontsize=11)
        ax1.plot(x, rents, color=color1, linewidth=2.5, marker='o', label='Asking Rent')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))

        # Vacancy on right axis
        ax2 = ax1.twinx()
        color2 = self.COLORS['vacancy']
        ax2.set_ylabel('Vacancy Rate (%)', color=color2, fontsize=11)
        ax2.plot(x, vacancies, color=color2, linewidth=2.5, marker='s', label='Vacancy')
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_xticks(x)
        ax1.set_xticklabels(periods, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        title = f"{submarket.replace('_', ' ').title()}: Rent & Vacancy Trends"
        ax1.set_title(title, fontsize=13, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        path = output_dir / 'rent_vacancy_trend.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        # 2. Supply/Demand Balance
        fig, ax = plt.subplots(figsize=(14, 7))

        absorptions = [m.absorption for m in history]
        deliveries_list = [m.deliveries for m in history]

        width = 0.35
        x = np.arange(len(periods))

        ax.bar(x - width/2, absorptions, width, label='Net Absorption', color=self.COLORS['absorption'])
        ax.bar(x + width/2, deliveries_list, width, label='Deliveries', color=self.COLORS['deliveries'])

        ax.set_xlabel('Period', fontsize=11)
        ax.set_ylabel('Units', fontsize=11)
        ax.set_title(f"{submarket.replace('_', ' ').title()}: Supply vs Demand",
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)

        plt.tight_layout()
        path = output_dir / 'supply_demand_balance.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        # 3. Rent by Star Rating (current period)
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['1 & 2 Star', '3 Star', '4 & 5 Star']
        values = [result.rent_12star, result.rent_3star, result.rent_45star]
        colors = [self.COLORS['12_star'], self.COLORS['3_star'], self.COLORS['45_star']]

        bars = ax.bar(categories, values, color=colors, edgecolor='black')

        ax.set_ylabel('Asking Rent ($)', fontsize=11)
        ax.set_title(f"{submarket.replace('_', ' ').title()}: Rent by Quality Tier ({result.analysis_period})",
                     fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'${val:,.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, val),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = output_dir / 'rent_by_star_rating.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        return paths

    def _generate_report(
        self,
        result: MarketPulseResult,
        history: List[MarketMetrics],
        audit: AuditTrail
    ) -> str:
        """Generate markdown report."""
        report = f"""# {result.submarket.replace('_', ' ').title()}: Market Pulse

**Analysis Period**: {result.analysis_period}
**Session ID**: {result.session_id}
**Skill Version**: {self.VERSION}

---

## Market Regime: {result.regime}

> {result.regime_description}

---

## Current Snapshot

| Metric | Value |
|--------|-------|
| **Asking Rent (All)** | ${result.current_asking_rent:,.0f} |
| **3-Star Rent** | ${result.rent_3star:,.0f} |
| **4/5-Star Rent** | ${result.rent_45star:,.0f} |
| **1/2-Star Rent** | ${result.rent_12star:,.0f} |
| **Vacancy Rate** | {result.current_vacancy:.1f}% |
| **Inventory** | {result.current_inventory:,} units |

---

## Year-over-Year Trends

| Metric | Change |
|--------|--------|
| **Rent Growth** | {result.yoy_rent_growth:+.1f}% |
| **Vacancy Change** | {result.yoy_vacancy_change:+.1f} ppts |
| **Net Absorption** | {result.yoy_absorption:,} units |
| **Deliveries** | {result.yoy_deliveries:,} units |

---

## Historical Trends

| Period | Asking Rent | Vacancy | Absorption | Deliveries |
|--------|-------------|---------|------------|------------|
"""

        for m in reversed(history):
            report += f"| {m.period} | ${m.asking_rent:,.0f} | {m.vacancy_rate:.1f}% | {m.absorption:,} | {m.deliveries:,} |\n"

        report += f"""
---

## Quality Tier Analysis

The rent spread between quality tiers indicates market segmentation:

| Comparison | Spread | Interpretation |
|------------|--------|----------------|
| 4/5 vs 3 Star | ${result.rent_45star - result.rent_3star:,.0f} ({((result.rent_45star/result.rent_3star)-1)*100:.1f}%) | Premium for Class A |
| 3 vs 1/2 Star | ${result.rent_3star - result.rent_12star:,.0f} ({((result.rent_3star/result.rent_12star)-1)*100:.1f}%) | Class B premium |

---

## Visualizations

- `figures/rent_vacancy_trend.png` - Rent and vacancy over time
- `figures/supply_demand_balance.png` - Absorption vs deliveries
- `figures/rent_by_star_rating.png` - Current rents by quality tier

---

{audit.generate_markdown_summary()}

---

*Generated by `/market-pulse` skill v{self.VERSION}*
"""

        return report

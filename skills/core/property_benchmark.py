"""
Property Benchmark Skill

Benchmarks a property against submarket comps and equilibrium values.

Usage:
    from skills import PropertyBenchmark

    skill = PropertyBenchmark()
    result = skill.run(
        property_name="sample_property",
        rent_roll_path="path/to/rent_roll.csv",
        submarket="far_north_dallas"
    )

Outputs:
    - Markdown report with benchmark comparison
    - CSV with floorplan-level analysis
    - PNG visualizations

Author: Claude Code
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json

from skills.utils.config import get_config, SkillConfig, CoStarBenchmarks, EquilibriumBenchmarks
from skills.utils.logging import SkillLogger, create_session_id
from skills.utils.validation import DataValidator, ValidationReport, validate_file_exists
from skills.utils.audit import AuditTrail, hash_file


@dataclass
class BenchmarkResult:
    """Result of property benchmark analysis."""
    property_name: str
    submarket: str
    session_id: str

    # Summary metrics
    total_units: int = 0
    occupied_units: int = 0
    avg_market_rent: float = 0.0
    avg_inplace_rent: float = 0.0

    # Benchmark gaps
    gap_vs_3star: float = 0.0
    gap_vs_45star: float = 0.0
    gap_vs_hp_equilibrium: float = 0.0
    gap_vs_income_equilibrium: float = 0.0

    # Position distribution
    premium_units: int = 0
    competitive_units: int = 0
    slight_discount_units: int = 0
    deep_discount_units: int = 0

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
            "total_units": self.total_units,
            "occupied_units": self.occupied_units,
            "avg_market_rent": self.avg_market_rent,
            "avg_inplace_rent": self.avg_inplace_rent,
            "gap_vs_3star": self.gap_vs_3star,
            "gap_vs_45star": self.gap_vs_45star,
            "gap_vs_hp_equilibrium": self.gap_vs_hp_equilibrium,
            "premium_units": self.premium_units,
            "competitive_units": self.competitive_units,
            "slight_discount_units": self.slight_discount_units,
            "deep_discount_units": self.deep_discount_units
        }


class PropertyBenchmark:
    """
    Property Benchmark Skill

    Benchmarks a property's rents against:
    - 3-Star CoStar asking rents (comp average)
    - 4/5-Star CoStar asking rents (premium positioning)
    - HP Equilibrium (fair value trend)
    - Income-based equilibrium (affordability ceiling)
    """

    SKILL_NAME = "property-benchmark"
    VERSION = "1.0.0"

    # Position classification thresholds
    PREMIUM_THRESHOLD = 15.0      # >= +15% above 3-star
    COMPETITIVE_THRESHOLD = 0.0   # 0% to +15%
    SLIGHT_DISCOUNT_THRESHOLD = -10.0  # -10% to 0%
    # Below -10% = DEEP_DISCOUNT

    # Color palette
    COLORS = {
        'PREMIUM_POSITIONING': '#9b59b6',
        'COMPETITIVE': '#2ecc71',
        'SLIGHT_DISCOUNT': '#f39c12',
        'DEEP_DISCOUNT': '#e74c3c'
    }

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        property_name: str,
        rent_roll_path: Path,
        submarket: str,
        output_dir: Optional[Path] = None,
        costar_period: str = "2025 Q4",
        include_45star: bool = True,
        include_equilibrium: bool = True
    ) -> BenchmarkResult:
        """
        Run property benchmark analysis.

        Args:
            property_name: Name of the property (e.g., "sample_property")
            rent_roll_path: Path to standardized rent roll CSV
            submarket: Submarket name (e.g., "far_north_dallas")
            output_dir: Output directory (default: property_analysis/output/{property})
            costar_period: CoStar data period (default: "2025 Q4")
            include_45star: Include 4/5-star benchmark comparison
            include_equilibrium: Include equilibrium benchmark comparison

        Returns:
            BenchmarkResult with summary metrics and output paths
        """
        # Initialize logging and audit
        session_id = create_session_id()
        logger = SkillLogger(self.SKILL_NAME, session_id)
        audit = AuditTrail(self.SKILL_NAME, session_id)

        logger.print_header(f"PROPERTY BENCHMARK: {property_name.upper()}")
        logger.info(f"Submarket: {submarket}")
        logger.info(f"Session: {session_id}")
        logger.print_separator()

        # Set parameters for logging
        logger.set_parameters(
            property_name=property_name,
            rent_roll_path=str(rent_roll_path),
            submarket=submarket,
            costar_period=costar_period,
            include_45star=include_45star,
            include_equilibrium=include_equilibrium
        )

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.property_output_dir / property_name
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            # 1. Load and validate rent roll
            logger.info("Loading rent roll...")
            rent_roll_path = Path(rent_roll_path)
            validation = validate_file_exists(rent_roll_path, "Rent roll")
            if not validation.is_valid:
                raise FileNotFoundError(validation.message)

            rent_roll = pd.read_csv(rent_roll_path)
            audit.add_data_source("rent_roll", rent_roll_path, rent_roll,
                                  f"{property_name} rent roll")

            validator = DataValidator()
            validation_report = validator.validate_rent_roll(rent_roll)
            if not validation_report.is_valid:
                logger.warning(f"Validation issues: {validation_report.warnings}")

            logger.info(f"Loaded {len(rent_roll)} units")

            # 2. Load CoStar benchmarks
            logger.info(f"Loading CoStar benchmarks ({costar_period})...")
            benchmarks = self.config.load_costar_benchmarks(submarket, costar_period)
            logger.info(f"3-Star: ${benchmarks.star_3.get('overall', 0):,.0f}/unit")
            if include_45star:
                logger.info(f"4/5-Star: ${benchmarks.star_45.get('overall', 0):,.0f}/unit")

            # Record benchmark sources
            costar_file = self.config.get_submarket_file(submarket)
            audit.add_data_source("costar_benchmarks", costar_file,
                                  description=f"CoStar {submarket} {costar_period}")

            # 3. Load equilibrium benchmarks
            equilibrium = self.config.equilibrium
            if include_equilibrium:
                logger.info(f"HP Equilibrium: ${equilibrium.hp_filter:,.0f}")
                logger.info(f"Income Equilibrium: ${equilibrium.income_based:,.0f}")

            # Record model versions
            audit.add_model_version(
                "benchmark_model", self.VERSION,
                parameters={
                    "premium_threshold": self.PREMIUM_THRESHOLD,
                    "competitive_threshold": self.COMPETITIVE_THRESHOLD,
                    "discount_threshold": self.SLIGHT_DISCOUNT_THRESHOLD
                },
                methodology_doc="methodology/04_submarket_scoring.md"
            )

            audit.add_model_version(
                "equilibrium_model", "2.1",
                parameters={
                    "hp_filter_lambda": 1600,
                    "income_rent_ratio": 0.30
                },
                methodology_doc="dislocation_analysis/methodology/01_regime_classification.md"
            )

            # 4. Process rent roll
            logger.print_separator()
            logger.info("Processing rent roll data...")
            processed_df = self._process_rent_roll(rent_roll, benchmarks, equilibrium)
            audit.add_transformation(
                "process_rent_roll",
                "Classify bed types, calculate gaps vs benchmarks",
                input_shape=rent_roll.shape,
                output_shape=processed_df.shape
            )

            # 5. Aggregate by floorplan
            logger.info("Aggregating by floorplan...")
            floorplan_df = self._aggregate_by_floorplan(processed_df, benchmarks)
            audit.add_transformation(
                "aggregate_floorplan",
                "Group by floorplan, compute weighted averages",
                output_shape=floorplan_df.shape
            )

            # 6. Classify positions
            logger.info("Classifying competitive positions...")
            floorplan_df = self._classify_positions(floorplan_df)

            # 7. Calculate summary metrics
            logger.print_separator()
            result = self._calculate_summary(
                property_name, submarket, session_id,
                rent_roll, processed_df, floorplan_df,
                benchmarks, equilibrium
            )

            logger.info(f"Total units analyzed: {result.occupied_units}")
            logger.info(f"Avg market rent: ${result.avg_market_rent:,.0f}")
            logger.info(f"Gap to 3-Star: {result.gap_vs_3star:+.1f}%")
            if include_45star:
                logger.info(f"Gap to 4/5-Star: {result.gap_vs_45star:+.1f}%")

            # 8. Generate outputs
            logger.print_separator()
            logger.info("Generating outputs...")

            # CSV
            csv_path = output_dir / "floorplan_benchmark_comparison.csv"
            floorplan_df.round(2).to_csv(csv_path, index=False)
            logger.add_output(str(csv_path))
            audit.add_output("floorplan_comparison", csv_path,
                            "Floorplan-level benchmark comparison")
            result.csv_path = csv_path
            logger.success(f"Saved: {csv_path.name}")

            # Figures
            fig_paths = self._generate_visualizations(
                floorplan_df, benchmarks, equilibrium,
                property_name, figures_dir
            )
            for fp in fig_paths:
                logger.add_output(str(fp))
                audit.add_output(fp.stem, fp, f"Visualization: {fp.stem}")
                result.figures.append(fp)
                logger.success(f"Saved: {fp.name}")

            # Markdown report
            report_path = output_dir / f"{property_name.upper()}_BENCHMARK_REPORT.md"
            report_content = self._generate_report(
                result, floorplan_df, benchmarks, equilibrium,
                audit, include_45star, include_equilibrium
            )
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.add_output(str(report_path))
            audit.add_output("report", report_path, "Full benchmark analysis report")
            result.report_path = report_path
            logger.success(f"Saved: {report_path.name}")

            # Audit trail
            audit_path = output_dir / f"audit_{session_id}.json"
            audit.save(audit_path)
            result.audit_path = audit_path
            logger.success(f"Saved: {audit_path.name}")

            # Complete logging
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

    def _process_rent_roll(
        self,
        df: pd.DataFrame,
        benchmarks: CoStarBenchmarks,
        equilibrium: EquilibriumBenchmarks
    ) -> pd.DataFrame:
        """Process rent roll and add benchmark comparisons."""
        df = df.copy()

        # Standardize bed type
        if 'bed_type_std' not in df.columns:
            df['bed_type_std'] = df.apply(self._classify_bed_type, axis=1)

        # Filter to occupied with valid data
        mask = (
            (df['status'] == 'Occupied') &
            (df['market_rent'].notna()) &
            (df['market_rent'] > 0)
        )

        # Handle lease_rent column (might be named differently)
        rent_col = None
        for col in ['lease_rent', 'base_rent', 'total_rent']:
            if col in df.columns:
                rent_col = col
                break

        if rent_col:
            mask = mask & (df[rent_col].notna()) & (df[rent_col] > 0)
            df['inplace_rent'] = df[rent_col]
        else:
            df['inplace_rent'] = df['market_rent']

        df = df[mask].copy()

        # Add benchmark values based on bed type
        df['star3_benchmark'] = df['bed_type_std'].map(
            lambda x: benchmarks.star_3.get(x, benchmarks.star_3.get('overall', 0))
        )
        df['star45_benchmark'] = df['bed_type_std'].map(
            lambda x: benchmarks.star_45.get(x, benchmarks.star_45.get('overall', 0))
        )

        # Calculate gaps
        df['gap_vs_3star'] = ((df['market_rent'] / df['star3_benchmark']) - 1) * 100
        df['gap_vs_45star'] = ((df['market_rent'] / df['star45_benchmark']) - 1) * 100
        df['gap_vs_hp'] = ((df['market_rent'] / equilibrium.hp_filter) - 1) * 100
        df['gap_vs_income'] = ((df['market_rent'] / equilibrium.income_based) - 1) * 100

        return df

    def _classify_bed_type(self, row) -> str:
        """Classify bed type from row data."""
        # Check explicit bed_type column first
        if 'bed_type' in row.index and pd.notna(row['bed_type']):
            bt = str(row['bed_type']).upper()
            if 'STUDIO' in bt or bt == '0BR':
                return 'Studio'
            elif '1' in bt:
                return '1BR'
            elif '2' in bt:
                return '2BR'
            elif '3' in bt:
                return '3BR'
            elif '4' in bt:
                return '4BR'

        # Fall back to floorplan code parsing
        fp_col = None
        for col in ['floorplan_code', 'PlanCode', 'unit_type']:
            if col in row.index:
                fp_col = col
                break

        if fp_col and pd.notna(row[fp_col]):
            code = str(row[fp_col]).upper().split('-')[0]
            if code.startswith('E'):
                return 'Studio'
            elif code.startswith('B3'):
                return '3BR'
            elif code.startswith('B2'):
                return '2BR'
            elif code.startswith('B1') or code.startswith('A'):
                return '1BR'

        return '1BR'  # Default

    def _aggregate_by_floorplan(
        self,
        df: pd.DataFrame,
        benchmarks: CoStarBenchmarks
    ) -> pd.DataFrame:
        """Aggregate metrics by floorplan."""
        agg_df = df.groupby(['floorplan_code', 'bed_type_std']).agg({
            'sqft': 'mean',
            'market_rent': 'mean',
            'inplace_rent': 'mean',
            'star3_benchmark': 'first',
            'star45_benchmark': 'first',
            'gap_vs_3star': 'mean',
            'gap_vs_45star': 'mean',
            'gap_vs_hp': 'mean',
            'gap_vs_income': 'mean',
            'unit_id': 'count'
        }).reset_index()

        agg_df.columns = [
            'floorplan', 'bed_type', 'avg_sqft', 'market_rent', 'inplace_rent',
            'star3_benchmark', 'star45_benchmark',
            'gap_vs_3star', 'gap_vs_45star', 'gap_vs_hp', 'gap_vs_income', 'units'
        ]

        return agg_df

    def _classify_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify competitive positioning."""
        def classify(gap):
            if gap >= self.PREMIUM_THRESHOLD:
                return 'PREMIUM_POSITIONING'
            elif gap >= self.COMPETITIVE_THRESHOLD:
                return 'COMPETITIVE'
            elif gap >= self.SLIGHT_DISCOUNT_THRESHOLD:
                return 'SLIGHT_DISCOUNT'
            else:
                return 'DEEP_DISCOUNT'

        df['position_class'] = df['gap_vs_3star'].apply(classify)
        return df

    def _calculate_summary(
        self,
        property_name: str,
        submarket: str,
        session_id: str,
        rent_roll: pd.DataFrame,
        processed: pd.DataFrame,
        floorplan_df: pd.DataFrame,
        benchmarks: CoStarBenchmarks,
        equilibrium: EquilibriumBenchmarks
    ) -> BenchmarkResult:
        """Calculate summary metrics."""
        total_units = len(rent_roll)
        occupied_units = int(floorplan_df['units'].sum())

        # Weighted averages
        weights = floorplan_df['units']
        avg_market = np.average(floorplan_df['market_rent'], weights=weights)
        avg_inplace = np.average(floorplan_df['inplace_rent'], weights=weights)

        # Benchmark gaps
        star3_overall = benchmarks.star_3.get('overall', 0)
        star45_overall = benchmarks.star_45.get('overall', 0)

        gap_3star = ((avg_market / star3_overall) - 1) * 100 if star3_overall else 0
        gap_45star = ((avg_market / star45_overall) - 1) * 100 if star45_overall else 0
        gap_hp = ((avg_market / equilibrium.hp_filter) - 1) * 100
        gap_income = ((avg_market / equilibrium.income_based) - 1) * 100

        # Position distribution
        pos_counts = floorplan_df.groupby('position_class')['units'].sum()

        return BenchmarkResult(
            property_name=property_name,
            submarket=submarket,
            session_id=session_id,
            total_units=total_units,
            occupied_units=occupied_units,
            avg_market_rent=avg_market,
            avg_inplace_rent=avg_inplace,
            gap_vs_3star=gap_3star,
            gap_vs_45star=gap_45star,
            gap_vs_hp_equilibrium=gap_hp,
            gap_vs_income_equilibrium=gap_income,
            premium_units=int(pos_counts.get('PREMIUM_POSITIONING', 0)),
            competitive_units=int(pos_counts.get('COMPETITIVE', 0)),
            slight_discount_units=int(pos_counts.get('SLIGHT_DISCOUNT', 0)),
            deep_discount_units=int(pos_counts.get('DEEP_DISCOUNT', 0))
        )

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        benchmarks: CoStarBenchmarks,
        equilibrium: EquilibriumBenchmarks,
        property_name: str,
        output_dir: Path
    ) -> List[Path]:
        """Generate visualization PNGs."""
        paths = []

        # 1. Gap vs 3-Star Waterfall
        fig, ax = plt.subplots(figsize=(14, 10))
        df_sorted = df.sort_values('gap_vs_3star', ascending=True)

        y = np.arange(len(df_sorted))
        colors = [self.COLORS.get(p, '#95a5a6') for p in df_sorted['position_class']]

        ax.barh(y, df_sorted['gap_vs_3star'], color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=1.5)

        for i, (gap, bed) in enumerate(zip(df_sorted['gap_vs_3star'], df_sorted['bed_type'])):
            label_x = gap + (2 if gap >= 0 else -2)
            ha = 'left' if gap >= 0 else 'right'
            ax.text(label_x, i, f'{gap:+.1f}%', va='center', ha=ha, fontsize=8, fontweight='bold')
            ax.text(-40, i, f'[{bed}]', va='center', ha='right', fontsize=7, color='#7f8c8d')

        ax.set_yticks(y)
        ax.set_yticklabels(df_sorted['floorplan'], fontsize=9)
        ax.set_xlabel('Gap vs. 3-Star CoStar Asking Rent (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{property_name.title()}: Market Rent Positioning vs. 3-Star Benchmark\n'
                     'Negative = Below Market | Positive = Premium Pricing',
                     fontsize=12, fontweight='bold')

        ax.axvspan(-50, 0, alpha=0.1, color='red')
        ax.axvspan(0, 50, alpha=0.1, color='green')

        patches = [mpatches.Patch(color=c, label=p.replace('_', ' ').title())
                   for p, c in self.COLORS.items()]
        ax.legend(handles=patches, loc='lower right', fontsize=9)
        ax.set_xlim(-40, 40)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        path = output_dir / 'floorplan_gap_vs_3star.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        # 2. Position Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        pos_units = df.groupby('position_class')['units'].sum()
        pos_order = ['PREMIUM_POSITIONING', 'COMPETITIVE', 'SLIGHT_DISCOUNT', 'DEEP_DISCOUNT']
        pos_units = pos_units.reindex([p for p in pos_order if p in pos_units.index])

        if len(pos_units) > 0:
            colors = [self.COLORS.get(p, '#95a5a6') for p in pos_units.index]
            axes[0].pie(pos_units, labels=[p.replace('_', '\n').title() for p in pos_units.index],
                        autopct='%1.1f%%', colors=colors, explode=[0.02]*len(pos_units),
                        textprops={'fontsize': 9})
            axes[0].set_title('Distribution by Unit Count', fontsize=11, fontweight='bold')

            # Average gaps
            avg_gaps = []
            for pos in pos_order:
                if pos in df['position_class'].values:
                    pos_df = df[df['position_class'] == pos]
                    avg_gap = (pos_df['gap_vs_3star'] * pos_df['units']).sum() / pos_df['units'].sum()
                    avg_gaps.append((pos, avg_gap))

            if avg_gaps:
                positions = [p[0] for p in avg_gaps]
                gaps = [p[1] for p in avg_gaps]
                bar_colors = [self.COLORS[p] for p in positions]

                bars = axes[1].bar([p.replace('_', '\n').title() for p in positions], gaps,
                                   color=bar_colors, edgecolor='black')
                axes[1].axhline(y=0, color='black', linewidth=1)
                axes[1].set_ylabel('Avg Gap vs. 3-Star (%)', fontsize=10, fontweight='bold')
                axes[1].set_title('Average Position by Class', fontsize=11, fontweight='bold')
                axes[1].grid(axis='y', alpha=0.3)

        fig.suptitle(f'{property_name.title()}: Competitive Position Distribution',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = output_dir / 'position_class_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

        return paths

    def _generate_report(
        self,
        result: BenchmarkResult,
        floorplan_df: pd.DataFrame,
        benchmarks: CoStarBenchmarks,
        equilibrium: EquilibriumBenchmarks,
        audit: AuditTrail,
        include_45star: bool,
        include_equilibrium: bool
    ) -> str:
        """Generate markdown report."""
        report = f"""# {result.property_name.title()}: Property Benchmark Analysis

**Submarket**: {result.submarket.replace('_', ' ').title()}
**Analysis Date**: {audit.created_at[:10]}
**Session ID**: {result.session_id}
**Skill Version**: {self.VERSION}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Units Analyzed** | {result.occupied_units} |
| **Avg Market Rent** | ${result.avg_market_rent:,.0f} |
| **Avg In-Place Rent** | ${result.avg_inplace_rent:,.0f} |
| **Gap to 3-Star** | **{result.gap_vs_3star:+.1f}%** |
"""

        if include_45star:
            report += f"| **Gap to 4/5-Star** | {result.gap_vs_45star:+.1f}% |\n"

        if include_equilibrium:
            report += f"| **Gap to HP Equilibrium** | {result.gap_vs_hp_equilibrium:+.1f}% |\n"

        report += f"""
### Key Finding

"""
        if result.gap_vs_3star > 10:
            report += f"""> {result.property_name.title()} is positioned **{result.gap_vs_3star:.0f}% above** the 3-Star benchmark, indicating **premium positioning** in the submarket.
"""
        elif result.gap_vs_3star > 0:
            report += f"""> {result.property_name.title()} is positioned **at market** (within {abs(result.gap_vs_3star):.0f}% of 3-Star benchmark).
"""
        else:
            report += f"""> {result.property_name.title()} is positioned **{abs(result.gap_vs_3star):.0f}% below** the 3-Star benchmark, indicating potential pricing upside.
"""

        report += f"""
---

## 1. Benchmark Comparison

### CoStar Benchmarks ({benchmarks.data_vintage})

| Benchmark | Overall | Studio | 1BR | 2BR | 3BR |
|-----------|---------|--------|-----|-----|-----|
| **3-Star** | ${benchmarks.star_3.get('overall', 0):,.0f} | ${benchmarks.star_3.get('Studio', 0):,.0f} | ${benchmarks.star_3.get('1BR', 0):,.0f} | ${benchmarks.star_3.get('2BR', 0):,.0f} | ${benchmarks.star_3.get('3BR', 0):,.0f} |
| **4/5-Star** | ${benchmarks.star_45.get('overall', 0):,.0f} | ${benchmarks.star_45.get('Studio', 0):,.0f} | ${benchmarks.star_45.get('1BR', 0):,.0f} | ${benchmarks.star_45.get('2BR', 0):,.0f} | ${benchmarks.star_45.get('3BR', 0):,.0f} |
"""

        if include_equilibrium:
            report += f"""
### Equilibrium Benchmarks

| Benchmark | Value | Property Gap |
|-----------|-------|--------------|
| HP Filter (Fair Value) | ${equilibrium.hp_filter:,.0f} | {result.gap_vs_hp_equilibrium:+.1f}% |
| Income-Based | ${equilibrium.income_based:,.0f} | {result.gap_vs_income_equilibrium:+.1f}% |
| Replacement Cost | ${equilibrium.replacement_cost:,.0f} | {((result.avg_market_rent / equilibrium.replacement_cost) - 1) * 100:+.1f}% |
"""

        report += f"""
---

## 2. Floor Plan Analysis

| Floor Plan | Bed Type | Units | Sq Ft | Market Rent | In-Place | 3-Star | Gap | Position |
|------------|----------|-------|-------|-------------|----------|--------|-----|----------|
"""

        df_sorted = floorplan_df.sort_values('gap_vs_3star', ascending=False)
        for _, row in df_sorted.iterrows():
            report += f"| {row['floorplan']} | {row['bed_type']} | {int(row['units'])} | {row['avg_sqft']:.0f} | ${row['market_rent']:,.0f} | ${row['inplace_rent']:,.0f} | ${row['star3_benchmark']:,.0f} | {row['gap_vs_3star']:+.1f}% | {row['position_class']} |\n"

        report += f"""
---

## 3. Position Distribution

| Position Class | Units | % of Property | Avg Gap |
|----------------|-------|---------------|---------|
| Premium (>+15%) | {result.premium_units} | {result.premium_units/result.occupied_units*100:.1f}% | Premium pricing |
| Competitive (0-15%) | {result.competitive_units} | {result.competitive_units/result.occupied_units*100:.1f}% | At market |
| Slight Discount (-10-0%) | {result.slight_discount_units} | {result.slight_discount_units/result.occupied_units*100:.1f}% | Minor discount |
| Deep Discount (<-10%) | {result.deep_discount_units} | {result.deep_discount_units/result.occupied_units*100:.1f}% | Pricing opportunity |

---

## 4. Visualizations

- `figures/floorplan_gap_vs_3star.png` - Gap analysis by floor plan
- `figures/position_class_distribution.png` - Position distribution charts

---

{audit.generate_markdown_summary()}

---

*Generated by `/property-benchmark` skill v{self.VERSION}*
"""

        return report

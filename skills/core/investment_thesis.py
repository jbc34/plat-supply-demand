"""
Investment Thesis Skill

Generates a comprehensive investment memo combining all analysis.

Usage:
    from skills import InvestmentThesis

    skill = InvestmentThesis()
    result = skill.run(
        property_name="sample_property",
        rent_roll_path="path/to/rent_roll.csv",
        submarket="far_north_dallas"
    )

Combines:
    - Property benchmarking
    - Rent projections
    - Market analysis
    - Investment scoring

Author: Claude Code
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from skills.utils.config import get_config, SkillConfig
from skills.utils.logging import SkillLogger, create_session_id
from skills.utils.audit import AuditTrail, hash_file

from skills.core.property_benchmark import PropertyBenchmark, BenchmarkResult
from skills.core.rent_projection import RentProjection, RentProjectionResult
from skills.core.market_pulse import MarketPulse, MarketPulseResult


@dataclass
class InvestmentRecommendation:
    """Investment recommendation summary."""
    rating: str  # BUY, HOLD, SELL
    confidence: str  # HIGH, MEDIUM, LOW
    summary: str
    key_positives: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    target_return: float = 0.0  # Expected 3-year IRR


@dataclass
class InvestmentThesisResult:
    """Result of investment thesis generation."""
    property_name: str
    submarket: str
    session_id: str
    analysis_date: str

    # Component results
    benchmark: Optional[BenchmarkResult] = None
    projection: Optional[RentProjectionResult] = None
    market: Optional[MarketPulseResult] = None

    # Recommendation
    recommendation: Optional[InvestmentRecommendation] = None

    # Output files
    report_path: Optional[Path] = None
    audit_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "submarket": self.submarket,
            "session_id": self.session_id,
            "analysis_date": self.analysis_date,
            "recommendation": {
                "rating": self.recommendation.rating if self.recommendation else None,
                "confidence": self.recommendation.confidence if self.recommendation else None,
                "target_return": self.recommendation.target_return if self.recommendation else None
            }
        }


class InvestmentThesis:
    """
    Investment Thesis Skill

    Generates a comprehensive investment memo by:
    1. Running property benchmark analysis
    2. Generating rent projections
    3. Analyzing market conditions
    4. Synthesizing an investment recommendation
    """

    SKILL_NAME = "investment-thesis"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        property_name: str,
        rent_roll_path: Path,
        submarket: str,
        output_dir: Optional[Path] = None,
        costar_period: str = "2025 Q4",
        projection_years: int = 3,
        theta: float = 0.307,
        organic_growth: float = 0.035
    ) -> InvestmentThesisResult:
        """
        Generate comprehensive investment thesis.

        Args:
            property_name: Name of the property
            rent_roll_path: Path to rent roll CSV
            submarket: Submarket name
            output_dir: Output directory
            costar_period: CoStar data period
            projection_years: Years for rent projection
            theta: Convergence speed parameter
            organic_growth: Organic growth rate

        Returns:
            InvestmentThesisResult with full analysis
        """
        # Initialize logging and audit
        session_id = create_session_id()
        logger = SkillLogger(self.SKILL_NAME, session_id)
        audit = AuditTrail(self.SKILL_NAME, session_id)
        analysis_date = datetime.utcnow().strftime("%Y-%m-%d")

        logger.print_header(f"INVESTMENT THESIS: {property_name.upper()}")
        logger.info(f"Submarket: {submarket}")
        logger.info(f"Session: {session_id}")
        logger.print_separator()

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.property_output_dir / property_name / "thesis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize result
        result = InvestmentThesisResult(
            property_name=property_name,
            submarket=submarket,
            session_id=session_id,
            analysis_date=analysis_date
        )

        try:
            # 1. Run Property Benchmark
            logger.print_separator()
            logger.info("STEP 1: Property Benchmark Analysis")
            logger.info("-" * 40)

            benchmark_skill = PropertyBenchmark(self.config)
            benchmark_result = benchmark_skill.run(
                property_name=property_name,
                rent_roll_path=rent_roll_path,
                submarket=submarket,
                costar_period=costar_period
            )
            result.benchmark = benchmark_result

            logger.success(f"Benchmark complete: Gap to 3-Star = {benchmark_result.gap_vs_3star:+.1f}%")

            # 2. Run Rent Projection
            logger.print_separator()
            logger.info("STEP 2: Rent Growth Projections")
            logger.info("-" * 40)

            projection_skill = RentProjection(self.config)
            projection_result = projection_skill.run(
                property_name=property_name,
                current_rent=benchmark_result.avg_market_rent,
                submarket=submarket,
                projection_years=projection_years,
                costar_period=costar_period,
                theta=theta,
                organic_growth=organic_growth
            )
            result.projection = projection_result

            # Find equilibrium scenario
            eq_scenario = next((s for s in projection_result.scenarios
                               if s.name == "Converge to Equilibrium"), None)
            if eq_scenario:
                logger.success(f"Projections complete: {eq_scenario.total_growth:+.1f}% total growth")

            # 3. Run Market Pulse
            logger.print_separator()
            logger.info("STEP 3: Market Analysis")
            logger.info("-" * 40)

            market_skill = MarketPulse(self.config)
            market_result = market_skill.run(
                submarket=submarket,
                costar_period=costar_period
            )
            result.market = market_result

            logger.success(f"Market analysis complete: {market_result.regime}")

            # 4. Generate Investment Recommendation
            logger.print_separator()
            logger.info("STEP 4: Investment Recommendation")
            logger.info("-" * 40)

            recommendation = self._generate_recommendation(
                benchmark_result, projection_result, market_result
            )
            result.recommendation = recommendation

            logger.info(f"Recommendation: {recommendation.rating}")
            logger.info(f"Confidence: {recommendation.confidence}")
            logger.info(f"Target Return: {recommendation.target_return:.1f}%")

            # 5. Generate Comprehensive Report
            logger.print_separator()
            logger.info("Generating Investment Memo...")

            report_path = output_dir / f"{property_name.upper()}_INVESTMENT_THESIS.md"
            report_content = self._generate_report(result, audit)
            with open(report_path, 'w') as f:
                f.write(report_content)
            result.report_path = report_path
            logger.success(f"Saved: {report_path.name}")

            # Audit trail
            audit.add_model_version(
                "investment_thesis", self.VERSION,
                parameters={
                    "projection_years": projection_years,
                    "theta": theta,
                    "organic_growth": organic_growth
                }
            )
            audit_path = output_dir / f"audit_thesis_{session_id}.json"
            audit.save(audit_path)
            result.audit_path = audit_path
            logger.success(f"Saved: {audit_path.name}")

            # Complete
            logger.complete("success")
            logger.save()

            logger.print_separator()
            logger.info(f"Investment Thesis complete. Session: {session_id}")
            logger.info(f"Output: {report_path}")

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.complete("failed")
            logger.save()
            raise

    def _generate_recommendation(
        self,
        benchmark: BenchmarkResult,
        projection: RentProjectionResult,
        market: MarketPulseResult
    ) -> InvestmentRecommendation:
        """Generate investment recommendation based on all analysis."""
        positives = []
        risks = []

        # Scoring factors
        score = 50  # Start neutral

        # 1. Current positioning vs market
        if benchmark.gap_vs_3star < -5:
            score += 15
            positives.append(f"Below-market pricing ({benchmark.gap_vs_3star:+.1f}% vs 3-Star) indicates value opportunity")
        elif benchmark.gap_vs_3star > 5:
            score -= 10
            risks.append(f"Premium pricing ({benchmark.gap_vs_3star:+.1f}% vs 3-Star) limits rent growth potential")
        else:
            positives.append("Rents competitively positioned at market")

        # 2. Gap to equilibrium (upside potential)
        if benchmark.gap_vs_hp_equilibrium < -10:
            score += 20
            positives.append(f"Significant upside to equilibrium ({benchmark.gap_vs_hp_equilibrium:+.1f}%)")
        elif benchmark.gap_vs_hp_equilibrium > 10:
            score -= 15
            risks.append(f"Above equilibrium pricing ({benchmark.gap_vs_hp_equilibrium:+.1f}%) suggests limited appreciation")

        # 3. Rent growth projections
        eq_scenario = next((s for s in projection.scenarios
                           if s.name == "Converge to Equilibrium"), None)
        if eq_scenario:
            if eq_scenario.cagr > 5:
                score += 15
                positives.append(f"Strong projected rent growth ({eq_scenario.cagr:.1f}% CAGR)")
            elif eq_scenario.cagr < 2:
                score -= 10
                risks.append(f"Limited rent growth potential ({eq_scenario.cagr:.1f}% CAGR)")

        # 4. Market regime
        if market.regime in ["EXPANSION", "TIGHTENING"]:
            score += 10
            positives.append(f"Favorable market conditions ({market.regime})")
        elif market.regime in ["CONTRACTION", "SOFTENING"]:
            score -= 15
            risks.append(f"Challenging market conditions ({market.regime})")

        # 5. YoY rent growth
        if market.yoy_rent_growth > 3:
            score += 5
            positives.append(f"Positive market momentum ({market.yoy_rent_growth:+.1f}% YoY)")
        elif market.yoy_rent_growth < 0:
            score -= 10
            risks.append(f"Negative rent growth trend ({market.yoy_rent_growth:+.1f}% YoY)")

        # Determine rating
        if score >= 70:
            rating = "BUY"
            confidence = "HIGH" if score >= 80 else "MEDIUM"
        elif score >= 50:
            rating = "HOLD"
            confidence = "MEDIUM"
        else:
            rating = "SELL"
            confidence = "HIGH" if score <= 30 else "MEDIUM"

        # Generate summary
        if rating == "BUY":
            summary = f"{benchmark.property_name.title()} presents an attractive investment opportunity with below-market rents and significant upside to equilibrium values."
        elif rating == "HOLD":
            summary = f"{benchmark.property_name.title()} is fairly valued at current levels with moderate growth potential."
        else:
            summary = f"{benchmark.property_name.title()} faces headwinds from premium pricing and/or challenging market conditions."

        # Calculate target return (simplified - just use equilibrium scenario CAGR)
        target_return = eq_scenario.cagr if eq_scenario else projection.organic_growth * 100

        return InvestmentRecommendation(
            rating=rating,
            confidence=confidence,
            summary=summary,
            key_positives=positives[:4],  # Top 4
            key_risks=risks[:4],
            target_return=target_return
        )

    def _generate_report(
        self,
        result: InvestmentThesisResult,
        audit: AuditTrail
    ) -> str:
        """Generate comprehensive investment memo."""
        b = result.benchmark
        p = result.projection
        m = result.market
        r = result.recommendation

        # Find scenarios
        eq_scenario = next((s for s in p.scenarios if s.name == "Converge to Equilibrium"), None)
        drift_scenario = next((s for s in p.scenarios if s.name == "Market Drift"), None)

        report = f"""# Investment Thesis: {result.property_name.title()}

**Submarket**: {result.submarket.replace('_', ' ').title()}
**Analysis Date**: {result.analysis_date}
**Session ID**: {result.session_id}
**Version**: {self.VERSION}

---

## Executive Summary

### Recommendation: **{r.rating}** (Confidence: {r.confidence})

{r.summary}

| Metric | Current | Target (3Y) | Upside |
|--------|---------|-------------|--------|
| Avg Market Rent | ${b.avg_market_rent:,.0f} | ${eq_scenario.final_rent:,.0f} | {eq_scenario.total_growth:+.1f}% |
| Gap to Equilibrium | {b.gap_vs_hp_equilibrium:+.1f}% | ~0% | {abs(b.gap_vs_hp_equilibrium):.0f}pp closure |
| **Target IRR** | - | - | **{r.target_return:.1f}%** |

---

## Key Investment Highlights

### Positives
"""
        for pos in r.key_positives:
            report += f"- {pos}\n"

        report += f"""
### Risks
"""
        for risk in r.key_risks:
            report += f"- {risk}\n"

        report += f"""
---

## 1. Property Overview

| Metric | Value |
|--------|-------|
| **Property** | {result.property_name.title()} |
| **Submarket** | {result.submarket.replace('_', ' ').title()} |
| **Total Units** | {b.total_units} |
| **Occupied Units** | {b.occupied_units} |
| **Avg Market Rent** | ${b.avg_market_rent:,.0f} |
| **Avg In-Place Rent** | ${b.avg_inplace_rent:,.0f} |

---

## 2. Market Positioning

### Benchmark Comparison

| Benchmark | Value | Gap |
|-----------|-------|-----|
| 3-Star CoStar | ${result.benchmark.gap_vs_3star:,.0f} | **{b.gap_vs_3star:+.1f}%** |
| 4/5-Star CoStar | - | {b.gap_vs_45star:+.1f}% |
| HP Equilibrium | - | {b.gap_vs_hp_equilibrium:+.1f}% |

### Position Distribution

| Class | Units | % |
|-------|-------|---|
| Premium (>+15%) | {b.premium_units} | {b.premium_units/b.occupied_units*100:.1f}% |
| Competitive (0-15%) | {b.competitive_units} | {b.competitive_units/b.occupied_units*100:.1f}% |
| Slight Discount (-10-0%) | {b.slight_discount_units} | {b.slight_discount_units/b.occupied_units*100:.1f}% |
| Deep Discount (<-10%) | {b.deep_discount_units} | {b.deep_discount_units/b.occupied_units*100:.1f}% |

---

## 3. Rent Growth Projections

### Scenario Analysis ({p.half_life:.1f}-year half-life convergence)

| Scenario | Year 1 | Year 2 | Year 3 | Total | CAGR |
|----------|--------|--------|--------|-------|------|
"""
        for s in p.scenarios:
            report += f"| {s.name} | {s.year1_growth:+.1f}% | {s.year2_growth:+.1f}% | {s.year3_growth:+.1f}% | **{s.total_growth:+.1f}%** | {s.cagr:.1f}% |\n"

        report += f"""
### Base Case: Converge to Equilibrium

Starting rent of ${b.avg_market_rent:,.0f} converges toward HP equilibrium of ${self.config.equilibrium.hp_filter:,.0f} over {p.half_life:.1f} years (half-life).

**Projected rent trajectory:**
- Year 1: ${eq_scenario.projections[12]:,.0f} ({eq_scenario.year1_growth:+.1f}%)
- Year 2: ${eq_scenario.projections[24]:,.0f} ({eq_scenario.year2_growth:+.1f}%)
- Year 3: ${eq_scenario.final_rent:,.0f} ({eq_scenario.year3_growth:+.1f}%)

---

## 4. Market Analysis

### Market Regime: {m.regime}

> {m.regime_description}

| Metric | Current | YoY Change |
|--------|---------|------------|
| Asking Rent | ${m.current_asking_rent:,.0f} | {m.yoy_rent_growth:+.1f}% |
| Vacancy Rate | {m.current_vacancy:.1f}% | {m.yoy_vacancy_change:+.1f} ppts |
| Net Absorption | {m.yoy_absorption:,} units | - |
| Deliveries | {m.yoy_deliveries:,} units | - |

### Quality Tier Rents

| Tier | Asking Rent | vs Property |
|------|-------------|-------------|
| 1/2 Star | ${m.rent_12star:,.0f} | {((b.avg_market_rent/m.rent_12star)-1)*100:+.1f}% |
| 3 Star | ${m.rent_3star:,.0f} | {b.gap_vs_3star:+.1f}% |
| 4/5 Star | ${m.rent_45star:,.0f} | {b.gap_vs_45star:+.1f}% |

---

## 5. Investment Thesis

### Why {r.rating}?

{r.summary}

### Valuation Support

The property is currently priced **{abs(b.gap_vs_hp_equilibrium):.0f}% {'below' if b.gap_vs_hp_equilibrium < 0 else 'above'}** our HP equilibrium fair value estimate of ${self.config.equilibrium.hp_filter:,.0f}. Using our mean reversion model with θ={p.theta} (half-life of {p.half_life:.1f} years), we project:

- **Conservative (Market Drift)**: {drift_scenario.total_growth:+.1f}% total return over 3 years
- **Base Case (Equilibrium)**: {eq_scenario.total_growth:+.1f}% total return over 3 years

### Risk Factors

1. **Market Risk**: {m.regime} conditions may persist longer than modeled
2. **Execution Risk**: Rent increases require operational excellence
3. **Supply Risk**: New deliveries could pressure rents
4. **Interest Rate Risk**: Higher rates may compress cap rates

---

## 6. Appendix: Model Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Convergence Speed (θ) | {p.theta} | Completed cycle analysis |
| Half-Life | {p.half_life:.2f} years | Derived from θ |
| Organic Growth | {p.organic_growth:.1%} | Long-run nominal growth |
| HP Equilibrium | ${self.config.equilibrium.hp_filter:,.0f} | HP filter trend |
| Income Equilibrium | ${self.config.equilibrium.income_based:,.0f} | 30% rent-to-income |

---

{audit.generate_markdown_summary()}

---

*Generated by `/investment-thesis` skill v{self.VERSION}*
"""
        return report

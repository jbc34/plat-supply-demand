"""
Claude Skills for Institutional Multifamily Analysis

A suite of reproducible, auditable analysis tools for institutional investors.

Skills:
    - property_benchmark: Benchmark property vs market & equilibrium
    - rent_projection: Project rent growth with convergence model
    - market_pulse: Quick market state assessment
    - submarket_score: Score & rank submarkets for investment
    - investment_thesis: Generate full investment memo
    - data_refresh: Update data and recalculate benchmarks

Usage:
    from skills import PropertyBenchmark

    skill = PropertyBenchmark()
    result = skill.run(
        property_name="sample_property",
        rent_roll_path="path/to/rent_roll.csv",
        submarket="far_north_dallas"
    )
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from skills.core.property_benchmark import PropertyBenchmark
from skills.core.rent_projection import RentProjection
from skills.core.market_pulse import MarketPulse
from skills.core.submarket_score import SubmarketScore
from skills.core.investment_thesis import InvestmentThesis
from skills.core.data_refresh import DataRefresh
# from skills.core.submarket_score import SubmarketScore
# from skills.core.investment_thesis import InvestmentThesis
# from skills.core.data_refresh import DataRefresh

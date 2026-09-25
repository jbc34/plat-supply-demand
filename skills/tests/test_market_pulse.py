"""
Test script for MarketPulse skill.

Tests the skill with Far North Dallas submarket.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skills import MarketPulse


def test_far_north_dallas():
    """Test MarketPulse with Far North Dallas."""
    print("=" * 70)
    print("TESTING MARKET PULSE SKILL")
    print("=" * 70)

    # Initialize skill
    skill = MarketPulse()

    # Run the skill
    try:
        result = skill.run(
            submarket="far_north_dallas",
            costar_period="2025 Q4",
            lookback_quarters=8
        )

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nSession ID: {result.session_id}")
        print(f"Submarket: {result.submarket}")
        print(f"Period: {result.analysis_period}")

        print(f"\nMarket Regime: {result.regime}")
        print(f"  {result.regime_description}")

        print(f"\nCurrent State:")
        print(f"  Asking Rent (All): ${result.current_asking_rent:,.0f}")
        print(f"  3-Star Rent: ${result.rent_3star:,.0f}")
        print(f"  4/5-Star Rent: ${result.rent_45star:,.0f}")
        print(f"  1/2-Star Rent: ${result.rent_12star:,.0f}")
        print(f"  Vacancy Rate: {result.current_vacancy:.1f}%")

        print(f"\nYoY Trends:")
        print(f"  Rent Growth: {result.yoy_rent_growth:+.1f}%")
        print(f"  Vacancy Change: {result.yoy_vacancy_change:+.1f} ppts")
        print(f"  Net Absorption: {result.yoy_absorption:,} units")
        print(f"  Deliveries: {result.yoy_deliveries:,} units")

        print(f"\nOutputs:")
        print(f"  Report: {result.report_path}")
        print(f"  CSV: {result.csv_path}")
        print(f"  Figures: {len(result.figures)} generated")
        print(f"  Audit: {result.audit_path}")

        print("\n" + "=" * 70)
        print("TEST PASSED!")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_far_north_dallas()

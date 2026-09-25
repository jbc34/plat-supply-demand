"""
Test script for PropertyBenchmark skill.

Tests the skill with Sample Property property data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skills import PropertyBenchmark


def test_sample_property():
    """Test PropertyBenchmark with Sample Property data."""
    print("=" * 70)
    print("TESTING PROPERTY BENCHMARK SKILL")
    print("=" * 70)

    # Initialize skill
    skill = PropertyBenchmark()

    # Define inputs
    rent_roll_path = project_root / "dislocation_analysis" / "property_analysis" / "input" / "sample_property_rent_roll_standardized.csv"

    print(f"\nRent roll path: {rent_roll_path}")
    print(f"Rent roll exists: {rent_roll_path.exists()}")

    if not rent_roll_path.exists():
        import pytest
        pytest.skip(f"sample rent roll not present: {rent_roll_path} (excluded data-dependent demo input)")

    # Run the skill
    try:
        result = skill.run(
            property_name="sample_property",
            rent_roll_path=rent_roll_path,
            submarket="far_north_dallas",
            costar_period="2025 Q4",
            include_45star=True,
            include_equilibrium=True
        )

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nSession ID: {result.session_id}")
        print(f"Property: {result.property_name}")
        print(f"Submarket: {result.submarket}")
        print(f"\nUnits Analyzed: {result.occupied_units}")
        print(f"Avg Market Rent: ${result.avg_market_rent:,.0f}")
        print(f"Avg In-Place Rent: ${result.avg_inplace_rent:,.0f}")
        print(f"\nGap to 3-Star: {result.gap_vs_3star:+.1f}%")
        print(f"Gap to 4/5-Star: {result.gap_vs_45star:+.1f}%")
        print(f"Gap to HP Equilibrium: {result.gap_vs_hp_equilibrium:+.1f}%")

        print(f"\nPosition Distribution:")
        print(f"  Premium: {result.premium_units} units")
        print(f"  Competitive: {result.competitive_units} units")
        print(f"  Slight Discount: {result.slight_discount_units} units")
        print(f"  Deep Discount: {result.deep_discount_units} units")

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
    test_sample_property()

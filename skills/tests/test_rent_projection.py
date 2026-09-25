"""
Test script for RentProjection skill.

Tests the skill with Sample Property property data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skills import RentProjection


def test_sample_property():
    """Test RentProjection with Sample Property data."""
    print("=" * 70)
    print("TESTING RENT PROJECTION SKILL")
    print("=" * 70)

    # Initialize skill
    skill = RentProjection()

    # Sample Property's average market rent from benchmark analysis
    current_rent = 1278

    rent_roll_path = (
        project_root / "dislocation_analysis" / "property_analysis" / "input"
        / "sample_property_rent_roll_standardized.csv"
    )

    print(f"\nProperty: Sample Property")
    print(f"Current rent: ${current_rent:,.0f}")

    if not rent_roll_path.exists():
        import pytest
        pytest.skip(f"sample rent roll not present: {rent_roll_path} (excluded data-dependent demo input)")

    # Run the skill
    try:
        result = skill.run(
            property_name="sample_property",
            current_rent=current_rent,
            submarket="far_north_dallas",
            projection_years=3,
            costar_period="2025 Q4",
            theta=0.307,  # Completed cycles estimate
            organic_growth=0.035,  # 3.5% organic
            include_45star_scenario=True
        )

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nSession ID: {result.session_id}")
        print(f"Property: {result.property_name}")
        print(f"Submarket: {result.submarket}")
        print(f"\nCurrent Rent: ${result.current_rent:,.0f}")
        print(f"Gap to Equilibrium: {result.current_gap_to_equilibrium:+.1f}%")
        print(f"\nConvergence Parameters:")
        print(f"  θ: {result.theta}")
        print(f"  Half-life: {result.half_life:.2f} years")
        print(f"  Organic growth: {result.organic_growth:.1%}")

        print(f"\nScenario Results:")
        for s in result.scenarios:
            print(f"\n  {s.name}:")
            print(f"    Year 1: {s.year1_growth:+.1f}%")
            print(f"    Year 2: {s.year2_growth:+.1f}%")
            print(f"    Year 3: {s.year3_growth:+.1f}%")
            print(f"    Total: {s.total_growth:+.1f}% ({s.cagr:.1f}% CAGR)")
            print(f"    Final Rent: ${s.final_rent:,.0f}")

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

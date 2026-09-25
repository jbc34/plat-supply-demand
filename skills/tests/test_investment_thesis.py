"""
Test script for InvestmentThesis skill.

Tests the skill with Sample Property property data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skills import InvestmentThesis


def test_sample_property():
    """Test InvestmentThesis with Sample Property data."""
    print("=" * 70)
    print("TESTING INVESTMENT THESIS SKILL")
    print("=" * 70)

    # Initialize skill
    skill = InvestmentThesis()

    # Define inputs
    rent_roll_path = project_root / "dislocation_analysis" / "property_analysis" / "input" / "sample_property_rent_roll_standardized.csv"

    print(f"\nProperty: Sample Property")
    print(f"Rent roll: {rent_roll_path.name}")

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
            projection_years=3,
            theta=0.307,
            organic_growth=0.035
        )

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nSession ID: {result.session_id}")
        print(f"Property: {result.property_name}")
        print(f"Submarket: {result.submarket}")

        if result.recommendation:
            r = result.recommendation
            print(f"\n** RECOMMENDATION: {r.rating} **")
            print(f"Confidence: {r.confidence}")
            print(f"Target Return: {r.target_return:.1f}%")
            print(f"\nSummary: {r.summary}")

            print(f"\nKey Positives:")
            for p in r.key_positives:
                print(f"  + {p}")

            print(f"\nKey Risks:")
            for risk in r.key_risks:
                print(f"  - {risk}")

        print(f"\nOutput:")
        print(f"  Report: {result.report_path}")
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

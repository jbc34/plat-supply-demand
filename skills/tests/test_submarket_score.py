"""
Test script for SubmarketScore skill.

Tests the skill with multiple DFW submarkets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from skills import SubmarketScore


def test_dfw_submarkets():
    """Test SubmarketScore with DFW submarkets."""
    print("=" * 70)
    print("TESTING SUBMARKET SCORE SKILL")
    print("=" * 70)

    # Initialize skill
    skill = SubmarketScore()

    # Define submarkets to compare
    submarkets = [
        "far_north_dallas",
        "plano",
        "richardson",
        "frisco",
        "north_dallas",
        "las_colinas",
        "uptown"
    ]

    print(f"\nSubmarkets to analyze: {len(submarkets)}")

    # Run the skill
    try:
        result = skill.run(
            submarkets=submarkets,
            costar_period="2025 Q4",
            equilibrium_rent=1516.0  # HP equilibrium for DFW
        )

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nSession ID: {result.session_id}")
        print(f"Period: {result.analysis_period}")
        print(f"Submarkets: {result.num_submarkets}")

        print(f"\nRankings:")
        for m in result.rankings:
            print(f"  #{m.rank} {m.display_name}: {m.composite_score:.0f} ({m.rating})")
            print(f"      Growth: {m.rent_growth_score:.0f}, Vacancy: {m.vacancy_score:.0f}, "
                  f"S/D: {m.supply_demand_score:.0f}, Dislocation: {m.dislocation_score:.0f}")

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
    test_dfw_submarkets()

#!/usr/bin/env python
"""Generate data catalog from FRED and BLS API metadata."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from supply_demand.data.sources import FREDClient


def main():
    fred = FREDClient()

    # All series we use or plan to use
    fred_series = {
        # Currently active in model
        'MORTGAGE30US': {'category': 'Capital', 'usage': 'active', 'notes': 'Primary K proxy (negative loading)'},
        'DGS10': {'category': 'Capital', 'usage': 'active', 'notes': 'Secondary K proxy (negative loading)'},
        'PERMIT': {'category': 'Supply', 'usage': 'active', 'notes': 'Primary S proxy'},
        'HOUST': {'category': 'Supply', 'usage': 'active', 'notes': 'Secondary S proxy'},
        'WPUSI012011': {'category': 'Supply/Cost', 'usage': 'active', 'notes': 'Construction PPI for equilibrium model'},

        # Available but not currently used
        'FEDFUNDS': {'category': 'Capital', 'usage': 'available', 'notes': 'Potential K proxy'},
        'UNRATE': {'category': 'Demand', 'usage': 'available', 'notes': 'National unemployment'},
        'CPIAUCSL': {'category': 'Inflation', 'usage': 'available', 'notes': 'For real value adjustments'},
        'PAYEMS': {'category': 'Demand', 'usage': 'available', 'notes': 'National employment'},

        # NFCI family (planned for K enhancement)
        'NFCI': {'category': 'Capital', 'usage': 'planned', 'notes': 'Chicago Fed financial conditions - candidate for K'},
        'NFCIRISK': {'category': 'Capital', 'usage': 'planned', 'notes': 'NFCI risk subindex'},
        'NFCICREDIT': {'category': 'Capital', 'usage': 'planned', 'notes': 'NFCI credit subindex'},
        'NFCILEVERAGE': {'category': 'Capital', 'usage': 'planned', 'notes': 'NFCI leverage subindex'},

        # Other potential series
        'DRTSCILM': {'category': 'Capital', 'usage': 'planned', 'notes': 'Bank lending standards survey'},
        'DPRIME': {'category': 'Capital', 'usage': 'available', 'notes': 'Prime rate'},
        'TLRESCONS': {'category': 'Supply', 'usage': 'available', 'notes': 'Residential construction spending'},
        'DSPIC96': {'category': 'Demand', 'usage': 'available', 'notes': 'Real disposable income'},
    }

    catalog = []

    print("Fetching FRED series metadata...")
    for series_id, meta in fred_series.items():
        try:
            info = fred.get_series_info(series_id)
            catalog.append({
                'series_id': series_id,
                'title': info.get('title', ''),
                'source': 'FRED',
                'category': meta['category'],
                'usage': meta['usage'],
                'frequency': info.get('frequency_short', info.get('frequency', '')),
                'units': info.get('units', ''),
                'seasonal_adj': info.get('seasonal_adjustment_short', ''),
                'start_date': info.get('observation_start', ''),
                'end_date': info.get('observation_end', ''),
                'notes': meta.get('notes', ''),
            })
            print(f"  {series_id}: OK")
        except Exception as e:
            print(f"  {series_id}: Error - {e}")

    # Add CoStar series (planned - requires API access)
    costar_series = [
        {
            'series_id': 'costar:net_absorption',
            'title': 'Net Absorption (Units)',
            'source': 'CoStar',
            'category': 'Demand',
            'usage': 'planned',
            'frequency': 'Q',
            'units': 'Units',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
        {
            'series_id': 'costar:occupancy_rate',
            'title': 'Occupancy Rate',
            'source': 'CoStar',
            'category': 'Demand',
            'usage': 'planned',
            'frequency': 'M',
            'units': 'Percent',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
        {
            'series_id': 'costar:effective_rent_growth',
            'title': 'Effective Rent Growth YoY',
            'source': 'CoStar',
            'category': 'Demand',
            'usage': 'planned',
            'frequency': 'M',
            'units': 'Percent',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
        {
            'series_id': 'costar:under_construction',
            'title': 'Units Under Construction',
            'source': 'CoStar',
            'category': 'Supply',
            'usage': 'planned',
            'frequency': 'Q',
            'units': 'Units',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
        {
            'series_id': 'costar:deliveries',
            'title': 'Unit Deliveries',
            'source': 'CoStar',
            'category': 'Supply',
            'usage': 'planned',
            'frequency': 'Q',
            'units': 'Units',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
        {
            'series_id': 'costar:cap_rate',
            'title': 'Market Cap Rate',
            'source': 'CoStar',
            'category': 'Capital',
            'usage': 'planned',
            'frequency': 'Q',
            'units': 'Percent',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
        {
            'series_id': 'costar:sales_volume',
            'title': 'Transaction Volume',
            'source': 'CoStar',
            'category': 'Capital',
            'usage': 'planned',
            'frequency': 'Q',
            'units': 'Dollars',
            'seasonal_adj': 'NSA',
            'start_date': 'varies by market',
            'end_date': 'current',
            'notes': 'Requires CoStar API subscription',
        },
    ]
    catalog.extend(costar_series)

    # Add Census series (planned)
    census_series = [
        {
            'series_id': 'census:net_migration',
            'title': 'Net Domestic Migration',
            'source': 'Census',
            'category': 'Demand',
            'usage': 'available',
            'frequency': 'A',
            'units': 'Persons',
            'seasonal_adj': 'NSA',
            'start_date': '2010',
            'end_date': 'current',
            'notes': 'Annual county-to-county migration flows',
        },
        {
            'series_id': 'census:building_permits',
            'title': 'Building Permits by MSA',
            'source': 'Census',
            'category': 'Supply',
            'usage': 'available',
            'frequency': 'M',
            'units': 'Units',
            'seasonal_adj': 'NSA',
            'start_date': '1990',
            'end_date': 'current',
            'notes': 'MSA-level permits (more granular than FRED PERMIT)',
        },
    ]
    catalog.extend(census_series)

    # Add BLS series (manual since API is different)
    bls_series = [
        {
            'series_id': 'SMS48191000000000001',
            'title': 'Dallas-Fort Worth-Arlington, TX Employment',
            'source': 'BLS',
            'category': 'Demand',
            'usage': 'active',
            'frequency': 'M',
            'units': 'Thousands of Persons',
            'seasonal_adj': 'SA',
            'start_date': '1990-01-01',
            'end_date': 'current',
            'notes': 'Primary D proxy for Dallas market',
        },
        {
            'series_id': 'SMS48264200000000001',
            'title': 'Houston-The Woodlands-Sugar Land, TX Employment',
            'source': 'BLS',
            'category': 'Demand',
            'usage': 'available',
            'frequency': 'M',
            'units': 'Thousands of Persons',
            'seasonal_adj': 'SA',
            'start_date': '1990-01-01',
            'end_date': 'current',
            'notes': 'D proxy for Houston market',
        },
        {
            'series_id': 'SMS48120400000000001',
            'title': 'Austin-Round Rock-Georgetown, TX Employment',
            'source': 'BLS',
            'category': 'Demand',
            'usage': 'available',
            'frequency': 'M',
            'units': 'Thousands of Persons',
            'seasonal_adj': 'SA',
            'start_date': '1990-01-01',
            'end_date': 'current',
            'notes': 'D proxy for Austin market',
        },
        {
            'series_id': 'SMS48411700000000001',
            'title': 'San Antonio-New Braunfels, TX Employment',
            'source': 'BLS',
            'category': 'Demand',
            'usage': 'available',
            'frequency': 'M',
            'units': 'Thousands of Persons',
            'seasonal_adj': 'SA',
            'start_date': '1990-01-01',
            'end_date': 'current',
            'notes': 'D proxy for San Antonio market',
        },
    ]

    catalog.extend(bls_series)

    # Save JSON
    output_path = project_root / 'research' / 'data_catalog.json'
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    print(f"\nSaved {len(catalog)} series to {output_path}")

    # Generate markdown table
    md_path = project_root / 'research' / 'data_catalog.md'
    with open(md_path, 'w') as f:
        f.write("# Data Catalog\n\n")
        f.write("Reference table of all time series used or planned for the model.\n\n")
        f.write(f"*Auto-generated. Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}*\n\n")
        f.write("---\n\n")

        # Group by usage
        for usage in ['active', 'available', 'planned']:
            f.write(f"## {usage.title()} Series\n\n")
            f.write("| Series ID | Description | Source | Category | Frequency | Start Date | Notes |\n")
            f.write("|-----------|-------------|--------|----------|-----------|------------|-------|\n")

            for item in catalog:
                if item.get('usage') == usage:
                    f.write(f"| `{item['series_id']}` | {item['title'][:40]}{'...' if len(item.get('title','')) > 40 else ''} | {item['source']} | {item['category']} | {item['frequency']} | {item['start_date']} | {item.get('notes', '')} |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("## Usage Legend\n\n")
        f.write("- **Active**: Currently used in the model\n")
        f.write("- **Available**: Data accessible, not yet integrated\n")
        f.write("- **Planned**: Identified for future testing\n")

    print(f"Saved markdown catalog to {md_path}")


if __name__ == '__main__':
    main()

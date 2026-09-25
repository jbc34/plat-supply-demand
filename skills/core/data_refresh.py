"""
Data Refresh Skill

Checks data freshness and provides refresh guidance.

Usage:
    from skills import DataRefresh

    skill = DataRefresh()
    result = skill.run()

Checks:
    - CoStar submarket data freshness
    - FRED economic indicators
    - Equilibrium calculation dates

Author: Claude Code
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from skills.utils.config import get_config, SkillConfig
from skills.utils.logging import SkillLogger, create_session_id
from skills.utils.audit import AuditTrail


@dataclass
class DataSourceStatus:
    """Status of a data source."""
    name: str
    source_type: str  # costar, fred, bls, internal
    file_path: Optional[str] = None
    last_modified: Optional[str] = None
    latest_period: Optional[str] = None
    record_count: int = 0
    is_stale: bool = False
    staleness_days: int = 0
    refresh_command: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "file_path": self.file_path,
            "last_modified": self.last_modified,
            "latest_period": self.latest_period,
            "record_count": self.record_count,
            "is_stale": self.is_stale,
            "staleness_days": self.staleness_days,
            "notes": self.notes
        }


@dataclass
class DataRefreshResult:
    """Result of data refresh check."""
    session_id: str
    check_date: str

    # Data source statuses
    costar_sources: List[DataSourceStatus] = field(default_factory=list)
    fred_sources: List[DataSourceStatus] = field(default_factory=list)
    internal_sources: List[DataSourceStatus] = field(default_factory=list)

    # Summary
    total_sources: int = 0
    stale_sources: int = 0
    fresh_sources: int = 0

    # Output files
    report_path: Optional[Path] = None
    audit_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "check_date": self.check_date,
            "total_sources": self.total_sources,
            "stale_sources": self.stale_sources,
            "fresh_sources": self.fresh_sources,
            "costar_sources": [s.to_dict() for s in self.costar_sources],
            "fred_sources": [s.to_dict() for s in self.fred_sources],
            "internal_sources": [s.to_dict() for s in self.internal_sources]
        }


class DataRefresh:
    """
    Data Refresh Skill

    Checks freshness of all data sources and provides refresh guidance.
    """

    SKILL_NAME = "data-refresh"
    VERSION = "1.0.0"

    # Staleness thresholds (days)
    STALENESS_THRESHOLDS = {
        'costar': 90,    # Quarterly data - stale after 90 days
        'fred': 30,      # Monthly data - stale after 30 days
        'bls': 30,       # Monthly data - stale after 30 days
        'internal': 180  # Internal calculations - refresh semi-annually
    }

    # Key FRED series
    FRED_SERIES = [
        ('DALLCBSA', 'Dallas Employment'),
        ('TXUR', 'Texas Unemployment Rate'),
        ('CUUR0000SA0', 'CPI Urban All Items'),
        ('DGS10', '10-Year Treasury'),
        ('FEDFUNDS', 'Federal Funds Rate')
    ]

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or get_config()

    def run(
        self,
        output_dir: Optional[Path] = None,
        check_submarkets: Optional[List[str]] = None
    ) -> DataRefreshResult:
        """
        Check data freshness across all sources.

        Args:
            output_dir: Output directory for report
            check_submarkets: Specific submarkets to check (default: all)

        Returns:
            DataRefreshResult with status of all data sources
        """
        # Initialize logging and audit
        session_id = create_session_id()
        logger = SkillLogger(self.SKILL_NAME, session_id)
        audit = AuditTrail(self.SKILL_NAME, session_id)
        check_date = datetime.utcnow().strftime("%Y-%m-%d")

        logger.print_header("DATA REFRESH CHECK")
        logger.info(f"Date: {check_date}")
        logger.info(f"Session: {session_id}")
        logger.print_separator()

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.skills_output_dir / "data_refresh"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize result
        result = DataRefreshResult(
            session_id=session_id,
            check_date=check_date
        )

        try:
            # 1. Check CoStar submarket files
            logger.info("Checking CoStar submarket data...")
            result.costar_sources = self._check_costar_sources(check_submarkets)

            stale_costar = sum(1 for s in result.costar_sources if s.is_stale)
            logger.info(f"  Found {len(result.costar_sources)} submarkets, {stale_costar} stale")

            # 2. Check FRED data
            logger.info("Checking FRED economic data...")
            result.fred_sources = self._check_fred_sources()

            stale_fred = sum(1 for s in result.fred_sources if s.is_stale)
            logger.info(f"  Found {len(result.fred_sources)} series, {stale_fred} stale")

            # 3. Check internal calculations
            logger.info("Checking internal calculations...")
            result.internal_sources = self._check_internal_sources()

            stale_internal = sum(1 for s in result.internal_sources if s.is_stale)
            logger.info(f"  Found {len(result.internal_sources)} sources, {stale_internal} stale")

            # 4. Calculate summary
            all_sources = (result.costar_sources + result.fred_sources +
                          result.internal_sources)
            result.total_sources = len(all_sources)
            result.stale_sources = sum(1 for s in all_sources if s.is_stale)
            result.fresh_sources = result.total_sources - result.stale_sources

            # 5. Generate outputs
            logger.print_separator()
            logger.info("Generating report...")

            # Markdown report
            report_path = output_dir / f"DATA_FRESHNESS_{check_date}.md"
            report_content = self._generate_report(result)
            with open(report_path, 'w') as f:
                f.write(report_content)
            result.report_path = report_path
            logger.success(f"Saved: {report_path.name}")

            # Audit trail
            audit_path = output_dir / f"audit_{session_id}.json"
            audit.save(audit_path)
            result.audit_path = audit_path
            logger.success(f"Saved: {audit_path.name}")

            # Summary
            logger.print_separator()
            if result.stale_sources > 0:
                logger.warning(f"DATA REFRESH NEEDED: {result.stale_sources} sources are stale")
            else:
                logger.success("All data sources are fresh")

            logger.complete("success")
            logger.save()

            return result

        except Exception as e:
            logger.error(f"Check failed: {str(e)}")
            logger.complete("failed")
            logger.save()
            raise

    def _check_costar_sources(
        self,
        submarkets: Optional[List[str]] = None
    ) -> List[DataSourceStatus]:
        """Check CoStar submarket files."""
        sources = []
        costar_dir = self.config.costar_with_stars

        if not costar_dir.exists():
            return sources

        # Get all parquet files or specific submarkets
        if submarkets:
            files = [self.config.get_submarket_file(s) for s in submarkets]
            files = [f for f in files if f.exists()]
        else:
            files = list(costar_dir.glob("*.parquet"))
            files = [f for f in files if not f.name.startswith('_')]

        for file_path in files:
            try:
                # Get file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                days_old = (datetime.now() - mtime).days

                # Read file to get latest period
                df = pd.read_parquet(file_path)
                latest_period = df['date'].max() if 'date' in df.columns else "Unknown"

                # Parse period to check staleness
                is_stale = days_old > self.STALENESS_THRESHOLDS['costar']

                # Extract submarket name
                name = file_path.stem.replace('dallasfort_worth_tx_usa_', '').replace('_', ' ').title()

                sources.append(DataSourceStatus(
                    name=name,
                    source_type='costar',
                    file_path=str(file_path),
                    last_modified=mtime.strftime("%Y-%m-%d"),
                    latest_period=latest_period,
                    record_count=len(df),
                    is_stale=is_stale,
                    staleness_days=days_old,
                    refresh_command=f"# Update from CoStar export for {name}",
                    notes=f"Latest data: {latest_period}"
                ))
            except Exception as e:
                sources.append(DataSourceStatus(
                    name=file_path.stem,
                    source_type='costar',
                    file_path=str(file_path),
                    is_stale=True,
                    notes=f"Error reading file: {str(e)}"
                ))

        return sorted(sources, key=lambda x: x.staleness_days, reverse=True)

    def _check_fred_sources(self) -> List[DataSourceStatus]:
        """Check FRED economic data files."""
        sources = []
        fred_dir = self.config.data_dir / "fred"

        for series_id, series_name in self.FRED_SERIES:
            file_path = fred_dir / f"{series_id.lower()}.csv"

            if file_path.exists():
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    days_old = (datetime.now() - mtime).days

                    df = pd.read_csv(file_path)
                    latest_date = df['date'].max() if 'date' in df.columns else "Unknown"

                    sources.append(DataSourceStatus(
                        name=series_name,
                        source_type='fred',
                        file_path=str(file_path),
                        last_modified=mtime.strftime("%Y-%m-%d"),
                        latest_period=latest_date,
                        record_count=len(df),
                        is_stale=days_old > self.STALENESS_THRESHOLDS['fred'],
                        staleness_days=days_old,
                        refresh_command=f"python3 data/fred/scripts/fetch_fred.py --series {series_id}",
                        notes=f"FRED Series: {series_id}"
                    ))
                except Exception as e:
                    sources.append(DataSourceStatus(
                        name=series_name,
                        source_type='fred',
                        is_stale=True,
                        notes=f"Error: {str(e)}"
                    ))
            else:
                sources.append(DataSourceStatus(
                    name=series_name,
                    source_type='fred',
                    is_stale=True,
                    staleness_days=999,
                    refresh_command=f"python3 data/fred/scripts/fetch_fred.py --series {series_id}",
                    notes=f"File not found: {file_path}"
                ))

        return sources

    def _check_internal_sources(self) -> List[DataSourceStatus]:
        """Check internal calculation files."""
        sources = []

        # Check equilibrium calculations
        eq_file = self.config.results_dir / "equilibrium_estimates.json"
        if eq_file.exists():
            mtime = datetime.fromtimestamp(eq_file.stat().st_mtime)
            days_old = (datetime.now() - mtime).days

            sources.append(DataSourceStatus(
                name="Equilibrium Estimates",
                source_type='internal',
                file_path=str(eq_file),
                last_modified=mtime.strftime("%Y-%m-%d"),
                is_stale=days_old > self.STALENESS_THRESHOLDS['internal'],
                staleness_days=days_old,
                refresh_command="# Recalculate HP filter on updated rent series",
                notes="HP filter, income-based, replacement cost benchmarks"
            ))
        else:
            sources.append(DataSourceStatus(
                name="Equilibrium Estimates",
                source_type='internal',
                is_stale=True,
                staleness_days=999,
                notes="Using default config values - consider updating"
            ))

        # Check convergence parameters
        conv_file = self.config.results_dir / "convergence_estimates.json"
        if conv_file.exists():
            mtime = datetime.fromtimestamp(conv_file.stat().st_mtime)
            days_old = (datetime.now() - mtime).days

            sources.append(DataSourceStatus(
                name="Convergence Parameters",
                source_type='internal',
                file_path=str(conv_file),
                last_modified=mtime.strftime("%Y-%m-%d"),
                is_stale=days_old > self.STALENESS_THRESHOLDS['internal'],
                staleness_days=days_old,
                notes=f"θ, half-life, organic growth estimates"
            ))
        else:
            sources.append(DataSourceStatus(
                name="Convergence Parameters",
                source_type='internal',
                is_stale=False,  # Hardcoded params are stable
                staleness_days=0,
                notes="Using validated config defaults (θ=0.307)"
            ))

        return sources

    def _generate_report(self, result: DataRefreshResult) -> str:
        """Generate markdown report."""
        report = f"""# Data Freshness Report

**Check Date**: {result.check_date}
**Session ID**: {result.session_id}
**Skill Version**: {self.VERSION}

---

## Summary

| Category | Total | Fresh | Stale |
|----------|-------|-------|-------|
| CoStar Submarkets | {len(result.costar_sources)} | {sum(1 for s in result.costar_sources if not s.is_stale)} | {sum(1 for s in result.costar_sources if s.is_stale)} |
| FRED Economic | {len(result.fred_sources)} | {sum(1 for s in result.fred_sources if not s.is_stale)} | {sum(1 for s in result.fred_sources if s.is_stale)} |
| Internal | {len(result.internal_sources)} | {sum(1 for s in result.internal_sources if not s.is_stale)} | {sum(1 for s in result.internal_sources if s.is_stale)} |
| **Total** | **{result.total_sources}** | **{result.fresh_sources}** | **{result.stale_sources}** |

---

## Action Required

"""
        if result.stale_sources == 0:
            report += "> All data sources are fresh. No action required.\n"
        else:
            report += f"> **{result.stale_sources} data sources need refresh.**\n\n"

            # List stale sources
            stale = [s for s in (result.costar_sources + result.fred_sources +
                                result.internal_sources) if s.is_stale]
            stale = sorted(stale, key=lambda x: x.staleness_days, reverse=True)

            report += "### Priority Refresh List\n\n"
            report += "| Source | Type | Days Old | Action |\n"
            report += "|--------|------|----------|--------|\n"

            for s in stale[:10]:
                report += f"| {s.name} | {s.source_type} | {s.staleness_days} | {s.notes} |\n"

        report += f"""
---

## CoStar Submarket Data

| Submarket | Latest Period | Last Modified | Days Old | Status |
|-----------|---------------|---------------|----------|--------|
"""

        for s in result.costar_sources:
            status = "STALE" if s.is_stale else "Fresh"
            report += f"| {s.name} | {s.latest_period or 'N/A'} | {s.last_modified or 'N/A'} | {s.staleness_days} | {status} |\n"

        report += f"""
---

## FRED Economic Data

| Series | Latest Date | Last Modified | Days Old | Status |
|--------|-------------|---------------|----------|--------|
"""

        for s in result.fred_sources:
            status = "STALE" if s.is_stale else "Fresh"
            report += f"| {s.name} | {s.latest_period or 'N/A'} | {s.last_modified or 'N/A'} | {s.staleness_days} | {status} |\n"

        report += f"""
---

## Internal Calculations

| Source | Last Modified | Days Old | Status | Notes |
|--------|---------------|----------|--------|-------|
"""

        for s in result.internal_sources:
            status = "STALE" if s.is_stale else "Fresh"
            report += f"| {s.name} | {s.last_modified or 'N/A'} | {s.staleness_days} | {status} | {s.notes} |\n"

        report += f"""
---

## Refresh Commands

### CoStar Data
```bash
# Export new data from CoStar and run:
python3 data/costar/scripts/update_parquet_with_star_ratings.py
```

### FRED Data
```bash
# Fetch latest FRED data:
python3 data/fred/scripts/fetch_all.py
```

### Equilibrium Recalculation
```bash
# After updating rent data:
python3 supply_demand/scripts/calculate_equilibrium.py
```

---

*Generated by `/data-refresh` skill v{self.VERSION}*
"""

        return report

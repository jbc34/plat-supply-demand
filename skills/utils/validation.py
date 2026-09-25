"""
Data validation utilities for skills.

Ensures data quality before analysis execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    check_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅" if self.is_valid else "❌"
        return f"{status} {self.check_name}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return all(r.is_valid for r in self.results)

    @property
    def warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.is_valid]

    def add(self, result: ValidationResult):
        self.results.append(result)

    def __str__(self) -> str:
        lines = ["Data Validation Report", "-" * 40]
        for r in self.results:
            lines.append(str(r))
        lines.append("-" * 40)
        status = "PASSED" if self.is_valid else "FAILED"
        lines.append(f"Overall: {status}")
        return "\n".join(lines)


class DataValidator:
    """Validates data for skill execution."""

    # Standard rent roll columns
    RENT_ROLL_REQUIRED = [
        'unit_id', 'floorplan_code', 'bed_type', 'sqft',
        'market_rent', 'status'
    ]

    RENT_ROLL_RECOMMENDED = [
        'lease_rent', 'move_in_date', 'lease_end',
        'base_rent', 'total_rent'
    ]

    def __init__(self):
        self.report = ValidationReport()

    def validate_rent_roll(self, df: pd.DataFrame) -> ValidationReport:
        """Validate a rent roll DataFrame."""
        self.report = ValidationReport()

        # Check required columns
        self._check_required_columns(df, self.RENT_ROLL_REQUIRED)

        # Check minimum rows
        self._check_min_rows(df, min_rows=10)

        # Check data types and ranges
        if 'market_rent' in df.columns:
            self._check_numeric_range(df, 'market_rent', min_val=100, max_val=10000)

        if 'sqft' in df.columns:
            self._check_numeric_range(df, 'sqft', min_val=100, max_val=5000)

        # Check null percentages
        self._check_null_percentage(df, self.RENT_ROLL_REQUIRED, max_pct=0.20)

        # Check for valid bed types
        if 'bed_type' in df.columns:
            self._check_valid_values(
                df, 'bed_type',
                valid_values=['Studio', '1BR', '2BR', '3BR', '4BR'],
                allow_null=True
            )

        # Check for valid status
        if 'status' in df.columns:
            self._check_valid_values(
                df, 'status',
                valid_values=['Occupied', 'Vacant', 'Model', 'Down', 'Notice'],
                allow_null=False
            )

        return self.report

    def validate_costar_benchmarks(self, df: pd.DataFrame,
                                   required_period: str = "2025 Q4") -> ValidationReport:
        """Validate CoStar benchmark data."""
        self.report = ValidationReport()

        # Check required columns
        required = ['date', 'star_rating', 'asking_rent_unit']
        self._check_required_columns(df, required)

        # Check required period exists
        if 'date' in df.columns:
            periods = df['date'].unique()
            has_period = required_period in periods
            self.report.add(ValidationResult(
                is_valid=has_period,
                check_name="Required period",
                message=f"Period {required_period} {'found' if has_period else 'NOT found'}",
                details={"available_periods": list(periods[-5:])}
            ))

        # Check star ratings
        if 'star_rating' in df.columns:
            required_stars = ['3 Star', '4 & 5 Star']
            available_stars = df['star_rating'].unique()
            has_stars = all(s in available_stars for s in required_stars)
            self.report.add(ValidationResult(
                is_valid=has_stars,
                check_name="Star ratings",
                message=f"Required ratings {'found' if has_stars else 'missing'}",
                details={"available": list(available_stars), "required": required_stars}
            ))

        return self.report

    def _check_required_columns(self, df: pd.DataFrame, required: List[str]):
        """Check that required columns exist."""
        missing = [c for c in required if c not in df.columns]
        self.report.add(ValidationResult(
            is_valid=len(missing) == 0,
            check_name="Required columns",
            message=f"Missing: {missing}" if missing else "All present",
            details={"missing": missing, "available": list(df.columns)}
        ))

    def _check_min_rows(self, df: pd.DataFrame, min_rows: int):
        """Check minimum row count."""
        n_rows = len(df)
        self.report.add(ValidationResult(
            is_valid=n_rows >= min_rows,
            check_name="Minimum rows",
            message=f"{n_rows} rows (min: {min_rows})",
            details={"row_count": n_rows, "minimum": min_rows}
        ))

    def _check_numeric_range(self, df: pd.DataFrame, column: str,
                             min_val: float, max_val: float):
        """Check numeric column is within expected range."""
        if column not in df.columns:
            return

        values = df[column].dropna()
        if len(values) == 0:
            self.report.add(ValidationResult(
                is_valid=False,
                check_name=f"{column} range",
                message="No non-null values",
                details={}
            ))
            return

        actual_min = values.min()
        actual_max = values.max()
        in_range = (actual_min >= min_val * 0.5) and (actual_max <= max_val * 2)

        self.report.add(ValidationResult(
            is_valid=in_range,
            check_name=f"{column} range",
            message=f"Range: {actual_min:.0f} to {actual_max:.0f}",
            details={
                "actual_min": actual_min,
                "actual_max": actual_max,
                "expected_min": min_val,
                "expected_max": max_val
            }
        ))

    def _check_null_percentage(self, df: pd.DataFrame, columns: List[str],
                               max_pct: float):
        """Check null percentage is below threshold."""
        for col in columns:
            if col not in df.columns:
                continue

            null_pct = df[col].isnull().mean()
            self.report.add(ValidationResult(
                is_valid=null_pct <= max_pct,
                check_name=f"{col} nulls",
                message=f"{null_pct:.1%} null (max: {max_pct:.0%})",
                details={"null_pct": null_pct, "max_allowed": max_pct}
            ))

    def _check_valid_values(self, df: pd.DataFrame, column: str,
                            valid_values: List[str], allow_null: bool = True):
        """Check column contains only valid values."""
        if column not in df.columns:
            return

        values = df[column].dropna().unique() if allow_null else df[column].unique()
        invalid = [v for v in values if v not in valid_values and pd.notna(v)]

        self.report.add(ValidationResult(
            is_valid=len(invalid) == 0,
            check_name=f"{column} values",
            message=f"Invalid: {invalid}" if invalid else "All valid",
            details={"invalid": invalid, "valid": valid_values}
        ))


def validate_file_exists(file_path: Path, description: str = "File") -> ValidationResult:
    """Check that a file exists."""
    exists = file_path.exists()
    return ValidationResult(
        is_valid=exists,
        check_name=f"{description} exists",
        message=str(file_path) if exists else f"NOT FOUND: {file_path}",
        details={"path": str(file_path)}
    )

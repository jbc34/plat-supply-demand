"""
Audit trail utilities for skills.

Provides file hashing and data lineage tracking.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict


def hash_file(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file for data lineage tracking.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # Truncate for readability


def hash_dataframe(df, algorithm: str = "sha256") -> str:
    """
    Compute hash of a DataFrame for data lineage tracking.

    Args:
        df: pandas DataFrame
        algorithm: Hash algorithm

    Returns:
        Hex digest of DataFrame hash
    """
    import pandas as pd
    hasher = hashlib.new(algorithm)
    # Hash the pickled representation
    hasher.update(df.to_json().encode())
    return hasher.hexdigest()[:16]


@dataclass
class DataSource:
    """Record of a data source used in analysis."""
    name: str
    path: str
    file_hash: str
    loaded_at: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    date_range: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelVersion:
    """Record of a model/methodology version."""
    name: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    methodology_doc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputArtifact:
    """Record of an output file generated."""
    name: str
    path: str
    file_hash: str
    created_at: str
    file_type: str
    size_bytes: int
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditTrail:
    """
    Comprehensive audit trail for skill executions.

    Tracks all inputs, transformations, and outputs for reproducibility.
    """

    def __init__(self, skill_name: str, session_id: str):
        self.skill_name = skill_name
        self.session_id = session_id
        self.created_at = datetime.utcnow().isoformat() + "Z"

        self.data_sources: List[DataSource] = []
        self.model_versions: List[ModelVersion] = []
        self.outputs: List[OutputArtifact] = []
        self.transformations: List[Dict[str, Any]] = []
        self.assumptions: List[Dict[str, str]] = []

    def add_data_source(self, name: str, path: Path, df=None,
                        description: Optional[str] = None) -> DataSource:
        """Record a data source."""
        import pandas as pd

        source = DataSource(
            name=name,
            path=str(path),
            file_hash=hash_file(path) if path.exists() else "N/A",
            loaded_at=datetime.utcnow().isoformat() + "Z",
            description=description
        )

        if df is not None:
            source.row_count = len(df)
            source.column_count = len(df.columns)
            if 'date' in df.columns:
                source.date_range = f"{df['date'].min()} to {df['date'].max()}"

        self.data_sources.append(source)
        return source

    def add_model_version(self, name: str, version: str,
                          parameters: Optional[Dict] = None,
                          methodology_doc: Optional[str] = None) -> ModelVersion:
        """Record a model version."""
        model = ModelVersion(
            name=name,
            version=version,
            parameters=parameters or {},
            methodology_doc=methodology_doc
        )
        self.model_versions.append(model)
        return model

    def add_output(self, name: str, path: Path,
                   description: Optional[str] = None) -> OutputArtifact:
        """Record an output file."""
        import os

        artifact = OutputArtifact(
            name=name,
            path=str(path),
            file_hash=hash_file(path) if path.exists() else "PENDING",
            created_at=datetime.utcnow().isoformat() + "Z",
            file_type=path.suffix,
            size_bytes=os.path.getsize(path) if path.exists() else 0,
            description=description
        )
        self.outputs.append(artifact)
        return artifact

    def add_transformation(self, name: str, description: str,
                           input_shape: Optional[tuple] = None,
                           output_shape: Optional[tuple] = None):
        """Record a data transformation step."""
        self.transformations.append({
            "name": name,
            "description": description,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    def add_assumption(self, name: str, value: str, rationale: str):
        """Record an analytical assumption."""
        self.assumptions.append({
            "name": name,
            "value": value,
            "rationale": rationale
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_name": self.skill_name,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "data_sources": [s.to_dict() for s in self.data_sources],
            "model_versions": [m.to_dict() for m in self.model_versions],
            "outputs": [o.to_dict() for o in self.outputs],
            "transformations": self.transformations,
            "assumptions": self.assumptions
        }

    def save(self, output_path: Path):
        """Save audit trail to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def generate_markdown_summary(self) -> str:
        """Generate a human-readable audit summary."""
        lines = [
            "## Data Lineage & Audit Trail",
            f"**Session**: {self.session_id}",
            f"**Generated**: {self.created_at}",
            "",
            "### Data Sources",
            "| Source | Path | Hash | Rows | Date Range |",
            "|--------|------|------|------|------------|"
        ]

        for ds in self.data_sources:
            lines.append(
                f"| {ds.name} | `{Path(ds.path).name}` | "
                f"`{ds.file_hash[:8]}` | {ds.row_count or 'N/A'} | "
                f"{ds.date_range or 'N/A'} |"
            )

        lines.extend([
            "",
            "### Models & Parameters",
            "| Model | Version | Key Parameters |",
            "|-------|---------|----------------|"
        ])

        for mv in self.model_versions:
            params_str = ", ".join(f"{k}={v}" for k, v in list(mv.parameters.items())[:3])
            lines.append(f"| {mv.name} | {mv.version} | {params_str} |")

        if self.assumptions:
            lines.extend([
                "",
                "### Key Assumptions",
                "| Assumption | Value | Rationale |",
                "|------------|-------|-----------|"
            ])
            for a in self.assumptions:
                lines.append(f"| {a['name']} | {a['value']} | {a['rationale']} |")

        lines.extend([
            "",
            "### Output Files",
            "| File | Type | Hash |",
            "|------|------|------|"
        ])

        for out in self.outputs:
            lines.append(f"| `{Path(out.path).name}` | {out.file_type} | `{out.file_hash[:8]}` |")

        return "\n".join(lines)

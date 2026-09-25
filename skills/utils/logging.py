"""
Structured logging for skills.

Provides session-based logging with JSON output for audit trails.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import sys


def create_session_id() -> str:
    """Create a unique session identifier."""
    return uuid.uuid4().hex[:12]


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillExecution:
    """Complete record of a skill execution."""
    session_id: str
    skill_name: str
    invoked_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "running"
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillLogger:
    """Logger for skill executions with structured output."""

    def __init__(self, skill_name: str, session_id: Optional[str] = None):
        self.skill_name = skill_name
        self.session_id = session_id or create_session_id()
        self.start_time = datetime.utcnow()

        self.execution = SkillExecution(
            session_id=self.session_id,
            skill_name=skill_name,
            invoked_at=self.start_time.isoformat() + "Z"
        )

        self._console_output = True

    def set_parameters(self, **params):
        """Record execution parameters."""
        self.execution.parameters = params

    def add_data_source(self, file_path: str, file_hash: Optional[str] = None,
                        description: Optional[str] = None):
        """Record a data source used."""
        self.execution.data_sources.append({
            "file": file_path,
            "hash": file_hash,
            "description": description
        })

    def set_model_versions(self, **versions):
        """Record model versions used."""
        self.execution.model_versions = versions

    def add_output(self, file_path: str):
        """Record an output file generated."""
        self.execution.outputs.append(file_path)

    def _log(self, level: str, message: str, data: Optional[Dict] = None):
        """Internal logging method."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level,
            message=message,
            data=data or {}
        )
        self.execution.logs.append(entry.to_dict())

        if self._console_output:
            prefix = {
                "INFO": "  ",
                "WARNING": "⚠️ ",
                "ERROR": "❌ ",
                "SUCCESS": "✅ "
            }.get(level, "  ")
            print(f"{prefix}{message}")

    def info(self, message: str, **data):
        """Log info message."""
        self._log("INFO", message, data)

    def warning(self, message: str, **data):
        """Log warning message."""
        self._log("WARNING", message, data)
        self.execution.warnings.append(message)

    def error(self, message: str, **data):
        """Log error message."""
        self._log("ERROR", message, data)
        self.execution.errors.append(message)

    def success(self, message: str, **data):
        """Log success message."""
        self._log("SUCCESS", message, data)

    def complete(self, status: str = "success"):
        """Mark execution as complete."""
        end_time = datetime.utcnow()
        self.execution.completed_at = end_time.isoformat() + "Z"
        self.execution.duration_seconds = (end_time - self.start_time).total_seconds()
        self.execution.status = status

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save execution log to JSON file."""
        if output_dir is None:
            from skills.utils.config import get_project_root
            output_dir = get_project_root() / "skills" / "logs" / self.skill_name

        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = self.start_time.strftime("%Y-%m-%d")
        filename = f"{date_str}_{self.session_id}.json"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.execution.to_dict(), f, indent=2)

        return output_path

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        return f"""
Skill Execution Summary
=======================
Session: {self.session_id}
Skill: {self.skill_name}
Status: {self.execution.status}
Duration: {self.execution.duration_seconds:.1f}s
Data Sources: {len(self.execution.data_sources)}
Outputs: {len(self.execution.outputs)}
Warnings: {len(self.execution.warnings)}
Errors: {len(self.execution.errors)}
"""

    def print_header(self, title: str):
        """Print a section header."""
        if self._console_output:
            print("=" * 70)
            print(title)
            print("=" * 70)

    def print_separator(self):
        """Print a separator line."""
        if self._console_output:
            print("-" * 70)

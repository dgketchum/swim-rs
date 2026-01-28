"""
Provenance tracking for SWIM data container.

Records all operations performed on the container, including
data ingestion, transformations, and exports.
"""

import getpass
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _get_swim_version() -> str:
    """Get the current swimrs package version."""
    try:
        from swimrs import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def _get_username() -> str:
    """Get current username."""
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


@dataclass
class ProvenanceEvent:
    """A single provenance event recording an operation on the container."""

    id: str
    timestamp: str
    operation: str
    swim_version: str
    user: str
    target: str | None = None
    source: str | None = None
    source_format: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    fields_affected: list[str] = field(default_factory=list)
    date_range: list[str] | None = None
    records_count: int | None = None
    duration_seconds: float | None = None
    success: bool = True
    error_message: str | None = None

    @classmethod
    def create(cls, operation: str, **kwargs) -> "ProvenanceEvent":
        """Create a new provenance event with auto-generated metadata."""
        return cls(
            id=f"evt_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(UTC).isoformat(),
            operation=operation,
            swim_version=_get_swim_version(),
            user=_get_username(),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceEvent":
        """Create from dictionary."""
        return cls(**data)


class ProvenanceLog:
    """
    Manages the provenance log for a SwimContainer.

    Stores events as an append-only log in the container's metadata.
    """

    def __init__(self):
        self.events: list[ProvenanceEvent] = []
        self.container_created_at: str | None = None
        self.container_created_by: str | None = None
        self.schema_version: str = "1.0"

    def add_event(self, event: ProvenanceEvent) -> None:
        """Add an event to the log."""
        self.events.append(event)

    def record(self, operation: str, **kwargs) -> ProvenanceEvent:
        """Create and add a new provenance event."""
        event = ProvenanceEvent.create(operation, **kwargs)
        self.add_event(event)
        return event

    def get_events_for_target(self, target: str) -> list[ProvenanceEvent]:
        """Get all events that affected a specific data path."""
        return [e for e in self.events if e.target == target]

    def get_events_by_operation(self, operation: str) -> list[ProvenanceEvent]:
        """Get all events of a specific operation type."""
        return [e for e in self.events if e.operation == operation]

    def get_latest_event_for_target(self, target: str) -> ProvenanceEvent | None:
        """Get the most recent event for a data path."""
        events = self.get_events_for_target(target)
        if events:
            return events[-1]
        return None

    def get_version_history(self) -> list[str]:
        """Get list of SWIM versions that have modified this container."""
        versions = []
        for event in self.events:
            if event.swim_version not in versions:
                versions.append(event.swim_version)
        return versions

    def to_dict(self) -> dict[str, Any]:
        """Convert entire log to dictionary."""
        return {
            "container_created_at": self.container_created_at,
            "container_created_by": self.container_created_by,
            "schema_version": self.schema_version,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceLog":
        """Create log from dictionary."""
        log = cls()
        log.container_created_at = data.get("container_created_at")
        log.container_created_by = data.get("container_created_by")
        log.schema_version = data.get("schema_version", "1.0")
        log.events = [ProvenanceEvent.from_dict(e) for e in data.get("events", [])]
        return log

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ProvenanceLog":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Generate a human-readable summary of the provenance log."""
        lines = [
            f"Container created: {self.container_created_at or 'unknown'}",
            f"Created by: {self.container_created_by or 'unknown'}",
            f"Schema version: {self.schema_version}",
            f"Total events: {len(self.events)}",
            "",
            "Event history:",
        ]

        for event in self.events[-10:]:  # Show last 10 events
            target_str = f" -> {event.target}" if event.target else ""
            lines.append(f"  [{event.timestamp[:19]}] {event.operation}{target_str}")

        if len(self.events) > 10:
            lines.append(f"  ... and {len(self.events) - 10} more events")

        return "\n".join(lines)


class DatasetProvenance:
    """
    Provenance metadata for a specific dataset within the container.

    Stored in the .zattrs of each Zarr array/group.
    """

    def __init__(self):
        self.created_at: str | None = None
        self.updated_at: str | None = None
        self.swim_version_created: str | None = None
        self.swim_version_updated: str | None = None
        self.source_type: str | None = None
        self.event_ids: list[str] = []
        self.coverage: dict[str, Any] = {}

    def record_creation(self, event_id: str, source_type: str = None):
        """Record initial creation of this dataset."""
        now = datetime.now(UTC).isoformat()
        self.created_at = now
        self.updated_at = now
        self.swim_version_created = _get_swim_version()
        self.swim_version_updated = self.swim_version_created
        self.source_type = source_type
        self.event_ids.append(event_id)

    def record_update(self, event_id: str):
        """Record an update to this dataset."""
        self.updated_at = datetime.now(UTC).isoformat()
        self.swim_version_updated = _get_swim_version()
        self.event_ids.append(event_id)

    def set_coverage(
        self,
        fields_present: int,
        fields_total: int,
        fields_complete: int = None,
        date_range: tuple = None,
        missing_fields: list[str] = None,
    ):
        """Set coverage information for this dataset."""
        self.coverage = {
            "fields_present": fields_present,
            "fields_total": fields_total,
            "fields_complete": fields_complete or fields_present,
            "date_range": list(date_range) if date_range else None,
            "missing_fields": missing_fields or [],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Zarr attrs."""
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "swim_version_created": self.swim_version_created,
            "swim_version_updated": self.swim_version_updated,
            "source_type": self.source_type,
            "event_ids": self.event_ids,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetProvenance":
        """Create from dictionary."""
        prov = cls()
        prov.created_at = data.get("created_at")
        prov.updated_at = data.get("updated_at")
        prov.swim_version_created = data.get("swim_version_created")
        prov.swim_version_updated = data.get("swim_version_updated")
        prov.source_type = data.get("source_type")
        prov.event_ids = data.get("event_ids", [])
        prov.coverage = data.get("coverage", {})
        return prov

"""
Provenance tracking for SWIM data container.

Records all operations performed on the container, including
data ingestion, transformations, and exports.
"""

import getpass
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import json


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
    target: Optional[str] = None
    source: Optional[str] = None
    source_format: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    fields_affected: List[str] = field(default_factory=list)
    date_range: Optional[List[str]] = None
    records_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    @classmethod
    def create(cls, operation: str, **kwargs) -> "ProvenanceEvent":
        """Create a new provenance event with auto-generated metadata."""
        return cls(
            id=f"evt_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            swim_version=_get_swim_version(),
            user=_get_username(),
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceEvent":
        """Create from dictionary."""
        return cls(**data)


class ProvenanceLog:
    """
    Manages the provenance log for a SwimContainer.

    Stores events as an append-only log in the container's metadata.
    """

    def __init__(self):
        self.events: List[ProvenanceEvent] = []
        self.container_created_at: Optional[str] = None
        self.container_created_by: Optional[str] = None
        self.schema_version: str = "1.0"

    def add_event(self, event: ProvenanceEvent) -> None:
        """Add an event to the log."""
        self.events.append(event)

    def record(self, operation: str, **kwargs) -> ProvenanceEvent:
        """Create and add a new provenance event."""
        event = ProvenanceEvent.create(operation, **kwargs)
        self.add_event(event)
        return event

    def get_events_for_target(self, target: str) -> List[ProvenanceEvent]:
        """Get all events that affected a specific data path."""
        return [e for e in self.events if e.target == target]

    def get_events_by_operation(self, operation: str) -> List[ProvenanceEvent]:
        """Get all events of a specific operation type."""
        return [e for e in self.events if e.operation == operation]

    def get_latest_event_for_target(self, target: str) -> Optional[ProvenanceEvent]:
        """Get the most recent event for a data path."""
        events = self.get_events_for_target(target)
        if events:
            return events[-1]
        return None

    def get_version_history(self) -> List[str]:
        """Get list of SWIM versions that have modified this container."""
        versions = []
        for event in self.events:
            if event.swim_version not in versions:
                versions.append(event.swim_version)
        return versions

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire log to dictionary."""
        return {
            "container_created_at": self.container_created_at,
            "container_created_by": self.container_created_by,
            "schema_version": self.schema_version,
            "events": [e.to_dict() for e in self.events]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceLog":
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
            lines.append(
                f"  [{event.timestamp[:19]}] {event.operation}{target_str}"
            )

        if len(self.events) > 10:
            lines.append(f"  ... and {len(self.events) - 10} more events")

        return "\n".join(lines)


class DatasetProvenance:
    """
    Provenance metadata for a specific dataset within the container.

    Stored in the .zattrs of each Zarr array/group.
    """

    def __init__(self):
        self.created_at: Optional[str] = None
        self.updated_at: Optional[str] = None
        self.swim_version_created: Optional[str] = None
        self.swim_version_updated: Optional[str] = None
        self.source_type: Optional[str] = None
        self.event_ids: List[str] = []
        self.coverage: Dict[str, Any] = {}

    def record_creation(self, event_id: str, source_type: str = None):
        """Record initial creation of this dataset."""
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = now
        self.updated_at = now
        self.swim_version_created = _get_swim_version()
        self.swim_version_updated = self.swim_version_created
        self.source_type = source_type
        self.event_ids.append(event_id)

    def record_update(self, event_id: str):
        """Record an update to this dataset."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.swim_version_updated = _get_swim_version()
        self.event_ids.append(event_id)

    def set_coverage(self, fields_present: int, fields_total: int,
                     fields_complete: int = None, date_range: tuple = None,
                     missing_fields: List[str] = None):
        """Set coverage information for this dataset."""
        self.coverage = {
            "fields_present": fields_present,
            "fields_total": fields_total,
            "fields_complete": fields_complete or fields_present,
            "date_range": list(date_range) if date_range else None,
            "missing_fields": missing_fields or [],
        }

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetProvenance":
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

"""Tests for swimrs.container.provenance module.

Tests cover:
- ProvenanceLog queries: get_events_for_target, get_events_by_operation,
  get_latest_event_for_target, get_version_history
- Serialization round-trip: to_dict -> from_dict
- ProvenanceEvent round-trip
- summary(): truncation at 10 events, empty log
"""

import pytest

from swimrs.container.provenance import ProvenanceEvent, ProvenanceLog


@pytest.fixture
def populated_log():
    """Create a log with multiple events."""
    log = ProvenanceLog()
    log.container_created_at = "2024-01-01T00:00:00"
    log.container_created_by = "test_user"
    log.record("ingest", target="remote_sensing/ndvi/landsat/irr", records_count=100)
    log.record("ingest", target="remote_sensing/ndvi/landsat/irr", records_count=200)
    log.record("ingest", target="meteorology/gridmet/eto", records_count=300)
    log.record("compute", target="derived/dynamics/ke_max", records_count=50)
    return log


class TestProvenanceLogQueries:
    """Tests for ProvenanceLog query methods."""

    def test_get_events_for_target(self, populated_log):
        """get_events_for_target returns matching events."""
        events = populated_log.get_events_for_target("remote_sensing/ndvi/landsat/irr")
        assert len(events) == 2
        assert all(e.target == "remote_sensing/ndvi/landsat/irr" for e in events)

    def test_get_events_for_target_nonexistent(self, populated_log):
        """get_events_for_target returns empty for unknown target."""
        events = populated_log.get_events_for_target("nonexistent/path")
        assert events == []

    def test_get_events_by_operation(self, populated_log):
        """get_events_by_operation returns matching events."""
        events = populated_log.get_events_by_operation("ingest")
        assert len(events) == 3

    def test_get_events_by_operation_compute(self, populated_log):
        """get_events_by_operation returns compute events."""
        events = populated_log.get_events_by_operation("compute")
        assert len(events) == 1

    def test_get_latest_event_for_target(self, populated_log):
        """get_latest_event_for_target returns the most recent event."""
        event = populated_log.get_latest_event_for_target("remote_sensing/ndvi/landsat/irr")
        assert event is not None
        assert event.records_count == 200  # The second ingest

    def test_get_latest_event_for_target_empty(self, populated_log):
        """get_latest_event_for_target returns None for unknown target."""
        event = populated_log.get_latest_event_for_target("nonexistent")
        assert event is None

    def test_get_version_history_deduplicates(self):
        """get_version_history returns unique versions in order."""
        log = ProvenanceLog()
        # Record events with same version
        log.record("ingest", target="a")
        log.record("ingest", target="b")
        versions = log.get_version_history()
        assert len(versions) == len(set(versions))


class TestProvenanceSerialization:
    """Tests for serialization round-trips."""

    def test_log_round_trip(self, populated_log):
        """to_dict -> from_dict preserves log structure."""
        d = populated_log.to_dict()
        restored = ProvenanceLog.from_dict(d)

        assert restored.container_created_at == populated_log.container_created_at
        assert restored.container_created_by == populated_log.container_created_by
        assert restored.schema_version == populated_log.schema_version
        assert len(restored.events) == len(populated_log.events)

    def test_event_round_trip(self):
        """ProvenanceEvent to_dict -> from_dict preserves all fields."""
        event = ProvenanceEvent.create(
            "ingest",
            target="remote_sensing/ndvi",
            source="/path/to/data",
            params={"instrument": "landsat"},
            fields_affected=["A", "B"],
            records_count=42,
        )
        d = event.to_dict()
        restored = ProvenanceEvent.from_dict(d)

        assert restored.id == event.id
        assert restored.operation == event.operation
        assert restored.target == event.target
        assert restored.source == event.source
        assert restored.params == event.params
        assert restored.fields_affected == event.fields_affected
        assert restored.records_count == event.records_count

    def test_json_round_trip(self, populated_log):
        """to_json -> from_json preserves log."""
        json_str = populated_log.to_json()
        restored = ProvenanceLog.from_json(json_str)
        assert len(restored.events) == len(populated_log.events)


class TestProvenanceSummary:
    """Tests for ProvenanceLog.summary()."""

    def test_summary_empty_log(self):
        """Empty log summary shows 0 events."""
        log = ProvenanceLog()
        s = log.summary()
        assert "Total events: 0" in s

    def test_summary_shows_recent_events(self, populated_log):
        """summary shows recent event operations."""
        s = populated_log.summary()
        assert "ingest" in s
        assert "compute" in s

    def test_summary_truncation_over_10(self):
        """summary truncates at 10 events and shows count."""
        log = ProvenanceLog()
        for i in range(15):
            log.record("ingest", target=f"path/{i}")

        s = log.summary()
        assert "... and 5 more events" in s

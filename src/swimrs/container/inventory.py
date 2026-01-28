"""
Inventory tracking for SWIM data container.

Provides observability into what data exists in the container,
coverage statistics, and readiness checks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from swimrs.container.schema import SwimSchema


class DataStatus(str, Enum):
    """Status of a data path in the container."""
    COMPLETE = "complete"       # All fields have data
    PARTIAL = "partial"         # Some fields have data
    NOT_PRESENT = "not_present" # No data ingested
    ERROR = "error"             # Data exists but has issues


@dataclass
class Coverage:
    """Coverage statistics for a data path."""
    path: str
    status: DataStatus
    fields_present: int
    fields_total: int
    fields_missing: List[str]
    date_range: Optional[tuple] = None
    date_gaps: List[tuple] = None
    event_ids: List[str] = None

    @property
    def percent_complete(self) -> float:
        """Percentage of fields with data."""
        if self.fields_total == 0:
            return 0.0
        return 100.0 * self.fields_present / self.fields_total

    @property
    def status_symbol(self) -> str:
        """Unicode symbol for status display."""
        if self.status == DataStatus.COMPLETE:
            return "\u2713"  # checkmark
        elif self.status == DataStatus.PARTIAL:
            return "\u25d0"  # half circle
        elif self.status == DataStatus.NOT_PRESENT:
            return "\u2717"  # x mark
        else:
            return "!"

    def summary_line(self) -> str:
        """Single-line summary for status display."""
        date_str = ""
        if self.date_range:
            date_str = f"  {self.date_range[0]} to {self.date_range[1]}"

        if self.status == DataStatus.NOT_PRESENT:
            return f"  {self.status_symbol} {self.path}: not ingested"
        else:
            pct = f"{self.percent_complete:.0f}%" if self.percent_complete < 100 else ""
            return f"  {self.status_symbol} {self.path}: {self.fields_present}/{self.fields_total} fields  {pct}{date_str}"


@dataclass
class FieldIssue:
    """A single issue found during field validation."""
    field_uid: str
    issue_type: str
    message: str
    value: Any = None
    threshold: Any = None


@dataclass
class FieldValidationReport:
    """
    Comprehensive report of field-level validation results.

    Provides detailed breakdown of which fields pass/fail validation
    and why, with easy-to-read formatting for observability.
    """
    total_fields: int
    valid_fields: List[str]
    invalid_fields: List[str]
    issues_by_field: Dict[str, List[FieldIssue]]
    issues_by_type: Dict[str, List[str]]  # issue_type -> list of field UIDs
    thresholds: Dict[str, Any]

    @property
    def valid_count(self) -> int:
        return len(self.valid_fields)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_fields)

    @property
    def percent_valid(self) -> float:
        if self.total_fields == 0:
            return 0.0
        return 100.0 * self.valid_count / self.total_fields

    def summary(self) -> str:
        """Generate a concise summary of validation results."""
        lines = [
            "=" * 60,
            "FIELD VALIDATION REPORT",
            "=" * 60,
            f"Total fields:   {self.total_fields}",
            f"Valid fields:   {self.valid_count} ({self.percent_valid:.1f}%)",
            f"Invalid fields: {self.invalid_count}",
            "",
        ]

        if self.issues_by_type:
            lines.append("Issues by type:")
            for issue_type, fields in sorted(self.issues_by_type.items()):
                lines.append(f"  {issue_type}: {len(fields)} fields")

        if self.thresholds:
            lines.append("")
            lines.append("Validation thresholds:")
            for name, value in self.thresholds.items():
                lines.append(f"  {name}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def detailed_report(self, max_fields_per_issue: int = 10) -> str:
        """Generate detailed report showing which fields have which issues."""
        lines = [self.summary(), ""]

        if not self.issues_by_type:
            lines.append("All fields passed validation.")
            return "\n".join(lines)

        lines.append("DETAILED BREAKDOWN BY ISSUE TYPE")
        lines.append("-" * 40)

        for issue_type, fields in sorted(self.issues_by_type.items()):
            lines.append(f"\n{issue_type} ({len(fields)} fields):")

            # Show sample of affected fields
            display_fields = fields[:max_fields_per_issue]
            for uid in display_fields:
                # Get the specific issue details for this field
                field_issues = self.issues_by_field.get(uid, [])
                for issue in field_issues:
                    if issue.issue_type == issue_type:
                        if issue.value is not None:
                            lines.append(f"  - {uid}: {issue.value}")
                        else:
                            lines.append(f"  - {uid}")
                        break

            if len(fields) > max_fields_per_issue:
                lines.append(f"  ... and {len(fields) - max_fields_per_issue} more")

        return "\n".join(lines)

    def field_report(self, field_uid: str) -> str:
        """Generate report for a specific field."""
        if field_uid in self.valid_fields:
            return f"{field_uid}: VALID (passed all checks)"

        issues = self.issues_by_field.get(field_uid, [])
        if not issues:
            return f"{field_uid}: Unknown status"

        lines = [f"{field_uid}: INVALID ({len(issues)} issues)"]
        for issue in issues:
            if issue.value is not None and issue.threshold is not None:
                lines.append(f"  - {issue.issue_type}: {issue.message} (value={issue.value}, threshold={issue.threshold})")
            elif issue.value is not None:
                lines.append(f"  - {issue.issue_type}: {issue.message} (value={issue.value})")
            else:
                lines.append(f"  - {issue.issue_type}: {issue.message}")
        return "\n".join(lines)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert validation results to a pandas DataFrame for analysis."""
        import pandas as pd

        records = []
        for uid in self.valid_fields:
            records.append({
                "field_uid": uid,
                "valid": True,
                "issue_count": 0,
                "issues": "",
            })

        for uid in self.invalid_fields:
            issues = self.issues_by_field.get(uid, [])
            issue_types = [i.issue_type for i in issues]
            records.append({
                "field_uid": uid,
                "valid": False,
                "issue_count": len(issues),
                "issues": ", ".join(issue_types),
            })

        return pd.DataFrame(records)


@dataclass
class ValidationResult:
    """Result of validating readiness for an operation."""
    operation: str
    ready: bool
    missing_data: List[str]
    incomplete_data: List[Coverage]
    warnings: List[str]
    ready_fields: List[str]
    not_ready_fields: List[str]

    def summary(self) -> str:
        """Generate summary of validation result."""
        lines = [f"Validation for: {self.operation}"]

        if self.ready:
            lines.append(f"  Status: READY ({len(self.ready_fields)} fields)")
        else:
            lines.append(f"  Status: NOT READY")

        if self.missing_data:
            lines.append(f"  Missing data:")
            for path in self.missing_data[:5]:
                lines.append(f"    - {path}")
            if len(self.missing_data) > 5:
                lines.append(f"    ... and {len(self.missing_data) - 5} more")

        if self.incomplete_data:
            lines.append(f"  Incomplete data:")
            for cov in self.incomplete_data[:3]:
                lines.append(f"    - {cov.path}: {cov.percent_complete:.0f}% complete")

        if self.warnings:
            lines.append(f"  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")

        if self.not_ready_fields:
            lines.append(f"  Fields not ready: {len(self.not_ready_fields)}")

        return "\n".join(lines)


class Inventory:
    """
    Tracks what data exists in a SwimContainer and provides coverage statistics.
    """

    def __init__(self, zarr_root, field_uids: List[str]):
        """
        Args:
            zarr_root: The root Zarr group of the container
            field_uids: List of all field UIDs in the container
        """
        self._root = zarr_root
        self._field_uids = field_uids
        self._coverage_cache: Dict[str, Coverage] = {}

    @property
    def field_count(self) -> int:
        """Total number of fields in the container."""
        return len(self._field_uids)

    def refresh(self):
        """Clear cached coverage data."""
        self._coverage_cache.clear()

    def get_coverage(self, path: str) -> Coverage:
        """
        Get coverage statistics for a data path.

        Args:
            path: Data path like 'remote_sensing/ndvi/landsat/irr'
        """
        if path in self._coverage_cache:
            return self._coverage_cache[path]

        # Check if path exists in Zarr
        try:
            arr = self._root[path]
        except KeyError:
            coverage = Coverage(
                path=path,
                status=DataStatus.NOT_PRESENT,
                fields_present=0,
                fields_total=self.field_count,
                fields_missing=self._field_uids.copy(),
            )
            self._coverage_cache[path] = coverage
            return coverage

        # Analyze the array for coverage
        import numpy as np

        # Determine which fields have data (not all NaN)
        if arr.ndim == 2:  # (time, field)
            has_data = ~np.all(np.isnan(arr[:]), axis=0)
        elif arr.ndim == 1:  # (field,) for properties
            has_data = ~np.isnan(arr[:])
        else:
            has_data = np.ones(len(self._field_uids), dtype=bool)

        fields_present = int(np.sum(has_data))
        missing_indices = np.where(~has_data)[0]
        fields_missing = [self._field_uids[i] for i in missing_indices if i < len(self._field_uids)]

        # Determine status
        if fields_present == 0:
            status = DataStatus.NOT_PRESENT
        elif fields_present == self.field_count:
            status = DataStatus.COMPLETE
        else:
            status = DataStatus.PARTIAL

        # Get date range if time series
        date_range = None
        if arr.ndim == 2 and "time" in self._root:
            try:
                time_arr = self._root["time/daily"][:]
                date_range = (str(time_arr[0])[:10], str(time_arr[-1])[:10])
            except (KeyError, IndexError):
                pass

        # Get provenance event IDs
        event_ids = arr.attrs.get("event_ids", [])

        coverage = Coverage(
            path=path,
            status=status,
            fields_present=fields_present,
            fields_total=self.field_count,
            fields_missing=fields_missing,
            date_range=date_range,
            event_ids=event_ids,
        )
        self._coverage_cache[path] = coverage
        return coverage

    def list_present_paths(self) -> List[str]:
        """List all data paths that have at least some data."""
        present = []
        for path in SwimSchema.list_all_paths():
            cov = self.get_coverage(path)
            if cov.status != DataStatus.NOT_PRESENT:
                present.append(path)
        return present

    def list_missing_paths(self) -> List[str]:
        """List all schema paths that have no data."""
        missing = []
        for path in SwimSchema.list_all_paths():
            cov = self.get_coverage(path)
            if cov.status == DataStatus.NOT_PRESENT:
                missing.append(path)
        return missing

    def validate_for_calibration(self, model: str = "ssebop", mask: str = "irr",
                                  met_source: str = "gridmet",
                                  snow_source: str = "snodas",
                                  instrument: str = "landsat") -> ValidationResult:
        """Check if container has data required for calibration."""
        required = SwimSchema.required_for_calibration(
            model=model, mask=mask, met_source=met_source,
            snow_source=snow_source, instrument=instrument
        )
        return self._validate_requirements(required, f"calibration ({model}, {met_source})")

    def validate_for_forward_run(self, model: str = "ssebop", mask: str = "irr",
                                  met_source: str = "gridmet",
                                  instrument: str = "landsat") -> ValidationResult:
        """Check if container has data required for forward model run."""
        required = SwimSchema.required_for_forward_run(
            model=model, mask=mask, met_source=met_source, instrument=instrument
        )
        return self._validate_requirements(required, f"forward_run ({model}, {mask}, {met_source})")

    def _validate_requirements(self, required_paths: List[str], operation: str) -> ValidationResult:
        """Validate that required data paths are present and complete."""
        missing_data = []
        incomplete_data = []
        warnings = []

        # Track which fields are ready (have all required data)
        field_ready_count = {uid: 0 for uid in self._field_uids}
        total_requirements = len(required_paths)

        for path in required_paths:
            cov = self.get_coverage(path)

            if cov.status == DataStatus.NOT_PRESENT:
                missing_data.append(path)
            elif cov.status == DataStatus.PARTIAL:
                incomplete_data.append(cov)
                # Track which fields are missing
                for uid in self._field_uids:
                    if uid not in cov.fields_missing:
                        field_ready_count[uid] += 1
            else:
                # Complete - all fields have this data
                for uid in self._field_uids:
                    field_ready_count[uid] += 1

        # A field is ready if it has all required data
        ready_fields = [uid for uid, count in field_ready_count.items()
                       if count == total_requirements - len(missing_data)]
        not_ready_fields = [uid for uid in self._field_uids if uid not in ready_fields]

        # Generate warnings for partial data
        if incomplete_data:
            pct = min(c.percent_complete for c in incomplete_data)
            warnings.append(f"Some datasets are incomplete (minimum {pct:.0f}% coverage)")

        ready = len(missing_data) == 0 and len(ready_fields) > 0

        return ValidationResult(
            operation=operation,
            ready=ready,
            missing_data=missing_data,
            incomplete_data=incomplete_data,
            warnings=warnings,
            ready_fields=ready_fields,
            not_ready_fields=not_ready_fields,
        )

    def suggest_next_steps(self) -> List[str]:
        """Suggest what the user should do next based on current state."""
        suggestions = []

        # Check for missing core data
        has_ndvi = self.get_coverage("remote_sensing/ndvi/landsat/irr").status != DataStatus.NOT_PRESENT
        has_etf = any(
            self.get_coverage(f"remote_sensing/etf/landsat/{m.value}/irr").status != DataStatus.NOT_PRESENT
            for m in SwimSchema.REMOTE_SENSING_STRUCTURE["etf"]["models"]
        )
        has_met_gridmet = self.get_coverage("meteorology/gridmet/eto").status != DataStatus.NOT_PRESENT
        has_met_era5 = self.get_coverage("meteorology/era5/eto").status != DataStatus.NOT_PRESENT
        has_met = has_met_gridmet or has_met_era5
        has_soils = self.get_coverage("properties/soils/awc").status != DataStatus.NOT_PRESENT
        has_irr = self.get_coverage("properties/irrigation/irr").status != DataStatus.NOT_PRESENT
        has_dynamics = self.get_coverage("derived/dynamics/ke_max").status != DataStatus.NOT_PRESENT

        if not has_ndvi:
            suggestions.append("Ingest Landsat NDVI: container.ingest_ee_ndvi(source_dir, instrument='landsat', mask='irr')")

        if not has_etf:
            suggestions.append("Ingest ETF data: container.ingest_ee_etf(source_dir, model='ssebop', mask='irr')")

        if not has_met:
            suggestions.append("Ingest meteorology: container.ingest_gridmet(met_dir) or container.ingest_era5(era5_dir)")

        if not has_soils:
            suggestions.append("Ingest properties: container.ingest_properties(soils_csv='path/to/soils.csv')")

        if not has_irr:
            suggestions.append("Ingest irrigation: container.ingest_properties(irrigation_csv='path/to/irr.csv')")

        if has_ndvi and has_etf and has_met and has_irr and not has_dynamics:
            suggestions.append("Compute dynamics: container.compute_dynamics()")

        if has_dynamics:
            suggestions.append("Export model inputs: container.export_model_inputs(...)")

        if not suggestions:
            suggestions.append("Container appears ready. Run validation: container.validate_for_calibration()")

        return suggestions

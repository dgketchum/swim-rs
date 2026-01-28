"""
Operation metrics and observability for SwimContainer.

Provides:
- OperationMetrics: Dataclass for capturing operation statistics
- OperationContext: Context manager for automatic metric collection
- MetricsCollector: Aggregate metrics across operations

This module enhances the provenance system with performance observability,
making it easier to track and optimize long-running data operations.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swimrs.container.provenance import ProvenanceLog


@dataclass
class OperationMetrics:
    """
    Metrics captured during a container operation.

    Provides detailed performance and quality statistics for:
    - Ingestion operations
    - Compute operations
    - Export operations

    Attributes:
        duration_seconds: Wall clock time for the operation
        records_processed: Number of records/rows processed
        records_failed: Number of records that failed processing
        missing_value_count: Count of NaN/missing values encountered
        fields_processed: Number of fields processed
        fields_skipped: Number of fields skipped
        peak_memory_mb: Peak memory usage during operation (if available)
        warnings: List of warning messages generated
        error: Error message if operation failed
    """

    duration_seconds: float = 0.0
    records_processed: int = 0
    records_failed: int = 0
    missing_value_count: int = 0
    fields_processed: int = 0
    fields_skipped: int = 0
    peak_memory_mb: float | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the operation completed without error."""
        return self.error is None

    @property
    def success_rate(self) -> float:
        """Fraction of records successfully processed."""
        total = self.records_processed + self.records_failed
        if total == 0:
            return 1.0
        return self.records_processed / total

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "missing_value_count": self.missing_value_count,
            "fields_processed": self.fields_processed,
            "fields_skipped": self.fields_skipped,
            "peak_memory_mb": self.peak_memory_mb,
            "warnings": self.warnings,
            "error": self.error,
            "success": self.success,
        }

    def summary(self) -> str:
        """Human-readable summary of metrics."""
        lines = [
            f"Duration: {self.duration_seconds:.2f}s",
            f"Records: {self.records_processed} processed",
        ]
        if self.records_failed:
            lines.append(f"         {self.records_failed} failed")
        if self.fields_processed:
            lines.append(f"Fields: {self.fields_processed} processed")
        if self.fields_skipped:
            lines.append(f"        {self.fields_skipped} skipped")
        if self.missing_value_count:
            lines.append(f"Missing values: {self.missing_value_count}")
        if self.peak_memory_mb:
            lines.append(f"Peak memory: {self.peak_memory_mb:.1f} MB")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class OperationContext:
    """
    Context manager for automatic operation metrics collection.

    Captures timing, memory usage, and error information automatically.
    Integrates with provenance logging to record enriched events.

    Example:
        with OperationContext(provenance, "ingest_ndvi", source="path/to/data") as ctx:
            for file in files:
                # Process file...
                ctx.records_processed += len(data)
                ctx.missing_value_count += nan_count
            ctx.fields_processed = len(fields)

        # Metrics are automatically recorded to provenance on exit
    """

    def __init__(
        self,
        provenance: ProvenanceLog,
        operation: str,
        target: str | None = None,
        source: str | None = None,
        **extra_params: Any,
    ):
        """
        Initialize the operation context.

        Args:
            provenance: ProvenanceLog to record the event to
            operation: Operation name (e.g., "ingest", "compute", "export")
            target: Target data path
            source: Source data location
            **extra_params: Additional parameters to include in provenance
        """
        self.provenance = provenance
        self.operation = operation
        self.target = target
        self.source = source
        self.extra_params = extra_params

        # Metrics to be updated during operation
        self.records_processed: int = 0
        self.records_failed: int = 0
        self.missing_value_count: int = 0
        self.fields_processed: int = 0
        self.fields_skipped: int = 0
        self.fields_affected: list[str] = []
        self.warnings: list[str] = []

        # Internal state
        self._start_time: float | None = None
        self._start_memory: float | None = None
        self._metrics: OperationMetrics | None = None

    def add_warning(self, message: str) -> None:
        """Add a warning message to the operation."""
        self.warnings.append(message)

    def __enter__(self) -> OperationContext:
        """Start timing and memory tracking."""
        self._start_time = time.perf_counter()
        self._start_memory = self._get_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Record metrics and provenance event on exit.

        Captures:
        - Duration
        - Peak memory (if tracemalloc available)
        - Error information (if exception occurred)
        """
        duration = time.perf_counter() - self._start_time

        # Calculate peak memory if available
        peak_memory = None
        if self._start_memory is not None:
            current = self._get_memory_usage()
            if current is not None:
                peak_memory = max(current - self._start_memory, 0)

        # Capture error information
        error = None
        if exc_type is not None:
            error = f"{exc_type.__name__}: {exc_val}"

        # Build metrics
        self._metrics = OperationMetrics(
            duration_seconds=duration,
            records_processed=self.records_processed,
            records_failed=self.records_failed,
            missing_value_count=self.missing_value_count,
            fields_processed=self.fields_processed,
            fields_skipped=self.fields_skipped,
            peak_memory_mb=peak_memory,
            warnings=self.warnings.copy(),
            error=error,
        )

        # Record provenance event with metrics
        self.provenance.record(
            operation=self.operation,
            target=self.target,
            source=self.source,
            params=self.extra_params,
            fields_affected=self.fields_affected if self.fields_affected else None,
            records_count=self.records_processed,
            duration_seconds=duration,
            success=error is None,
            error_message=error,
        )

        # Don't suppress exceptions
        return False

    @property
    def metrics(self) -> OperationMetrics | None:
        """Get the collected metrics (available after context exit)."""
        return self._metrics

    def _get_memory_usage(self) -> float | None:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
        except Exception:
            return None


@contextmanager
def track_operation(
    provenance: ProvenanceLog,
    operation: str,
    target: str | None = None,
    source: str | None = None,
    **params: Any,
):
    """
    Context manager for tracking an operation with provenance.

    This is a convenience function that wraps OperationContext.

    Example:
        with track_operation(prov, "ingest", target="ndvi", source="data/") as ctx:
            # Do work...
            ctx.records_processed = 100

    Args:
        provenance: ProvenanceLog instance
        operation: Operation name
        target: Target path
        source: Source path
        **params: Additional parameters

    Yields:
        OperationContext for updating metrics
    """
    ctx = OperationContext(
        provenance=provenance,
        operation=operation,
        target=target,
        source=source,
        **params,
    )
    with ctx:
        yield ctx


@dataclass
class MetricsSummary:
    """
    Aggregate metrics summary across multiple operations.

    Provides an overview of container operations performance.
    """

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_seconds: float = 0.0
    total_records_processed: int = 0
    total_missing_values: int = 0
    operations_by_type: dict[str, int] = field(default_factory=dict)

    def add_operation(self, operation: str, metrics: OperationMetrics) -> None:
        """Add an operation's metrics to the summary."""
        self.total_operations += 1
        if metrics.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        self.total_duration_seconds += metrics.duration_seconds
        self.total_records_processed += metrics.records_processed
        self.total_missing_values += metrics.missing_value_count

        if operation not in self.operations_by_type:
            self.operations_by_type[operation] = 0
        self.operations_by_type[operation] += 1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Container Metrics Summary",
            "=" * 40,
            f"Total operations: {self.total_operations}",
            f"  Successful: {self.successful_operations}",
            f"  Failed: {self.failed_operations}",
            f"Total duration: {self.total_duration_seconds:.1f}s",
            f"Total records: {self.total_records_processed:,}",
            f"Total missing values: {self.total_missing_values:,}",
            "",
            "Operations by type:",
        ]
        for op_type, count in sorted(self.operations_by_type.items()):
            lines.append(f"  {op_type}: {count}")
        return "\n".join(lines)


class MetricsCollector:
    """
    Collects and aggregates metrics from container operations.

    Can be used to track performance over time and identify bottlenecks.

    Example:
        collector = MetricsCollector()

        with collector.track("ingest", prov, target="ndvi") as ctx:
            # Do work...
            ctx.records_processed = 100

        print(collector.summary())
    """

    def __init__(self):
        self._operations: list[tuple] = []  # (operation, metrics)

    def add(self, operation: str, metrics: OperationMetrics) -> None:
        """Add operation metrics."""
        self._operations.append((operation, metrics))

    @contextmanager
    def track(
        self,
        operation: str,
        provenance: ProvenanceLog,
        target: str | None = None,
        source: str | None = None,
        **params: Any,
    ):
        """
        Track an operation and add its metrics to the collector.

        Args:
            operation: Operation name
            provenance: ProvenanceLog instance
            target: Target path
            source: Source path
            **params: Additional parameters

        Yields:
            OperationContext for updating metrics
        """
        ctx = OperationContext(
            provenance=provenance,
            operation=operation,
            target=target,
            source=source,
            **params,
        )
        with ctx:
            yield ctx

        if ctx.metrics is not None:
            self.add(operation, ctx.metrics)

    def get_summary(self) -> MetricsSummary:
        """Get aggregate metrics summary."""
        summary = MetricsSummary()
        for operation, metrics in self._operations:
            summary.add_operation(operation, metrics)
        return summary

    def summary(self) -> str:
        """Human-readable summary string."""
        return self.get_summary().summary()

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._operations.clear()

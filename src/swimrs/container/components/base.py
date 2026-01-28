"""
Base component class for SwimContainer components.

Provides the foundation for component classes (Ingestor, Calculator,
Exporter, Query) that operate on the shared ContainerState.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from swimrs.container.logging import ContainerLogger, get_logger

if TYPE_CHECKING:
    from swimrs.container.state import ContainerState


class Component:
    """
    Base class for container components.

    Components operate on shared ContainerState, providing focused functionality
    while maintaining access to the underlying zarr storage and metadata.

    Attributes:
        _state: The shared ContainerState object
        _container: Optional reference to parent SwimContainer
        _log: Structured logger for this component

    Example:
        class MyComponent(Component):
            def my_method(self, ...):
                # Access zarr root
                arr = self._state.root["path/to/data"]

                # Access provenance
                self._state.provenance.record(...)

                # Use xarray interface
                ds = self._state.get_dataset(...)

                # Structured logging
                self._log.info("operation_complete", records=100)
    """

    def __init__(self, state: ContainerState, container: Any | None = None):
        """
        Initialize the component.

        Args:
            state: ContainerState instance providing access to container data
            container: Optional reference to parent SwimContainer for backward compatibility
        """
        self._state = state
        self._container = container
        self._log = get_logger(self.__class__.__name__.lower())

    @property
    def state(self) -> ContainerState:
        """Access the container state."""
        return self._state

    @property
    def log(self) -> ContainerLogger:
        """Access the component logger."""
        return self._log

    def _ensure_writable(self) -> None:
        """
        Ensure the container is open in writable mode.

        Raises:
            ValueError: If container is read-only
        """
        if not self._state.is_writable:
            raise ValueError(
                "Cannot modify: container opened in read-only mode. "
                "Use mode='r+' or mode='a' to enable writes."
            )

    def _safe_delete_path(self, path: str) -> bool:
        """
        Safely delete a path from the zarr store.

        Handles stores that don't support deletion (like ZipStore) gracefully.
        For ZipStore, deletion is not supported but writing to the same path
        will create a new entry that overrides the old one when reading.

        Args:
            path: The zarr path to delete

        Returns:
            True if deletion succeeded, False if skipped due to store limitation
        """
        if path not in self._state.root:
            return True  # Nothing to delete

        try:
            del self._state.root[path]
            return True
        except NotImplementedError:
            # ZipStore and some other stores don't support deletion
            # Writing new data will override the old entry
            self._log.debug("cannot_delete_path", path=path, reason="store_limitation")
            return False

    @contextmanager
    def _track_operation(
        self,
        operation: str,
        target: str | None = None,
        **params: Any,
    ) -> Generator[dict]:
        """
        Context manager for tracking operations with logging and provenance.

        Args:
            operation: Name of the operation (e.g., "ingest_ndvi", "compute_dynamics")
            target: Target path in the container (e.g., "remote_sensing/ndvi/landsat/irr")
            **params: Additional parameters to log and record in provenance

        Yields:
            A context dict for accumulating operation metrics (records_processed, etc.)

        Example:
            with self._track_operation("ingest_ndvi", target=path, source=str(source_dir)) as ctx:
                # ... perform operation ...
                ctx["records_processed"] = n_records
                ctx["fields_processed"] = n_fields
        """
        import time

        context = {
            "records_processed": 0,
            "fields_processed": 0,
            "warnings": [],
        }

        bound_log = self._log.bind(operation=operation, target=target, **params)
        bound_log.info("operation_started")

        start_time = time.time()
        try:
            yield context
            duration = time.time() - start_time
            bound_log.info(
                "operation_complete",
                duration_seconds=round(duration, 2),
                records_processed=context.get("records_processed", 0),
                fields_processed=context.get("fields_processed", 0),
            )
        except Exception as e:
            duration = time.time() - start_time
            bound_log.error(
                "operation_failed",
                duration_seconds=round(duration, 2),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

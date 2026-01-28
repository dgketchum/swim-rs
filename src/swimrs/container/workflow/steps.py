"""
Workflow step definitions for automated data preparation.

Each step represents a discrete operation that can be:
- Checked for completion
- Executed independently
- Tracked for progress
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swimrs.container import SwimContainer

    from .config import (
        ETFSourceConfig,
        MeteorologyConfig,
        NDVISourceConfig,
        PropertiesConfig,
    )


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a workflow step."""

    status: StepStatus
    message: str = ""
    records_processed: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the step completed successfully."""
        return self.status == StepStatus.COMPLETED


class WorkflowStep(ABC):
    """
    Abstract base class for workflow steps.

    Each step must implement:
    - name: Unique identifier for the step
    - description: Human-readable description
    - check_complete: Check if step output already exists
    - execute: Perform the step operation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique step identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @abstractmethod
    def check_complete(self, container: SwimContainer) -> bool:
        """
        Check if step output already exists in container.

        Args:
            container: SwimContainer to check

        Returns:
            True if step can be skipped
        """
        pass

    @abstractmethod
    def execute(self, container: SwimContainer) -> StepResult:
        """
        Execute the step.

        Args:
            container: SwimContainer to operate on

        Returns:
            StepResult with status and details
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


@dataclass
class IngestNDVIStep(WorkflowStep):
    """Step for ingesting NDVI data."""

    config: NDVISourceConfig

    @property
    def name(self) -> str:
        return f"ingest_ndvi_{self.config.instrument}_{self.config.mask}"

    @property
    def description(self) -> str:
        return f"Ingest NDVI from {self.config.instrument} ({self.config.mask} mask)"

    def check_complete(self, container: SwimContainer) -> bool:
        path = f"remote_sensing/ndvi/{self.config.instrument}/{self.config.mask}"
        return path in container._root

    def execute(self, container: SwimContainer) -> StepResult:
        import time

        start = time.perf_counter()
        try:
            event = container.ingest_ee_ndvi(
                source_dir=self.config.path,
                instrument=self.config.instrument,
                mask=self.config.mask,
            )
            duration = time.perf_counter() - start
            return StepResult(
                status=StepStatus.COMPLETED,
                message=f"Ingested {event.records_count} NDVI records",
                records_processed=event.records_count or 0,
                duration_seconds=duration,
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                message=str(e),
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )


@dataclass
class IngestETFStep(WorkflowStep):
    """Step for ingesting ETF data."""

    config: ETFSourceConfig

    @property
    def name(self) -> str:
        return f"ingest_etf_{self.config.instrument}_{self.config.model}_{self.config.mask}"

    @property
    def description(self) -> str:
        return f"Ingest ETF from {self.config.model} ({self.config.mask} mask)"

    def check_complete(self, container: SwimContainer) -> bool:
        path = f"remote_sensing/etf/{self.config.instrument}/{self.config.model}/{self.config.mask}"
        return path in container._root

    def execute(self, container: SwimContainer) -> StepResult:
        import time

        start = time.perf_counter()
        try:
            event = container.ingest_ee_etf(
                source_dir=self.config.path,
                model=self.config.model,
                mask=self.config.mask,
                instrument=self.config.instrument,
            )
            duration = time.perf_counter() - start
            return StepResult(
                status=StepStatus.COMPLETED,
                message=f"Ingested {event.records_count} ETF records",
                records_processed=event.records_count or 0,
                duration_seconds=duration,
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                message=str(e),
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )


@dataclass
class IngestMeteorologyStep(WorkflowStep):
    """Step for ingesting meteorology data."""

    config: MeteorologyConfig

    @property
    def name(self) -> str:
        return f"ingest_meteorology_{self.config.source}"

    @property
    def description(self) -> str:
        return f"Ingest meteorology from {self.config.source}"

    def check_complete(self, container: SwimContainer) -> bool:
        path = f"meteorology/{self.config.source}/eto"
        return path in container._root

    def execute(self, container: SwimContainer) -> StepResult:
        import time

        start = time.perf_counter()
        try:
            if self.config.source == "gridmet":
                event = container.ingest_gridmet(
                    source_dir=self.config.path,
                    variables=self.config.variables,
                )
            elif self.config.source == "era5":
                event = container.ingest_era5(
                    source_dir=self.config.path,
                    variables=self.config.variables,
                )
            else:
                raise ValueError(f"Unknown meteorology source: {self.config.source}")

            duration = time.perf_counter() - start
            return StepResult(
                status=StepStatus.COMPLETED,
                message=f"Ingested meteorology ({event.records_count} records)",
                records_processed=event.records_count or 0,
                duration_seconds=duration,
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                message=str(e),
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )


@dataclass
class IngestPropertiesStep(WorkflowStep):
    """Step for ingesting field properties."""

    config: PropertiesConfig

    @property
    def name(self) -> str:
        return "ingest_properties"

    @property
    def description(self) -> str:
        return "Ingest field properties (LULC, soils, irrigation)"

    def check_complete(self, container: SwimContainer) -> bool:
        # Check for at least one property
        return (
            "properties/soils/awc" in container._root
            or "properties/land_cover/modis_lc" in container._root
        )

    def execute(self, container: SwimContainer) -> StepResult:
        import time

        start = time.perf_counter()
        try:
            event = container.ingest_properties(
                lulc_csv=self.config.lulc,
                soils_csv=self.config.soils,
                irrigation_csv=self.config.irrigation,
                location_csv=self.config.location,
            )
            duration = time.perf_counter() - start
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Ingested field properties",
                records_processed=event.records_count or 0,
                duration_seconds=duration,
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                message=str(e),
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )


@dataclass
class ComputeFusedNDVIStep(WorkflowStep):
    """Step for computing fused NDVI."""

    @property
    def name(self) -> str:
        return "compute_fused_ndvi"

    @property
    def description(self) -> str:
        return "Compute fused Landsat+Sentinel NDVI"

    def check_complete(self, container: SwimContainer) -> bool:
        return "derived/merged_ndvi/irr" in container._root

    def execute(self, container: SwimContainer) -> StepResult:
        import time

        start = time.perf_counter()
        try:
            event = container.compute_fused_ndvi()
            duration = time.perf_counter() - start
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Computed fused NDVI",
                duration_seconds=duration,
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                message=str(e),
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )


@dataclass
class ComputeDynamicsStep(WorkflowStep):
    """Step for computing field dynamics."""

    etf_model: str = "ssebop"
    irr_threshold: float = 0.1
    masks: tuple = ("irr", "inv_irr")
    instrument: str = "landsat"

    @property
    def name(self) -> str:
        return f"compute_dynamics_{self.etf_model}"

    @property
    def description(self) -> str:
        return f"Compute field dynamics using {self.etf_model}"

    @property
    def dependencies(self) -> list:
        """Optional dependencies - fused_ndvi is preferred but not required."""
        return ["compute_fused_ndvi"]

    def check_dependencies(self, container: SwimContainer) -> tuple:
        """
        Check if dependencies are satisfied or skippable.

        Returns:
            Tuple of (satisfied: bool, message: str)
        """
        # Fused NDVI is preferred but not required
        fused_path = f"derived/merged_ndvi/{self.masks[0]}"
        ndvi_path = f"remote_sensing/ndvi/{self.instrument}/{self.masks[0]}"

        if fused_path in container._root:
            return True, "Using fused NDVI (recommended)"
        elif ndvi_path in container._root:
            return (
                True,
                f"Using single-instrument NDVI from {self.instrument} (fused not available)",
            )
        else:
            return False, "No NDVI data available. Run ingest.ndvi() first."

    def check_complete(self, container: SwimContainer) -> bool:
        return "derived/dynamics/ke_max" in container._root

    def execute(self, container: SwimContainer) -> StepResult:
        import time

        start = time.perf_counter()

        # Check dependencies and warn if fused NDVI is missing
        deps_ok, deps_msg = self.check_dependencies(container)
        if not deps_ok:
            return StepResult(
                status=StepStatus.FAILED,
                message=deps_msg,
                error="Missing required NDVI data",
                duration_seconds=time.perf_counter() - start,
            )

        try:
            event = container.compute_dynamics(
                etf_model=self.etf_model,
                irr_threshold=self.irr_threshold,
            )
            duration = time.perf_counter() - start

            # Include dependency message in result
            message = f"Computed dynamics for {event.records_count or 'all'} fields"
            if "single-instrument" in deps_msg:
                message += f" (Note: {deps_msg})"

            return StepResult(
                status=StepStatus.COMPLETED,
                message=message,
                duration_seconds=duration,
            )
        except Exception as e:
            return StepResult(
                status=StepStatus.FAILED,
                message=str(e),
                error=str(e),
                duration_seconds=time.perf_counter() - start,
            )

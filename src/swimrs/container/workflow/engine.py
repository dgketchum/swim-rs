"""
Workflow engine for automated data preparation.

Provides:
- WorkflowEngine: Orchestrates execution of workflow steps
- Resumable execution with checkpoint support
- Progress tracking and reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from swimrs.container import SwimContainer
from swimrs.container.logging import get_logger

from .config import WorkflowConfig
from .steps import (
    ComputeDynamicsStep,
    ComputeFusedNDVIStep,
    IngestETFStep,
    IngestMeteorologyStep,
    IngestNDVIStep,
    IngestPropertiesStep,
    StepResult,
    StepStatus,
    WorkflowStep,
)

logger = get_logger("workflow")


@dataclass
class WorkflowProgress:
    """Tracks progress of workflow execution."""

    total_steps: int = 0
    completed_steps: int = 0
    skipped_steps: int = 0
    failed_steps: int = 0
    step_results: dict[str, StepResult] = field(default_factory=dict)

    @property
    def percent_complete(self) -> float:
        """Percentage of steps completed or skipped."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps + self.skipped_steps) / self.total_steps * 100

    @property
    def success(self) -> bool:
        """Whether all steps completed successfully."""
        return self.failed_steps == 0

    def summary(self) -> str:
        """Human-readable progress summary."""
        lines = [
            f"Workflow Progress: {self.percent_complete:.0f}%",
            f"  Completed: {self.completed_steps}/{self.total_steps}",
            f"  Skipped: {self.skipped_steps}",
            f"  Failed: {self.failed_steps}",
        ]
        if self.failed_steps > 0:
            lines.append("Failed steps:")
            for name, result in self.step_results.items():
                if result.status == StepStatus.FAILED:
                    lines.append(f"  - {name}: {result.error}")
        return "\n".join(lines)


class WorkflowEngine:
    """
    Engine for executing data preparation workflows.

    Reads configuration, builds execution plan, and runs steps
    with resumability and progress tracking.

    Example:
        # From configuration file
        engine = WorkflowEngine.from_yaml("project.yaml")
        engine.run()

        # Check status
        engine.status()

        # Resume after failure
        engine.run(resume=True)
    """

    def __init__(
        self,
        config: WorkflowConfig,
        container: SwimContainer | None = None,
    ):
        """
        Initialize the workflow engine.

        Args:
            config: WorkflowConfig instance
            container: Optional existing container (created if not provided)
        """
        self.config = config
        self._container = container
        self._steps: list[WorkflowStep] = []
        self._progress = WorkflowProgress()

        # Build execution plan
        self._build_execution_plan()

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> WorkflowEngine:
        """
        Create workflow engine from YAML configuration file.

        Args:
            config_path: Path to YAML configuration

        Returns:
            WorkflowEngine instance
        """
        config = WorkflowConfig.from_yaml(config_path)
        return cls(config)

    def _build_execution_plan(self) -> None:
        """Build ordered list of workflow steps from configuration."""
        self._steps = []

        # NDVI ingestion steps
        for ndvi_config in self.config.sources.ndvi:
            self._steps.append(IngestNDVIStep(config=ndvi_config))

        # ETF ingestion steps
        for etf_config in self.config.sources.etf:
            self._steps.append(IngestETFStep(config=etf_config))

        # Meteorology step
        if self.config.sources.meteorology:
            self._steps.append(IngestMeteorologyStep(config=self.config.sources.meteorology))

        # Properties step
        if self.config.sources.properties:
            self._steps.append(IngestPropertiesStep(config=self.config.sources.properties))

        # Compute steps
        if self.config.compute.compute_fused_ndvi:
            self._steps.append(ComputeFusedNDVIStep())

        if self.config.compute.compute_dynamics:
            params = self.config.compute.dynamics_params
            self._steps.append(
                ComputeDynamicsStep(
                    etf_model=params.get("etf_model", "ssebop"),
                    irr_threshold=params.get("irr_threshold", 0.1),
                )
            )

        self._progress.total_steps = len(self._steps)

    def _ensure_container(self) -> SwimContainer:
        """Ensure container exists, creating if necessary."""
        if self._container is not None:
            return self._container

        # Determine container path
        project = self.config.project
        if project.output_path:
            container_path = self.config.resolve_path(project.output_path)
        else:
            container_path = self.config.resolve_path(f"{project.name}.swim")

        # Check if container exists
        if container_path.exists():
            logger.info("opening_existing_container", path=str(container_path))
            self._container = SwimContainer(container_path, mode="r+")
        else:
            logger.info("creating_new_container", path=str(container_path))
            self._container = SwimContainer.create(
                uri=container_path,
                fields_shapefile=self.config.resolve_path(project.shapefile),
                uid_column=project.uid_column,
                start_date=project.start_date,
                end_date=project.end_date,
                project_name=project.name,
            )

        return self._container

    def run(
        self,
        resume: bool = True,
        step: str | None = None,
        dry_run: bool = False,
    ) -> WorkflowProgress:
        """
        Execute the workflow.

        Args:
            resume: If True, skip completed steps
            step: If provided, run only this specific step
            dry_run: If True, only report what would be done

        Returns:
            WorkflowProgress with execution results
        """
        container = self._ensure_container()

        # Filter steps if specific step requested
        steps_to_run = self._steps
        if step:
            steps_to_run = [s for s in self._steps if s.name == step]
            if not steps_to_run:
                raise ValueError(
                    f"Step not found: {step}. Available: {[s.name for s in self._steps]}"
                )

        logger.info(
            "starting_workflow",
            total_steps=len(steps_to_run),
            resume=resume,
            dry_run=dry_run,
        )

        for workflow_step in steps_to_run:
            # Check if step is already complete
            if resume and workflow_step.check_complete(container):
                logger.info(
                    "skipping_step",
                    step=workflow_step.name,
                    reason="already_complete",
                )
                self._progress.skipped_steps += 1
                self._progress.step_results[workflow_step.name] = StepResult(
                    status=StepStatus.SKIPPED,
                    message="Already complete",
                )
                continue

            # Dry run - just report
            if dry_run:
                print(f"[DRY RUN] Would execute: {workflow_step.name}")
                print(f"          {workflow_step.description}")
                continue

            # Execute step
            logger.info(
                "executing_step",
                step=workflow_step.name,
                description=workflow_step.description,
            )
            print(f"Executing: {workflow_step.description}...")

            result = workflow_step.execute(container)
            self._progress.step_results[workflow_step.name] = result

            if result.success:
                self._progress.completed_steps += 1
                logger.info(
                    "step_completed",
                    step=workflow_step.name,
                    duration=result.duration_seconds,
                    records=result.records_processed,
                )
                print(f"  ✓ {result.message} ({result.duration_seconds:.1f}s)")
            else:
                self._progress.failed_steps += 1
                logger.error(
                    "step_failed",
                    step=workflow_step.name,
                    error=result.error,
                )
                print(f"  ✗ Failed: {result.error}")
                # Stop on failure
                break

        # Save container
        if not dry_run:
            container.save()

        logger.info(
            "workflow_complete",
            completed=self._progress.completed_steps,
            skipped=self._progress.skipped_steps,
            failed=self._progress.failed_steps,
        )

        return self._progress

    def status(self) -> str:
        """
        Get current workflow status.

        Shows which steps are complete, pending, or failed.

        Returns:
            Formatted status string
        """
        container = self._ensure_container()

        lines = [
            f"Workflow: {self.config.project.name}",
            f"Container: {container.path}",
            f"Fields: {container.n_fields}",
            f"Date range: {container.start_date.date()} to {container.end_date.date()}",
            "",
            "Steps:",
        ]

        for step in self._steps:
            if step.name in self._progress.step_results:
                result = self._progress.step_results[step.name]
                status_icon = {
                    StepStatus.COMPLETED: "✓",
                    StepStatus.SKIPPED: "○",
                    StepStatus.FAILED: "✗",
                    StepStatus.PENDING: "·",
                    StepStatus.IN_PROGRESS: "▶",
                }[result.status]
                lines.append(f"  {status_icon} {step.name}: {result.message}")
            elif step.check_complete(container):
                lines.append(f"  ✓ {step.name}: Complete (pre-existing)")
            else:
                lines.append(f"  · {step.name}: Pending")

        return "\n".join(lines)

    def list_steps(self) -> list[str]:
        """Get list of step names in execution order."""
        return [step.name for step in self._steps]

    @property
    def progress(self) -> WorkflowProgress:
        """Get current progress."""
        return self._progress

    @property
    def container(self) -> SwimContainer | None:
        """Get the container (if opened)."""
        return self._container

"""
Workflow orchestration for automated data preparation.

Provides YAML-based configuration and automated execution of
the standard data preparation pipeline.

Example YAML Configuration:
    project:
      name: "Flux_Network"
      shapefile: "data/sites.shp"
      uid_column: "site_id"
      date_range: ["2017-01-01", "2023-12-31"]

    sources:
      ndvi:
        landsat:
          irr: "data/ndvi/landsat_irr/"
          inv_irr: "data/ndvi/landsat_inv_irr/"
      etf:
        landsat:
          ssebop:
            irr: "data/etf/ssebop_irr/"
      meteorology:
        source: gridmet
        path: "data/gridmet/"
      properties:
        lulc: "data/lulc.csv"
        soils: "data/soils.csv"

    workflow:
      compute_fused_ndvi: true
      compute_dynamics:
        etf_model: ssebop
        irr_threshold: 0.1

Example Usage:
    from swimrs.container.workflow import WorkflowEngine

    # From YAML config
    engine = WorkflowEngine.from_yaml("project.yaml")

    # Check what will be done
    print(engine.status())

    # Run workflow (resumes from last successful step)
    progress = engine.run(resume=True)

    # Run specific step
    progress = engine.run(step="compute_dynamics")
"""

from .config import (
    ComputeConfig,
    ETFSourceConfig,
    MeteorologyConfig,
    NDVISourceConfig,
    ProjectConfig,
    PropertiesConfig,
    SourcesConfig,
    ValidationConfig,
    WorkflowConfig,
)
from .engine import (
    WorkflowEngine,
    WorkflowProgress,
)
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

__all__ = [
    # Configuration
    "WorkflowConfig",
    "ProjectConfig",
    "SourcesConfig",
    "NDVISourceConfig",
    "ETFSourceConfig",
    "MeteorologyConfig",
    "PropertiesConfig",
    "ComputeConfig",
    "ValidationConfig",
    # Steps
    "WorkflowStep",
    "StepStatus",
    "StepResult",
    "IngestNDVIStep",
    "IngestETFStep",
    "IngestMeteorologyStep",
    "IngestPropertiesStep",
    "ComputeFusedNDVIStep",
    "ComputeDynamicsStep",
    # Engine
    "WorkflowEngine",
    "WorkflowProgress",
]

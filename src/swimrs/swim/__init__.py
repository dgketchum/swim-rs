"""
SWIM-RS core simulation module.

Provides configuration and data containers for SWIM-RS model runs.

Key Classes:
    ProjectConfig: Configuration management for SWIM-RS projects.
    SamplePlots: Container for field data loaded from JSON files.
    ContainerPlots: Adapter for using SwimContainer data with the model.
"""

from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import ContainerPlots, SamplePlots

__all__ = [
    "ProjectConfig",
    "SamplePlots",
    "ContainerPlots",
]

if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================

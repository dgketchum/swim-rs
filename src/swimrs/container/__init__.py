"""
SWIM Data Container module.

Provides a unified data container for SWIM-RS projects using Zarr as the backend.
"""

from swimrs.container.container import SwimContainer
from swimrs.container.schema import (
    SwimSchema,
    Instrument,
    MaskType,
    ETModel,
    MetSource,
    SnowSource,
    SoilSource,
    Parameter,
)

__all__ = [
    "SwimContainer",
    "SwimSchema",
    "Instrument",
    "MaskType",
    "ETModel",
    "MetSource",
    "SnowSource",
    "SoilSource",
    "Parameter",
]

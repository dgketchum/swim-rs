"""
SWIM Data Container module.

Provides a unified data container for SWIM-RS projects using Zarr as the backend.

Key Classes:
    SwimContainer: Main container class for unified SWIM-RS data storage.
    ContainerBase: Base class with lifecycle management (for extension).
    SwimSchema: Schema definitions and validation rules.

Convenience Functions:
    open_container: Open an existing .swim file.
    create_container: Create a new container from a shapefile.

Architecture:
    SwimContainer composes functionality from mixins:
    - IngestionMixin: Data ingestion (ingest_ee_ndvi, ingest_gridmet, etc.)
    - ComputeMixin: Derived data computation (compute_dynamics, compute_fused_ndvi)
    - ExportMixin: Data export (export_prepped_input_json, export_shapefile)
    - QueryMixin: Data access (status, get_dataframe, get_geodataframe)

Example:
    >>> from swimrs.container import create_container, open_container
    >>>
    >>> # Create a new container
    >>> container = create_container(
    ...     path="project.swim",
    ...     fields_shapefile="fields.shp",
    ...     uid_column="FID",
    ...     start_date="2017-01-01",
    ...     end_date="2023-12-31",
    ... )
    >>>
    >>> # Ingest data
    >>> container.ingest_gridmet("gridmet_dir/")
    >>> container.ingest_ee_ndvi("ndvi_dir/", instrument="landsat", mask="irr")
    >>>
    >>> # Compute dynamics and export for model
    >>> container.compute_dynamics(etf_model="ssebop")
    >>> container.compute_fused_ndvi()
    >>> container.export_prepped_input_json("prepped_input.json")
    >>>
    >>> # Or use directly with model via ContainerPlots
    >>> from swimrs.swim.sampleplots import ContainerPlots
    >>> plots = ContainerPlots(container, etf_model="ssebop")
"""

from swimrs.container.container import (
    SwimContainer,
    open_container,
    create_container,
)
from swimrs.container.base import ContainerBase
from swimrs.container.mixins import (
    IngestionMixin,
    ComputeMixin,
    ExportMixin,
    QueryMixin,
)
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
    # Main classes
    "SwimContainer",
    "ContainerBase",
    "SwimSchema",
    # Mixins (for extension)
    "IngestionMixin",
    "ComputeMixin",
    "ExportMixin",
    "QueryMixin",
    # Convenience functions
    "open_container",
    "create_container",
    # Enums
    "Instrument",
    "MaskType",
    "ETModel",
    "MetSource",
    "SnowSource",
    "SoilSource",
    "Parameter",
]

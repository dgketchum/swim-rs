"""
SWIM Data Container module.

Provides a unified data container for SWIM-RS projects using Zarr as the backend.

Key Classes:
    SwimContainer: Main container class for unified SWIM-RS data storage.
    ContainerState: Centralized state with xarray interface.
    SwimSchema: Schema definitions and validation rules.

Storage Backends:
    The container supports pluggable storage backends:
    - ZipStoreProvider: Local .swim files (default)
    - DirectoryStoreProvider: Local directories (faster for development)
    - S3StoreProvider: Amazon S3 / S3-compatible storage
    - GCSStoreProvider: Google Cloud Storage
    - MemoryStoreProvider: In-memory (for testing)

Components:
    Clean, namespace-organized API via component attributes:
    - container.ingest: Data ingestion (ndvi, etf, gridmet, etc.)
    - container.compute: Derived computation (dynamics, fused_ndvi)
    - container.export: Data export (prepped_input_json, shapefile)
    - container.query: Data access (status, xarray, dataframe)

Convenience Functions:
    open_container: Open an existing container (local or cloud).
    create_container: Create a new container from a shapefile.

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
    >>> # Ingest data via component API
    >>> container.ingest.gridmet("gridmet_dir/")
    >>> container.ingest.ndvi("ndvi_dir/", instrument="landsat", mask="irr")
    >>>
    >>> # Compute dynamics and export for model
    >>> container.compute.dynamics(etf_model="ssebop")
    >>> container.export.prepped_input_json("prepped_input.json")
    >>>
    >>> # Open from cloud storage (requires s3fs)
    >>> container = SwimContainer.open("s3://bucket/project.zarr", mode="r")
"""

from swimrs.container.container import (
    SwimContainer,
    open_container,
    create_container,
)
from swimrs.container.state import ContainerState
from swimrs.container.components import (
    Component,
    Ingestor,
    Calculator,
    Exporter,
    Query,
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
from swimrs.container.storage import (
    StorageProvider,
    StorageProviderFactory,
    ZipStoreProvider,
    DirectoryStoreProvider,
    MemoryStoreProvider,
    open_storage,
)
from swimrs.container.metrics import (
    OperationMetrics,
    OperationContext,
    MetricsCollector,
    MetricsSummary,
    track_operation,
)
from swimrs.container.logging import (
    ContainerLogger,
    get_logger,
    configure_logging,
)
from swimrs.container.workflow import (
    WorkflowEngine,
    WorkflowConfig,
    WorkflowProgress,
)

__all__ = [
    # Main classes
    "SwimContainer",
    "SwimSchema",
    "ContainerState",
    # Components
    "Component",
    "Ingestor",
    "Calculator",
    "Exporter",
    "Query",
    # Convenience functions
    "open_container",
    "create_container",
    # Storage providers
    "StorageProvider",
    "StorageProviderFactory",
    "ZipStoreProvider",
    "DirectoryStoreProvider",
    "MemoryStoreProvider",
    "open_storage",
    # Metrics and observability
    "OperationMetrics",
    "OperationContext",
    "MetricsCollector",
    "MetricsSummary",
    "track_operation",
    # Logging
    "ContainerLogger",
    "get_logger",
    "configure_logging",
    # Workflow
    "WorkflowEngine",
    "WorkflowConfig",
    "WorkflowProgress",
    # Enums
    "Instrument",
    "MaskType",
    "ETModel",
    "MetSource",
    "SnowSource",
    "SoilSource",
    "Parameter",
]

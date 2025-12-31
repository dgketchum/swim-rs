"""
SWIM Data Container - unified data management for SWIM-RS projects.

Provides a single-file container (Zarr ZipStore) that holds all project data
including geometries, remote sensing, meteorology, properties, and derived products.

The SwimContainer class composes functionality from:
- ContainerBase: Lifecycle management, state, and helper methods
- IngestionMixin: Data ingestion from various sources
- ComputeMixin: Derived data computation
- ExportMixin: Data export in various formats
- QueryMixin: Data access and status queries
"""

from datetime import datetime
from pathlib import Path
from typing import Union

from swimrs.container.base import ContainerBase
from swimrs.container.mixins import (
    IngestionMixin,
    ComputeMixin,
    ExportMixin,
    QueryMixin,
)


class SwimContainer(
    IngestionMixin,
    ComputeMixin,
    ExportMixin,
    QueryMixin,
    ContainerBase,
):
    """
    Unified data container for SWIM-RS projects.

    Stores all project data in a single Zarr archive (.swim file) including:
    - Field geometries (from shapefile)
    - Remote sensing data (NDVI, ETF from Landsat/Sentinel/ECOSTRESS)
    - Meteorology (GridMET, ERA5)
    - Static properties (soils, land cover, irrigation masks)
    - Snow data (SNODAS)
    - Derived products (dynamics, fused NDVI)

    Provides full provenance tracking and observability into data completeness.

    This class composes functionality from multiple mixins:
    - IngestionMixin: ingest_ee_ndvi, ingest_ee_etf, ingest_gridmet, etc.
    - ComputeMixin: compute_dynamics, compute_fused_ndvi
    - ExportMixin: export_shapefile, export_csv, export_prepped_input_json
    - QueryMixin: status, validate, get_dataframe, get_geodataframe

    Example:
        # Create a new container
        container = SwimContainer.create(
            "project.swim",
            fields_shapefile="fields.shp",
            uid_column="FID",
            start_date="2016-01-01",
            end_date="2023-12-31",
            project_name="My Project"
        )

        # Check status
        container.status()

        # Ingest data
        container.ingest_ee_ndvi("path/to/csvs/", instrument="landsat", mask="irr")

        # Save and close
        container.save()
    """

    # Class attributes inherited from ContainerBase
    # EXTENSION = ".swim"
    # SCHEMA_VERSION = "1.0"

    pass  # All functionality comes from mixins and base class


# -------------------------------------------------------------------------
# Convenience functions
# -------------------------------------------------------------------------

def open_container(path: Union[str, Path], mode: str = "r") -> SwimContainer:
    """
    Open an existing SWIM container.

    Args:
        path: Path to .swim file
        mode: 'r' for read-only, 'r+' for read-write

    Returns:
        SwimContainer instance
    """
    return SwimContainer(path, mode=mode)


def create_container(path: Union[str, Path],
                    fields_shapefile: Union[str, Path],
                    uid_column: str,
                    start_date: Union[str, datetime],
                    end_date: Union[str, datetime],
                    project_name: str = None,
                    overwrite: bool = False) -> SwimContainer:
    """
    Create a new SWIM container from a shapefile.

    Args:
        path: Path for the new .swim file
        fields_shapefile: Path to shapefile with field geometries
        uid_column: Column name containing unique field identifiers
        start_date: Start of analysis period
        end_date: End of analysis period
        project_name: Optional project name
        overwrite: If True, overwrite existing file

    Returns:
        New SwimContainer instance
    """
    return SwimContainer.create(
        path=path,
        fields_shapefile=fields_shapefile,
        uid_column=uid_column,
        start_date=start_date,
        end_date=end_date,
        project_name=project_name,
        overwrite=overwrite,
    )

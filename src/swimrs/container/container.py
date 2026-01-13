"""
SWIM Data Container - unified data management for SWIM-RS projects.

Provides a single-file container (Zarr ZipStore) that holds all project data
including geometries, remote sensing, meteorology, properties, and derived products.

SwimContainer provides:
- Core: Lifecycle management, state, and helper methods
- Components: Ingestor, Calculator, Exporter, Query for clean API access
- Xarray interface for vectorized data operations

Storage backends are pluggable via the storage module:
- ZipStoreProvider: Local .swim files (default)
- DirectoryStoreProvider: Local directories (faster for development)
- S3StoreProvider: Amazon S3 / S3-compatible storage
- GCSStoreProvider: Google Cloud Storage
- MemoryStoreProvider: In-memory (for testing)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from zarr.core.dtype import VariableLengthUTF8, VariableLengthBytes

from swimrs.container.provenance import ProvenanceLog, DatasetProvenance
from swimrs.container.inventory import Inventory
from swimrs.container.state import ContainerState
from swimrs.container.storage import (
    StorageProvider,
    StorageProviderFactory,
    ZipStoreProvider,
    DirectoryStoreProvider,
)
from swimrs.container.components import (
    Ingestor,
    Calculator,
    Exporter,
    Query,
)


class SwimContainer:
    """
    Unified data container for SWIM-RS projects.

    Stores all project data in a Zarr archive including:
    - Field geometries (from shapefile)
    - Remote sensing data (NDVI, ETF from Landsat/Sentinel/ECOSTRESS)
    - Meteorology (GridMET, ERA5)
    - Static properties (soils, land cover, irrigation masks)
    - Snow data (SNODAS)
    - Derived products (dynamics, fused NDVI)

    Provides full provenance tracking and observability into data completeness.

    Storage backends are pluggable:
    - Local .swim files (ZipStoreProvider) - default
    - Local directories (DirectoryStoreProvider) - faster for development
    - S3 buckets (S3StoreProvider) - cloud storage
    - GCS buckets (GCSStoreProvider) - cloud storage
    - Memory (MemoryStoreProvider) - for testing

    Component-based API:
    - container.ingest: Data ingestion operations
    - container.compute: Derived data computation
    - container.export: Data export operations
    - container.query: Data access and status queries

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

        # Ingest data via component API
        container.ingest.ndvi("path/to/csvs/", instrument="landsat", mask="irr")
        container.ingest.gridmet("path/to/met/")

        # Compute derived products
        container.compute.dynamics(etf_model="ssebop")

        # Export for model
        container.export.prepped_input_json("output/prepped.json")

        # Save and close
        container.save()

        # Open existing container from URI (auto-selects backend)
        container = SwimContainer.open("s3://bucket/project.zarr", mode="r")
    """

    EXTENSION = ".swim"
    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        path_or_provider: Union[str, Path, StorageProvider],
        mode: str = "r",
    ):
        """
        Open an existing SwimContainer.

        Args:
            path_or_provider: Path to container file, URI string, or StorageProvider
            mode: 'r' for read-only, 'r+' for read-write, 'a' for append
                  (ignored if path_or_provider is a StorageProvider)

        Note:
            For URI-based access (S3, GCS), use SwimContainer.open() which provides
            a cleaner interface for passing provider-specific options.
        """
        # Handle different input types
        if isinstance(path_or_provider, StorageProvider):
            self._provider = path_or_provider
            self._mode = path_or_provider.mode
            # Extract path for backward compatibility
            location = path_or_provider.location
            self.path = Path(location) if isinstance(location, str) and not location.startswith(("s3://", "gs://")) else Path(str(location))
        else:
            # Legacy path-based initialization
            self.path = Path(path_or_provider)
            self._mode = mode
            self._provider = StorageProviderFactory.from_uri(self.path, mode=mode)

        self._root: Optional[zarr.Group] = None
        self._provenance: Optional[ProvenanceLog] = None
        self._inventory: Optional[Inventory] = None
        self._field_uids: List[str] = []
        self._uid_to_index: Dict[str, int] = {}
        self._time_index: Optional[pd.DatetimeIndex] = None
        self._modified: bool = False

        # Check existence for read modes
        if self._mode in ("r", "r+") and not self._provider.exists():
            raise FileNotFoundError(f"Container not found: {self._provider.uri}")

        self._open_storage()

    def _open_storage(self):
        """Open the storage and load metadata."""
        # Suppress ZipStore duplicate name warnings during writes.
        # This is expected during ingestion and the data is correct.
        if self._mode in ("r+", "a", "w"):
            warnings.filterwarnings(
                "ignore",
                message="Duplicate name:",
                category=UserWarning,
                module="zipfile",
            )

        # Open storage provider and get root group
        self._root = self._provider.open()

        # Load metadata
        self._load_metadata()

    @classmethod
    def open(
        cls,
        uri: Union[str, Path],
        mode: str = "r",
        **kwargs: Any,
    ) -> "SwimContainer":
        """
        Open an existing container from a URI.

        This is the preferred way to open containers, especially for cloud storage.
        Automatically selects the appropriate storage backend based on the URI.

        Args:
            uri: Location of the container:
                - Local path: "project.swim", "/path/to/project.zarr"
                - File URI: "file:///path/to/project.swim"
                - S3 URI: "s3://bucket/key" (requires s3fs)
                - GCS URI: "gs://bucket/key" (requires gcsfs)
            mode: Access mode ('r', 'r+', 'a')
            **kwargs: Additional arguments passed to the storage provider:
                - S3: aws_access_key_id, aws_secret_access_key, endpoint_url, etc.
                - GCS: project, token, etc.

        Returns:
            SwimContainer: Opened container instance

        Example:
            # Local file
            container = SwimContainer.open("project.swim")

            # S3 with IAM credentials
            container = SwimContainer.open("s3://bucket/project.zarr")

            # S3 with explicit credentials
            container = SwimContainer.open(
                "s3://bucket/project.zarr",
                aws_access_key_id="AKIA...",
                aws_secret_access_key="..."
            )
        """
        provider = StorageProviderFactory.from_uri(uri, mode=mode, **kwargs)
        return cls(provider)

    @property
    def uri(self) -> str:
        """URI identifying the storage location."""
        return self._provider.uri

    @property
    def storage_provider(self) -> StorageProvider:
        """Access the underlying storage provider."""
        return self._provider

    def _load_metadata(self):
        """Load container metadata from Zarr attrs."""
        # Load field UIDs and index mapping
        if "geometry/uid" in self._root:
            self._field_uids = list(self._root["geometry/uid"][:])
            # Handle bytes if stored as such
            if self._field_uids and isinstance(self._field_uids[0], bytes):
                self._field_uids = [u.decode("utf-8") for u in self._field_uids]
        else:
            self._field_uids = []

        self._uid_to_index = {uid: i for i, uid in enumerate(self._field_uids)}

        # Load time index
        if "time/daily" in self._root:
            self._time_index = pd.DatetimeIndex(self._root["time/daily"][:])
        else:
            self._time_index = None

        # Load provenance log
        prov_data = self._root.attrs.get("provenance", None)
        if prov_data:
            self._provenance = ProvenanceLog.from_dict(prov_data)
        else:
            self._provenance = ProvenanceLog()

        # Initialize inventory
        self._inventory = Inventory(self._root, self._field_uids)

        # Create centralized state object for component access
        self._state = ContainerState(
            provider=self._provider,
            field_uids=self._field_uids,
            time_index=self._time_index,
            provenance=self._provenance,
            inventory=self._inventory,
            mode=self._mode,
        )

        # Initialize component instances for clean API access
        # These provide container.ingest.ndvi() style API
        self.ingest = Ingestor(self._state, container=self)
        self.compute = Calculator(self._state, container=self)
        self.export = Exporter(self._state, container=self)
        self.query = Query(self._state, container=self)

    @classmethod
    def create(
        cls,
        uri: Union[str, Path],
        fields_shapefile: Union[str, Path],
        uid_column: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        project_name: str = None,
        overwrite: bool = False,
        storage: str = "auto",
        **storage_kwargs: Any,
    ) -> "SwimContainer":
        """
        Create a new SwimContainer from a shapefile.

        Args:
            uri: Location for the new container:
                - Local path: "project.swim" (default .swim extension added)
                - S3 URI: "s3://bucket/project.zarr"
                - GCS URI: "gs://bucket/project.zarr"
            fields_shapefile: Path to shapefile with field geometries
            uid_column: Column name containing unique field identifiers
            start_date: Start of analysis period
            end_date: End of analysis period
            project_name: Optional project name
            overwrite: If True, overwrite existing storage
            storage: Storage backend for local paths:
                - "auto" (default): directory store (better for development)
                - "directory": explicit directory store
                - "zip": explicit zip store (better for sharing)
            **storage_kwargs: Additional arguments passed to storage provider

        Returns:
            New SwimContainer instance

        Example:
            # Local directory store (default)
            container = SwimContainer.create(
                "project.swim",
                fields_shapefile="fields.shp",
                uid_column="FID",
                start_date="2016-01-01",
                end_date="2023-12-31"
            )

            # Explicit zip store for sharing
            container = SwimContainer.create(
                "project.swim",
                fields_shapefile="fields.shp",
                uid_column="FID",
                start_date="2016-01-01",
                end_date="2023-12-31",
                storage="zip"
            )

            # S3
            container = SwimContainer.create(
                "s3://bucket/project.zarr",
                fields_shapefile="fields.shp",
                uid_column="FID",
                start_date="2016-01-01",
                end_date="2023-12-31",
                aws_access_key_id="...",
                aws_secret_access_key="...")
        """
        # Determine storage location
        uri_str = str(uri)
        is_local = not uri_str.startswith(("s3://", "gs://", "memory://"))

        if is_local:
            path = Path(uri)
            if not path.suffix:
                path = path.with_suffix(cls.EXTENSION)
            uri_str = str(path)

            # Handle overwrite BEFORE creating provider so auto-detection
            # uses the default (DirectoryStore) instead of the existing format
            if overwrite and path.exists():
                import shutil
                if path.is_file():
                    path.unlink()
                    # Also remove lock file if present
                    lock_path = Path(str(path) + ".lock")
                    if lock_path.exists():
                        lock_path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                    lock_path = Path(str(path) + ".lock")
                    if lock_path.exists():
                        lock_path.unlink()

        # Create storage provider for write mode
        provider = StorageProviderFactory.from_uri(
            uri_str, mode="w", storage=storage, **storage_kwargs
        )

        # Check for existing container (only relevant if overwrite=False)
        if provider.exists():
            raise FileExistsError(
                f"Container already exists: {provider.uri}. "
                "Use overwrite=True to replace."
            )

        # Parse dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Read shapefile
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required to create a container from a shapefile")

        gdf = gpd.read_file(fields_shapefile)

        if uid_column not in gdf.columns:
            raise ValueError(f"UID column '{uid_column}' not found in shapefile. "
                           f"Available columns: {list(gdf.columns)}")

        # Extract UIDs and validate uniqueness
        uids = gdf[uid_column].astype(str).tolist()
        if len(uids) != len(set(uids)):
            raise ValueError(f"UID column '{uid_column}' contains duplicate values")

        n_fields = len(uids)

        # Create time index
        time_index = pd.date_range(start_date, end_date, freq="D")
        n_days = len(time_index)

        # Open storage for writing
        root = provider.open()

        # Initialize provenance early so we can include it in single attrs update
        provenance = ProvenanceLog()
        provenance.container_created_at = datetime.now(timezone.utc).isoformat()

        # Determine project name from path if not provided
        if project_name is None:
            if is_local:
                project_name = path.stem
            else:
                # Extract name from URI (e.g., s3://bucket/project.zarr -> project)
                project_name = uri_str.rstrip("/").split("/")[-1].split(".")[0]

        # Set all root attributes in a single update to avoid duplicate .zattrs in ZipStore
        root.attrs.update({
            "project_name": project_name,
            "schema_version": cls.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "n_fields": n_fields,
            "n_days": n_days,
            "uid_column": uid_column,
            "source_shapefile": str(fields_shapefile),
        })

        # Create time coordinate
        time_grp = root.create_group("time")
        time_grp.create_array(
            "daily",
            data=time_index.values.astype("datetime64[ns]"),
        )

        # Create geometry group
        geom_grp = root.create_group("geometry")

        # Store UIDs
        uid_arr = geom_grp.create_array(
            "uid",
            shape=(len(uids),),
            dtype=VariableLengthUTF8(),
        )
        uid_arr[:] = list(uids)

        # Store centroids
        centroids = gdf.geometry.centroid
        geom_grp.create_array("lon", data=centroids.x.values)
        geom_grp.create_array("lat", data=centroids.y.values)

        # Store area
        if gdf.crs and gdf.crs.is_projected:
            areas = gdf.geometry.area
        else:
            # Reproject to equal area for area calculation
            areas = gdf.to_crs("EPSG:6933").geometry.area
        geom_grp.create_array("area_m2", data=areas.values)

        # Store WKB geometries
        wkb_data = gdf.geometry.apply(lambda g: g.wkb).values
        wkb_arr = geom_grp.create_array(
            "wkb",
            shape=wkb_data.shape,
            dtype=VariableLengthBytes(),
        )
        wkb_arr[:] = wkb_data.tolist()

        # Store original shapefile properties
        props_grp = geom_grp.create_group("properties")
        for col in gdf.columns:
            if col == "geometry" or col == uid_column:
                continue
            try:
                data = gdf[col].values
                if data.dtype == object:
                    str_arr = props_grp.create_array(
                        col,
                        shape=data.shape,
                        dtype=VariableLengthUTF8(),
                    )
                    str_arr[:] = [str(x) for x in data]
                else:
                    props_grp.create_array(col, data=data)
            except Exception as e:
                print(f"Warning: Could not store property '{col}': {e}")

        # Create empty groups for data categories
        root.create_group("remote_sensing")
        root.create_group("meteorology")
        root.create_group("properties")
        root.create_group("snow")
        root.create_group("derived")

        # Record creation event in provenance
        event = provenance.record(
            "create",
            params={
                "fields_shapefile": str(fields_shapefile),
                "uid_column": uid_column,
                "start_date": str(start_date.date()),
                "end_date": str(end_date.date()),
                "project_name": project_name,
            },
            fields_affected=uids,
            records_count=n_fields,
        )

        # Update attrs with provenance
        root.attrs.update({"provenance": provenance.to_dict()})

        # Close the provider to flush writes
        provider.close()

        print(f"Created container: {provider.uri} ({n_fields} fields, {n_days} days)")

        # Reopen in append mode for continued use
        return cls.open(uri_str, mode="r+", **storage_kwargs)

    def close(self):
        """Close the container and release resources."""
        if self._modified:
            self._save_metadata()

        if self._provider is not None and self._provider.is_open:
            self._provider.close()
            self._root = None

    def save(self):
        """Save any pending changes to the container."""
        if self._mode == "r":
            raise ValueError("Cannot save: container opened in read-only mode")
        self._save_metadata()
        self._modified = False

    def _save_metadata(self):
        """Save metadata back to Zarr attrs."""
        if self._root is None:
            return
        self._root.attrs["provenance"] = self._provenance.to_dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def project_name(self) -> str:
        """Project name."""
        return self._root.attrs.get("project_name", "")

    @property
    def n_fields(self) -> int:
        """Number of fields in the container."""
        return len(self._field_uids)

    @property
    def field_uids(self) -> List[str]:
        """List of field UIDs."""
        return self._field_uids.copy()

    @property
    def start_date(self) -> pd.Timestamp:
        """Start date of the analysis period."""
        return pd.Timestamp(self._root.attrs.get("start_date"))

    @property
    def end_date(self) -> pd.Timestamp:
        """End date of the analysis period."""
        return pd.Timestamp(self._root.attrs.get("end_date"))

    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Date range as (start, end) tuple."""
        return (self.start_date, self.end_date)

    @property
    def n_days(self) -> int:
        """Number of days in the analysis period."""
        return len(self._time_index) if self._time_index is not None else 0

    @property
    def provenance(self) -> ProvenanceLog:
        """Access the provenance log."""
        return self._provenance

    @property
    def inventory(self) -> Inventory:
        """Access the inventory tracker."""
        return self._inventory

    @property
    def state(self) -> ContainerState:
        """
        Access the centralized container state.

        The state object provides:
        - Low-level zarr access via state.root
        - High-level xarray access via state.get_xarray(), state.get_dataset()
        - Field/time index utilities
        - Array creation helpers

        This is primarily intended for component classes and advanced usage.
        For most use cases, prefer the container's public methods.
        """
        return self._state

    # -------------------------------------------------------------------------
    # Xarray Interface (delegated to state)
    # -------------------------------------------------------------------------

    def to_xarray(
        self,
        path: str,
        fields: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        name: Optional[str] = None,
    ):
        """
        Get data as a labeled xarray DataArray.

        This provides a high-level interface for working with container data
        using xarray's powerful vectorized operations.

        Args:
            path: Zarr path to the array (e.g., "remote_sensing/ndvi/landsat/irr")
            fields: Optional list of field UIDs to include (default: all)
            start_date: Optional start date filter
            end_date: Optional end date filter
            name: Optional name for the DataArray

        Returns:
            xr.DataArray with 'time' and 'site' coordinates

        Example:
            ndvi = container.to_xarray("remote_sensing/ndvi/landsat/irr")

            # Use xarray operations
            smoothed = ndvi.rolling(time=32, center=True).mean()
            annual = ndvi.resample(time="YE").mean()
            by_site = ndvi.sel(site="US-Ne1")
        """
        return self._state.get_xarray(
            path, fields=fields, start_date=start_date, end_date=end_date, name=name
        )

    def to_dataset(
        self,
        paths: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ):
        """
        Get multiple variables as an xarray Dataset.

        Args:
            paths: Mapping of variable names to zarr paths.
                   If None, loads all available time series variables.
            fields: Optional list of field UIDs to include
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            xr.Dataset with requested variables

        Example:
            # Load specific variables
            ds = container.to_dataset({
                "ndvi": "remote_sensing/ndvi/landsat/irr",
                "etf": "remote_sensing/etf/landsat/ssebop/irr",
                "eto": "meteorology/gridmet/eto",
            })

            # Vectorized operations across all sites
            et = ds["etf"] * ds["eto"]
        """
        return self._state.get_dataset(
            paths=paths, fields=fields, start_date=start_date, end_date=end_date
        )

    # -------------------------------------------------------------------------
    # Index Access Helpers
    # -------------------------------------------------------------------------

    def get_field_index(self, uid: str) -> int:
        """Get the array index for a field UID."""
        if uid not in self._uid_to_index:
            raise KeyError(f"Unknown field UID: {uid}")
        return self._uid_to_index[uid]

    def get_time_index(self, date: Union[str, datetime, pd.Timestamp]) -> int:
        """Get the array index for a date."""
        if isinstance(date, str):
            date = pd.Timestamp(date)
        try:
            return self._time_index.get_loc(date)
        except KeyError:
            raise KeyError(f"Date {date} not in container time range")

    # -------------------------------------------------------------------------
    # Array Creation Helpers
    # -------------------------------------------------------------------------

    def _ensure_group(self, path: str):
        """Ensure a group path exists, creating parent groups as needed."""
        parts = path.strip("/").split("/")
        current = self._root
        for part in parts:
            if part not in current:
                current = current.create_group(part)
            else:
                current = current[part]
        return current

    def _create_timeseries_array(self, path: str, dtype: str = "float32",
                                 fill_value: float = np.nan) -> zarr.Array:
        """Create a new time series array at the given path."""
        parent_path = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1]
        parent = self._ensure_group(parent_path)

        arr = parent.create_array(
            name,
            shape=(self.n_days, self.n_fields),
            chunks=(365, min(100, self.n_fields)),
            dtype=dtype,
            fill_value=fill_value,
        )
        return arr

    def _create_property_array(self, path: str, dtype: str = "float32",
                              fill_value: float = np.nan) -> zarr.Array:
        """Create a new static property array at the given path."""
        parent_path = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1]
        parent = self._ensure_group(parent_path)

        arr = parent.create_array(
            name,
            shape=(self.n_fields,),
            chunks=(min(100, self.n_fields),),
            dtype=dtype,
            fill_value=fill_value,
        )
        return arr

    def _mark_modified(self):
        """Mark the container as having unsaved changes."""
        self._modified = True

    # -------------------------------------------------------------------------
    # Pack / Unpack Methods
    # -------------------------------------------------------------------------

    def pack(self, output_path: Union[str, Path]) -> Path:
        """
        Pack container to zip file for sharing.

        Creates a compressed copy of this container. The original
        directory is preserved.

        Args:
            output_path: Path for output zip file (.swim extension added if missing)

        Returns:
            Path to created zip file

        Raises:
            ValueError: If container is not using DirectoryStore
            FileExistsError: If output file already exists

        Example:
            # Create directory container
            container = SwimContainer.create("project.swim", ...)

            # Pack for sharing
            zip_path = container.pack("project_share.swim")
            print(f"Packed to: {zip_path}")
        """
        import shutil

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(self.EXTENSION)

        # Check if output already exists
        if output_path.exists():
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Delete it first or choose a different name."
            )

        # Check that we have a directory-based container
        if not isinstance(self._provider, DirectoryStoreProvider):
            if isinstance(self._provider, ZipStoreProvider):
                # Already a zip - just copy
                shutil.copy2(self._provider.location, output_path)
                print(f"Copied zip to: {output_path}")
                return output_path
            else:
                raise ValueError(
                    "pack() is only supported for local containers. "
                    f"Current storage type: {type(self._provider).__name__}"
                )

        # Save any pending changes first
        if self._modified:
            self._save_metadata()

        # Create zip from directory using shutil.make_archive
        source_dir = self._provider.location

        # shutil.make_archive wants the archive name without extension
        archive_base = str(output_path.with_suffix(""))

        # Create the zip archive
        shutil.make_archive(archive_base, "zip", source_dir)

        # Rename to desired extension if not .zip
        if output_path.suffix != ".zip":
            created_path = Path(archive_base + ".zip")
            created_path.rename(output_path)

        print(f"Packed to: {output_path}")
        return output_path

    def unpack(self, output_path: Union[str, Path]) -> "SwimContainer":
        """
        Unpack zip container to directory format.

        Creates a directory copy of this container. The original
        zip file is preserved.

        Args:
            output_path: Path for output directory (.swim extension added if missing)

        Returns:
            New SwimContainer pointing to unpacked directory

        Raises:
            ValueError: If container is not using ZipStore
            FileExistsError: If output directory already exists

        Example:
            # Open existing zip container
            container = SwimContainer.open("project.swim")

            # Unpack for development
            dev_container = container.unpack("project_dev.swim")
            print(f"Unpacked to: {dev_container.path}")
        """
        import shutil

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(self.EXTENSION)

        # Check if output already exists
        if output_path.exists():
            raise FileExistsError(
                f"Output path already exists: {output_path}. "
                "Delete it first or choose a different name."
            )

        # Check that we have a zip-based container
        if not isinstance(self._provider, ZipStoreProvider):
            if isinstance(self._provider, DirectoryStoreProvider):
                # Already a directory - just copy
                shutil.copytree(self._provider.location, output_path)
                print(f"Copied directory to: {output_path}")
                return SwimContainer.open(output_path, mode="r+")
            else:
                raise ValueError(
                    "unpack() is only supported for local containers. "
                    f"Current storage type: {type(self._provider).__name__}"
                )

        # Extract zip to directory using shutil.unpack_archive
        source_zip = self._provider.location
        shutil.unpack_archive(str(source_zip), str(output_path), "zip")

        print(f"Unpacked to: {output_path}")

        # Return new container pointing to the unpacked directory
        return SwimContainer.open(output_path, mode="r+")


# -------------------------------------------------------------------------
# Convenience functions
# -------------------------------------------------------------------------

def open_container(
    uri: Union[str, Path],
    mode: str = "r",
    **storage_kwargs: Any,
) -> SwimContainer:
    """
    Open an existing SWIM container.

    Auto-detects storage format:
    - If path is a file → ZipStore
    - If path is a directory → DirectoryStore

    Args:
        uri: Location of the container:
            - Local path: "project.swim", "/path/to/project.zarr"
            - S3 URI: "s3://bucket/key" (requires s3fs)
            - GCS URI: "gs://bucket/key" (requires gcsfs)
        mode: 'r' for read-only, 'r+' for read-write
        **storage_kwargs: Additional arguments for cloud storage providers

    Returns:
        SwimContainer instance

    Example:
        # Local file (auto-detects zip or directory)
        container = open_container("project.swim", mode="r+")

        # S3 storage
        container = open_container(
            "s3://bucket/project.zarr",
            mode="r",
            aws_access_key_id="...",
            aws_secret_access_key="..."
        )
    """
    return SwimContainer.open(uri, mode=mode, **storage_kwargs)


def create_container(
    uri: Union[str, Path],
    fields_shapefile: Union[str, Path],
    uid_column: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    project_name: str = None,
    overwrite: bool = False,
    storage: str = "auto",
    **storage_kwargs: Any,
) -> SwimContainer:
    """
    Create a new SWIM container from a shapefile.

    Args:
        uri: Location for the new container:
            - Local path: "project.swim" (default .swim extension added)
            - S3 URI: "s3://bucket/project.zarr"
            - GCS URI: "gs://bucket/project.zarr"
        fields_shapefile: Path to shapefile with field geometries
        uid_column: Column name containing unique field identifiers
        start_date: Start of analysis period
        end_date: End of analysis period
        project_name: Optional project name
        overwrite: If True, overwrite existing container
        storage: Storage backend for local paths:
            - "auto" (default): directory store (better for development)
            - "directory": explicit directory store
            - "zip": explicit zip store (better for sharing)
        **storage_kwargs: Additional arguments for cloud storage providers

    Returns:
        New SwimContainer instance

    Example:
        # Local directory store (default)
        container = create_container(
            "project.swim",
            fields_shapefile="fields.shp",
            uid_column="FID",
            start_date="2017-01-01",
            end_date="2023-12-31"
        )

        # Explicit zip store for sharing
        container = create_container(
            "project.swim",
            fields_shapefile="fields.shp",
            uid_column="FID",
            start_date="2017-01-01",
            end_date="2023-12-31",
            storage="zip"
        )

        # S3 storage
        container = create_container(
            "s3://bucket/project.zarr",
            fields_shapefile="fields.shp",
            uid_column="FID",
            start_date="2017-01-01",
            end_date="2023-12-31",
            aws_access_key_id="...",
            aws_secret_access_key="..."
        )
    """
    return SwimContainer.create(
        uri=uri,
        fields_shapefile=fields_shapefile,
        uid_column=uid_column,
        start_date=start_date,
        end_date=end_date,
        project_name=project_name,
        overwrite=overwrite,
        storage=storage,
        **storage_kwargs,
    )

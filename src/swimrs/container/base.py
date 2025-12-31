"""
SWIM Container Base - lifecycle management and shared infrastructure.

Provides ContainerBase with:
- File I/O and lifecycle management (open, close, save)
- State management (Zarr store, metadata, indexes)
- Context manager support
- Properties for accessing container metadata
- Helper methods for array creation
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import zarr
from filelock import FileLock

from swimrs.container.provenance import ProvenanceLog, DatasetProvenance
from swimrs.container.inventory import Inventory


class ContainerBase:
    """
    Base class providing lifecycle management and shared infrastructure.

    This class handles:
    - Opening/closing Zarr ZipStore
    - File locking for concurrent access
    - Metadata loading and saving
    - Index management (field UIDs, time index)
    - Provenance and inventory tracking
    - Helper methods for creating arrays

    Mixins inherit from this base to add domain-specific functionality
    (ingestion, computation, export, queries).
    """

    EXTENSION = ".swim"
    SCHEMA_VERSION = "1.0"

    def __init__(self, path: Union[str, Path], mode: str = "r"):
        """
        Open an existing SwimContainer.

        Args:
            path: Path to .swim file
            mode: 'r' for read-only, 'r+' for read-write, 'a' for append
        """
        self.path = Path(path)
        self._mode = mode
        self._store = None
        self._root = None
        self._lock = None
        self._provenance = None
        self._inventory = None
        self._field_uids = None
        self._uid_to_index = None
        self._time_index = None
        self._modified = False

        if not self.path.exists():
            raise FileNotFoundError(f"Container not found: {self.path}")

        self._open()

    def _open(self):
        """Open the Zarr store."""
        # Acquire file lock for write modes
        if self._mode in ("r+", "a", "w"):
            self._lock = FileLock(str(self.path) + ".lock", timeout=30)
            self._lock.acquire()

        # ZipStore only accepts 'r', 'w', 'x', 'a' - map 'r+' to 'a' for read-write
        if self._mode == "r":
            zarr_mode = "r"
        elif self._mode in ("r+", "a"):
            zarr_mode = "a"  # append mode allows read-write on existing file
        else:
            zarr_mode = self._mode

        self._store = zarr.ZipStore(str(self.path), mode=zarr_mode)
        self._root = zarr.open_group(self._store, mode="a" if zarr_mode != "r" else "r")

        # Load metadata
        self._load_metadata()

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

    @classmethod
    def create(cls, path: Union[str, Path],
               fields_shapefile: Union[str, Path],
               uid_column: str,
               start_date: Union[str, datetime],
               end_date: Union[str, datetime],
               project_name: str = None,
               overwrite: bool = False) -> "ContainerBase":
        """
        Create a new SwimContainer from a shapefile.

        Args:
            path: Path for the new .swim file
            fields_shapefile: Path to shapefile with field geometries
            uid_column: Column name containing unique field identifiers
            start_date: Start of analysis period
            end_date: End of analysis period
            project_name: Optional project name
            overwrite: If True, overwrite existing file

        Returns:
            New container instance
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(cls.EXTENSION)

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"Container already exists: {path}. Use overwrite=True to replace.")

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

        # Create Zarr store
        store = zarr.ZipStore(str(path), mode="w")
        root = zarr.open_group(store, mode="w")

        # Initialize provenance early so we can include it in single attrs update
        provenance = ProvenanceLog()
        provenance.container_created_at = datetime.now(timezone.utc).isoformat()

        # Set all root attributes in a single update to avoid duplicate .zattrs in ZipStore
        root.attrs.update({
            "project_name": project_name or path.stem,
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
        time_grp.create_dataset(
            "daily",
            data=time_index.values.astype("datetime64[ns]"),
            dtype="datetime64[ns]"
        )

        # Create geometry group
        geom_grp = root.create_group("geometry")

        # Store UIDs
        geom_grp.create_dataset(
            "uid",
            data=np.array(uids, dtype=object),
            dtype=object,
            object_codec=zarr.codecs.VLenUTF8()
        )

        # Store centroids
        centroids = gdf.geometry.centroid
        geom_grp.create_dataset("lon", data=centroids.x.values, dtype="float64")
        geom_grp.create_dataset("lat", data=centroids.y.values, dtype="float64")

        # Store area
        if gdf.crs and gdf.crs.is_projected:
            areas = gdf.geometry.area
        else:
            # Reproject to equal area for area calculation
            areas = gdf.to_crs("EPSG:6933").geometry.area
        geom_grp.create_dataset("area_m2", data=areas.values, dtype="float64")

        # Store WKB geometries
        wkb_data = gdf.geometry.apply(lambda g: g.wkb).values
        geom_grp.create_dataset(
            "wkb",
            data=wkb_data,
            dtype=object,
            object_codec=zarr.codecs.VLenBytes()
        )

        # Store original shapefile properties
        props_grp = geom_grp.create_group("properties")
        for col in gdf.columns:
            if col == "geometry" or col == uid_column:
                continue
            try:
                data = gdf[col].values
                if data.dtype == object:
                    props_grp.create_dataset(
                        col,
                        data=np.array(data.astype(str), dtype=object),
                        dtype=object,
                        object_codec=zarr.codecs.VLenUTF8()
                    )
                else:
                    props_grp.create_dataset(col, data=data)
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

        # Flush and close the store properly
        store.close()

        print(f"Created container: {path} ({n_fields} fields, {n_days} days)")

        # Reopen in append mode for continued use
        return cls(path, mode="r+")

    def close(self):
        """Close the container and release locks."""
        if self._modified:
            self._save_metadata()

        if self._store is not None:
            self._store.close()
            self._store = None
            self._root = None

        if self._lock is not None:
            self._lock.release()
            self._lock = None

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

        arr = parent.create_dataset(
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

        arr = parent.create_dataset(
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

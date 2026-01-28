"""
Container state management with native xarray support.

Provides the ContainerState class that manages the shared state between
component classes (Ingestor, Calculator, Exporter, Query) and offers
a high-level xarray interface for data access.

This module bridges the gap between low-level zarr storage and high-level
xarray operations, enabling vectorized computations while maintaining
compatibility with the existing zarr-based infrastructure.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import zarr

if TYPE_CHECKING:
    import xarray as xr
    from swimrs.container.storage import StorageProvider
    from swimrs.container.provenance import ProvenanceLog
    from swimrs.container.inventory import Inventory


class ContainerState:
    """
    Centralized state management for SwimContainer.

    Holds all shared state including the zarr root group, field UIDs,
    time index, provenance log, and inventory tracker. Provides a
    high-level xarray interface for efficient data access and computation.

    This class is designed to be passed to component classes (Ingestor,
    Calculator, Exporter, Query) enabling clean separation of concerns
    while maintaining shared access to container data.

    Attributes:
        root: The zarr root group for low-level access
        field_uids: List of field unique identifiers
        uid_to_index: Mapping from UID to array index
        time_index: DatetimeIndex for the time dimension
        provenance: ProvenanceLog for audit trail
        inventory: Inventory tracker for coverage analysis
        mode: Access mode ('r', 'r+', 'a', 'w')
        modified: Whether container has unsaved changes

    Example:
        state = ContainerState(provider, field_uids, time_index, provenance, inventory)

        # Low-level zarr access
        arr = state.root["remote_sensing/ndvi/landsat/irr"]

        # High-level xarray access
        ds = state.dataset  # Full dataset
        subset = state.get_subset(["ndvi_landsat_irr", "eto"], fields=["US-Ne1"])

        # Vectorized operations
        smoothed = subset["ndvi_landsat_irr"].rolling(time=32, center=True).mean()
    """

    def __init__(
        self,
        provider: "StorageProvider",
        field_uids: List[str],
        time_index: pd.DatetimeIndex,
        provenance: "ProvenanceLog",
        inventory: "Inventory",
        mode: str = "r",
    ):
        """
        Initialize container state.

        Args:
            provider: Storage provider with open zarr root
            field_uids: List of field UIDs
            time_index: Time index for time series data
            provenance: ProvenanceLog instance
            inventory: Inventory instance
            mode: Access mode
        """
        self._provider = provider
        self._field_uids = field_uids
        self._uid_to_index = {uid: i for i, uid in enumerate(field_uids)}
        self._time_index = time_index
        self._provenance = provenance
        self._inventory = inventory
        self._mode = mode
        self._modified = False

        # Clear cached dataset when state is modified
        self._dataset_cache: Optional["xr.Dataset"] = None

    @property
    def root(self) -> zarr.Group:
        """The zarr root group."""
        return self._provider.root

    @property
    def field_uids(self) -> List[str]:
        """List of field UIDs (copy for safety)."""
        return self._field_uids.copy()

    @property
    def uid_to_index(self) -> Dict[str, int]:
        """Mapping from UID to array index (copy for safety)."""
        return self._uid_to_index.copy()

    @property
    def time_index(self) -> pd.DatetimeIndex:
        """Time index for time series data."""
        return self._time_index

    @property
    def provenance(self) -> "ProvenanceLog":
        """Provenance log for audit trail."""
        return self._provenance

    @property
    def inventory(self) -> "Inventory":
        """Inventory tracker."""
        return self._inventory

    @property
    def mode(self) -> str:
        """Access mode."""
        return self._mode

    @property
    def modified(self) -> bool:
        """Whether container has unsaved changes."""
        return self._modified

    @property
    def n_fields(self) -> int:
        """Number of fields."""
        return len(self._field_uids)

    @property
    def n_days(self) -> int:
        """Number of days in time series."""
        return len(self._time_index) if self._time_index is not None else 0

    @property
    def is_writable(self) -> bool:
        """Whether container is open in writable mode."""
        return self._mode in ("r+", "a", "w")

    def mark_modified(self) -> None:
        """Mark the container as having unsaved changes."""
        self._modified = True
        # Invalidate dataset cache when data is modified
        self._dataset_cache = None

    def get_field_index(self, uid: str) -> int:
        """Get array index for a field UID."""
        if uid not in self._uid_to_index:
            raise KeyError(f"Unknown field UID: {uid}")
        return self._uid_to_index[uid]

    def get_time_slice(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> slice:
        """
        Get a slice for the time dimension.

        Args:
            start_date: Start date (inclusive), defaults to first date
            end_date: End date (inclusive), defaults to last date

        Returns:
            Slice object for indexing time arrays
        """
        if start_date is None:
            start_idx = 0
        else:
            start_ts = pd.Timestamp(start_date)
            start_idx = self._time_index.get_loc(start_ts)
            if isinstance(start_idx, slice):
                start_idx = start_idx.start

        if end_date is None:
            end_idx = len(self._time_index)
        else:
            end_ts = pd.Timestamp(end_date)
            end_idx = self._time_index.get_loc(end_ts)
            if isinstance(end_idx, slice):
                end_idx = end_idx.stop
            else:
                end_idx = end_idx + 1  # Make inclusive

        return slice(start_idx, end_idx)

    # -------------------------------------------------------------------------
    # Xarray Interface
    # -------------------------------------------------------------------------

    @cached_property
    def _available_paths(self) -> Dict[str, str]:
        """
        Discover available data paths in the zarr store.

        Returns a mapping from variable names to zarr paths.
        Variable names are derived from paths (e.g., "ndvi_landsat_irr"
        from "remote_sensing/ndvi/landsat/irr").
        """
        paths = {}

        def _scan_group(group: zarr.Group, prefix: str = "") -> None:
            for name in group.keys():
                item = group[name]
                full_path = f"{prefix}/{name}" if prefix else name
                if isinstance(item, zarr.Array):
                    # Convert path to variable name
                    var_name = full_path.replace("/", "_").lstrip("_")
                    # Skip geometry and time coordinate arrays
                    if not full_path.startswith(("geometry", "time")):
                        paths[var_name] = full_path
                elif isinstance(item, zarr.Group):
                    _scan_group(item, full_path)

        _scan_group(self.root)
        return paths

    def get_xarray(
        self,
        path: str,
        fields: Optional[Sequence[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        name: Optional[str] = None,
    ) -> "xr.DataArray":
        """
        Get data as a labeled xarray DataArray.

        Args:
            path: Zarr path to the array (e.g., "remote_sensing/ndvi/landsat/irr")
            fields: Optional list of field UIDs to include (default: all)
            start_date: Optional start date filter
            end_date: Optional end date filter
            name: Optional name for the DataArray

        Returns:
            xr.DataArray with 'time' and 'site' coordinates

        Example:
            ndvi = state.get_xarray("remote_sensing/ndvi/landsat/irr")
            # Use xarray operations
            smoothed = ndvi.rolling(time=32, center=True).mean()
            annual = ndvi.resample(time="YE").mean()
        """
        import xarray as xr

        if path not in self.root:
            raise KeyError(f"Path not found in container: {path}")

        arr = self.root[path]

        # Determine array type by shape
        if len(arr.shape) == 2:
            # Time series array (time, fields)
            time_slice = self.get_time_slice(start_date, end_date)
            time_coords = self._time_index[time_slice]

            if fields is not None:
                field_indices = [self.get_field_index(f) for f in fields]
                data = arr[time_slice, field_indices]
                site_coords = list(fields)
            else:
                data = arr[time_slice, :]
                site_coords = self._field_uids

            da = xr.DataArray(
                data,
                dims=["time", "site"],
                coords={"time": time_coords, "site": site_coords},
                name=name or path.replace("/", "_").lstrip("_"),
            )

        elif len(arr.shape) == 1:
            # Property array (fields,)
            if fields is not None:
                field_indices = [self.get_field_index(f) for f in fields]
                data = arr[field_indices]
                site_coords = list(fields)
            else:
                data = arr[:]
                site_coords = self._field_uids

            da = xr.DataArray(
                data,
                dims=["site"],
                coords={"site": site_coords},
                name=name or path.replace("/", "_").lstrip("_"),
            )

        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

        return da

    def get_dataset(
        self,
        paths: Optional[Dict[str, str]] = None,
        fields: Optional[Sequence[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> "xr.Dataset":
        """
        Get multiple variables as an xarray Dataset.

        Args:
            paths: Mapping of variable names to zarr paths
                   If None, loads all available time series variables
            fields: Optional list of field UIDs to include
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            xr.Dataset with requested variables

        Example:
            # Load specific variables
            ds = state.get_dataset({
                "ndvi": "remote_sensing/ndvi/landsat/irr",
                "etf": "remote_sensing/etf/landsat/ssebop/irr",
                "eto": "meteorology/gridmet/eto",
            })

            # Vectorized operations
            et = ds["etf"] * ds["eto"]
        """
        import xarray as xr

        if paths is None:
            # Load all available time series variables
            paths = {
                name: path
                for name, path in self._available_paths.items()
                if path in self.root and len(self.root[path].shape) == 2
            }

        data_vars = {}
        for var_name, path in paths.items():
            try:
                data_vars[var_name] = self.get_xarray(
                    path, fields=fields, start_date=start_date, end_date=end_date
                )
            except KeyError:
                # Skip missing paths
                continue

        return xr.Dataset(data_vars)

    def get_properties_dataset(
        self,
        properties: Optional[List[str]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> "xr.Dataset":
        """
        Get property data as an xarray Dataset.

        Args:
            properties: List of property names to include
                       (e.g., ["awc", "clay", "irr"])
                       If None, loads all available properties
            fields: Optional list of field UIDs to include

        Returns:
            xr.Dataset with property variables (site dimension only)
        """
        import xarray as xr

        # Define common property paths
        property_paths = {
            "awc": "properties/soils/awc",
            "clay": "properties/soils/clay",
            "sand": "properties/soils/sand",
            "ksat": "properties/soils/ksat",
            "modis_lc": "properties/land_cover/modis_lc",
            "cdl": "properties/land_cover/cdl",
            "irr": "properties/irrigation/irr",
            "lanid": "properties/irrigation/lanid",
            "lat": "properties/location/lat",
            "lon": "properties/location/lon",
            "elevation": "properties/location/elevation",
            "area_m2": "geometry/area_m2",
        }

        if properties is not None:
            property_paths = {
                k: v for k, v in property_paths.items() if k in properties
            }

        data_vars = {}
        for var_name, path in property_paths.items():
            if path in self.root:
                try:
                    data_vars[var_name] = self.get_xarray(path, fields=fields)
                except Exception:
                    continue

        return xr.Dataset(data_vars)

    def to_xarray_dataset(
        self,
        include_properties: bool = True,
        fields: Optional[Sequence[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> "xr.Dataset":
        """
        Load entire container as a single xarray Dataset.

        This is useful for comprehensive analysis but may use significant memory
        for large containers. For targeted access, use get_dataset() instead.

        Args:
            include_properties: Whether to include static property variables
            fields: Optional list of field UIDs to include
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            xr.Dataset with all available variables
        """
        import xarray as xr

        # Get time series data
        ds = self.get_dataset(
            fields=fields, start_date=start_date, end_date=end_date
        )

        # Add properties if requested
        if include_properties:
            props = self.get_properties_dataset(fields=fields)
            ds = xr.merge([ds, props])

        return ds

    # -------------------------------------------------------------------------
    # Array Creation (for ingestion)
    # -------------------------------------------------------------------------

    def ensure_group(self, path: str) -> zarr.Group:
        """
        Ensure a group path exists, creating parent groups as needed.

        Args:
            path: Path to the group (e.g., "remote_sensing/ndvi/landsat")

        Returns:
            The zarr Group at the specified path
        """
        parts = path.strip("/").split("/")
        current = self.root
        for part in parts:
            if part not in current:
                current = current.create_group(part)
            else:
                current = current[part]
        return current

    def create_timeseries_array(
        self,
        path: str,
        dtype: str = "float32",
        fill_value: float = np.nan,
        overwrite: bool = False,
    ) -> zarr.Array:
        """
        Create a new time series array at the given path.

        Args:
            path: Full zarr path (e.g., "remote_sensing/ndvi/landsat/irr")
            dtype: Data type for the array
            fill_value: Fill value for missing data
            overwrite: If True, overwrite existing array

        Returns:
            The created zarr Array
        """
        parent_path = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1]
        parent = self.ensure_group(parent_path)

        arr = parent.create_array(
            name,
            shape=(self.n_days, self.n_fields),
            chunks=(365, min(100, self.n_fields)),
            dtype=dtype,
            fill_value=fill_value,
            overwrite=overwrite,
        )
        return arr

    def create_property_array(
        self,
        path: str,
        dtype: str = "float32",
        fill_value: float = np.nan,
        overwrite: bool = False,
    ) -> zarr.Array:
        """
        Create a new static property array at the given path.

        Args:
            path: Full zarr path (e.g., "properties/soils/awc")
            dtype: Data type for the array
            fill_value: Fill value for missing data
            overwrite: If True, overwrite existing array

        Returns:
            The created zarr Array
        """
        parent_path = "/".join(path.split("/")[:-1])
        name = path.split("/")[-1]
        parent = self.ensure_group(parent_path)

        arr = parent.create_array(
            name,
            shape=(self.n_fields,),
            chunks=(min(100, self.n_fields),),
            dtype=dtype,
            overwrite=overwrite,
            fill_value=fill_value,
        )
        return arr

    def refresh(self) -> None:
        """
        Refresh cached data after modifications.

        Call this after ingesting new data or computing derived products
        to ensure subsequent queries see the updated data.
        """
        self._dataset_cache = None
        self._inventory.refresh()
        # Clear the cached property
        if "_available_paths" in self.__dict__:
            del self.__dict__["_available_paths"]

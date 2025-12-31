"""
SWIM Data Container - unified data management for SWIM-RS projects.

Provides a single-file container (Zarr ZipStore) that holds all project data
including geometries, remote sensing, meteorology, properties, and derived products.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from filelock import FileLock

import numpy as np
import pandas as pd
import zarr

from swimrs.container.schema import SwimSchema, Instrument, MaskType, ETModel, MetSource
from swimrs.container.provenance import ProvenanceLog, ProvenanceEvent, DatasetProvenance
from swimrs.container.inventory import Inventory, Coverage, ValidationResult, DataStatus


class SwimContainer:
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

        # Open Zarr store
        zarr_mode = "r" if self._mode == "r" else "r+"
        self._store = zarr.ZipStore(str(self.path), mode=zarr_mode)
        self._root = zarr.open_group(self._store, mode=zarr_mode)

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
               overwrite: bool = False) -> "SwimContainer":
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
            New SwimContainer instance
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

        # Set root attributes
        root.attrs["project_name"] = project_name or path.stem
        root.attrs["schema_version"] = cls.SCHEMA_VERSION
        root.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        root.attrs["start_date"] = str(start_date.date())
        root.attrs["end_date"] = str(end_date.date())
        root.attrs["n_fields"] = n_fields
        root.attrs["n_days"] = n_days
        root.attrs["uid_column"] = uid_column
        root.attrs["source_shapefile"] = str(fields_shapefile)

        # Create time coordinate
        time_grp = root.create_group("time")
        time_arr = time_grp.create_dataset(
            "daily",
            data=time_index.values.astype("datetime64[ns]"),
            dtype="datetime64[ns]"
        )

        # Create geometry group
        geom_grp = root.create_group("geometry")

        # Store UIDs
        uid_arr = geom_grp.create_dataset(
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

        # Initialize provenance
        provenance = ProvenanceLog()
        provenance.container_created_at = datetime.now(timezone.utc).isoformat()
        provenance.container_created_by = ProvenanceEvent.create("_")._asdict() if hasattr(ProvenanceEvent, '_asdict') else None

        # Record creation event
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

        root.attrs["provenance"] = provenance.to_dict()

        # Close and reopen in read-write mode
        store.close()

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
    # Status and Observability
    # -------------------------------------------------------------------------

    def status(self, detailed: bool = False) -> str:
        """
        Generate a status report for the container.

        Args:
            detailed: If True, show per-field details

        Returns:
            Formatted status string (also prints to stdout)
        """
        lines = []

        # Header
        lines.append("\u256d" + "\u2500" * 70 + "\u256e")
        lines.append(f"\u2502  SWIM Container: {self.path.name:<56}\u2502")
        lines.append(f"\u2502  Project: {self.project_name:<60}\u2502")
        lines.append("\u251c" + "\u2500" * 70 + "\u2524")
        lines.append(f"\u2502  Fields: {self.n_fields:<10} Date range: {self.start_date.date()} to {self.end_date.date():<14}\u2502")

        # Get container size
        size_mb = self.path.stat().st_size / (1024 * 1024)
        lines.append(f"\u2502  Container size: {size_mb:.1f} MB{' ' * 51}\u2502")
        lines.append("\u2570" + "\u2500" * 70 + "\u256f")
        lines.append("")

        # Geometry
        lines.append("Geometry:")
        lines.append(f"  \u2713 Imported ({self.n_fields} polygons)")
        lines.append("")

        # Remote Sensing - NDVI
        lines.append("Remote Sensing - NDVI:")
        for inst in [Instrument.LANDSAT, Instrument.SENTINEL, Instrument.ECOSTRESS]:
            for mask in [MaskType.IRR, MaskType.INV_IRR, MaskType.NO_MASK]:
                path = f"remote_sensing/ndvi/{inst.value}/{mask.value}"
                cov = self._inventory.get_coverage(path)
                lines.append(cov.summary_line())
        lines.append("")

        # Remote Sensing - ETF
        lines.append("Remote Sensing - ETF:")
        for model in ETModel:
            for mask in [MaskType.IRR, MaskType.INV_IRR, MaskType.NO_MASK]:
                path = f"remote_sensing/etf/landsat/{model.value}/{mask.value}"
                cov = self._inventory.get_coverage(path)
                if cov.status != DataStatus.NOT_PRESENT or model in [ETModel.SSEBOP, ETModel.PTJPL]:
                    lines.append(cov.summary_line())
        lines.append("")

        # Meteorology
        lines.append("Meteorology:")
        for source in [MetSource.GRIDMET, MetSource.ERA5]:
            path = f"meteorology/{source.value}/eto"
            cov = self._inventory.get_coverage(path)
            lines.append(cov.summary_line())
        lines.append("")

        # Properties
        lines.append("Properties:")
        for prop_path in ["properties/soils/awc", "properties/land_cover/modis_lc",
                         "properties/irrigation/irr"]:
            cov = self._inventory.get_coverage(prop_path)
            lines.append(cov.summary_line())
        lines.append("")

        # Snow
        lines.append("Snow:")
        cov = self._inventory.get_coverage("snow/snodas/swe")
        lines.append(cov.summary_line())
        lines.append("")

        # Derived
        lines.append("Derived:")
        for derived_path in ["derived/dynamics/ke_max", "derived/dynamics/kc_max",
                            "derived/dynamics/irr_data", "derived/combined_ndvi/ndvi"]:
            cov = self._inventory.get_coverage(derived_path)
            lines.append(cov.summary_line())
        lines.append("")

        # Model readiness
        lines.append("\u2500" * 72)
        lines.append("Model Readiness:")
        lines.append("")

        val_calib = self._inventory.validate_for_calibration()
        if val_calib.ready:
            lines.append(f"  Calibration: \u2713 Ready ({len(val_calib.ready_fields)} fields)")
        else:
            lines.append(f"  Calibration: \u2717 Not ready")
            if val_calib.missing_data:
                lines.append(f"    Missing: {', '.join(val_calib.missing_data[:3])}")

        val_forward = self._inventory.validate_for_forward_run()
        if val_forward.ready:
            lines.append(f"  Forward run: \u2713 Ready ({len(val_forward.ready_fields)} fields)")
        else:
            lines.append(f"  Forward run: \u2717 Not ready")

        lines.append("")

        # Next steps
        lines.append("\u2500" * 72)
        lines.append("Suggested next steps:")
        for step in self._inventory.suggest_next_steps()[:3]:
            lines.append(f"  {step}")

        output = "\n".join(lines)
        print(output)
        return output

    def validate(self, operation: str = "calibration", model: str = "ssebop",
                mask: str = "irr", met_source: str = "gridmet",
                snow_source: str = "snodas", instrument: str = "landsat") -> ValidationResult:
        """
        Validate container readiness for a specific operation.

        Args:
            operation: "calibration" or "forward_run"
            model: ET model to validate for
            mask: Mask type to validate for
            met_source: Meteorology source (gridmet, era5, nldas)
            snow_source: Snow data source (snodas, era5)
            instrument: Remote sensing instrument (landsat, ecostress)

        Returns:
            ValidationResult with details
        """
        if operation == "calibration":
            return self._inventory.validate_for_calibration(
                model=model, mask=mask, met_source=met_source,
                snow_source=snow_source, instrument=instrument
            )
        elif operation == "forward_run":
            return self._inventory.validate_for_forward_run(
                model=model, mask=mask, met_source=met_source, instrument=instrument
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # -------------------------------------------------------------------------
    # Data Access
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

    def get_dataframe(self, path: str,
                     fields: List[str] = None,
                     start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
        """
        Get data as a pandas DataFrame.

        Args:
            path: Data path (e.g., 'remote_sensing/ndvi/landsat/irr')
            fields: List of field UIDs (None for all)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with DatetimeIndex and field UIDs as columns
        """
        if path not in self._root:
            raise KeyError(f"Data path not found: {path}")

        arr = self._root[path][:]

        # Build field selection
        if fields is None:
            field_indices = slice(None)
            field_cols = self._field_uids
        else:
            field_indices = [self.get_field_index(f) for f in fields]
            field_cols = fields

        # Handle different array dimensions
        if arr.ndim == 2:  # Time series (time, field)
            data = arr[:, field_indices]
            df = pd.DataFrame(data, index=self._time_index, columns=field_cols)

            # Apply date filter
            if start_date:
                df = df.loc[start_date:]
            if end_date:
                df = df.loc[:end_date]

        elif arr.ndim == 1:  # Static property (field,)
            data = arr[field_indices] if isinstance(field_indices, list) else arr[:]
            df = pd.DataFrame({"value": data}, index=field_cols)

        else:
            raise ValueError(f"Unexpected array dimensions: {arr.ndim}")

        return df

    def get_field_timeseries(self, uid: str,
                            parameters: List[str] = None) -> pd.DataFrame:
        """
        Get all time series data for a single field.

        Args:
            uid: Field UID
            parameters: List of parameters to include (None for all available)

        Returns:
            DataFrame with DatetimeIndex and parameter columns
        """
        idx = self.get_field_index(uid)
        data = {}

        # Collect all available time series for this field
        paths_to_check = []
        if parameters:
            # Build paths from parameters
            for param in parameters:
                if param == "ndvi":
                    paths_to_check.append("remote_sensing/ndvi/landsat/irr")
                elif param == "etf":
                    paths_to_check.append("remote_sensing/etf/landsat/ssebop/irr")
                elif param in ["eto", "prcp", "tmin", "tmax"]:
                    paths_to_check.append(f"meteorology/gridmet/{param}")
        else:
            # Get all present paths
            paths_to_check = self._inventory.list_present_paths()

        for path in paths_to_check:
            if path in self._root:
                arr = self._root[path]
                if arr.ndim == 2:
                    col_name = path.split("/")[-1]
                    if "etf" in path:
                        col_name = path.split("/")[3] + "_etf"  # model_etf
                    elif "ndvi" in path:
                        inst = path.split("/")[2]
                        mask = path.split("/")[3]
                        col_name = f"{inst}_ndvi_{mask}"
                    data[col_name] = arr[:, idx]

        df = pd.DataFrame(data, index=self._time_index)
        return df

    def get_geodataframe(self):
        """
        Get field geometries as a GeoDataFrame.

        Returns:
            GeoDataFrame with geometries and properties
        """
        try:
            import geopandas as gpd
            from shapely import wkb
        except ImportError:
            raise ImportError("geopandas and shapely are required for get_geodataframe()")

        # Load WKB geometries
        wkb_data = self._root["geometry/wkb"][:]
        geometries = [wkb.loads(w) for w in wkb_data]

        # Build GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"uid": self._field_uids},
            geometry=geometries,
            crs="EPSG:4326"  # Assume WGS84
        )

        # Add centroids
        gdf["lon"] = self._root["geometry/lon"][:]
        gdf["lat"] = self._root["geometry/lat"][:]
        gdf["area_m2"] = self._root["geometry/area_m2"][:]

        # Add stored properties
        if "geometry/properties" in self._root:
            props_grp = self._root["geometry/properties"]
            for key in props_grp.keys():
                gdf[key] = props_grp[key][:]

        return gdf

    # -------------------------------------------------------------------------
    # Data Ingestion (stubs - to be implemented)
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

    def ingest_ee_ndvi(self, source_dir: Union[str, Path],
                       instrument: str,
                       mask: str,
                       overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest NDVI data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files
            instrument: 'landsat' or 'sentinel'
            mask: 'irr', 'inv_irr', or 'no_mask'
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        path = f"remote_sensing/ndvi/{instrument}/{mask}"

        # Check if data exists
        if path in self._root and not overwrite:
            raise ValueError(f"Data already exists at {path}. Use overwrite=True to replace.")

        # Create or get array
        if path not in self._root or overwrite:
            if path in self._root:
                del self._root[path]
            arr = self._create_timeseries_array(path)
        else:
            arr = self._root[path]

        # Read CSV files and populate array
        source_dir = Path(source_dir)
        csv_files = list(source_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {source_dir}")

        records_count = 0
        fields_found = set()

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=["date"])

            for uid in df.columns:
                if uid in ["date", "system:index", ".geo"]:
                    continue
                if uid not in self._uid_to_index:
                    continue

                field_idx = self._uid_to_index[uid]
                fields_found.add(uid)

                for _, row in df.iterrows():
                    try:
                        time_idx = self.get_time_index(row["date"])
                        value = row[uid]
                        if pd.notna(value):
                            arr[time_idx, field_idx] = value
                            records_count += 1
                    except (KeyError, IndexError):
                        continue

        # Record provenance
        event = self._provenance.record(
            "ingest",
            target=path,
            source=str(source_dir),
            source_format="earth_engine_csv",
            params={"instrument": instrument, "mask": mask},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        # Update dataset provenance
        prov = DatasetProvenance()
        prov.record_creation(event.id, source_type="earth_engine_csv")
        prov.set_coverage(
            fields_present=len(fields_found),
            fields_total=self.n_fields,
            date_range=(str(self.start_date.date()), str(self.end_date.date())),
            missing_fields=[u for u in self._field_uids if u not in fields_found],
        )
        arr.attrs.update(prov.to_dict())

        self._modified = True
        self._inventory.refresh()

        return event

    def ingest_ee_etf(self, source_dir: Union[str, Path],
                      model: str,
                      mask: str,
                      overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest ETF data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files
            model: ET model ('ssebop', 'ptjpl', etc.)
            mask: 'irr', 'inv_irr', or 'no_mask'
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        path = f"remote_sensing/etf/landsat/{model}/{mask}"

        if path in self._root and not overwrite:
            raise ValueError(f"Data already exists at {path}. Use overwrite=True to replace.")

        if path not in self._root or overwrite:
            if path in self._root:
                del self._root[path]
            arr = self._create_timeseries_array(path)
        else:
            arr = self._root[path]

        source_dir = Path(source_dir)
        csv_files = list(source_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {source_dir}")

        records_count = 0
        fields_found = set()

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=["date"])

            for uid in df.columns:
                if uid in ["date", "system:index", ".geo"]:
                    continue
                if uid not in self._uid_to_index:
                    continue

                field_idx = self._uid_to_index[uid]
                fields_found.add(uid)

                for _, row in df.iterrows():
                    try:
                        time_idx = self.get_time_index(row["date"])
                        value = row[uid]
                        if pd.notna(value):
                            arr[time_idx, field_idx] = value
                            records_count += 1
                    except (KeyError, IndexError):
                        continue

        event = self._provenance.record(
            "ingest",
            target=path,
            source=str(source_dir),
            source_format="earth_engine_csv",
            params={"model": model, "mask": mask},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        prov = DatasetProvenance()
        prov.record_creation(event.id, source_type="earth_engine_csv")
        prov.set_coverage(
            fields_present=len(fields_found),
            fields_total=self.n_fields,
            date_range=(str(self.start_date.date()), str(self.end_date.date())),
            missing_fields=[u for u in self._field_uids if u not in fields_found],
        )
        arr.attrs.update(prov.to_dict())

        self._modified = True
        self._inventory.refresh()

        return event

    def ingest_gridmet(self, source_dir: Union[str, Path],
                       variables: List[str] = None,
                       include_corrected: bool = True,
                       overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest GridMET meteorology data.

        Args:
            source_dir: Directory containing per-field CSV files
            variables: List of variables to ingest. If None, ingest all available.
                       Options: eto, etr, prcp, tmin, tmax, srad, vpd, ea, u2, elev,
                                eto_corr, etr_corr
            include_corrected: If True, also ingest bias-corrected versions if available
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        source_dir = Path(source_dir)
        csv_files = list(source_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {source_dir}")

        # All supported GridMET variables
        all_vars = ["eto", "etr", "prcp", "tmin", "tmax", "srad", "vpd", "ea", "u2", "elev"]
        corrected_vars = ["eto_corr", "etr_corr"]

        if variables is None:
            met_vars = all_vars.copy()
            if include_corrected:
                met_vars.extend(corrected_vars)
        else:
            met_vars = variables

        arrays = {}

        for var in met_vars:
            path = f"meteorology/gridmet/{var}"
            if path in self._root and not overwrite:
                continue
            if path in self._root:
                del self._root[path]
            arrays[var] = self._create_timeseries_array(path)

        if not arrays:
            raise ValueError("All meteorology data already exists. Use overwrite=True to replace.")

        records_count = 0
        fields_found = set()

        for csv_file in csv_files:
            # Field UID from filename
            uid = csv_file.stem
            if uid not in self._uid_to_index:
                continue

            field_idx = self._uid_to_index[uid]
            fields_found.add(uid)

            df = pd.read_csv(csv_file, parse_dates=["date"], index_col="date")

            for var, arr in arrays.items():
                if var not in df.columns:
                    continue
                for date, value in df[var].items():
                    try:
                        time_idx = self.get_time_index(date)
                        if pd.notna(value):
                            arr[time_idx, field_idx] = value
                            records_count += 1
                    except (KeyError, IndexError):
                        continue

        event = self._provenance.record(
            "ingest",
            target="meteorology/gridmet",
            source=str(source_dir),
            source_format="gridmet_csv",
            params={"variables": list(arrays.keys()), "include_corrected": include_corrected},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        for var, arr in arrays.items():
            prov = DatasetProvenance()
            prov.record_creation(event.id, source_type="gridmet_csv")
            prov.set_coverage(
                fields_present=len(fields_found),
                fields_total=self.n_fields,
                date_range=(str(self.start_date.date()), str(self.end_date.date())),
                missing_fields=[u for u in self._field_uids if u not in fields_found],
            )
            arr.attrs.update(prov.to_dict())

        self._modified = True
        self._inventory.refresh()

        return event

    def ingest_era5(self, source_dir: Union[str, Path],
                    variables: List[str] = None,
                    param_mapping: Dict[str, str] = None,
                    overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest ERA5-Land meteorology data from monthly Earth Engine CSV exports.

        Expected CSV format: index column is field ID, data columns are
        "<param>_<YYYYMMDD>" for each day (e.g., "eto_20170115").

        Args:
            source_dir: Directory containing monthly ERA5 CSV files
            variables: List of variables to ingest. If None, ingest all found.
                       Options: swe, eto, tmean, tmin, tmax, prcp, srad
            param_mapping: Dict to rename parameters (e.g., {"precip": "prcp"})
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        source_dir = Path(source_dir)
        csv_files = sorted([f for f in source_dir.glob("*.csv")])

        if not csv_files:
            raise ValueError(f"No CSV files found in {source_dir}")

        if param_mapping is None:
            param_mapping = {"precip": "prcp"}

        # Collect all data by field and parameter
        from collections import defaultdict
        from datetime import datetime as dt

        all_data: Dict[str, Dict[str, Dict[pd.Timestamp, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)

            for uid, row in df.iterrows():
                uid = str(uid)
                if uid not in self._uid_to_index:
                    continue

                for col_name, value in row.items():
                    parts = str(col_name).rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    param_name, date_str = parts
                    if not date_str.isdigit() or len(date_str) != 8:
                        continue

                    # Apply parameter mapping
                    param_name = param_mapping.get(param_name, param_name)

                    if variables and param_name not in variables:
                        continue

                    try:
                        date = pd.Timestamp(dt.strptime(date_str, "%Y%m%d"))
                        if pd.notna(value):
                            all_data[uid][param_name][date] = float(value)
                    except (ValueError, TypeError):
                        continue

        if not all_data:
            raise ValueError(f"No valid ERA5 data found in {source_dir}")

        # Determine which parameters were found
        all_params = set()
        for uid_data in all_data.values():
            all_params.update(uid_data.keys())

        # Create arrays for each parameter
        arrays = {}
        for param in all_params:
            path = f"meteorology/era5/{param}"
            if path in self._root and not overwrite:
                continue
            if path in self._root:
                del self._root[path]
            arrays[param] = self._create_timeseries_array(path)

        if not arrays:
            raise ValueError("All ERA5 data already exists. Use overwrite=True to replace.")

        # Populate arrays
        records_count = 0
        fields_found = set()

        for uid, param_data in all_data.items():
            if uid not in self._uid_to_index:
                continue
            field_idx = self._uid_to_index[uid]
            fields_found.add(uid)

            for param, date_values in param_data.items():
                if param not in arrays:
                    continue
                arr = arrays[param]
                for date, value in date_values.items():
                    try:
                        time_idx = self.get_time_index(date)
                        arr[time_idx, field_idx] = value
                        records_count += 1
                    except (KeyError, IndexError):
                        continue

        event = self._provenance.record(
            "ingest",
            target="meteorology/era5",
            source=str(source_dir),
            source_format="era5_land_csv",
            params={"variables": list(arrays.keys())},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        for param, arr in arrays.items():
            prov = DatasetProvenance()
            prov.record_creation(event.id, source_type="era5_land_csv")
            prov.set_coverage(
                fields_present=len(fields_found),
                fields_total=self.n_fields,
                date_range=(str(self.start_date.date()), str(self.end_date.date())),
                missing_fields=[u for u in self._field_uids if u not in fields_found],
            )
            arr.attrs.update(prov.to_dict())

        self._modified = True
        self._inventory.refresh()

        return event

    def ingest_snodas(self, source_dir: Union[str, Path],
                      overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest SNODAS snow water equivalent data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing SNODAS CSV files
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        path = "snow/snodas/swe"

        if path in self._root and not overwrite:
            raise ValueError(f"Data already exists at {path}. Use overwrite=True to replace.")

        if path in self._root:
            del self._root[path]
        arr = self._create_timeseries_array(path)

        source_dir = Path(source_dir)
        csv_files = list(source_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {source_dir}")

        records_count = 0
        fields_found = set()

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=["date"])

            for uid in df.columns:
                if uid in ["date", "system:index", ".geo"]:
                    continue
                if uid not in self._uid_to_index:
                    continue

                field_idx = self._uid_to_index[uid]
                fields_found.add(uid)

                for _, row in df.iterrows():
                    try:
                        time_idx = self.get_time_index(row["date"])
                        value = row[uid]
                        if pd.notna(value):
                            arr[time_idx, field_idx] = value
                            records_count += 1
                    except (KeyError, IndexError):
                        continue

        event = self._provenance.record(
            "ingest",
            target=path,
            source=str(source_dir),
            source_format="snodas_csv",
            params={},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        prov = DatasetProvenance()
        prov.record_creation(event.id, source_type="snodas_csv")
        prov.set_coverage(
            fields_present=len(fields_found),
            fields_total=self.n_fields,
            date_range=(str(self.start_date.date()), str(self.end_date.date())),
            missing_fields=[u for u in self._field_uids if u not in fields_found],
        )
        arr.attrs.update(prov.to_dict())

        self._modified = True
        self._inventory.refresh()

        return event

    def ingest_properties(self, lulc_csv: Union[str, Path] = None,
                          soils_csv: Union[str, Path] = None,
                          irrigation_csv: Union[str, Path] = None,
                          cdl_csv: Union[str, Path] = None,
                          uid_column: str = None,
                          lulc_key: str = "mode",
                          overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest static field properties from CSV files.

        Args:
            lulc_csv: Path to land cover CSV (MODIS LC or similar)
            soils_csv: Path to soils CSV (SSURGO or HWSD)
            irrigation_csv: Path to irrigation status CSV (IrrMapper/LANID)
            cdl_csv: Path to Crop Data Layer CSV
            uid_column: Column name for field UIDs (uses container's default if None)
            lulc_key: Column name in LULC CSV for land cover code
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        if uid_column is None:
            uid_column = self._root.attrs.get("uid_column", "FID")

        sources = []
        records_count = 0
        fields_found = set()

        # Land cover
        if lulc_csv is not None:
            lulc_path = Path(lulc_csv)
            if not lulc_path.exists():
                raise FileNotFoundError(f"LULC file not found: {lulc_csv}")

            lc_df = pd.read_csv(lulc_path, index_col=uid_column)
            sources.append(str(lulc_csv))

            # Create array for MODIS LC code
            path = "properties/land_cover/modis_lc"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path, dtype="int16", fill_value=-1)

                for uid in lc_df.index:
                    uid = str(uid)
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)

                    if lulc_key in lc_df.columns:
                        value = lc_df.loc[uid, lulc_key] if isinstance(uid, str) else lc_df.loc[str(uid), lulc_key]
                        if pd.notna(value):
                            arr[field_idx] = int(value)
                            records_count += 1

        # Soils
        if soils_csv is not None:
            soils_path = Path(soils_csv)
            if not soils_path.exists():
                raise FileNotFoundError(f"Soils file not found: {soils_csv}")

            soil_df = pd.read_csv(soils_path, index_col=uid_column)
            sources.append(str(soils_csv))

            # Handle HWSD format (single "mode" column for AWC)
            if "awc" not in soil_df.columns and "mode" in soil_df.columns:
                soil_df = soil_df.rename(columns={"mode": "awc"})

            soil_props = ["awc", "ksat", "clay", "sand"]
            for prop in soil_props:
                if prop not in soil_df.columns:
                    continue

                path = f"properties/soils/{prop}"
                if path not in self._root or overwrite:
                    if path in self._root:
                        del self._root[path]
                    arr = self._create_property_array(path)

                    for uid in soil_df.index:
                        uid = str(uid)
                        if uid not in self._uid_to_index:
                            continue
                        field_idx = self._uid_to_index[uid]
                        fields_found.add(uid)

                        value = soil_df.loc[uid, prop] if isinstance(uid, str) else soil_df.loc[str(uid), prop]
                        if pd.notna(value):
                            arr[field_idx] = float(value)
                            records_count += 1

        # Irrigation status (multi-year)
        if irrigation_csv is not None:
            irr_path = Path(irrigation_csv)
            if not irr_path.exists():
                raise FileNotFoundError(f"Irrigation file not found: {irrigation_csv}")

            irr_df = pd.read_csv(irr_path, index_col=uid_column)
            sources.append(str(irrigation_csv))

            # Drop lat/lon if present
            for col in ["LAT", "LON", "lat", "lon"]:
                if col in irr_df.columns:
                    irr_df = irr_df.drop(columns=[col])

            # Store mean irrigation fraction
            path = "properties/irrigation/irr"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path)

                for uid in irr_df.index:
                    uid = str(uid)
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)

                    # Calculate mean irrigation fraction across years
                    values = irr_df.loc[uid].values if isinstance(uid, str) else irr_df.loc[str(uid)].values
                    mean_irr = np.nanmean([float(v) for v in values if pd.notna(v)])
                    if pd.notna(mean_irr):
                        arr[field_idx] = mean_irr
                        records_count += 1

        event = self._provenance.record(
            "ingest",
            target="properties",
            source=", ".join(sources),
            source_format="property_csv",
            params={"lulc_key": lulc_key},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        self._modified = True
        self._inventory.refresh()

        return event

    def ingest_dynamics(self, dynamics_json: Union[str, Path],
                        overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest pre-computed dynamics data from JSON file.

        The JSON file should have the structure produced by SamplePlotDynamics:
        {
            "irr": {field_id: {year: {"irr_doys": [...], "f_irr": float, "irrigated": int}}},
            "gwsub": {field_id: {year: {"f_sub": float, ...}}},
            "ke_max": {field_id: float},
            "kc_max": {field_id: float}
        }

        Args:
            dynamics_json: Path to dynamics JSON file
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        import json

        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        dynamics_path = Path(dynamics_json)
        if not dynamics_path.exists():
            raise FileNotFoundError(f"Dynamics file not found: {dynamics_json}")

        with open(dynamics_path, "r") as f:
            dynamics = json.load(f)

        records_count = 0
        fields_found = set()

        # ke_max - single value per field
        if "ke_max" in dynamics:
            path = "derived/dynamics/ke_max"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path)

                for uid, value in dynamics["ke_max"].items():
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)
                    if pd.notna(value):
                        arr[field_idx] = float(value)
                        records_count += 1

        # kc_max - single value per field
        if "kc_max" in dynamics:
            path = "derived/dynamics/kc_max"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path)

                for uid, value in dynamics["kc_max"].items():
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)
                    if pd.notna(value):
                        arr[field_idx] = float(value)
                        records_count += 1

        # Irrigation data (per field per year)
        # Store as JSON strings in object arrays since structure is complex
        if "irr" in dynamics:
            path = "derived/dynamics/irr_data"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                # Store as JSON strings for each field
                parent_path = "/".join(path.split("/")[:-1])
                name = path.split("/")[-1]
                parent = self._ensure_group(parent_path)
                arr = parent.create_dataset(
                    name,
                    shape=(self.n_fields,),
                    dtype=object,
                    object_codec=zarr.codecs.VLenUTF8(),
                )

                for uid, year_data in dynamics["irr"].items():
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)
                    arr[field_idx] = json.dumps(year_data)
                    records_count += 1

        # Groundwater subsidy data (per field per year)
        if "gwsub" in dynamics:
            path = "derived/dynamics/gwsub_data"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                parent_path = "/".join(path.split("/")[:-1])
                name = path.split("/")[-1]
                parent = self._ensure_group(parent_path)
                arr = parent.create_dataset(
                    name,
                    shape=(self.n_fields,),
                    dtype=object,
                    object_codec=zarr.codecs.VLenUTF8(),
                )

                for uid, year_data in dynamics["gwsub"].items():
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)
                    arr[field_idx] = json.dumps(year_data)
                    records_count += 1

        event = self._provenance.record(
            "ingest",
            target="derived/dynamics",
            source=str(dynamics_json),
            source_format="dynamics_json",
            params={},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        self._modified = True
        self._inventory.refresh()

        return event

    def get_dynamics(self, uid: str) -> Dict[str, Any]:
        """
        Get dynamics data for a specific field.

        Args:
            uid: Field UID

        Returns:
            Dict with ke_max, kc_max, irr (per-year), gwsub (per-year)
        """
        import json

        idx = self.get_field_index(uid)
        result = {}

        # Get scalar values
        for key in ["ke_max", "kc_max"]:
            path = f"derived/dynamics/{key}"
            if path in self._root:
                result[key] = float(self._root[path][idx])

        # Get JSON data
        for key in ["irr_data", "gwsub_data"]:
            path = f"derived/dynamics/{key}"
            if path in self._root:
                json_str = self._root[path][idx]
                if json_str:
                    result[key.replace("_data", "")] = json.loads(json_str)

        return result

    def compute_dynamics(self, etf_model: str = "ssebop",
                         irr_threshold: float = 0.1,
                         masks: tuple = ("irr", "inv_irr"),
                         instruments: tuple = ("landsat",),
                         use_mask: bool = True,
                         use_lulc: bool = False,
                         lookback: int = 10,
                         fields: List[str] = None) -> ProvenanceEvent:
        """
        Compute dynamics (irrigation detection, K-parameters) from container data.

        This method exports data from the container, runs the dynamics analysis,
        and stores the results back in the container.

        Requires: NDVI, ETf, and meteorology data to already be ingested.

        Args:
            etf_model: ET model to use for analysis (e.g., 'ssebop', 'ptjpl')
            irr_threshold: Irrigation fraction threshold for classification
            masks: Mask types to use
            instruments: Instruments to use for NDVI
            use_mask: Use irrigation mask for analysis
            use_lulc: Use land cover for analysis
            lookback: Number of days to look back for irrigation detection
            fields: List of field UIDs to process (None for all)

        Returns:
            ProvenanceEvent recording the computation
        """
        if self._mode == "r":
            raise ValueError("Cannot compute: container opened in read-only mode")

        if not use_mask and not use_lulc:
            raise ValueError("Must set either use_mask=True or use_lulc=True")

        # Validate required data exists
        met_source = "gridmet" if "meteorology/gridmet/eto" in self._root else "era5"
        required = [
            f"remote_sensing/ndvi/{instruments[0]}/{masks[0]}",
            f"remote_sensing/etf/{instruments[0]}/{etf_model}/{masks[0]}",
            f"meteorology/{met_source}/eto",
            f"meteorology/{met_source}/prcp",
        ]

        missing = [p for p in required if p not in self._root]
        if missing:
            raise ValueError(f"Missing required data for dynamics computation: {missing}")

        if fields is None:
            fields = self._field_uids

        # Export temporary Parquet files for dynamics processing
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            ts_dir = Path(tmpdir) / "timeseries"
            ts_dir.mkdir()
            props_file = Path(tmpdir) / "properties.json"
            dynamics_file = Path(tmpdir) / "dynamics.json"

            # Export per-field timeseries as Parquet
            self._export_field_timeseries_for_dynamics(ts_dir, fields, met_source, etf_model, masks, instruments)

            # Export properties as JSON
            self._export_properties_for_dynamics(props_file, fields)

            # Run dynamics computation
            from swimrs.prep.dynamics import process_dynamics_batch

            process_dynamics_batch(
                str(ts_dir),
                str(props_file),
                str(dynamics_file),
                etf_target=etf_model,
                irr_threshold=irr_threshold,
                select=fields,
                masks=masks,
                instruments=instruments,
                use_mask=use_mask,
                use_lulc=use_lulc,
                lookback=lookback,
                num_workers=1,  # Single-threaded to avoid issues with temp files
            )

            # Import dynamics results back into container
            if dynamics_file.exists():
                event = self.ingest_dynamics(dynamics_file, overwrite=True)
            else:
                raise RuntimeError("Dynamics computation failed - no output file produced")

        # Update provenance to reflect computation (not just ingestion)
        self._provenance.record(
            "compute",
            target="derived/dynamics",
            params={
                "etf_model": etf_model,
                "irr_threshold": irr_threshold,
                "masks": list(masks),
                "instruments": list(instruments),
                "use_mask": use_mask,
                "use_lulc": use_lulc,
            },
            fields_affected=fields,
        )

        return event

    def _export_field_timeseries_for_dynamics(self, output_dir: Path, fields: List[str],
                                               met_source: str, etf_model: str,
                                               masks: tuple, instruments: tuple):
        """Export field timeseries in format expected by dynamics.py."""
        from swimrs.prep import COLUMN_MULTIINDEX, ACCEPTED_UNITS_MAP

        for uid in fields:
            if uid not in self._uid_to_index:
                continue

            idx = self._uid_to_index[uid]
            data = {}

            # Collect all time series for this field
            # NDVI
            for inst in instruments:
                for mask in masks:
                    path = f"remote_sensing/ndvi/{inst}/{mask}"
                    if path in self._root:
                        col = (uid, inst, "ndvi", "unitless", "none", mask)
                        data[col] = self._root[path][:, idx]

            # ETf
            for inst in instruments:
                for mask in masks:
                    path = f"remote_sensing/etf/{inst}/{etf_model}/{mask}"
                    if path in self._root:
                        col = (uid, inst, "etf", "unitless", etf_model, mask)
                        data[col] = self._root[path][:, idx]

            # Meteorology
            for var in ["eto", "prcp", "tmin", "tmax", "srad"]:
                path = f"meteorology/{met_source}/{var}"
                if path in self._root:
                    units = ACCEPTED_UNITS_MAP.get(var, "none")
                    col = (uid, "none", var, units, met_source, "no_mask")
                    data[col] = self._root[path][:, idx]

            # Snow
            path = "snow/snodas/swe"
            if path in self._root:
                col = (uid, "none", "swe", "mm", "none", "no_mask")
                data[col] = self._root[path][:, idx]

            if data:
                df = pd.DataFrame(data, index=self._time_index)
                df.columns = pd.MultiIndex.from_tuples(df.columns, names=COLUMN_MULTIINDEX)
                df.to_parquet(output_dir / f"{uid}.parquet")

    def _export_properties_for_dynamics(self, output_file: Path, fields: List[str]):
        """Export properties in format expected by dynamics.py."""
        import json
        from swimrs.prep import MAX_EFFECTIVE_ROOTING_DEPTH as RZ

        props = {}

        for uid in fields:
            if uid not in self._uid_to_index:
                continue

            idx = self._uid_to_index[uid]
            field_props = {}

            # LULC code
            path = "properties/land_cover/modis_lc"
            if path in self._root:
                lulc_code = int(self._root[path][idx])
                field_props["lulc_code"] = lulc_code
                field_props["root_depth"] = RZ.get(str(lulc_code), {}).get("rooting_depth", np.nan)
                field_props["zr_mult"] = RZ.get(str(lulc_code), {}).get("zr_multiplier", np.nan)

            # AWC
            path = "properties/soils/awc"
            if path in self._root:
                field_props["awc"] = float(self._root[path][idx])

            # Irrigation fraction (store as dict by year for compatibility)
            path = "properties/irrigation/irr"
            if path in self._root:
                mean_irr = float(self._root[path][idx])
                # Create year dict with same value (simplified)
                years = range(self.start_date.year, self.end_date.year + 1)
                field_props["irr"] = {str(yr): mean_irr for yr in years}

            # Area
            if "geometry/area_m2" in self._root:
                field_props["area_sq_m"] = float(self._root["geometry/area_m2"][idx])

            if field_props:
                props[uid] = field_props

        with open(output_file, "w") as f:
            json.dump(props, f, indent=2)

    # -------------------------------------------------------------------------
    # Export Methods (stubs - to be implemented)
    # -------------------------------------------------------------------------

    def export_shapefile(self, output_path: Union[str, Path]):
        """Export field geometries to a shapefile."""
        gdf = self.get_geodataframe()
        gdf.to_file(output_path)
        print(f"Exported shapefile to {output_path}")

    def export_csv(self, path: str, output_dir: Union[str, Path],
                  format: str = "earth_engine") -> ProvenanceEvent:
        """
        Export data to CSV format.

        Args:
            path: Data path to export
            output_dir: Output directory
            format: 'earth_engine' or 'standard'

        Returns:
            ProvenanceEvent recording the export
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_dataframe(path)

        if format == "earth_engine":
            # EE format: date column + one column per field
            df = df.reset_index()
            df = df.rename(columns={"index": "date"})
            output_file = output_dir / f"{path.replace('/', '_')}.csv"
            df.to_csv(output_file, index=False)
        else:
            # Standard format
            output_file = output_dir / f"{path.replace('/', '_')}.csv"
            df.to_csv(output_file)

        event = self._provenance.record(
            "export",
            source=path,
            target=str(output_file),
            params={"format": format},
        )

        return event

    def export_model_inputs(self, output_dir: Union[str, Path],
                           model: str = "ssebop",
                           mask: str = "irr",
                           fields: List[str] = None) -> ProvenanceEvent:
        """
        Export data in the format required by the SWIM-RS model.

        Args:
            output_dir: Output directory
            model: ET model to use
            mask: Mask type
            fields: List of field UIDs (None for all ready fields)

        Returns:
            ProvenanceEvent recording the export
        """
        # Validate readiness
        validation = self.validate("forward_run", model=model, mask=mask)

        if fields is None:
            fields = validation.ready_fields

        if not fields:
            raise ValueError("No fields are ready for model export")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export per-field JSON files (matching existing prep_plots.py format)
        # This is a placeholder - actual implementation would match the model's expected format

        event = self._provenance.record(
            "export",
            target=str(output_dir),
            params={"model": model, "mask": mask, "format": "swim_model"},
            fields_affected=fields,
        )

        print(f"Exported model inputs for {len(fields)} fields to {output_dir}")
        return event


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

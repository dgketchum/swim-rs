"""
Exporter component for data export operations.

Provides a clean API for exporting container data in various formats.
Usage: container.export.prepped_input_json(...) instead of container.export_prepped_input_json(...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import Component
from swimrs.container.schema import (
    REQUIRED_MET_BASE,
    REQUIRED_MET_HOURLY,
    REQUIRED_MET_IRR,
    REQUIRED_MET_UNIRR,
)

if TYPE_CHECKING:
    import xarray as xr
    from swimrs.container.state import ContainerState
    from swimrs.container.provenance import ProvenanceEvent


class Exporter(Component):
    """
    Component for exporting container data.

    Provides methods for exporting data to various formats including
    shapefiles, CSVs, and model-ready JSON.

    Example:
        container.export.prepped_input_json("output/prepped.json")
        container.export.shapefile("output/fields.shp")
        container.export.csv("remote_sensing/ndvi/landsat/irr", "output/ndvi/")
    """

    # Required meteorology parameters (from centralized schema)
    REQUIRED_MET = REQUIRED_MET_BASE + REQUIRED_MET_HOURLY

    def __init__(self, state: "ContainerState", container=None):
        """
        Initialize the Exporter.

        Args:
            state: ContainerState instance
            container: Optional reference to parent SwimContainer
        """
        super().__init__(state, container)

    def prepped_input_json(
        self,
        output_path: Union[str, Path],
        etf_model: str = "ssebop",
        masks: Tuple[str, ...] = ("irr", "inv_irr"),
        met_source: str = "gridmet",
        instrument: str = "landsat",
        fields: Optional[List[str]] = None,
        use_merged_ndvi: bool = True,
        irr_threshold: float = 0.1,
        include_switched_etf: bool = True,
        validate: bool = False,
        validation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ProvenanceEvent":
        """
        Export model-ready prepped input JSON file.

        Creates a JSONL file suitable for the SWIM-RS model containing:
        - Field order and properties
        - Daily time series (met, NDVI, ETf)
        - Per-year irrigation and groundwater subsidy data
        - Scalar dynamics (Ke, Kc parameters)

        Args:
            output_path: Output file path (.json)
            etf_model: ET model to use
            masks: Mask types to include
            met_source: Meteorology source
            instrument: NDVI instrument
            fields: Optional list of field UIDs (default: all)
            use_merged_ndvi: Use combined Landsat+Sentinel NDVI
            irr_threshold: Threshold for irrigation classification
            include_switched_etf: Apply ETf mask switching logic
            validate: If True, filter fields by validation rules
            validation_kwargs: Arguments for field validation

        Returns:
            ProvenanceEvent recording the operation
        """
        output_path = Path(output_path)

        with self._track_operation(
            "export_prepped_json",
            target=str(output_path),
            etf_model=etf_model,
            met_source=met_source,
        ) as ctx:
            # Determine target fields
            target_fields = fields if fields else self._state.field_uids

            # Validate fields if requested
            if validate and validation_kwargs:
                validation_result = self._state.inventory.validate_for_calibration(
                    model=etf_model,
                    met_source=met_source,
                    **validation_kwargs
                )
                target_fields = [f for f in target_fields if f in validation_result.ready_fields]

            if not target_fields:
                self._log.warning("no_fields_to_export")
                return self._state.provenance.record(
                    "export",
                    target=str(output_path),
                    params={"etf_model": etf_model},
                    records_count=0,
                    success=True,
                )

            # Build output dictionary
            output = self._build_prepped_json(
                target_fields,
                etf_model,
                masks,
                met_source,
                instrument,
                use_merged_ndvi,
                irr_threshold,
                include_switched_etf,
            )

            # Write JSONL
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for key, value in output.items():
                    json.dump({key: value}, f)
                    f.write("\n")

            ctx["records_processed"] = len(target_fields)
            ctx["fields_processed"] = len(target_fields)

            self._log.info("export_complete", path=str(output_path), fields=len(target_fields))

            event = self._state.provenance.record(
                "export",
                target=str(output_path),
                source_format="prepped_json",
                params={
                    "etf_model": etf_model,
                    "masks": list(masks),
                    "met_source": met_source,
                    "instrument": instrument,
                    "use_merged_ndvi": use_merged_ndvi,
                    "irr_threshold": irr_threshold,
                },
                fields_affected=target_fields,
                records_count=len(target_fields),
            )

            return event

    def shapefile(
        self,
        output_path: Union[str, Path],
        fields: Optional[List[str]] = None,
    ) -> "ProvenanceEvent":
        """
        Export field geometries to shapefile.

        Args:
            output_path: Output shapefile path (.shp)
            fields: Optional list of field UIDs to export

        Returns:
            ProvenanceEvent recording the operation
        """
        import geopandas as gpd
        from shapely import wkb

        output_path = Path(output_path)

        with self._track_operation(
            "export_shapefile",
            target=str(output_path),
        ) as ctx:
            target_fields = fields if fields else self._state.field_uids

            # Build GeoDataFrame from container geometry
            geometries = []
            uids = []

            wkb_arr = self._state.root["geometry/wkb"]

            for field_uid in target_fields:
                if field_uid not in self._state._uid_to_index:
                    continue
                idx = self._state._uid_to_index[field_uid]
                wkb_bytes = wkb_arr[idx]
                if wkb_bytes is not None:
                    try:
                        geom = wkb.loads(bytes(wkb_bytes))
                        geometries.append(geom)
                        uids.append(field_uid)
                    except Exception:
                        continue

            if not geometries:
                self._log.warning("no_geometries_to_export")
                return self._state.provenance.record(
                    "export",
                    target=str(output_path),
                    params={},
                    records_count=0,
                    success=True,
                )

            gdf = gpd.GeoDataFrame({"FID": uids}, geometry=geometries, crs="EPSG:4326")

            # Add properties if available
            props_ds = self._state.get_properties_dataset(fields=uids)
            for var in props_ds.data_vars:
                gdf[var] = props_ds[var].values

            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_path)

            ctx["records_processed"] = len(geometries)
            ctx["fields_processed"] = len(geometries)

            event = self._state.provenance.record(
                "export",
                target=str(output_path),
                source_format="shapefile",
                params={},
                fields_affected=uids,
                records_count=len(geometries),
            )

            return event

    def csv(
        self,
        path: str,
        output_dir: Union[str, Path],
        format: str = "wide",
        fields: Optional[List[str]] = None,
    ) -> "ProvenanceEvent":
        """
        Export data at a zarr path to CSV files.

        Args:
            path: Zarr path to export (e.g., "remote_sensing/ndvi/landsat/irr")
            output_dir: Directory for output CSV files
            format: Output format ("wide" or "long")
            fields: Optional list of field UIDs

        Returns:
            ProvenanceEvent recording the operation
        """
        output_dir = Path(output_dir)

        with self._track_operation(
            "export_csv",
            target=str(output_dir),
            source_path=path,
        ) as ctx:
            if path not in self._state.root:
                raise ValueError(f"Path not found in container: {path}")

            da = self._state.get_xarray(path, fields=fields)
            df = da.to_pandas()

            output_dir.mkdir(parents=True, exist_ok=True)

            if format == "wide":
                # Single CSV with fields as columns
                output_file = output_dir / f"{path.replace('/', '_')}.csv"
                df.to_csv(output_file)
                ctx["records_processed"] = df.size
            else:
                # One CSV per field
                for col in df.columns:
                    output_file = output_dir / f"{col}.csv"
                    df[[col]].to_csv(output_file)
                ctx["records_processed"] = df.size

            ctx["fields_processed"] = len(df.columns)

            event = self._state.provenance.record(
                "export",
                target=str(output_dir),
                source_format="csv",
                params={"source_path": path, "format": format},
                fields_affected=list(df.columns),
                records_count=int(df.size),
            )

            return event

    def model_inputs(
        self,
        output_dir: Union[str, Path],
        etf_model: str = "ssebop",
        met_source: str = "gridmet",
        fields: Optional[List[str]] = None,
    ) -> "ProvenanceEvent":
        """
        Export model inputs to directory structure.

        Creates separate files for each data type in a directory structure
        suitable for batch processing.

        Args:
            output_dir: Base directory for outputs
            etf_model: ET model
            met_source: Meteorology source
            fields: Optional list of field UIDs

        Returns:
            ProvenanceEvent recording the operation
        """
        output_dir = Path(output_dir)

        with self._track_operation(
            "export_model_inputs",
            target=str(output_dir),
            etf_model=etf_model,
        ) as ctx:
            target_fields = fields if fields else self._state.field_uids

            output_dir.mkdir(parents=True, exist_ok=True)

            # Export meteorology
            met_dir = output_dir / "meteorology"
            met_dir.mkdir(exist_ok=True)
            for var in ["eto", "prcp", "tmin", "tmax", "srad"]:
                met_path = f"meteorology/{met_source}/{var}"
                if met_path in self._state.root:
                    da = self._state.get_xarray(met_path, fields=target_fields)
                    df = da.to_pandas()
                    df.to_csv(met_dir / f"{var}.csv")

            # Export remote sensing
            rs_dir = output_dir / "remote_sensing"
            rs_dir.mkdir(exist_ok=True)
            for mask in ["irr", "inv_irr", "no_mask"]:
                ndvi_path = f"remote_sensing/ndvi/landsat/{mask}"
                if ndvi_path in self._state.root:
                    da = self._state.get_xarray(ndvi_path, fields=target_fields)
                    df = da.to_pandas()
                    df.to_csv(rs_dir / f"ndvi_{mask}.csv")

                etf_path = f"remote_sensing/etf/landsat/{etf_model}/{mask}"
                if etf_path in self._state.root:
                    da = self._state.get_xarray(etf_path, fields=target_fields)
                    df = da.to_pandas()
                    df.to_csv(rs_dir / f"etf_{mask}.csv")

            # Export dynamics
            dynamics_dir = output_dir / "dynamics"
            dynamics_dir.mkdir(exist_ok=True)
            dynamics = self._get_dynamics_dict(target_fields)
            with open(dynamics_dir / "dynamics.json", "w") as f:
                json.dump(dynamics, f, indent=2)

            ctx["records_processed"] = len(target_fields)
            ctx["fields_processed"] = len(target_fields)

            event = self._state.provenance.record(
                "export",
                target=str(output_dir),
                source_format="model_inputs",
                params={"etf_model": etf_model, "met_source": met_source},
                fields_affected=target_fields,
                records_count=len(target_fields),
            )

            return event

    def to_xarray(
        self,
        output_path: Union[str, Path],
        variables: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
    ) -> "ProvenanceEvent":
        """
        Export data as a NetCDF file via xarray.

        Args:
            output_path: Output NetCDF path (.nc)
            variables: Variables to include (default: all time series)
            fields: Fields to include (default: all)

        Returns:
            ProvenanceEvent recording the operation
        """
        import xarray as xr

        output_path = Path(output_path)

        with self._track_operation(
            "export_netcdf",
            target=str(output_path),
        ) as ctx:
            ds = self._state.get_dataset(fields=fields)

            if variables is not None:
                ds = ds[variables]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(output_path)

            ctx["records_processed"] = ds.sizes.get("time", 0) * ds.sizes.get("site", 0)
            ctx["fields_processed"] = ds.sizes.get("site", 0)

            event = self._state.provenance.record(
                "export",
                target=str(output_path),
                source_format="netcdf",
                params={"variables": variables or list(ds.data_vars)},
                fields_affected=list(ds.coords.get("site", {}).values) if "site" in ds.coords else [],
            )

            return event

    def to_dataframe(
        self,
        path: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Export a single variable as a pandas DataFrame.

        Args:
            path: Zarr path to the variable
            fields: Optional list of field UIDs

        Returns:
            pd.DataFrame with DatetimeIndex and field columns
        """
        da = self._state.get_xarray(path, fields=fields)
        return da.to_pandas()

    def observations(
        self,
        output_dir: Union[str, Path],
        etf_model: str = "ssebop",
        masks: Tuple[str, ...] = ("irr", "inv_irr"),
        irr_threshold: float = 0.1,
        fields: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> "ProvenanceEvent":
        """
        Export observation files for model calibration.

        Creates per-field numpy files compatible with the SWIM-RS calibration workflow:
        - obs_etf_{fid}.np: ETf observations with mask switching
        - obs_swe_{fid}.np: SWE observations

        The ETf mask switching logic matches prep_plots.preproc():
        - Default to inv_irr (non-irrigated) mask
        - Switch to irr mask for years where f_irr >= irr_threshold

        Args:
            output_dir: Directory for output files
            etf_model: ET model to use (e.g., "ssebop", "ptjpl")
            masks: Mask types for ETf switching
            irr_threshold: Threshold for irrigated year classification
            fields: Fields to export (default: all)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            ProvenanceEvent recording the operation
        """
        output_dir = Path(output_dir)

        with self._track_operation(
            "export_observations",
            target=str(output_dir),
            etf_model=etf_model,
        ) as ctx:
            target_fields = fields if fields else self._state.field_uids
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get time slice
            time_slice = self._state.get_time_slice(start_date, end_date)
            time_index = self._state.time_index[time_slice]

            # Get dynamics data for mask switching
            dynamics = self._get_dynamics_dict(target_fields)
            irr_data = dynamics.get("irr", {})

            # Load ETf data for both masks
            etf_data = {}
            for mask in masks:
                etf_path = f"remote_sensing/etf/landsat/{etf_model}/{mask}"
                if etf_path in self._state.root:
                    etf_data[mask] = self._state.get_xarray(
                        etf_path, fields=target_fields,
                        start_date=start_date, end_date=end_date
                    )

            # Load SWE data
            swe_data = None
            for source in ["snodas", "era5"]:
                swe_path = f"snow/{source}/swe"
                if swe_path in self._state.root:
                    swe_data = self._state.get_xarray(
                        swe_path, fields=target_fields,
                        start_date=start_date, end_date=end_date
                    )
                    break

            exported_count = 0

            for fid in target_fields:
                # Build switched ETf series
                etf_values = self._build_switched_etf(
                    fid, etf_data, irr_data, masks, irr_threshold, time_index
                )

                if etf_values is not None:
                    etf_file = output_dir / f"obs_etf_{fid}.np"
                    np.savetxt(etf_file, etf_values)
                    exported_count += 1

                # Export SWE
                if swe_data is not None:
                    try:
                        swe_values = swe_data.sel(site=fid).values
                        swe_file = output_dir / f"obs_swe_{fid}.np"
                        np.savetxt(swe_file, swe_values)
                    except KeyError:
                        pass

            ctx["records_processed"] = exported_count
            ctx["fields_processed"] = len(target_fields)

            self._log.info(
                "observations_export_complete",
                path=str(output_dir),
                fields=exported_count,
            )

            event = self._state.provenance.record(
                "export",
                target=str(output_dir),
                source_format="observations",
                params={
                    "etf_model": etf_model,
                    "masks": list(masks),
                    "irr_threshold": irr_threshold,
                },
                fields_affected=target_fields,
                records_count=exported_count,
            )

            return event

    def _build_switched_etf(
        self,
        fid: str,
        etf_data: Dict[str, "xr.DataArray"],
        irr_data: Dict[str, Dict],
        masks: Tuple[str, ...],
        irr_threshold: float,
        time_index: pd.DatetimeIndex,
    ) -> Optional[np.ndarray]:
        """
        Build ETf array with mask switching based on irrigation status.

        Logic matches prep_plots.preproc():
        - Start with inv_irr (non-irrigated) mask as base
        - For years where f_irr >= irr_threshold, use irr mask
        """
        # Determine base mask (prefer inv_irr, fall back to no_mask or irr)
        if "inv_irr" in etf_data:
            base_mask = "inv_irr"
        elif "no_mask" in etf_data:
            base_mask = "no_mask"
        elif "irr" in etf_data:
            base_mask = "irr"
        else:
            return None

        try:
            etf_values = etf_data[base_mask].sel(site=fid).values.copy()
        except KeyError:
            return None

        # Switch to irrigated mask for irrigated years
        if "irr" in etf_data and fid in irr_data:
            field_irr = irr_data[fid]

            # Find irrigated years
            irr_years = []
            for k, v in field_irr.items():
                if k == "fallow_years":
                    continue
                try:
                    if isinstance(v, dict) and v.get("f_irr", 0.0) >= irr_threshold:
                        irr_years.append(int(k))
                except (ValueError, TypeError):
                    continue

            # Switch mask for irrigated years
            if irr_years:
                try:
                    irr_etf = etf_data["irr"].sel(site=fid).values
                    year_array = time_index.year

                    for yr in irr_years:
                        yr_mask = year_array == yr
                        etf_values[yr_mask] = irr_etf[yr_mask]
                except (KeyError, IndexError):
                    pass

        return etf_values

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_prepped_json(
        self,
        fields: List[str],
        etf_model: str,
        masks: Tuple[str, ...],
        met_source: str,
        instrument: str,
        use_merged_ndvi: bool,
        irr_threshold: float,
        include_switched_etf: bool,
    ) -> Dict[str, Any]:
        """Build the complete prepped JSON structure."""
        output = {}

        # Properties
        output["props"] = self._get_properties_dict(fields)

        # Dynamics
        dynamics = self._get_dynamics_dict(fields)
        output["irr_data"] = dynamics.get("irr", {})
        output["gwsub_data"] = dynamics.get("gwsub", {})
        output["ke_max"] = dynamics.get("ke_max", {})
        output["kc_max"] = dynamics.get("kc_max", {})

        # Build time series
        order = []
        time_series = {}

        # Get required data
        ds = self._build_export_dataset(fields, etf_model, masks, met_source, instrument, use_merged_ndvi)

        if ds is None:
            output["order"] = []
            output["time_series"] = {}
            return output

        # Apply ETf switching if requested
        if include_switched_etf:
            ds = self._apply_etf_switching(ds, etf_model, masks, irr_threshold, dynamics.get("irr", {}))

        # Build time series data
        time_index = pd.DatetimeIndex(ds.coords["time"].values)

        for field_uid in fields:
            if field_uid not in ds.coords["site"].values:
                continue
            order.append(field_uid)

        # Initialize time series structure
        for i, t in enumerate(time_index):
            dt_str = f"{t.year}-{t.month:02d}-{t.day:02d}"
            time_series[dt_str] = {"doy": int(t.dayofyear)}

        # Add data for each parameter
        for var_name in ds.data_vars:
            var_data = ds[var_name]
            for i, t in enumerate(time_index):
                dt_str = f"{t.year}-{t.month:02d}-{t.day:02d}"
                values = var_data.isel(time=i).values.tolist()
                time_series[dt_str][var_name] = values

        # Add hourly precip (derived from daily)
        if "prcp" in ds:
            prcp_data = ds["prcp"]
            for hr in range(24):
                hr_name = f"prcp_hr_{hr:02d}"
                for i, t in enumerate(time_index):
                    dt_str = f"{t.year}-{t.month:02d}-{t.day:02d}"
                    values = (prcp_data.isel(time=i).values / 24.0).tolist()
                    time_series[dt_str][hr_name] = values

        # Add nld_ppt_d (same as prcp)
        if "prcp" in ds:
            for i, t in enumerate(time_index):
                dt_str = f"{t.year}-{t.month:02d}-{t.day:02d}"
                time_series[dt_str]["nld_ppt_d"] = time_series[dt_str].get("prcp", [])

        output["order"] = order
        output["time_series"] = time_series
        output["missing"] = [f for f in fields if f not in order]

        return output

    def _get_properties_dict(self, fields: List[str]) -> Dict[str, Dict]:
        """Get properties for all fields as a dictionary."""
        props = {}

        props_ds = self._state.get_properties_dataset(fields=fields)

        for field_uid in fields:
            if field_uid not in props_ds.coords["site"].values:
                continue

            field_props = {}
            for var in props_ds.data_vars:
                value = props_ds[var].sel(site=field_uid).values
                if not np.isnan(value):
                    field_props[var] = float(value)

            # Add LULC code mapping
            if "modis_lc" in field_props:
                field_props["lulc_code"] = int(field_props["modis_lc"])

            props[field_uid] = field_props

        return props

    def _get_dynamics_dict(self, fields: List[str]) -> Dict[str, Dict]:
        """Get dynamics data for all fields as a dictionary."""
        dynamics = {"irr": {}, "gwsub": {}, "ke_max": {}, "kc_max": {}}

        # K parameters
        for k_type in ["ke_max", "kc_max"]:
            path = f"derived/dynamics/{k_type}"
            if path in self._state.root:
                arr = self._state.root[path]
                for field_uid in fields:
                    if field_uid in self._state._uid_to_index:
                        idx = self._state._uid_to_index[field_uid]
                        value = arr[idx]
                        if not np.isnan(value):
                            dynamics[k_type][field_uid] = float(value)

        # Complex data (JSON strings)
        for data_type in ["irr_data", "gwsub_data"]:
            path = f"derived/dynamics/{data_type}"
            key = data_type.replace("_data", "")
            if path in self._state.root:
                arr = self._state.root[path]
                for field_uid in fields:
                    if field_uid in self._state._uid_to_index:
                        idx = self._state._uid_to_index[field_uid]
                        data = arr[idx]
                        # zarr v3 returns 0-d ndarray for scalar indexing
                        if hasattr(data, 'item'):
                            data = data.item()
                        if data is not None and data != "":
                            try:
                                dynamics[key][field_uid] = json.loads(data)
                            except (json.JSONDecodeError, TypeError):
                                pass

        return dynamics

    def _build_export_dataset(
        self,
        fields: List[str],
        etf_model: str,
        masks: Tuple[str, ...],
        met_source: str,
        instrument: str,
        use_merged_ndvi: bool,
    ) -> Optional["xr.Dataset"]:
        """Build dataset with all required variables for export."""
        import xarray as xr

        paths = {}

        # Meteorology
        for var in ["eto", "eto_corr", "prcp", "tmin", "tmax", "srad"]:
            met_path = f"meteorology/{met_source}/{var}"
            if met_path in self._state.root:
                paths[var] = met_path

        # Snow
        for source in ["snodas", "era5"]:
            swe_path = f"snow/{source}/swe"
            if swe_path in self._state.root:
                paths["swe"] = swe_path
                break

        # ETf for each mask
        for mask in masks:
            etf_path = f"remote_sensing/etf/{instrument}/{etf_model}/{mask}"
            if etf_path in self._state.root:
                paths[f"etf_{mask}"] = etf_path

        if not paths:
            return None

        # Load base dataset (met, snow, ETf)
        ds = self._state.get_dataset(paths, fields=fields)

        # NDVI: combine from all masks using fillna() to handle fields
        # that only have data in one mask (e.g., non-irrigated stations
        # only have inv_irr data, irrigated stations only have irr data)
        ndvi_data = None
        for mask in masks:
            if use_merged_ndvi:
                merged_path = f"derived/merged_ndvi/{mask}"
                ndvi_path = f"remote_sensing/ndvi/{instrument}/{mask}"
                path = merged_path if merged_path in self._state.root else ndvi_path
            else:
                path = f"remote_sensing/ndvi/{instrument}/{mask}"

            if path not in self._state.root:
                continue

            mask_ndvi = self._state.get_xarray(path, fields=fields)
            if ndvi_data is None:
                ndvi_data = mask_ndvi
            else:
                # Fill NaN values from secondary mask
                ndvi_data = ndvi_data.fillna(mask_ndvi)

        if ndvi_data is not None:
            ds["ndvi"] = ndvi_data

        return ds

    def _apply_etf_switching(
        self,
        ds: "xr.Dataset",
        etf_model: str,
        masks: Tuple[str, ...],
        irr_threshold: float,
        irr_data: Dict[str, Dict],
    ) -> "xr.Dataset":
        """
        Apply ETf mask switching based on irrigation status.

        Logic matches legacy prep_plots.preproc():
        1. Start with non-irrigated (inv_irr) mask as base
        2. Fill NaN from irrigated (irr) mask for fields with only irr data
        3. For irrigated years (f_irr >= threshold), use irr mask values

        This ensures fields that only have data in one mask still get values,
        while applying proper mask switching for mixed fields.
        """
        import xarray as xr

        # Determine which ETf variables are available
        has_inv_irr = "etf_inv_irr" in ds
        has_irr = "etf_irr" in ds
        has_no_mask = "etf_no_mask" in ds

        if not (has_inv_irr or has_irr or has_no_mask):
            return ds

        # Build base ETf by combining masks with fillna()
        # Priority: inv_irr -> irr -> no_mask (for base values)
        if has_inv_irr:
            etf = ds["etf_inv_irr"].copy()
            # Fill NaN from irr mask (for fields that only have irrigated data)
            if has_irr:
                etf = etf.fillna(ds["etf_irr"])
            if has_no_mask:
                etf = etf.fillna(ds["etf_no_mask"])
        elif has_irr:
            etf = ds["etf_irr"].copy()
            if has_no_mask:
                etf = etf.fillna(ds["etf_no_mask"])
        else:
            etf = ds["etf_no_mask"].copy()

        # Apply year-based switching: for irrigated years, use irr mask values
        # This overrides the base values for years where the field is irrigated
        if has_irr:
            for site in ds.coords["site"].values:
                site_irr = irr_data.get(str(site), {})

                # Find irrigated years (f_irr >= threshold)
                irr_years = []
                for k, v in site_irr.items():
                    if k == "fallow_years":
                        continue
                    try:
                        if isinstance(v, dict) and v.get("f_irr", 0.0) >= irr_threshold:
                            irr_years.append(int(k))
                    except (ValueError, TypeError):
                        continue

                if irr_years:
                    time_idx = pd.DatetimeIndex(ds.coords["time"].values)
                    irr_mask = time_idx.year.isin(irr_years)
                    etf.loc[dict(site=site, time=irr_mask)] = \
                        ds["etf_irr"].sel(site=site).isel(time=irr_mask)

        # Add combined ETf to dataset
        ds["etf"] = etf

        # Add model-specific ETf name
        ds[f"{etf_model}_etf"] = etf

        return ds

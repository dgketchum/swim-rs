"""
Exporter component for data export operations.

Provides a clean API for exporting container data in various formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from swimrs.container.schema import (
    find_swe_path,
    get_rooting_code,
    get_rooting_depth,
    is_cropland,
)

from .base import Component

if TYPE_CHECKING:
    import xarray as xr

    from swimrs.container.provenance import ProvenanceEvent
    from swimrs.container.state import ContainerState


class Exporter(Component):
    """
    Component for exporting container data.

    Provides methods for exporting data to various formats including
    shapefiles, CSVs, and observation files for calibration.

    Example:
        container.export.shapefile("output/fields.shp")
        container.export.csv("remote_sensing/ndvi/landsat/irr", "output/ndvi/")
        container.export.observations("output/obs/", etf_model="ssebop")
    """

    def __init__(self, state: ContainerState, container=None):
        """
        Initialize the Exporter.

        Args:
            state: ContainerState instance
            container: Optional reference to parent SwimContainer
        """
        super().__init__(state, container)

    def shapefile(
        self,
        output_path: str | Path,
        fields: list[str] | None = None,
    ) -> ProvenanceEvent:
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
        output_dir: str | Path,
        format: str = "wide",
        fields: list[str] | None = None,
    ) -> ProvenanceEvent:
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
        output_dir: str | Path,
        etf_model: str = "ssebop",
        met_source: str = "gridmet",
        fields: list[str] | None = None,
    ) -> ProvenanceEvent:
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
        output_path: str | Path,
        variables: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> ProvenanceEvent:
        """
        Export data as a NetCDF file via xarray.

        Args:
            output_path: Output NetCDF path (.nc)
            variables: Variables to include (default: all time series)
            fields: Fields to include (default: all)

        Returns:
            ProvenanceEvent recording the operation
        """

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
                fields_affected=list(ds.coords.get("site", {}).values)
                if "site" in ds.coords
                else [],
            )

            return event

    def to_dataframe(
        self,
        path: str,
        fields: list[str] | None = None,
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
        output_dir: str | Path,
        etf_model: str = "ssebop",
        etf_instrument: str = "landsat",
        masks: tuple[str, ...] = ("irr", "inv_irr"),
        irr_threshold: float = 0.1,
        fields: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        ensemble_source: str | None = None,
        ensemble_members: list[str] | None = None,
        auxiliary_model: str | None = None,
        auxiliary_instrument: str | None = None,
    ) -> ProvenanceEvent:
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
            auxiliary_model: Optional auxiliary ETf model filling only dates
                where the primary target has no retrieval (additional-date
                design: a date with any primary value never takes the
                auxiliary). Requires auxiliary_instrument.
            auxiliary_instrument: Instrument for the auxiliary ETf source.

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

            # Load ETf data for masks (optionally ensemble mean across models)
            etf_data = {}
            if etf_model == "ensemble":
                _ens_source = ensemble_source or "computed"
                if _ens_source == "openet":
                    # Use OpenET's pre-computed ensemble directly
                    for mask in masks:
                        etf_path = f"remote_sensing/etf/{etf_instrument}/ensemble/{mask}"
                        if etf_path in self._state.root:
                            etf_data[mask] = self._state.get_xarray(
                                etf_path,
                                fields=target_fields,
                                start_date=start_date,
                                end_date=end_date,
                            )
                else:
                    # Compute mean across individual models (DIY).
                    # Use frozen member list if provided, else discover from container.
                    import xarray as xr

                    etf_prefix = f"remote_sensing/etf/{etf_instrument}"
                    if ensemble_members:
                        model_names = list(ensemble_members)
                    else:
                        model_names = []
                        if etf_prefix in self._state.root:
                            try:
                                model_names = sorted(self._state.root[etf_prefix].keys())
                            except Exception:
                                model_names = []

                    for mask in masks:
                        model_arrays = []
                        for model_name in model_names:
                            etf_path = f"{etf_prefix}/{model_name}/{mask}"
                            if etf_path not in self._state.root:
                                continue
                            model_arrays.append(
                                self._state.get_xarray(
                                    etf_path,
                                    fields=target_fields,
                                    start_date=start_date,
                                    end_date=end_date,
                                )
                            )
                        if model_arrays:
                            stacked = xr.concat(model_arrays, dim="model")
                            etf_data[mask] = stacked.mean(dim="model")
            else:
                for mask in masks:
                    etf_path = f"remote_sensing/etf/{etf_instrument}/{etf_model}/{mask}"
                    if etf_path in self._state.root:
                        etf_data[mask] = self._state.get_xarray(
                            etf_path,
                            fields=target_fields,
                            start_date=start_date,
                            end_date=end_date,
                        )

            # Load auxiliary ETf (additional-date source, e.g. ECOSTRESS PT-JPL)
            aux_data = {}
            if auxiliary_model and auxiliary_instrument:
                for mask in masks:
                    aux_path = f"remote_sensing/etf/{auxiliary_instrument}/{auxiliary_model}/{mask}"
                    if aux_path in self._state.root:
                        aux_data[mask] = self._state.get_xarray(
                            aux_path,
                            fields=target_fields,
                            start_date=start_date,
                            end_date=end_date,
                        )
                if not aux_data:
                    raise ValueError(
                        f"Auxiliary ETf source remote_sensing/etf/{auxiliary_instrument}/"
                        f"{auxiliary_model} not found in container for masks {masks}."
                    )

            # Load SWE data
            swe_data = None
            swe_path = find_swe_path(self._state.root)
            if swe_path is not None:
                swe_data = self._state.get_xarray(
                    swe_path, fields=target_fields, start_date=start_date, end_date=end_date
                )

            exported_count = 0

            for fid in target_fields:
                # Build switched ETf series
                etf_values = self._build_switched_etf(
                    fid, etf_data, irr_data, masks, irr_threshold, time_index
                )

                # Always write an ETf file so calibration tooling has a consistent
                # set of inputs; missing observations remain NaN and should be
                # given zero weight downstream.
                if etf_values is None:
                    etf_values = np.full(len(time_index), np.nan, dtype=float)

                if aux_data:
                    aux_values = self._build_switched_etf(
                        fid, aux_data, irr_data, masks, irr_threshold, time_index
                    )
                    if aux_values is not None:
                        etf_values = self._fill_auxiliary(etf_values, aux_values)

                etf_file = output_dir / f"obs_etf_{fid}.np"
                np.savetxt(etf_file, etf_values)
                exported_count += 1

                # Export SWE
                swe_values = None
                if swe_data is not None:
                    try:
                        swe_values = swe_data.sel(site=fid).values
                    except KeyError:
                        swe_values = None
                if swe_values is None:
                    swe_values = np.full(len(time_index), np.nan, dtype=float)
                swe_file = output_dir / f"obs_swe_{fid}.np"
                np.savetxt(swe_file, swe_values)

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
                    "auxiliary_model": auxiliary_model,
                    "auxiliary_instrument": auxiliary_instrument,
                },
                fields_affected=target_fields,
                records_count=exported_count,
            )

            return event

    @staticmethod
    def _fill_auxiliary(primary: np.ndarray, auxiliary: np.ndarray) -> np.ndarray:
        """Fill primary-ETf gaps with auxiliary values (additional-date design).

        A date takes the auxiliary value only when the primary has no retrieval
        at all; any date with a finite primary value keeps it untouched, so the
        primary observation set is preserved exactly.
        """
        out = np.asarray(primary, dtype=float).copy()
        aux = np.asarray(auxiliary, dtype=float)
        fill = ~np.isfinite(out) & np.isfinite(aux)
        out[fill] = aux[fill]
        return out

    def _build_switched_etf(
        self,
        fid: str,
        etf_data: dict[str, xr.DataArray],
        irr_data: dict[str, dict],
        masks: tuple[str, ...],
        irr_threshold: float,
        time_index: pd.DatetimeIndex,
    ) -> np.ndarray | None:
        """
        Build ETf array with mask switching based on irrigation status.

        Logic matches prep_plots.preproc():
        - Start with inv_irr (non-irrigated) mask as base
        - For years where f_irr >= irr_threshold, use irr mask
        """
        # Only consider masks that were requested
        available = [m for m in masks if m in etf_data]
        if not available:
            return None

        if "no_mask" in available and len(available) == 1:
            # Strict no_mask mode — use unmasked ETf directly, no switching
            try:
                return etf_data["no_mask"].sel(site=fid).values.copy()
            except KeyError:
                return None

        # Irrigation mode — base on inv_irr, switch to irr for irrigated years
        if "inv_irr" in available:
            base_mask = "inv_irr"
        elif "irr" in available:
            base_mask = "irr"
        else:
            return None

        try:
            etf_values = etf_data[base_mask].sel(site=fid).values.copy()
        except KeyError:
            return None

        if "irr" in available and fid in irr_data:
            field_irr = irr_data[fid]

            irr_years = []
            for k, v in field_irr.items():
                if k == "fallow_years":
                    continue
                try:
                    if isinstance(v, dict) and v.get("f_irr", 0.0) >= irr_threshold:
                        irr_years.append(int(k))
                except (ValueError, TypeError):
                    continue

            if irr_years:
                try:
                    irr_etf = etf_data["irr"].sel(site=fid).values
                    year_array = time_index.year

                    for yr in irr_years:
                        yr_mask = year_array == yr
                        # A year with no irr-mask retrievals means IrrMapper
                        # mapped no pixels for the field that year, so the
                        # inv_irr composite covers the whole field — keep it
                        # rather than blanking the year (mirrors the NDVI
                        # switching guard in process/input.py).
                        if not np.any(np.isfinite(irr_etf[yr_mask])):
                            continue
                        etf_values[yr_mask] = irr_etf[yr_mask]
                except (KeyError, IndexError):
                    pass

        return etf_values

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_properties_dict(self, fields: list[str]) -> dict[str, dict]:
        """
        Get field properties as a dictionary.

        Used by build_swim_input to extract properties for HDF5 construction.

        Args:
            fields: List of field UIDs to get properties for

        Returns:
            Dict mapping field UIDs to their property dictionaries
        """
        props = {}

        # Get properties dataset
        props_ds = self._state.get_properties_dataset(fields=fields)

        # Property names mapping (container names -> standard names)
        prop_map = {
            "awc": "awc",
            "rew": "rew",
            "tew": "tew",
            "ksat": "ksat",
            "cn2": "cn2",
            "zr_min": "zr_min",
            "zr_max": "zr_max",
            "mad": "mad",
            # Handle legacy name
            "p_depletion": "mad",
        }

        for field_uid in fields:
            if field_uid not in self._state._uid_to_index:
                continue

            field_props = {}

            for ds_name, out_name in prop_map.items():
                if ds_name in props_ds:
                    try:
                        val = props_ds[ds_name].sel(site=field_uid).values
                        if not np.isnan(val):
                            field_props[out_name] = float(val)
                    except (KeyError, TypeError):
                        continue

            # Add boolean properties
            for bool_prop in ["irr_status", "perennial", "gw_status"]:
                if bool_prop in props_ds:
                    try:
                        val = props_ds[bool_prop].sel(site=field_uid).values
                        field_props[bool_prop] = bool(val)
                    except (KeyError, TypeError):
                        continue

            # Derive lulc_class, lulc_source, root_depth from GLC10 (primary) or MODIS (fallback)
            lulc_class = None
            lulc_source = None

            if "glc10" in props_ds:
                try:
                    val = props_ds["glc10"].sel(site=field_uid).values
                    if not np.isnan(val) and int(val) != -1:
                        lulc_class = int(val)
                        lulc_source = "glc10"
                except (KeyError, TypeError):
                    pass

            if lulc_class is None and "modis_lc" in props_ds:
                try:
                    val = props_ds["modis_lc"].sel(site=field_uid).values
                    if not np.isnan(val) and int(val) != -1:
                        lulc_class = int(val)
                        lulc_source = "modis"
                except (KeyError, TypeError):
                    pass

            # CDL-cultivated override: asymmetric — can only rescue a unit
            # from perennial mechanics, never push one into them.
            cultivated = False
            if "cdl_cultivated" in props_ds:
                try:
                    val = props_ds["cdl_cultivated"].sel(site=field_uid).values
                    cultivated = int(val) == 1
                except (KeyError, TypeError, ValueError):
                    pass
            field_props["cultivated"] = cultivated

            if lulc_class is not None:
                field_props["lulc_class"] = lulc_class
                field_props["lulc_source"] = lulc_source
                if cultivated and not is_cropland(lulc_class, lulc_source):
                    # Cultivated history: cropland rooting, not desert/grassland
                    rooting_code = 12
                else:
                    rooting_code = get_rooting_code(lulc_class, lulc_source)
                field_props["root_depth"] = get_rooting_depth(rooting_code)

            if field_props:
                props[field_uid] = field_props

        return props

    def _get_dynamics_dict(self, fields: list[str]) -> dict[str, dict]:
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
                        if hasattr(data, "item"):
                            data = data.item()
                        if data is not None and data != "":
                            try:
                                dynamics[key][field_uid] = json.loads(data)
                            except (json.JSONDecodeError, TypeError):
                                pass

        return dynamics

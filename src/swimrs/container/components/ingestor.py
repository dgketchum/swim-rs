"""
Ingestor component for data ingestion operations.

Provides a clean, namespace-organized API for ingesting data into the container.
Usage: container.ingest.ndvi(...) instead of container.ingest_ee_ndvi(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import Component

if TYPE_CHECKING:
    from swimrs.container.components.grid_mapping import GridMapping
    from swimrs.container.provenance import ProvenanceEvent
    from swimrs.container.state import ContainerState


class Ingestor(Component):
    """
    Component for ingesting data into the container.

    Provides methods for ingesting remote sensing data, meteorology,
    properties, and other data sources. All methods use bulk xarray
    operations for efficiency and record provenance for audit trails.

    Example:
        container.ingest.ndvi(source_dir, instrument="landsat", mask="irr")
        container.ingest.gridmet(met_dir)
        container.ingest.properties(lulc_csv="lulc.csv", soils_csv="soils.csv")
    """

    def __init__(self, state: ContainerState, container=None):
        """
        Initialize the Ingestor.

        Args:
            state: ContainerState instance
            container: Optional reference to parent SwimContainer
        """
        super().__init__(state, container)

    # -------------------------------------------------------------------------
    # Remote Sensing Ingestion
    # -------------------------------------------------------------------------

    def ndvi(
        self,
        source_dir: str | Path,
        uid_column: str = "FID",
        instrument: str = "landsat",
        mask: str = "irr",
        fields: list[str] | None = None,
        overwrite: bool = False,
        min_ndvi: float = 0.05,
        apply_consecutive_filter: bool = True,
    ) -> ProvenanceEvent:
        """
        Ingest NDVI data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files
            uid_column: Column name for field UID in CSVs (default: "FID")
            instrument: Source instrument ("landsat", "sentinel", "ecostress")
            mask: Mask type ("irr", "inv_irr", "no_mask")
            fields: Optional list of field UIDs to process (default: all)
            overwrite: If True, replace existing data
            min_ndvi: Minimum valid NDVI value (default: 0.05)
            apply_consecutive_filter: Remove lower of consecutive-day observations

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        source_dir = Path(source_dir)
        path = f"remote_sensing/ndvi/{instrument}/{mask}"

        with self._track_operation(
            "ingest_ndvi",
            target=path,
            source=str(source_dir),
            instrument=instrument,
            mask=mask,
        ) as ctx:
            # Check if data exists
            if path in self._state.root and not overwrite:
                raise ValueError(f"Data exists at {path}. Use overwrite=True.")

            # Parse all CSVs into unified DataFrame
            all_data = self._parse_ee_remote_sensing_csvs(
                source_dir, instrument, "ndvi", uid_column, fields, mask=mask
            )

            if all_data.empty:
                self._log.warning("no_data_found", source=str(source_dir))
                return self._state.provenance.record(
                    "ingest",
                    target=path,
                    source=str(source_dir),
                    params={"instrument": instrument, "mask": mask},
                    records_count=0,
                    success=True,
                )

            # Apply quality filters
            all_data = self._apply_ndvi_filters(all_data, min_ndvi, apply_consecutive_filter)

            # Align to container grid and write
            records = self._write_timeseries(path, all_data, fields, overwrite=overwrite)

            ctx["records_processed"] = records
            ctx["fields_processed"] = len(all_data.columns)

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target=path,
                source=str(source_dir),
                source_format="earth_engine_csv",
                params={
                    "instrument": instrument,
                    "mask": mask,
                    "min_ndvi": min_ndvi,
                    "apply_consecutive_filter": apply_consecutive_filter,
                },
                fields_affected=list(all_data.columns),
                records_count=records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def etf(
        self,
        source_dir: str | Path,
        uid_column: str = "FID",
        model: str = "ssebop",
        mask: str = "irr",
        instrument: str = "landsat",
        fields: list[str] | None = None,
        overwrite: bool = False,
        min_etf: float = 0.05,
    ) -> ProvenanceEvent:
        """
        Ingest ET fraction data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files
            uid_column: Column name for field UID in CSVs (default: "FID")
            model: ET model ("ssebop", "ptjpl", "sims", "eemetric", etc.)
            mask: Mask type ("irr", "inv_irr", "no_mask")
            instrument: Source instrument ("landsat", "ecostress")
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data
            min_etf: Minimum valid ETf value (default: 0.05). Values below
                this are treated as noise/artifacts and set to NaN.

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        source_dir = Path(source_dir)
        path = f"remote_sensing/etf/{instrument}/{model}/{mask}"

        with self._track_operation(
            "ingest_etf",
            target=path,
            source=str(source_dir),
            model=model,
            instrument=instrument,
            mask=mask,
        ) as ctx:
            # Check if data exists
            if path in self._state.root and not overwrite:
                raise ValueError(f"Data exists at {path}. Use overwrite=True.")

            # Parse all CSVs into unified DataFrame
            all_data = self._parse_ee_remote_sensing_csvs(
                source_dir, instrument, "etf", uid_column, fields, mask=mask
            )

            if all_data.empty:
                self._log.warning("no_data_found", source=str(source_dir))
                return self._state.provenance.record(
                    "ingest",
                    target=path,
                    source=str(source_dir),
                    params={"model": model, "instrument": instrument, "mask": mask},
                    records_count=0,
                    success=True,
                )

            # Filter values below min_etf (default 0.05) as noise/artifacts
            # This matches legacy sparse_time_series() behavior which replaces
            # 0.0 with NaN and filters values < 0.05
            all_data = all_data.where(all_data >= min_etf)

            # Align to container grid and write
            records = self._write_timeseries(path, all_data, fields, overwrite=overwrite)

            ctx["records_processed"] = records
            ctx["fields_processed"] = len(all_data.columns)

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target=path,
                source=str(source_dir),
                source_format="earth_engine_csv",
                params={"model": model, "instrument": instrument, "mask": mask},
                fields_affected=list(all_data.columns),
                records_count=records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    # -------------------------------------------------------------------------
    # Meteorology Ingestion
    # -------------------------------------------------------------------------

    def gridmet(
        self,
        source_dir: str | Path,
        grid_shapefile: str | Path | None = None,
        grid_mapping: str | Path | dict[str, int] | GridMapping | None = None,
        uid_column: str = "FID",
        grid_column: str = "GFID",
        variables: list[str] | None = None,
        include_corrected: bool = True,
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """
        Ingest GridMET meteorology data from Parquet files.

        GridMET data is downloaded at grid cell resolution (4km), where multiple
        fields may share the same grid cell. This method can operate in two modes:

        1. **Mapped mode** (grid_shapefile or grid_mapping provided): Uses a
           UID-to-GFID mapping to replicate grid cell data across fields that
           share the same cell. Files are named {gfid}.parquet.

        2. **Direct mode** (no mapping provided): Looks for files named
           {uid}.parquet directly. Use this when each field has its own
           unique parquet file (e.g., sparse flux stations).

        Args:
            source_dir: Directory containing Parquet files
            grid_shapefile: Shapefile with UID and GFID columns for mapping
            grid_mapping: Alternative to grid_shapefile - can be:
                - Path to JSON file with {uid: gfid, ...} mapping
                - Dict with {uid: gfid, ...} mapping
                - GridMapping instance
            uid_column: Column name for field UID in shapefile (default: "FID")
            grid_column: Column name for grid ID in shapefile (default: "GFID")
            variables: Variables to ingest (default: all available)
            include_corrected: Include bias-corrected ET variables (eto_corr, etr_corr)
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        from .grid_mapping import GridMapping

        self._ensure_writable()
        source_dir = Path(source_dir)

        # Determine mode: mapped vs direct
        use_mapping = grid_shapefile is not None or grid_mapping is not None
        mapping = None
        n_grid_cells = 0

        if use_mapping:
            # Build grid mapping
            if grid_shapefile is not None:
                mapping = GridMapping.from_shapefile(
                    grid_shapefile, uid_column, grid_column, source_name="gridmet"
                )
            elif isinstance(grid_mapping, (str, Path)):
                mapping = GridMapping.from_json(grid_mapping, source_name="gridmet")
            elif isinstance(grid_mapping, dict):
                mapping = GridMapping(grid_mapping, source_name="gridmet")
            else:
                # Assume it's already a GridMapping instance
                mapping = grid_mapping

            self._log.info(
                "gridmet_mapping_loaded",
                n_fields=mapping.n_fields,
                n_grid_cells=mapping.n_grid_cells,
            )
            n_grid_cells = mapping.n_grid_cells
        else:
            # Direct mode - files named by UID
            self._log.info(
                "gridmet_direct_mode",
                message="No mapping provided, looking for {uid}.parquet files",
            )

        # Default variables
        if variables is None:
            variables = ["eto", "etr", "prcp", "tmin", "tmax", "srad", "ea", "u2"]
            if include_corrected:
                variables.extend(["eto_corr", "etr_corr"])

        with self._track_operation(
            "ingest_gridmet",
            target="meteorology/gridmet",
            source=str(source_dir),
            variables=variables,
        ) as ctx:
            total_records = 0
            fields_processed = set()

            for var in variables:
                path = f"meteorology/gridmet/{var}"

                if path in self._state.root and not overwrite:
                    self._log.debug("skipping_existing", path=path)
                    continue

                # Load data from Parquet files
                if use_mapping:
                    var_data = self._load_gridded_variable(source_dir, var, mapping)
                else:
                    var_data = self._load_uid_variable(source_dir, var)

                if var_data.empty:
                    self._log.debug("no_data_for_variable", variable=var)
                    continue

                # Write to container
                records = self._write_timeseries(path, var_data, None, overwrite=overwrite)
                total_records += records
                fields_processed.update(var_data.columns)

            ctx["records_processed"] = total_records
            ctx["fields_processed"] = len(fields_processed)

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target="meteorology/gridmet",
                source=str(source_dir),
                source_format="parquet",
                params={
                    "variables": variables,
                    "include_corrected": include_corrected,
                    "grid_cells": n_grid_cells,
                    "direct_mode": not use_mapping,
                },
                fields_affected=list(fields_processed),
                records_count=total_records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def era5(
        self,
        source_dir: str | Path,
        variables: list[str] | None = None,
        field_mapping: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """
        Ingest ERA5 meteorology data from monthly CSV exports.

        Handles the column format: {param}_{YYYYMMDD} (e.g., eto_20170115)

        Args:
            source_dir: Directory containing ERA5 CSV files
            variables: Variables to ingest (default: swe, eto, tmean, tmin, tmax, prcp, srad)
            field_mapping: Optional UID to met-file mapping
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        source_dir = Path(source_dir)

        # Default variables for ERA5
        if variables is None:
            variables = ["swe", "eto", "tmean", "tmin", "tmax", "prcp", "srad"]

        # Parameter name mapping (ERA5 uses different names)
        param_mapping = {
            "precip": "prcp",
            "precipitation": "prcp",
        }

        with self._track_operation(
            "ingest_era5",
            target="meteorology/era5",
            source=str(source_dir),
            variables=variables,
        ) as ctx:
            # Parse ERA5 monthly CSVs into site-level data
            site_data = self._parse_era5_csvs(source_dir, param_mapping)

            if not site_data:
                self._log.warning("no_data_found", source=str(source_dir))
                return self._state.provenance.record(
                    "ingest",
                    target="meteorology/era5",
                    source=str(source_dir),
                    params={"variables": variables},
                    records_count=0,
                    success=True,
                )

            total_records = 0
            fields_processed = set()

            # Process each variable
            for var in variables:
                # Apply mapping to get the normalized variable name
                # This matches the mapping applied during CSV parsing
                normalized_var = param_mapping.get(var, var)
                path = f"meteorology/era5/{normalized_var}"

                if path in self._state.root and not overwrite:
                    self._log.debug("skipping_existing", path=path)
                    continue

                # Extract variable data from site_data using normalized name
                var_df = self._extract_variable_from_site_data(site_data, normalized_var)

                if var_df.empty:
                    self._log.debug("no_data_for_variable", variable=var)
                    continue

                # Write to container
                records = self._write_timeseries(path, var_df, None, overwrite=overwrite)
                total_records += records
                fields_processed.update(var_df.columns)

            ctx["records_processed"] = total_records
            ctx["fields_processed"] = len(fields_processed)

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target="meteorology/era5",
                source=str(source_dir),
                source_format="era5_csv",
                params={"variables": variables},
                fields_affected=list(fields_processed),
                records_count=total_records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def snodas(
        self,
        source_dir: str | Path,
        uid_column: str = "FID",
        fields: list[str] | None = None,
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """
        Ingest SNODAS snow water equivalent data from Earth Engine CSV extracts.

        Args:
            source_dir: Directory containing CSV files from Earth Engine export.
                CSV format: rows=fields, columns=dates (YYYYMMDD), values=SWE in meters.
            uid_column: Column name for field UID in CSVs (default: "FID")
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        source_dir = Path(source_dir)
        path = "snow/snodas/swe"

        with self._track_operation(
            "ingest_snodas",
            target=path,
            source=str(source_dir),
        ) as ctx:
            if path in self._state.root and not overwrite:
                raise ValueError(f"Data exists at {path}. Use overwrite=True.")

            # Load SWE data from CSV extracts
            swe_data = self._load_snodas_extracts(source_dir, uid_column, fields)

            if swe_data.empty:
                self._log.warning("no_data_found", source=str(source_dir))
                return self._state.provenance.record(
                    "ingest",
                    target=path,
                    source=str(source_dir),
                    params={},
                    records_count=0,
                    success=True,
                )

            # Write to container
            records = self._write_timeseries(path, swe_data, fields, overwrite=overwrite)

            ctx["records_processed"] = records
            ctx["fields_processed"] = len(swe_data.columns)

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target=path,
                source=str(source_dir),
                source_format="earth_engine_csv",
                params={"uid_column": uid_column},
                fields_affected=list(swe_data.columns),
                records_count=records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    # -------------------------------------------------------------------------
    # Properties Ingestion
    # -------------------------------------------------------------------------

    def properties(
        self,
        lulc_csv: str | Path | None = None,
        soils_csv: str | Path | None = None,
        irr_csv: str | Path | None = None,
        location_csv: str | Path | None = None,
        uid_column: str = "FID",
        lulc_column: str = "modis_lc",
        extra_lulc_column: str | None = "glc10_lc",
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """
        Ingest static field properties from CSV files.

        Applies LULC override logic:
        1. GLCLand10 crop code (10) overrides non-crop MODIS codes to cropland (12)
        2. Mean irrigation > 0.3 overrides to cropland (12)

        Args:
            lulc_csv: CSV with land use/land cover data
            soils_csv: CSV with soil properties (AWC, clay, sand, ksat)
            irr_csv: CSV with irrigation fraction data
            location_csv: CSV with location data (lat, lon, elevation)
            uid_column: Column name for field UID in CSVs
            lulc_column: Column name for LULC code (default: modis_lc)
            extra_lulc_column: Column for secondary LULC (default: glc10_lc)
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        sources = []
        properties_ingested = []

        with self._track_operation(
            "ingest_properties",
            target="properties",
        ) as ctx:
            # Process LULC
            if lulc_csv:
                lulc_csv = Path(lulc_csv)
                sources.append(str(lulc_csv))
                self._ingest_lulc(
                    lulc_csv,
                    uid_column,
                    lulc_column,
                    extra_lulc_column,
                    irr_csv,
                    overwrite,
                )
                properties_ingested.append("land_cover")

            # Process soils
            if soils_csv:
                soils_csv = Path(soils_csv)
                sources.append(str(soils_csv))
                self._ingest_soils(soils_csv, uid_column, overwrite)
                properties_ingested.append("soils")

            # Process irrigation
            if irr_csv:
                irr_csv = Path(irr_csv)
                sources.append(str(irr_csv))
                self._ingest_irrigation(irr_csv, uid_column, overwrite)
                properties_ingested.append("irrigation")

            # Process location
            if location_csv:
                location_csv = Path(location_csv)
                sources.append(str(location_csv))
                self._ingest_location(location_csv, uid_column, overwrite)
                properties_ingested.append("location")

            ctx["fields_processed"] = self._state.n_fields

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target="properties",
                source="; ".join(sources),
                source_format="csv",
                params={
                    "uid_column": uid_column,
                    "properties": properties_ingested,
                },
                fields_affected=self._state.field_uids,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def dynamics(
        self,
        dynamics_json: str | Path,
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """
        Ingest pre-computed dynamics data from JSON file.

        Args:
            dynamics_json: Path to JSON file with dynamics data
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        dynamics_json = Path(dynamics_json)
        import json

        with self._track_operation(
            "ingest_dynamics",
            target="derived/dynamics",
            source=str(dynamics_json),
        ) as ctx:
            with open(dynamics_json) as f:
                data = json.load(f)

            # Write ke_max
            ke_path = "derived/dynamics/ke_max"
            if ke_path in self._state.root and not overwrite:
                pass
            else:
                if ke_path in self._state.root:
                    self._safe_delete_path(ke_path)
                arr = self._state.create_property_array(ke_path)
                for uid in self._state.field_uids:
                    if uid in data.get("ke_max", {}):
                        idx = self._state.get_field_index(uid)
                        arr[idx] = data["ke_max"][uid]

            # Write kc_max
            kc_path = "derived/dynamics/kc_max"
            if kc_path in self._state.root and not overwrite:
                pass
            else:
                if kc_path in self._state.root:
                    self._safe_delete_path(kc_path)
                arr = self._state.create_property_array(kc_path)
                for uid in self._state.field_uids:
                    if uid in data.get("kc_max", {}):
                        idx = self._state.get_field_index(uid)
                        arr[idx] = data["kc_max"][uid]

            # Write irr_data and gwsub_data as JSON strings
            from zarr.core.dtype import VariableLengthUTF8

            for key in ["irr", "gwsub"]:
                data_key = f"{key}_data" if key in ["irr", "gwsub"] else key
                if data_key not in data:
                    continue

                data_path = f"derived/dynamics/{key}_data"
                if data_path in self._state.root and not overwrite:
                    continue
                if data_path in self._state.root:
                    self._safe_delete_path(data_path)

                parent = self._state.ensure_group("derived/dynamics")
                arr = parent.create_array(
                    f"{key}_data",
                    shape=(self._state.n_fields,),
                    dtype=VariableLengthUTF8(),
                )

                # Build list of values then assign at once
                values = [""] * self._state.n_fields
                for uid in self._state.field_uids:
                    if uid in data.get(data_key, {}):
                        idx = self._state.get_field_index(uid)
                        values[idx] = json.dumps(data[data_key][uid])
                arr[:] = values

            ctx["fields_processed"] = len(data.get("ke_max", {}))

            event = self._state.provenance.record(
                "ingest",
                target="derived/dynamics",
                source=str(dynamics_json),
                source_format="dynamics_json",
                params={},
                fields_affected=list(data.get("ke_max", {}).keys()),
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _parse_ee_remote_sensing_csvs(
        self,
        source_dir: Path,
        instrument: str,
        parameter: str,
        uid_column: str,
        fields: list[str] | None = None,
        mask: str | None = None,
    ) -> pd.DataFrame:
        """
        Parse Earth Engine CSV exports into a unified DataFrame.

        CSV format: rows=fields (identified by uid_column), columns=dates (YYYYMMDD).
        Handles date parsing from column names:
        - Landsat: PARAM_YYYYMMDD (e.g., NDVI_20170115)
        - Sentinel: YYYYMMDD_... (e.g., 20170115_S2A)

        File naming convention for mask filtering:
        - ndvi_{field_id}_{mask}_{year}.csv
        - {model}_etf_{field_id}_{mask}_{year}.csv

        Args:
            source_dir: Directory containing CSV files
            instrument: Source instrument ("landsat", "sentinel", etc.)
            parameter: Data type ("ndvi" or "etf")
            uid_column: Column name for field UID in CSVs
            fields: Optional list of field UIDs to process
            mask: Optional mask type to filter files ("irr", "inv_irr", "no_mask")

        Returns:
            DataFrame with DatetimeIndex and field UIDs as columns
        """
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            self._log.warning("no_csv_files", directory=str(source_dir))
            return pd.DataFrame()

        # Filter files by mask if specified
        if mask is not None:
            # Match files that contain the mask pattern in the filename
            # File naming conventions vary:
            #   - ndvi_{year}_{mask}.csv (e.g., ndvi_2020_irr.csv)
            #   - ndvi_{field}_{mask}_{year}.csv (e.g., ndvi_US-FPe_irr_2020.csv)
            #   - {model}_etf_{field}_{mask}_{year}.csv
            # Handle "irr" vs "inv_irr" - "inv_irr" should NOT match mask="irr"
            filtered_files = []
            for f in csv_files:
                filename = f.stem  # filename without extension
                # Check for mask at end of filename (_irr) or in middle (_irr_)
                if mask == "irr":
                    # Match _irr at end or _irr_ in middle, but exclude inv_irr
                    has_irr = filename.endswith("_irr") or "_irr_" in filename
                    has_inv_irr = "inv_irr" in filename
                    if has_irr and not has_inv_irr:
                        filtered_files.append(f)
                elif mask == "inv_irr":
                    # Match _inv_irr at end or _inv_irr_ in middle
                    if filename.endswith("_inv_irr") or "_inv_irr_" in filename:
                        filtered_files.append(f)
                elif mask == "no_mask":
                    # Match _no_mask at end or _no_mask_ in middle
                    if filename.endswith("_no_mask") or "_no_mask_" in filename:
                        filtered_files.append(f)
                else:
                    # Generic mask pattern
                    if filename.endswith(f"_{mask}") or f"_{mask}_" in filename:
                        filtered_files.append(f)
            csv_files = filtered_files

            if not csv_files:
                self._log.debug("no_files_for_mask", mask=mask, directory=str(source_dir))
                return pd.DataFrame()

        all_series = []
        fields_found = set()

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                self._log.debug("csv_parse_error", file=str(csv_file), error=str(e))
                continue

            # Handle single-field CSVs where field ID is the first column header
            # Format: field_id, date1, date2, ...
            #         (empty), val1, val2, ...
            # Convert to standard format by renaming first column and setting field ID as row value
            if uid_column not in df.columns:
                first_col = df.columns[0]
                # Check if first column header is a known field ID
                if first_col in self._state._uid_to_index:
                    # This is a single-field sparse CSV - convert to standard format
                    field_id = first_col
                    new_cols = [uid_column] + list(df.columns[1:])
                    df.columns = new_cols
                    # Cast first column to object dtype before assigning string
                    df[uid_column] = df[uid_column].astype(object)
                    df.iloc[0, 0] = field_id
                    self._log.debug("converted_sparse_csv", file=str(csv_file), field_id=field_id)
                else:
                    self._log.warning(
                        "uid_column_missing",
                        file=str(csv_file),
                        uid_column=uid_column,
                        available_columns=list(df.columns[:5]),
                    )
                    continue

            # Parse data columns (those with dates in the column name) - do once per file
            data_cols = []
            dates = []
            non_data_cols = {uid_column, "system:index", ".geo", "lat", "lon", "LAT", "LON"}

            for col in df.columns:
                if col in non_data_cols:
                    continue

                try:
                    if instrument == "landsat":
                        # Format: PARAM_YYYYMMDD (e.g., NDVI_20170115)
                        parts = col.split("_")
                        if len(parts) >= 2:
                            date_str = parts[-1]
                            if len(date_str) == 8 and date_str.isdigit():
                                dates.append(pd.to_datetime(date_str))
                                data_cols.append(col)
                    elif instrument == "sentinel":
                        # Format: YYYYMMDD... (e.g., 20170115_S2A)
                        date_str = col[:8]
                        if len(date_str) == 8 and date_str.isdigit():
                            dates.append(pd.to_datetime(date_str))
                            data_cols.append(col)
                except Exception:
                    continue

            if not data_cols:
                self._log.warning(
                    "no_date_columns_found", file=str(csv_file), sample_columns=list(df.columns[:5])
                )
                continue

            # Iterate over all rows (each row is a field)
            for _, row in df.iterrows():
                field_id = str(row[uid_column])

                # Filter by requested fields
                if fields and field_id not in fields:
                    continue

                # Skip if field not in container
                if field_id not in self._state._uid_to_index:
                    continue

                fields_found.add(field_id)

                # Extract values for this row and create series
                values = row[data_cols].values
                series = pd.Series(values, index=dates, name=field_id)
                series = series.sort_index()

                # Remove duplicates by taking the max value
                if series.index.duplicated().any():
                    series = series.groupby(series.index).max()

                all_series.append(series)

        if not all_series:
            self._log.warning(
                "no_series_created",
                fields_found=list(fields_found),
                container_fields_sample=list(self._state._uid_to_index.keys())[:5],
            )
            return pd.DataFrame()

        # Group series by field ID and combine (handle multiple CSV files per field)
        from collections import defaultdict

        field_series = defaultdict(list)
        for s in all_series:
            field_series[s.name].append(s)

        combined_series = []
        for field_id, series_list in field_series.items():
            if len(series_list) == 1:
                combined = series_list[0]
            else:
                # Combine multiple series for the same field
                combined = series_list[0]
                for s in series_list[1:]:
                    combined = combined.combine_first(s)
                combined.name = field_id
            combined_series.append(combined)

        # Combine all series into a DataFrame
        result = pd.concat(combined_series, axis=1)
        result = result.sort_index()

        # Ensure we have a proper DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.DatetimeIndex(result.index)

        return result

    def _apply_ndvi_filters(
        self,
        df: pd.DataFrame,
        min_ndvi: float,
        apply_consecutive_filter: bool,
    ) -> pd.DataFrame:
        """
        Apply quality filters to NDVI data.

        1. Replace values below min_ndvi with NaN
        2. Remove lower of two consecutive-day observations
        """
        # Filter by minimum NDVI
        df = df.where(df >= min_ndvi)

        if not apply_consecutive_filter:
            return df

        # Consecutive day filtering (vectorized approach)
        # For each field, where two consecutive days both have data,
        # remove the lower value
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            # Find consecutive days
            idx = series.index.to_series()
            diffs = idx.diff()
            consecutive = diffs == pd.Timedelta(days=1)

            for day in series.index[consecutive]:
                prev_day = day - pd.Timedelta(days=1)
                if prev_day in series.index:
                    if series[prev_day] > series[day]:
                        df.loc[day, col] = np.nan
                    else:
                        df.loc[prev_day, col] = np.nan

        return df

    def _write_timeseries(
        self,
        path: str,
        data: pd.DataFrame,
        fields: list[str] | None,
        overwrite: bool = False,
    ) -> int:
        """
        Write time series DataFrame to container Zarr array.

        Args:
            path: Target path in container
            data: DataFrame with DatetimeIndex and field columns
            fields: Optional field filter
            overwrite: If True, overwrite existing array

        Returns:
            Number of non-NaN values written
        """

        # Create the array
        arr = self._state.create_timeseries_array(path, overwrite=overwrite)

        # Align data to container grid
        # Reindex to container time and field dimensions
        container_fields = fields if fields else self._state.field_uids
        common_fields = [f for f in container_fields if f in data.columns]

        if not common_fields:
            self._log.warning("no_matching_fields", path=path)
            return 0

        # Reindex data to container time index
        aligned = data.reindex(index=self._state.time_index, columns=common_fields)

        # Ensure numeric dtype (CSV parsing can produce object dtype)
        aligned = aligned.apply(pd.to_numeric, errors="coerce")

        # Write each field
        for field_uid in common_fields:
            if field_uid not in self._state._uid_to_index:
                continue
            field_idx = self._state._uid_to_index[field_uid]
            arr[:, field_idx] = aligned[field_uid].values

        return int(np.count_nonzero(~np.isnan(aligned.values.astype(float))))

    def _load_gridded_variable(
        self,
        source_dir: Path,
        variable: str,
        grid_mapping: GridMapping,
    ) -> pd.DataFrame:
        """
        Load a variable from grid-cell-based parquet files.

        Replicates timeseries across all fields mapped to each grid cell.
        This handles the case where multiple fields share the same GridMET
        cell (or other coarse-resolution grid).

        Args:
            source_dir: Directory containing {grid_id}.parquet files
            variable: Variable name to extract (e.g., 'eto', 'tmax')
            grid_mapping: GridMapping with UID→grid_id relationships

        Returns:
            DataFrame with columns=field_uids, index=dates

        Raises:
            ValueError: If legacy MultiIndex format is detected
        """

        result_series = []
        valid_uids = set(self._state._uid_to_index.keys())

        # Filter mapping to only UIDs in this container
        mapping = grid_mapping.filter_to_valid_uids(valid_uids)

        if not mapping.unique_grid_ids:
            self._log.warning(
                "no_grid_cells_mapped",
                n_container_fields=len(valid_uids),
                n_mapping_fields=len(grid_mapping),
            )
            return pd.DataFrame()

        for grid_id in mapping.unique_grid_ids:
            # Find parquet file for this grid cell
            pq_file = source_dir / f"{grid_id}.parquet"
            if not pq_file.exists():
                self._log.debug(
                    "grid_file_missing",
                    grid_id=grid_id,
                    file=str(pq_file),
                )
                continue

            try:
                df = pd.read_parquet(pq_file)
            except Exception as e:
                self._log.debug(
                    "parquet_read_error",
                    file=str(pq_file),
                    variable=variable,
                    error=str(e),
                )
                continue

            # Require simple column format (no legacy MultiIndex support)
            if isinstance(df.columns, pd.MultiIndex):
                raise ValueError(
                    f"Legacy MultiIndex format not supported. "
                    f"Re-download gridmet data with simple column format: {pq_file}"
                )

            if variable not in df.columns:
                self._log.debug(
                    "variable_not_in_file",
                    variable=variable,
                    file=str(pq_file),
                    available=list(df.columns),
                )
                continue

            series = df[variable]

            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.DatetimeIndex(series.index)

            # Replicate for all UIDs mapped to this grid cell
            for uid in mapping.get_uids_for_grid(grid_id):
                if uid not in valid_uids:
                    continue
                uid_series = series.copy()
                uid_series.name = uid
                result_series.append(uid_series)

        if not result_series:
            return pd.DataFrame()

        return pd.concat(result_series, axis=1).sort_index()

    def _load_uid_variable(
        self,
        source_dir: Path,
        variable: str,
    ) -> pd.DataFrame:
        """
        Load a variable from UID-named parquet files (direct mode).

        Looks for files named {uid}.parquet directly, without grid mapping.
        Use this for sparse field networks where each field has its own
        unique parquet file.

        Args:
            source_dir: Directory containing {uid}.parquet files
            variable: Variable name to extract (e.g., 'eto', 'tmax')

        Returns:
            DataFrame with columns=field_uids, index=dates
        """
        result_series = []
        valid_uids = set(self._state._uid_to_index.keys())

        for uid in valid_uids:
            # Find parquet file for this UID
            pq_file = source_dir / f"{uid}.parquet"
            if not pq_file.exists():
                self._log.debug(
                    "uid_file_missing",
                    uid=uid,
                    file=str(pq_file),
                )
                continue

            try:
                df = pd.read_parquet(pq_file)
            except Exception as e:
                self._log.debug(
                    "parquet_read_error",
                    file=str(pq_file),
                    variable=variable,
                    error=str(e),
                )
                continue

            # Require simple column format (no legacy MultiIndex support)
            if isinstance(df.columns, pd.MultiIndex):
                raise ValueError(
                    f"Legacy MultiIndex format not supported. "
                    f"Re-download gridmet data with simple column format: {pq_file}"
                )

            if variable not in df.columns:
                self._log.debug(
                    "variable_not_in_file",
                    variable=variable,
                    file=str(pq_file),
                    available=list(df.columns),
                )
                continue

            series = df[variable]

            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.DatetimeIndex(series.index)

            series.name = uid
            result_series.append(series)

        if not result_series:
            return pd.DataFrame()

        return pd.concat(result_series, axis=1).sort_index()

    def _parse_era5_csvs(
        self,
        source_dir: Path,
        param_mapping: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        """
        Parse ERA5 monthly CSV exports using vectorized operations.

        Column format: {param}_{YYYYMMDD} (e.g., eto_20170115)

        Returns:
            Dict mapping field_uid to DataFrame with parameter columns
        """
        site_data = {}
        valid_uids = set(self._state._uid_to_index.keys())
        csv_files = list(source_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            # Determine field ID column
            uid_col = None
            for col in ["FID", "fid", "site_id", "SITE_ID", "sid", "SID"]:
                if col in df.columns:
                    uid_col = col
                    break

            if uid_col is None:
                continue

            # Parse column headers ONCE to identify data columns and create MultiIndex
            col_tuples = []  # (param, date) tuples for MultiIndex
            valid_cols = []  # corresponding column names
            for col in df.columns:
                if col == uid_col:
                    continue
                if "_" not in col:
                    continue

                parts = col.rsplit("_", 1)
                if len(parts) != 2:
                    continue

                param, date_str = parts
                if len(date_str) != 8 or not date_str.isdigit():
                    continue

                # Apply parameter mapping
                param = param_mapping.get(param, param)

                try:
                    date = pd.to_datetime(date_str)
                    col_tuples.append((param, date))
                    valid_cols.append(col)
                except Exception:
                    continue

            if not valid_cols:
                continue

            # Filter to valid sites and set UID as index
            df[uid_col] = df[uid_col].astype(str)
            df = df[df[uid_col].isin(valid_uids)]
            if df.empty:
                continue

            df = df.set_index(uid_col)

            # Extract just the data columns and set MultiIndex
            data_df = df[valid_cols].copy()
            data_df.columns = pd.MultiIndex.from_tuples(col_tuples, names=["param", "date"])

            # For each site, unstack to get DataFrame with date index, param columns
            for uid in data_df.index:
                row = data_df.loc[uid]
                # Handle case where uid appears multiple times
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                # Unstack: param level becomes columns, date becomes index
                site_df = row.unstack(level="param")

                if uid in site_data:
                    site_data[uid] = pd.concat([site_data[uid], site_df])
                    site_data[uid] = site_data[uid][~site_data[uid].index.duplicated(keep="last")]
                else:
                    site_data[uid] = site_df

        return site_data

    def _extract_variable_from_site_data(
        self,
        site_data: dict[str, pd.DataFrame],
        variable: str,
    ) -> pd.DataFrame:
        """Extract a single variable from site-level DataFrames."""
        series_list = []

        for field_uid, df in site_data.items():
            if variable not in df.columns:
                continue

            series = df[variable].copy()
            series.name = field_uid
            series_list.append(series)

        if not series_list:
            return pd.DataFrame()

        result = pd.concat(series_list, axis=1)
        return result.sort_index()

    def _load_snodas_extracts(
        self,
        source_dir: Path,
        uid_column: str,
        fields: list[str] | None,
    ) -> pd.DataFrame:
        """
        Load SNODAS SWE data from Earth Engine CSV extracts.

        CSV format: rows=fields, columns=dates (YYYYMMDD), values=SWE in meters.
        Values are converted to millimeters (*1000). See `src/swimrs/units.py`
        (SNODAS_DAILY_UNITS).

        Args:
            source_dir: Directory containing CSV files
            uid_column: Column name for field UID
            fields: Optional list of field UIDs to filter

        Returns:
            DataFrame with DatetimeIndex and field UIDs as columns, SWE in mm
        """
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            self._log.warning("no_csv_files", directory=str(source_dir))
            return pd.DataFrame()

        # Accumulate data across all CSV files (each file is one month)
        all_data: dict[str, dict[str, float]] = {}  # {field_uid: {date: value}}

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=uid_column)
            except Exception as e:
                self._log.debug("csv_parse_error", file=str(csv_file), error=str(e))
                continue

            # Each row is a field, each column is a date (YYYYMMDD format)
            for field_uid, row in df.iterrows():
                field_uid = str(field_uid)

                # Filter by requested fields
                if fields and field_uid not in fields:
                    continue
                # Filter by fields in container
                if field_uid not in self._state._uid_to_index:
                    continue

                if field_uid not in all_data:
                    all_data[field_uid] = {}

                for date_str, value in row.items():
                    # Convert meters to millimeters
                    all_data[field_uid][str(date_str)] = value * 1000.0

        if not all_data:
            return pd.DataFrame()

        # Convert to DataFrame with DatetimeIndex
        series_list = []
        for field_uid, date_values in all_data.items():
            dates = [pd.to_datetime(d, format="%Y%m%d") for d in date_values.keys()]
            values = list(date_values.values())
            series = pd.Series(values, index=dates, name=field_uid)
            series_list.append(series)

        result = pd.concat(series_list, axis=1)
        return result.sort_index()

    def _ingest_lulc(
        self,
        lulc_csv: Path,
        uid_column: str,
        lulc_column: str,
        extra_lulc_column: str | None,
        irrigation_csv: str | Path | None,
        overwrite: bool,
    ) -> None:
        """Ingest LULC data with override logic."""
        path = "properties/land_cover/modis_lc"

        if path in self._state.root and not overwrite:
            return
        if path in self._state.root:
            self._safe_delete_path(path)

        df = pd.read_csv(lulc_csv)
        df = df.set_index(uid_column)

        # Apply GLCLand10 crop override
        # If extra LULC column has value 10 (crop) and MODIS is not 12, override to 12
        if extra_lulc_column and extra_lulc_column in df.columns:
            crop_override = (df[extra_lulc_column] == 10) & (df[lulc_column] != 12)
            df.loc[crop_override, lulc_column] = 12

        # Apply irrigation-based crop override
        # If mean irrigation > 0.3 and MODIS is not 12, override to 12
        if irrigation_csv:
            irr_df = pd.read_csv(Path(irrigation_csv))
            irr_df = irr_df.set_index(uid_column)

            # Drop non-numeric columns
            numeric_cols = irr_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                mean_irr = irr_df[numeric_cols].mean(axis=1)
                irr_crop_override = (mean_irr > 0.3) & (df[lulc_column] != 12)
                df.loc[irr_crop_override, lulc_column] = 12

        # Write to container (use -1 as fill_value for integer types)
        arr = self._state.create_property_array(path, dtype="int16", fill_value=-1)

        for uid in self._state.field_uids:
            if uid in df.index:
                idx = self._state.get_field_index(uid)
                arr[idx] = int(df.loc[uid, lulc_column])

    def _ingest_soils(
        self,
        soils_csv: Path,
        uid_column: str,
        overwrite: bool,
    ) -> None:
        """Ingest soil properties.

        Expected units (canonical SWIM-RS):
        - `awc`: meters of water per meter soil (m/m) in source CSV; stored as-is
          in the container and converted to mm/m when building SwimInput.
        - `ksat`: mm/day. This is converted to mm/hr internally for IER runoff.
          See `src/swimrs/units.py` (PROCESS_CANONICAL_UNITS).
        """
        df = pd.read_csv(soils_csv)
        df = df.set_index(uid_column)

        # Common soil property columns
        soil_props = {
            "awc": ["awc", "AWC", "available_water_capacity"],
            "clay": ["clay", "CLAY", "clay_pct"],
            "sand": ["sand", "SAND", "sand_pct"],
            "ksat": ["ksat", "KSAT", "saturated_conductivity"],
        }

        for prop, possible_cols in soil_props.items():
            path = f"properties/soils/{prop}"

            if path in self._state.root and not overwrite:
                continue
            if path in self._state.root:
                self._safe_delete_path(path)

            # Find the matching column
            col = None
            for c in possible_cols:
                if c in df.columns:
                    col = c
                    break

            if col is None:
                continue

            arr = self._state.create_property_array(path)

            for uid in self._state.field_uids:
                if uid in df.index:
                    idx = self._state.get_field_index(uid)
                    value = df.loc[uid, col]
                    if pd.notna(value):
                        arr[idx] = float(value)

    def _ingest_irrigation(
        self,
        irrigation_csv: Path,
        uid_column: str,
        overwrite: bool,
    ) -> None:
        """Ingest irrigation fraction data (mean and per-year)."""
        mean_path = "properties/irrigation/irr"
        yearly_path = "properties/irrigation/irr_yearly"

        df = pd.read_csv(irrigation_csv)
        df = df.set_index(uid_column)

        # Extract year columns (format: irr_YYYY)
        year_cols = [c for c in df.columns if c.startswith("irr_") and c[4:].isdigit()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return

        # Store mean irrigation
        if mean_path not in self._state.root or overwrite:
            if mean_path in self._state.root:
                self._safe_delete_path(mean_path)

            mean_irr = df[numeric_cols].mean(axis=1)
            arr = self._state.create_property_array(mean_path)

            for uid in self._state.field_uids:
                if uid in mean_irr.index:
                    idx = self._state.get_field_index(uid)
                    value = mean_irr[uid]
                    if pd.notna(value):
                        arr[idx] = float(value)

        # Store per-year irrigation as JSON strings
        if yearly_path not in self._state.root or overwrite:
            if yearly_path in self._state.root:
                self._safe_delete_path(yearly_path)

            if year_cols:
                import json

                from zarr.core.dtype import VariableLengthUTF8

                parent = self._state.ensure_group("properties/irrigation")
                arr = parent.create_array(
                    "irr_yearly",
                    shape=(self._state.n_fields,),
                    dtype=VariableLengthUTF8(),
                )

                values = ["{}"] * self._state.n_fields
                for uid in self._state.field_uids:
                    if uid in df.index:
                        idx = self._state.get_field_index(uid)
                        # Build dict: {"2020": 0.5, "2021": 0.8, ...}
                        yearly_data = {}
                        for col in year_cols:
                            year_str = col[4:]  # Extract YYYY from irr_YYYY
                            val = df.loc[uid, col]
                            if pd.notna(val):
                                yearly_data[year_str] = float(val)
                        values[idx] = json.dumps(yearly_data)

                arr[:] = values

    def _ingest_location(
        self,
        location_csv: Path,
        uid_column: str,
        overwrite: bool,
    ) -> None:
        """Ingest location data (lat, lon, elevation)."""
        df = pd.read_csv(location_csv)
        df = df.set_index(uid_column)

        location_props = {
            "lat": ["lat", "LAT", "latitude", "LATITUDE"],
            "lon": ["lon", "LON", "longitude", "LONGITUDE"],
            "elevation": ["elevation", "ELEVATION", "elev", "ELEV"],
        }

        for prop, possible_cols in location_props.items():
            path = f"properties/location/{prop}"

            if path in self._state.root and not overwrite:
                continue
            if path in self._state.root:
                self._safe_delete_path(path)

            col = None
            for c in possible_cols:
                if c in df.columns:
                    col = c
                    break

            if col is None:
                continue

            arr = self._state.create_property_array(path)

            for uid in self._state.field_uids:
                if uid in df.index:
                    idx = self._state.get_field_index(uid)
                    value = df.loc[uid, col]
                    if pd.notna(value):
                        arr[idx] = float(value)

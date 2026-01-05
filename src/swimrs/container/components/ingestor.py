"""
Ingestor component for data ingestion operations.

Provides a clean, namespace-organized API for ingesting data into the container.
Usage: container.ingest.ndvi(...) instead of container.ingest_ee_ndvi(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import Component

if TYPE_CHECKING:
    import xarray as xr
    from swimrs.container.state import ContainerState
    from swimrs.container.provenance import ProvenanceEvent


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

    def __init__(self, state: "ContainerState", container=None):
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
        source_dir: Union[str, Path],
        instrument: str = "landsat",
        mask: str = "irr",
        fields: Optional[List[str]] = None,
        overwrite: bool = False,
        min_ndvi: float = 0.05,
        apply_consecutive_filter: bool = True,
    ) -> "ProvenanceEvent":
        """
        Ingest NDVI data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files (one per field per year)
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
            if path in self._state.root:
                del self._state.root[path]

            # Parse all CSVs into unified DataFrame
            all_data = self._parse_ee_csvs(source_dir, instrument, "ndvi", fields)

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
            all_data = self._apply_ndvi_filters(
                all_data, min_ndvi, apply_consecutive_filter
            )

            # Align to container grid and write
            records = self._write_timeseries(path, all_data, fields)

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
        source_dir: Union[str, Path],
        model: str = "ssebop",
        mask: str = "irr",
        instrument: str = "landsat",
        fields: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
        """
        Ingest ET fraction data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files (one per field per year)
            model: ET model ("ssebop", "ptjpl", "sims", "eemetric", etc.)
            mask: Mask type ("irr", "inv_irr", "no_mask")
            instrument: Source instrument ("landsat", "ecostress")
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data

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
            if path in self._state.root:
                del self._state.root[path]

            # Parse all CSVs into unified DataFrame
            all_data = self._parse_ee_csvs(source_dir, instrument, "etf", fields)

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

            # Apply basic quality filters (ETf should be 0-1.5 range typically)
            all_data = all_data.where((all_data >= 0) & (all_data <= 2.0))

            # Align to container grid and write
            records = self._write_timeseries(path, all_data, fields)

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
        source_dir: Union[str, Path],
        variables: Optional[List[str]] = None,
        include_corrected: bool = True,
        field_mapping: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
        """
        Ingest GridMET meteorology data from Parquet files.

        Args:
            source_dir: Directory containing Parquet files (one per field)
            variables: Variables to ingest (default: all available)
            include_corrected: Include bias-corrected ET variables
            field_mapping: Optional UID to met-file mapping
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        source_dir = Path(source_dir)

        # Default variables
        if variables is None:
            variables = ["eto", "etr", "prcp", "tmin", "tmax", "srad", "vpd", "ea", "u2"]
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
                if path in self._state.root:
                    del self._state.root[path]

                # Load data from Parquet files
                var_data = self._load_met_variable(
                    source_dir, var, "gridmet", field_mapping
                )

                if var_data.empty:
                    self._log.debug("no_data_for_variable", variable=var)
                    continue

                # Write to container
                records = self._write_timeseries(path, var_data, None)
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
                params={"variables": variables, "include_corrected": include_corrected},
                fields_affected=list(fields_processed),
                records_count=total_records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def era5(
        self,
        source_dir: Union[str, Path],
        variables: Optional[List[str]] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
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
                path = f"meteorology/era5/{var}"

                if path in self._state.root and not overwrite:
                    self._log.debug("skipping_existing", path=path)
                    continue
                if path in self._state.root:
                    del self._state.root[path]

                # Extract variable data from site_data
                var_df = self._extract_variable_from_site_data(site_data, var)

                if var_df.empty:
                    self._log.debug("no_data_for_variable", variable=var)
                    continue

                # Write to container
                records = self._write_timeseries(path, var_df, None)
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
        source_file: Optional[Union[str, Path]] = None,
        source_dir: Optional[Union[str, Path]] = None,
        fields: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
        """
        Ingest SNODAS snow water equivalent data.

        Args:
            source_file: JSON file with SWE data (keyed by field UID)
            source_dir: Alternative: directory with per-field files
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        path = "snow/snodas/swe"

        source = source_file or source_dir
        if source is None:
            raise ValueError("Must provide either source_file or source_dir")

        with self._track_operation(
            "ingest_snodas",
            target=path,
            source=str(source),
        ) as ctx:
            if path in self._state.root and not overwrite:
                raise ValueError(f"Data exists at {path}. Use overwrite=True.")
            if path in self._state.root:
                del self._state.root[path]

            # Load SWE data
            if source_file:
                swe_data = self._load_snodas_json(Path(source_file), fields)
            else:
                swe_data = self._load_snodas_dir(Path(source_dir), fields)

            if swe_data.empty:
                self._log.warning("no_data_found", source=str(source))
                return self._state.provenance.record(
                    "ingest",
                    target=path,
                    source=str(source),
                    params={},
                    records_count=0,
                    success=True,
                )

            # Write to container
            records = self._write_timeseries(path, swe_data, fields)

            ctx["records_processed"] = records
            ctx["fields_processed"] = len(swe_data.columns)

            # Record provenance
            event = self._state.provenance.record(
                "ingest",
                target=path,
                source=str(source),
                source_format="snodas_json" if source_file else "snodas_dir",
                params={},
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
        lulc_csv: Optional[Union[str, Path]] = None,
        soils_csv: Optional[Union[str, Path]] = None,
        irrigation_csv: Optional[Union[str, Path]] = None,
        location_csv: Optional[Union[str, Path]] = None,
        uid_column: str = "FID",
        lulc_column: str = "MODIS_LC",
        extra_lulc_column: Optional[str] = "glcland10",
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
        """
        Ingest static field properties from CSV files.

        Applies LULC override logic:
        1. GLCLand10 crop code (10) overrides non-crop MODIS codes to cropland (12)
        2. Mean irrigation > 0.3 overrides to cropland (12)

        Args:
            lulc_csv: CSV with land use/land cover data
            soils_csv: CSV with soil properties (AWC, clay, sand, ksat)
            irrigation_csv: CSV with irrigation fraction data
            location_csv: CSV with location data (lat, lon, elevation)
            uid_column: Column name for field UID in CSVs
            lulc_column: Column name for LULC code
            extra_lulc_column: Optional column for secondary LULC (e.g., glcland10)
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
                    irrigation_csv,
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
            if irrigation_csv:
                irrigation_csv = Path(irrigation_csv)
                sources.append(str(irrigation_csv))
                self._ingest_irrigation(irrigation_csv, uid_column, overwrite)
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
        dynamics_json: Union[str, Path],
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
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
                    del self._state.root[ke_path]
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
                    del self._state.root[kc_path]
                arr = self._state.create_property_array(kc_path)
                for uid in self._state.field_uids:
                    if uid in data.get("kc_max", {}):
                        idx = self._state.get_field_index(uid)
                        arr[idx] = data["kc_max"][uid]

            # Write irr_data and gwsub_data as JSON strings
            import zarr
            from numcodecs import VLenUTF8

            for key in ["irr", "gwsub"]:
                data_key = f"{key}_data" if key in ["irr", "gwsub"] else key
                if data_key not in data:
                    continue

                data_path = f"derived/dynamics/{key}_data"
                if data_path in self._state.root and not overwrite:
                    continue
                if data_path in self._state.root:
                    del self._state.root[data_path]

                parent = self._state.ensure_group("derived/dynamics")
                arr = parent.create_dataset(
                    f"{key}_data",
                    shape=(self._state.n_fields,),
                    dtype=object,
                    object_codec=VLenUTF8(),
                )

                for uid in self._state.field_uids:
                    if uid in data.get(data_key, {}):
                        idx = self._state.get_field_index(uid)
                        arr[idx] = json.dumps(data[data_key][uid])

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

    def _parse_ee_csvs(
        self,
        source_dir: Path,
        instrument: str,
        parameter: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Parse Earth Engine CSV exports into a unified DataFrame.

        Handles date parsing from column names:
        - Landsat: NDVI_YYYYMMDD (e.g., NDVI_20170115)
        - Sentinel: YYYYMMDD_... (e.g., 20170115_S2A)

        Returns:
            DataFrame with DatetimeIndex and field UIDs as columns
        """
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            self._log.warning("no_csv_files", directory=str(source_dir))
            return pd.DataFrame()

        all_series = []
        fields_found = set()

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                self._log.debug("csv_parse_error", file=str(csv_file), error=str(e))
                continue

            # Try to determine the field ID
            if "FID" in df.columns:
                field_id = str(df.iloc[0]["FID"])
            elif df.columns[0] not in ["date", "Date", "time"]:
                field_id = str(df.columns[0])
            else:
                field_id = csv_file.stem.split("_")[0]

            # Filter by requested fields
            if fields and field_id not in fields:
                continue

            # Skip if field not in container
            if field_id not in self._state._uid_to_index:
                continue

            fields_found.add(field_id)

            # Parse data columns (those with dates in the column name)
            data_cols = []
            dates = []

            for col in df.columns:
                # Skip non-data columns
                if col in ["FID", "system:index", ".geo", "lat", "lon", "LAT", "LON"]:
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
                continue

            # Extract values and create series
            values = df[data_cols].iloc[0].values
            series = pd.Series(values, index=dates, name=field_id)
            series = series.sort_index()

            # Remove duplicates by taking the max value
            if series.index.duplicated().any():
                series = series.groupby(series.index).max()

            all_series.append(series)

        if not all_series:
            return pd.DataFrame()

        # Combine all series into a DataFrame
        result = pd.concat(all_series, axis=1)
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
        fields: Optional[List[str]],
    ) -> int:
        """
        Write time series DataFrame to container Zarr array.

        Args:
            path: Target path in container
            data: DataFrame with DatetimeIndex and field columns
            fields: Optional field filter

        Returns:
            Number of non-NaN values written
        """
        import xarray as xr

        # Create the array
        arr = self._state.create_timeseries_array(path)

        # Align data to container grid
        # Reindex to container time and field dimensions
        container_fields = fields if fields else self._state.field_uids
        common_fields = [f for f in container_fields if f in data.columns]

        if not common_fields:
            self._log.warning("no_matching_fields", path=path)
            return 0

        # Reindex data to container time index
        aligned = data.reindex(index=self._state.time_index, columns=common_fields)

        # Write each field
        for field_uid in common_fields:
            if field_uid not in self._state._uid_to_index:
                continue
            field_idx = self._state._uid_to_index[field_uid]
            arr[:, field_idx] = aligned[field_uid].values

        return int(np.count_nonzero(~np.isnan(aligned.values)))

    def _load_met_variable(
        self,
        source_dir: Path,
        variable: str,
        source: str,
        field_mapping: Optional[Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Load a meteorology variable from Parquet files.

        Expects files named {field_uid}.parquet with the variable as a column.
        """
        result_series = []

        parquet_files = list(source_dir.glob("*.parquet"))

        for pq_file in parquet_files:
            # Determine field UID
            field_uid = pq_file.stem
            if field_mapping and field_uid in field_mapping:
                field_uid = field_mapping[field_uid]

            if field_uid not in self._state._uid_to_index:
                continue

            try:
                df = pd.read_parquet(pq_file)

                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    # Find the variable in the MultiIndex
                    matching_cols = [
                        c for c in df.columns
                        if (len(c) > 2 and c[2] == variable) or
                           (len(c) > 0 and c[0] == variable)
                    ]
                    if matching_cols:
                        series = df[matching_cols[0]]
                    else:
                        continue
                elif variable in df.columns:
                    series = df[variable]
                else:
                    continue

                series.name = field_uid
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.DatetimeIndex(series.index)

                result_series.append(series)

            except Exception as e:
                self._log.debug(
                    "parquet_read_error",
                    file=str(pq_file),
                    variable=variable,
                    error=str(e),
                )
                continue

        if not result_series:
            return pd.DataFrame()

        result = pd.concat(result_series, axis=1)
        return result.sort_index()

    def _parse_era5_csvs(
        self,
        source_dir: Path,
        param_mapping: Dict[str, str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Parse ERA5 monthly CSV exports.

        Column format: {param}_{YYYYMMDD} (e.g., eto_20170115)

        Returns:
            Dict mapping field_uid to DataFrame with parameter columns
        """
        site_data = {}

        csv_files = list(source_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            # Determine field ID column
            uid_col = None
            for col in ["FID", "fid", "site_id", "SITE_ID"]:
                if col in df.columns:
                    uid_col = col
                    break

            if uid_col is None:
                continue

            # Process each row (each row is a site)
            for _, row in df.iterrows():
                field_uid = str(row[uid_col])

                if field_uid not in self._state._uid_to_index:
                    continue

                # Parse parameter columns
                records = []
                for col in df.columns:
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
                        value = row[col]
                        records.append({"date": date, "param": param, "value": value})
                    except Exception:
                        continue

                if not records:
                    continue

                # Convert to DataFrame
                rec_df = pd.DataFrame(records)
                pivoted = rec_df.pivot(index="date", columns="param", values="value")

                if field_uid in site_data:
                    # Merge with existing data
                    site_data[field_uid] = pd.concat([site_data[field_uid], pivoted])
                    site_data[field_uid] = site_data[field_uid][
                        ~site_data[field_uid].index.duplicated(keep="last")
                    ]
                else:
                    site_data[field_uid] = pivoted

        return site_data

    def _extract_variable_from_site_data(
        self,
        site_data: Dict[str, pd.DataFrame],
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

    def _load_snodas_json(
        self,
        source_file: Path,
        fields: Optional[List[str]],
    ) -> pd.DataFrame:
        """Load SNODAS SWE data from JSON file."""
        import json

        with open(source_file) as f:
            data = json.load(f)

        series_list = []

        for field_uid, records in data.items():
            if fields and field_uid not in fields:
                continue
            if field_uid not in self._state._uid_to_index:
                continue

            # Records format: {date_str: value} or [{date: ..., swe: ...}]
            if isinstance(records, dict):
                dates = [pd.to_datetime(d) for d in records.keys()]
                values = list(records.values())
            elif isinstance(records, list):
                dates = [pd.to_datetime(r.get("date", r.get("time"))) for r in records]
                values = [r.get("swe", r.get("value")) for r in records]
            else:
                continue

            series = pd.Series(values, index=dates, name=field_uid)
            series_list.append(series)

        if not series_list:
            return pd.DataFrame()

        result = pd.concat(series_list, axis=1)
        return result.sort_index()

    def _load_snodas_dir(
        self,
        source_dir: Path,
        fields: Optional[List[str]],
    ) -> pd.DataFrame:
        """Load SNODAS SWE data from directory of files."""
        series_list = []

        for f in source_dir.glob("*.json"):
            field_uid = f.stem
            if fields and field_uid not in fields:
                continue
            if field_uid not in self._state._uid_to_index:
                continue

            import json
            with open(f) as fp:
                records = json.load(fp)

            if isinstance(records, dict):
                dates = [pd.to_datetime(d) for d in records.keys()]
                values = list(records.values())
            elif isinstance(records, list):
                dates = [pd.to_datetime(r.get("date", r.get("time"))) for r in records]
                values = [r.get("swe", r.get("value")) for r in records]
            else:
                continue

            series = pd.Series(values, index=dates, name=field_uid)
            series_list.append(series)

        if not series_list:
            return pd.DataFrame()

        result = pd.concat(series_list, axis=1)
        return result.sort_index()

    def _ingest_lulc(
        self,
        lulc_csv: Path,
        uid_column: str,
        lulc_column: str,
        extra_lulc_column: Optional[str],
        irrigation_csv: Optional[Union[str, Path]],
        overwrite: bool,
    ) -> None:
        """Ingest LULC data with override logic."""
        path = "properties/land_cover/modis_lc"

        if path in self._state.root and not overwrite:
            return
        if path in self._state.root:
            del self._state.root[path]

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

        # Write to container
        arr = self._state.create_property_array(path, dtype="int16")

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
        """Ingest soil properties."""
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
                del self._state.root[path]

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
        """Ingest irrigation fraction data (mean across years)."""
        path = "properties/irrigation/irr"

        if path in self._state.root and not overwrite:
            return
        if path in self._state.root:
            del self._state.root[path]

        df = pd.read_csv(irrigation_csv)
        df = df.set_index(uid_column)

        # Calculate mean irrigation across all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return

        mean_irr = df[numeric_cols].mean(axis=1)

        arr = self._state.create_property_array(path)

        for uid in self._state.field_uids:
            if uid in mean_irr.index:
                idx = self._state.get_field_index(uid)
                value = mean_irr[uid]
                if pd.notna(value):
                    arr[idx] = float(value)

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
                del self._state.root[path]

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

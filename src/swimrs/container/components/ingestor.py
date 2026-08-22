"""
Ingestor component for data ingestion operations.

Provides a clean, namespace-organized API for ingesting data into the container.
Usage: container.ingest.ndvi(...)
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import Component

if TYPE_CHECKING:
    from swimrs.container.components.grid_mapping import GridMapping
    from swimrs.container.provenance import ProvenanceEvent
    from swimrs.container.state import ContainerState

# Parameters stored in the calibration group
CALIBRATION_PARAMS = (
    "aw",
    "mad",
    "ndvi_k",
    "ndvi_0",
    # Linear NDVI-Kcb curve (kcb_ndvi_mode="linear"); absent from sigmoid runs
    "ndvi_alpha",
    "ndvi_beta",
    "swe_alpha",
    "swe_beta",
    "ks_damp",
    "kr_damp",
)

# PEST++ column name mapping → internal names
_PEST_NAME_MAP = {
    "ks_alpha": "ks_damp",
    "kr_alpha": "kr_damp",
    "ndvi_k": "ndvi_k",
    "ndvi_0": "ndvi_0",
    "ndvi_alpha": "ndvi_alpha",
    "ndvi_beta": "ndvi_beta",
    "swe_alpha": "swe_alpha",
    "swe_beta": "swe_beta",
    "aw": "aw",
    "mad": "mad",
}

# Shared ETf validity ceiling for ingested calibration targets.
MAX_VALID_ETF = 2.0


def parse_pest_par_csv(
    par_csv: str | Path,
    fids: list[str],
    summary_stat: str = "median",
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Parse PEST++ .par.csv into per-parameter, per-field values and uncertainties.

    Parameters
    ----------
    par_csv : str | Path
        Path to a PEST++ .par.csv file (realizations × parameters).
    fids : list[str]
        Field UIDs expected in the column names.
    summary_stat : str
        Summary statistic across realizations: ``"median"`` or ``"mean"``.

    Returns
    -------
    values : dict[str, dict[str, float]]
        ``{param_name: {fid: value}}`` using internal names (e.g. ``ks_damp``).
    stds : dict[str, dict[str, float]]
        ``{param_name: {fid: std}}`` across realizations.
    """
    df = pd.read_csv(par_csv, index_col=0)
    numeric_rows = df.loc[df.index != "base"]

    if summary_stat == "median":
        center = numeric_rows.median()
    else:
        center = numeric_rows.mean()
    spread = numeric_rows.std()

    fids_lower = {f.lower(): f for f in fids}

    values: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}

    for col in df.columns:
        # Column format: pname:p_{param}_{fid}_ptype:tied_0_:0
        parts = col.split("_ptype:")[0]
        parts = parts.replace("pname:p_", "")
        parts = parts.rsplit("_:0", 1)[0]

        matched_fid = None
        param_name = None
        for fid_orig in fids:
            if parts.lower().endswith(f"_{fid_orig.lower()}"):
                matched_fid = fid_orig
                param_name = parts[: -(len(fid_orig) + 1)]
                break

        if matched_fid is None or param_name is None:
            continue

        internal_name = _PEST_NAME_MAP.get(param_name, param_name)
        if internal_name not in CALIBRATION_PARAMS:
            continue

        values.setdefault(internal_name, {})[matched_fid] = float(center[col])
        stds.setdefault(internal_name, {})[matched_fid] = float(spread[col])

    return values, stds


def _parse_single_csv(
    csv_file: Path,
    uid_column: str,
    instrument: str,
    known_uids: set[str],
    fields_set: set[str] | None,
) -> list[pd.Series]:
    """Parse one Earth Engine CSV into a list of per-field Series.

    This is a module-level function so it can be dispatched to threads
    without serialising the entire Ingestor instance.

    Args:
        csv_file: Path to a single CSV.
        uid_column: Expected column name for the field UID.
        instrument: ``"landsat"`` or ``"sentinel"`` (controls date parsing).
        known_uids: Set of field UIDs present in the container.
        fields_set: Optional allowlist; ``None`` means accept all known UIDs.

    Returns:
        List of ``pd.Series`` (may be empty if the file is irrelevant).
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return []

    # Handle single-field CSVs where field ID is the first column header
    if uid_column not in df.columns:
        first_col = df.columns[0]
        if first_col in known_uids:
            field_id = first_col
            new_cols = [uid_column] + list(df.columns[1:])
            df.columns = new_cols
            df[uid_column] = df[uid_column].astype(object)
            df.iloc[0, 0] = field_id
        else:
            return []

    # Parse date columns
    non_data_cols = {uid_column, "system:index", ".geo", "lat", "lon", "LAT", "LON"}
    data_cols = []
    dates = []

    for col in df.columns:
        if col in non_data_cols:
            continue
        try:
            if instrument == "landsat":
                parts = col.split("_")
                if len(parts) >= 2:
                    date_str = parts[-1]
                    if len(date_str) == 8 and date_str.isdigit():
                        dates.append(pd.to_datetime(date_str))
                        data_cols.append(col)
            elif instrument == "sentinel":
                date_str = col[:8]
                if len(date_str) == 8 and date_str.isdigit():
                    dates.append(pd.to_datetime(date_str))
                    data_cols.append(col)
            elif instrument in ("ecostress", "merged"):
                parts = col.split("_")
                date_str = parts[-1]
                if len(date_str) == 8 and date_str.isdigit():
                    dates.append(pd.to_datetime(date_str))
                    data_cols.append(col)
        except Exception:
            continue

    if not data_cols:
        return []

    series_list = []
    for _, row in df.iterrows():
        raw_id = row[uid_column]
        # iterrows() upcasts int→float; normalize "1.0" → "1".
        # Only coerce plain numeric strings — do NOT coerce underscore-delimited
        # UIDs like "001_000001" (Python's int() accepts "_" as a separator,
        # which would silently corrupt "001_000001" → 1000001).
        raw_str = str(raw_id)
        if raw_str.replace(".", "", 1).isdigit():
            try:
                raw_id = int(float(raw_id))
            except (ValueError, TypeError):
                pass
        field_id = str(raw_id)

        if fields_set and field_id not in fields_set:
            continue
        if field_id not in known_uids:
            continue

        values = row[data_cols].values
        series = pd.Series(values, index=dates, name=field_id)
        # iterrows() rows take the frame's common dtype; with a string UID
        # column, an all-NaN row becomes str dtype (pandas 3), which breaks
        # the numeric duplicate collapse below. Data columns are numeric by
        # contract, so astype raises on genuine junk instead of masking it.
        series = series.astype("float64")
        series = series.sort_index()

        if series.index.duplicated().any():
            if instrument == "sentinel":
                # S2 granules are cut with a ~10 km apron on the MGRS grid, so
                # fields in the overlap strip appear in 2+ same-date columns
                # from the same overpass — average them, they are not
                # independent observations.
                series = series.groupby(series.index).mean()
            else:
                series = series.groupby(series.index).max()

        series_list.append(series)

    return series_list


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
        workers: int = 1,
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
            workers: Number of threads for parallel CSV reading (default: 1)

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
                source_dir,
                instrument,
                "ndvi",
                uid_column,
                fields,
                mask=mask,
                workers=workers,
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
        workers: int = 1,
        scale_factor: float = 1.0,
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
                Values above ``MAX_VALID_ETF`` are also treated as invalid
                and set to NaN.
            workers: Number of threads for parallel CSV reading (default: 1)
            scale_factor: Multiply all ETf values by this factor (default: 1.0).
                Use to apply a uniform algorithm bias correction (e.g., 0.80
                for a 20% reduction in PT-JPL ETf).

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
                source_dir,
                instrument,
                "etf",
                uid_column,
                fields,
                mask=mask,
                workers=workers,
            )

            if all_data.empty:
                self._log.warning("no_data_found", source=str(source_dir))
                return self._state.provenance.record(
                    "ingest",
                    target=path,
                    source=str(source_dir),
                    params={
                        "model": model,
                        "instrument": instrument,
                        "mask": mask,
                        "min_etf": min_etf,
                        "max_etf": MAX_VALID_ETF,
                        "scale_factor": scale_factor,
                    },
                    records_count=0,
                    success=True,
                )

            # Filter obvious ETf artifacts before writing:
            # - below min_etf: legacy sparse_time_series-style noise removal
            # - above MAX_VALID_ETF: impossible ETf values
            all_data = all_data.where(all_data >= min_etf)
            all_data = all_data.where(all_data <= MAX_VALID_ETF)

            if scale_factor != 1.0:
                all_data = all_data * scale_factor
                # Guard against scaled values crossing the shared ETf ceiling.
                all_data = all_data.where(all_data <= MAX_VALID_ETF)

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
                    "model": model,
                    "instrument": instrument,
                    "mask": mask,
                    "min_etf": min_etf,
                    "max_etf": MAX_VALID_ETF,
                    "scale_factor": scale_factor,
                },
                fields_affected=list(all_data.columns),
                records_count=records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def eta(
        self,
        source_dir: str | Path,
        uid_column: str = "FID",
        mask: str = "irr",
        instrument: str = "landsat",
        fields: list[str] | None = None,
        overwrite: bool = False,
        workers: int = 1,
    ) -> ProvenanceEvent:
        """Ingest monthly ETa from OpenET ensemble CSV exports.

        Expects column names in format ``ensemble_eta_YYYYMM01`` (first of
        month), as produced by ``sid_prepped.py``.

        Args:
            source_dir: Directory containing CSV files
            uid_column: Column name for field UID in CSVs (default: "FID")
            mask: Mask type ("irr", "inv_irr", "no_mask")
            instrument: Source instrument (default: "landsat")
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data
            workers: Number of threads for parallel CSV reading (default: 1)

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()
        source_dir = Path(source_dir)
        path = f"remote_sensing/eta/{instrument}/ensemble/{mask}"

        with self._track_operation(
            "ingest_eta",
            target=path,
            source=str(source_dir),
            instrument=instrument,
            mask=mask,
        ) as ctx:
            if path in self._state.root and not overwrite:
                raise ValueError(f"Data exists at {path}. Use overwrite=True.")

            all_data = self._parse_ee_remote_sensing_csvs(
                source_dir,
                instrument,
                "eta",
                uid_column,
                fields,
                mask=mask,
                workers=workers,
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

            records = self._write_timeseries(path, all_data, fields, overwrite=overwrite)

            ctx["records_processed"] = records
            ctx["fields_processed"] = len(all_data.columns)

            event = self._state.provenance.record(
                "ingest",
                target=path,
                source=str(source_dir),
                source_format="earth_engine_csv",
                params={"instrument": instrument, "mask": mask},
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
            elif isinstance(grid_mapping, str | Path):
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
    # Calibration Ingestion
    # -------------------------------------------------------------------------

    def calibration(
        self,
        par_csv: str | Path,
        fields: list[str] | None = None,
        batch_id: int | None = None,
        summary_stat: str = "median",
    ) -> ProvenanceEvent:
        """Ingest calibrated parameters from a PEST++ .par.csv into the container.

        Creates (or updates) the ``calibration/`` group with best-estimate
        parameter values and uncertainty (std across realizations).  Supports
        incremental batch-by-batch ingestion — only the fields present in
        *par_csv* are overwritten; other indices retain their NaN fill.

        Parameters
        ----------
        par_csv : str | Path
            Path to a PEST++ ``.par.csv`` file.
        fields : list[str], optional
            Subset of field UIDs to ingest.  Defaults to all container fields.
        batch_id : int, optional
            Batch identifier written to ``calibration/metadata/batch_id``.
        summary_stat : str
            ``"median"`` or ``"mean"`` across ensemble realizations.

        Returns
        -------
        ProvenanceEvent
        """
        self._ensure_writable()
        par_csv = Path(par_csv)

        fids = fields if fields else self._state.field_uids

        with self._track_operation(
            "ingest_calibration",
            target="calibration",
            source=str(par_csv),
        ) as ctx:
            values, stds = parse_pest_par_csv(par_csv, fids, summary_stat)

            # Ensure calibration groups exist
            self._state.ensure_group("calibration/parameters")
            self._state.ensure_group("calibration/uncertainty")
            self._state.ensure_group("calibration/metadata")

            # Write parameter and uncertainty arrays
            for param in CALIBRATION_PARAMS:
                for prefix, data_dict in [
                    ("calibration/parameters", values),
                    ("calibration/uncertainty", stds),
                ]:
                    path = f"{prefix}/{param}"
                    if path not in self._state.root:
                        self._state.create_property_array(path, dtype="float64", fill_value=np.nan)
                    arr = self._state.root[path]
                    if param in data_dict:
                        for fid, val in data_dict[param].items():
                            if fid in self._state._uid_to_index:
                                idx = self._state.get_field_index(fid)
                                arr[idx] = val

            # Write metadata arrays
            for meta_name, dtype, fill in [
                ("batch_id", "int32", -1),
                ("calibrated", "uint8", 0),
            ]:
                meta_path = f"calibration/metadata/{meta_name}"
                if meta_path not in self._state.root:
                    self._state.create_property_array(meta_path, dtype=dtype, fill_value=fill)

            meta_batch = self._state.root["calibration/metadata/batch_id"]
            meta_cal = self._state.root["calibration/metadata/calibrated"]

            calibrated_fids = set()
            for param_vals in values.values():
                calibrated_fids.update(param_vals.keys())

            for fid in calibrated_fids:
                if fid in self._state._uid_to_index:
                    idx = self._state.get_field_index(fid)
                    meta_cal[idx] = 1
                    if batch_id is not None:
                        meta_batch[idx] = batch_id

            # Update group-level attrs
            cal_group = self._state.root["calibration"]
            existing_attrs = dict(cal_group.attrs) if cal_group.attrs else {}
            existing_attrs["summary_stat"] = summary_stat
            n_calibrated = int(np.sum(np.asarray(meta_cal[:]) > 0))
            existing_attrs["n_fields_calibrated"] = n_calibrated

            batches_meta = existing_attrs.get("batches", {})
            if isinstance(batches_meta, str):
                batches_meta = json.loads(batches_meta)
            if batch_id is not None:
                batches_meta[str(batch_id)] = {
                    "n_fields": len(calibrated_fids),
                    "status": "ingested",
                }
            existing_attrs["batches"] = json.dumps(batches_meta)
            existing_attrs["n_batches_completed"] = len(batches_meta)

            cal_group.attrs.update(existing_attrs)

            ctx["fields_processed"] = len(calibrated_fids)

            event = self._state.provenance.record(
                "ingest",
                target="calibration",
                source=str(par_csv),
                source_format="pest_par_csv",
                params={"batch_id": batch_id, "summary_stat": summary_stat},
                fields_affected=list(calibrated_fids),
                records_count=len(calibrated_fids),
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
        workers: int = 1,
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
            workers: Number of threads for parallel CSV reading (default: 1)

        Returns:
            DataFrame with DatetimeIndex and field UIDs as columns
        """
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            self._log.warning("no_csv_files", directory=str(source_dir))
            return pd.DataFrame()

        # Filter files by mask if specified
        if mask is not None:
            csv_files = self._filter_files_by_mask(csv_files, mask)
            if not csv_files:
                self._log.debug("no_files_for_mask", mask=mask, directory=str(source_dir))
                return pd.DataFrame()

        known_uids = set(self._state._uid_to_index.keys())
        fields_set = set(fields) if fields else None

        # Parse CSVs — serial or threaded
        if workers <= 1:
            all_series = []
            for csv_file in csv_files:
                all_series.extend(
                    _parse_single_csv(csv_file, uid_column, instrument, known_uids, fields_set)
                )
        else:
            all_series = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _parse_single_csv, csv_file, uid_column, instrument, known_uids, fields_set
                    ): csv_file
                    for csv_file in csv_files
                }
                for future in as_completed(futures):
                    try:
                        all_series.extend(future.result())
                    except Exception as e:
                        self._log.warning(
                            "csv_parse_error", file=str(futures[future]), error=str(e)
                        )

        if not all_series:
            self._log.warning(
                "no_series_created",
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

    @staticmethod
    def _filter_files_by_mask(csv_files: list[Path], mask: str) -> list[Path]:
        """Filter CSV file list by mask pattern in filename."""
        filtered = []
        for f in csv_files:
            filename = f.stem
            if mask == "irr":
                has_irr = filename.endswith("_irr") or "_irr_" in filename
                has_inv_irr = "inv_irr" in filename
                if has_irr and not has_inv_irr:
                    filtered.append(f)
            elif mask == "inv_irr":
                if filename.endswith("_inv_irr") or "_inv_irr_" in filename:
                    filtered.append(f)
            elif mask == "no_mask":
                if filename.endswith("_no_mask") or "_no_mask_" in filename:
                    filtered.append(f)
            else:
                if filename.endswith(f"_{mask}") or f"_{mask}_" in filename:
                    filtered.append(f)
        return filtered

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

        # Coerce to float in bulk (faster than per-column pd.to_numeric)
        values = aligned.values
        if values.dtype == object:
            values = pd.to_numeric(values.ravel(), errors="coerce").reshape(values.shape)
        values = values.astype(np.float64, copy=False)

        # Build column-index mapping and write all fields in one zarr slice
        col_indices = np.array(
            [self._state._uid_to_index[f] for f in common_fields if f in self._state._uid_to_index],
            dtype=np.intp,
        )
        arr[:, col_indices] = values[:, : len(col_indices)]

        return int(np.count_nonzero(~np.isnan(values)))

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
        """Ingest LULC data with override logic.

        Stores GLC10 independently (if available) and MODIS as fallback.
        GLC10 is the primary classification; MODIS is kept for backward
        compatibility with older containers.
        """
        modis_path = "properties/land_cover/modis_lc"
        glc10_path = "properties/land_cover/glc10"

        if modis_path in self._state.root and not overwrite:
            return
        if modis_path in self._state.root:
            self._safe_delete_path(modis_path)
        if glc10_path in self._state.root:
            self._safe_delete_path(glc10_path)

        df = pd.read_csv(lulc_csv)
        df = df.set_index(uid_column)
        df.index = df.index.astype(str)

        # Store GLC10 independently (no mutation)
        if extra_lulc_column and extra_lulc_column in df.columns:
            glc10_arr = self._state.create_property_array(glc10_path, dtype="int16", fill_value=-1)
            for uid in self._state.field_uids:
                if uid in df.index:
                    val = df.loc[uid, extra_lulc_column]
                    if not pd.isna(val):
                        idx = self._state.get_field_index(uid)
                        glc10_arr[idx] = int(val)

        # Apply irrigation-based crop override on MODIS only
        # If mean irrigation > 0.3 and MODIS is not 12, override to 12
        if irrigation_csv:
            irr_df = pd.read_csv(Path(irrigation_csv))
            irr_df = irr_df.set_index(uid_column)
            irr_df.index = irr_df.index.astype(str)

            # Drop non-numeric columns
            numeric_cols = irr_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                mean_irr = irr_df[numeric_cols].mean(axis=1)
                irr_crop_override = (mean_irr > 0.3) & (df[lulc_column] != 12)
                df.loc[irr_crop_override, lulc_column] = 12

        # Write MODIS to container (use -1 as fill_value for integer types)
        arr = self._state.create_property_array(modis_path, dtype="int16", fill_value=-1)

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
        df.index = df.index.astype(str)

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
        df.index = df.index.astype(str)

        # Extract year columns (format: irr_YYYY)
        year_cols = [c for c in df.columns if c.startswith("irr_") and c[4:].isdigit()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return

        # Store mean irrigation
        if mean_path not in self._state.root or overwrite:
            if mean_path in self._state.root:
                self._safe_delete_path(mean_path)

            # Only irr_YYYY columns — irr CSVs may carry LAT/LON, which would
            # poison a mean over all numeric columns.
            mean_irr = df[year_cols if year_cols else numeric_cols].mean(axis=1)
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
        df.index = df.index.astype(str)

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

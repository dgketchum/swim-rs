"""
SWIM Container Ingestion Mixin - data ingestion from various sources.

Provides methods to ingest:
- Remote sensing data (NDVI, ETf) from Earth Engine CSV exports
- Meteorology data (GridMET, ERA5)
- Snow data (SNODAS)
- Static properties (soils, land cover, irrigation)
- Pre-computed dynamics
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import zarr

from swimrs.container.provenance import ProvenanceEvent, DatasetProvenance


class IngestionMixin:
    """
    Mixin providing data ingestion methods.

    Requires ContainerBase attributes:
    - _mode, _root, _uid_to_index, _field_uids, _provenance, _inventory
    - get_time_index(), _create_timeseries_array(), _create_property_array()
    - _mark_modified()
    """

    def ingest_ee_ndvi(self, source_dir: Union[str, Path],
                       instrument: str,
                       mask: str,
                       fields: List[str] = None,
                       overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest NDVI data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files
            instrument: 'landsat' or 'sentinel'
            mask: 'irr', 'inv_irr', or 'no_mask'
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        path = f"remote_sensing/ndvi/{instrument}/{mask}"

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
        target_fields = set(fields) if fields else None

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=["date"])

            for uid in df.columns:
                if uid in ["date", "system:index", ".geo"]:
                    continue
                if uid not in self._uid_to_index:
                    continue
                if target_fields and uid not in target_fields:
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
            params={"instrument": instrument, "mask": mask},
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

        self._mark_modified()
        self._inventory.refresh()

        return event

    def ingest_ee_etf(self, source_dir: Union[str, Path],
                      model: str,
                      mask: str,
                      instrument: str = "landsat",
                      fields: List[str] = None,
                      overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest ETF data from Earth Engine CSV exports.

        Args:
            source_dir: Directory containing CSV files
            model: ET model ('ssebop', 'ptjpl', etc.)
            mask: 'irr', 'inv_irr', or 'no_mask'
            instrument: Remote sensing instrument
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing data

        Returns:
            ProvenanceEvent recording the ingestion
        """
        if self._mode == "r":
            raise ValueError("Cannot ingest: container opened in read-only mode")

        path = f"remote_sensing/etf/{instrument}/{model}/{mask}"

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
        target_fields = set(fields) if fields else None

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=["date"])

            for uid in df.columns:
                if uid in ["date", "system:index", ".geo"]:
                    continue
                if uid not in self._uid_to_index:
                    continue
                if target_fields and uid not in target_fields:
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
            params={"model": model, "mask": mask, "instrument": instrument},
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

        self._mark_modified()
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

        self._mark_modified()
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

        all_params = set()
        for uid_data in all_data.values():
            all_params.update(uid_data.keys())

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

        self._mark_modified()
        self._inventory.refresh()

        return event

    def ingest_snodas(self, source_dir: Union[str, Path] = None,
                      json_path: Union[str, Path] = None,
                      overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest SNODAS snow water equivalent data.

        Can ingest from either Earth Engine CSV exports or a preprocessed JSON file.

        Args:
            source_dir: Directory containing SNODAS CSV files
            json_path: Path to preprocessed SNODAS JSON file
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

        records_count = 0
        fields_found = set()

        if json_path is not None:
            # Ingest from JSON
            import json
            json_path = Path(json_path)
            with open(json_path, "r") as f:
                snodas_data = json.load(f)

            for uid, ts_data in snodas_data.items():
                if uid not in self._uid_to_index:
                    continue
                field_idx = self._uid_to_index[uid]
                fields_found.add(uid)

                for date_str, value in ts_data.items():
                    try:
                        time_idx = self.get_time_index(date_str)
                        if pd.notna(value):
                            arr[time_idx, field_idx] = value
                            records_count += 1
                    except (KeyError, IndexError):
                        continue

            source = str(json_path)
            source_format = "snodas_json"

        elif source_dir is not None:
            # Ingest from CSV
            source_dir = Path(source_dir)
            csv_files = list(source_dir.glob("*.csv"))

            if not csv_files:
                raise ValueError(f"No CSV files found in {source_dir}")

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

            source = str(source_dir)
            source_format = "snodas_csv"
        else:
            raise ValueError("Must provide either source_dir or json_path")

        event = self._provenance.record(
            "ingest",
            target=path,
            source=source,
            source_format=source_format,
            params={},
            fields_affected=list(fields_found),
            records_count=records_count,
        )

        prov = DatasetProvenance()
        prov.record_creation(event.id, source_type=source_format)
        prov.set_coverage(
            fields_present=len(fields_found),
            fields_total=self.n_fields,
            date_range=(str(self.start_date.date()), str(self.end_date.date())),
            missing_fields=[u for u in self._field_uids if u not in fields_found],
        )
        arr.attrs.update(prov.to_dict())

        self._mark_modified()
        self._inventory.refresh()

        return event

    def ingest_properties(self, lulc_csv: Union[str, Path] = None,
                          soils_csv: Union[str, Path] = None,
                          irrigation_csv: Union[str, Path] = None,
                          cdl_csv: Union[str, Path] = None,
                          properties_json: Union[str, Path] = None,
                          uid_column: str = None,
                          lulc_key: str = "mode",
                          fields: List[str] = None,
                          overwrite: bool = False) -> ProvenanceEvent:
        """
        Ingest static field properties from CSV or JSON files.

        Args:
            lulc_csv: Path to land cover CSV (MODIS LC or similar)
            soils_csv: Path to soils CSV (SSURGO or HWSD)
            irrigation_csv: Path to irrigation status CSV (IrrMapper/LANID)
            cdl_csv: Path to Crop Data Layer CSV
            properties_json: Path to pre-built properties JSON (alternative to CSVs)
            uid_column: Column name for field UIDs (uses container's default if None)
            lulc_key: Column name in LULC CSV for land cover code
            fields: Optional list of field UIDs to process
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
        target_fields = set(fields) if fields else None

        # Handle properties JSON format
        if properties_json is not None:
            import json
            props_path = Path(properties_json)
            with open(props_path, "r") as f:
                props_data = json.load(f)

            sources.append(str(properties_json))

            # LULC code
            path = "properties/land_cover/modis_lc"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path, dtype="int16", fill_value=-1)

                for uid, field_props in props_data.items():
                    if uid not in self._uid_to_index:
                        continue
                    if target_fields and uid not in target_fields:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)

                    if "lulc_code" in field_props:
                        arr[field_idx] = int(field_props["lulc_code"])
                        records_count += 1

            # AWC
            path = "properties/soils/awc"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path)

                for uid, field_props in props_data.items():
                    if uid not in self._uid_to_index:
                        continue
                    if target_fields and uid not in target_fields:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)

                    if "awc" in field_props and pd.notna(field_props["awc"]):
                        arr[field_idx] = float(field_props["awc"])
                        records_count += 1

        # Land cover from CSV
        if lulc_csv is not None:
            lulc_path = Path(lulc_csv)
            if not lulc_path.exists():
                raise FileNotFoundError(f"LULC file not found: {lulc_csv}")

            lc_df = pd.read_csv(lulc_path, index_col=uid_column)
            sources.append(str(lulc_csv))

            path = "properties/land_cover/modis_lc"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path, dtype="int16", fill_value=-1)

                for uid in lc_df.index:
                    uid = str(uid)
                    if uid not in self._uid_to_index:
                        continue
                    if target_fields and uid not in target_fields:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)

                    if lulc_key in lc_df.columns:
                        value = lc_df.loc[uid, lulc_key] if uid in lc_df.index else None
                        if pd.notna(value):
                            arr[field_idx] = int(value)
                            records_count += 1

        # Soils from CSV
        if soils_csv is not None:
            soils_path = Path(soils_csv)
            if not soils_path.exists():
                raise FileNotFoundError(f"Soils file not found: {soils_csv}")

            soil_df = pd.read_csv(soils_path, index_col=uid_column)
            sources.append(str(soils_csv))

            # Handle HWSD format
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
                        if target_fields and uid not in target_fields:
                            continue
                        field_idx = self._uid_to_index[uid]
                        fields_found.add(uid)

                        value = soil_df.loc[uid, prop] if uid in soil_df.index else None
                        if pd.notna(value):
                            arr[field_idx] = float(value)
                            records_count += 1

        # Irrigation status
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

            path = "properties/irrigation/irr"
            if path not in self._root or overwrite:
                if path in self._root:
                    del self._root[path]
                arr = self._create_property_array(path)

                for uid in irr_df.index:
                    uid = str(uid)
                    if uid not in self._uid_to_index:
                        continue
                    if target_fields and uid not in target_fields:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)

                    values = irr_df.loc[uid].values if uid in irr_df.index else []
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

        self._mark_modified()
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

        # ke_max
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

        # kc_max
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
        if "irr" in dynamics:
            path = "derived/dynamics/irr_data"
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

                for uid, year_data in dynamics["irr"].items():
                    if uid not in self._uid_to_index:
                        continue
                    field_idx = self._uid_to_index[uid]
                    fields_found.add(uid)
                    arr[field_idx] = json.dumps(year_data)
                    records_count += 1

        # Groundwater subsidy data
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

        self._mark_modified()
        self._inventory.refresh()

        return event

"""
SWIM Container Query Mixin - data access and status queries.

Provides methods to:
- Query container status and validation
- Access data as DataFrames
- Get field time series and geometries
- Retrieve dynamics data
"""

from typing import Dict, List, Union, Any

import numpy as np
import pandas as pd

from swimrs.container.schema import Instrument, MaskType, ETModel, MetSource
from swimrs.container.inventory import ValidationResult, DataStatus


class QueryMixin:
    """
    Mixin providing data access and query methods.

    Requires ContainerBase attributes:
    - _root, _uid_to_index, _field_uids, _time_index, _inventory
    - get_field_index(), get_time_index()
    - n_fields, start_date, end_date, project_name, path
    """

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

        if fields is None:
            field_indices = slice(None)
            field_cols = self._field_uids
        else:
            field_indices = [self.get_field_index(f) for f in fields]
            field_cols = fields

        if arr.ndim == 2:  # Time series (time, field)
            data = arr[:, field_indices]
            df = pd.DataFrame(data, index=self._time_index, columns=field_cols)

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

        paths_to_check = []
        if parameters:
            for param in parameters:
                if param == "ndvi":
                    paths_to_check.append("remote_sensing/ndvi/landsat/irr")
                elif param == "etf":
                    paths_to_check.append("remote_sensing/etf/landsat/ssebop/irr")
                elif param in ["eto", "prcp", "tmin", "tmax"]:
                    paths_to_check.append(f"meteorology/gridmet/{param}")
        else:
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

        wkb_data = self._root["geometry/wkb"][:]
        geometries = [wkb.loads(w) for w in wkb_data]

        gdf = gpd.GeoDataFrame(
            {"uid": self._field_uids},
            geometry=geometries,
            crs="EPSG:4326"
        )

        gdf["lon"] = self._root["geometry/lon"][:]
        gdf["lat"] = self._root["geometry/lat"][:]
        gdf["area_m2"] = self._root["geometry/area_m2"][:]

        if "geometry/properties" in self._root:
            props_grp = self._root["geometry/properties"]
            for key in props_grp.keys():
                gdf[key] = props_grp[key][:]

        return gdf

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

        for key in ["ke_max", "kc_max"]:
            path = f"derived/dynamics/{key}"
            if path in self._root:
                result[key] = float(self._root[path][idx])

        for key in ["irr_data", "gwsub_data"]:
            path = f"derived/dynamics/{key}"
            if path in self._root:
                json_str = self._root[path][idx]
                if json_str:
                    result[key.replace("_data", "")] = json.loads(json_str)

        return result

"""
SWIM Container Export Mixin - data export in various formats.

Provides methods to export:
- Shapefiles (field geometries)
- CSV files (time series data)
- prepped_input.json (model-ready format)
"""

from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np
import pandas as pd

from swimrs.container.provenance import ProvenanceEvent


class ExportMixin:
    """
    Mixin providing data export methods.

    Requires ContainerBase attributes:
    - _root, _uid_to_index, _field_uids, _provenance, _time_index
    - get_field_index(), start_date, end_date, n_fields
    - get_dataframe() from QueryMixin
    """

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
            df = df.reset_index()
            df = df.rename(columns={"index": "date"})
            output_file = output_dir / f"{path.replace('/', '_')}.csv"
            df.to_csv(output_file, index=False)
        else:
            output_file = output_dir / f"{path.replace('/', '_')}.csv"
            df.to_csv(output_file)

        event = self._provenance.record(
            "export",
            source=path,
            target=str(output_file),
            params={"format": format},
        )

        return event

    def export_prepped_input_json(self, output_path: Union[str, Path],
                                    etf_model: str = "ssebop",
                                    masks: tuple = ("irr", "inv_irr"),
                                    met_source: str = None,
                                    instrument: str = "landsat",
                                    fields: List[str] = None,
                                    use_fused_ndvi: bool = True) -> ProvenanceEvent:
        """
        Export data in prepped_input.json format for SWIM-RS model consumption.

        This replaces the prep_fields_json() workflow by reading directly from
        the container instead of multiple Parquet files. The output JSON can be
        loaded directly by SamplePlots.initialize_plot_data().

        Args:
            output_path: Path for the output JSON file
            etf_model: ET model to use (ssebop, ptjpl, etc.)
            masks: Mask types to include (irr, inv_irr, no_mask)
            met_source: Meteorology source (gridmet, era5). Auto-detected if None.
            instrument: Remote sensing instrument (landsat, sentinel)
            fields: List of field UIDs to include (None for all)
            use_fused_ndvi: If True, use fused NDVI from derived/combined_ndvi

        Returns:
            ProvenanceEvent recording the export
        """
        import json as _json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fields is None:
            fields = self._field_uids

        if met_source is None:
            if "meteorology/gridmet/eto" in self._root:
                met_source = "gridmet"
            elif "meteorology/era5/eto" in self._root:
                met_source = "era5"
            else:
                raise ValueError("No meteorology data found in container")

        output = {
            'order': fields,
            'props': self._build_props_dict(fields),
            'time_series': self._build_timeseries_dict(
                fields, etf_model, masks, met_source, instrument, use_fused_ndvi
            ),
            'irr_data': self._get_irr_data_dict(fields),
            'gwsub_data': self._get_gwsub_data_dict(fields),
            'ke_max': self._get_scalar_dynamics_dict(fields, "ke_max"),
            'kc_max': self._get_scalar_dynamics_dict(fields, "kc_max"),
            'missing': [],
        }

        with open(output_path, 'w') as f:
            for key, value in output.items():
                _json.dump({key: value}, f)
                f.write('\n')

        print(f"Exported prepped_input.json to {output_path} ({len(fields)} fields)")

        event = self._provenance.record(
            "export",
            target=str(output_path),
            params={
                "format": "prepped_input_json",
                "etf_model": etf_model,
                "masks": list(masks),
                "met_source": met_source,
                "instrument": instrument,
            },
            fields_affected=fields,
            records_count=len(fields),
        )

        return event

    def _build_props_dict(self, fields: List[str]) -> Dict[str, Dict[str, Any]]:
        """Build properties dict for each field in prepped_input format."""
        from swimrs.prep import MAX_EFFECTIVE_ROOTING_DEPTH as RZ

        props = {}

        for uid in fields:
            if uid not in self._uid_to_index:
                continue

            idx = self._uid_to_index[uid]
            field_props = {}

            # LULC code and derived values
            lulc_path = "properties/land_cover/modis_lc"
            if lulc_path in self._root:
                lulc_code = int(self._root[lulc_path][idx])
                field_props["lulc_code"] = lulc_code

                rz_info = RZ.get(str(lulc_code), {})
                field_props["root_depth"] = rz_info.get("rooting_depth", 0.55)
                field_props["zr_mult"] = rz_info.get("zr_multiplier", 1.0)
            else:
                field_props["lulc_code"] = 12  # Default to cropland
                field_props["root_depth"] = 0.55
                field_props["zr_mult"] = 1.0

            # Soils
            for soil_prop in ["awc", "ksat", "clay", "sand"]:
                path = f"properties/soils/{soil_prop}"
                if path in self._root:
                    val = self._root[path][idx]
                    if not np.isnan(val):
                        field_props[soil_prop] = float(val)

            # Area
            if "geometry/area_m2" in self._root:
                field_props["area_sq_m"] = float(self._root["geometry/area_m2"][idx])

            # Centroid
            if "geometry/lon" in self._root:
                field_props["centroid_lon"] = float(self._root["geometry/lon"][idx])
            if "geometry/lat" in self._root:
                field_props["centroid_lat"] = float(self._root["geometry/lat"][idx])

            props[uid] = field_props

        return props

    def _build_timeseries_dict(self, fields: List[str], etf_model: str,
                                masks: tuple, met_source: str, instrument: str,
                                use_fused_ndvi: bool) -> Dict[str, Dict[str, Any]]:
        """Build time_series dict in prepped_input format."""
        met_var_map = {
            "tmin": f"meteorology/{met_source}/tmin",
            "tmax": f"meteorology/{met_source}/tmax",
            "prcp": f"meteorology/{met_source}/prcp",
            "srad": f"meteorology/{met_source}/srad",
            "eto": f"meteorology/{met_source}/eto",
            "eto_corr": f"meteorology/{met_source}/eto_corr",
            "etr": f"meteorology/{met_source}/etr",
            "etr_corr": f"meteorology/{met_source}/etr_corr",
        }

        field_indices = [self._uid_to_index[f] for f in fields if f in self._uid_to_index]

        time_series = {}

        for i, date in enumerate(self._time_index):
            dt_str = date.strftime("%Y-%m-%d")
            day_data = {"doy": date.dayofyear}

            # Meteorology variables
            for var_name, path in met_var_map.items():
                if path in self._root:
                    arr = self._root[path]
                    values = [float(arr[i, idx]) if not np.isnan(arr[i, idx]) else None
                              for idx in field_indices]
                    day_data[var_name] = values
                elif var_name in ["eto_corr", "etr_corr"]:
                    day_data[var_name] = [None] * len(field_indices)

            # SWE
            swe_path = "snow/snodas/swe"
            if swe_path not in self._root:
                swe_path = "snow/era5/swe"
            if swe_path in self._root:
                arr = self._root[swe_path]
                day_data["swe"] = [float(arr[i, idx]) if not np.isnan(arr[i, idx]) else 0.0
                                   for idx in field_indices]
            else:
                day_data["swe"] = [0.0] * len(field_indices)

            # NLDAS precip placeholders
            day_data["nld_ppt_d"] = day_data.get("prcp", [0.0] * len(field_indices))
            for hr in range(24):
                hr_key = f"prcp_hr_{hr:02d}"
                if "prcp" in day_data:
                    day_data[hr_key] = [v / 24.0 if v is not None else 0.0
                                        for v in day_data["prcp"]]
                else:
                    day_data[hr_key] = [0.0] * len(field_indices)

            # NDVI per mask
            for mask in masks:
                if use_fused_ndvi:
                    ndvi_path = f"derived/combined_ndvi/{mask}"
                else:
                    ndvi_path = f"remote_sensing/ndvi/{instrument}/{mask}"

                key = f"ndvi_{mask}"
                if ndvi_path in self._root:
                    arr = self._root[ndvi_path]
                    day_data[key] = [float(arr[i, idx]) if not np.isnan(arr[i, idx]) else None
                                     for idx in field_indices]
                else:
                    fallback_path = f"remote_sensing/ndvi/{instrument}/{mask}"
                    if fallback_path in self._root:
                        arr = self._root[fallback_path]
                        day_data[key] = [float(arr[i, idx]) if not np.isnan(arr[i, idx]) else None
                                         for idx in field_indices]
                    else:
                        day_data[key] = [None] * len(field_indices)

            # ETf per model and mask
            for mask in masks:
                etf_path = f"remote_sensing/etf/{instrument}/{etf_model}/{mask}"
                key = f"{etf_model}_etf_{mask}"
                if etf_path in self._root:
                    arr = self._root[etf_path]
                    day_data[key] = [float(arr[i, idx]) if not np.isnan(arr[i, idx]) else None
                                     for idx in field_indices]
                else:
                    day_data[key] = [None] * len(field_indices)

            time_series[dt_str] = day_data

        return time_series

    def _get_irr_data_dict(self, fields: List[str]) -> Dict[str, Dict]:
        """Get irrigation data dict from dynamics."""
        import json as _json

        path = "derived/dynamics/irr_data"
        if path not in self._root:
            return {f: {} for f in fields}

        result = {}
        for uid in fields:
            if uid not in self._uid_to_index:
                result[uid] = {}
                continue

            idx = self._uid_to_index[uid]
            json_str = self._root[path][idx]
            if json_str:
                data = _json.loads(json_str)
                result[uid] = {int(k) if k.isdigit() else k: v for k, v in data.items()}
            else:
                result[uid] = {}

        return result

    def _get_gwsub_data_dict(self, fields: List[str]) -> Dict[str, Dict]:
        """Get groundwater subsidy data dict from dynamics."""
        import json as _json

        path = "derived/dynamics/gwsub_data"
        if path not in self._root:
            return {f: {} for f in fields}

        result = {}
        for uid in fields:
            if uid not in self._uid_to_index:
                result[uid] = {}
                continue

            idx = self._uid_to_index[uid]
            json_str = self._root[path][idx]
            if json_str:
                data = _json.loads(json_str)
                result[uid] = {int(k) if k.isdigit() else k: v for k, v in data.items()}
            else:
                result[uid] = {}

        return result

    def _get_scalar_dynamics_dict(self, fields: List[str], param: str) -> Dict[str, float]:
        """Get scalar dynamics parameter (ke_max or kc_max) as dict."""
        path = f"derived/dynamics/{param}"
        if path not in self._root:
            default = 1.0 if param == "ke_max" else 1.25
            return {f: default for f in fields}

        result = {}
        for uid in fields:
            if uid not in self._uid_to_index:
                result[uid] = 1.0 if param == "ke_max" else 1.25
                continue

            idx = self._uid_to_index[uid]
            val = self._root[path][idx]
            result[uid] = float(val) if not np.isnan(val) else (1.0 if param == "ke_max" else 1.25)

        return result

    def export_model_inputs(self, output_dir: Union[str, Path],
                           model: str = "ssebop",
                           mask: str = "irr",
                           fields: List[str] = None) -> ProvenanceEvent:
        """
        Export data in the format required by the SWIM-RS model.

        This is a convenience wrapper around export_prepped_input_json().

        Args:
            output_dir: Output directory
            model: ET model to use
            mask: Mask type
            fields: List of field UIDs (None for all ready fields)

        Returns:
            ProvenanceEvent recording the export
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "prepped_input.json"

        if mask == "irr":
            masks = ("irr", "inv_irr")
        elif mask == "inv_irr":
            masks = ("inv_irr",)
        else:
            masks = ("no_mask",)

        return self.export_prepped_input_json(
            output_path=output_path,
            etf_model=model,
            masks=masks,
            fields=fields,
        )

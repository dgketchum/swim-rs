"""
SWIM Container Compute Mixin - derived data computation.

Provides methods to compute:
- Dynamics (irrigation detection, K-parameters)
- Fused NDVI (Landsat + Sentinel using quantile mapping)
"""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from swimrs.container.provenance import ProvenanceEvent


class ComputeMixin:
    """
    Mixin providing derived data computation methods.

    Requires ContainerBase attributes:
    - _mode, _root, _uid_to_index, _field_uids, _provenance, _inventory, _time_index
    - get_time_index(), _create_timeseries_array(), _create_property_array()
    - _mark_modified(), _ensure_group()
    - start_date, end_date, n_fields, n_days
    """

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

        with tempfile.TemporaryDirectory() as tmpdir:
            ts_dir = Path(tmpdir) / "timeseries"
            ts_dir.mkdir()
            props_file = Path(tmpdir) / "properties.json"
            dynamics_file = Path(tmpdir) / "dynamics.json"

            # Export per-field timeseries as Parquet
            self._export_field_timeseries_for_dynamics(
                ts_dir, fields, met_source, etf_model, masks, instruments
            )

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

            # Irrigation fraction
            path = "properties/irrigation/irr"
            if path in self._root:
                mean_irr = float(self._root[path][idx])
                years = range(self.start_date.year, self.end_date.year + 1)
                field_props["irr"] = {str(yr): mean_irr for yr in years}

            # Area
            if "geometry/area_m2" in self._root:
                field_props["area_sq_m"] = float(self._root["geometry/area_m2"][idx])

            if field_props:
                props[uid] = field_props

        with open(output_file, "w") as f:
            json.dump(props, f, indent=2)

    def compute_fused_ndvi(self, masks: tuple = ("irr", "inv_irr", "no_mask"),
                           min_pairs: int = 20,
                           window_days: int = 1,
                           overwrite: bool = False) -> ProvenanceEvent:
        """
        Compute fused NDVI from Landsat and Sentinel observations.

        Uses quantile mapping to adjust Sentinel NDVI to match Landsat,
        then combines both sources. This replicates the fusion done in
        prep_plots.py and dynamics.py.

        Args:
            masks: Mask types to process
            min_pairs: Minimum number of paired observations for quantile mapping
            window_days: Rolling window for matching observations
            overwrite: If True, replace existing fused NDVI

        Returns:
            ProvenanceEvent recording the computation
        """
        if self._mode == "r":
            raise ValueError("Cannot compute: container opened in read-only mode")

        from swimrs.prep.ndvi_regression import sentinel_adjust_quantile_mapping

        records_count = 0
        fields_processed = set()

        for mask in masks:
            landsat_path = f"remote_sensing/ndvi/landsat/{mask}"
            sentinel_path = f"remote_sensing/ndvi/sentinel/{mask}"
            output_path = f"derived/combined_ndvi/{mask}"

            if output_path in self._root and not overwrite:
                print(f"Fused NDVI already exists at {output_path}, skipping")
                continue

            has_landsat = landsat_path in self._root
            has_sentinel = sentinel_path in self._root

            if not has_landsat and not has_sentinel:
                print(f"No NDVI data found for mask={mask}, skipping")
                continue

            if output_path in self._root:
                del self._root[output_path]
            output_arr = self._create_timeseries_array(output_path)

            for uid in self._field_uids:
                idx = self._uid_to_index[uid]
                fields_processed.add(uid)

                if has_landsat and has_sentinel:
                    landsat_data = self._root[landsat_path][:, idx]
                    sentinel_data = self._root[sentinel_path][:, idx]

                    landsat_df = pd.DataFrame(
                        {uid: landsat_data},
                        index=self._time_index
                    )
                    sentinel_df = pd.DataFrame(
                        {uid: sentinel_data},
                        index=self._time_index
                    )

                    try:
                        sentinel_adjusted = sentinel_adjust_quantile_mapping(
                            sentinel_ndvi_df=sentinel_df,
                            landsat_ndvi_df=landsat_df,
                            min_pairs=min_pairs,
                            window_days=window_days
                        )

                        combined = pd.concat(
                            [landsat_df[uid], pd.Series(sentinel_adjusted, index=self._time_index)],
                            axis=1
                        ).mean(axis=1)

                        output_arr[:, idx] = combined.values
                        records_count += int(combined.notna().sum())

                    except Exception:
                        output_arr[:, idx] = landsat_data
                        records_count += int(np.sum(~np.isnan(landsat_data)))

                elif has_landsat:
                    output_arr[:, idx] = self._root[landsat_path][:, idx]
                    records_count += int(np.sum(~np.isnan(self._root[landsat_path][:, idx])))

                elif has_sentinel:
                    output_arr[:, idx] = self._root[sentinel_path][:, idx]
                    records_count += int(np.sum(~np.isnan(self._root[sentinel_path][:, idx])))

            print(f"Created fused NDVI at {output_path}")

        event = self._provenance.record(
            "compute",
            target="derived/combined_ndvi",
            params={
                "masks": list(masks),
                "min_pairs": min_pairs,
                "window_days": window_days,
            },
            fields_affected=list(fields_processed),
            records_count=records_count,
        )

        self._mark_modified()
        self._inventory.refresh()

        return event

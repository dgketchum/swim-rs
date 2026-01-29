#!/usr/bin/env python
"""
Convert Legacy prepped_input.json to SwimContainer
==================================================

This script converts the legacy prepped_input.json format (used by SamplePlots)
into a SwimContainer that can be used with the new process package.

This is essential for parity testing: running the same input data through
both the legacy code path and the new code path to verify equivalent outputs.

Usage
-----
    # Convert multi_station test fixture
    python scripts/convert_legacy_input.py \
        --json tests/fixtures/multi_station/golden/prepped_input.json \
        --shapefile tests/fixtures/multi_station/data/gis/multi_station.shp \
        --output tests/fixtures/multi_station/golden/converted.swim \
        --uid-column FID

    # With spinup and calibrated params
    python scripts/convert_legacy_input.py \
        --json tests/fixtures/multi_station/golden/prepped_input.json \
        --shapefile tests/fixtures/multi_station/data/gis/multi_station.shp \
        --output tests/fixtures/multi_station/golden/converted.swim \
        --spinup tests/fixtures/multi_station/golden/spinup.json \
        --calibrated-params tests/fixtures/multi_station/golden/calibrated_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_prepped_input(json_path: Path) -> dict:
    """Load legacy prepped_input.json file.

    Handles both single JSON and JSONL (line-delimited) formats.

    Parameters
    ----------
    json_path : Path
        Path to prepped_input.json

    Returns
    -------
    dict
        Loaded input data with keys: order, props, time_series,
        irr_data, gwsub_data, ke_max, kc_max
    """
    with open(json_path, encoding="utf-8") as f:
        content = f.read()

    # Try single JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try JSONL format
    data = {}
    for line in content.strip().split("\n"):
        if line.strip():
            data.update(json.loads(line))
    return data


def get_date_range(time_series: dict) -> tuple[datetime, datetime]:
    """Extract start and end dates from time_series keys.

    Parameters
    ----------
    time_series : dict
        Time series data keyed by date strings

    Returns
    -------
    tuple[datetime, datetime]
        (start_date, end_date)
    """
    dates = sorted(time_series.keys())
    start = datetime.strptime(dates[0], "%Y-%m-%d")
    end = datetime.strptime(dates[-1], "%Y-%m-%d")
    return start, end


def convert_to_container(
    json_path: Path,
    shapefile_path: Path,
    output_path: Path,
    uid_column: str = "FID",
    spinup_path: Path | None = None,
    calibrated_params_path: Path | None = None,
    met_source: str = "gridmet",
    overwrite: bool = False,
) -> None:
    """Convert prepped_input.json to SwimContainer.

    Parameters
    ----------
    json_path : Path
        Path to legacy prepped_input.json
    shapefile_path : Path
        Path to shapefile with field geometries
    output_path : Path
        Output path for .swim container
    uid_column : str
        Column name for field UIDs in shapefile
    spinup_path : Path, optional
        Path to spinup.json for initial state
    calibrated_params_path : Path, optional
        Path to calibrated_params.json
    met_source : str
        Meteorology source name (gridmet, era5)
    overwrite : bool
        If True, overwrite existing container
    """
    from swimrs.container import SwimContainer

    print(f"Loading prepped_input.json from {json_path}...")
    data = load_prepped_input(json_path)

    # Extract metadata
    field_order = data["order"]
    n_fields = len(field_order)
    start_date, end_date = get_date_range(data["time_series"])
    n_days = (end_date - start_date).days + 1

    print(f"  Fields: {field_order}")
    print(f"  Date range: {start_date.date()} to {end_date.date()} ({n_days} days)")

    # Check if output exists
    if output_path.exists():
        if overwrite:
            print(f"Removing existing container: {output_path}")
            import shutil

            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        else:
            raise FileExistsError(
                f"Container already exists: {output_path}. Use --overwrite to replace."
            )

    # Create container
    print(f"\nCreating SwimContainer at {output_path}...")
    container = SwimContainer.create(
        uri=str(output_path),
        fields_shapefile=str(shapefile_path),
        uid_column=uid_column,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        project_name="legacy_conversion",
        storage="directory",
    )

    # Build field index mapping (legacy order -> container order)
    legacy_to_container_idx = {}
    for legacy_idx, fid in enumerate(field_order):
        if fid in container._uid_to_index:
            legacy_to_container_idx[legacy_idx] = container._uid_to_index[fid]
        else:
            print(f"  WARNING: Field {fid} not found in shapefile")

    # Write properties
    print("\nWriting properties...")
    _write_properties(container, data, field_order, legacy_to_container_idx)

    # Write time series
    print("\nWriting time series...")
    _write_time_series(
        container, data, field_order, legacy_to_container_idx, start_date, end_date, met_source
    )

    # Write dynamics
    print("\nWriting dynamics...")
    _write_dynamics(container, data, field_order)

    # Save and close
    container.save()
    container.close()

    print(f"\nContainer created successfully: {output_path}")

    # Verify
    print("\nVerifying container...")
    container = SwimContainer.open(str(output_path), mode="r")
    print(f"  Fields: {container.field_uids}")
    print(f"  Date range: {container.start_date} to {container.end_date}")
    container.close()


def _write_properties(
    container,
    data: dict,
    field_order: list[str],
    legacy_to_container_idx: dict[int, int],
) -> None:
    """Write static properties to container."""
    from zarr.core.dtype import VariableLengthUTF8

    props = data["props"]
    n_fields = len(container.field_uids)

    # Ensure groups exist
    container._root.require_group("properties/soils")
    container._root.require_group("properties/land_cover")
    container._root.require_group("properties/irrigation")

    # Soils: awc, ksat, clay, sand
    for var in ["awc", "ksat", "clay", "sand"]:
        path = f"properties/soils/{var}"
        parent = container._root["properties/soils"]
        if var in parent:
            arr = parent[var]
        else:
            arr = parent.create_array(var, shape=(n_fields,), dtype="float32", fill_value=np.nan)

        for legacy_idx, fid in enumerate(field_order):
            if legacy_idx in legacy_to_container_idx:
                container_idx = legacy_to_container_idx[legacy_idx]
                val = props.get(fid, {}).get(var)
                if val is not None:
                    arr[container_idx] = float(val)
        print(f"  Written: {path}")

    # Land cover: modis_lc
    path = "properties/land_cover/modis_lc"
    parent = container._root["properties/land_cover"]
    if "modis_lc" in parent:
        arr = parent["modis_lc"]
    else:
        arr = parent.create_array("modis_lc", shape=(n_fields,), dtype="int32", fill_value=-1)
    for legacy_idx, fid in enumerate(field_order):
        if legacy_idx in legacy_to_container_idx:
            container_idx = legacy_to_container_idx[legacy_idx]
            val = props.get(fid, {}).get("lulc_code")
            if val is not None:
                arr[container_idx] = int(val)
    print(f"  Written: {path}")

    # Irrigation: mean and per-year
    path = "properties/irrigation/irr"
    parent = container._root["properties/irrigation"]
    if "irr" in parent:
        arr = parent["irr"]
    else:
        arr = parent.create_array("irr", shape=(n_fields,), dtype="float32", fill_value=np.nan)
    for legacy_idx, fid in enumerate(field_order):
        if legacy_idx in legacy_to_container_idx:
            container_idx = legacy_to_container_idx[legacy_idx]
            irr = props.get(fid, {}).get("irr")
            if isinstance(irr, dict):
                # Average across years
                vals = [v for v in irr.values() if isinstance(v, (int, float))]
                arr[container_idx] = np.mean(vals) if vals else 0.0
            elif irr is not None:
                arr[container_idx] = float(irr)
    print(f"  Written: {path}")

    # Per-year irrigation as JSON strings
    path = "properties/irrigation/irr_yearly"
    if "irr_yearly" in parent:
        arr = parent["irr_yearly"]
    else:
        arr = parent.create_array(
            "irr_yearly",
            shape=(n_fields,),
            dtype=VariableLengthUTF8(),
        )
    # Build list of values then assign at once
    values = ["{}"] * n_fields
    for legacy_idx, fid in enumerate(field_order):
        if legacy_idx in legacy_to_container_idx:
            container_idx = legacy_to_container_idx[legacy_idx]
            irr = props.get(fid, {}).get("irr")
            if isinstance(irr, dict):
                values[container_idx] = json.dumps(irr)
    arr[:] = values
    print(f"  Written: {path}")


def _write_time_series(
    container,
    data: dict,
    field_order: list[str],
    legacy_to_container_idx: dict[int, int],
    start_date: datetime,
    end_date: datetime,
    met_source: str,
) -> None:
    """Write time series data to container."""
    ts = data["time_series"]
    n_days = (end_date - start_date).days + 1
    n_fields = len(container.field_uids)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Build date to index mapping
    date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}

    # Meteorology variables
    met_vars = ["tmin", "tmax", "prcp", "srad", "eto", "eto_corr"]

    met_group = container._root.require_group(f"meteorology/{met_source}")
    for var_name in met_vars:
        if var_name in met_group:
            arr = met_group[var_name]
        else:
            arr = met_group.create_array(
                var_name, shape=(n_days, n_fields), dtype="float32", fill_value=np.nan
            )

        for date_str, day_data in ts.items():
            if date_str not in date_to_idx:
                continue
            day_idx = date_to_idx[date_str]
            values = day_data.get(var_name)
            if values is None:
                continue

            for legacy_idx, val in enumerate(values):
                if legacy_idx in legacy_to_container_idx and val is not None:
                    container_idx = legacy_to_container_idx[legacy_idx]
                    arr[day_idx, container_idx] = float(val)

        print(f"  Written: meteorology/{met_source}/{var_name}")

    # Snow
    snow_group = container._root.require_group("snow/snodas")
    if "swe" in snow_group:
        arr = snow_group["swe"]
    else:
        arr = snow_group.create_array(
            "swe", shape=(n_days, n_fields), dtype="float32", fill_value=np.nan
        )
    for date_str, day_data in ts.items():
        if date_str not in date_to_idx:
            continue
        day_idx = date_to_idx[date_str]
        values = day_data.get("swe")
        if values is None:
            continue
        for legacy_idx, val in enumerate(values):
            if legacy_idx in legacy_to_container_idx and val is not None:
                container_idx = legacy_to_container_idx[legacy_idx]
                arr[day_idx, container_idx] = float(val)
    print("  Written: snow/snodas/swe")

    # NDVI (irr and inv_irr masks)
    # In container schema: remote_sensing/ndvi/{instrument}/{mask} is an ARRAY (n_days, n_fields)
    ndvi_parent = container._root.require_group("remote_sensing/ndvi/landsat")
    for mask in ["irr", "inv_irr"]:
        if mask in ndvi_parent:
            arr = ndvi_parent[mask]
        else:
            arr = ndvi_parent.create_array(
                mask, shape=(n_days, n_fields), dtype="float32", fill_value=np.nan
            )

        var_name = f"ndvi_{mask}"
        for date_str, day_data in ts.items():
            if date_str not in date_to_idx:
                continue
            day_idx = date_to_idx[date_str]
            values = day_data.get(var_name)
            if values is None:
                continue
            for legacy_idx, val in enumerate(values):
                if legacy_idx in legacy_to_container_idx and val is not None:
                    container_idx = legacy_to_container_idx[legacy_idx]
                    arr[day_idx, container_idx] = float(val)
        print(f"  Written: remote_sensing/ndvi/landsat/{mask}")

    # Also create merged_ndvi (combine masks)
    merged_group = container._root.require_group("derived/merged_ndvi")
    for mask in ["irr", "inv_irr"]:
        if mask not in merged_group:
            # Copy from raw NDVI - data parameter infers shape and dtype
            source_arr = ndvi_parent[mask][:].astype("float32")
            merged_group.create_array(mask, data=source_arr)
        print(f"  Written: derived/merged_ndvi/{mask}")

    # ETf (multiple models, irr and inv_irr masks)
    etf_models = ["ssebop", "ptjpl", "sims"]
    for model in etf_models:
        for mask in ["irr", "inv_irr"]:
            var_name = f"{model}_etf_{mask}"

            # Check if this variable exists in time series
            sample_date = list(ts.keys())[0]
            if var_name not in ts[sample_date]:
                continue

            etf_group = container._root.require_group(f"remote_sensing/etf/landsat/{model}")
            if mask in etf_group:
                arr = etf_group[mask]
            else:
                arr = etf_group.create_array(
                    mask, shape=(n_days, n_fields), dtype="float32", fill_value=np.nan
                )

            for date_str, day_data in ts.items():
                if date_str not in date_to_idx:
                    continue
                day_idx = date_to_idx[date_str]
                values = day_data.get(var_name)
                if values is None:
                    continue
                for legacy_idx, val in enumerate(values):
                    if legacy_idx in legacy_to_container_idx and val is not None:
                        container_idx = legacy_to_container_idx[legacy_idx]
                        arr[day_idx, container_idx] = float(val)
            print(f"  Written: remote_sensing/etf/landsat/{model}/{mask}")


def _write_dynamics(
    container,
    data: dict,
    field_order: list[str],
) -> None:
    """Write dynamics data to container."""
    from zarr.core.dtype import VariableLengthUTF8

    n_fields = len(container.field_uids)

    dyn_group = container._root.require_group("derived/dynamics")

    # ke_max
    ke_max = data.get("ke_max", {})
    if "ke_max" in dyn_group:
        arr = dyn_group["ke_max"]
    else:
        arr = dyn_group.create_array(
            "ke_max", shape=(n_fields,), dtype="float32", fill_value=np.nan
        )
    for fid in field_order:
        if fid in container._uid_to_index:
            idx = container._uid_to_index[fid]
            val = ke_max.get(fid)
            if val is not None:
                arr[idx] = float(val)
    print("  Written: derived/dynamics/ke_max")

    # kc_max
    kc_max = data.get("kc_max", {})
    if "kc_max" in dyn_group:
        arr = dyn_group["kc_max"]
    else:
        arr = dyn_group.create_array(
            "kc_max", shape=(n_fields,), dtype="float32", fill_value=np.nan
        )
    for fid in field_order:
        if fid in container._uid_to_index:
            idx = container._uid_to_index[fid]
            val = kc_max.get(fid)
            if val is not None:
                arr[idx] = float(val)
    print("  Written: derived/dynamics/kc_max")

    # irr_data (as JSON strings)
    irr_data = data.get("irr_data", {})
    if "irr_data" in dyn_group:
        arr = dyn_group["irr_data"]
    else:
        arr = dyn_group.create_array(
            "irr_data",
            shape=(n_fields,),
            dtype=VariableLengthUTF8(),
        )
    # Build list of values then assign at once
    values = ["{}"] * n_fields
    for fid in field_order:
        if fid in container._uid_to_index:
            idx = container._uid_to_index[fid]
            field_irr = irr_data.get(fid, {})
            values[idx] = json.dumps(field_irr)
    arr[:] = values
    print("  Written: derived/dynamics/irr_data")

    # gwsub_data (as JSON strings)
    gwsub_data = data.get("gwsub_data", {})
    if "gwsub_data" in dyn_group:
        arr = dyn_group["gwsub_data"]
    else:
        arr = dyn_group.create_array(
            "gwsub_data",
            shape=(n_fields,),
            dtype=VariableLengthUTF8(),
        )
    # Build list of values then assign at once
    values = ["{}"] * n_fields
    for fid in field_order:
        if fid in container._uid_to_index:
            idx = container._uid_to_index[fid]
            field_gw = gwsub_data.get(fid, {})
            values[idx] = json.dumps(field_gw)
    arr[:] = values
    print("  Written: derived/dynamics/gwsub_data")


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy prepped_input.json to SwimContainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--json", "-j", required=True, type=Path, help="Path to prepped_input.json")
    parser.add_argument(
        "--shapefile",
        "-s",
        required=True,
        type=Path,
        help="Path to shapefile with field geometries",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output path for .swim container"
    )
    parser.add_argument(
        "--uid-column",
        "-u",
        default="FID",
        help="Column name for field UIDs in shapefile (default: FID)",
    )
    parser.add_argument(
        "--spinup", type=Path, default=None, help="Path to spinup.json for initial state"
    )
    parser.add_argument(
        "--calibrated-params", type=Path, default=None, help="Path to calibrated_params.json"
    )
    parser.add_argument(
        "--met-source",
        default="gridmet",
        choices=["gridmet", "era5"],
        help="Meteorology source name (default: gridmet)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing container")

    args = parser.parse_args()

    # Validate paths
    if not args.json.exists():
        print(f"ERROR: JSON file not found: {args.json}")
        sys.exit(1)
    if not args.shapefile.exists():
        print(f"ERROR: Shapefile not found: {args.shapefile}")
        sys.exit(1)

    convert_to_container(
        json_path=args.json,
        shapefile_path=args.shapefile,
        output_path=args.output,
        uid_column=args.uid_column,
        spinup_path=args.spinup,
        calibrated_params_path=args.calibrated_params,
        met_source=args.met_source,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

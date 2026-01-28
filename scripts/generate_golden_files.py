#!/usr/bin/env python3
"""
Generate golden reference files for SwimContainer regression tests.

Run this script ONCE with the current working code to generate reference outputs.
These outputs become the "ground truth" for future regression tests.

Usage:
    # Generate S2 single-station golden files
    python scripts/generate_golden_files.py s2 \
        --shapefile tests/fixtures/S2/data/gis/flux_footprint_s2.shp \
        --uid-column FID \
        --ndvi-dir /path/to/ndvi/csvs \
        --etf-dir /path/to/etf/csvs \
        --met-dir /path/to/met/parquets \
        --properties-json /path/to/properties.json \
        --start-date 2020-01-01 \
        --end-date 2022-12-31 \
        --output-dir tests/fixtures/S2/golden

    # Generate multi-station golden files
    python scripts/generate_golden_files.py multi \
        --shapefile /path/to/multi_station.shp \
        --uid-column site_id \
        --ndvi-dir /path/to/ndvi/csvs \
        --etf-dir /path/to/etf/csvs \
        --met-dir /path/to/met/parquets \
        --properties-json /path/to/properties.json \
        --start-date 2020-01-01 \
        --end-date 2022-12-31 \
        --output-dir tests/fixtures/multi_station/golden
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy and datetime types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def generate_golden_files(
    shapefile: Path,
    uid_column: str,
    ndvi_dir: Path,
    etf_dir: Path,
    met_dir: Path,
    properties_dir: Optional[Path],
    start_date: str,
    end_date: str,
    output_dir: Path,
    etf_model: str = "ssebop",
    instrument: str = "landsat",
    mask: str = "irr",
) -> Dict[str, Path]:
    """
    Generate golden files for a container configuration.

    Args:
        shapefile: Path to fields shapefile
        uid_column: Column name for field UIDs
        ndvi_dir: Directory containing NDVI CSV exports
        etf_dir: Directory containing ETf CSV exports
        met_dir: Directory containing meteorology parquet files
        properties_dir: Directory containing property CSV files (lulc.csv, ssurgo.csv, irr.csv)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save golden files
        etf_model: ET model name (default: ssebop)
        instrument: Instrument name (default: landsat)
        mask: Mask type (default: irr)

    Returns:
        Dict mapping output names to file paths
    """
    from swimrs.container import SwimContainer
    import tempfile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating golden files for {shapefile}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")

    # Create container in temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        container_path = Path(tmp_dir) / "golden_gen.swim"

        logger.info("Creating container...")
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(shapefile),
            uid_column=uid_column,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"Container created with {container.n_fields} field(s)")
        logger.info(f"Field UIDs: {container.field_uids}")

        # Ingest data - ingest both irr and inv_irr masks for mixed irrigated/non-irrigated stations
        masks_to_ingest = [mask, "inv_irr"] if mask == "irr" else [mask]

        for m in masks_to_ingest:
            logger.info(f"Ingesting NDVI data ({m} mask)...")
            container.ingest.ndvi(
                source_dir=str(ndvi_dir),
                instrument=instrument,
                mask=m,
            )

        for m in masks_to_ingest:
            logger.info(f"Ingesting ETf data ({m} mask)...")
            container.ingest.etf(
                source_dir=str(etf_dir),
                model=etf_model,
                instrument=instrument,
                mask=m,
            )

        logger.info("Ingesting meteorology data...")
        container.ingest.gridmet(source_dir=str(met_dir))

        # Ingest properties if available
        if properties_dir and properties_dir.exists():
            lulc_csv = properties_dir / "lulc.csv"
            ssurgo_csv = properties_dir / "ssurgo.csv"
            irr_csv = properties_dir / "irr.csv"

            if lulc_csv.exists() or ssurgo_csv.exists():
                logger.info("Ingesting properties...")
                container.ingest.properties(
                    lulc_csv=str(lulc_csv) if lulc_csv.exists() else None,
                    soils_csv=str(ssurgo_csv) if ssurgo_csv.exists() else None,
                    irrigation_csv=str(irr_csv) if irr_csv.exists() else None,
                    uid_column="site_id",
                    lulc_column="modis_lc",
                    extra_lulc_column="glc10_lc",
                )

        # Compute dynamics with all ingested masks
        logger.info("Computing dynamics...")
        container.compute.dynamics(
            etf_model=etf_model,
            masks=tuple(masks_to_ingest),
            instruments=(instrument,),
            use_lulc=True,
        )

        # Extract golden outputs
        golden_outputs = {}

        # 1. ke_max values
        ke_path = "derived/dynamics/ke_max"
        if ke_path in container._state.root:
            ke_values = container._state.root[ke_path][:].tolist()
            ke_data = {
                uid: val for uid, val in zip(container.field_uids, ke_values)
            }
            golden_outputs["ke_max"] = ke_data
            logger.info(f"Extracted ke_max: {ke_data}")

        # 2. kc_max values
        kc_path = "derived/dynamics/kc_max"
        if kc_path in container._state.root:
            kc_values = container._state.root[kc_path][:].tolist()
            kc_data = {
                uid: val for uid, val in zip(container.field_uids, kc_values)
            }
            golden_outputs["kc_max"] = kc_data
            logger.info(f"Extracted kc_max: {kc_data}")

        # 3. Irrigation data (JSON-encoded strings)
        irr_path = "derived/dynamics/irr_data"
        if irr_path in container._state.root:
            irr_arr = container._state.root[irr_path]
            irr_data = {}
            for i, uid in enumerate(container.field_uids):
                val = irr_arr[i]
                if val:
                    irr_data[uid] = json.loads(val)
                else:
                    irr_data[uid] = None
            golden_outputs["irr_data"] = irr_data
            logger.info(f"Extracted irr_data for {len(irr_data)} fields")

        # 4. Groundwater subsidy data
        gwsub_path = "derived/dynamics/gwsub_data"
        if gwsub_path in container._state.root:
            gwsub_arr = container._state.root[gwsub_path]
            gwsub_data = {}
            for i, uid in enumerate(container.field_uids):
                val = gwsub_arr[i]
                if val:
                    gwsub_data[uid] = json.loads(val)
                else:
                    gwsub_data[uid] = None
            golden_outputs["gwsub_data"] = gwsub_data
            logger.info(f"Extracted gwsub_data for {len(gwsub_data)} fields")

        # 5. Export prepped_input.json
        prepped_path = Path(tmp_dir) / "prepped_input.json"
        logger.info("Exporting prepped_input.json...")
        container.export.prepped_input_json(
            output_path=str(prepped_path),
            etf_model=etf_model,
            masks=tuple(masks_to_ingest),
        )

        # Read and parse the prepped input
        with open(prepped_path, 'r') as f:
            # It's a JSONL file, so read first line as sample
            first_line = f.readline()
            if first_line:
                prepped_sample = json.loads(first_line)
                # Store just the structure and a subset for testing
                prepped_summary = {
                    "field_count": len(container.field_uids),
                    "fields": container.field_uids,
                    "first_field_keys": list(prepped_sample.keys()) if prepped_sample else [],
                }
                golden_outputs["prepped_input_summary"] = prepped_summary

        # Save full prepped input (copy the file)
        import shutil
        shutil.copy(prepped_path, output_dir / "prepped_input.json")

        # 6. Generate spinup by running the model
        logger.info("Generating spinup state by running model...")
        try:
            from swimrs.swim.config import ProjectConfig
            from swimrs.swim.sampleplots import SamplePlots
            from swimrs.model.obs_field_cycle import field_day_loop

            # Create a minimal config for running the model
            # We need to run with the prepped_input.json we just generated
            config = ProjectConfig()

            # Set minimal required attributes
            config.prepped_input = str(prepped_path)
            config.start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            config.end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            config.fields_shapefile = str(shapefile)
            config.feature_id_col = uid_column
            config.refet_type = "eto"
            config.irrigation_threshold = 0.3
            config.runoff_process = "cn"
            config.mode_forecast = False
            config.mode_calib = False

            # Initialize plots and run model
            plots = SamplePlots()
            plots.initialize_plot_data(config)
            output = field_day_loop(config, plots, debug_flag=False)

            # Extract final state for each field
            spinup_data = {}
            for field_id, field_df in output.items():
                spinup_data[field_id] = field_df.iloc[-1].to_dict()

            golden_outputs["spinup"] = spinup_data
            logger.info(f"Generated spinup for {len(spinup_data)} field(s)")

        except Exception as e:
            logger.warning(f"Failed to generate spinup: {e}")
            logger.warning("Spinup file will not be generated")

        container.close()

    # Save golden files
    saved_files = {}

    for name, data in golden_outputs.items():
        filepath = output_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        saved_files[name] = filepath
        logger.info(f"Saved {filepath}")

    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "shapefile": str(shapefile),
        "uid_column": uid_column,
        "start_date": start_date,
        "end_date": end_date,
        "etf_model": etf_model,
        "instrument": instrument,
        "mask": mask,
        "field_uids": container.field_uids if hasattr(container, 'field_uids') else [],
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files["metadata"] = metadata_path

    logger.info(f"Golden file generation complete. Files saved to {output_dir}")
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference files for SwimContainer regression tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments for both commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--shapefile", type=Path, required=True,
        help="Path to fields shapefile"
    )
    common_parser.add_argument(
        "--uid-column", type=str, required=True,
        help="Column name for field UIDs"
    )
    common_parser.add_argument(
        "--ndvi-dir", type=Path, required=True,
        help="Directory containing NDVI CSV exports"
    )
    common_parser.add_argument(
        "--etf-dir", type=Path, required=True,
        help="Directory containing ETf CSV exports"
    )
    common_parser.add_argument(
        "--met-dir", type=Path, required=True,
        help="Directory containing meteorology parquet files"
    )
    common_parser.add_argument(
        "--properties-dir", type=Path, default=None,
        help="Directory containing property CSV files (lulc.csv, ssurgo.csv, irr.csv)"
    )
    common_parser.add_argument(
        "--start-date", type=str, required=True,
        help="Start date (YYYY-MM-DD)"
    )
    common_parser.add_argument(
        "--end-date", type=str, required=True,
        help="End date (YYYY-MM-DD)"
    )
    common_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to save golden files"
    )
    common_parser.add_argument(
        "--etf-model", type=str, default="ssebop",
        help="ET model name (default: ssebop)"
    )
    common_parser.add_argument(
        "--instrument", type=str, default="landsat",
        help="Instrument name (default: landsat)"
    )
    common_parser.add_argument(
        "--mask", type=str, default="irr",
        help="Mask type (default: irr)"
    )

    # S2 single-station command
    s2_parser = subparsers.add_parser(
        "s2",
        parents=[common_parser],
        help="Generate golden files for S2 single-station test"
    )

    # Multi-station command
    multi_parser = subparsers.add_parser(
        "multi",
        parents=[common_parser],
        help="Generate golden files for multi-station test"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.shapefile.exists():
        logger.error(f"Shapefile not found: {args.shapefile}")
        sys.exit(1)
    if not args.ndvi_dir.exists():
        logger.error(f"NDVI directory not found: {args.ndvi_dir}")
        sys.exit(1)
    if not args.etf_dir.exists():
        logger.error(f"ETf directory not found: {args.etf_dir}")
        sys.exit(1)
    if not args.met_dir.exists():
        logger.error(f"Meteorology directory not found: {args.met_dir}")
        sys.exit(1)

    # Generate golden files
    try:
        generate_golden_files(
            shapefile=args.shapefile,
            uid_column=args.uid_column,
            ndvi_dir=args.ndvi_dir,
            etf_dir=args.etf_dir,
            met_dir=args.met_dir,
            properties_dir=args.properties_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            etf_model=args.etf_model,
            instrument=args.instrument,
            mask=args.mask,
        )
    except Exception as e:
        logger.exception(f"Failed to generate golden files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

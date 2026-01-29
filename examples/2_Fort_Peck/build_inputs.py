#!/usr/bin/env python
"""
Build Model Inputs via SwimContainer
=====================================

This script processes extracted Earth Engine and GridMET data into the
SWIM-RS container format.

The container approach provides:
- Provenance tracking for all operations
- Unified data storage in Zarr format
- Standardized ingestion and export workflows
- Data validation and completeness checking

Run this after extract_data.py and syncing from the bucket.

Requirements
------------
- Extracted data from extract_data.py in the data/ subdirectories
- GridMET parquet files in met_timeseries/gridmet/

Input Data Structure
--------------------
    data/
    ├── properties/
    │   ├── 2_Fort_Peck_landcover.csv
    │   ├── 2_Fort_Peck_irr.csv
    │   └── 2_Fort_Peck_ssurgo.csv
    ├── remote_sensing/landsat/extracts/
    │   ├── ptjpl_etf/{irr,inv_irr}/*.csv
    │   └── ndvi/{irr,inv_irr}/*.csv
    ├── snow/snodas/extracts/*.csv
    ├── met_timeseries/gridmet/*.parquet
    └── gis/
        ├── flux_fields.shp
        └── flux_fields_gfid.shp

Output
------
    data/2_Fort_Peck.swim/  (container directory)

Usage
-----
    python build_inputs.py                    # Build for US-FPe only
    python build_inputs.py --all-fields       # Build for all fields
    python build_inputs.py --fields US-FPe US-ARM  # Specific fields
    python build_inputs.py --rebuild          # Rebuild container from scratch
"""

import argparse
import os
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "../.."))
sys.path.insert(0, ROOT_DIR)

from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig

# Default field to process
DEFAULT_FIELDS = ["US-FPe"]


def load_config():
    """Load project configuration."""
    config_file = os.path.join(PROJECT_DIR, "2_Fort_Peck.toml")
    cfg = ProjectConfig()
    cfg.read_config(config_file, project_root_override=PROJECT_DIR)
    return cfg


def create_container(cfg, overwrite=False):
    """Create a new SwimContainer from the fields shapefile."""
    container_path = os.path.join(SCRIPT_DIR, f"{cfg.project_name}.swim")

    container = SwimContainer.create(
        container_path,
        fields_shapefile=cfg.fields_shapefile,
        uid_column=cfg.feature_id_col,
        start_date=cfg.start_dt,
        end_date=cfg.end_dt,
        project_name=cfg.project_name,
        overwrite=overwrite,
    )
    return container


def ingest_data(container, cfg, select_fields=None):
    """Ingest all extracted data into the container."""
    uid_col = cfg.feature_id_col

    # 1. Ingest properties (landcover, soils, irrigation)
    print("\n=== Ingesting Properties ===")
    props_dir = os.path.join(SCRIPT_DIR, "properties")
    lulc_csv = os.path.join(props_dir, f"{cfg.project_name}_landcover.csv")
    soils_csv = os.path.join(props_dir, f"{cfg.project_name}_ssurgo.csv")
    irr_csv = os.path.join(props_dir, f"{cfg.project_name}_irr.csv")

    container.ingest.properties(
        lulc_csv=lulc_csv if os.path.exists(lulc_csv) else None,
        soils_csv=soils_csv if os.path.exists(soils_csv) else None,
        irr_csv=irr_csv if os.path.exists(irr_csv) else None,
        uid_column=uid_col,
        overwrite=True,
    )

    # 2. Ingest GridMET meteorology
    print("\n=== Ingesting GridMET ===")
    met_dir = cfg.met_dir
    grid_shp = cfg.gridmet_mapping_shp

    if os.path.exists(met_dir) and os.path.exists(grid_shp):
        container.ingest.gridmet(
            source_dir=met_dir,
            grid_shapefile=grid_shp,
            uid_column=uid_col,
            grid_column="GFID",
            include_corrected=True,
            overwrite=True,
        )
    else:
        print(f"  Warning: GridMET data not found at {met_dir}")

    # 3. Ingest SNODAS SWE
    # Note: SNODAS data starts in 2004, earlier dates will be filled with zeros
    print("\n=== Ingesting SNODAS ===")
    snodas_dir = os.path.join(SCRIPT_DIR, "snow", "snodas", "extracts")

    if os.path.exists(snodas_dir):
        container.ingest.snodas(
            source_dir=snodas_dir,
            uid_column=uid_col,
            fields=select_fields,
            overwrite=True,
        )
    else:
        print(f"  Warning: SNODAS data not found at {snodas_dir}")

    # 4. Ingest ETf data (PT-JPL)
    print("\n=== Ingesting ETf (PT-JPL) ===")
    etf_base = os.path.join(SCRIPT_DIR, "remote_sensing", "landsat", "extracts", "ptjpl_etf")

    for mask in ["irr", "inv_irr"]:
        mask_dir = os.path.join(etf_base, mask)
        if os.path.exists(mask_dir):
            container.ingest.etf(
                source_dir=mask_dir,
                uid_column=uid_col,
                model="ptjpl",
                mask=mask,
                instrument="landsat",
                fields=select_fields,
                overwrite=True,
            )

    # 5. Ingest NDVI data
    # Note: Single-field EE exports may use the field ID as the first column header
    # (e.g., 'US-FPe' instead of 'site_id'). The ingestor auto-detects and converts
    # these to standard format if the column header matches a known field ID.
    print("\n=== Ingesting NDVI ===")
    ndvi_base = os.path.join(SCRIPT_DIR, "remote_sensing", "landsat", "extracts", "ndvi")

    for mask in ["irr", "inv_irr"]:
        mask_dir = os.path.join(ndvi_base, mask)
        if os.path.exists(mask_dir):
            container.ingest.ndvi(
                source_dir=mask_dir,
                uid_column=uid_col,
                instrument="landsat",
                mask=mask,
                fields=select_fields,
                overwrite=True,
            )


def compute_dynamics(container, select_fields=None):
    """Compute merged NDVI and field dynamics (ke_max, kc_max, irrigation)."""

    # 1. Merge NDVI (Landsat only for this example)
    print("\n=== Computing Merged NDVI ===")
    container.compute.merged_ndvi(
        masks=("irr", "inv_irr"),
        instruments=("landsat",),
        overwrite=True,
    )

    # 2. Compute dynamics (ke_max, kc_max, irrigation detection)
    print("\n=== Computing Dynamics ===")
    container.compute.dynamics(
        etf_model="ptjpl",
        masks=("irr", "inv_irr"),
        instruments=("landsat",),
        use_mask=True,  # Use irrigation mask for CONUS
        use_lulc=False,
        fields=select_fields,
        overwrite=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build SWIM-RS model inputs using SwimContainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all-fields", action="store_true", help="Process all fields (default: US-FPe only)"
    )
    parser.add_argument("--fields", nargs="+", default=None, help="Specific field IDs to process")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild container from scratch (overwrites existing)",
    )
    args = parser.parse_args()

    # Determine fields to process
    if args.fields:
        select_fields = args.fields
    elif args.all_fields:
        select_fields = None
    else:
        select_fields = DEFAULT_FIELDS

    cfg = load_config()

    print(f"\nProject: {cfg.project_name}")
    print(f"Fields: {select_fields if select_fields else 'ALL'}")

    # Check if container exists
    container_path = os.path.join(SCRIPT_DIR, f"{cfg.project_name}.swim")
    container_exists = os.path.exists(container_path)

    if args.rebuild or not container_exists:
        print(f"\nCreating new container at {container_path}")
        container = create_container(cfg, overwrite=True)
    else:
        print(f"\nOpening existing container at {container_path}")
        container = SwimContainer.open(container_path, mode="r+")

    try:
        # Ingest all data
        ingest_data(container, cfg, select_fields)

        # Compute derived products (merged NDVI, dynamics)
        compute_dynamics(container, select_fields)

        # Save container
        container.save()

        print("\nDone! Built inputs for fields:")
        for fid in select_fields or container.field_uids:
            print(f"  - {fid}")

    finally:
        container.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Build Model Inputs via SwimContainer
=====================================

This script processes extracted Earth Engine and GridMET data into the
SWIM-RS container format, then exports to prepped_input.json.

The container approach provides:
- Provenance tracking for all operations
- Unified data storage in Zarr format
- Standardized ingestion and export workflows
- Data validation and completeness checking

Run this after extract_data.py and syncing from the bucket.

Requirements
------------
- Extracted data from extract_data.py in the data/ subdirectories
- GridMET parquet files in met/

Input Data Structure
--------------------
    data/
    ├── properties/
    │   ├── 3_Crane_landcover.csv
    │   ├── 3_Crane_irr.csv
    │   └── 3_Crane_ssurgo.csv
    ├── remote_sensing/landsat/extracts/
    │   ├── ptjpl_etf/{irr,inv_irr}/*.csv
    │   ├── sims_etf/{irr,inv_irr}/*.csv
    │   ├── ssebop_etf/{irr,inv_irr}/*.csv
    │   ├── geesebal_etf/{irr,inv_irr}/*.csv
    │   └── ndvi/{irr,inv_irr}/*.csv
    ├── snow/snodas/extracts/*.csv
    ├── met/*.parquet
    └── gis/
        ├── flux_fields.shp
        └── flux_fields_gfid.shp

Output
------
    data/3_Crane.swim/  (container directory)
    data/prepped_input.json

Usage
-----
    python build_inputs.py                    # Build for S2 only
    python build_inputs.py --all-fields       # Build for all fields
    python build_inputs.py --fields S2 S3     # Specific fields
    python build_inputs.py --rebuild          # Rebuild container from scratch
"""

import argparse
import os
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '../..'))
sys.path.insert(0, ROOT_DIR)

from swimrs.swim.config import ProjectConfig
from swimrs.container import SwimContainer

# Default field to process
DEFAULT_FIELDS = ['S2']

# OpenET ensemble models to ingest
ENSEMBLE_MODELS = ['ptjpl', 'sims', 'ssebop', 'geesebal']


def load_config():
    """Load project configuration."""
    config_file = os.path.join(PROJECT_DIR, '3_Crane.toml')
    cfg = ProjectConfig()
    cfg.read_config(config_file, project_root_override=PROJECT_DIR)
    return cfg


def create_container(cfg, overwrite=False):
    """Create a new SwimContainer from the fields shapefile."""
    container_path = os.path.join(SCRIPT_DIR, f'{cfg.project_name}.swim')

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
    props_dir = os.path.join(SCRIPT_DIR, 'properties')
    lulc_csv = os.path.join(props_dir, f'{cfg.project_name}_landcover.csv')
    soils_csv = os.path.join(props_dir, f'{cfg.project_name}_ssurgo.csv')
    irr_csv = os.path.join(props_dir, f'{cfg.project_name}_irr.csv')

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
            grid_column='GFID',
            include_corrected=True,
            overwrite=True,
        )
    else:
        print(f"  Warning: GridMET data not found at {met_dir}")

    # 3. Ingest SNODAS SWE
    print("\n=== Ingesting SNODAS ===")
    snodas_dir = os.path.join(SCRIPT_DIR, 'snow', 'snodas', 'extracts')

    if os.path.exists(snodas_dir):
        container.ingest.snodas(
            source_dir=snodas_dir,
            uid_column=uid_col,
            fields=select_fields,
            overwrite=True,
        )
    else:
        print(f"  Warning: SNODAS data not found at {snodas_dir}")

    # 4. Ingest ETf data from all OpenET ensemble models
    print("\n=== Ingesting ETf (OpenET Ensemble) ===")
    for model in ENSEMBLE_MODELS:
        etf_base = os.path.join(SCRIPT_DIR, 'remote_sensing', 'landsat', 'extracts', f'{model}_etf')
        if not os.path.exists(etf_base):
            print(f"  Skipping {model} - directory not found: {etf_base}")
            continue

        for mask in ['irr', 'inv_irr']:
            mask_dir = os.path.join(etf_base, mask)
            if os.path.exists(mask_dir):
                print(f"  Ingesting {model} ETf ({mask})...")
                container.ingest.etf(
                    source_dir=mask_dir,
                    uid_column=uid_col,
                    model=model,
                    mask=mask,
                    instrument='landsat',
                    fields=select_fields,
                    overwrite=True,
                )

    # 5. Ingest NDVI data
    # Note: Single-field EE exports may use the field ID as the first column header
    # (e.g., 'S2' instead of 'site_id'). The ingestor auto-detects and converts
    # these to standard format if the column header matches a known field ID.
    print("\n=== Ingesting NDVI ===")
    ndvi_base = os.path.join(SCRIPT_DIR, 'remote_sensing', 'landsat', 'extracts', 'ndvi')

    for mask in ['irr', 'inv_irr']:
        mask_dir = os.path.join(ndvi_base, mask)
        if os.path.exists(mask_dir):
            container.ingest.ndvi(
                source_dir=mask_dir,
                uid_column=uid_col,
                instrument='landsat',
                mask=mask,
                fields=select_fields,
                overwrite=True,
            )


def compute_dynamics(container, select_fields=None):
    """Compute merged NDVI and field dynamics (ke_max, kc_max, irrigation)."""

    # 1. Merge NDVI (Landsat only for this example)
    print("\n=== Computing Merged NDVI ===")
    container.compute.merged_ndvi(
        masks=('irr', 'inv_irr'),
        instruments=('landsat',),
        overwrite=True,
    )

    # 2. Compute dynamics (ke_max, kc_max, irrigation detection)
    print("\n=== Computing Dynamics ===")
    container.compute.dynamics(
        etf_model='ssebop',
        masks=('irr', 'inv_irr'),
        instruments=('landsat',),
        use_mask=True,  # Use irrigation mask for CONUS
        use_lulc=False,
        fields=select_fields,
        overwrite=True,
    )


def export_prepped_json(container, output_path, select_fields=None):
    """Export container data to prepped_input.json format."""
    print(f"\n=== Exporting to {output_path} ===")

    container.export.prepped_input_json(
        output_path=output_path,
        etf_model='ssebop',
        masks=('irr', 'inv_irr'),
        met_source='gridmet',
        instrument='landsat',
        fields=select_fields,
        use_merged_ndvi=False,  # Use Landsat-only NDVI
        irr_threshold=0.1,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Build SWIM-RS model inputs using SwimContainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--all-fields', action='store_true',
                        help='Process all fields (default: S2 only)')
    parser.add_argument('--fields', nargs='+', default=None,
                        help='Specific field IDs to process')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild container from scratch (overwrites existing)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: data/prepped_input.json)')
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
    container_path = os.path.join(SCRIPT_DIR, f'{cfg.project_name}.swim')
    container_exists = os.path.exists(container_path)

    if args.rebuild or not container_exists:
        print(f"\nCreating new container at {container_path}")
        container = create_container(cfg, overwrite=True)
    else:
        print(f"\nOpening existing container at {container_path}")
        container = SwimContainer.open(container_path, mode='r+')

    try:
        # Ingest all data
        ingest_data(container, cfg, select_fields)

        # Compute derived products (merged NDVI, dynamics)
        compute_dynamics(container, select_fields)

        # Export to prepped_input.json
        output_path = args.output or os.path.join(SCRIPT_DIR, 'prepped_input.json')
        export_prepped_json(container, output_path, select_fields)

        # Save container
        container.save()

        print(f"\nDone! Built inputs for fields:")
        for fid in (select_fields or container.field_uids):
            print(f"  - {fid}")

    finally:
        container.close()


if __name__ == '__main__':
    main()

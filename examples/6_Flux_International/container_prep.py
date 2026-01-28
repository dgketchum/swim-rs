"""
SwimContainer-based data preparation for Example 6 (Flux International).

This replaces the data_prep.py workflow with direct container ingestion.
All processed data is stored in the .swim zarr file instead of intermediate files.

Usage:
    python container_prep.py [--overwrite] [--sites SITE1,SITE2,...] [--skip-sentinel]
"""

import os
import warnings
from pathlib import Path

# Suppress zarr 3.x unstable spec warning for VariableLengthBytes (used for WKB geometry)
# The warning comes from zarr.core.dtype.npy.bytes
warnings.filterwarnings(
    "ignore",
    message="The data type.*VariableLengthBytes.*does not have a Zarr V3 specification",
)
# Also suppress the UnstableSpecificationWarning directly
try:
    from zarr.core.dtype.common import UnstableSpecificationWarning

    warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
except ImportError:
    pass

from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    """Load project configuration."""
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "6_Flux_International.toml"

    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def create_container(cfg: ProjectConfig, overwrite: bool = False) -> SwimContainer:
    """Create a new SwimContainer for the project."""
    container_path = cfg.container_path

    if os.path.exists(container_path):
        if overwrite:
            os.remove(container_path)
            print(f"Removed existing container: {container_path}")
        else:
            raise FileExistsError(
                f"Container already exists: {container_path}. Use overwrite=True to replace."
            )

    print(f"Creating container: {container_path}")
    print(f"  Fields: {cfg.fields_shapefile}")
    print(f"  UID column: {cfg.feature_id_col}")
    print(f"  Date range: {cfg.start_dt} to {cfg.end_dt}")

    container = SwimContainer.create(
        uri=container_path,
        fields_shapefile=cfg.fields_shapefile,
        uid_column=cfg.feature_id_col,
        start_date=str(cfg.start_dt.date()),
        end_date=str(cfg.end_dt.date()),
    )

    print(f"  Created with {container.n_fields} fields")
    return container


def ingest_ndvi(
    container: SwimContainer, cfg: ProjectConfig, sites: list = None, add_sentinel: bool = True
):
    """Ingest NDVI data from Landsat and Sentinel.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        sites: Optional list of site IDs to include
        add_sentinel: If True, also ingest Sentinel NDVI
    """
    # Landsat NDVI
    landsat_ndvi_dir = Path(cfg.landsat_dir) / "extracts" / "ndvi" / "no_mask"
    if landsat_ndvi_dir.exists():
        print(f"Ingesting Landsat NDVI from: {landsat_ndvi_dir}")
        container.ingest.ndvi(
            source_dir=str(landsat_ndvi_dir),
            uid_column=cfg.feature_id_col,
            instrument="landsat",
            mask="no_mask",
            fields=sites,
        )
    else:
        print(f"  WARNING: Landsat NDVI directory not found: {landsat_ndvi_dir}")

    # Sentinel NDVI
    if add_sentinel:
        sentinel_ndvi_dir = Path(cfg.sentinel_dir) / "extracts" / "ndvi" / "no_mask"
        if sentinel_ndvi_dir.exists():
            print(f"Ingesting Sentinel NDVI from: {sentinel_ndvi_dir}")
            container.ingest.ndvi(
                source_dir=str(sentinel_ndvi_dir),
                uid_column=cfg.feature_id_col,
                instrument="sentinel",
                mask="no_mask",
                fields=sites,
            )
        else:
            print(f"  WARNING: Sentinel NDVI directory not found: {sentinel_ndvi_dir}")


def ingest_etf(container: SwimContainer, cfg: ProjectConfig):
    """Ingest ETf data (PT-JPL model for international)."""
    etf_model = cfg.etf_target_model  # 'ptjpl'

    # Landsat ETf
    landsat_etf_dir = Path(cfg.landsat_dir) / "extracts" / f"{etf_model}_etf" / "no_mask"
    if landsat_etf_dir.exists():
        print(f"Ingesting Landsat {etf_model} ETf from: {landsat_etf_dir}")
        container.ingest.etf(
            source_dir=str(landsat_etf_dir),
            uid_column=cfg.feature_id_col,
            model=etf_model,
            instrument="landsat",
            mask="no_mask",
        )
    else:
        print(f"  WARNING: Landsat ETf directory not found: {landsat_etf_dir}")

    # ECOSTRESS: Data is daily ET (not ETf), would need ETo division - skipping for now


def ingest_meteorology(container: SwimContainer, cfg: ProjectConfig):
    """Ingest ERA5-Land meteorology data."""
    met_dir = Path(cfg.met_dir)

    if met_dir.exists():
        print(f"Ingesting ERA5-Land meteorology from: {met_dir}")
        container.ingest.era5(
            source_dir=str(met_dir),
            variables=cfg.era5_params or ["swe", "eto", "tmean", "tmin", "tmax", "prcp", "srad"],
        )
    else:
        print(f"  WARNING: ERA5 meteorology directory not found: {met_dir}")


def ingest_properties(container: SwimContainer, cfg: ProjectConfig):
    """Ingest static field properties (LULC, HWSD soils)."""
    print("Ingesting field properties...")

    lulc_csv = cfg.lulc_csv if cfg.lulc_csv and os.path.exists(cfg.lulc_csv) else None
    soils_csv = cfg.hwsd_csv if cfg.hwsd_csv and os.path.exists(cfg.hwsd_csv) else None

    if lulc_csv:
        print(f"  LULC: {lulc_csv}")
    if soils_csv:
        print(f"  Soils (HWSD): {soils_csv}")

    if lulc_csv or soils_csv:
        container.ingest.properties(
            lulc_csv=lulc_csv,
            soils_csv=soils_csv,
            irrigation_csv=None,  # No irrigation data for international
            uid_column=cfg.feature_id_col,
            lulc_column="modis_lc",
            extra_lulc_column="glc10_lc",
        )
    else:
        print("  WARNING: No property files found to ingest")


def compute_fused_ndvi(container: SwimContainer, cfg: ProjectConfig):
    """Compute fused NDVI by combining Landsat and Sentinel observations."""
    print("Computing fused NDVI...")

    container.compute.fused_ndvi(
        masks=("no_mask",),
        instrument1="landsat",
        instrument2="sentinel",
        min_pairs=20,
        window_days=5,
    )

    print("  Fused NDVI computation complete")


def compute_dynamics(container: SwimContainer, cfg: ProjectConfig):
    """Compute field dynamics (ke_max, kc_max, irrigation detection)."""
    print("Computing dynamics...")

    container.compute.dynamics(
        etf_model=cfg.etf_target_model,
        masks=("no_mask",),
        instruments=("landsat", "sentinel"),
        use_lulc=True,
        irr_threshold=cfg.irrigation_threshold,
        met_source=cfg.met_source,  # "era5" for international
    )

    print("  Dynamics computation complete")


def export_model_inputs(container: SwimContainer, cfg: ProjectConfig, output_path: str = None):
    """Export data in prepped_input.json format for model consumption."""
    print("Exporting model inputs...")

    if output_path is None:
        output_path = cfg.input_data

    container.export.prepped_input_json(
        output_path=output_path,
        etf_model=cfg.etf_target_model,
        masks=("no_mask",),
        met_source="era5",
        instrument="landsat",
        use_fused_ndvi=True,
    )

    print(f"  Exported to: {output_path}")


def run_full_pipeline(overwrite: bool = False, sites: list = None, add_sentinel: bool = True):
    """Run the complete container preparation pipeline.

    Args:
        overwrite: If True, overwrite existing container
        sites: Optional list of site IDs to process
        add_sentinel: If True, include Sentinel NDVI ingestion
    """
    cfg = _load_config()

    print("=" * 60)
    print(f"Container Preparation: {cfg.project_name}")
    print(f"  Met source: {cfg.met_source}")
    print(f"  Soil source: {cfg.soil_source}")
    print(f"  Mask mode: {cfg.mask_mode}")
    print("=" * 60)

    # Create container
    container = create_container(cfg, overwrite=overwrite)

    try:
        # Ingest data
        ingest_ndvi(container, cfg, sites=sites, add_sentinel=add_sentinel)
        ingest_etf(container, cfg)
        ingest_meteorology(container, cfg)
        ingest_properties(container, cfg)

        # Compute derived products
        compute_fused_ndvi(container, cfg)
        compute_dynamics(container, cfg)

        # Export model inputs
        export_model_inputs(container, cfg)

        # Save
        print("Saving container...")
        container.save()
        print(f"Container saved: {cfg.container_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Fields: {container.n_fields}")
        print(f"  Date range: {container.start_date} to {container.end_date}")
        print(f"  Days: {container.n_days}")
        print("=" * 60)

    finally:
        container.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SwimContainer for Example 6")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing container if it exists",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated site IDs to process (default: all)",
    )
    parser.add_argument(
        "--skip-sentinel",
        action="store_true",
        help="Skip Sentinel NDVI ingestion",
    )
    args = parser.parse_args()

    # Parse sites argument
    select_sites = None
    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]

    run_full_pipeline(
        overwrite=args.overwrite,
        sites=select_sites,
        add_sentinel=not args.skip_sentinel,
    )

"""
Container-based data preparation for Example 6 (Flux International).

The container workflow:
    1. Create container from shapefile
    2. Ingest meteorology (ERA5-Land)
    3. Ingest remote sensing (NDVI from Landsat/Sentinel, ETf from Landsat PT-JPL + ECOSTRESS)
    4. Ingest properties (HWSD soils, LULC)
    5. Compute fused NDVI (Landsat + Sentinel)
    6. Compute dynamics (ke_max, kc_max, irrigation detection)

Usage:
    python container_prep.py [--overwrite] [--sites SITE1,SITE2,...] [--skip-sentinel]
                             [--exclude-sites PR-xLA,...]

    # Or use functions directly:
    from container_prep import create_project_container, prep_all
    container = create_project_container(overwrite=True)
    prep_all(container)
"""

import os
from pathlib import Path

from swimrs.container import SwimContainer, create_container, open_container
from swimrs.swim.config import ProjectConfig

EXAMPLE_DIR = Path(__file__).resolve().parent
if not (EXAMPLE_DIR / "6_Flux_International.toml").exists():
    EXAMPLE_DIR = EXAMPLE_DIR.parent


def _load_config(config_path: str | None = None) -> ProjectConfig:
    """Load project configuration."""
    conf = Path(config_path) if config_path else EXAMPLE_DIR / "6_Flux_International.toml"

    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def create_project_container(cfg: ProjectConfig = None, overwrite: bool = False) -> SwimContainer:
    """Create a new SwimContainer for this project.

    Args:
        cfg: ProjectConfig instance (loaded if None)
        overwrite: If True, overwrite existing container

    Returns:
        SwimContainer instance
    """
    if cfg is None:
        cfg = _load_config()

    container_path = cfg.container_path or os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")

    if os.path.exists(container_path) and not overwrite:
        print(f"Opening existing container: {container_path}")
        return open_container(container_path, mode="r+")

    print(f"Creating new container: {container_path}")
    container = create_container(
        uri=container_path,
        fields_shapefile=cfg.fields_shapefile,
        uid_column=cfg.feature_id_col,
        start_date=cfg.start_dt,
        end_date=cfg.end_dt,
        project_name=cfg.project_name,
        overwrite=overwrite,
    )

    return container


def ingest_remote_sensing(
    container: SwimContainer,
    cfg: ProjectConfig,
    sites: list = None,
    add_sentinel: bool = True,
    etf_scale_factor: float = 1.0,
):
    """Ingest NDVI and ETf into the container.

    NDVI: Landsat + Sentinel (no_mask only for international sites).
    ETf: Landsat PT-JPL + ECOSTRESS PT-JPL (pre-converted from daily ET).
    """
    print("\n=== Ingesting Remote Sensing ===")

    etf_model = cfg.etf_target_model  # 'ptjpl'

    # Landsat NDVI
    landsat_ndvi_dir = os.path.join(cfg.landsat_dir, "extracts", "ndvi", "no_mask")
    if os.path.isdir(landsat_ndvi_dir):
        print("Ingesting Landsat NDVI (no_mask)...")
        container.ingest.ndvi(
            source_dir=landsat_ndvi_dir,
            uid_column=cfg.feature_id_col,
            instrument="landsat",
            mask="no_mask",
            fields=sites,
        )
    else:
        print(f"  WARNING: Landsat NDVI directory not found: {landsat_ndvi_dir}")

    # Sentinel NDVI
    if add_sentinel:
        sentinel_ndvi_dir = os.path.join(cfg.sentinel_dir, "extracts", "ndvi", "no_mask")
        if os.path.isdir(sentinel_ndvi_dir):
            print("Ingesting Sentinel NDVI (no_mask)...")
            container.ingest.ndvi(
                source_dir=sentinel_ndvi_dir,
                uid_column=cfg.feature_id_col,
                instrument="sentinel",
                mask="no_mask",
                fields=sites,
            )
        else:
            print(f"  WARNING: Sentinel NDVI directory not found: {sentinel_ndvi_dir}")

    # Landsat ETf
    landsat_etf_dir = os.path.join(cfg.landsat_dir, "extracts", f"{etf_model}_etf", "no_mask")
    if os.path.isdir(landsat_etf_dir):
        scale_msg = f" (scale_factor={etf_scale_factor})" if etf_scale_factor != 1.0 else ""
        print(f"Ingesting Landsat {etf_model} ETf (no_mask){scale_msg}...")
        container.ingest.etf(
            source_dir=landsat_etf_dir,
            uid_column=cfg.feature_id_col,
            model=etf_model,
            instrument="landsat",
            mask="no_mask",
            fields=sites,
            scale_factor=etf_scale_factor,
        )
    else:
        print(f"  WARNING: Landsat ETf directory not found: {landsat_etf_dir}")

    # ECOSTRESS ETf (converted from daily ET by ecostress_etf_convert.py)
    ecostress_etf_dir = os.path.join(cfg.ecostress_dir, "extracts", "etf", "no_mask")
    if os.path.isdir(ecostress_etf_dir):
        print(f"Ingesting ECOSTRESS {etf_model} ETf (no_mask)...")
        container.ingest.etf(
            source_dir=ecostress_etf_dir,
            uid_column=cfg.feature_id_col,
            model=etf_model,
            instrument="ecostress",
            mask="no_mask",
            fields=sites,
        )
    else:
        print(
            f"  NOTE: ECOSTRESS ETf not found at {ecostress_etf_dir}"
            " (run ecostress_etf_convert.py first)"
        )


def ingest_meteorology(container: SwimContainer, cfg: ProjectConfig):
    """Ingest ERA5-Land meteorology data."""
    print("\n=== Ingesting Meteorology (ERA5-Land) ===")

    met_dir = cfg.met_dir
    if os.path.isdir(met_dir):
        container.ingest.era5(
            source_dir=met_dir,
            variables=cfg.era5_params or ["swe", "eto", "tmean", "tmin", "tmax", "prcp", "srad"],
        )
    else:
        print(f"  WARNING: ERA5 meteorology directory not found: {met_dir}")


def ingest_properties(container: SwimContainer, cfg: ProjectConfig):
    """Ingest static field properties (HWSD soils, LULC)."""
    print("\n=== Ingesting Properties ===")

    container.ingest.properties(
        soils_csv=cfg.hwsd_csv,
        lulc_csv=cfg.lulc_csv,
        irr_csv=None,  # No irrigation data for international sites
        uid_column=cfg.feature_id_col,
        lulc_column="modis_lc",
        extra_lulc_column="glc10_lc",
        awc_units="mm/m",  # HWSD v2 AWC is mm/m; container stores m/m
    )


def compute_fused_ndvi(container: SwimContainer, overwrite: bool = False):
    """Compute fused NDVI by combining Landsat and Sentinel observations."""
    print("\n=== Computing Fused NDVI ===")

    container.compute.fused_ndvi(
        masks=("no_mask",),
        overwrite=overwrite,
    )


def compute_dynamics(container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False):
    """Compute field dynamics (ke_max, kc_max, irrigation detection)."""
    print("\n=== Computing Dynamics ===")

    container.compute.dynamics(
        etf_model=cfg.etf_target_model,
        masks=("no_mask",),
        instruments=("landsat", "sentinel"),
        use_lulc=True,
        irr_threshold=cfg.irrigation_threshold or 0.3,
        met_source=cfg.met_source,  # "era5" for international
        overwrite=overwrite,
    )


def prep_all(
    container: SwimContainer,
    cfg: ProjectConfig = None,
    sites: list = None,
    overwrite: bool = False,
    add_sentinel: bool = True,
    etf_scale_factor: float = 1.0,
):
    """Run the complete data preparation workflow.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance (loaded if None)
        sites: Optional list of site IDs to include
        overwrite: If True, replace existing data
        add_sentinel: If True, include Sentinel NDVI
    """
    if cfg is None:
        cfg = _load_config()

    # Step 1: Ingest meteorology
    ingest_meteorology(container, cfg)

    # Step 2: Ingest remote sensing (NDVI, ETf)
    ingest_remote_sensing(
        container,
        cfg,
        sites=sites,
        add_sentinel=add_sentinel,
        etf_scale_factor=etf_scale_factor,
    )

    # Step 3: Ingest properties
    ingest_properties(container, cfg)

    # Step 4: Compute fused NDVI
    compute_fused_ndvi(container, overwrite=overwrite)

    # Step 5: Compute dynamics
    compute_dynamics(container, cfg, overwrite=overwrite)

    # Step 6: Health check
    print("\n=== Running Health Check ===")
    health_dir = os.path.join(cfg.data_dir, "health")
    container.report(config=cfg, output_dir=health_dir)

    print("\n=== Container Preparation Complete ===")
    print(container.inventory)


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
        "--exclude-sites",
        type=str,
        default=None,
        help="Comma-separated site IDs to exclude",
    )
    parser.add_argument(
        "--skip-sentinel",
        action="store_true",
        help="Skip Sentinel NDVI ingestion",
    )
    parser.add_argument(
        "--etf-scale-factor",
        type=float,
        default=1.0,
        help="Multiply ETf values by this factor during ingestion (default: 1.0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config (default: 6_Flux_International.toml)",
    )
    args = parser.parse_args()

    select_sites = None
    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]

    # Load configuration
    config = _load_config(args.config)

    # Exclude sites if requested (filter shapefile fields before container creation)
    if args.exclude_sites:
        import geopandas as gpd

        exclude = [s.strip() for s in args.exclude_sites.split(",")]
        shp = config.fields_shapefile
        gdf = gpd.read_file(shp, engine="fiona")
        before = len(gdf)
        gdf = gdf[~gdf[config.feature_id_col].isin(exclude)]
        # Write filtered shapefile to a temp location for container creation
        filtered_shp = shp.replace(".shp", "_filtered.shp")
        gdf.to_file(filtered_shp, engine="fiona")
        config.fields_shapefile = filtered_shp
        print(f"Excluded {before - len(gdf)} sites: {exclude} ({len(gdf)} remaining)")

    # Create or open container
    container = create_project_container(config, overwrite=args.overwrite)

    # Run full preparation workflow
    prep_all(
        container,
        config,
        sites=select_sites,
        overwrite=args.overwrite,
        add_sentinel=not args.skip_sentinel,
        etf_scale_factor=args.etf_scale_factor,
    )

    container.close()

    print(f"\nContainer saved to: {container.path}")

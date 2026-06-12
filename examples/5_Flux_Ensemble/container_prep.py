"""
Container-based data preparation workflow for 5_Flux_Ensemble.

This module replicates the functionality of data_prep.py but uses the
SwimContainer approach instead of the multi-file Parquet/JSON approach.

The container workflow:
    1. Create container from shapefile
    2. Ingest meteorology (GridMET)
    3. Ingest remote sensing (NDVI, ETf from Landsat/Sentinel)
    4. Ingest snow (SNODAS)
    5. Ingest properties (soils, LULC, irrigation)
    6. Compute fused NDVI (Landsat + Sentinel)
    7. Compute dynamics (irrigation, groundwater, ke_max, kc_max)

Usage:
    python container_prep.py [--overwrite] [--sites SITE1,SITE2,...] [--skip-sentinel]
                             [--openet-source {diy,ee}] [--getinfo]

    --getinfo ingests from the .getInfo() extraction outputs (data_extract.py
    default steps): NDVI from {landsat,sentinel}/getinfo/ndvi/{mask}, SNODAS
    from snow/snodas/getinfo, properties from properties/getinfo/. Bucket ETf
    ingest is skipped entirely — ETf enters the run container from the getInfo
    v2.1 CSVs via build_container.py:ingest_new_etf.

    # Or use functions directly:
    from container_prep import create_project_container, prep_all
    container = create_project_container(overwrite=True)
    prep_all(container)
"""

import os
import tempfile
from pathlib import Path

from swimrs.container import SwimContainer, create_container, open_container
from swimrs.swim.config import ProjectConfig
from swimrs.utils.flux_stations import create_master_shapefile, filter_by_classification

# Canonical source data (shipped with the repo)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOOTPRINTS_SHP = _REPO_ROOT / "examples" / "data" / "flux_footprints_3p_clean.shp"
_METADATA_CSV = _REPO_ROOT / "examples" / "data" / "station_metadata.csv"


def _load_config() -> ProjectConfig:
    """Load project configuration from TOML file."""
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def build_shapefile(
    cfg: ProjectConfig,
    classification: str = "Croplands",
    overwrite: bool = False,
    exclude_sites: list[str] | None = None,
):
    """Regenerate the flux fields shapefile from canonical repo data.

    Builds a master shapefile from shipped footprints and metadata, then
    filters to the requested land-cover classification.  The result is
    written to the GIS directory expected by the TOML config.

    Args:
        cfg: ProjectConfig instance.
        classification: IGBP class to keep (default ``"Croplands"``).
        overwrite: Replace an existing shapefile.
        exclude_sites: Site IDs to drop after filtering.
    """
    output_shp = cfg.fields_shapefile
    if os.path.exists(output_shp) and not overwrite:
        print(f"Shapefile already exists: {output_shp}")
        return

    os.makedirs(os.path.dirname(output_shp), exist_ok=True)
    print(f"\n=== Building {classification} shapefile ===")
    print(f"  Footprints: {_FOOTPRINTS_SHP}")
    print(f"  Metadata:   {_METADATA_CSV}")

    with tempfile.TemporaryDirectory() as tmpdir:
        master = os.path.join(tmpdir, "master.shp")
        create_master_shapefile(str(_FOOTPRINTS_SHP), str(_METADATA_CSV), master, overwrite=True)
        filtered = filter_by_classification(master, classification, output_shp, overwrite=overwrite)

    if exclude_sites:
        import geopandas as gpd

        gdf = gpd.read_file(output_shp, engine="fiona")
        before = len(gdf)
        gdf = gdf[~gdf[cfg.feature_id_col].isin(exclude_sites)]
        gdf.to_file(output_shp, engine="fiona")
        dropped = before - len(gdf)
        print(f"  Excluded {dropped} sites: {exclude_sites}")
        filtered = gdf

    print(f"  Created {len(filtered)} {classification} stations → {output_shp}")


def build_gridmet_mapping(cfg: ProjectConfig, overwrite: bool = False):
    """Create the GridMET mapping shapefile used by the canonical builder."""
    from swimrs.data_extraction.gridmet.gridmet import assign_gridmet_ids

    mapping_shp = cfg.gridmet_mapping_shp
    if os.path.exists(mapping_shp) and not overwrite:
        print(f"GridMET mapping already exists: {mapping_shp}")
        return

    if not os.path.exists(cfg.fields_shapefile):
        raise FileNotFoundError(
            f"Fields shapefile not found: {cfg.fields_shapefile}. Build it before GridMET mapping."
        )

    os.makedirs(os.path.dirname(mapping_shp), exist_ok=True)
    print("\n=== Building GridMET mapping shapefile ===")
    assign_gridmet_ids(
        fields=cfg.fields_shapefile,
        fields_join=mapping_shp,
        gridmet_points=cfg.gridmet_centroids,
        feature_id=cfg.feature_id_col,
        gridmet_id_col=cfg.gridmet_id_col,
    )
    print(f"  Created mapping → {mapping_shp}")


def create_project_container(
    cfg: ProjectConfig = None,
    overwrite: bool = False,
    exclude_sites: list[str] | None = None,
) -> SwimContainer:
    """
    Create a new SwimContainer for this project.

    Args:
        cfg: ProjectConfig instance (loaded if None)
        overwrite: If True, overwrite existing container
        exclude_sites: Optional site IDs to drop when rebuilding the canonical
            cropland shapefile for a fresh container.

    Returns:
        SwimContainer instance
    """
    if cfg is None:
        cfg = _load_config()

    container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")

    if os.path.exists(container_path) and not overwrite:
        print(f"Opening existing container: {container_path}")
        return open_container(container_path, mode="r+")

    # Ensure the canonical cropland shapefile exists before container creation.
    build_shapefile(cfg, overwrite=overwrite, exclude_sites=exclude_sites)

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


def ingest_meteorology(container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False):
    """
    Ingest GridMET meteorology data into the container.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        overwrite: If True, replace existing data
    """
    print("\n=== Ingesting Meteorology (GridMET) ===")

    # Check if already ingested
    if "meteorology/gridmet/eto" in container._root and not overwrite:
        print("GridMET data already ingested, skipping")
        return

    # Ingest GridMET with all available variables
    # GridMET parquet files are named by grid cell ID (GFID), not by station UID,
    # so we must provide the mapping shapefile that links site_id → GFID.
    container.ingest.gridmet(
        source_dir=cfg.met_dir,
        grid_shapefile=cfg.gridmet_mapping_shp,
        uid_column=cfg.feature_id_col,
        grid_column="GFID",
        variables=[
            "eto",
            "etr",
            "prcp",
            "tmin",
            "tmax",
            "srad",
            "u2",
            "ea",
        ],
        overwrite=overwrite,
    )


def _resolve_etf_dir(landsat_dir: str, model: str, mask: str, openet_source: str) -> str:
    """Build the ETf directory path for the given source.

    Args:
        landsat_dir: Root Landsat data directory.
        model: ETf model name (e.g. ``"ptjpl"``, ``"ssebop"``).
        mask: Mask name (``"no_mask"`` in Example 5).
        openet_source: ``"diy"`` for image-level extracts, ``"ee"`` for
            pre-computed OpenET EE asset extracts.

    Returns:
        Absolute path to the ETf CSV directory.
    """
    if openet_source == "ee":
        return os.path.join(landsat_dir, "extracts", "openet", f"{model}_etf", mask)
    return os.path.join(landsat_dir, "extracts", f"{model}_etf", mask)


def ingest_remote_sensing(
    container: SwimContainer,
    cfg: ProjectConfig,
    sites: list = None,
    overwrite: bool = False,
    add_sentinel: bool = True,
    openet_source: str = "diy",
    getinfo: bool = False,
):
    """
    Ingest remote sensing data (NDVI, ETf) into the container.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        sites: Optional list of site IDs to include
        overwrite: If True, replace existing data
        add_sentinel: If True, also ingest Sentinel NDVI
        openet_source: ``"diy"`` (default) or ``"ee"`` for OpenET EE assets
        getinfo: If True, ingest NDVI from the getInfo extraction dirs
            (``getinfo/ndvi/{mask}``) and skip bucket ETf ingest entirely
            (ETf comes from the getInfo v2.1 CSVs via build_container.py)
    """
    print("\n=== Ingesting Remote Sensing ===")

    # Example 5 is a no_mask experiment: IrrMapper/LANID enters only as the
    # per-year irrigation fraction table used for status enforcement in dynamics.
    masks = ["no_mask"]
    models = list(dict.fromkeys([cfg.etf_target_model] + (cfg.etf_ensemble_members or [])))
    models = [m for m in models if m != "ensemble"]
    if cfg.ensemble_source == "openet" and "ensemble" not in models:
        models.append("ensemble")
    n_workers = cfg.workers or 1

    ndvi_subdir = ("getinfo", "ndvi") if getinfo else ("extracts", "ndvi")

    # Ingest Landsat NDVI
    for mask in masks:
        ndvi_dir = os.path.join(cfg.landsat_dir, *ndvi_subdir, mask)
        if os.path.isdir(ndvi_dir):
            print(f"Ingesting Landsat NDVI ({mask})...")
            container.ingest.ndvi(
                source_dir=ndvi_dir,
                uid_column=cfg.feature_id_col,
                instrument="landsat",
                mask=mask,
                fields=sites,
                overwrite=overwrite,
                workers=n_workers,
            )

    # Ingest Sentinel NDVI
    if add_sentinel:
        for mask in masks:
            sentinel_dir = getattr(cfg, "sentinel_dir", None)
            if sentinel_dir is None:
                # Derive sentinel dir from landsat dir
                sentinel_dir = cfg.landsat_dir.replace("landsat", "sentinel")
            ndvi_dir = os.path.join(sentinel_dir, *ndvi_subdir, mask)
            if os.path.isdir(ndvi_dir):
                print(f"Ingesting Sentinel NDVI ({mask})...")
                container.ingest.ndvi(
                    source_dir=ndvi_dir,
                    uid_column=cfg.feature_id_col,
                    instrument="sentinel",
                    mask=mask,
                    fields=sites,
                    overwrite=overwrite,
                    workers=n_workers,
                )

    # Ingest ETf for each model (no_mask, per Example 5 policy)
    if getinfo:
        print(
            "Skipping bucket ETf ingest (--getinfo): ETf comes from the getInfo "
            "v2.1 CSVs via build_container.py:ingest_new_etf"
        )
        return
    for model in models:
        for mask in masks:
            etf_dir = _resolve_etf_dir(cfg.landsat_dir, model, mask, openet_source)
            if os.path.isdir(etf_dir):
                print(f"Ingesting ETf ({model}, {mask})...")
                container.ingest.etf(
                    source_dir=etf_dir,
                    uid_column=cfg.feature_id_col,
                    instrument="landsat",
                    model=model,
                    mask=mask,
                    fields=sites,
                    overwrite=overwrite,
                    workers=n_workers,
                )


def ingest_snow(
    container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False, getinfo: bool = False
):
    """
    Ingest SNODAS snow data into the container.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        overwrite: If True, replace existing data
        getinfo: If True, ingest from the getInfo extraction dir
            (``snow/snodas/getinfo``) instead of ``cfg.snodas_in_dir``
    """
    print("\n=== Ingesting Snow (SNODAS) ===")

    if "snow/snodas/swe" in container._root and not overwrite:
        print("SNODAS data already ingested, skipping")
        return

    snodas_dir = (
        os.path.join(cfg.data_dir, "snow", "snodas", "getinfo") if getinfo else cfg.snodas_in_dir
    )

    # Ingest directly from extracts directory
    if snodas_dir and os.path.isdir(snodas_dir):
        container.ingest.snodas(
            source_dir=snodas_dir,
            uid_column=cfg.feature_id_col,
            overwrite=overwrite,
        )
    else:
        print("Warning: No SNODAS extracts found, skipping")


def ingest_properties(
    container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False, getinfo: bool = False
):
    """
    Ingest field properties (soils, LULC, irrigation) into the container.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        overwrite: If True, replace existing data
        getinfo: If True, read the property CSVs from ``properties/getinfo/``
            (same basenames as the configured paths)
    """
    print("\n=== Ingesting Properties ===")

    soils_csv, lulc_csv, irr_csv = cfg.ssurgo_csv, cfg.lulc_csv, cfg.irr_csv
    if getinfo:
        prop_dir = os.path.join(cfg.data_dir, "properties", "getinfo")
        soils_csv = os.path.join(prop_dir, os.path.basename(soils_csv))
        lulc_csv = os.path.join(prop_dir, os.path.basename(lulc_csv))
        irr_csv = os.path.join(prop_dir, os.path.basename(irr_csv))

    # Build properties from individual sources
    container.ingest.properties(
        soils_csv=soils_csv,
        lulc_csv=lulc_csv,
        irr_csv=irr_csv,
        uid_column=cfg.feature_id_col,
        overwrite=overwrite,
    )


def compute_fused_ndvi(container: SwimContainer, overwrite: bool = False):
    """
    Compute fused NDVI from Landsat and Sentinel observations.

    Uses quantile mapping to adjust Sentinel NDVI to match Landsat,
    then combines both sources.

    Args:
        container: SwimContainer instance
        overwrite: If True, replace existing fused NDVI
    """
    print("\n=== Computing Fused NDVI ===")

    container.compute.fused_ndvi(
        masks=("no_mask",),
        overwrite=overwrite,
    )


def compute_dynamics(container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False):
    """
    Compute irrigation, groundwater subsidy, and K parameters.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        overwrite: If True, replace existing dynamics
    """
    print("\n=== Computing Dynamics ===")

    container.compute.dynamics(
        etf_model=cfg.etf_target_model,
        masks=("no_mask",),
        irr_threshold=cfg.irrigation_threshold or 0.3,
        use_mask=True,
        use_lulc=False,
        lookback=5,
        overwrite=overwrite,
    )


def prep_all(
    container: SwimContainer,
    cfg: ProjectConfig = None,
    sites: list = None,
    overwrite: bool = False,
    add_sentinel: bool = True,
    openet_source: str = "diy",
    getinfo: bool = False,
):
    """
    Run the complete data preparation workflow.

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance (loaded if None)
        sites: Optional list of site IDs to include
        overwrite: If True, replace existing data
        add_sentinel: If True, include Sentinel NDVI
        openet_source: ``"diy"`` (default) or ``"ee"`` for OpenET EE assets
        getinfo: If True, ingest NDVI/SNODAS/properties from the getInfo
            extraction outputs and skip bucket ETf ingest
    """
    if cfg is None:
        cfg = _load_config()

    # Step 1: GridMET site-to-cell mapping. Never regenerated under --overwrite:
    # GFIDs must stay consistent with the parquet met store already on disk
    # (regenerating from a different centroids file re-keys GFIDs and breaks the
    # met join). Delete the mapping shapefile manually to force a rebuild.
    build_gridmet_mapping(cfg, overwrite=False)

    # Step 2: Ingest meteorology
    ingest_meteorology(container, cfg, overwrite=overwrite)

    # Step 3: Ingest remote sensing (NDVI, ETf)
    ingest_remote_sensing(
        container,
        cfg,
        sites=sites,
        overwrite=overwrite,
        add_sentinel=add_sentinel,
        openet_source=openet_source,
        getinfo=getinfo,
    )

    # Step 4: Ingest snow
    ingest_snow(container, cfg, overwrite=overwrite, getinfo=getinfo)

    # Step 5: Ingest properties
    ingest_properties(container, cfg, overwrite=overwrite, getinfo=getinfo)

    # Step 6: Compute fused NDVI
    compute_fused_ndvi(container, overwrite=overwrite)

    # Step 7: Compute dynamics
    compute_dynamics(container, cfg, overwrite=overwrite)

    print("\n=== Container Preparation Complete ===")
    print(container.inventory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Container-based data preparation for 5_Flux_Ensemble"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing container",
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
    parser.add_argument(
        "--openet-source",
        choices=["diy", "ee"],
        default="diy",
        help="ETf source: 'diy' for image-level extracts (default), "
        "'ee' for pre-computed OpenET EE asset extracts",
    )
    parser.add_argument(
        "--exclude-sites",
        type=str,
        default=None,
        help="Comma-separated site IDs to exclude",
    )
    parser.add_argument(
        "--getinfo",
        action="store_true",
        help="Ingest NDVI/SNODAS/properties from the getInfo extraction "
        "outputs (getinfo/ dirs) and skip bucket ETf ingest",
    )
    args = parser.parse_args()

    # Parse sites argument
    select_sites = None
    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]

    exclude = None
    if args.exclude_sites:
        exclude = [s.strip() for s in args.exclude_sites.split(",")]

    # Load configuration
    config = _load_config()

    # Create or open container (builds the canonical shapefile if needed)
    container = create_project_container(config, overwrite=args.overwrite, exclude_sites=exclude)

    # Run full preparation workflow
    prep_all(
        container,
        config,
        sites=select_sites,
        overwrite=args.overwrite,
        add_sentinel=not args.skip_sentinel,
        openet_source=args.openet_source,
        getinfo=args.getinfo,
    )

    # Close container to ensure data is saved
    container.close()

    print(f"\nContainer saved to: {container.path}")
    print("\nTo run the model:")
    print("  python calibrate.py")

"""
Container-based data preparation workflow for 4_Flux_Network.

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
    8. Export prepped_input.json for model consumption

Usage:
    python container_prep.py

    # Or use functions directly:
    from prep_container import create_project_container, prep_all
    container = create_project_container(overwrite=True)
    prep_all(container)
"""

import os
from pathlib import Path

from swimrs.container import SwimContainer, create_container, open_container
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    """Load project configuration from TOML file."""
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "4_Flux_Network.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))
    return cfg


def create_project_container(cfg: ProjectConfig = None,
                             overwrite: bool = False) -> SwimContainer:
    """
    Create a new SwimContainer for this project.

    Args:
        cfg: ProjectConfig instance (loaded if None)
        overwrite: If True, overwrite existing container

    Returns:
        SwimContainer instance
    """
    if cfg is None:
        cfg = _load_config()

    container_path = os.path.join(cfg.project_ws, f"{cfg.project_name}.swim")

    if os.path.exists(container_path) and not overwrite:
        print(f"Opening existing container: {container_path}")
        return open_container(container_path, mode="r+")

    print(f"Creating new container: {container_path}")
    container = create_container(
        path=container_path,
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

    Corresponds to: prep_timeseries() meteorology portion

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
    container.ingest_gridmet(
        source_dir=cfg.met_dir,
        variables=["eto", "etr", "prcp", "tmin", "tmax", "srad", "u2", "ea"],
        include_corrected=True,  # Also ingest eto_corr, etr_corr if available
        overwrite=overwrite,
    )


def ingest_remote_sensing(container: SwimContainer, cfg: ProjectConfig,
                          sites: list = None, overwrite: bool = False,
                          add_sentinel: bool = True):
    """
    Ingest remote sensing data (NDVI, ETf) into the container.

    Corresponds to: prep_earthengine_extracts()

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        sites: Optional list of site IDs to include
        overwrite: If True, replace existing data
        add_sentinel: If True, also ingest Sentinel NDVI
    """
    print("\n=== Ingesting Remote Sensing ===")

    masks = ["irr", "inv_irr"]
    models = [cfg.etf_target_model] + (cfg.etf_ensemble_members or [])

    # Ingest Landsat NDVI
    for mask in masks:
        ndvi_dir = os.path.join(cfg.landsat_dir, "extracts", "ndvi", mask)
        if os.path.isdir(ndvi_dir):
            print(f"Ingesting Landsat NDVI ({mask})...")
            container.ingest_ee_ndvi(
                csv_dir=ndvi_dir,
                instrument="landsat",
                mask=mask,
                fields=sites,
                overwrite=overwrite,
            )

    # Ingest Sentinel NDVI
    if add_sentinel:
        for mask in masks:
            ndvi_dir = os.path.join(cfg.sentinel_dir, "extracts", "ndvi", mask)
            if os.path.isdir(ndvi_dir):
                print(f"Ingesting Sentinel NDVI ({mask})...")
                container.ingest_ee_ndvi(
                    csv_dir=ndvi_dir,
                    instrument="sentinel",
                    mask=mask,
                    fields=sites,
                    overwrite=overwrite,
                )

    # Ingest ETf for each model
    for model in models:
        for mask in masks:
            etf_dir = os.path.join(cfg.landsat_dir, "extracts", f"{model}_etf", mask)
            if os.path.isdir(etf_dir):
                print(f"Ingesting ETf ({model}, {mask})...")
                container.ingest_ee_etf(
                    csv_dir=etf_dir,
                    instrument="landsat",
                    model=model,
                    mask=mask,
                    fields=sites,
                    overwrite=overwrite,
                )


def ingest_snow(container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False):
    """
    Ingest SNODAS snow data into the container.

    Corresponds to: prep_snow()

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        overwrite: If True, replace existing data
    """
    print("\n=== Ingesting Snow (SNODAS) ===")

    if "snow/snodas/swe" in container._root and not overwrite:
        print("SNODAS data already ingested, skipping")
        return

    # SNODAS can be ingested from either the raw directory or the JSON
    if cfg.snodas_out_json and os.path.exists(cfg.snodas_out_json):
        container.ingest_snodas(
            json_path=cfg.snodas_out_json,
            overwrite=overwrite,
        )
    elif cfg.snodas_in_dir and os.path.isdir(cfg.snodas_in_dir):
        # If only raw SNODAS directory exists, need to process it first
        from swimrs.data_extraction.snodas.snodas import create_timeseries_json
        import tempfile

        # Create temporary JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_json = f.name

        create_timeseries_json(
            cfg.snodas_in_dir,
            temp_json,
            feature_id=cfg.feature_id_col,
        )

        container.ingest_snodas(json_path=temp_json, overwrite=overwrite)
        os.unlink(temp_json)
    else:
        print("Warning: No SNODAS data found, skipping")


def ingest_properties(container: SwimContainer, cfg: ProjectConfig,
                      sites: list = None, overwrite: bool = False):
    """
    Ingest field properties (soils, LULC, irrigation) into the container.

    Corresponds to: prep_field_properties()

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        sites: Optional list of site IDs to include
        overwrite: If True, replace existing data
    """
    print("\n=== Ingesting Properties ===")

    # For properties, we can either:
    # 1. Ingest from individual CSVs (soils, LULC, irrigation)
    # 2. Ingest from a pre-built properties.json

    if cfg.properties_json and os.path.exists(cfg.properties_json):
        # Use existing properties JSON
        container.ingest_properties(
            properties_json=cfg.properties_json,
            fields=sites,
            overwrite=overwrite,
        )
    else:
        # Build properties from individual sources
        container.ingest_properties(
            soils_csv=cfg.ssurgo_csv,
            lulc_csv=cfg.lulc_csv,
            irr_csv=cfg.irr_csv,
            fields=sites,
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

    container.compute_fused_ndvi(
        masks=("irr", "inv_irr"),
        min_pairs=20,
        window_days=1,
        overwrite=overwrite,
    )


def compute_dynamics(container: SwimContainer, cfg: ProjectConfig, overwrite: bool = False):
    """
    Compute irrigation, groundwater subsidy, and K parameters.

    Corresponds to: prep_dynamics()

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        overwrite: If True, replace existing dynamics
    """
    print("\n=== Computing Dynamics ===")

    container.compute_dynamics(
        etf_model=cfg.etf_target_model,
        masks=("irr", "inv_irr"),
        irr_threshold=cfg.irrigation_threshold or 0.1,
        use_mask=True,
        use_lulc=False,
        lookback=5,
        overwrite=overwrite,
    )


def export_model_inputs(container: SwimContainer, cfg: ProjectConfig, output_path: str = None):
    """
    Export data in prepped_input.json format for model consumption.

    Corresponds to: prep_input_json()

    Args:
        container: SwimContainer instance
        cfg: ProjectConfig instance
        output_path: Output path (defaults to cfg.input_data)
    """
    print("\n=== Exporting Model Inputs ===")

    if output_path is None:
        output_path = cfg.input_data

    container.export_prepped_input_json(
        output_path=output_path,
        etf_model=cfg.etf_target_model,
        masks=("irr", "inv_irr"),
        instrument="landsat",
        use_fused_ndvi=True,
    )


def prep_all(container: SwimContainer, cfg: ProjectConfig = None, sites: list = None,
             overwrite: bool = False, add_sentinel: bool = True):
    """
    Run the complete data preparation workflow.

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
    ingest_meteorology(container, cfg, overwrite=overwrite)

    # Step 2: Ingest remote sensing (NDVI, ETf)
    ingest_remote_sensing(container, cfg, sites=sites,
                          overwrite=overwrite, add_sentinel=add_sentinel)

    # Step 3: Ingest snow
    ingest_snow(container, cfg, overwrite=overwrite)

    # Step 4: Ingest properties
    ingest_properties(container, cfg, sites=sites, overwrite=overwrite)

    # Step 5: Compute fused NDVI
    compute_fused_ndvi(container, overwrite=overwrite)

    # Step 6: Compute dynamics
    compute_dynamics(container, cfg, overwrite=overwrite)

    # Step 7: Export model inputs
    export_model_inputs(container, cfg)

    print("\n=== Container Preparation Complete ===")
    print(container.inventory)


if __name__ == "__main__":
    # Load configuration
    config = _load_config()
    select_sites = None  # Set to list of site IDs to process subset

    # Create or open container
    container = create_project_container(config, overwrite=True)

    # Run full preparation workflow
    prep_all(container, config, sites=select_sites, overwrite=True, add_sentinel=True)

    # Close container to ensure data is saved
    container.close()

    print(f"\nContainer saved to: {container.path}")
    print("\nTo use with the model:")
    print("  Option A: Use exported prepped_input.json with SamplePlots")
    print("  Option B: Use ContainerPlots directly (no JSON export needed)")

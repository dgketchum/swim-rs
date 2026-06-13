"""
Container prep for LS Ensemble POR experiment (2008-2025).

Landsat SSEBop + Landsat PT-JPL ETf, full period of record where both
products are available. PEST builder's ensemble mode averages them.

Usage:
    python container_prep_ls_ensemble_por.py [--config PATH] [--overwrite]
"""

import os
from pathlib import Path

from swimrs.container import SwimContainer, create_container, open_container
from swimrs.swim.config import ProjectConfig

TOML = Path(__file__).resolve().parent / "6_Flux_International_LSEnsemble_POR.toml"


def _load_config(config_path: str | None = None) -> ProjectConfig:
    conf = Path(config_path) if config_path else TOML
    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def build_container(cfg: ProjectConfig, overwrite: bool = False) -> SwimContainer:
    container_path = cfg.container_path or os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")

    if os.path.exists(container_path) and not overwrite:
        print(f"Opening existing container: {container_path}")
        return open_container(container_path, mode="r+")

    print(f"Creating new container: {container_path}")
    return create_container(
        uri=container_path,
        fields_shapefile=cfg.fields_shapefile,
        uid_column=cfg.feature_id_col,
        start_date=cfg.start_dt,
        end_date=cfg.end_dt,
        project_name=cfg.project_name,
        overwrite=overwrite,
    )


def ingest_met(container: SwimContainer, cfg: ProjectConfig):
    print("\n=== Ingesting Meteorology (ERA5-Land) ===")
    container.ingest.era5(
        source_dir=cfg.met_dir,
        variables=cfg.era5_params or ["swe", "eto", "tmean", "tmin", "tmax", "prcp", "srad"],
    )


def ingest_ndvi(container: SwimContainer, cfg: ProjectConfig):
    print("\n=== Ingesting NDVI ===")
    for instrument, base_dir in [("landsat", cfg.landsat_dir), ("sentinel", cfg.sentinel_dir)]:
        ndvi_dir = os.path.join(base_dir, "extracts", "ndvi", "no_mask")
        if os.path.isdir(ndvi_dir):
            print(f"  {instrument} NDVI (no_mask)...")
            container.ingest.ndvi(
                source_dir=ndvi_dir,
                uid_column=cfg.feature_id_col,
                instrument=instrument,
                mask="no_mask",
            )


def ingest_etf(container: SwimContainer, cfg: ProjectConfig):
    print("\n=== Ingesting ETf (Landsat SSEBop + Landsat PT-JPL) ===")
    for model, subdir in [("ssebop", "ssebop_etf"), ("ptjpl", "ptjpl_etf")]:
        etf_dir = os.path.join(cfg.landsat_dir, "extracts", subdir, "no_mask")
        if os.path.isdir(etf_dir):
            print(f"  Landsat {model} ETf (no_mask)...")
            container.ingest.etf(
                source_dir=etf_dir,
                uid_column=cfg.feature_id_col,
                model=model,
                instrument="landsat",
                mask="no_mask",
            )


def ingest_properties(container: SwimContainer, cfg: ProjectConfig):
    print("\n=== Ingesting Properties ===")
    container.ingest.properties(
        soils_csv=cfg.hwsd_csv,
        lulc_csv=cfg.lulc_csv,
        irr_csv=None,
        uid_column=cfg.feature_id_col,
        lulc_column="modis_lc",
        extra_lulc_column="glc10_lc",
    )


def compute_fused_ndvi(container: SwimContainer):
    print("\n=== Computing Fused NDVI ===")
    container.compute.fused_ndvi(masks=("no_mask",), overwrite=True)


def compute_dynamics(container: SwimContainer, cfg: ProjectConfig):
    print("\n=== Computing Dynamics ===")
    container.compute.dynamics(
        etf_model="ssebop",
        masks=("no_mask",),
        instruments=("landsat", "sentinel"),
        use_lulc=True,
        irr_threshold=cfg.irrigation_threshold or 0.3,
        met_source=cfg.met_source,
        overwrite=True,
        lulc_irr_method="annual_2yr",
        annual_subsidy_ratio=1.3,
        etf_gap_fallback_model="ptjpl",
    )


def main(config_path: str | None = None, overwrite: bool = False):
    cfg = _load_config(config_path)
    container = build_container(cfg, overwrite=overwrite)

    ingest_met(container, cfg)
    ingest_ndvi(container, cfg)
    ingest_etf(container, cfg)
    ingest_properties(container, cfg)
    compute_fused_ndvi(container)
    compute_dynamics(container, cfg)

    print("\n=== Container Preparation Complete ===")
    import numpy as np

    root = container._root
    for p in [
        "remote_sensing/etf/landsat/ssebop/no_mask",
        "remote_sensing/etf/landsat/ptjpl/no_mask",
    ]:
        arr = np.array(root[p][:])
        print(f"  {p}: shape={arr.shape}, valid={int((~np.isnan(arr)).sum())}")

    container.close()
    print(f"\nContainer saved to: {container.path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, help="Path to TOML config (default: LSEnsemble POR)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(config_path=args.config, overwrite=args.overwrite)

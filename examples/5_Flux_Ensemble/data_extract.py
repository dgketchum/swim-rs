import os
from pathlib import Path

import geopandas as gpd

from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))
    return cfg


def extract_snodas(cfg: ProjectConfig) -> None:
    is_authorized()
    from swimrs.data_extraction.ee.snodas_export import sample_snodas_swe

    sample_snodas_swe(
        feature_coll=cfg.fields_shapefile,
        bucket=cfg.ee_bucket,
        dest='bucket',
        debug=False,
        check_dir=None,
        feature_id=cfg.feature_id_col,
        file_prefix=cfg.project_name,
    )


def extract_properties(cfg: ProjectConfig) -> None:
    is_authorized()
    from swimrs.data_extraction.ee.ee_props import get_cdl, get_irrigation, get_landcover, get_ssurgo

    project = cfg.project_name
    get_cdl(cfg.fields_shapefile, f"{project}_cdl", selector=cfg.feature_id_col, dest='bucket', bucket=cfg.ee_bucket, file_prefix=project)
    get_irrigation(cfg.fields_shapefile, f"{project}_irr", debug=True, selector=cfg.feature_id_col, lanid=True, dest='bucket', bucket=cfg.ee_bucket, file_prefix=project)
    get_ssurgo(cfg.fields_shapefile, f"{project}_ssurgo", debug=False, selector=cfg.feature_id_col, dest='bucket', bucket=cfg.ee_bucket, file_prefix=project)
    get_landcover(cfg.fields_shapefile, f"{project}_landcover", debug=False, selector=cfg.feature_id_col, out_fmt="CSV", dest='bucket', bucket=cfg.ee_bucket, file_prefix=project)


def extract_ndvi(cfg: ProjectConfig, sites=None, get_sentinel: bool = True) -> None:
    is_authorized()
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    for mask in ["irr", "inv_irr"]:
        dst = os.path.join(cfg.landsat_dir, "extracts", "ndvi", mask)
        sparse_sample_ndvi(
            cfg.fields_shapefile,
            bucket=cfg.ee_bucket,
            dest='bucket',
            debug=False,
            mask_type=mask,
            check_dir=dst,
            start_yr=cfg.start_dt.year,
            end_yr=cfg.end_dt.year,
            feature_id=cfg.feature_id_col,
            satellite="landsat",
            state_col=cfg.state_col,
            select=sites,
            file_prefix=cfg.project_name,
        )

        if get_sentinel:
            dst = os.path.join(cfg.sentinel_dir, "extracts", "ndvi", mask)
            sparse_sample_ndvi(
                cfg.fields_shapefile,
                bucket=cfg.ee_bucket,
                dest='bucket',
                debug=False,
                mask_type=mask,
                check_dir=dst,
                start_yr=max(2017, cfg.start_dt.year),
                end_yr=cfg.end_dt.year,
                feature_id=cfg.feature_id_col,
                satellite="sentinel",
                state_col=cfg.state_col,
                select=sites,
                file_prefix=cfg.project_name,
            )


def extract_openet_etf(
    cfg: ProjectConfig,
    sites=None,
    models=None,
    mask_types=None,
) -> None:
    """
    Extract ETf zonal statistics using the open source OpenET software packages.

    This function uses the etf/ package modules to export per-scene ET fraction
    zonal means for all 5 OpenET models: PT-JPL, SIMS, SSEBop, DisALEXI, and geeSEBAL.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object.
    sites : list, optional
        List of site IDs to process. If None, processes all sites.
    models : list, optional
        List of model names to extract. Options: 'ptjpl', 'sims', 'ssebop',
        'disalexi', 'geesebal'. If None, extracts all models.
    mask_types : list, optional
        List of irrigation mask types: 'no_mask', 'irr', 'inv_irr'.
        Default: ['irr', 'inv_irr'].
    """
    import ee
    ee.Initialize()

    from swimrs.data_extraction.ee import (
        export_ptjpl_zonal_stats,
        export_sims_zonal_stats,
        export_ssebop_zonal_stats,
        export_geesebal_zonal_stats,
    )

    # Map model names to export functions
    model_exporters = {
        'ptjpl': export_ptjpl_zonal_stats,
        'sims': export_sims_zonal_stats,
        'ssebop': export_ssebop_zonal_stats,
        'geesebal': export_geesebal_zonal_stats,
    }

    if models is None:
        models = list(model_exporters.keys())

    if mask_types is None:
        mask_types = ['irr', 'inv_irr']

    for mask_type in mask_types:
        print(f"\n{'#'*60}")
        print(f"Processing mask type: {mask_type}")
        print(f"{'#'*60}")

        for model in models:
            if model not in model_exporters:
                print(f"Unknown model: {model}, skipping")
                continue

            export_fn = model_exporters[model]
            chk_dir = os.path.join(cfg.landsat_dir, 'extracts', f'{model}_etf', mask_type)

            print(f"\n{'='*60}")
            print(f"Extracting {model.upper()} ETf zonal statistics ({mask_type})")
            print(f"{'='*60}")

            export_fn(
                shapefile=cfg.fields_shapefile,
                bucket=cfg.ee_bucket,
                feature_id=cfg.feature_id_col,
                select=sites,
                start_yr=cfg.start_dt.year,
                end_yr=cfg.end_dt.year,
                mask_type=mask_type,
                check_dir=chk_dir,
                state_col=cfg.state_col,
                buffer=None,
                batch_size=60,
                file_prefix=cfg.project_name,
            )


def extract_gridmet(cfg: ProjectConfig, sites=None) -> None:
    from swimrs.data_extraction.gridmet.gridmet import assign_gridmet_and_corrections, download_gridmet

    nldas_needed = (cfg.swb_mode == "ier")

    assign_gridmet_and_corrections(
        fields=cfg.fields_shapefile,
        gridmet_points=cfg.gridmet_centroids,
        gridmet_ras=cfg.correction_tifs,
        fields_join=cfg.gridmet_mapping_shp,
        factors_js=cfg.gridmet_factors,
        feature_id=cfg.feature_id_col,
        field_select=sites,
    )

    download_gridmet(
        cfg.gridmet_mapping_shp,
        cfg.gridmet_factors,
        cfg.met_dir,
        start=str(cfg.start_dt.date()),
        end=str(cfg.end_dt.date()),
        overwrite=False,
        append=True,
        use_nldas=nldas_needed,
        feature_id=cfg.gridmet_mapping_index_col,
        target_fields=sites,
    )


if __name__ == "__main__":
    config = _load_config()
    select_sites = gpd.read_file(config.fields_shapefile)['side_id'].to_list()

    # Standard extraction workflow
    # extract_snodas(config)
    # extract_properties(config)
    extract_ndvi(config, select_sites, get_sentinel=True)
    # extract_gridmet(config, select_sites)

    # OpenET ETf extraction using the etf/ package modules
    # This extracts zonal statistics for all 5 open source OpenET models:
    # PT-JPL, SIMS, SSEBop, DisALEXI, and geeSEBAL
    # Default mask_types=['irr', 'inv_irr'] matches example 4 workflow
    extract_openet_etf(
        config,
        sites=select_sites,
        models=['ptjpl', 'sims', 'ssebop','geesebal'],
        mask_types=['irr'],
    )


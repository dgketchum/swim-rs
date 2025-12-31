import os
from pathlib import Path

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
    )


def extract_properties(cfg: ProjectConfig) -> None:
    is_authorized()
    from swimrs.data_extraction.ee.ee_props import get_cdl, get_irrigation, get_landcover, get_ssurgo

    project = cfg.project_name
    get_cdl(cfg.fields_shapefile, f"{project}_cdl", selector=cfg.feature_id_col, dest='bucket', bucket=cfg.ee_bucket)
    get_irrigation(cfg.fields_shapefile, f"{project}_irr", debug=True, selector=cfg.feature_id_col, lanid=True, dest='bucket', bucket=cfg.ee_bucket)
    get_ssurgo(cfg.fields_shapefile, f"{project}_ssurgo", debug=False, selector=cfg.feature_id_col, dest='bucket', bucket=cfg.ee_bucket)
    get_landcover(cfg.fields_shapefile, f"{project}_landcover", debug=False, selector=cfg.feature_id_col, out_fmt="CSV", dest='bucket', bucket=cfg.ee_bucket)


def extract_remote_sensing(cfg: ProjectConfig, sites=None, get_sentinel: bool = True) -> None:
    is_authorized()
    from swimrs.data_extraction.ee.etf_export import sparse_sample_etf
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    models = [cfg.etf_target_model] + (cfg.etf_ensemble_members or [])

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
            )

        for model in models:
            # PT-JPL and SIMS extractions are handled separately (OpenET software workflows).
            if model in ["ptjpl", "sims"]:
                continue
            dst = os.path.join(cfg.landsat_dir, "extracts", f"{model}_etf", mask)
            sparse_sample_etf(
                cfg.fields_shapefile,
                bucket=cfg.ee_bucket,
                dest='bucket',
                debug=False,
                mask_type=mask,
                check_dir=dst,
                start_yr=max(2016, cfg.start_dt.year),
                end_yr=cfg.end_dt.year,
                feature_id=cfg.feature_id_col,
                state_col=cfg.state_col,
                select=sites,
                model=model,
            )


def extract_gridmet(cfg: ProjectConfig, sites=None) -> None:
    from swimrs.data_extraction.gridmet.gridmet import assign_gridmet_and_corrections, download_gridmet

    nldas_needed = (cfg.swb_mode == "ier")

    assign_gridmet_and_corrections(
        fields=cfg.gridmet_mapping_shp,
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
    select_sites = None
    extract_snodas(config)
    extract_properties(config)
    extract_remote_sensing(config, select_sites, get_sentinel=True)
    extract_gridmet(config, select_sites)


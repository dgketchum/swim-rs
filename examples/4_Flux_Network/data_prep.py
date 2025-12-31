import os
from pathlib import Path

from swimrs.prep import get_ensemble_parameters
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "4_Flux_Network.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))
    return cfg


def prep_earthengine_extracts(cfg: ProjectConfig, sites=None, overwrite: bool = False, add_sentinel: bool = True):
    from swimrs.prep.remote_sensing import join_remote_sensing, sparse_time_series

    masks = ["irr", "inv_irr"]
    rs_files = []

    models = [cfg.etf_target_model] + (cfg.etf_ensemble_members or [])
    years = list(range(cfg.start_dt.year, cfg.end_dt.year + 1))

    for model in models:
        for mask in masks:
            ee_data = os.path.join(cfg.landsat_dir, "extracts", f"{model}_etf", mask)
            out_parquet = os.path.join(cfg.landsat_tables_dir, f"{model}_etf_{mask}.parquet")
            rs_files.append(out_parquet)
            if os.path.exists(out_parquet) and not overwrite:
                continue
            sparse_time_series(
                cfg.fields_shapefile,
                ee_data,
                years,
                out_parquet,
                feature_id=cfg.feature_id_col,
                instrument="landsat",
                parameter="etf",
                algorithm=model,
                mask=mask,
                select=sites,
            )

    for mask in masks:
        ee_data = os.path.join(cfg.landsat_dir, "extracts", "ndvi", mask)
        out_parquet = os.path.join(cfg.landsat_tables_dir, f"ndvi_{mask}.parquet")
        rs_files.append(out_parquet)
        if not (os.path.exists(out_parquet) and not overwrite):
            sparse_time_series(
                cfg.fields_shapefile,
                ee_data,
                years,
                out_parquet,
                feature_id=cfg.feature_id_col,
                instrument="landsat",
                parameter="ndvi",
                algorithm="none",
                mask=mask,
                select=sites,
            )

        if add_sentinel:
            ee_data = os.path.join(cfg.sentinel_dir, "extracts", "ndvi", mask)
            out_parquet = os.path.join(cfg.sentinel_tables_dir, f"ndvi_{mask}.parquet")
            rs_files.append(out_parquet)
            if not (os.path.exists(out_parquet) and not overwrite):
                sparse_time_series(
                    cfg.fields_shapefile,
                    ee_data,
                    years,
                    out_parquet,
                    feature_id=cfg.feature_id_col,
                    instrument="sentinel",
                    parameter="ndvi",
                    algorithm="none",
                    mask=mask,
                    select=sites,
                )

    join_remote_sensing(rs_files, cfg.remote_sensing_tables_dir, station_selection="inclusive")


def prep_field_properties(cfg: ProjectConfig, sites=None):
    from swimrs.prep.field_properties import write_field_properties

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    write_field_properties(
        cfg.fields_shapefile,
        cfg.properties_json,
        cfg.lulc_csv,
        irr=cfg.irr_csv,
        lulc_key="modis_lc",
        soils=cfg.ssurgo_csv,
        index_col=cfg.feature_id_col,
        flux_meta=station_metadata if os.path.exists(station_metadata) else None,
        select=sites,
        **{"extra_lulc_key": "glc10_lc"},
    )


def prep_snow(cfg: ProjectConfig, index_col=None):
    from swimrs.data_extraction.snodas.snodas import create_timeseries_json

    create_timeseries_json(cfg.snodas_in_dir, cfg.snodas_out_json, feature_id=index_col or cfg.feature_id_col)


def prep_timeseries(cfg: ProjectConfig, sites=None):
    from swimrs.prep.field_timeseries import join_daily_timeseries

    join_daily_timeseries(
        fields=cfg.gridmet_mapping_shp,
        met_dir=cfg.met_dir,
        rs_dir=cfg.remote_sensing_tables_dir,
        dst_dir=cfg.joined_timeseries_dir,
        snow=cfg.snodas_out_json,
        overwrite=True,
        start_date=cfg.start_dt,
        end_date=cfg.end_dt,
        feature_id=cfg.gridmet_mapping_index_col,
        **{"met_mapping": "GFID", "target_fields": sites},
    )


def prep_dynamics(cfg: ProjectConfig, sites=None, sentinel: bool = True):
    from swimrs.prep.dynamics import process_dynamics_batch

    sensors = ("landsat", "sentinel") if sentinel else ("landsat",)
    process_dynamics_batch(
        cfg.joined_timeseries_dir,
        cfg.properties_json,
        cfg.dynamics_data_json,
        etf_target=cfg.etf_target_model,
        irr_threshold=cfg.irrigation_threshold,
        select=sites,
        masks=("irr", "inv_irr"),
        instruments=sensors,
        use_lulc=False,
        use_mask=True,
        lookback=5,
        num_workers=12,
    )


def prep_input_json(cfg: ProjectConfig, sites=None):
    from swimrs.prep.prep_plots import prep_fields_json

    params = get_ensemble_parameters()
    params = [p for p in params if p[0] in ["none", "ptjpl", "sims", "ssebop"]]
    prep_fields_json(
        cfg.properties_json,
        cfg.joined_timeseries_dir,
        cfg.dynamics_data_json,
        cfg.input_data,
        target_plots=sites,
        rs_params=params,
        interp_params=("ndvi",),
    )


if __name__ == "__main__":
    config = _load_config()
    select_sites = None

    prep_earthengine_extracts(config, select_sites, overwrite=True, add_sentinel=True)
    prep_field_properties(config, select_sites)
    prep_snow(config)
    prep_timeseries(config, select_sites)
    prep_dynamics(config, select_sites, sentinel=True)
    prep_input_json(config, select_sites)


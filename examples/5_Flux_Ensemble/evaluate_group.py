import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pandas as pd

from swimrs.analysis.metrics import compare_etf_estimates
from swimrs.model.obs_field_cycle import field_day_loop
from swimrs.prep import get_flux_sites
from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import SamplePlots


def compare_openet(fid: str, flux_file: str, model_output: str, openet_dir: str, plots: SamplePlots,
                   model: str, gap_tolerance: int = 5, ssebop_eto_source: str = "eto_corr"):
    openet_daily = os.path.join(openet_dir, "daily_data", f"{fid}.csv")
    openet_monthly = os.path.join(openet_dir, "monthly_data", f"{fid}.csv")
    irr_ = plots.input["irr_data"][fid]
    daily, overpass, monthly = compare_etf_estimates(
        model_output,
        flux_file,
        openet_daily_path=openet_daily,
        openet_monthly_path=openet_monthly,
        irr=irr_,
        target_model=model,
        gap_tolerance=gap_tolerance,
        ssebop_eto_source=ssebop_eto_source,
    )
    return monthly


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    target_dir = os.path.join(cfg.project_ws, "diy_ensemble_non_crop")
    cfg.forecast_parameters_csv = os.path.join(target_dir, f"{cfg.project_name}.3.par.csv")
    cfg.spinup = os.path.join(target_dir, "spinup.json")
    cfg.input_data = os.path.join(target_dir, "prepped_input.json")

    openet_dir = os.path.join(cfg.data_dir, "openet_flux")
    flux_dir = os.path.join(cfg.data_dir, "daily_flux_files")

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    sites, sdf = get_flux_sites(station_metadata, crop_only=False, return_df=True, western_only=True, header=1)

    plots = SamplePlots()
    plots.initialize_plot_data(cfg)

    cfg.calibrate = False
    cfg.forecast = True
    cfg.read_forecast_parameters()

    if os.path.exists(cfg.forecast_parameters_csv):
        modified = datetime.fromtimestamp(os.path.getmtime(cfg.forecast_parameters_csv))
        print(f"Calibration made {modified}")

    df_dct = field_day_loop(cfg, plots, debug_flag=True)
    sites = [k for k in df_dct.keys() if k in sites]

    results = {}
    for site_id in sites:
        out_csv = os.path.join(target_dir, f"{site_id}.csv")
        df = df_dct[site_id].copy()
        in_df = plots.input_to_dataframe(site_id)
        df = pd.concat([df, in_df], axis=1, ignore_index=False).loc[cfg.start_dt: cfg.end_dt]
        df.to_csv(out_csv)

        flux_file = os.path.join(flux_dir, f"{site_id}_daily_data.csv")
        monthly = compare_openet(site_id, flux_file, out_csv, openet_dir, plots, model=cfg.etf_target_model,
                                 gap_tolerance=5)
        if monthly:
            results[site_id] = monthly

    pprint({k: v.get("rmse_swim") for k, v in results.items() if isinstance(v, dict)})


import collections
import os
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple

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


def _verbose_monthly_summary(site_id: str, monthly: dict, target_model: str) -> Tuple[Optional[str], Optional[str]]:
    rmse_all = {
        k.split("_", 1)[1]: v
        for k, v in monthly.items()
        if isinstance(k, str) and k.startswith("rmse_") and isinstance(v, (int, float))
    }
    if not rmse_all:
        return None, None

    best_overall_model = min(rmse_all, key=rmse_all.get)

    best_pair_model = None
    if "rmse_swim" in monthly and "rmse_openet" in monthly:
        best_pair_model = "swim" if monthly["rmse_swim"] <= monthly["rmse_openet"] else "openet"

    n_samples = monthly.get("n_samples")
    print(f"n Samples: {n_samples}")
    print("Best overall:", best_overall_model)
    print("Best swim vs openet:", best_pair_model if best_pair_model else "NA")

    if target_model == "openet":
        print(f"Flux Mean: {monthly.get('mean_flux')}")
        print(f"SWIM Mean: {monthly.get('mean_swim')}")
        print(f"{best_overall_model} Mean: {monthly.get(f'mean_{best_overall_model}')}")
        print(f"OpenET Mean: {monthly.get('mean_openet')}")
        print(f"SWIM RMSE: {monthly.get('rmse_swim')}")
        print(f"{best_overall_model} RMSE: {monthly.get(f'rmse_{best_overall_model}')}")
        print(f"OpenET RMSE: {monthly.get('rmse_openet')}")
    elif target_model == "ssebop":
        print(f"Flux Mean: {monthly.get('mean_flux')}")
        print(f"SWIM Mean: {monthly.get('mean_swim')}")
        print(f"SSEBop NHM Mean: {monthly.get('mean_ssebop')}")
        print(f"SWIM RMSE: {monthly.get('rmse_swim')}")
        print(f"{best_overall_model} RMSE: {monthly.get(f'rmse_{best_overall_model}')}")
        print(f"SSEBop NHM RMSE: {monthly.get('rmse_ssebop')}")

    return best_overall_model, best_pair_model


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    target_dir = os.path.join(cfg.project_ws, "diy_ensemble")
    cfg.forecast_parameters_csv = os.path.join(target_dir, f"{cfg.project_name}.3.par.csv")
    cfg.spinup = os.path.join(target_dir, "spinup.json")
    cfg.input_data = os.path.join(target_dir, "prepped_input.json")

    openet_dir = os.path.join(cfg.data_dir, "openet_flux")
    flux_dir = os.path.join(cfg.data_dir, "daily_flux_files")

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    ec_sites, sdf = get_flux_sites(station_metadata, crop_only=True, return_df=True, western_only=True, header=1)

    plots = SamplePlots()
    plots.initialize_plot_data(cfg)

    cfg.calibrate = False
    cfg.forecast = True
    cfg.read_forecast_parameters()

    if os.path.exists(cfg.forecast_parameters_csv):
        modified = datetime.fromtimestamp(os.path.getmtime(cfg.forecast_parameters_csv))
        print(f"Calibration made {modified}")

    df_dct = field_day_loop(cfg, plots, debug_flag=True)
    sites = [k for k in df_dct.keys() if k in ec_sites]

    print(f"{len(sites)} sites to evalutate in 5_Flux_Ensemble")

    incomplete, complete = [], []
    results_overall, results_pair = [], []
    results = {}
    for ee, site_id in enumerate(sites):
        try:
            lulc = sdf.at[site_id, "General classification"]
        except Exception:
            lulc = "NA"

        print(f"\n{ee} {site_id}: {lulc}")

        out_csv = os.path.join(target_dir, f"{site_id}.csv")
        df = df_dct[site_id].copy()
        in_df = plots.input_to_dataframe(site_id)
        df = pd.concat([df, in_df], axis=1, ignore_index=False).loc[cfg.start_dt: cfg.end_dt]
        df.to_csv(out_csv)

        flux_file = os.path.join(flux_dir, f"{site_id}_daily_data.csv")
        monthly = compare_openet(site_id, flux_file, out_csv, openet_dir, plots, model=cfg.etf_target_model,
                                 gap_tolerance=5)
        if monthly and isinstance(monthly, dict):
            results[site_id] = monthly
            best_overall, best_pair = _verbose_monthly_summary(site_id, monthly, target_model=cfg.etf_target_model)
            if best_overall:
                results_overall.append((best_overall, lulc))
            if best_pair:
                results_pair.append((best_pair, lulc))
            complete.append(site_id)
        else:
            incomplete.append(site_id)

    pprint({k: v.get("rmse_swim") for k, v in results.items() if isinstance(v, dict)})
    pprint({s: [t[0] for t in results_overall].count(s) for s in set(t[0] for t in results_overall)})
    pprint(
        {
            category: [
                item[0]
                for item in collections.Counter(t[0] for t in results_overall if t[1] == category).most_common(3)
            ]
            for category in set(t[1] for t in results_overall)
        }
    )
    pprint({s: [t[0] for t in results_pair].count(s) for s in set(t[0] for t in results_pair)})
    pprint(
        {
            category: [
                item[0]
                for item in collections.Counter(t[0] for t in results_pair if t[1] == category).most_common(3)
            ]
            for category in set(t[1] for t in results_pair)
        }
    )
    print(f"complete: {complete}")
    print(f"incomplete: {incomplete}")

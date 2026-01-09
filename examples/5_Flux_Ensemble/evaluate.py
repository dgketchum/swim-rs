import os
import time
from pathlib import Path
from pprint import pprint

import pandas as pd

from openet_evaluation import evaluate_openet_site
from swimrs.model.obs_field_cycle import field_day_loop
from swimrs.prep import get_flux_sites
from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import SamplePlots


def run_flux_site(fid: str, cfg: ProjectConfig, plots: SamplePlots, outfile: str) -> None:
    start_time = time.time()
    df_dct = field_day_loop(cfg, plots, debug_flag=True)
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds\n")

    df = df_dct[fid].copy()
    in_df = plots.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)
    df = df.loc[cfg.start_dt: cfg.end_dt]
    df.to_csv(outfile)


def compare_openet(fid: str, flux_file: str, model_output: str, openet_dir: str, plots: SamplePlots,
                   return_comparison: bool = False, gap_tolerance: int = 5):
    """Compare SWIM and OpenET ensemble against flux observations for a single site."""
    openet_daily = os.path.join(openet_dir, "daily_data", f"{fid}.csv")
    openet_monthly = os.path.join(openet_dir, "monthly_data", f"{fid}.csv")
    irr_ = plots.input["irr_data"][fid]
    daily, overpass, monthly = evaluate_openet_site(
        model_output,
        flux_file,
        openet_daily_path=openet_daily,
        openet_monthly_path=openet_monthly,
        irr=irr_,
        gap_tolerance=gap_tolerance,
    )

    if monthly is None:
        return None

    agg_comp = monthly.copy()
    if len(agg_comp) < 3:
        return None

    rmse_values = {k.split("_", 1)[1]: v for k, v in agg_comp.items() if k.startswith("rmse_")}
    if not rmse_values:
        return None

    best_overall_model = min(rmse_values, key=rmse_values.get)
    if not return_comparison:
        return best_overall_model

    print(f"n Samples: {agg_comp.get('n_samples')}")
    print("Best overall:", best_overall_model)
    return best_overall_model


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    sites, sdf = get_flux_sites(station_metadata, crop_only=False, return_df=True, western_only=True, header=1)

    openet_dir = os.path.join(cfg.data_dir, "openet_flux")
    flux_dir = os.path.join(cfg.data_dir, "daily_flux_files")

    run_dir = os.path.join(cfg.project_ws, "results", "tight")
    os.makedirs(run_dir, exist_ok=True)

    plots_ = SamplePlots()
    plots_.initialize_plot_data(cfg)

    complete, incomplete = [], []
    for i, site_id in enumerate(sites):
        lulc = sdf.at[site_id, "General classification"]
        print(f"\n{i} {site_id}: {lulc}")

        flux_file = os.path.join(flux_dir, f"{site_id}_daily_data.csv")
        out_csv = os.path.join(run_dir, f"{site_id}.csv")

        try:
            run_flux_site(site_id, cfg, plots_, out_csv)
        except Exception as exc:
            print(f"{site_id} error: {exc}")
            incomplete.append(site_id)
            continue

        _ = compare_openet(site_id, flux_file, out_csv, openet_dir, plots_,
                           return_comparison=True, gap_tolerance=5)
        complete.append(site_id)

    print(f"complete: {complete}")
    print(f"incomplete: {incomplete}")


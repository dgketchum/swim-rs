import os
import time
from pathlib import Path

import pandas as pd

from swimrs.model.obs_field_cycle import field_day_loop
from swimrs.prep import get_ensemble_parameters
from swimrs.prep.prep_plots import prep_fields_json
from swimrs.swim.config import ProjectConfig
from swimrs.swim.sampleplots import SamplePlots


def run_flux_site(fid: str, cfg: ProjectConfig, overwrite_input: bool = False) -> None:
    start_time = time.time()

    models = [cfg.etf_target_model] + (cfg.etf_ensemble_members or [])
    rs_params = get_ensemble_parameters(include=models)

    target_dir = os.path.join(cfg.project_ws, "testrun", fid)
    os.makedirs(target_dir, exist_ok=True)
    station_prepped_input = os.path.join(target_dir, f"prepped_input_{fid}.json")

    if not os.path.isfile(station_prepped_input) or overwrite_input:
        prep_fields_json(
            cfg.properties_json,
            cfg.plot_timeseries,
            cfg.dynamics_data_json,
            station_prepped_input,
            target_plots=[fid],
            rs_params=rs_params,
            interp_params=("ndvi",),
        )

    cfg.input_data = station_prepped_input
    cfg.spinup = os.path.join(target_dir, f"spinup_{fid}.json")

    plots = SamplePlots()
    plots.initialize_plot_data(cfg)

    df_dct = field_day_loop(cfg, plots, debug_flag=True)
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds\n")

    df = df_dct[fid].copy()
    in_df = plots.input_to_dataframe(fid)
    df = pd.concat([df, in_df], axis=1, ignore_index=False)
    df = df.loc[cfg.start_dt: cfg.end_dt]

    out_csv = os.path.join(target_dir, f"{fid}.csv")
    df.to_csv(out_csv)
    print(f"run complete: {fid}, wrote {out_csv}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    run_flux_site("B_01", cfg, overwrite_input=True)


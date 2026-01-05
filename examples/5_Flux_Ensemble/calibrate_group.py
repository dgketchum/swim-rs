import os
import shutil
import tempfile
from pathlib import Path

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.prep import get_ensemble_parameters, get_flux_sites
from swimrs.prep.prep_plots import prep_fields_json, preproc
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)

    cfg.python_script = str(project_dir / "custom_forward_run.py")
    return cfg


def run_pest_sequence(cfg: ProjectConfig, results_dir: str, select_stations, pdc_remove: bool = False,
                      overwrite: bool = False):
    project = cfg.project_name

    if os.path.isdir(cfg.pest_run_dir):
        shutil.rmtree(cfg.pest_run_dir)
    os.makedirs(cfg.pest_run_dir, exist_ok=False)

    os.makedirs(results_dir, exist_ok=True)
    station_prepped_input = os.path.join(results_dir, "prepped_input.json")

    if not os.path.isfile(station_prepped_input) or overwrite:
        models = [cfg.etf_target_model] + (cfg.etf_ensemble_members or [])
        rs_params = get_ensemble_parameters(include=models)
        rs_params = [p for p in rs_params if p[0] in ["none", "ptjpl", "sims", "ssebop"]]
        prep_fields_json(
            cfg.properties_json,
            cfg.plot_timeseries,
            cfg.dynamics_data_json,
            cfg.input_data,
            target_plots=select_stations,
            rs_params=rs_params,
            interp_params=("ndvi",),
        )
        preproc(cfg)

    shutil.copyfile(cfg.input_data, station_prepped_input)
    shutil.copyfile(cfg.input_data, os.path.join(cfg.pest_run_dir, os.path.basename(cfg.input_data)))

    p_dir = os.path.join(cfg.pest_run_dir, "pest")
    m_dir = os.path.join(cfg.pest_run_dir, "master")
    w_dir = os.path.join(cfg.pest_run_dir, "workers")

    os.chdir(Path(__file__).resolve().parent)

    builder = PestBuilder(cfg, use_existing=False, python_script=getattr(cfg, "python_script", None),
                          conflicted_obs=None)
    builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
    builder.build_localizer()

    exe_ = "pestpp-ies"

    if pdc_remove:
        builder.write_control_settings(noptmax=-1, reals=5)
    else:
        builder.write_control_settings(noptmax=0)

    builder.spinup(overwrite=True)
    shutil.copyfile(builder.config.spinup, os.path.join(results_dir, "spinup.json"))

    builder.dry_run(exe_)

    pdc_file = os.path.join(p_dir, f"{project}.pdc.csv")
    if os.path.exists(pdc_file) and pdc_remove:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdc = os.path.join(temp_dir, f"{project}.pdc.csv")
            shutil.copyfile(pdc_file, temp_pdc)

            builder = PestBuilder(cfg, use_existing=False, python_script=getattr(cfg, "python_script", None),
                                  conflicted_obs=temp_pdc)
            builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
            builder.build_localizer()
            builder.write_control_settings(noptmax=0)
            builder.dry_run(exe_)

    builder.write_control_settings(noptmax=3, reals=cfg.realizations)
    pst_name = f"{project}.pst"
    run_pst(p_dir, exe_, pst_name, num_workers=cfg.workers, worker_root=w_dir, master_dir=m_dir, verbose=False,
            cleanup=False)

    for fname in [f"{project}.3.par.csv", f"{project}.2.par.csv", f"{project}.phi.meas.csv", f"{project}.pdc.csv",
                  f"{project}.idx.csv"]:
        src = os.path.join(m_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(results_dir, fname))

    shutil.rmtree(p_dir)
    shutil.rmtree(m_dir)
    shutil.rmtree(w_dir)


if __name__ == "__main__":
    cfg = _load_config()

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    # crop_sites = get_flux_sites(station_metadata, crop_only=True, western_only=True, header=1)
    all_sites = get_flux_sites(station_metadata, crop_only=False, western_only=True, header=1)
    non_crop_sites = [s for s in all_sites if s not in all_sites]

    results = os.path.join(cfg.project_ws, "diy_test_set")
    run_pest_sequence(cfg, results, select_stations=['MR', 'US-FPe', 'ALARC2_Smith6'], overwrite=True, pdc_remove=True)


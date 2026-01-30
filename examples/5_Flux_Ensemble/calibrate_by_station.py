import os
import shutil
import tempfile
from pathlib import Path

from swimrs.calibrate.flux_utils import get_flux_sites
from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _load_config(calibrate: bool = True) -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf), calibrate=calibrate)
    else:
        cfg.read_config(
            str(conf), project_root_override=str(project_dir.parent), calibrate=calibrate
        )

    cfg.python_script = str(project_dir / "custom_forward_run.py")
    return cfg


def run_pest_sequence(
    cfg: ProjectConfig,
    results_dir: str,
    select_stations: list[str],
    pdc_remove: bool = False,
    overwrite: bool = False,
):
    project = cfg.project_name

    container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    container = SwimContainer.open(container_path, mode="r")

    for i, fid in enumerate(select_stations, start=1):
        print(f"{fid}: {i} of {len(select_stations)} stations")

        if os.path.isdir(cfg.pest_run_dir):
            shutil.rmtree(cfg.pest_run_dir)
        os.makedirs(cfg.pest_run_dir, exist_ok=False)

        target_dir = os.path.join(results_dir, fid)
        os.makedirs(target_dir, exist_ok=True)

        p_dir = os.path.join(cfg.pest_run_dir, "pest")
        m_dir = os.path.join(cfg.pest_run_dir, "master")
        w_dir = os.path.join(cfg.pest_run_dir, "workers")

        os.chdir(Path(__file__).resolve().parent)

        builder = PestBuilder(
            cfg,
            container,
            use_existing=False,
            python_script=getattr(cfg, "python_script", None),
            conflicted_obs=None,
        )
        builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
        builder.build_localizer()
        builder.add_regularization()

        exe_ = "pestpp-ies"
        if pdc_remove:
            builder.write_control_settings(noptmax=-1, reals=5)
        else:
            builder.write_control_settings(noptmax=0)

        builder.spinup(overwrite=True)
        shutil.copyfile(builder.config.spinup, os.path.join(target_dir, f"spinup_{fid}.json"))

        builder.dry_run(exe_)

        pdc_file = os.path.join(p_dir, f"{project}.pdc.csv")
        if os.path.exists(pdc_file) and pdc_remove:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdc = os.path.join(temp_dir, f"{project}.pdc.csv")
                shutil.copyfile(pdc_file, temp_pdc)
                builder = PestBuilder(
                    cfg,
                    container,
                    use_existing=False,
                    python_script=getattr(cfg, "python_script", None),
                    conflicted_obs=temp_pdc,
                )
                builder.build_pest(
                    target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members
                )
                builder.build_localizer()
                builder.add_regularization()
                builder.write_control_settings(noptmax=0)
                builder.dry_run(exe_)

        builder.write_control_settings(noptmax=3, reals=cfg.realizations)
        pst_name = f"{project}.pst"
        run_pst(
            p_dir,
            exe_,
            pst_name,
            num_workers=cfg.workers,
            worker_root=w_dir,
            master_dir=m_dir,
            verbose=False,
            cleanup=False,
        )

        fcst_file = os.path.join(m_dir, f"{project}.3.par.csv")
        fcst_out = os.path.join(target_dir, f"{fid}.3.par.csv")
        if not os.path.exists(fcst_file):
            fcst_file = os.path.join(m_dir, f"{project}.2.par.csv")
            fcst_out = os.path.join(target_dir, f"{fid}.2.par.csv")
        if os.path.exists(fcst_file):
            shutil.copyfile(fcst_file, fcst_out)

        for fname in [f"{project}.phi.meas.csv", f"{project}.pdc.csv", f"{project}.idx.csv"]:
            src = os.path.join(m_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(target_dir, f"{fid}.{fname.split('.', 1)[1]}"))

        shutil.rmtree(p_dir)
        shutil.rmtree(m_dir)
        shutil.rmtree(w_dir)


if __name__ == "__main__":
    cfg = _load_config(calibrate=True)
    results_dir = os.path.join(cfg.project_ws, "ptjpl_test")
    os.makedirs(results_dir, exist_ok=True)

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    crop_sites = get_flux_sites(station_metadata, crop_only=True, western_only=True, header=1)
    all_sites = get_flux_sites(station_metadata, crop_only=False, western_only=True, header=1)
    non_crop_sites = [s for s in all_sites if s not in crop_sites]
    sites_ordered = crop_sites + non_crop_sites

    run_pest_sequence(
        cfg, results_dir, select_stations=sites_ordered, pdc_remove=True, overwrite=False
    )

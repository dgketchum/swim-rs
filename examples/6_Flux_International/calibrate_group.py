import os
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _load_config(calibrate: bool = True) -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf_path = project_dir / "6_Flux_International.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf_path), calibrate=calibrate)
    else:
        cfg.read_config(
            str(conf_path), project_root_override=str(project_dir.parent), calibrate=calibrate
        )
    return cfg


def _site_ids(cfg: ProjectConfig, select: list[str] | None = None) -> list[str]:
    gdf = gpd.read_file(cfg.fields_shapefile)
    if cfg.feature_id_col not in gdf.columns:
        raise ValueError(
            f"Feature ID column {cfg.feature_id_col} not found in {cfg.fields_shapefile}"
        )
    ids = sorted(set(gdf[cfg.feature_id_col].astype(str).tolist()))
    if select:
        ids = [i for i in ids if i in set(select)]
    return ids


def run_group_calibration(
    *,
    select_sites: list[str] | None = None,
    workers: int = 8,
    realizations: int | None = None,
    overwrite: bool = False,
    pdc_remove: bool = True,
) -> None:
    cfg = _load_config(calibrate=True)

    sites = _site_ids(cfg, select_sites)
    if not sites:
        raise ValueError("No sites selected for calibration")

    # Ensure calibration workspace exists/fresh
    if overwrite and os.path.isdir(cfg.pest_run_dir):
        shutil.rmtree(cfg.pest_run_dir)
    os.makedirs(cfg.pest_run_dir, exist_ok=True)

    container_path = getattr(cfg, "container_path", None)
    if container_path is None:
        container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    container = SwimContainer.open(container_path, mode="r")

    builder = PestBuilder(
        cfg, container, use_existing=False, python_script=getattr(cfg, "python_script", None)
    )
    builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
    builder.build_localizer()

    exe_ = "pestpp-ies"

    # Short run to detect prior-data conflict
    if pdc_remove:
        builder.write_control_settings(noptmax=-1, reals=5)
    else:
        builder.write_control_settings(noptmax=0)

    builder.spinup(overwrite=True)
    builder.dry_run(exe_)

    project = cfg.project_name
    pdc_file = os.path.join(builder.pest_dir, f"{project}.pdc.csv")
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
            builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
            builder.build_localizer()
            builder.write_control_settings(noptmax=0)
            builder.dry_run(exe_)

    # Main run
    reals = int(realizations) if realizations is not None else int(cfg.realizations or 250)
    builder.write_control_settings(noptmax=3, reals=reals)

    pst_name = f"{project}.pst"
    run_pst(
        builder.pest_dir,
        exe_,
        pst_name,
        num_workers=int(workers),
        worker_root=builder.workers_dir,
        master_dir=builder.master_dir,
        verbose=False,
        cleanup=False,
    )

    # Copy key outputs for inspection
    out_dir = os.path.join(cfg.project_ws, "results", "group")
    os.makedirs(out_dir, exist_ok=True)

    for fname in [
        f"{project}.3.par.csv",
        f"{project}.2.par.csv",
        f"{project}.phi.meas.csv",
        f"{project}.pdc.csv",
        f"{project}.idx.csv",
    ]:
        src = os.path.join(builder.master_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, fname))

    spinup_src = cfg.spinup
    if spinup_src and os.path.exists(spinup_src):
        shutil.copyfile(spinup_src, os.path.join(out_dir, "spinup.json"))

    print(f"Wrote group calibration outputs to {out_dir}")


if __name__ == "__main__":
    # Keep this script simple; edit/select sites as needed for the international run.
    run_group_calibration()

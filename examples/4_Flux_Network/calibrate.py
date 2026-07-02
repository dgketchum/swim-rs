"""Calibration for 4_Flux_Network using PEST++ IES.

Usage:
    python calibrate.py [--sites SITE1,SITE2,...] [--pdc-remove] [--workers N]
"""

import os
import shutil
import tempfile
from pathlib import Path

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "4_Flux_Network.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)

    return cfg


def run_pest_sequence(
    cfg: ProjectConfig,
    results_dir: str,
    pdc_remove: bool = False,
    debug_fields: list[str] | None = None,
    exclude_fields: list[str] | None = None,
    select_fields: list[str] | None = None,
    ies_num_threads: int | None = None,
    container_path: str | None = None,
):
    project = cfg.project_name

    if os.path.isdir(cfg.pest_run_dir):
        shutil.rmtree(cfg.pest_run_dir)
    os.makedirs(cfg.pest_run_dir, exist_ok=False)

    os.makedirs(results_dir, exist_ok=True)

    p_dir = os.path.join(cfg.pest_run_dir, "pest")
    m_dir = os.path.join(cfg.pest_run_dir, "master")
    w_dir = os.path.join(cfg.pest_run_dir, "workers")

    os.chdir(Path(__file__).resolve().parent)

    if container_path is None:
        container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    container = SwimContainer.open(container_path, mode="r")

    builder = PestBuilder(
        cfg,
        container,
        use_existing=False,
        conflicted_obs=None,
    )

    if exclude_fields:
        before = len(builder.plot_order)
        builder.plot_order = [f for f in builder.plot_order if f not in exclude_fields]
        builder.pest_args = builder.get_pest_builder_args()
        print(f"Excluded {before - len(builder.plot_order)} fields: {exclude_fields}")
        print(f"Calibrating {len(builder.plot_order)} fields")

    if select_fields is not None:
        missing = [f for f in select_fields if f not in builder.plot_order]
        if missing:
            raise ValueError(f"Selected fields not in container: {missing}")
        builder.plot_order = select_fields
        builder.pest_args = builder.get_pest_builder_args()
        print(f"Selected {len(select_fields)} fields for calibration")

    if debug_fields is not None:
        missing = [f for f in debug_fields if f not in builder.plot_order]
        if missing:
            raise ValueError(f"Debug fields not in container: {missing}")
        builder.plot_order = debug_fields
        builder.pest_args = builder.get_pest_builder_args()
        print(f"DEBUG: limiting to {len(debug_fields)} fields: {debug_fields}")

    # Spinup must run before build_pest so that _build_swim_input can
    # bake the spinup state into swim_input.h5 for workers.
    builder.spinup(overwrite=True)
    shutil.copyfile(builder.config.spinup, os.path.join(results_dir, "spinup.json"))

    builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
    builder.build_localizer()

    exe_ = "pestpp-ies"

    if pdc_remove:
        builder.write_control_settings(noptmax=-1, reals=5)
    else:
        builder.write_control_settings(noptmax=0)

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
                conflicted_obs=temp_pdc,
            )
            if exclude_fields:
                builder.plot_order = [f for f in builder.plot_order if f not in exclude_fields]
                builder.pest_args = builder.get_pest_builder_args()
            if select_fields is not None:
                builder.plot_order = select_fields
                builder.pest_args = builder.get_pest_builder_args()
            if debug_fields is not None:
                builder.plot_order = debug_fields
                builder.pest_args = builder.get_pest_builder_args()
            builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
            builder.build_localizer()
            builder.write_control_settings(noptmax=0)
            builder.dry_run(exe_)

    reals = 20 if debug_fields else cfg.realizations
    n_workers = min(10, cfg.workers) if debug_fields else cfg.workers
    noptmax = 3
    builder.write_control_settings(noptmax=noptmax, reals=reals, ies_num_threads=ies_num_threads)
    pst_name = f"{project}.pst"
    run_pst(
        p_dir,
        exe_,
        pst_name,
        num_workers=n_workers,
        worker_root=w_dir,
        master_dir=m_dir,
        verbose=False,
        cleanup=False,
    )

    for fname in [
        f"{project}.3.par.csv",
        f"{project}.2.par.csv",
        f"{project}.phi.meas.csv",
        f"{project}.pdc.csv",
        f"{project}.idx.csv",
    ]:
        src = os.path.join(m_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(results_dir, fname))

    shutil.rmtree(p_dir)
    shutil.rmtree(m_dir)
    shutil.rmtree(w_dir)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Calibrate 4_Flux_Network with PEST++ IES")
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated site IDs (debug subset)"
    )
    parser.add_argument(
        "--exclude", type=str, default=None, help="Comma-separated site IDs to exclude"
    )
    parser.add_argument(
        "--pdc-remove", action="store_true", default=False, help="Run PDC removal pass"
    )
    parser.add_argument("--workers", type=int, default=None, help="Override worker count")
    args = parser.parse_args()

    cfg = _load_config()

    debug_fields = None
    if args.sites:
        debug_fields = [s.strip() for s in args.sites.split(",")]

    exclude_fields = None
    if args.exclude:
        exclude_fields = [s.strip() for s in args.exclude.split(",")]

    if args.workers:
        cfg.workers = args.workers

    results = os.path.join(cfg.project_ws, "results")
    t0 = time.time()
    run_pest_sequence(
        cfg,
        results,
        pdc_remove=args.pdc_remove,
        debug_fields=debug_fields,
        exclude_fields=exclude_fields,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")

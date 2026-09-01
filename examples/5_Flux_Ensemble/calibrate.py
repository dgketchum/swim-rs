import glob
import os
import shutil
import tempfile
from pathlib import Path

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _copy_if_exists(src_dirs, name, dst_dir):
    """Copy the first existing `name` found across `src_dirs` into `dst_dir`."""
    for d in src_dirs:
        src = os.path.join(d, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst_dir, name))
            return True
    return False


def _archive_pest_outputs(m_dir, p_dir, results_dir, project, noptmax):
    """Copy the full PEST++ trajectory and problem definition into the
    RUN_POLICY archive BEFORE any working-directory cleanup.

    Category 4 (pest_outputs): every iteration .par.csv/.obs.csv (.0=prior
    through .{noptmax}=posterior), phi histories, .rec, .pdc, final .rei.
    Category 3 (problem_definition, raw): .pst and loc.mat. Decoded tables
    (observation_table, observation_metadata, parameter_bounds) are produced
    post-hoc by archive_run.py.
    """
    arc = os.path.join(results_dir, "archive")
    cat3 = os.path.join(arc, "3_problem_definition")
    cat4 = os.path.join(arc, "4_pest_outputs")
    os.makedirs(cat3, exist_ok=True)
    os.makedirs(cat4, exist_ok=True)

    # Category 4: full optimization trajectory (master dir is authoritative).
    for i in range(0, noptmax + 1):
        _copy_if_exists([m_dir], f"{project}.{i}.par.csv", cat4)
        _copy_if_exists([m_dir], f"{project}.{i}.obs.csv", cat4)
    for name in (
        f"{project}.rec",
        f"{project}.phi.meas.csv",
        f"{project}.phi.actual.csv",
        f"{project}.phi.composite.csv",
        f"{project}.pdc.csv",
        f"{project}.idx.csv",
        f"{project}.{noptmax}.rei",
        f"{project}.rei",
    ):
        _copy_if_exists([m_dir], name, cat4)
    # any other .pdc.csv variants pestpp may emit
    for pdc in glob.glob(os.path.join(m_dir, f"{project}.*.pdc.csv")):
        shutil.copyfile(pdc, os.path.join(cat4, os.path.basename(pdc)))

    # Category 3 (raw): control file, its external sidecar CSVs (so the .pst
    # is self-contained / re-loadable), localizer matrix + summary. The pcf-v2
    # .pst references the *_data.csv files by relative name, so they must sit
    # beside it in the archive.
    _copy_if_exists([m_dir, p_dir], f"{project}.pst", cat3)
    _copy_if_exists([p_dir, m_dir], "loc.mat", cat3)
    _copy_if_exists([p_dir, m_dir], "localizer_summary.json", cat3)
    for sidecar in glob.glob(os.path.join(p_dir, "*_data.csv")):
        name = os.path.basename(sidecar)
        if any(
            name.endswith(s)
            for s in (
                ".pargp_data.csv",
                ".par_data.csv",
                ".obs_data.csv",
                ".tplfile_data.csv",
                ".insfile_data.csv",
            )
        ):
            shutil.copyfile(sidecar, os.path.join(cat3, name))

    print(f"Archived PEST trajectory + problem definition to {arc}")


def _load_config(config_path: str | None = None) -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = Path(config_path) if config_path else project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)

    return cfg


def run_pest_sequence(
    cfg: ProjectConfig,
    results_dir: str,
    pdc_remove: bool = False,
    debug_fields: list[str] | None = None,
    ies_num_threads: int | None = None,
    container_path: str | None = None,
    keep_pestrun: bool = False,
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

    # Guardrail: validate NDVI and ETf masks match mask_mode before calibration
    mask_mode = getattr(cfg, "mask_mode", "irrigation")
    required_masks = ["no_mask"] if mask_mode == "none" else ["irr", "inv_irr"]

    for mask in required_masks:
        ndvi_key = f"remote_sensing/ndvi/landsat/{mask}"
        if ndvi_key not in container._root:
            raise RuntimeError(
                f"Missing Landsat NDVI ({mask}) for mask_mode={mask_mode!r} — cannot calibrate"
            )

    etf_model = cfg.etf_target_model
    models = cfg.etf_ensemble_members if etf_model == "ensemble" else [etf_model]
    for model in models:
        for mask in required_masks:
            etf_key = f"remote_sensing/etf/landsat/{model}/{mask}"
            if etf_key not in container._root:
                raise RuntimeError(
                    f"Missing ETf {model}/{mask} for mask_mode={mask_mode!r} — cannot calibrate"
                )

    builder = PestBuilder(
        cfg,
        container,
        use_existing=False,
        conflicted_obs=None,
    )

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
    builder.export_weight_audit(os.path.join(results_dir, "etf_weight_audit.csv"))
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

    # RUN_POLICY: archive the full trajectory + problem definition BEFORE cleanup.
    _archive_pest_outputs(m_dir, p_dir, results_dir, project, noptmax)

    for fname in [
        f"{project}.{noptmax}.par.csv",
        f"{project}.{noptmax - 1}.par.csv",
        f"{project}.phi.meas.csv",
        f"{project}.pdc.csv",
        f"{project}.idx.csv",
    ]:
        src = os.path.join(m_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(results_dir, fname))

    if keep_pestrun:
        print(f"keep_pestrun=True: leaving {cfg.pest_run_dir} intact")
    else:
        shutil.rmtree(p_dir)
        shutil.rmtree(m_dir)
        shutil.rmtree(w_dir)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Run Ex5 calibration")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Project TOML (default: 5_Flux_Ensemble.toml; E0 arms use variant configs)",
    )
    parser.add_argument(
        "--etf-weighting-mode",
        choices=["spread", "fixed_sd"],
        default=None,
        help="Override etf_weighting_mode from TOML config",
    )
    parser.add_argument(
        "--results-tag",
        default=None,
        help="Results subdirectory name (e.g. 'e1_spread', 'e2_fixed_sd')",
    )
    parser.add_argument(
        "--fixed-sd",
        type=float,
        default=None,
        help="Override etf_weighting_fixed_sd (default 0.33)",
    )
    parser.add_argument(
        "--spread-floor",
        type=float,
        default=None,
        help="Override etf_weighting_spread_floor (default 0.1)",
    )
    parser.add_argument(
        "--min-members",
        type=int,
        default=None,
        help="Override etf_weighting_min_members (default 2)",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help="Override container path",
    )
    parser.add_argument(
        "--etf-target-model",
        type=str,
        default=None,
        help="Override etf_target_model from TOML config (e.g. 'ensemble', 'ensemble_mad')",
    )
    parser.add_argument(
        "--ensemble-source",
        type=str,
        default=None,
        help="Override ensemble_source from TOML config (e.g. 'computed', 'openet')",
    )
    parser.add_argument(
        "--debug-fields",
        type=str,
        default=None,
        help="Comma-separated site IDs for debug subset",
    )
    parser.add_argument(
        "--keep-pestrun",
        action="store_true",
        help="Do not delete the pest/master/workers dirs after archiving "
        "(RUN_POLICY safety: preserve raw PEST outputs for publication runs)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Apply CLI overrides to config
    if args.etf_weighting_mode is not None:
        cfg.etf_weighting_mode = args.etf_weighting_mode
    if args.fixed_sd is not None:
        cfg.etf_weighting_fixed_sd = args.fixed_sd
    if args.spread_floor is not None:
        cfg.etf_weighting_spread_floor = args.spread_floor
    if args.min_members is not None:
        cfg.etf_weighting_min_members = args.min_members
    if args.etf_target_model is not None:
        cfg.etf_target_model = args.etf_target_model
    if args.ensemble_source is not None:
        cfg.ensemble_source = args.ensemble_source

    debug_fields = None
    if args.debug_fields:
        debug_fields = [s.strip() for s in args.debug_fields.split(",")]

    if args.results_tag:
        results = os.path.join(cfg.project_ws, "results", args.results_tag)
    else:
        # Never default into an archived run directory; untagged runs go to scratch.
        results = os.path.join(cfg.project_ws, "results", "scratch")

    print(f"ETf target: {cfg.etf_target_model}")
    print(f"Ensemble source: {cfg.ensemble_source}")
    print(f"Weighting mode: {cfg.etf_weighting_mode}")
    print(f"Results dir: {results}")

    t0 = time.time()
    run_pest_sequence(
        cfg,
        results,
        pdc_remove=False,
        debug_fields=debug_fields,
        container_path=args.container,
        keep_pestrun=args.keep_pestrun,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")

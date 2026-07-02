import os
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _load_config(calibrate: bool = True, conf_path: Path | None = None) -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    if conf_path is None:
        conf_path = project_dir / "6_Flux_International.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf_path), calibrate=calibrate)
    else:
        cfg.read_config(
            str(conf_path), project_root_override=str(project_dir.parent), calibrate=calibrate
        )
    return cfg


def _results_dir(cfg: ProjectConfig, conf_path: Path | None = None) -> str:
    base = os.path.join(cfg.project_ws, "results")
    if conf_path is None:
        return base
    return os.path.join(base, conf_path.stem)


def _group_results_dir(cfg: ProjectConfig, conf_path: Path | None = None) -> str:
    return os.path.join(_results_dir(cfg, conf_path), "group")


def _site_ids(cfg: ProjectConfig, select: list[str] | None = None) -> list[str]:
    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    if cfg.feature_id_col not in gdf.columns:
        raise ValueError(
            f"Feature ID column {cfg.feature_id_col} not found in {cfg.fields_shapefile}"
        )
    ids = sorted(set(gdf[cfg.feature_id_col].astype(str).tolist()))
    if select:
        ids = [i for i in ids if i in set(select)]
    return ids


def export_calibration_bundle(conf_path: Path | None = None) -> None:
    """Export calibration report and parameter CSV to the config-specific results dir.

    Must be run *after* batch_runner has ingested calibration into the container.
    This is a standalone post-ingest step, not part of run_group_calibration(),
    because PEST output must be ingested before the container has anything to export.

    Batch provenance (n_batches, phi convergence) is embedded in the container's
    ``calibration`` group attrs and written into ``calibration.json`` by the report.
    External metadata files (run_manifest.json, batch_log.json) are not copied
    because they live in the mutable pest_run_dir and may not correspond to the
    container state at export time.
    """
    cfg = _load_config(calibrate=True, conf_path=conf_path)
    container_path = getattr(cfg, "container_path", None)
    if container_path is None:
        container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    if not os.path.exists(container_path):
        raise FileNotFoundError(f"Container not found at {container_path}")

    results_dir = _results_dir(cfg, conf_path)
    os.makedirs(results_dir, exist_ok=True)

    container = SwimContainer.open(container_path, mode="r")
    try:
        report = container.calibration_report(output_dir=results_dir)
        df = report.to_dataframe()
        df.to_csv(os.path.join(results_dir, "calibration_parameters.csv"), index=False)
        print(report.summary())
        print(f"\nExported calibration artifacts ({len(df)} fields) to {results_dir}")
    finally:
        container.close()


def run_group_calibration(
    *,
    select_sites: list[str] | None = None,
    workers: int | None = None,
    realizations: int | None = None,
    overwrite: bool = False,
    pdc_remove: bool = False,
    conf_path: Path | None = None,
) -> None:
    cfg = _load_config(calibrate=True, conf_path=conf_path)

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
        cfg,
        container,
        use_existing=False,
        select_fields=sites if select_sites else None,
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
                conflicted_obs=temp_pdc,
                select_fields=sites if select_sites else None,
            )
            builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
            builder.build_localizer()
            builder.write_control_settings(noptmax=0)
            builder.dry_run(exe_)

    # Main run
    reals = int(realizations) if realizations is not None else int(cfg.realizations or 250)
    n_workers = int(workers) if workers is not None else int(getattr(cfg, "workers", 8))
    builder.write_control_settings(noptmax=3, reals=reals)

    pst_name = f"{project}.pst"
    run_pst(
        builder.pest_dir,
        exe_,
        pst_name,
        num_workers=n_workers,
        worker_root=builder.workers_dir,
        master_dir=builder.master_dir,
        verbose=False,
        cleanup=False,
    )

    # Copy key outputs for inspection
    out_dir = _group_results_dir(cfg, conf_path)
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
    config_flag = f" --config {conf_path}" if conf_path else ""
    print(
        "Next: ingest calibration into the container with batch_runner, "
        f"then run: calibrate_group.py{config_flag} --export-calibration"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate 6_Flux_International with PEST++ IES")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config (default: 6_Flux_International.toml)",
    )
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site IDs (subset)")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count")
    parser.add_argument("--realizations", type=int, default=None, help="Override realizations")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Wipe pest_run_dir")
    parser.add_argument("--pdc-remove", action="store_true", default=False, help="Run PDC removal")
    parser.add_argument(
        "--export-calibration",
        action="store_true",
        default=False,
        help="Export calibration artifacts from an already-ingested container (post-ingest step)",
    )
    args = parser.parse_args()

    conf = Path(args.config) if args.config else None

    if args.export_calibration:
        export_calibration_bundle(conf_path=conf)
    else:
        sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
        run_group_calibration(
            select_sites=sites,
            workers=args.workers,
            realizations=args.realizations,
            overwrite=args.overwrite,
            pdc_remove=args.pdc_remove,
            conf_path=conf,
        )

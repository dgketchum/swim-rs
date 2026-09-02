import argparse
import os
import shutil
import sys
import warnings

# Suppress noisy pyemu legacy warning about flopy (not needed for current workflow)
warnings.filterwarnings(
    "ignore",
    message="Failed to import legacy module.*flopy",
    category=UserWarning,
)


# Import helpers: support both installed (swimrs.*) and in-repo (src.swimrs.*)
def _try_import(installed_path: str, dev_path: str, name: str):
    try:
        module = __import__(installed_path, fromlist=[name])
        return getattr(module, name)
    except Exception:
        module = __import__(dev_path, fromlist=[name])
        return getattr(module, name)


ProjectConfig = _try_import("swimrs.swim.config", "src.swimrs.swim.config", "ProjectConfig")
health_report_output_dir = _try_import(
    "swimrs.container.health",
    "src.swimrs.container.health",
    "health_report_output_dir",
)

# Earth Engine utils and exports
is_authorized = _try_import(
    "swimrs.data_extraction.ee.ee_utils", "src.swimrs.data_extraction.ee.ee_utils", "is_authorized"
)
sparse_sample_ndvi = _try_import(
    "swimrs.data_extraction.ee.ndvi_export",
    "src.swimrs.data_extraction.ee.ndvi_export",
    "sparse_sample_ndvi",
)
export_etf = _try_import(
    "swimrs.data_extraction.ee.etf_export",
    "src.swimrs.data_extraction.ee.etf_export",
    "export_etf",
)
sample_snodas_swe = _try_import(
    "swimrs.data_extraction.ee.snodas_export",
    "src.swimrs.data_extraction.ee.snodas_export",
    "sample_snodas_swe",
)
get_irrigation = _try_import(
    "swimrs.data_extraction.ee.ee_props", "src.swimrs.data_extraction.ee.ee_props", "get_irrigation"
)
get_ssurgo = _try_import(
    "swimrs.data_extraction.ee.ee_props", "src.swimrs.data_extraction.ee.ee_props", "get_ssurgo"
)
get_cdl = _try_import(
    "swimrs.data_extraction.ee.ee_props", "src.swimrs.data_extraction.ee.ee_props", "get_cdl"
)
get_landcover = _try_import(
    "swimrs.data_extraction.ee.ee_props", "src.swimrs.data_extraction.ee.ee_props", "get_landcover"
)
sample_era5_land_variables_daily = _try_import(
    "swimrs.data_extraction.ee.ee_era5",
    "src.swimrs.data_extraction.ee.ee_era5",
    "sample_era5_land_variables_daily",
)

# GridMET
assign_gridmet_ids = _try_import(
    "swimrs.data_extraction.gridmet.gridmet",
    "src.swimrs.data_extraction.gridmet.gridmet",
    "assign_gridmet_ids",
)
sample_gridmet_corrections = _try_import(
    "swimrs.data_extraction.gridmet.gridmet",
    "src.swimrs.data_extraction.gridmet.gridmet",
    "sample_gridmet_corrections",
)
download_gridmet = _try_import(
    "swimrs.data_extraction.gridmet.gridmet",
    "src.swimrs.data_extraction.gridmet.gridmet",
    "download_gridmet",
)

# Calibration
PestBuilder = _try_import(
    "swimrs.calibrate.pest_builder", "src.swimrs.calibrate.pest_builder", "PestBuilder"
)
run_pst = _try_import("swimrs.calibrate.run_pest", "src.swimrs.calibrate.run_pest", "run_pst")

SwimContainer = _try_import("swimrs.container", "src.swimrs.container", "SwimContainer")


def _parse_sites_arg(sites: str | None) -> list[str] | None:
    if not sites:
        return None
    parts = [s.strip() for s in sites.split(",") if s.strip()]
    return parts or None


def _resolve_project_root(default_config_path: str, override: str | None) -> str | None:
    if override:
        return os.path.abspath(override)
    # default: directory containing the TOML
    return os.path.dirname(os.path.abspath(default_config_path))


def _resolve_container_path(
    config,
    conf_path: str,
    out_root: str | None,
    override: str | None = None,
) -> str:
    if override:
        return os.path.abspath(override)

    container_path = getattr(config, "container_path", None)
    if container_path:
        return container_path

    data_root = config.data_dir or out_root or os.path.dirname(os.path.abspath(conf_path))
    return os.path.join(data_root, f"{config.project_name or 'swim'}.swim")


def _resolve_calibrated_params_path(
    config, forecast_params: str | None, out_root: str
) -> str | None:
    calibrated_params_path = None
    if not forecast_params:
        return None

    os.makedirs(out_root, exist_ok=True)
    config.forecast_param_csv = forecast_params
    if os.path.isfile(config.forecast_param_csv):
        config.read_forecast_parameters()
        if hasattr(config, "forecast_parameters") and config.forecast_parameters is not None:
            calibrated_params_path = _convert_forecast_params_to_json(
                config.forecast_parameters, out_root
            )
    else:
        print(f"Forecast parameter CSV not found: {config.forecast_param_csv}")

    return calibrated_params_path


def _write_evaluation_csvs(result, out_root: str) -> list[str]:
    import pandas as pd

    os.makedirs(out_root, exist_ok=True)

    write_failures: list[str] = []
    for i, fid in enumerate(result.field_uids):
        df_data = {
            "et_act": result.output.eta[:, i],
            "etref": result.ref_et[:, i],
            "kc_act": result.output.etf[:, i],
            "kc_bas": result.output.kcb[:, i],
            "ks": result.output.ks[:, i],
            "ke": result.output.ke[:, i],
            "melt": result.output.melt[:, i],
            "rain": result.output.rain[:, i],
            "depl_root": result.output.depl_root[:, i],
            "dperc": result.output.dperc[:, i],
            "runoff": result.output.runoff[:, i],
            "swe": result.output.swe[:, i],
            "ppt": result.prcp[:, i],
            "tmin": result.tmin[:, i],
            "tmax": result.tmax[:, i],
            "tavg": (result.tmin[:, i] + result.tmax[:, i]) / 2.0,
            "irrigation": result.output.irr_sim[:, i],
            "gw_sim": result.output.gw_sim[:, i],
        }
        df = pd.DataFrame(df_data, index=result.dates)

        out_csv = os.path.join(out_root, f"{fid}.csv")
        try:
            df.to_csv(out_csv)
            print(f"Wrote {out_csv}")
        except Exception as e:
            print(f"Failed to write {out_csv}: {e}")
            write_failures.append(fid)

    return write_failures


def _ensure_shapefile(fields_path: str, conf_path: str, out_root: str | None) -> str | None:
    """Ensure the shapefile exists, copying from TOML-relative data/gis if needed."""
    if fields_path and os.path.exists(fields_path):
        return fields_path

    basename = os.path.basename(fields_path) if fields_path else None
    if not basename:
        return None

    conf_dir = os.path.dirname(os.path.abspath(conf_path))
    source_dir = os.path.join(conf_dir, "data", "gis")
    source_files = {
        ext: os.path.join(source_dir, f"{os.path.splitext(basename)[0]}.{ext}")
        for ext in ("shp", "shx", "dbf", "prj", "cpg")
    }

    # Only proceed if the source .shp exists
    if not os.path.exists(source_files["shp"]):
        return None

    # Copy into out_root/data/gis if out_root provided; otherwise copy alongside config
    target_root = out_root or conf_dir
    target_dir = os.path.join(target_root, "data", "gis")
    os.makedirs(target_dir, exist_ok=True)

    target_base = os.path.join(target_dir, os.path.splitext(basename)[0])
    for ext, src in source_files.items():
        if os.path.exists(src):
            shutil.copy2(src, f"{target_base}.{ext}")

    return f"{target_base}.shp"


def cmd_extract(args: argparse.Namespace) -> int:
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root)

    # Earth Engine auth gate for all EE extraction tasks
    try:
        is_authorized()
    except Exception as e:
        print(f"Earth Engine authorization check failed: {e}")
        return 1

    export_dest = args.export
    bucket_arg = args.bucket or getattr(config, "ee_bucket", None)
    file_prefix = args.file_prefix
    if export_dest == "bucket" and not bucket_arg:
        print("Export destination set to bucket, but no --bucket or config.ee_bucket provided")
        return 2
    failures: list[str] = []

    # 1) SNODAS SWE (builds EE FeatureCollection from fields shapefile by default)
    if not args.no_snodas:
        try:
            sample_snodas_swe(
                feature_coll=config.fields_shapefile,
                bucket=bucket_arg,
                debug=False,
                check_dir=None,
                feature_id=config.feature_id_col,
                dest=export_dest,
                drive_folder="swim",
                file_prefix=file_prefix,
                drive_categorize=args.drive_categorize,
            )
        except Exception as e:
            print(f"SNODAS export error: {e}")
            failures.append("snodas")

    # 2) Properties (CDL, irrigation fraction, SSURGO, landcover)
    if not args.no_properties:
        try:
            project = config.project_name or "swim"
            get_cdl(
                config.fields_shapefile,
                f"{project}_cdl",
                selector=config.feature_id_col,
                dest=export_dest,
                bucket=bucket_arg,
                drive_folder="swim",
                file_prefix=file_prefix,
                drive_categorize=args.drive_categorize,
            )
            get_irrigation(
                config.fields_shapefile,
                f"{project}_irr",
                debug=True,
                selector=config.feature_id_col,
                lanid=True,
                dest=export_dest,
                bucket=bucket_arg,
                drive_folder="swim",
                file_prefix=file_prefix,
                drive_categorize=args.drive_categorize,
            )
            get_ssurgo(
                config.fields_shapefile,
                f"{project}_ssurgo",
                debug=False,
                selector=config.feature_id_col,
                dest=export_dest,
                bucket=bucket_arg,
                drive_folder="swim",
                file_prefix=file_prefix,
                drive_categorize=args.drive_categorize,
            )
            get_landcover(
                config.fields_shapefile,
                f"{project}_landcover",
                debug=False,
                selector=config.feature_id_col,
                out_fmt="CSV",
                dest=export_dest,
                bucket=bucket_arg,
                drive_folder="swim",
                drive_categorize=args.drive_categorize,
                file_prefix=file_prefix,
            )
        except Exception as e:
            print(f"Properties export error: {e}")
            failures.append("properties")

    # 3) Remote sensing NDVI (and optionally Sentinel & ETF models)
    if not args.no_rs:
        try:
            masks = ["irr", "inv_irr"]
            years = list(range(config.start_dt.year, config.end_dt.year + 1))
            for m in masks:
                landsat_check = os.path.join(config.landsat_dir or "", "extracts", "ndvi", m)
                sparse_sample_ndvi(
                    config.fields_shapefile,
                    bucket=bucket_arg,
                    debug=False,
                    mask_type=m,
                    check_dir=landsat_check,
                    start_yr=years[0],
                    end_yr=years[-1],
                    feature_id=config.feature_id_col,
                    satellite="landsat",
                    state_col=config.state_col,
                    select=_parse_sites_arg(args.sites),
                    dest=export_dest,
                    drive_folder="swim",
                    file_prefix=file_prefix,
                    drive_categorize=args.drive_categorize,
                )
                if args.add_sentinel:
                    sentinel_check = os.path.join(config.sentinel_dir or "", "extracts", "ndvi", m)
                    sentinel_start = max(2017, years[0])
                    sparse_sample_ndvi(
                        config.fields_shapefile,
                        bucket=bucket_arg,
                        debug=False,
                        mask_type=m,
                        check_dir=sentinel_check,
                        start_yr=sentinel_start,
                        end_yr=years[-1],
                        feature_id=config.feature_id_col,
                        satellite="sentinel",
                        state_col=config.state_col,
                        select=_parse_sites_arg(args.sites),
                        dest=export_dest,
                        drive_folder="swim",
                        file_prefix=file_prefix,
                        drive_categorize=args.drive_categorize,
                    )

            # Optional ETF models (using OpenET FOSS packages)
            if args.etf_models:
                models = [m.strip() for m in args.etf_models.split(",") if m.strip()]
                clustered = not args.sparse  # default to clustered unless --sparse
                for m in masks:
                    for model in models:
                        etf_check = os.path.join(
                            config.landsat_dir or "", "extracts", f"{model}_etf", m
                        )
                        export_etf(
                            shapefile=config.fields_shapefile,
                            model=model,
                            feature_id=config.feature_id_col,
                            select=_parse_sites_arg(args.sites),
                            start_yr=max(2016, years[0]),
                            end_yr=years[-1],
                            mask_type=m,
                            check_dir=etf_check,
                            state_col=config.state_col,
                            dest=export_dest,
                            bucket=bucket_arg,
                            drive_folder="swim",
                            file_prefix=file_prefix,
                            clustered=clustered,
                        )
        except Exception as e:
            print(f"Remote sensing export error: {e}")
            failures.append("remote_sensing")

    # 4) Meteorology: GridMET or ERA5-Land based on config.met_source
    met_source = getattr(config, "met_source", "gridmet")
    if args.no_met:
        print("Skipping meteorology download (--no-met).")
    elif met_source == "gridmet":
        try:
            # Assign GFIDs (optionally from centroids), optionally sample corrections
            gridmet_points = (
                config.gridmet_centroids if getattr(args, "use_gridmet_centroids", False) else None
            )
            join_path = config.gridmet_mapping_shp
            fields_joined = assign_gridmet_ids(
                fields=config.fields_shapefile,
                fields_join=join_path,
                gridmet_points=gridmet_points,
                field_select=_parse_sites_arg(args.sites),
                feature_id=config.feature_id_col,
                gridmet_id_col=config.gridmet_mapping_index_col or "GFID",
            )

            factors_path = None
            if getattr(args, "gridmet_correction", False) and config.correction_tifs:
                factors_path = config.gridmet_factors
                sample_gridmet_corrections(
                    fields_join=join_path,
                    gridmet_ras=config.correction_tifs,
                    factors_js=factors_path,
                    gridmet_id_col=config.gridmet_mapping_index_col or "GFID",
                )
            download_gridmet(
                join_path,
                factors_path,
                config.met_dir,
                start=str(config.start_dt.date()),
                end=str(config.end_dt.date()),
                overwrite=False,
                append=True,
                target_fields=_parse_sites_arg(args.sites),
                feature_id=config.gridmet_mapping_index_col,
            )
        except Exception as e:
            print(f"GridMET error: {e}")
            failures.append("meteorology_gridmet")
    elif met_source == "era5":
        if export_dest != "bucket":
            print("ERA5-Land export requires --export=bucket and a configured bucket.")
            return 3
        try:
            sample_era5_land_variables_daily(
                shapefile=config.fields_shapefile,
                bucket=bucket_arg,
                debug=False,
                check_dir=config.era5_extracts_dir,
                overwrite=False,
                start_yr=config.start_dt.year,
                end_yr=config.end_dt.year,
                feature_id_col=config.feature_id_col,
                file_prefix=file_prefix,
            )
        except Exception as e:
            print(f"ERA5-Land export error: {e}")
            failures.append("meteorology_era5")
    else:
        print(f"Unknown met_source '{met_source}' in config")
        failures.append("meteorology_source")

    if failures:
        print(f"Extract failed for stage(s): {', '.join(failures)}")
        return 1

    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root, calibrate=True)

    # Resolve container path (same pattern as cmd_prep/cmd_evaluate)
    container_path = getattr(config, "container_path", None)
    if not container_path:
        data_root = config.data_dir or out_root or os.path.dirname(os.path.abspath(conf_path))
        container_path = os.path.join(data_root, f"{config.project_name or 'swim'}.swim")

    if not os.path.exists(container_path):
        print(f"Container not found: {container_path}")
        print("Run 'swim prep' first to create the container.")
        return 1

    # Open container (read-only is sufficient for calibration data access)
    try:
        container = SwimContainer.open(container_path, mode="r")
    except Exception as e:
        print(f"Failed to open container: {e}")
        return 1

    # Build and run PEST++
    try:
        builder = PestBuilder(
            config,
            container,
            use_existing=False,
        )
        builder.spinup(
            overwrite=True
        )  # CRITICAL: before build_pest (bakes spinup into swim_input.h5)
        builder.build_pest(target_etf=config.etf_target_model, members=config.etf_ensemble_members)
        builder.build_localizer()

        reals = int(args.realizations) if args.realizations else (config.realizations or 250)
        builder.write_control_settings(noptmax=3, reals=reals)

        exe_ = "pestpp-ies"
        project = config.project_name
        p_dir = os.path.join(config.pest_run_dir, "pest")
        m_dir = os.path.join(config.pest_run_dir, "master")
        w_dir = os.path.join(config.pest_run_dir, "workers")
        pst_name = f"{project}.pst"

        run_pst(
            p_dir,
            exe_,
            pst_name,
            num_workers=int(args.workers),
            worker_root=w_dir,
            master_dir=m_dir,
            verbose=False,
            cleanup=False,
        )
    except Exception as e:
        print(f"Calibration run failed: {e}")
        return 1
    finally:
        try:
            container.close()
        except Exception:
            pass

    return 0


def cmd_calibrate_batch(args: argparse.Namespace) -> int:
    """Dispatch batch calibration actions via the native batch runner."""
    from swimrs.calibrate.batch_runner import (
        build_batch,
        calibrate_all,
        cleanup_failed,
        ingest_all,
        ingest_batch,
        prep,
        resolve_context,
        run_batch,
        show_status,
    )

    ctx = resolve_context(
        args.config,
        container_override=args.container,
        output_override=args.output,
        shapefile_override=args.shapefile,
        batch_size=args.batch_size,
        workers=args.workers,
        realizations=args.reals,
        noptmax=args.noptmax,
    )

    action = args.action

    if action == "prep":
        prep(
            ctx,
            exclude_uncovered=args.exclude_uncovered,
            skip_fids_path=args.skip_fids,
            override=args.override,
            skip_health=args.skip_health,
        )
    elif action == "status":
        show_status(ctx)
    elif action == "cleanup-failed":
        cleanup_failed(ctx)
    elif action == "calibrate-all":
        n_failed = calibrate_all(
            ctx,
            resume=args.resume,
            override=args.override,
            skip_health=args.skip_health,
            exclude_uncovered=args.exclude_uncovered,
            skip_fids_path=args.skip_fids,
        )
        return min(n_failed, 1)
    elif action == "ingest-all":
        from swimrs.calibrate.batch_support import persist_calibration_resolved_state

        ingest_all(ctx)
        persist_calibration_resolved_state(
            ctx.container_path,
            ctx.toml_path,
            ctx.output_root,
            command="calibrate-batch --action ingest-all",
        )
    elif action == "ingest-batch":
        if args.batch_id is None:
            print("Error: --batch-id required for ingest-batch")
            return 1
        ingest_batch(ctx, args.batch_id)
    elif action == "run-batch-task":
        from swimrs.calibrate.batch_runner import run_batch_task

        if args.batch_id is None:
            print("Error: --batch-id required for run-batch-task")
            return 1
        return run_batch_task(ctx, args.batch_id, rebuild=args.rebuild)
    elif action == "run-batch":
        if args.batch_id is None:
            print("Error: --batch-id required for run-batch")
            return 1
        from pathlib import Path

        batch_dir = Path(ctx.output_root) / f"batch_{args.batch_id:03d}"
        if not batch_dir.exists():
            print(f"Batch directory not found: {batch_dir}")
            return 1
        run_batch(batch_dir, num_workers=ctx.workers)
    elif action == "run-all":
        from pathlib import Path

        from swimrs.calibrate.batch_support import find_par_csv

        batch_dirs = sorted(Path(ctx.output_root).glob("batch_*"))
        if not batch_dirs:
            print(f"No batch directories found in {ctx.output_root}")
            return 1
        print(f"Running {len(batch_dirs)} batches sequentially...")
        for bd in batch_dirs:
            if args.resume and find_par_csv(bd) is not None:
                print(f"\n=== {bd.name} === SKIP (has .par.csv)")
                continue
            print(f"\n=== {bd.name} ===")
            run_batch(bd, num_workers=ctx.workers)
    elif action == "build-all":
        from pathlib import Path

        from swimrs.calibrate.batch_support import load_batches_from_manifest, partition_fields

        manifest_path = Path(ctx.output_root) / "batch_manifest.csv"
        if manifest_path.exists():
            batches = load_batches_from_manifest(ctx.output_root, ctx.feature_id_col)
        else:
            raw = partition_fields(
                ctx.grouping_shapefile or ctx.fields_shapefile,
                ctx.feature_id_col,
                ctx.batch_size,
                grouping_column=ctx.grouping_column,
            )
            batches = list(enumerate(raw))
        print(f"Building {len(batches)} batches...")
        for batch_id, batch_fids in batches:
            print(f"\n--- Batch {batch_id:03d} ({len(batch_fids)} fields) ---")
            build_batch(ctx, batch_fids, batch_id)
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Build model-ready inputs using SwimContainer (ingest → compute → export)."""
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root)

    # Ensure shapefile exists (copy from TOML-relative data/gis if missing under out-dir)
    resolved_shp = _ensure_shapefile(config.fields_shapefile, conf_path, out_root)
    if resolved_shp is None or not os.path.exists(resolved_shp):
        print(f"Fields shapefile not found: {config.fields_shapefile}")
        print("Place the shapefile under the configured root or alongside the TOML at data/gis/.")
        return 1
    config.fields_shapefile = resolved_shp

    container_path = getattr(config, "container_path", None)
    if not container_path:
        data_root = config.data_dir or out_root or os.path.dirname(os.path.abspath(conf_path))
        container_path = os.path.join(data_root, f"{config.project_name or 'swim'}.swim")

    try:
        if os.path.exists(container_path):
            container = SwimContainer.open(container_path, mode="a")
        else:
            container = SwimContainer.create(
                container_path,
                fields_shapefile=config.fields_shapefile,
                uid_column=config.feature_id_col,
                start_date=str(config.start_dt.date()),
                end_date=str(config.end_dt.date()),
                project_name=config.project_name,
            )
    except Exception as e:
        print(f"Failed to open/create container: {e}")
        return 1

    sites = _parse_sites_arg(args.sites)
    use_lulc = bool(args.use_lulc_irr or args.international)
    masks = ("no_mask",) if use_lulc else ("irr", "inv_irr")
    instruments = ["landsat"]
    failures: list[str] = []

    try:
        # Properties
        try:
            container.ingest.properties(
                lulc_csv=config.lulc_csv,
                soils_csv=config.ssurgo_csv,
                irr_csv=config.irr_csv,
                uid_column=config.feature_id_col,
                overwrite=args.overwrite,
            )
            print("Ingested properties")
        except Exception as e:
            print(f"Properties ingest failed: {e}")
            failures.append("properties")

        # NDVI
        if not args.no_ndvi:
            for mask in masks:
                ndvi_dir = os.path.join(config.landsat_dir or "", "extracts", "ndvi", mask)
                if os.path.isdir(ndvi_dir):
                    try:
                        container.ingest.ndvi(
                            ndvi_dir,
                            uid_column=config.feature_id_col,
                            instrument="landsat",
                            mask=mask,
                            fields=sites,
                            overwrite=args.overwrite,
                        )
                        print(f"Ingested Landsat NDVI ({mask})")
                    except Exception as e:
                        print(f"Landsat NDVI ingest failed ({mask}): {e}")
                        failures.append(f"ndvi_landsat_{mask}")
                if args.add_sentinel and not args.landsat_only_ndvi:
                    s2_dir = os.path.join(config.sentinel_dir or "", "extracts", "ndvi", mask)
                    if os.path.isdir(s2_dir):
                        try:
                            container.ingest.ndvi(
                                s2_dir,
                                uid_column=config.feature_id_col,
                                instrument="sentinel",
                                mask=mask,
                                fields=sites,
                                overwrite=args.overwrite,
                            )
                            if "sentinel" not in instruments:
                                instruments.append("sentinel")
                            print(f"Ingested Sentinel NDVI ({mask})")
                        except Exception as e:
                            print(f"Sentinel NDVI ingest failed ({mask}): {e}")
                            failures.append(f"ndvi_sentinel_{mask}")

        # ETF
        if not args.no_etf:
            etf_models = [
                m for m in [config.etf_target_model] + (config.etf_ensemble_members or []) if m
            ]
            for model in etf_models:
                for mask in masks:
                    etf_dir = os.path.join(
                        config.landsat_dir or "", "extracts", f"{model}_etf", mask
                    )
                    if os.path.isdir(etf_dir):
                        try:
                            container.ingest.etf(
                                etf_dir,
                                uid_column=config.feature_id_col,
                                model=model,
                                mask=mask,
                                instrument="landsat",
                                fields=sites,
                                overwrite=args.overwrite,
                            )
                            print(f"Ingested ETf {model} ({mask})")
                        except Exception as e:
                            print(f"ETf ingest failed ({model}, {mask}): {e}")
                            failures.append(f"etf_{model}_{mask}")

        # Meteorology
        if not args.no_met:
            met_source = getattr(config, "met_source", "gridmet")
            if met_source == "gridmet":
                try:
                    container.ingest.gridmet(
                        config.met_dir,
                        grid_shapefile=config.gridmet_mapping_shp,
                        uid_column=config.feature_id_col,
                        grid_column=config.gridmet_id_col
                        or config.gridmet_mapping_index_col
                        or "GFID",
                        overwrite=args.overwrite,
                    )
                    print("Ingested GridMET")
                except Exception as e:
                    print(f"GridMET ingest failed: {e}")
                    failures.append("meteorology_gridmet")
            elif met_source == "era5":
                try:
                    container.ingest.era5(
                        config.met_dir,
                        overwrite=args.overwrite,
                    )
                    print("Ingested ERA5-Land")
                except Exception as e:
                    print(f"ERA5 ingest failed: {e}")
                    failures.append("meteorology_era5")
            else:
                print(f"Unsupported met_source '{met_source}'")
                failures.append("meteorology_source")

        # SNODAS (optional)
        if not args.no_snow and getattr(config, "snow_source", "snodas") == "snodas":
            try:
                container.ingest.snodas(
                    config.snodas_in_dir,
                    uid_column=config.feature_id_col,
                    fields=sites,
                    overwrite=args.overwrite,
                )
                print("Ingested SNODAS")
            except Exception as e:
                print(f"SNODAS ingest failed: {e}")
                failures.append("snow")

        # Derived products
        try:
            container.compute.merged_ndvi(
                masks=masks,
                instruments=tuple(instruments),
                overwrite=args.overwrite,
            )
            print("Computed merged NDVI")
        except Exception as e:
            print(f"Merged NDVI compute failed: {e}")
            failures.append("merged_ndvi")

        try:
            container.compute.dynamics(
                etf_model=config.etf_target_model or "ssebop",
                irr_threshold=config.irrigation_threshold or 0.1,
                masks=masks,
                instruments=tuple(instruments),
                use_mask=not use_lulc,
                use_lulc=use_lulc,
                met_source=getattr(config, "met_source", "gridmet"),
                fields=sites,
                overwrite=args.overwrite,
            )
            print("Computed dynamics")
        except Exception as e:
            print(f"Dynamics compute failed: {e}")
            failures.append("dynamics")

        # Post-build health check
        if not args.skip_health and not failures:
            print("\n--- Health Check ---")
            try:
                health_config = {}
                use_lulc_mode = bool(args.use_lulc_irr or args.international)
                health_config["mask_mode"] = "no_mask" if use_lulc_mode else "irrigation"
                if config.etf_target_model:
                    health_config["etf_target_model"] = config.etf_target_model
                if config.etf_ensemble_members:
                    health_config["etf_ensemble_members"] = config.etf_ensemble_members
                met_source = getattr(config, "met_source", "gridmet")
                if met_source:
                    health_config["met_source"] = met_source
                snow_source = getattr(config, "snow_source", None)
                if snow_source:
                    health_config["snow_source"] = snow_source

                container.report(
                    config=health_config,
                    output_dir=str(health_report_output_dir(container.uri)),
                    health_profile="calibration",
                )
            except Exception as e:
                print(f"Health check failed: {e}")

    finally:
        try:
            container.close()
        except Exception:
            pass

    if failures:
        print(f"Prep failed for stage(s): {', '.join(failures)}")
        return 1

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a .swim container file."""
    from swimrs.container import SwimContainer

    container_path = args.container
    if not os.path.exists(container_path):
        print(f"Container not found: {container_path}")
        return 1

    container = SwimContainer.open(container_path)
    try:
        print(container.query.status(detailed=args.detailed))
    finally:
        container.close()

    return 0


def cmd_project(args: argparse.Namespace) -> int:
    """Build a hindcast or forecast run container from a calibrated source."""
    from swimrs.container.project import create_run_container

    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root)

    mode = args.mode
    scenarios = [args.scenario] if args.scenario else None

    if mode == "forecast" and not scenarios:
        # Use all scenarios from config
        scenarios = getattr(config, "forecast_scenarios", None)
        if not scenarios:
            print("No --scenario provided and no forecast.scenarios in TOML")
            return 1

    try:
        if mode == "hindcast":
            create_run_container(
                config,
                mode="hindcast",
                overwrite=args.overwrite,
                skip_health=args.skip_health,
            )
        elif mode == "forecast":
            for scenario in scenarios:
                print(f"\n=== Scenario: {scenario} ===")
                create_run_container(
                    config,
                    mode="forecast",
                    scenario=scenario,
                    overwrite=args.overwrite,
                    skip_health=args.skip_health,
                )
        else:
            print(f"Unknown mode: {mode}")
            return 1
    except Exception as e:
        print(f"Project build failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run simulation and write per-site output CSVs using the process package."""
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    forecast_flag = bool(args.forecast_params)
    config.read_config(conf_path, project_root_override=out_root, forecast=forecast_flag)

    container_path = _resolve_container_path(config, conf_path, out_root)
    if not os.path.exists(container_path):
        print(f"Container not found: {container_path}")
        print("Run 'swim prep' first to create the container.")
        return 1

    spinup_path = args.spinup or getattr(config, "spinup", None)
    calibrated_params_path = _resolve_calibrated_params_path(config, args.forecast_params, out_root)

    try:
        container = SwimContainer.open(container_path)
    except Exception as e:
        print(f"Failed to open container: {e}")
        return 1

    try:
        fields = _parse_sites_arg(args.sites)
        result = container.run(
            run_id="evaluate",
            profile="core",
            persist=False,
            engine="fast",
            spinup_json_path=spinup_path,
            calibrated_params_path=calibrated_params_path,
            refet_type=getattr(config, "refet_type", "eto") or "eto",
            etf_model=getattr(config, "etf_target_model", "ssebop"),
            met_source=getattr(config, "met_source", "gridmet"),
            fields=fields,
            mask_mode=getattr(config, "mask_mode", "irrigation"),
            max_irr_rate=getattr(config, "max_irr_rate", 100.0) or 100.0,
        )
        print(
            f"Running daily loop for {len(result.field_uids)} site(s) "
            f"over {len(result.dates)} day(s)..."
        )
        write_failures = _write_evaluation_csvs(result, out_root)

        if write_failures:
            print(f"Evaluation output write failed for site(s): {', '.join(write_failures)}")
            return 1

    except Exception as e:
        print(f"Evaluation run failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        container.close()
        if calibrated_params_path and os.path.exists(calibrated_params_path):
            try:
                os.remove(calibrated_params_path)
            except Exception:
                pass

    return 0


def _load_kc_max_overrides_from_config(config):
    """Resolve the optional [crop_coefficient] kc_max_overrides JSON into a dict."""
    from swimrs.process.input import load_kc_max_overrides

    return load_kc_max_overrides(getattr(config, "kc_max_overrides", None))


def cmd_run(args: argparse.Namespace) -> int:
    """Run the model and persist outputs in the target container."""
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    forecast_flag = bool(args.forecast_params)
    config.read_config(conf_path, project_root_override=out_root, forecast=forecast_flag)

    container_path = _resolve_container_path(config, conf_path, out_root, override=args.container)
    if not os.path.exists(container_path):
        print(f"Container not found: {container_path}")
        return 1

    spinup_path = args.spinup or getattr(config, "spinup", None)
    calibrated_params_path = _resolve_calibrated_params_path(config, args.forecast_params, out_root)

    try:
        container = SwimContainer.open(container_path, mode="r+")
    except Exception as e:
        print(f"Failed to open container: {e}")
        return 1

    try:
        result = container.run(
            run_id=args.run_id,
            profile=args.profile,
            overwrite=args.overwrite,
            restart_from=args.restart_from,
            spinup_json_path=spinup_path,
            calibrated_params_path=calibrated_params_path,
            start_date=args.start_date,
            end_date=args.end_date,
            refet_type=getattr(config, "refet_type", "eto") or "eto",
            etf_model=getattr(config, "etf_target_model", "ssebop"),
            met_source=getattr(config, "met_source", "gridmet"),
            fields=_parse_sites_arg(args.sites),
            kc_max_overrides=_load_kc_max_overrides_from_config(config),
            mask_mode=getattr(config, "mask_mode", "irrigation"),
            ndvi_mode=args.ndvi_mode,
            max_irr_rate=getattr(config, "max_irr_rate", 100.0) or 100.0,
            command=" ".join(sys.argv),
        )
        container.save()
        run_meta = container.runs.metadata(result.run_id)
        print(f"Persisted run: {result.run_id}")
        print(f"  Container: {container_path}")
        print(f"  Path: simulation/runs/{result.run_id}")
        print(f"  Profile: {run_meta.get('profile')}")
        print(f"  Days: {run_meta.get('n_days')}")
        print(f"  Fields: {run_meta.get('field_count')}")
    except Exception as e:
        print(f"Run failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        container.close()
        if calibrated_params_path and os.path.exists(calibrated_params_path):
            try:
                os.remove(calibrated_params_path)
            except Exception:
                pass

    return 0


def _convert_forecast_params_to_json(forecast_params, out_dir: str) -> str:
    """Convert forecast_parameters Series to JSON format for build_swim_input.

    The forecast_parameters Series has index like 'kc_max_FID1', 'ndvi_k_FID1', etc.
    We convert to: {FID1: {kc_max: val, ndvi_k: val, ...}, ...}
    """
    import json
    import tempfile

    params_by_fid = {}
    for param_name in forecast_params.index:
        # Parse param name: expect format like 'kc_max_FID1' or 'ndvi_k_FID1'
        parts = param_name.rsplit("_", 1)
        if len(parts) == 2:
            base_param, fid = parts
            if fid not in params_by_fid:
                params_by_fid[fid] = {}
            params_by_fid[fid][base_param] = float(forecast_params[param_name])

    # Write to temp JSON file
    fd, json_path = tempfile.mkstemp(suffix=".json", prefix="calib_params_", dir=out_dir)
    os.close(fd)
    with open(json_path, "w") as f:
        json.dump(params_by_fid, f)

    return json_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swim",
        description="SWIM-RS workflow CLI: extract -> prep -> calibrate -> evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="store_true", help="Print version and exit")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument(
            "config",
            help="Path to project TOML (e.g., examples/5_Flux_Ensemble/5_Flux_Ensemble.toml)",
        )
        sp.add_argument(
            "--out-dir",
            default=None,
            help="Override project root for outputs; defaults to the directory containing the TOML",
        )
        sp.add_argument(
            "--workers",
            type=int,
            default=6,
            help="Worker count for parallelizable steps (e.g., dynamics, calibration)",
        )
        sp.add_argument(
            "--sites",
            default=None,
            help="Comma-separated site IDs to restrict processing; default processes all sites",
        )

    # extract
    pe = sub.add_parser(
        "extract",
        help="Run data extraction (Earth Engine + GridMET)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Exports SNODAS, properties (CDL/irrigation/soils/landcover), NDVI/ETF, and GridMET time series.",
    )
    add_common(pe)
    pe.add_argument(
        "--add-sentinel",
        action="store_true",
        help="Also export Sentinel-2 NDVI (>=2017). Default: off",
    )
    pe.add_argument(
        "--etf-models",
        default=None,
        help="Comma-separated ETF models to export (options: ptjpl, ssebop, sims, geesebal). Requires swimrs[openet].",
    )
    pe.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse ETf export (one task per field-year). Default is clustered (one task per year).",
    )
    pe.add_argument(
        "--no-snodas", action="store_true", help="Skip SNODAS SWE extraction (default: run)"
    )
    pe.add_argument(
        "--no-properties",
        action="store_true",
        help="Skip CDL/irrigation/soils/landcover extraction (default: run)",
    )
    pe.add_argument(
        "--no-rs",
        action="store_true",
        help="Skip remote sensing (NDVI/ETF) extraction (default: run)",
    )
    pe.add_argument(
        "--no-met", action="store_true", help="Skip meteorology download (GridMET or ERA5-Land)"
    )
    pe.add_argument(
        "--export",
        choices=["drive", "bucket"],
        default="drive",
        help="Earth Engine export destination",
    )
    pe.add_argument(
        "--bucket", default=None, help="Cloud Storage bucket when --export=bucket (e.g., my-bucket)"
    )
    pe.add_argument(
        "--drive-categorize",
        action="store_true",
        help="Place Drive exports into per-category folders (e.g., swim_properties, swim_ndvi)",
    )
    pe.add_argument(
        "--file-prefix",
        default="swim",
        help="Prefix path under the bucket for exports (dest=bucket)",
    )
    pe.add_argument(
        "--use-gridmet-centroids",
        action="store_true",
        help="Assign GridMET GFIDs using provided centroids shapefile (paths.gis.gridmet_centroids)",
    )
    pe.add_argument(
        "--gridmet-correction",
        action="store_true",
        help="Sample GridMET correction rasters (paths.conus.correction_tifs) when mapping GFIDs",
    )
    pe.set_defaults(func=cmd_extract)

    # prep (container-based)
    pp = sub.add_parser(
        "prep",
        help="Ingest data into SwimContainer and compute dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Ingest properties/RS/met into a .swim container and compute dynamics.",
    )
    add_common(pp)
    pp.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing datasets in the container"
    )
    pp.add_argument("--no-ndvi", action="store_true", help="Skip NDVI ingestion")
    pp.add_argument(
        "--landsat-only-ndvi",
        action="store_true",
        help="Force Landsat-only NDVI (skip Sentinel even if present and export uses Landsat NDVI)",
    )
    pp.add_argument("--no-etf", action="store_true", help="Skip ETf ingestion")
    pp.add_argument("--no-met", action="store_true", help="Skip meteorology ingestion")
    pp.add_argument("--no-snow", action="store_true", help="Skip SNODAS ingestion")
    pp.add_argument(
        "--add-sentinel", action="store_true", help="Ingest Sentinel-2 NDVI if available"
    )
    pp.add_argument(
        "--use-lulc-irr",
        action="store_true",
        help="Use LULC-based irrigation detection (no masks) instead of mask-based (CONUS)",
    )
    pp.add_argument(
        "--international",
        action="store_true",
        help="Alias for LULC-based irrigation detection with no-mask NDVI/ETf (non-CONUS workflows)",
    )
    pp.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip post-build health check (runs by default)",
    )
    pp.set_defaults(func=cmd_prep)

    # calibrate
    pc = sub.add_parser(
        "calibrate",
        help="Build and run calibration with PEST++ IES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Builds a PEST++ project, runs spinup (noptmax=0), then IES (noptmax=3). Uses workers for parallel execution.",
    )
    add_common(pc)
    pc.add_argument(
        "--realizations",
        type=int,
        default=None,
        help="Override number of realizations; uses config value if set, otherwise 250",
    )
    pc.set_defaults(func=cmd_calibrate)

    # inspect
    pi = sub.add_parser(
        "inspect",
        help="Inspect a .swim container file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Show contents and status of a .swim container file.",
    )
    pi.add_argument("container", help="Path to .swim container file")
    pi.add_argument(
        "--detailed", action="store_true", help="Show detailed status with provenance log"
    )
    pi.set_defaults(func=cmd_inspect)

    # project (run container factory)
    pj = sub.add_parser(
        "project",
        help="Build hindcast or forecast run container from calibrated source",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Creates a new container with a different date range, copying calibrated "
        "parameters from the source container and ingesting time-varying data for "
        "the target period.",
    )
    add_common(pj)
    pj.add_argument(
        "--mode",
        required=True,
        choices=["hindcast", "forecast"],
        help="Container type to build",
    )
    pj.add_argument(
        "--scenario",
        default=None,
        help="Forecast scenario (e.g. rcp85_cesm). If omitted for forecast mode, "
        "builds all scenarios listed in the TOML.",
    )
    pj.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run container",
    )
    pj.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip post-build health check",
    )
    pj.set_defaults(func=cmd_project)

    # run
    pr = sub.add_parser(
        "run",
        help="Run the model and persist outputs inside a container",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runs SWIM from an existing container and stores outputs under simulation/runs/<run_id>.",
    )
    add_common(pr)
    pr.add_argument(
        "--container",
        default=None,
        help="Explicit container path override; defaults to config.container_path",
    )
    pr.add_argument(
        "--run-id",
        default=None,
        help="Run identifier to store under simulation/runs/<run_id>; defaults to a timestamped id",
    )
    pr.add_argument(
        "--profile",
        choices=["core", "full", "state_only"],
        default="core",
        help="Persistence profile for stored outputs",
    )
    pr.add_argument(
        "--restart-from",
        default=None,
        help="Persisted run id to use as the initial state",
    )
    pr.add_argument("--spinup", default=None, help="Path to spinup JSON (optional)")
    pr.add_argument(
        "--forecast-params", default=None, help="Path to forecast parameter CSV (optional)"
    )
    pr.add_argument(
        "--start-date",
        default=None,
        help="Optional simulation start date override (YYYY-MM-DD)",
    )
    pr.add_argument(
        "--end-date",
        default=None,
        help="Optional simulation end date override (YYYY-MM-DD)",
    )
    pr.add_argument(
        "--ndvi-mode",
        choices=["observed", "climatology"],
        default="observed",
        help="NDVI source mode passed to build_swim_input",
    )
    pr.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing persisted run with the same run id",
    )
    pr.set_defaults(func=cmd_run)

    # evaluate
    pv = sub.add_parser(
        "evaluate",
        help="Run model in debug mode and write per-site CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runs the model (debug detail) and writes per-site CSV. Optionally computes metrics vs OpenET and flux data.",
    )
    add_common(pv)
    pv.add_argument(
        "--forecast-params", default=None, help="Path to forecast parameter CSV (optional)"
    )
    pv.add_argument("--spinup", default=None, help="Path to spinup JSON (optional)")
    pv.add_argument(
        "--flux-dir",
        default=None,
        help="Directory containing per-site flux CSVs named <FID>_daily_data.csv (e.g., config.data_dir/daily_flux_files)",
    )
    pv.add_argument(
        "--openet-dir",
        default=None,
        help="Directory with subfolders daily_data/ and monthly_data/ containing <FID>.csv files from OpenET",
    )
    pv.add_argument(
        "--metrics-out",
        default=None,
        help="Directory to write metrics summaries; defaults to --out-dir",
    )
    pv.set_defaults(func=cmd_evaluate)

    # calibrate-batch
    pb = sub.add_parser(
        "calibrate-batch",
        help="Batch calibration: partition fields, build/run/ingest PEST++ per batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Config-driven batch PEST++ IES calibration. Partitions fields into batches, "
        "builds PEST++ setups, runs them, ingests results, and cleans up.",
    )
    pb.add_argument(
        "config",
        help="Path to project TOML (e.g., examples/4_Flux_Network/4_Flux_Network.toml)",
    )
    pb.add_argument(
        "--action",
        default="calibrate-all",
        choices=[
            "prep",
            "build-all",
            "run-batch",
            "run-batch-task",
            "run-all",
            "ingest-batch",
            "ingest-all",
            "status",
            "calibrate-all",
            "cleanup-failed",
        ],
        help="Batch action to perform",
    )
    pb.add_argument(
        "--batch-id",
        type=int,
        default=None,
        help="Batch ID for run-batch/run-batch-task/ingest-batch",
    )
    pb.add_argument(
        "--rebuild",
        action="store_true",
        help="run-batch-task: rebuild and rerun even when the batch is already complete",
    )
    pb.add_argument(
        "--batch-size", type=int, default=None, help="Fields per batch (default: config, else 50)"
    )
    pb.add_argument(
        "--workers", type=int, default=None, help="PEST workers per batch (default: from config)"
    )
    pb.add_argument(
        "--noptmax", type=int, default=None, help="Max PEST iterations (default: config, else 3)"
    )
    pb.add_argument(
        "--reals", type=int, default=None, help="Ensemble realizations (default: from config)"
    )
    pb.add_argument("--resume", action="store_true", help="Skip already-ingested batches")
    pb.add_argument("--override", action="store_true", help="Override preflight gate failures")
    pb.add_argument(
        "--skip-health", action="store_true", help="Use prior health check from container"
    )
    pb.add_argument("--exclude-uncovered", action="store_true", help="Exclude zero-coverage fields")
    pb.add_argument(
        "--skip-fids", type=str, default=None, help="Text file of FIDs to exclude (one per line)"
    )
    pb.add_argument("--container", type=str, default=None, help="Override container path")
    pb.add_argument("--output", type=str, default=None, help="Override output directory")
    pb.add_argument("--shapefile", type=str, default=None, help="Override fields shapefile")
    pb.set_defaults(func=cmd_calibrate_batch)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        try:
            import importlib.metadata as importlib_metadata  # py3.8+
        except Exception:
            import importlib_metadata  # type: ignore
        try:
            ver = importlib_metadata.version("swimrs")
        except Exception:
            ver = "unknown"
        print(ver)
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

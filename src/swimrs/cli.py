import argparse
import os
import sys
from typing import List, Optional


# Import helpers: support both installed (swimrs.*) and in-repo (src.swimrs.*)
def _try_import(installed_path: str, dev_path: str, name: str):
    try:
        module = __import__(installed_path, fromlist=[name])
        return getattr(module, name)
    except Exception:
        module = __import__(dev_path, fromlist=[name])
        return getattr(module, name)


ProjectConfig = _try_import('swimrs.swim.config', 'src.swimrs.swim.config', 'ProjectConfig')

# Earth Engine utils and exports
is_authorized = _try_import('swimrs.data_extraction.ee.ee_utils', 'src.swimrs.data_extraction.ee.ee_utils',
                            'is_authorized')
sparse_sample_ndvi = _try_import('swimrs.data_extraction.ee.ndvi_export', 'src.swimrs.data_extraction.ee.ndvi_export',
                                 'sparse_sample_ndvi')
sparse_sample_etf = _try_import('swimrs.data_extraction.ee.etf_export', 'src.swimrs.data_extraction.ee.etf_export',
                                'sparse_sample_etf')
sample_snodas_swe = _try_import('swimrs.data_extraction.ee.snodas_export',
                                'src.swimrs.data_extraction.ee.snodas_export', 'sample_snodas_swe')
get_irrigation = _try_import('swimrs.data_extraction.ee.ee_props', 'src.swimrs.data_extraction.ee.ee_props',
                             'get_irrigation')
get_ssurgo = _try_import('swimrs.data_extraction.ee.ee_props', 'src.swimrs.data_extraction.ee.ee_props', 'get_ssurgo')
get_cdl = _try_import('swimrs.data_extraction.ee.ee_props', 'src.swimrs.data_extraction.ee.ee_props', 'get_cdl')
get_landcover = _try_import('swimrs.data_extraction.ee.ee_props', 'src.swimrs.data_extraction.ee.ee_props',
                            'get_landcover')
sample_era5_land_variables_daily = _try_import(
    'swimrs.data_extraction.ee.ee_era5', 'src.swimrs.data_extraction.ee.ee_era5',
    'sample_era5_land_variables_daily'
)

# GridMET
assign_gridmet_ids = _try_import(
    'swimrs.data_extraction.gridmet.gridmet',
    'src.swimrs.data_extraction.gridmet.gridmet',
    'assign_gridmet_ids',
)
sample_gridmet_corrections = _try_import(
    'swimrs.data_extraction.gridmet.gridmet',
    'src.swimrs.data_extraction.gridmet.gridmet',
    'sample_gridmet_corrections',
)
download_gridmet = _try_import(
    'swimrs.data_extraction.gridmet.gridmet',
    'src.swimrs.data_extraction.gridmet.gridmet',
    'download_gridmet',
)

# Calibration
PestBuilder = _try_import('swimrs.calibrate.pest_builder', 'src.swimrs.calibrate.pest_builder', 'PestBuilder')
run_pst = _try_import('swimrs.calibrate.run_pest', 'src.swimrs.calibrate.run_pest', 'run_pst')

# Evaluate
SamplePlots = _try_import('swimrs.swim.sampleplots', 'src.swimrs.swim.sampleplots', 'SamplePlots')
field_day_loop = _try_import('swimrs.model.obs_field_cycle', 'src.swimrs.model.obs_field_cycle', 'field_day_loop')
compare_etf_estimates = _try_import('swimrs.analysis.metrics', 'src.swimrs.analysis.metrics', 'compare_etf_estimates')
SwimContainer = _try_import('swimrs.container', 'src.swimrs.container', 'SwimContainer')

def _parse_sites_arg(sites: Optional[str]) -> Optional[List[str]]:
    if not sites:
        return None
    parts = [s.strip() for s in sites.split(',') if s.strip()]
    return parts or None


def _resolve_project_root(default_config_path: str, override: Optional[str]) -> Optional[str]:
    if override:
        return os.path.abspath(override)
    # default: directory containing the TOML
    return os.path.dirname(os.path.abspath(default_config_path))


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
    bucket_arg = args.bucket or getattr(config, 'ee_bucket', None)
    file_prefix = args.file_prefix
    if export_dest == 'bucket' and not bucket_arg:
        print('Export destination set to bucket, but no --bucket or config.ee_bucket provided')
        return 2

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
                drive_folder='swim',
                file_prefix=file_prefix,
                drive_categorize=args.drive_categorize,
            )
        except Exception as e:
            print(f"SNODAS export error: {e}")

    # 2) Properties (CDL, irrigation fraction, SSURGO, landcover)
    if not args.no_properties:
        try:
            project = config.project_name or 'swim'
            get_cdl(
                config.fields_shapefile,
                f"{project}_cdl",
                selector=config.feature_id_col,
                dest=export_dest,
                bucket=bucket_arg,
                drive_folder='swim',
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
                drive_folder='swim',
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
                drive_folder='swim',
                file_prefix=file_prefix,
                drive_categorize=args.drive_categorize,
            )
            get_landcover(
                config.fields_shapefile,
                f"{project}_landcover",
                debug=False,
                selector=config.feature_id_col,
                out_fmt='CSV',
                dest=export_dest,
                bucket=bucket_arg,
                drive_folder='swim',
                drive_categorize=args.drive_categorize,
                file_prefix=file_prefix,
            )
        except Exception as e:
            print(f"Properties export error: {e}")

    # 3) Remote sensing NDVI (and optionally Sentinel & ETF models)
    if not args.no_rs:
        try:
            masks = ['irr', 'inv_irr']
            years = list(range(config.start_dt.year, config.end_dt.year + 1))
            for m in masks:
                landsat_check = os.path.join(config.landsat_dir or '', 'extracts', 'ndvi', m)
                sparse_sample_ndvi(
                    config.fields_shapefile,
                    bucket=bucket_arg,
                    debug=False,
                    mask_type=m,
                    check_dir=landsat_check,
                    start_yr=years[0],
                    end_yr=years[-1],
                    feature_id=config.feature_id_col,
                    satellite='landsat',
                    state_col=config.state_col,
                    select=_parse_sites_arg(args.sites),
                    dest=export_dest,
                    drive_folder='swim',
                    file_prefix=file_prefix,
                    drive_categorize=args.drive_categorize,
                )
                if args.add_sentinel:
                    sentinel_check = os.path.join(config.sentinel_dir or '', 'extracts', 'ndvi', m)
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
                        satellite='sentinel',
                        state_col=config.state_col,
                        select=_parse_sites_arg(args.sites),
                        dest=export_dest,
                        drive_folder='swim',
                        file_prefix=file_prefix,
                        drive_categorize=args.drive_categorize,
                    )

            # Optional ETF models
            if args.etf_models:
                models = [m.strip() for m in args.etf_models.split(',') if m.strip()]
                for m in masks:
                    for model in models:
                        etf_check = os.path.join(config.landsat_dir or '', 'extracts', f'{model}_etf', m)
                        sparse_sample_etf(
                            config.fields_shapefile,
                            bucket=bucket_arg,
                            debug=False,
                            mask_type=m,
                            check_dir=etf_check,
                            feature_id=config.feature_id_col,
                            select=_parse_sites_arg(args.sites),
                            start_yr=max(2016, years[0]),
                            end_yr=years[-1],
                            state_col=config.state_col,
                            model=model,
                            dest=export_dest,
                            drive_folder='swim',
                            file_prefix=file_prefix,
                            drive_categorize=args.drive_categorize,
                        )
        except Exception as e:
            print(f"Remote sensing export error: {e}")

    # 4) Meteorology: GridMET or ERA5-Land based on config.met_source
    met_source = getattr(config, 'met_source', 'gridmet')
    if args.no_met:
        print("Skipping meteorology download (--no-met).")
    elif met_source == 'gridmet':
        try:
            use_nldas = (getattr(config, 'runoff_process', None) == 'ier')
            # Assign GFIDs (optionally from centroids), optionally sample corrections
            gridmet_points = config.gridmet_centroids if getattr(args, 'use_gridmet_centroids', False) else None
            join_path = config.gridmet_mapping_shp
            fields_joined = assign_gridmet_ids(
                fields=config.gridmet_mapping_shp,
                fields_join=join_path,
                gridmet_points=gridmet_points,
                field_select=_parse_sites_arg(args.sites),
                feature_id=config.feature_id_col,
                gridmet_id_col=config.gridmet_mapping_index_col or 'GFID',
            )

            factors_path = None
            if getattr(args, 'gridmet_correction', False) and config.correction_tifs:
                factors_path = config.gridmet_factors
                sample_gridmet_corrections(
                    fields_join=join_path,
                    gridmet_ras=config.correction_tifs,
                    factors_js=factors_path,
                    gridmet_id_col=config.gridmet_mapping_index_col or 'GFID',
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
                use_nldas=use_nldas,
            )
        except Exception as e:
            print(f"GridMET error: {e}")
    elif met_source == 'era5':
        if export_dest != 'bucket':
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
    else:
        print(f"Unknown met_source '{met_source}' in config; skipping meteorology export.")

    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root, calibrate=True)

    # Calibration now expects prepped_input.json to exist (built via container workflow).
    if not os.path.isfile(config.input_data):
        print(f"Missing prepped input: {config.input_data}. Build it via SwimContainer export before calibrating.")
        return 1

    # Build and run PEST++
    try:
        # Allow CLI override of python script
        if args.python_script:
            config.python_script = args.python_script
        builder = PestBuilder(config, use_existing=False, python_script=getattr(config, 'python_script', None))
        builder.build_pest(target_etf=config.etf_target_model, members=config.etf_ensemble_members)
        builder.build_localizer()

        # Spinup (noptmax=0), then set real run
        builder.write_control_settings(noptmax=0)
        builder.spinup(overwrite=True)

        reals = int(args.realizations) if args.realizations else (config.realizations or 250)
        builder.write_control_settings(noptmax=3, reals=reals)

        exe_ = 'pestpp-ies'
        project = config.project_name
        p_dir = os.path.join(config.pest_run_dir, 'pest')
        m_dir = os.path.join(config.pest_run_dir, 'master')
        w_dir = os.path.join(config.pest_run_dir, 'workers')
        pst_name = f'{project}.pst'

        run_pst(p_dir, exe_, pst_name, num_workers=int(args.workers), worker_root=w_dir,
                master_dir=m_dir, verbose=False, cleanup=False)
    except Exception as e:
        print(f"Calibration run failed: {e}")
        return 1

    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Build model-ready inputs using SwimContainer (ingest → compute → export)."""
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root)

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
    masks = ("irr", "inv_irr")
    instruments = ["landsat"]

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
            print(f"Properties ingest skipped/failed: {e}")

        # NDVI
        if not args.no_ndvi:
            for mask in masks:
                ndvi_dir = os.path.join(config.landsat_dir or "", "extracts", "ndvi", mask)
                if os.path.isdir(ndvi_dir):
                    container.ingest.ndvi(
                        ndvi_dir,
                        uid_column=config.feature_id_col,
                        instrument="landsat",
                        mask=mask,
                        fields=sites,
                        overwrite=args.overwrite,
                    )
                    print(f"Ingested Landsat NDVI ({mask})")
                if args.add_sentinel:
                    s2_dir = os.path.join(config.sentinel_dir or "", "extracts", "ndvi", mask)
                    if os.path.isdir(s2_dir):
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

        # ETF
        if not args.no_etf:
            etf_models = [m for m in [config.etf_target_model] + (config.etf_ensemble_members or []) if m]
            for model in etf_models:
                for mask in masks:
                    etf_dir = os.path.join(config.landsat_dir or "", "extracts", f"{model}_etf", mask)
                    if os.path.isdir(etf_dir):
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

        # Meteorology
        if not args.no_met:
            met_source = getattr(config, "met_source", "gridmet")
            if met_source == "gridmet":
                try:
                    container.ingest.gridmet(
                        config.met_dir,
                        grid_shapefile=config.gridmet_mapping_shp,
                        uid_column=config.feature_id_col,
                        grid_column=config.gridmet_id_col or config.gridmet_mapping_index_col or "GFID",
                        overwrite=args.overwrite,
                    )
                    print("Ingested GridMET")
                except Exception as e:
                    print(f"GridMET ingest failed: {e}")
            elif met_source == "era5":
                try:
                    container.ingest.era5(
                        config.met_dir,
                        overwrite=args.overwrite,
                    )
                    print("Ingested ERA5-Land")
                except Exception as e:
                    print(f"ERA5 ingest failed: {e}")

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
                print(f"SNODAS ingest skipped/failed: {e}")

        # Derived products
        try:
            container.compute.merged_ndvi(
                masks=masks,
                instruments=tuple(instruments),
                overwrite=args.overwrite,
            )
            print("Computed merged NDVI")
        except Exception as e:
            print(f"Merged NDVI compute skipped/failed: {e}")

        try:
            container.compute.dynamics(
                etf_model=config.etf_target_model or "ssebop",
                irr_threshold=config.irrigation_threshold or 0.1,
                masks=masks,
                instruments=tuple(instruments),
                use_mask=True,
                use_lulc=False,
                met_source=getattr(config, "met_source", "gridmet"),
                fields=sites,
                overwrite=args.overwrite,
            )
            print("Computed dynamics")
        except Exception as e:
            print(f"Dynamics compute skipped/failed: {e}")

        # Export prepped input
        export_path = config.input_data or config.prepped_input or os.path.join(out_root or ".", "prepped_input.json")
        try:
            container.export.prepped_input_json(
                export_path,
                etf_model=config.etf_target_model or "ssebop",
                masks=masks,
                met_source=getattr(config, "met_source", "gridmet"),
                instrument=instruments[0] if instruments else "landsat",
                fields=sites,
                irr_threshold=config.irrigation_threshold or 0.1,
                include_switched_etf=True,
            )
            print(f"Wrote {export_path}")
        except Exception as e:
            print(f"Prepped input export failed: {e}")

    finally:
        try:
            container.close()
        except Exception:
            pass

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


def cmd_evaluate(args: argparse.Namespace) -> int:
    conf_path = args.config
    out_root = _resolve_project_root(conf_path, args.out_dir)

    config = ProjectConfig()
    config.read_config(conf_path, project_root_override=out_root, forecast=True)

    # Optional overrides for forecast params and spinup
    if args.forecast_params:
        config.forecast_param_csv = args.forecast_params
    if args.spinup:
        config.spinup = args.spinup
    config.read_forecast_parameters()

    # Load input and run model at debug granularity; write per-site CSV
    plots = SamplePlots()
    plots.initialize_plot_data(config)

    os.makedirs(out_root, exist_ok=True)
    targets = plots.input.get('order', [])
    if args.sites:
        requested = set(_parse_sites_arg(args.sites) or [])
        targets = [t for t in targets if t in requested]

    try:
        result = field_day_loop(config, plots, debug_flag=True)
    except Exception as e:
        print(f"Evaluation run failed: {e}")
        return 1

    metrics_by_site = {}
    for fid in targets:
        try:
            df = result[fid]
        except Exception:
            continue
        out_csv = os.path.join(out_root, f'{fid}.csv')
        try:
            df.to_csv(out_csv)
            print(f'Wrote {out_csv}')
        except Exception as e:
            print(f'Failed to write {out_csv}: {e}')

        # Optional metrics vs OpenET
        if args.flux_dir and args.openet_dir:
            try:
                flux_file = os.path.join(args.flux_dir, f'{fid}_daily_data.csv')
                openet_daily = os.path.join(args.openet_dir, 'daily_data', f'{fid}.csv')
                openet_monthly = os.path.join(args.openet_dir, 'monthly_data', f'{fid}.csv')
                irr = (plots.input.get('irr_data') or {}).get(fid, {})
                daily, overpass, monthly = compare_etf_estimates(
                    combined_output_path=df,
                    flux_data_path=flux_file,
                    openet_daily_path=openet_daily,
                    openet_monthly_path=openet_monthly,
                    irr=irr,
                    target_model=getattr(config, 'etf_target_model', 'ssebop'),
                    gap_tolerance=5,
                )
                metrics_by_site[fid] = {
                    'daily': daily or {},
                    'overpass': overpass or {},
                    'monthly': monthly or {},
                }
            except Exception as e:
                print(f'Metrics failed for {fid}: {e}')

    # Write metrics summary if requested
    if metrics_by_site:
        import json as _json
        metrics_dir = args.metrics_out or out_root
        try:
            os.makedirs(metrics_dir, exist_ok=True)
        except Exception:
            pass
        metrics_json = os.path.join(metrics_dir, 'metrics_by_site.json')
        try:
            with open(metrics_json, 'w') as fp:
                _json.dump(metrics_by_site, fp, indent=2)
            print(f'Wrote {metrics_json}')
        except Exception as e:
            print(f'Failed to write metrics JSON: {e}')

        # Also emit a flat CSV of monthly RMSE/R2 if available
        try:
            import pandas as _pd
            rows = []
            for site, d in metrics_by_site.items():
                monthly = d.get('monthly') or {}
                if monthly:
                    row = {'site': site}
                    row.update({k: v for k, v in monthly.items()})
                    rows.append(row)
            if rows:
                dfm = _pd.DataFrame(rows)
                dfm.to_csv(os.path.join(metrics_dir, 'metrics_monthly.csv'), index=False)
                print(f"Wrote {os.path.join(metrics_dir, 'metrics_monthly.csv')}")
        except Exception as e:
            print(f'Failed to write metrics CSV: {e}')

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='swim',
        description='SWIM-RS workflow CLI: extract -> prep -> calibrate -> evaluate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--version', action='store_true', help='Print version and exit')
    sub = p.add_subparsers(dest='command', required=True)

    def add_common(sp):
        sp.add_argument(
            'config',
            help='Path to project TOML (e.g., examples/5_Flux_Ensemble/5_Flux_Ensemble.toml)'
        )
        sp.add_argument(
            '--out-dir',
            default=None,
            help='Override project root for outputs; defaults to the directory containing the TOML'
        )
        sp.add_argument(
            '--workers', type=int, default=6,
            help='Worker count for parallelizable steps (e.g., dynamics, calibration)'
        )
        sp.add_argument(
            '--sites', default=None,
            help='Comma-separated site IDs to restrict processing; default processes all sites'
        )

    # extract
    pe = sub.add_parser(
        'extract',
        help='Run data extraction (Earth Engine + GridMET)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Exports SNODAS, properties (CDL/irrigation/soils/landcover), NDVI/ETF, and GridMET time series.',
    )
    add_common(pe)
    pe.add_argument(
        '--add-sentinel', action='store_true',
        help='Also export Sentinel-2 NDVI (>=2017). Default: off'
    )
    pe.add_argument(
        '--etf-models', default=None,
        help='Comma-separated ETF models to export (options: ssebop, ptjpl, sims, eemetric, geesebal, disalexi, openet)'
    )
    pe.add_argument('--no-snodas', action='store_true', help='Skip SNODAS SWE extraction (default: run)')
    pe.add_argument('--no-properties', action='store_true',
                    help='Skip CDL/irrigation/soils/landcover extraction (default: run)')
    pe.add_argument('--no-rs', action='store_true', help='Skip remote sensing (NDVI/ETF) extraction (default: run)')
    pe.add_argument('--no-met', action='store_true', help='Skip meteorology download (GridMET or ERA5-Land)')
    pe.add_argument('--export', choices=['drive', 'bucket'], default='drive', help='Earth Engine export destination')
    pe.add_argument('--bucket', default=None, help='Cloud Storage bucket when --export=bucket (e.g., my-bucket)')
    pe.add_argument('--drive-categorize', action='store_true',
                    help='Place Drive exports into per-category folders (e.g., swim_properties, swim_ndvi)')
    pe.add_argument('--file-prefix', default='swim', help='Prefix path under the bucket for exports (dest=bucket)')
    pe.add_argument('--use-gridmet-centroids', action='store_true',
                    help='Assign GridMET GFIDs using provided centroids shapefile (paths.gis.gridmet_centroids)')
    pe.add_argument('--gridmet-correction', action='store_true',
                    help='Sample GridMET correction rasters (paths.conus.correction_tifs) when mapping GFIDs')
    pe.set_defaults(func=cmd_extract)

    # prep (container-based)
    pp = sub.add_parser(
        'prep',
        help='Ingest data into SwimContainer and export prepped_input.json',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Ingest properties/RS/met into a .swim container, compute dynamics, and export prepped_input.json.',
    )
    add_common(pp)
    pp.add_argument('--overwrite', action='store_true', help='Overwrite existing datasets in the container')
    pp.add_argument('--no-ndvi', action='store_true', help='Skip NDVI ingestion')
    pp.add_argument('--no-etf', action='store_true', help='Skip ETf ingestion')
    pp.add_argument('--no-met', action='store_true', help='Skip meteorology ingestion')
    pp.add_argument('--no-snow', action='store_true', help='Skip SNODAS ingestion')
    pp.add_argument('--add-sentinel', action='store_true', help='Ingest Sentinel-2 NDVI if available')
    pp.set_defaults(func=cmd_prep)

    # calibrate
    pc = sub.add_parser(
        'calibrate',
        help='Build and run calibration with PEST++ IES',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Builds a PEST++ project, runs spinup (noptmax=0), then IES (noptmax=3). Uses workers for parallel execution.',
    )
    add_common(pc)
    pc.add_argument(
        '--realizations', type=int, default=None,
        help='Override number of realizations; uses config value if set, otherwise 250'
    )
    pc.add_argument('--python-script', default=None,
                    help='Override custom forward runner script (default: package script)')
    pc.set_defaults(func=cmd_calibrate)

    # inspect
    pi = sub.add_parser(
        'inspect',
        help='Inspect a .swim container file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Show contents and status of a .swim container file.',
    )
    pi.add_argument('container', help='Path to .swim container file')
    pi.add_argument('--detailed', action='store_true', help='Show detailed status with provenance log')
    pi.set_defaults(func=cmd_inspect)

    # evaluate
    pv = sub.add_parser(
        'evaluate',
        help='Run model in debug mode and write per-site CSVs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Runs the model (debug detail) and writes per-site CSV. Optionally computes metrics vs OpenET and flux data.',
    )
    add_common(pv)
    pv.add_argument('--forecast-params', default=None, help='Path to forecast parameter CSV (optional)')
    pv.add_argument('--spinup', default=None, help='Path to spinup JSON (optional)')
    pv.add_argument(
        '--flux-dir', default=None,
        help='Directory containing per-site flux CSVs named <FID>_daily_data.csv (e.g., config.data_dir/daily_flux_files)'
    )
    pv.add_argument(
        '--openet-dir', default=None,
        help='Directory with subfolders daily_data/ and monthly_data/ containing <FID>.csv files from OpenET'
    )
    pv.add_argument('--metrics-out', default=None, help='Directory to write metrics summaries; defaults to --out-dir')
    pv.set_defaults(func=cmd_evaluate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, 'version', False):
        try:
            import importlib.metadata as importlib_metadata  # py3.8+
        except Exception:
            import importlib_metadata  # type: ignore
        try:
            ver = importlib_metadata.version('swimrs')
        except Exception:
            ver = 'unknown'
        print(ver)
        return 0
    return int(args.func(args))


if __name__ == '__main__':
    sys.exit(main())

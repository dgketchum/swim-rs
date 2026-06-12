"""Re-extract stale remote sensing data for 5_Flux_Ensemble.

Audit (2026-02-02) found the following data predates the current extraction code:

  - PT-JPL ETf:   Extracted Sep 2025 with old code (pre-FOSS migration Jan 2026).
                   0/556 dates match 3_Crane. Mean diff -0.14.
  - geeSEBAL ETf:  Extracted Apr 2025 from OpenET daily assets (not custom API).
                   198 vs 542 irr obs, 0 vs 93 inv_irr. 51 of 77 fields missing.
  - SSEBop ETf:   Extracted Feb 2025. 272/533 dates differ from 3_Crane re-extraction.
  - NDVI:         Extracted Feb 2025, before SBAF harmonization (Jan 12 2026).
                   126/582 dates match 3_Crane. Mean abs diff 0.018.
  - SIMS ETf:     OK — 561/561 exact match with 3_Crane. NOT re-extracted.

This script:
  1. Removes old extract CSVs for the four stale datasets
  2. Submits Earth Engine export tasks via the current FOSS packages
  3. Prints the gsutil sync + container rebuild commands to run after EE tasks finish

Usage:
    python reextract_stale_data.py                # submit all
    python reextract_stale_data.py --models ptjpl  # submit one model
    python reextract_stale_data.py --dry-run       # show what would be deleted
"""

import argparse
import os
import shutil
from pathlib import Path

from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parents[1]
    conf = project_dir / "5_Flux_Ensemble.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))
    return cfg


STALE_ETF_MODELS = ["ptjpl", "ssebop", "geesebal"]
MASKS = ["irr", "inv_irr"]


def clear_old_extracts(cfg, models, clear_ndvi, dry_run=False):
    """Remove old extract CSVs so EE re-export isn't skipped by check_dir."""
    landsat_extracts = os.path.join(cfg.landsat_dir, "extracts")

    for model in models:
        for mask in MASKS:
            d = os.path.join(landsat_extracts, f"{model}_etf", mask)
            if os.path.isdir(d):
                n = len(os.listdir(d))
                if dry_run:
                    print(f"  [dry-run] would remove {n} files from {d}")
                else:
                    shutil.rmtree(d)
                    os.makedirs(d, exist_ok=True)
                    print(f"  Cleared {n} files from {d}")

    if clear_ndvi:
        for mask in MASKS:
            d = os.path.join(landsat_extracts, "ndvi", mask)
            if os.path.isdir(d):
                n = len(os.listdir(d))
                if dry_run:
                    print(f"  [dry-run] would remove {n} files from {d}")
                else:
                    shutil.rmtree(d)
                    os.makedirs(d, exist_ok=True)
                    print(f"  Cleared {n} files from {d}")


def submit_etf_exports(cfg, models):
    """Submit EE export tasks for ETf models."""
    import ee

    ee.Initialize()

    from swimrs.data_extraction.ee import (
        export_geesebal_zonal_stats,
        export_ptjpl_zonal_stats,
        export_ssebop_zonal_stats,
    )

    exporters = {
        "ptjpl": export_ptjpl_zonal_stats,
        "ssebop": export_ssebop_zonal_stats,
        "geesebal": export_geesebal_zonal_stats,
    }

    for mask in MASKS:
        for model in models:
            fn = exporters[model]
            chk = os.path.join(cfg.landsat_dir, "extracts", f"{model}_etf", mask)
            print(f"\nSubmitting {model.upper()} ETf ({mask})...")
            fn(
                shapefile=cfg.fields_shapefile,
                bucket=cfg.ee_bucket,
                feature_id=cfg.feature_id_col,
                select=None,
                start_yr=cfg.start_dt.year,
                end_yr=cfg.end_dt.year,
                mask_type=mask,
                check_dir=chk,
                state_col=cfg.state_col,
                buffer=None,
                batch_size=60,
                file_prefix=cfg.project_name,
            )


def submit_ndvi_export(cfg):
    """Submit EE export tasks for Landsat NDVI with SBAF harmonization."""
    import ee

    ee.Initialize()

    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    for mask in MASKS:
        chk = os.path.join(cfg.landsat_dir, "extracts", "ndvi", mask)
        print(f"\nSubmitting Landsat NDVI ({mask})...")
        sparse_sample_ndvi(
            cfg.fields_shapefile,
            bucket=cfg.ee_bucket,
            dest="bucket",
            debug=False,
            mask_type=mask,
            check_dir=chk,
            start_yr=cfg.start_dt.year,
            end_yr=cfg.end_dt.year,
            feature_id=cfg.feature_id_col,
            satellite="landsat",
            state_col=cfg.state_col,
            select=None,
            file_prefix=cfg.project_name,
        )


def print_next_steps(cfg):
    """Print commands to run after EE tasks finish."""
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Monitor Earth Engine tasks:
   https://code.earthengine.google.com/tasks

2. Once ALL tasks complete, sync from GCS:""")
    print(f"   gsutil -m rsync -r gs://{cfg.ee_bucket}/{cfg.project_name}/ {cfg.data_dir}/")
    print("""
3. Rebuild the container (from examples/5_Flux_Ensemble/):
   python container_prep.py --overwrite
""")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=STALE_ETF_MODELS,
        help="ETf models to re-extract (default: all stale)",
    )
    parser.add_argument("--skip-ndvi", action="store_true", help="Skip NDVI re-extraction")
    parser.add_argument("--skip-etf", action="store_true", help="Skip ETf re-extraction")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without submitting tasks"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Skip clearing old extracts (resume mode — uses check_dir to skip existing)",
    )
    args = parser.parse_args()

    models = args.models or STALE_ETF_MODELS
    do_ndvi = not args.skip_ndvi
    do_etf = not args.skip_etf

    cfg = _load_config()

    print(f"Project: {cfg.project_name}")
    print(f"Date range: {cfg.start_dt.date()} to {cfg.end_dt.date()}")
    print(f"Shapefile: {cfg.fields_shapefile}")
    print(f"Bucket: {cfg.ee_bucket}")
    print(f"ETf models to re-extract: {models if do_etf else 'SKIPPED'}")
    print(f"NDVI re-extract: {'yes' if do_ndvi else 'SKIPPED'}")
    print()

    # Step 1: clear old extracts (unless --no-clear)
    if args.no_clear:
        print("Skipping clear (--no-clear). Will resume using check_dir to skip existing.")
    else:
        print("Clearing old extract CSVs...")
        etf_to_clear = models if do_etf else []
        clear_old_extracts(cfg, etf_to_clear, clear_ndvi=do_ndvi, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No EE tasks submitted.")
        return

    # Step 2: submit EE export tasks
    if do_etf:
        submit_etf_exports(cfg, models)

    if do_ndvi:
        submit_ndvi_export(cfg)

    # Step 3: print next steps
    print_next_steps(cfg)


if __name__ == "__main__":
    main()

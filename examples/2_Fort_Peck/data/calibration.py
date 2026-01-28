#!/usr/bin/env python
"""
PEST++ Calibration Script for Fort Peck
========================================

This script runs PEST++ IES calibration using SSEBop ETf and SNODAS SWE
observations. It reproduces the workflow from 02_calibration.ipynb.

The calibration uses:
- SSEBop ETf observations from Landsat (not flux tower data)
- SNODAS SWE observations
- Iterative Ensemble Smoother (pestpp-ies)

Requirements
------------
- PEST++ installed (pestpp-ies in PATH)
- SwimContainer with ingested data at data/{project}.swim
- Project configuration at {project}.toml

Usage
-----
    python calibration.py                      # Run with defaults
    python calibration.py --workers 12         # Use 12 parallel workers
    python calibration.py --noptmax 5          # Run 5 optimization iterations
    python calibration.py --reals 100          # Use 100 realizations
    python calibration.py --dry-run            # Build files only, don't run PEST++
    python calibration.py --skip-build         # Skip build, run existing setup
    python calibration.py --archive ./results  # Archive results to custom directory
    python calibration.py --no-archive         # Skip cleanup/archiving step

"""

import argparse
import os
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '../..'))
sys.path.insert(0, ROOT_DIR)

from swimrs.container import SwimContainer
from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.pest_cleanup import PestResults
from swimrs.swim.config import ProjectConfig
from swimrs.calibrate.run_pest import run_pst


def load_config():
    """Load project configuration."""
    config_file = os.path.join(PROJECT_DIR, '2_Fort_Peck.toml')
    cfg = ProjectConfig()
    cfg.read_config(config_file, project_root_override=PROJECT_DIR)
    return cfg


def export_observations(container, project_ws, etf_model='ssebop'):
    """Export observation files for PEST++ calibration."""
    # Match the notebook: obs files go in data/pestrun/obs
    obs_dir = os.path.join(project_ws, 'data', 'pestrun', 'obs')
    os.makedirs(obs_dir, exist_ok=True)

    print("\n=== Exporting Observation Files ===")
    container.export.observations(
        output_dir=obs_dir,
        etf_model=etf_model,
        masks=('irr', 'inv_irr'),
        irr_threshold=0.1,
    )
    print(f"Observation files written to {obs_dir}")


def build_pest_setup(config, container, python_script=None):
    """Build PEST++ control files and setup."""
    print("\n=== Building PEST++ Control Files ===")

    # python_script=None uses the default from swimrs.calibrate package
    builder = PestBuilder(
        config,
        container=container,
        use_existing=False,
        python_script=python_script,
    )

    # Build the .pst control file
    print("Building .pst control file...")
    builder.build_pest(
        target_etf=config.etf_target_model,
        members=config.etf_ensemble_members
    )

    # Show created files
    pest_files = [f for f in sorted(os.listdir(builder.pest_dir))
                  if os.path.isfile(os.path.join(builder.pest_dir, f))]
    print(f"\nFiles in pest directory ({builder.pest_dir}):")
    for f in pest_files:
        print(f"  {f}")

    return builder


def configure_and_test(builder, noptmax=3, reals=20):
    """Configure PEST++ settings and run dry run."""
    print("\n=== Configuring PEST++ ===")

    # Build localizer matrix
    print("Building localizer matrix...")
    builder.build_localizer()

    # Run spinup to save water balance state as initial conditions
    print("Running spinup...")
    builder.spinup(overwrite=True)

    # Run dry run to verify setup
    print("Running dry run to verify setup...")
    builder.dry_run()

    # Write control settings
    print(f"Setting noptmax={noptmax}, realizations={reals}")
    builder.write_control_settings(noptmax=noptmax, reals=reals)

    print(f"\nControl file: {builder.pst_file}")


def run_calibration(builder, config, workers=None, cleanup=True, verbose=True):
    """Run PEST++ calibration."""
    print("\n=== Running PEST++ Calibration ===")

    if workers is None:
        workers = getattr(config, 'workers', None) or 6

    pst_file = f"{config.project_name}.pst"

    print(f"Starting PEST++ with {workers} workers...")
    print(f"Control file: {pst_file}")
    print(f"Master directory: {builder.master_dir}")
    print(f"Workers directory: {builder.workers_dir}")

    run_pst(
        builder.pest_dir,
        'pestpp-ies',
        pst_file,
        num_workers=workers,
        worker_root=builder.workers_dir,
        master_dir=builder.master_dir,
        cleanup=cleanup,
        verbose=verbose
    )


def process_results(builder, config, archive_dir=None, skip_archive=False):
    """Process calibration results: check success, summarize, and cleanup."""
    results = PestResults(builder.pest_dir, config.project_name)

    # Print summary
    results.print_summary()

    # Check success
    success, issues = results.is_successful()

    if skip_archive:
        print("Skipping archive/cleanup (--no-archive specified)")
        return success

    # Cleanup and archive
    print("\n=== Archiving Results ===")
    keep_debug = not success  # Keep debug files on failure

    report = results.cleanup(
        archive_dir=archive_dir,
        keep_debug=keep_debug,
        dry_run=False,
    )

    if report['files_archived']:
        archive_path = archive_dir or os.path.join(builder.pest_dir, 'archive')
        print(f"Archived {len(report['files_archived'])} files to {archive_path}")

    if report.get('space_recovered_mb', 0) > 0:
        print(f"Space recovered: {report['space_recovered_mb']:.1f} MB")

    if report.get('debug_preserved'):
        print("\nDebug files preserved for troubleshooting.")
        if 'recommendations' in report:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Run PEST++ IES calibration for Fort Peck',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: from config or 6)')
    parser.add_argument('--noptmax', type=int, default=3,
                        help='Number of optimization iterations (default: 3)')
    parser.add_argument('--reals', type=int, default=20,
                        help='Number of realizations (default: 20, use 100-200 for production)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build files only, do not run PEST++')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip build step, run existing PEST++ setup')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep worker directories after calibration (useful for debugging)')
    parser.add_argument('--python-script', default=None,
                        help='Path to custom forward run script')
    parser.add_argument('--archive', default=None,
                        help='Directory to archive results (default: pest/archive)')
    parser.add_argument('--no-archive', action='store_true',
                        help='Skip the cleanup/archiving step')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config()
    print(f"\nProject: {cfg.project_name}")
    print(f"ETf target model: {cfg.etf_target_model}")
    print(f"Date range: {cfg.start_dt} to {cfg.end_dt}")

    # Open container
    container_path = os.path.join(SCRIPT_DIR, f'{cfg.project_name}.swim')
    if not os.path.exists(container_path):
        print(f"\nError: Container not found at {container_path}")
        print("Run build_inputs.py first to create the container.")
        sys.exit(1)

    print(f"\nOpening container: {container_path}")
    container = SwimContainer.open(container_path, mode='r')

    try:
        if not args.skip_build:
            # Export observations
            export_observations(container, PROJECT_DIR, etf_model=cfg.etf_target_model)

            # Build PEST++ setup
            builder = build_pest_setup(cfg, container, python_script=args.python_script)

            # Configure and test
            configure_and_test(builder, noptmax=args.noptmax, reals=args.reals)
        else:
            # Load existing builder (minimal init for running)
            print("\nSkipping build, using existing PEST++ setup...")
            builder = PestBuilder(
                cfg,
                container=container,
                use_existing=True,
                python_script=args.python_script
            )

        if args.dry_run:
            print("\n=== Dry Run Complete ===")
            print("PEST++ files have been built but calibration was not run.")
            print(f"To run manually: cd {builder.pest_dir} && pestpp-ies {cfg.project_name}.pst")
        else:
            # Run calibration
            run_calibration(
                builder, cfg,
                workers=args.workers,
                cleanup=not args.no_cleanup,
                verbose=True
            )

            # Process results: summarize, check success, cleanup/archive
            success = process_results(
                builder, cfg,
                archive_dir=args.archive,
                skip_archive=args.no_archive
            )

            if success:
                print(f"\nCalibration complete!")
                print(f"Use calibrated parameters in 03_calibrated_model.ipynb")
            else:
                print(f"\nCalibration completed with issues. Check output above.")

    finally:
        container.close()
        print("\nContainer closed.")


if __name__ == '__main__':
    main()

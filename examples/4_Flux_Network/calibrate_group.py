"""
Group calibration for 4_Flux_Network using modern SwimContainer workflow.

This module runs PEST++ calibration for the flux network using the
container-based workflow and the modern PestBuilder API.

Usage:
    python calibrate_group.py [--sites SITE1,SITE2,...] [--pdc-remove] [--workers N]
"""
import os
import shutil
from pathlib import Path

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.run_pest import run_pst
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    """Load project configuration from TOML file."""
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "4_Flux_Network.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    # Match pattern in existing examples: forward runner lives in the project directory
    cfg.python_script = str(project_dir / "custom_forward_run.py")
    return cfg


def run_pest_sequence(cfg: ProjectConfig, results_dir: str, select_stations: list = None,
                      pdc_remove: bool = False, overwrite: bool = False):
    """
    Run PEST++ calibration sequence using modern SwimContainer workflow.

    Args:
        cfg: ProjectConfig instance
        results_dir: Directory to save results
        select_stations: Optional list of station IDs to calibrate (default: all)
        pdc_remove: If True, run PDC detection and remove conflicted observations
        overwrite: If True, overwrite existing results
    """
    project = cfg.project_name

    # Rebuild PEST run dir from scratch
    if os.path.isdir(cfg.pest_run_dir):
        shutil.rmtree(cfg.pest_run_dir)
    os.makedirs(cfg.pest_run_dir, exist_ok=False)

    os.makedirs(results_dir, exist_ok=True)

    # Open container
    container_path = os.path.join(cfg.project_ws, f"{project}.swim")
    if not os.path.exists(container_path):
        raise FileNotFoundError(
            f"Container not found at {container_path}. "
            "Run container_prep.py first to create the container."
        )

    container = SwimContainer.open(container_path, mode='r')

    try:
        # Export observation files for PEST++ calibration
        obs_dir = os.path.join(cfg.pest_run_dir, 'obs')
        os.makedirs(obs_dir, exist_ok=True)

        print("\n=== Exporting Observations ===")
        container.export.observations(
            output_dir=obs_dir,
            etf_model=cfg.etf_target_model,
            masks=('irr', 'inv_irr'),
            irr_threshold=cfg.irrigation_threshold or 0.1,
            fields=select_stations,
        )
        print(f"Observation files written to {obs_dir}")

        # Change to project directory for relative paths in PEST
        os.chdir(Path(__file__).resolve().parent)

        # Build PestBuilder with container
        print("\n=== Building PEST++ Control Files ===")
        builder = PestBuilder(
            cfg,
            container=container,
            use_existing=False,
            python_script=getattr(cfg, "python_script", None),
            conflicted_obs=None,
        )
        builder.build_pest(target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members)
        builder.build_localizer()

        # Run spinup
        print("\n=== Running Spinup ===")
        builder.spinup(overwrite=True)
        shutil.copyfile(cfg.spinup, os.path.join(results_dir, "spinup.json"))

        exe_ = "pestpp-ies"

        # Handle PDC detection if requested
        if pdc_remove:
            print("\n=== Running PDC Detection ===")
            builder.write_control_settings(noptmax=-1, reals=5)
            builder.dry_run(exe_)

            pdc_file = os.path.join(builder.pest_dir, f"{project}.pdc.csv")
            if os.path.exists(pdc_file):
                print(f"PDC file found: {pdc_file}")
                print("Rebuilding PEST with conflicted observations removed...")

                # Rebuild with conflicted observations removed
                builder = PestBuilder(
                    cfg,
                    container=container,
                    use_existing=False,
                    python_script=getattr(cfg, "python_script", None),
                    conflicted_obs=pdc_file,
                )
                builder.build_pest(
                    target_etf=cfg.etf_target_model, members=cfg.etf_ensemble_members
                )
                builder.build_localizer()
                builder.write_control_settings(noptmax=0)
                builder.dry_run(exe_)
        else:
            # Standard dry run
            print("\n=== Running Dry Run ===")
            builder.write_control_settings(noptmax=0)
            builder.dry_run(exe_)

        # Configure and run full calibration
        print("\n=== Running PEST++ Calibration ===")
        noptmax = 3
        reals = cfg.realizations or 20
        workers = cfg.workers or 6

        builder.write_control_settings(noptmax=noptmax, reals=reals)

        pst_name = f"{project}.pst"
        run_pst(
            builder.pest_dir,
            exe_,
            pst_name,
            num_workers=workers,
            worker_root=builder.workers_dir,
            master_dir=builder.master_dir,
            verbose=False,
            cleanup=False,
        )

        # Copy key outputs to results directory
        print("\n=== Copying Results ===")
        for i in range(noptmax + 1):
            fname = f"{project}.{i}.par.csv"
            src = os.path.join(builder.master_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(results_dir, fname))
                print(f"  Copied {fname}")

        for fname in [f"{project}.phi.meas.csv", f"{project}.pdc.csv", f"{project}.idx.csv"]:
            src = os.path.join(builder.master_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(results_dir, fname))
                print(f"  Copied {fname}")

        # Clean up PEST directories
        print("\n=== Cleanup ===")
        for d in [builder.pest_dir, builder.master_dir, builder.workers_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"  Removed {d}")

    finally:
        container.close()

    print("\n=== Calibration Complete ===")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Group calibration for 4_Flux_Network using PEST++"
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated site IDs to calibrate (default: all)",
    )
    parser.add_argument(
        "--pdc-remove",
        action="store_true",
        help="Run PDC detection and remove conflicted observations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of PEST++ workers (default: from config or 6)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    args = parser.parse_args()

    config = _load_config()

    # Override workers if specified
    if args.workers:
        config.workers = args.workers

    # Parse sites argument
    select_sites = None
    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]

    results = os.path.join(config.project_ws, "group_calibration")
    run_pest_sequence(
        config,
        results,
        select_stations=select_sites,
        overwrite=args.overwrite,
        pdc_remove=args.pdc_remove,
    )

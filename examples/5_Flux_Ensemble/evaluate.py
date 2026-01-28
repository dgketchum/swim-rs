"""
Evaluate SWIM model against flux tower observations using the process package API.

This module runs the model for all flux sites and compares against
OpenET ensemble and flux tower observations.

Usage:
    python evaluate.py [--output-dir PATH] [--sites SITE1,SITE2,...] [--gap-tolerance N]
"""
import os
import time
from pathlib import Path

import pandas as pd

from openet_evaluation import evaluate_openet_site
from swimrs.container import SwimContainer
from swimrs.prep import get_flux_sites
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
from swimrs.swim.config import ProjectConfig


def output_to_dataframe(output, swim_input, field_idx: int) -> pd.DataFrame:
    """Convert DailyOutput arrays to DataFrame for a single field."""
    dates = pd.date_range(swim_input.start_date, periods=output.n_days, freq='D')

    df = pd.DataFrame({
        'et_act': output.eta[:, field_idx],
        'kc_act': output.etf[:, field_idx],
        'kc_bas': output.kcb[:, field_idx],
        'ke': output.ke[:, field_idx],
        'ks': output.ks[:, field_idx],
        'kr': output.kr[:, field_idx],
        'runoff': output.runoff[:, field_idx],
        'rain': output.rain[:, field_idx],
        'melt': output.melt[:, field_idx],
        'swe': output.swe[:, field_idx],
        'depl_root': output.depl_root[:, field_idx],
        'dperc': output.dperc[:, field_idx],
        'irrigation': output.irr_sim[:, field_idx],
        'soil_water': output.gw_sim[:, field_idx],
    }, index=dates)

    return df


def input_to_dataframe(swim_input, field_idx: int) -> pd.DataFrame:
    """Extract input time series for a field."""
    dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq='D')

    etr = swim_input.get_time_series('etr')
    prcp = swim_input.get_time_series('prcp')
    tmin = swim_input.get_time_series('tmin')
    tmax = swim_input.get_time_series('tmax')

    df = pd.DataFrame({
        'etref': etr[:, field_idx],
        'ppt': prcp[:, field_idx],
        'tmin': tmin[:, field_idx],
        'tmax': tmax[:, field_idx],
    }, index=dates)

    # Add ETf observations if available
    try:
        etf_irr = swim_input.get_time_series('etf_irr')
        etf_inv_irr = swim_input.get_time_series('etf_inv_irr')
        df['etf_irr'] = etf_irr[:, field_idx]
        df['etf_inv_irr'] = etf_inv_irr[:, field_idx]
    except (KeyError, ValueError):
        pass

    return df


def run_flux_site(fid: str, cfg: ProjectConfig, container: SwimContainer,
                  outfile: str) -> None:
    """Run SWIM model for a single flux site and save output."""
    start_time = time.time()

    # Build swim_input.h5 for this site (use temp location)
    h5_path = outfile.replace('.csv', '.h5')

    swim_input = build_swim_input(
        container,
        output_h5=h5_path,
        spinup_json_path=None,
        etf_model=cfg.etf_target_model,
        met_source="gridmet",
        fields=[fid],
    )

    # Run simulation
    output, final_state = run_daily_loop(swim_input)

    print(f"\nExecution time: {time.time() - start_time:.2f} seconds\n")

    # Convert to DataFrame
    field_idx = swim_input.fids.index(fid)
    out_df = output_to_dataframe(output, swim_input, field_idx)
    in_df = input_to_dataframe(swim_input, field_idx)
    df = pd.concat([out_df, in_df], axis=1)

    # Filter to config date range
    df = df.loc[cfg.start_dt: cfg.end_dt]
    df.to_csv(outfile)


def get_irr_data_from_container(container: SwimContainer, fid: str) -> dict:
    """Extract irrigation data for a field from container."""
    try:
        props = container.query.properties(fields=[fid])
        if fid in props and 'irr' in props[fid]:
            return props[fid]['irr']
    except Exception:
        pass
    return {}


def compare_openet(fid: str, flux_file: str, model_output: str, openet_dir: str,
                   irr_data: dict, return_comparison: bool = False, gap_tolerance: int = 5):
    """Compare SWIM and OpenET ensemble against flux observations for a single site."""
    openet_daily = os.path.join(openet_dir, "daily_data", f"{fid}.csv")
    openet_monthly = os.path.join(openet_dir, "monthly_data", f"{fid}.csv")
    irr_ = irr_data.get(fid, {})
    daily, overpass, monthly = evaluate_openet_site(
        model_output,
        flux_file,
        openet_daily_path=openet_daily,
        openet_monthly_path=openet_monthly,
        irr=irr_,
        gap_tolerance=gap_tolerance,
    )

    if monthly is None:
        return None

    agg_comp = monthly.copy()
    if len(agg_comp) < 3:
        return None

    rmse_values = {k.split("_", 1)[1]: v for k, v in agg_comp.items() if k.startswith("rmse_")}
    if not rmse_values:
        return None

    best_overall_model = min(rmse_values, key=rmse_values.get)
    if not return_comparison:
        return best_overall_model

    print(f"n Samples: {agg_comp.get('n_samples')}")
    print("Best overall:", best_overall_model)
    return best_overall_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SWIM model against flux tower observations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {project_ws}/results)",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated site IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--gap-tolerance",
        type=int,
        default=5,
        help="Gap tolerance for evaluation (default: 5)",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    cfg.read_config(str(conf))

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    all_sites, sdf = get_flux_sites(station_metadata, crop_only=False, return_df=True,
                                    western_only=True, header=1)

    # Filter sites if specified
    if args.sites:
        sites = [s.strip() for s in args.sites.split(",")]
    else:
        sites = all_sites

    openet_dir = os.path.join(cfg.data_dir, "openet_flux")
    flux_dir = os.path.join(cfg.data_dir, "daily_flux_files")

    # Use output-dir if specified, otherwise default to project_ws/results
    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join(cfg.project_ws, "results")
    os.makedirs(run_dir, exist_ok=True)

    # Open container
    container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    if not os.path.exists(container_path):
        raise FileNotFoundError(
            f"Container not found at {container_path}. "
            "Run container_prep.py first to create the container."
        )

    container = SwimContainer.open(container_path, mode='r')

    # Load irrigation data from container for all sites
    irr_data = {}
    try:
        props = container.query.properties()
        for fid in sites:
            if fid in props and 'irr' in props[fid]:
                irr_data[fid] = props[fid]['irr']
    except Exception:
        pass

    complete, incomplete = [], []

    try:
        for i, site_id in enumerate(sites):
            lulc = sdf.at[site_id, "General classification"]
            print(f"\n{i} {site_id}: {lulc}")

            flux_file = os.path.join(flux_dir, f"{site_id}_daily_data.csv")
            out_csv = os.path.join(run_dir, f"{site_id}.csv")

            try:
                run_flux_site(site_id, cfg, container, out_csv)
            except Exception as exc:
                print(f"{site_id} error: {exc}")
                incomplete.append(site_id)
                continue

            _ = compare_openet(site_id, flux_file, out_csv, openet_dir, irr_data,
                               return_comparison=True, gap_tolerance=args.gap_tolerance)
            complete.append(site_id)

        print(f"complete: {complete}")
        print(f"incomplete: {incomplete}")
    finally:
        container.close()

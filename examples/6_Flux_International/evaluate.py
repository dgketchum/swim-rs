"""
Evaluate SWIM model against flux tower observations for international sites.

This module runs the model for international flux sites and compares against
flux tower observations.

Key differences from CONUS examples:
    - Uses ERA5-Land meteorology (not GridMET)
    - Uses HWSD soils (not SSURGO)
    - No irrigation masking (mask_mode="none")
    - ETf from PT-JPL only

Usage:
    python evaluate.py [--output-dir PATH] [--sites SITE1,SITE2,...] [--gap-tolerance N]
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from swimrs.container import SwimContainer
from swimrs.prep import get_flux_sites
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
from swimrs.swim.config import ProjectConfig


def output_to_dataframe(output, swim_input, field_idx: int) -> pd.DataFrame:
    """Convert DailyOutput arrays to DataFrame for a single field."""
    dates = pd.date_range(swim_input.start_date, periods=output.n_days, freq="D")

    df = pd.DataFrame(
        {
            "et_act": output.eta[:, field_idx],
            "kc_act": output.etf[:, field_idx],
            "kc_bas": output.kcb[:, field_idx],
            "ke": output.ke[:, field_idx],
            "ks": output.ks[:, field_idx],
            "kr": output.kr[:, field_idx],
            "runoff": output.runoff[:, field_idx],
            "rain": output.rain[:, field_idx],
            "melt": output.melt[:, field_idx],
            "swe": output.swe[:, field_idx],
            "depl_root": output.depl_root[:, field_idx],
            "dperc": output.dperc[:, field_idx],
            "irrigation": output.irr_sim[:, field_idx],
            "soil_water": output.gw_sim[:, field_idx],
        },
        index=dates,
    )

    return df


def input_to_dataframe(swim_input, field_idx: int) -> pd.DataFrame:
    """Extract input time series for a field."""
    dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")

    # ERA5 uses 'eto' as the reference ET variable
    try:
        etr = swim_input.get_time_series("eto")
    except (KeyError, ValueError):
        etr = swim_input.get_time_series("etr")

    prcp = swim_input.get_time_series("prcp")
    tmin = swim_input.get_time_series("tmin")
    tmax = swim_input.get_time_series("tmax")

    df = pd.DataFrame(
        {
            "etref": etr[:, field_idx],
            "ppt": prcp[:, field_idx],
            "tmin": tmin[:, field_idx],
            "tmax": tmax[:, field_idx],
        },
        index=dates,
    )

    # Add ETf observations if available (no mask for international)
    try:
        etf = swim_input.get_time_series("etf_no_mask")
        df["etf"] = etf[:, field_idx]
    except (KeyError, ValueError):
        pass

    return df


def run_flux_site(fid: str, cfg: ProjectConfig, container: SwimContainer, outfile: str) -> None:
    """Run SWIM model for a single flux site and save output."""
    start_time = time.time()

    # Build swim_input.h5 for this site (use temp location)
    h5_path = outfile.replace(".csv", ".h5")

    swim_input = build_swim_input(
        container,
        output_h5=h5_path,
        spinup_json_path=None,
        etf_model=cfg.etf_target_model,  # "ptjpl" for international
        met_source="era5",  # ERA5-Land for international
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
    df = df.loc[cfg.start_dt : cfg.end_dt]
    df.to_csv(outfile)


def compare_with_flux(fid: str, model_output: str, flux_file: str, return_comparison: bool = False):
    """Compare model output against flux tower observations.

    Args:
        fid: Site ID
        model_output: Path to model output CSV
        flux_file: Path to flux tower data CSV
        return_comparison: If True, return comparison dict

    Returns:
        Comparison dict if return_comparison=True, else None
    """
    if not os.path.exists(flux_file):
        print(f"  Flux file not found: {flux_file}")
        return None

    try:
        # Load model output
        model_df = pd.read_csv(model_output, index_col=0, parse_dates=True)

        # Load flux data (assumes 'ET' or 'LE' column exists)
        flux_df = pd.read_csv(flux_file, index_col="date", parse_dates=True)

        # Find common dates
        common_idx = model_df.index.intersection(flux_df.index)
        if len(common_idx) < 10:
            print(f"  Insufficient overlapping data ({len(common_idx)} days)")
            return None

        # Get ET values (model uses 'et_act', flux may use 'ET' or 'LE_corr')
        model_et = model_df.loc[common_idx, "et_act"]

        if "ET" in flux_df.columns:
            flux_et = flux_df.loc[common_idx, "ET"]
        elif "LE_corr" in flux_df.columns:
            # Convert latent heat flux to ET (mm/day)
            # LE (W/m2) * 86400 / 2.45e6 = ET (mm/day)
            flux_et = flux_df.loc[common_idx, "LE_corr"] * 86400 / 2.45e6
        else:
            print("  No ET or LE column in flux file")
            return None

        # Drop NaN values
        valid_mask = ~(model_et.isna() | flux_et.isna())
        model_et = model_et[valid_mask]
        flux_et = flux_et[valid_mask]

        if len(model_et) < 10:
            print(f"  Insufficient valid data ({len(model_et)} days)")
            return None

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(flux_et, model_et))
        r2 = r2_score(flux_et, model_et)
        bias = (model_et - flux_et).mean()

        comparison = {
            "n_samples": len(model_et),
            "rmse": rmse,
            "r2": r2,
            "bias": bias,
            "mean_flux": flux_et.mean(),
            "mean_model": model_et.mean(),
        }

        print(f"  n={comparison['n_samples']}, RMSE={rmse:.2f}, R2={r2:.3f}, Bias={bias:.2f}")

        if return_comparison:
            return comparison
        return None

    except Exception as exc:
        print(f"  Error comparing {fid}: {exc}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SWIM model against flux tower observations for international sites"
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
    conf = project_dir / "6_Flux_International.toml"

    cfg = ProjectConfig()
    cfg.read_config(str(conf))

    # Try to load station metadata if available
    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    if os.path.exists(station_metadata):
        all_sites, sdf = get_flux_sites(station_metadata, crop_only=False, return_df=True)
    else:
        # Use all sites from container
        all_sites = None
        sdf = None

    # Filter sites if specified
    if args.sites:
        sites = [s.strip() for s in args.sites.split(",")]
    elif all_sites is not None:
        sites = all_sites
    else:
        sites = None  # Will be populated from container

    flux_dir = os.path.join(cfg.data_dir, "daily_flux_files")

    # Use output-dir if specified, otherwise default to project_ws/results
    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join(cfg.project_ws, "results")
    os.makedirs(run_dir, exist_ok=True)

    # Open container
    container_path = cfg.container_path
    if not os.path.exists(container_path):
        raise FileNotFoundError(
            f"Container not found at {container_path}. "
            "Run container_prep.py first to create the container."
        )

    container = SwimContainer.open(container_path, mode="r")

    # If no sites from metadata or CLI, use container field UIDs
    if sites is None:
        sites = container.field_uids

    complete, incomplete = [], []
    results = []

    try:
        for i, site_id in enumerate(sites):
            lulc = sdf.at[site_id, "General classification"] if sdf is not None else "Unknown"
            print(f"\n{i} {site_id}: {lulc}")

            flux_file = os.path.join(flux_dir, f"{site_id}_daily_data.csv")
            out_csv = os.path.join(run_dir, f"{site_id}.csv")

            try:
                run_flux_site(site_id, cfg, container, out_csv)
            except Exception as exc:
                print(f"{site_id} error: {exc}")
                incomplete.append(site_id)
                continue

            result = compare_with_flux(site_id, out_csv, flux_file, return_comparison=True)
            if result:
                results.append((site_id, result))
            complete.append(site_id)

        print(f"\n{'=' * 60}")
        print(f"Complete: {len(complete)}")
        print(f"Incomplete: {len(incomplete)}")

        if results:
            print("\nSummary Statistics:")
            rmses = [r[1]["rmse"] for r in results]
            r2s = [r[1]["r2"] for r in results]
            print(f"  Mean RMSE: {np.mean(rmses):.2f} mm/day")
            print(f"  Mean R2: {np.mean(r2s):.3f}")

    finally:
        container.close()

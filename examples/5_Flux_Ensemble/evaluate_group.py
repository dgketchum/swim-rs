"""
Flux Ensemble Evaluation Script

Evaluates SWIM model against flux tower observations for multiple sites.
Uses the modern SwimContainer + process package workflow.
"""

import collections
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from openet_evaluation import evaluate_openet_site
from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
from swimrs.prep import get_flux_sites
from swimrs.swim.config import ProjectConfig


def run_single_site(config, container, site_id, output_csv):
    """Run SWIM model for a single site using the process package.

    Parameters
    ----------
    config : ProjectConfig
        Project configuration
    container : SwimContainer
        Open container with input data
    site_id : str
        Site identifier
    output_csv : str
        Path to output CSV file

    Returns
    -------
    pd.DataFrame
        Combined input/output DataFrame
    """
    # Create temporary HDF5 for SwimInput
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        temp_h5_path = tmp.name

    try:
        # Build SwimInput from container for single site
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5_path,
            runoff_process=getattr(config, 'runoff_process', 'cn'),
            etf_model=getattr(config, 'etf_target_model', 'ssebop'),
            met_source=getattr(config, 'met_source', 'gridmet'),
            fields=[site_id],
        )

        # Run simulation
        output, final_state = run_daily_loop(swim_input)

        # Get time series data
        n_days = swim_input.n_days
        dates = pd.date_range(swim_input.start_date, periods=n_days, freq='D')

        # Get input time series
        etr = swim_input.get_time_series('etr')
        prcp = swim_input.get_time_series('prcp')
        tmin = swim_input.get_time_series('tmin')
        tmax = swim_input.get_time_series('tmax')
        ndvi = swim_input.get_time_series('ndvi')

        # Build DataFrame (field index 0 since we're doing single field)
        i = 0
        df_data = {
            # Model outputs
            'et_act': output.eta[:, i],
            'etref': etr[:, i],
            'kc_act': output.etf[:, i],
            'kc_bas': output.kcb[:, i],
            'ks': output.ks[:, i],
            'ke': output.ke[:, i],
            'melt': output.melt[:, i],
            'rain': output.rain[:, i],
            'depl_root': output.depl_root[:, i],
            'dperc': output.dperc[:, i],
            'runoff': output.runoff[:, i],
            'swe': output.swe[:, i],
            'irrigation': output.irr_sim[:, i],
            'gw_sim': output.gw_sim[:, i],
            # Input time series
            'ppt': prcp[:, i],
            'tmin': tmin[:, i],
            'tmax': tmax[:, i],
            'tavg': (tmin[:, i] + tmax[:, i]) / 2.0,
            'ndvi': ndvi[:, i],
        }

        # Calculate derived columns
        df_data['soil_water'] = swim_input.properties.awc[i] - output.depl_root[:, i]

        # Add ETf observations for comparison
        etf_model = getattr(config, 'etf_target_model', 'ssebop')

        # Determine irrigation status
        try:
            irr_path = "properties/irrigation/irr"
            irr_arr = container.root[irr_path][:]
            field_idx = container.field_uids.index(site_id)
            is_irr = bool(irr_arr[field_idx])
        except Exception:
            is_irr = False

        mask = 'irr' if is_irr else 'inv_irr'

        for m in ['inv_irr', 'irr']:
            etf_path = f"remote_sensing/etf/landsat/{etf_model}/{m}"
            try:
                etf_df = container.query.dataframe(etf_path, fields=[site_id])
                etf_series = etf_df[site_id].reindex(dates)
                df_data[f'etf_{m}'] = etf_series.values
            except Exception:
                df_data[f'etf_{m}'] = np.nan

        df = pd.DataFrame(df_data, index=dates)

        # Trim to config date range
        df = df.loc[config.start_dt:config.end_dt]

        swim_input.close()

    finally:
        # Clean up temp file
        if os.path.exists(temp_h5_path):
            os.remove(temp_h5_path)

    df.to_csv(output_csv)
    return df


def get_irrigation_status(container, site_id):
    """Get irrigation status for a site from container."""
    try:
        irr_path = "properties/irrigation/irr"
        irr_arr = container.root[irr_path][:]
        field_idx = container.field_uids.index(site_id)
        return bool(irr_arr[field_idx])
    except Exception:
        return False


def compare_openet(fid: str, flux_file: str, model_output: str, openet_dir: str,
                   irr: bool, gap_tolerance: int = 5,
                   ensemble_members: Optional[List[str]] = None):
    """Compare SWIM and OpenET ensemble against flux observations for a single site."""
    openet_daily = os.path.join(openet_dir, "daily_data", f"{fid}.csv")
    openet_monthly = os.path.join(openet_dir, "monthly_data", f"{fid}.csv")

    daily, overpass, monthly = evaluate_openet_site(
        model_output,
        flux_file,
        openet_daily_path=openet_daily,
        openet_monthly_path=openet_monthly,
        irr=irr,
        gap_tolerance=gap_tolerance,
        ensemble_members=ensemble_members,
    )
    return monthly


def _verbose_monthly_summary(site_id: str, monthly: dict,
                             ensemble_members: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Print summary statistics for SWIM vs OpenET comparison."""
    rmse_all = {
        k.split("_", 1)[1]: v
        for k, v in monthly.items()
        if isinstance(k, str) and k.startswith("rmse_") and isinstance(v, (int, float))
    }
    if not rmse_all:
        return None, None

    best_overall_model = min(rmse_all, key=rmse_all.get)

    best_pair_model = None
    if "rmse_swim" in monthly and "rmse_openet" in monthly:
        best_pair_model = "swim" if monthly["rmse_swim"] <= monthly["rmse_openet"] else "openet"

    n_samples = monthly.get("n_samples")
    print(f"n Samples: {n_samples}")
    print("Best overall:", best_overall_model)
    print("Best swim vs openet:", best_pair_model if best_pair_model else "NA")

    print(f"Flux Mean: {monthly.get('mean_flux')}")
    print(f"SWIM Mean: {monthly.get('mean_swim')}")
    print(f"SWIM RMSE: {monthly.get('rmse_swim')}")
    print(f"OpenET Ensemble Mean: {monthly.get('mean_openet')}")
    print(f"OpenET Ensemble RMSE: {monthly.get('rmse_openet')}")
    print(f"{best_overall_model} Mean: {monthly.get(f'mean_{best_overall_model}')}")
    print(f"{best_overall_model} RMSE: {monthly.get(f'rmse_{best_overall_model}')}")

    # Show individual ensemble member stats if provided
    if ensemble_members:
        print("Ensemble member comparisons:")
        for member in ensemble_members:
            rmse_key = f"rmse_{member}"
            mean_key = f"mean_{member}"
            if rmse_key in monthly:
                print(f"  {member.upper()} RMSE: {monthly.get(rmse_key)}, Mean: {monthly.get(mean_key)}")

    return best_overall_model, best_pair_model


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    target_dir = os.path.join(cfg.project_ws, "diy_ensemble")

    # Check for calibration parameters (forecasting mode)
    cfg.forecast_parameters_csv = os.path.join(target_dir, f"{cfg.project_name}.3.par.csv")
    cfg.spinup = os.path.join(target_dir, "spinup.json")

    openet_dir = os.path.join(cfg.data_dir, "openet_flux")
    flux_dir = os.path.join(cfg.data_dir, "daily_flux_files")

    station_metadata = os.path.join(cfg.data_dir, "station_metadata.csv")
    ec_sites, sdf = get_flux_sites(station_metadata, crop_only=True, return_df=True, western_only=False, header=1)

    cfg.calibrate = False
    cfg.forecast = True
    cfg.read_forecast_parameters()

    if os.path.exists(cfg.forecast_parameters_csv):
        modified = datetime.fromtimestamp(os.path.getmtime(cfg.forecast_parameters_csv))
        print(f"Calibration made {modified}")

    # Open container
    container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    container = SwimContainer.open(container_path, mode='r')

    try:
        # Get sites available in container that are also EC sites
        available_sites = set(container.field_uids)
        sites = [s for s in ec_sites if s in available_sites]

        print(f"{len(sites)} sites to evaluate in 5_Flux_Ensemble")

        # Use ensemble_members for comparison models
        ensemble_members = getattr(cfg, 'etf_ensemble_members', ['ssebop', 'ptjpl', 'sims'])
        print(f"Ensemble members for comparison: {ensemble_members}")

        incomplete, complete = [], []
        results_overall, results_pair = [], []
        results = {}

        for ee, site_id in enumerate(sites):
            try:
                lulc = sdf.at[site_id, "General classification"]
            except Exception:
                lulc = "NA"

            print(f"\n{ee} {site_id}: {lulc}")

            out_csv = os.path.join(target_dir, f"{site_id}.csv")

            start_time = time.time()
            try:
                df = run_single_site(cfg, container, site_id, out_csv)
            except Exception as exc:
                print(f"Error running {site_id}: {exc}")
                incomplete.append(site_id)
                continue
            end_time = time.time()
            print(f"Execution time: {end_time - start_time:.2f} seconds")

            flux_file = os.path.join(flux_dir, f"{site_id}_daily_data.csv")
            irr = get_irrigation_status(container, site_id)

            monthly = compare_openet(site_id, flux_file, out_csv, openet_dir, irr,
                                     gap_tolerance=5, ensemble_members=ensemble_members)
            if monthly and isinstance(monthly, dict):
                results[site_id] = monthly
                best_overall, best_pair = _verbose_monthly_summary(site_id, monthly,
                                                                   ensemble_members=ensemble_members)
                if best_overall:
                    results_overall.append((best_overall, lulc))
                if best_pair:
                    results_pair.append((best_pair, lulc))
                complete.append(site_id)
            else:
                incomplete.append(site_id)

    finally:
        container.close()

    pprint({k: v.get("rmse_swim") for k, v in results.items() if isinstance(v, dict)})
    pprint({s: [t[0] for t in results_overall].count(s) for s in set(t[0] for t in results_overall)})
    pprint(
        {
            category: [
                item[0]
                for item in collections.Counter(t[0] for t in results_overall if t[1] == category).most_common(3)
            ]
            for category in set(t[1] for t in results_overall)
        }
    )
    pprint({s: [t[0] for t in results_pair].count(s) for s in set(t[0] for t in results_pair)})
    pprint(
        {
            category: [
                item[0]
                for item in collections.Counter(t[0] for t in results_pair if t[1] == category).most_common(3)
            ]
            for category in set(t[1] for t in results_pair)
        }
    )
    print(f"complete: {complete}")
    print(f"incomplete: {incomplete}")

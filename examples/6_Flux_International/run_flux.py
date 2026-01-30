"""
Flux International Evaluation Script

Evaluates SWIM model against flux tower observations for international sites.
Uses the modern SwimContainer + process package workflow.
"""

import collections
import os
import tempfile
import time
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from ssebop_evaluation import evaluate_ssebop_site

from swimrs.calibrate.flux_utils import get_flux_sites
from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
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
    start_time = time.time()

    # Create temporary HDF5 for SwimInput
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5_path = tmp.name

    try:
        # Build SwimInput from container for single site
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5_path,
            runoff_process=getattr(config, "runoff_process", "cn"),
            etf_model=getattr(config, "etf_target_model", "ptjpl"),
            met_source=getattr(config, "met_source", "era5"),
            fields=[site_id],
        )

        # Run simulation
        output, final_state = run_daily_loop(swim_input)

        # Get time series data
        n_days = swim_input.n_days
        dates = pd.date_range(swim_input.start_date, periods=n_days, freq="D")

        # Get input time series
        etr = swim_input.get_time_series("etr")
        prcp = swim_input.get_time_series("prcp")
        tmin = swim_input.get_time_series("tmin")
        tmax = swim_input.get_time_series("tmax")
        ndvi = swim_input.get_time_series("ndvi")

        # Build DataFrame (field index 0 since we're doing single field)
        i = 0
        df_data = {
            # Model outputs
            "et_act": output.eta[:, i],
            "etref": etr[:, i],
            "kc_act": output.etf[:, i],
            "kc_bas": output.kcb[:, i],
            "ks": output.ks[:, i],
            "ke": output.ke[:, i],
            "melt": output.melt[:, i],
            "rain": output.rain[:, i],
            "depl_root": output.depl_root[:, i],
            "dperc": output.dperc[:, i],
            "runoff": output.runoff[:, i],
            "swe": output.swe[:, i],
            "irrigation": output.irr_sim[:, i],
            "gw_sim": output.gw_sim[:, i],
            # Input time series
            "ppt": prcp[:, i],
            "tmin": tmin[:, i],
            "tmax": tmax[:, i],
            "tavg": (tmin[:, i] + tmax[:, i]) / 2.0,
            "ndvi": ndvi[:, i],
        }

        # Calculate derived columns
        df_data["soil_water"] = swim_input.properties.awc[i] - output.depl_root[:, i]

        # Add ETf observations for comparison
        etf_model = getattr(config, "etf_target_model", "ptjpl")

        for mask in ["inv_irr", "irr"]:
            etf_path = f"remote_sensing/etf/landsat/{etf_model}/{mask}"
            try:
                etf_df = container.query.dataframe(etf_path, fields=[site_id])
                etf_series = etf_df[site_id].reindex(dates)
                df_data[f"etf_{mask}"] = etf_series.values
            except Exception:
                df_data[f"etf_{mask}"] = np.nan

        df = pd.DataFrame(df_data, index=dates)

        # Trim to config date range
        df = df.loc[config.start_dt : config.end_dt]

        swim_input.close()

    finally:
        # Clean up temp file
        if os.path.exists(temp_h5_path):
            os.remove(temp_h5_path)

    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds\n")

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


def compare_ssebop(fid, flux_file, model_output, irr, return_comparison=False, gap_tolerance=5):
    """Compare SWIM and SSEBop against flux observations for a single site."""
    daily, overpass, monthly = evaluate_ssebop_site(
        model_output, flux_file, irr=irr, gap_tolerance=gap_tolerance
    )

    if monthly is None:
        return None

    agg_comp = monthly.copy()
    if len(agg_comp) < 3:
        return None

    rmse_values = {
        k.split("_")[1]: v
        for k, v in agg_comp.items()
        if k.startswith("rmse_")
        if "swim" in k or "ssebop" in k
    }

    if len(rmse_values) == 0:
        return None

    lowest_rmse_model = min(rmse_values, key=rmse_values.get)
    print(f"n Samples: {agg_comp['n_samples']}")
    print("Lowest RMSE:", lowest_rmse_model)

    if not return_comparison:
        return lowest_rmse_model

    try:
        print(f"Flux Mean: {agg_comp['mean_flux']}")
        print(f"SWIM Mean: {agg_comp['mean_swim']}")
        print(f"SSEBop NHM Mean: {agg_comp.get('mean_ssebop')}")
        print(f"SWIM RMSE: {agg_comp['rmse_swim']}")
        print(f"SSEBop NHM RMSE: {agg_comp.get('rmse_ssebop')}")
        return lowest_rmse_model

    except KeyError as exc:
        print(fid, exc)
        return None


if __name__ == "__main__":
    project = "6_Flux_International"

    root = "/data/ssd2/swim"
    data = os.path.join(root, project, "data")
    project_ws_ = os.path.join(root, project)
    if not os.path.isdir(root):
        root = "/home/dgketchum/code/swim-rs"
        project_ws_ = os.path.join(root, "examples", project)
        data = os.path.join(project_ws_, "data")

    config_file = os.path.join(project_ws_, "6_Flux_International.toml")

    station_file = os.path.join(data, "station_metadata.csv")
    sites, sdf = get_flux_sites(station_file, crop_only=False, return_df=True)

    incomplete, complete, results = [], [], []

    overwrite_ = False

    # Load config
    cfg = ProjectConfig()
    cfg.read_config(config_file, project_ws_)

    for ee, site_ in enumerate(sites):
        lulc = sdf.at[site_, "General classification"]

        if site_ in ["US-Bi2", "US-Dk1", "JPL1_JV114"]:
            continue

        if site_ not in ["US-Ro4"]:
            continue

        print(f"\n{ee} {site_}: {lulc}")

        run_const = os.path.join(project_ws_, "results", "verify")
        output_ = os.path.join(run_const, site_)

        # Check for site-specific container or use main container
        site_container_path = os.path.join(output_, f"{site_}.swim")
        main_container_path = os.path.join(data, f"{project}.swim")

        container_path = (
            site_container_path if os.path.exists(site_container_path) else main_container_path
        )

        if not os.path.exists(container_path):
            print(f"Container not found at {container_path}")
            incomplete.append(site_)
            continue

        flux_dir = os.path.join(project_ws_, "data", "daily_flux_files")
        flux_data = os.path.join(flux_dir, f"{site_}_daily_data.csv")

        fcst_params = os.path.join(output_, f"{site_}.3.par.csv")
        if not os.path.exists(fcst_params):
            continue

        modified_date = datetime.fromtimestamp(os.path.getmtime(fcst_params))
        print(f"Calibration made {modified_date}")
        if modified_date < pd.to_datetime("2025-07-01"):
            continue

        out_csv = os.path.join(output_, f"{site_}.csv")

        # Open container and run
        container = SwimContainer.open(container_path, mode="r")

        try:
            # Set forecast parameters on config
            cfg.forecast = True
            cfg.forecast_parameters_csv = fcst_params
            cfg.read_forecast_parameters()

            if not os.path.exists(out_csv) or overwrite_:
                run_single_site(cfg, container, site_, out_csv)

            irr = get_irrigation_status(container, site_)

        except ValueError as exc:
            print(f"{site_} error: {exc}")
            container.close()
            continue
        finally:
            container.close()

        result = compare_ssebop(
            site_, flux_data, out_csv, irr, return_comparison=True, gap_tolerance=5
        )

        if result:
            results.append((result, lulc))

        complete.append(site_)

        out_fig_dir_ = os.path.join(root, "examples", project, "figures", "model_output", "png")

    pprint({s: [t[0] for t in results].count(s) for s in set(t[0] for t in results)})
    pprint(
        {
            category: [
                item[0]
                for item in collections.Counter(
                    t[0] for t in results if t[1] == category
                ).most_common(3)
            ]
            for category in set(t[1] for t in results)
        }
    )
    print(f"complete: {complete}")
    print(f"incomplete: {incomplete}")
# ========================= EOF ====================================================================

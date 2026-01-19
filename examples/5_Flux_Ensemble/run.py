"""
Run SWIM model for individual flux sites using the process package API.

This module uses the modern container-based workflow:
    1. Open SwimContainer
    2. Build SwimInput from container
    3. Run simulation with run_daily_loop
    4. Convert output to DataFrame

Usage:
    python run.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
from swimrs.swim.config import ProjectConfig


def output_to_dataframe(output, swim_input, field_idx: int) -> pd.DataFrame:
    """Convert DailyOutput arrays to DataFrame for a single field.

    Args:
        output: DailyOutput from run_daily_loop()
        swim_input: SwimInput used in simulation
        field_idx: Index of field in swim_input.fids

    Returns:
        DataFrame with daily outputs indexed by date
    """
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
    """Extract input time series for a field.

    Args:
        swim_input: SwimInput container
        field_idx: Index of field in swim_input.fids

    Returns:
        DataFrame with input time series indexed by date
    """
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

    # Add NDVI observations if available
    try:
        ndvi_irr = swim_input.get_time_series('ndvi_irr')
        ndvi_inv_irr = swim_input.get_time_series('ndvi_inv_irr')
        df['ndvi_irr'] = ndvi_irr[:, field_idx]
        df['ndvi_inv_irr'] = ndvi_inv_irr[:, field_idx]
    except (KeyError, ValueError):
        pass

    return df


def run_flux_site(fid: str, cfg: ProjectConfig, container: SwimContainer,
                  overwrite_input: bool = False) -> None:
    """Run SWIM model for a single flux site.

    Args:
        fid: Field/site ID
        cfg: ProjectConfig instance
        container: SwimContainer with ingested data
        overwrite_input: If True, rebuild swim_input.h5 (ignored, always rebuilds)
    """
    start_time = time.time()

    target_dir = os.path.join(cfg.project_ws, "testrun", fid)
    os.makedirs(target_dir, exist_ok=True)

    # Build swim_input.h5 for this site
    h5_path = os.path.join(target_dir, f"swim_input_{fid}.h5")

    swim_input = build_swim_input(
        container,
        output_h5=h5_path,
        spinup_json_path=None,  # Use default spinup
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

    out_csv = os.path.join(target_dir, f"{fid}.csv")
    df.to_csv(out_csv)
    print(f"run complete: {fid}, wrote {out_csv}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))

    # Open container
    container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    if not os.path.exists(container_path):
        raise FileNotFoundError(
            f"Container not found at {container_path}. "
            "Run container_prep.py first to create the container."
        )

    container = SwimContainer.open(container_path, mode='r')

    try:
        run_flux_site("B_01", cfg, container, overwrite_input=True)
    finally:
        container.close()

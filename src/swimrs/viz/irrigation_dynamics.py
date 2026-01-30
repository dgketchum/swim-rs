import json
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from swimrs.calibrate.flux_utils import get_flux_sites
from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig


def irrigation_timeseries(container, fid, out_dir=None):
    """Plot NDVI time series with irrigation event markers for each year.

    Parameters
    ----------
    container : SwimContainer
        Open container with dynamics computed.
    fid : str
        Field UID.
    out_dir : str, optional
        Output directory for figures. Shows interactively if None.
    """
    df = container.query.field_timeseries(fid)

    # Find the NDVI irrigated column
    ndvi_col = None
    for col in df.columns:
        if "ndvi" in col and "irr" in col and "inv" not in col:
            ndvi_col = col
            break

    if ndvi_col is None:
        print(f"No irrigated NDVI column found for {fid}")
        return

    # Get per-year irr_doys from container dynamics
    field_idx = container.state.get_field_index(fid)
    irr_arr = container.state.root["derived/dynamics/irr_data"]
    irr_json = irr_arr[field_idx]
    if hasattr(irr_json, "item"):
        irr_json = irr_json.item()

    irr_data = {}
    if irr_json and isinstance(irr_json, str):
        try:
            irr_data = json.loads(irr_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    years = sorted(set(df.index.year))

    for year in years:
        year_data = irr_data.get(str(year), {})
        irr_doys = year_data.get("irr_doys", []) if isinstance(year_data, dict) else []

        df_year = df.loc[f"{year}-01-01" : f"{year}-12-31"].copy()
        df_year["doy"] = df_year.index.dayofyear

        df_year["ndvi_rolling"] = df_year[ndvi_col].rolling(window=32, center=True).mean()

        irr_dates = [
            pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
            for doy in irr_doys
            if doy in df_year["doy"].tolist()
        ]
        irr_values = df_year.loc[irr_dates, "ndvi_rolling"] if irr_dates else pd.Series()

        fig = make_subplots()

        fig.add_trace(
            go.Scatter(
                x=df_year.index,
                y=df_year["ndvi_rolling"],
                mode="lines",
                name=f"Irrigated NDVI (Smoothed) - {fid} 32-day Mean",
                line=dict(color="green"),
            )
        )

        if len(irr_values) > 0:
            fig.add_trace(
                go.Scatter(
                    x=irr_dates,
                    y=irr_values,
                    mode="markers",
                    name="Potential Irrigation Day",
                    marker=dict(size=12, color="blue", symbol="circle-open", line=dict(width=2)),
                )
            )

        fig.update_layout(
            xaxis_title="Date", yaxis_title="Value", title=f"NDVI Time Series for {year}"
        )

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            if "html" in out_dir:
                fig_file = os.path.join(out_dir, f"{fid}_{year}.html")
                fig.write_html(fig_file)
            else:
                fig_file = os.path.join(out_dir, f"{fid}_{year}.png")
                fig.write_image(fig_file)
            print(fig_file)
        else:
            fig.show()


if __name__ == "__main__":
    project = "5_Flux_Ensemble"

    home = os.path.expanduser("~")
    config_file = os.path.join(home, "code", "swim-rs", "examples", project, f"{project}.toml")

    config = ProjectConfig()
    config.read_config(config_file)

    western_only = project == "5_Flux_Ensemble"

    sites, sdf = get_flux_sites(
        config.station_metadata_csv,
        crop_only=False,
        return_df=True,
        western_only=western_only,
        header=1,
    )

    print(f"{len(sites)} sites to evaluate in {project}")

    container_path = os.path.join(config.data_dir, f"{config.project_name}.swim")
    container = SwimContainer.open(container_path, mode="r")

    try:
        for ee, site_ in enumerate(sites):
            lulc = sdf.at[site_, "General classification"]

            if site_ not in ["ALARC2_Smith6"]:
                continue

            print(f"\n{ee} {site_}: {lulc}")

            out_fig_dir_ = os.path.join(
                os.path.expanduser("~"), "Downloads", "figures", "irrigation", "reorg"
            )

            irrigation_timeseries(container, site_, out_dir=out_fig_dir_)
    finally:
        container.close()
# ========================= EOF ====================================================================

"""
Figure 7a: Site-Level Win Rate Map

160 sites plotted geographically, marker color = SWIM wins (blue) or
SSEBop wins (orange) on monthly R², marker size = magnitude of R² delta.

Usage:
    python paper/figures/fig7a_winrate.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
    }
)

MONTHLY_CSV = "/data/ssd1/swim/4_Flux_Network/results/evaluation_monthly_metrics.csv"
SHP = "/data/ssd1/swim/4_Flux_Network/data/gis/flux_fields.shp"

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig7a_winrate.png"
OUT_PDF = OUT_DIR / "fig7a_winrate.pdf"

SWIM_COLOR = "#4C72B0"
SSEBOP_COLOR = "#DD8452"


def main():
    df = pd.read_csv(MONTHLY_CSV)
    shp = gpd.read_file(SHP, engine="fiona")
    lulc_map = dict(zip(shp["site_id"], shp["lc_class"]))

    # Derive centroids in a projected CRS, then transform to lon/lat for plotting.
    shp_proj = shp.to_crs(epsg=5070)
    centroids = gpd.GeoSeries(shp_proj.geometry.centroid, crs=shp_proj.crs).to_crs(epsg=4326)
    coord_map = dict(zip(shp["site_id"], zip(centroids.x, centroids.y)))

    df["lulc"] = df["fid"].map(lulc_map)
    df["lon"] = df["fid"].map(lambda f: coord_map.get(f, (np.nan, np.nan))[0])
    df["lat"] = df["fid"].map(lambda f: coord_map.get(f, (np.nan, np.nan))[1])
    df["delta"] = df["r2_swim"] - df["r2_ssebop"]
    df["swim_wins"] = df["delta"] > 0
    df = df.dropna(subset=["lon", "lat", "delta"])

    # Background: CONUS state boundaries
    STATES_SHP = "/nas/boundaries/states/us_state_20m/cb_2016_us_state_20m.shp"
    states = gpd.read_file(STATES_SHP, engine="fiona").to_crs(epsg=4326)
    conus = states[~states["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    conus.boundary.plot(ax=ax, linewidth=0.4, color="gray", zorder=1)
    ax.set_xlim(-126, -65)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")

    # Size by |delta|, color by winner
    sizes = np.clip(np.abs(df["delta"].values) * 150, 15, 200)
    colors = [SWIM_COLOR if w else SSEBOP_COLOR for w in df["swim_wins"]]

    ax.scatter(
        df["lon"],
        df["lat"],
        s=sizes,
        c=colors,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
    )

    # Legend
    wins = df["swim_wins"].sum()
    total = len(df)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SWIM_COLOR,
            markersize=10,
            label=f"SWIM wins ({wins}/{total}, {100 * wins / total:.0f}%)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SSEBOP_COLOR,
            markersize=10,
            label=f"SSEBop wins ({total - wins}/{total}, {100 * (total - wins) / total:.0f}%)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            label="Small |R² delta|",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=12,
            label="Large |R² delta|",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9, fontsize=9)

    ax.set_title(f"Monthly R²: SWIM vs SSEBop ({total} paired sites)", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()

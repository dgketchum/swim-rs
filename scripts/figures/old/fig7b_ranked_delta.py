"""
Figure 7b: Ranked Site-Level R² Delta (Lollipop Chart)

Sites ranked by monthly R² delta (SWIM - SSEBop), colored by LULC class.
Horizontal lollipops with vertical line at zero.

Usage:
    python paper/figures/fig7b_ranked_delta.py
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
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
    }
)

MONTHLY_CSV = "/data/ssd1/swim/4_Flux_Network/results/evaluation_monthly_metrics.csv"
SHP = "/data/ssd1/swim/4_Flux_Network/data/gis/flux_fields.shp"

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig7b_ranked_delta.png"
OUT_PDF = OUT_DIR / "fig7b_ranked_delta.pdf"

LULC_COLORS = {
    "Croplands": "#4C72B0",
    "Grasslands": "#55A868",
    "Shrublands": "#C4A24D",
    "Evergreen Forests": "#1B7837",
    "Mixed Forests": "#8172B2",
    "Wetland/Riparian": "#64B5CD",
}


def main():
    df = pd.read_csv(MONTHLY_CSV)
    shp = gpd.read_file(SHP, engine="fiona")
    lulc_map = dict(zip(shp["site_id"], shp["lc_class"]))

    df["lulc"] = df["fid"].map(lulc_map)
    df["delta"] = df["r2_swim"] - df["r2_ssebop"]
    df = df.dropna(subset=["lulc", "delta"])

    # Sort by delta descending (best SWIM advantage at top)
    df = df.sort_values("delta", ascending=True).reset_index(drop=True)

    n = len(df)
    y_pos = np.arange(n)
    colors = [LULC_COLORS.get(lulc, "gray") for lulc in df["lulc"]]

    fig, _ = plt.subplots(figsize=(7, 0.12 * n + 1.5))
    ax = fig.axes[0]

    # Lollipop: stem + dot
    ax.hlines(y_pos, 0, df["delta"], colors=colors, linewidth=0.8, alpha=0.7)
    ax.scatter(df["delta"], y_pos, c=colors, s=15, zorder=5, edgecolors="white", linewidths=0.3)

    # Zero line
    ax.axvline(0, color="black", lw=0.8, ls="-")

    # Site labels on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["fid"], fontsize=5.5)
    ax.set_xlabel("Monthly R² Delta (SWIM − SSEBop)")
    ax.set_title("Site-Level Monthly R² Advantage", loc="left", fontweight="bold")

    # Shade SWIM-wins region
    ax.axvspan(0, df["delta"].max() * 1.1, color=LULC_COLORS["Croplands"], alpha=0.03)
    ax.axvspan(df["delta"].min() * 1.1, 0, color="#DD8452", alpha=0.03)

    ax.text(
        0.02,
        0.01,
        "← SSEBop better",
        transform=ax.transAxes,
        fontsize=8,
        color="#DD8452",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.98,
        0.01,
        "SWIM better →",
        transform=ax.transAxes,
        fontsize=8,
        color=LULC_COLORS["Croplands"],
        ha="right",
        va="bottom",
    )

    # Legend for LULC colors
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=lulc)
        for lulc, c in LULC_COLORS.items()
    ]
    wins = (df["delta"] > 0).sum()
    legend_elements.append(
        Line2D(
            [0], [0], marker="", color="w", label=f"SWIM wins {wins}/{n} ({100 * wins / n:.0f}%)"
        )
    )
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Sites: {n}, SWIM wins: {wins} ({100 * wins / n:.0f}%)")


if __name__ == "__main__":
    main()

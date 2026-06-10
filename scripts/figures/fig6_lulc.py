"""
Figure 6: Stratified Performance by Land Cover (NOVELTY FIGURE)

Box plots of site-level monthly R² for SWIM vs SSEBop, faceted by LULC class.
160 sites from Example 4, 6 land cover classes.

Usage:
    python paper/figures/fig6_lulc.py
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
OUT_PNG = OUT_DIR / "fig6_lulc.png"
OUT_PDF = OUT_DIR / "fig6_lulc.pdf"

LULC_ORDER = [
    "Croplands",
    "Grasslands",
    "Shrublands",
    "Evergreen Forests",
    "Mixed Forests",
    "Wetland/Riparian",
]

SWIM_COLOR = "#4C72B0"
SSEBOP_COLOR = "#DD8452"

Y_FLOOR = -2.0  # Clip display below this; annotate clipped outliers


def main():
    # Load data — filter to strictly paired sites (both SWIM and SSEBop finite)
    df = pd.read_csv(MONTHLY_CSV)
    shp = gpd.read_file(SHP, engine="fiona")
    lulc_map = dict(zip(shp["site_id"], shp["lc_class"]))
    df["lulc"] = df["fid"].map(lulc_map)
    df = df.dropna(subset=["lulc", "r2_swim", "r2_ssebop"])

    fig, _ = plt.subplots(2, 3, figsize=(13, 7))
    axes = fig.axes

    for idx, lulc in enumerate(LULC_ORDER):
        ax = axes[idx]
        grp = df[df["lulc"] == lulc]
        n = len(grp)

        swim_vals = grp["r2_swim"].dropna().values
        ssebop_vals = grp["r2_ssebop"].dropna().values

        # Identify outliers below floor (will annotate separately)
        swim_clipped = swim_vals[swim_vals < Y_FLOOR]
        ssebop_clipped = ssebop_vals[ssebop_vals < Y_FLOOR]

        # Clip for box plot display
        swim_display = np.clip(swim_vals, Y_FLOOR, None)
        ssebop_display = np.clip(ssebop_vals, Y_FLOOR, None)

        bp = ax.boxplot(
            [swim_display, ssebop_display],
            positions=[0, 1],
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
            medianprops=dict(color="black", lw=1.5),
            whiskerprops=dict(color="gray", lw=1),
            capprops=dict(color="gray", lw=1),
        )

        bp["boxes"][0].set_facecolor(SWIM_COLOR)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(SSEBOP_COLOR)
        bp["boxes"][1].set_alpha(0.7)

        # Annotate clipped outliers at floor with downward arrow and value
        for clipped, x_pos in [(swim_clipped, 0), (ssebop_clipped, 1)]:
            if len(clipped) > 0:
                worst = min(clipped)
                count = len(clipped)
                label = f"{count}× below\n({worst:.1f})" if count > 1 else f"({worst:.1f})"
                ax.annotate(
                    label,
                    xy=(x_pos, Y_FLOOR),
                    xytext=(x_pos + 0.3, Y_FLOOR + 0.15),
                    fontsize=7,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                )

        ax.set_ylim(Y_FLOOR - 0.15, 1.1)

        # Win rate
        paired = grp.dropna(subset=["r2_swim", "r2_ssebop"])
        wins = (paired["r2_swim"] > paired["r2_ssebop"]).sum()
        total = len(paired)
        pct = 100 * wins / total if total > 0 else 0

        ax.set_title(f"{lulc} (n={n})", fontweight="bold", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SWIM", "SSEBop"])
        ax.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.5)

        # Win rate annotation
        ax.text(
            0.5,
            0.03,
            f"SWIM wins {wins}/{total} ({pct:.0f}%)",
            transform=ax.transAxes,
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Shared y-axis label
    axes[0].set_ylabel("Monthly R²")
    axes[3].set_ylabel("Monthly R²")

    n_total = len(df)
    fig.suptitle(
        f"Monthly ET Performance: SWIM vs SSEBop by Land Cover ({n_total} paired sites)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    # Summary table
    print(f"\n{'LULC':25s} {'n':>3s} {'SWIM med':>9s} {'SSEBop med':>11s} {'Wins':>8s}")
    for lulc in LULC_ORDER:
        grp = df[df["lulc"] == lulc]
        paired = grp.dropna(subset=["r2_swim", "r2_ssebop"])
        wins = (paired["r2_swim"] > paired["r2_ssebop"]).sum()
        print(
            f"{lulc:25s} {len(grp):3d}"
            f" {grp['r2_swim'].median():9.3f}"
            f" {grp['r2_ssebop'].median():11.3f}"
            f" {wins}/{len(paired)}"
        )


if __name__ == "__main__":
    main()

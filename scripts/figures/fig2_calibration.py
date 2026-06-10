"""
Figure 2: Calibration Mechanics

(a) Phi convergence across PEST++ IES iterations for Ex4 and Ex5.
(b-d) Prior vs posterior parameter distributions for aw, ndvi_0, mad by land cover.

Usage:
    python paper/figures/fig2_calibration.py
"""

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
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

# Paths
PHI_EX4 = "/data/ssd1/swim/4_Flux_Network/results/4_Flux_Network.phi.meas.csv"
PHI_EX5 = "/data/ssd1/swim/5_Flux_Ensemble/results/run13_mad_ensemble/5_Flux_Ensemble.phi.meas.csv"
PAR_EX4 = "/data/ssd1/swim/4_Flux_Network/results/4_Flux_Network.3.par.csv"
SHP_EX4 = "/data/ssd1/swim/4_Flux_Network/data/gis/flux_fields.shp"

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig2_calibration.png"
OUT_PDF = OUT_DIR / "fig2_calibration.pdf"

# Prior bounds from pest_builder.py:933-1029
PARAM_INFO = {
    "aw": {"lower": 100.0, "upper": 400.0, "initial": 150.0, "label": "Available Water (mm)"},
    "ndvi_0": {"lower": 0.1, "upper": 0.80, "initial": 0.55, "label": "NDVI Inflection Point"},
    "mad": {"lower": 0.10, "upper": 0.90, "initial": 0.45, "label": "Management Allowed Depletion"},
}

LULC_ORDER = [
    "Croplands",
    "Grasslands",
    "Shrublands",
    "Evergreen Forests",
    "Mixed Forests",
    "Wetland/Riparian",
]

LULC_SHORT = {
    "Croplands": "Crop",
    "Grasslands": "Grass",
    "Shrublands": "Shrub",
    "Evergreen Forests": "Evrgn",
    "Mixed Forests": "Mixed",
    "Wetland/Riparian": "Wetland",
}

LULC_COLORS = {
    "Croplands": "#4C72B0",
    "Grasslands": "#55A868",
    "Shrublands": "#C4A24D",
    "Evergreen Forests": "#1B7837",
    "Mixed Forests": "#8172B2",
    "Wetland/Riparian": "#64B5CD",
}


def load_phi(path):
    df = pd.read_csv(path)
    return df[["iteration", "mean", "standard_deviation"]].copy()


def parse_par_columns(columns):
    """Parse parameter columns to extract (param_name, site_id) tuples."""
    pat = re.compile(r"pname:p_(.+?)_([^_:]+)_:0_")
    result = {}
    for col in columns:
        m = pat.match(col)
        if m:
            param, site = m.group(1), m.group(2)
            result[col] = (param, site)
    return result


def load_posterior_long(par_path, shp_path, target_params):
    """Load posterior par.csv and reshape to long format with LULC."""
    shp = gpd.read_file(shp_path, engine="fiona")
    lulc_map = {sid.lower(): lc for sid, lc in zip(shp["site_id"], shp["lc_class"])}

    par = pd.read_csv(par_path, index_col=0)
    col_map = parse_par_columns(par.columns)

    records = []
    for col, (param, site) in col_map.items():
        if param not in target_params:
            continue
        lulc = lulc_map.get(site)
        if lulc is None:
            continue
        vals = par[col].dropna().values
        for v in vals:
            records.append({"param": param, "site": site, "lulc": lulc, "value": float(v)})

    return pd.DataFrame(records)


def plot_phi(ax, phi_ex4, phi_ex5):
    """Panel (a): normalized phi convergence."""
    for phi, label, color in [
        (phi_ex4, "Ex4: 160 sites, SSEBop target", "#4C72B0"),
        (phi_ex5, "Ex5: 60 sites, Ensemble target", "#DD8452"),
    ]:
        x = phi["iteration"].values
        y_norm = phi["mean"].values / phi["mean"].values[0]
        std_norm = phi["standard_deviation"].values / phi["mean"].values[0]

        ax.plot(x, y_norm, "o-", color=color, lw=2, markersize=6, label=label)
        ax.fill_between(x, y_norm - std_norm, y_norm + std_norm, color=color, alpha=0.15)

    ax.set_xlabel("PEST++ IES Iteration")
    ax.set_ylabel("Normalized Mean $\\Phi$ (fraction of prior)")
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.set_title("(a) Objective Function Convergence", loc="left", fontweight="bold")


def plot_param(ax, df_long, param, info):
    """Panel for one parameter: box plots by LULC."""
    sub = df_long[df_long["param"] == param].copy()
    sub["lulc"] = pd.Categorical(sub["lulc"], categories=LULC_ORDER, ordered=True)
    sub = sub.dropna(subset=["lulc"]).sort_values("lulc")

    positions = []
    box_data = []
    colors = []
    tick_labels = []
    for i, lulc in enumerate(LULC_ORDER):
        grp = sub[sub["lulc"] == lulc]["value"]
        if len(grp) == 0:
            continue
        positions.append(i)
        box_data.append(grp.values)
        colors.append(LULC_COLORS[lulc])
        n = len(set(sub[sub["lulc"] == lulc]["site"]))
        tick_labels.append(f"{LULC_SHORT[lulc]}\n(n={n})")

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.5),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(info["lower"], ls="--", color="red", lw=0.8, alpha=0.6)
    ax.axhline(info["upper"], ls="--", color="red", lw=0.8, alpha=0.6)
    ax.axhline(info["initial"], ls=":", color="black", lw=1.0, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_ylabel(info["label"])


def main():
    phi_ex4 = load_phi(PHI_EX4)
    phi_ex5 = load_phi(PHI_EX5)
    df_long = load_posterior_long(PAR_EX4, SHP_EX4, set(PARAM_INFO.keys()))

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.30)

    ax_phi = fig.add_subplot(gs[0, :])
    plot_phi(ax_phi, phi_ex4, phi_ex5)

    params = list(PARAM_INFO.keys())
    panel_labels = ["(b)", "(c)", "(d)"]
    for i, param in enumerate(params):
        ax = fig.add_subplot(gs[1, i])
        plot_param(ax, df_long, param, PARAM_INFO[param])
        ax.set_title(
            f"{panel_labels[i]} {PARAM_INFO[param]['label']}", loc="left", fontweight="bold"
        )

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()

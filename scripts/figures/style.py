"""Shared style constants and helpers for SWIM-RS publication figures."""

from pathlib import Path

import matplotlib as mpl

MM_PER_INCH = 25.4


def mm_to_in(mm):
    return mm / MM_PER_INCH


# ---------- Figure / panel dimensions (mm) ----------
FIG_WIDTH_MM = 190
FIG_HEIGHT_MM = 120

PANEL_A_W_MM, PANEL_A_H_MM = 86, 82
PANEL_B_W_MM, PANEL_B_H_MM = 100, 82
PANEL_C_W_MM, PANEL_C_H_MM = 190, 34

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_DIR = REPO_ROOT / "paper" / "figures" / "fig1_handoff"
CARTOGRAPHIC_DIR = REPO_ROOT / "data" / "cartographic"
WORLD_SHP = CARTOGRAPHIC_DIR / "ne_110m_admin_0_countries.shp"
STATES_SHP = Path("/data/hdd1/data/Hoylman/hoylman_calle_data/shp/states.shp")

# Site shapefiles — use config fields_shapefile (calibration input) for E1/E2,
# publication cohort for E3 (75 sites matching manuscript Table 1).
E1_SHP = Path("/data/ssd1/swim/4_Flux_Network/data/gis/flux_fields.shp")
E2_SHP = Path("/data/ssd1/swim/5_Flux_Ensemble/data/gis/flux_fields.shp")
E3_SHP = Path("/data/ssd1/swim/6_Flux_International/data/gis/flux_crop_pub_75_150m.shp")

# ---------- Land-cover colors (Tol qualitative, from plan) ----------
LULC_COLORS = {
    "Cropland": "#117733",
    "Croplands": "#117733",
    "Grassland": "#88CCEE",
    "Grasslands": "#88CCEE",
    "Shrubland": "#DDCC77",
    "Shrublands": "#DDCC77",
    "Evergreen forest": "#332288",
    "Evergreen Forests": "#332288",
    "Mixed forest": "#AA4499",
    "Mixed Forests": "#AA4499",
    "Wetland/riparian": "#CC6677",
    "Wetland/Riparian": "#CC6677",
    "Other": "#999999",
}

# Canonical display names (shapefile names -> figure labels)
LULC_DISPLAY = {
    "Croplands": "Cropland",
    "Grasslands": "Grassland",
    "Shrublands": "Shrubland",
    "Evergreen Forests": "Evergreen forest",
    "Mixed Forests": "Mixed forest",
    "Wetland/Riparian": "Wetland/riparian",
}

# Ordered for legend
LULC_ORDER = [
    "Cropland",
    "Grassland",
    "Shrubland",
    "Evergreen forest",
    "Mixed forest",
    "Wetland/riparian",
]

# ---------- Experiment accent colors ----------
EXPERIMENT_COLORS = {
    "E1": "#4477AA",
    "E2": "#228833",
    "E3-LS": "#EE6677",
    "E3-Triple": "#CCBB44",
}

# ---------- Architecture region fills ----------
REGION_COLORS = {
    "inputs": "#DCEAF7",
    "model": "#FFF9DB",
    "calibration": "#FDECEC",
    "validation": "#E8E8E8",
}


def set_publication_style():
    """Set matplotlib rcParams for publication-quality vector output."""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Source Sans 3", "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300


def panel_label(ax, label, x=-0.02, y=1.04):
    """Add a bold panel label like (a), (b), (c) at top-left."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def export_figure(fig, stem):
    """Export figure as SVG, PDF, and PNG to *stem* (pathlib.Path, no suffix)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    for suffix, kwargs in [
        (".svg", dict(transparent=True)),
        (".pdf", dict(transparent=True)),
        (".png", dict(dpi=300, transparent=False)),
    ]:
        path = stem.with_suffix(suffix)
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0.01, **kwargs)
        print(f"  {path}")

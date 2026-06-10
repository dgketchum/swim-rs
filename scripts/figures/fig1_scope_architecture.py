#!/usr/bin/env python3
"""
Figure 1 — Experimental scope and SWIM-RS architecture.

Generates three independent vector panels for artist handoff plus an assembly
reference.  Run from repo root:

    uv run python scripts/figures/fig1_scope_architecture.py --panel all
    uv run python scripts/figures/fig1_scope_architecture.py --panel a
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.style import (
    E1_SHP,
    E2_SHP,
    E3_SHP,
    EXPERIMENT_COLORS,
    HANDOFF_DIR,
    LULC_COLORS,
    LULC_DISPLAY,
    LULC_ORDER,
    PANEL_A_H_MM,
    PANEL_A_W_MM,
    PANEL_B_H_MM,
    PANEL_B_W_MM,
    PANEL_C_H_MM,
    PANEL_C_W_MM,
    REGION_COLORS,
    STATES_SHP,
    WORLD_SHP,
    export_figure,
    mm_to_in,
    panel_label,
    set_publication_style,
)

# ── CRS ──────────────────────────────────────────────────────────────────────
CRS_CONUS = "EPSG:5070"  # NAD83 Conus Albers

# ── manuscript site counts (used for annotation, not filtering) ──────────────
N_E1 = 151
N_E2 = 58
N_E3 = 71


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_e1_e2_sites():
    """Return E1 GeoDataFrame (projected to CONUS Albers) with 'is_e2' flag."""
    e1 = gpd.read_file(E1_SHP, engine="fiona").to_crs(CRS_CONUS)
    # Use centroids for plotting (geometries are field polygons)
    e1["geometry"] = e1.geometry.centroid
    e2_ids = set(gpd.read_file(E2_SHP, engine="fiona")["site_id"])
    e1["is_e2"] = e1["site_id"].isin(e2_ids)
    # Normalize land-cover names
    e1["lc_display"] = e1["lc_class"].map(LULC_DISPLAY).fillna("Other")
    e1["lc_color"] = e1["lc_display"].map(LULC_COLORS).fillna("#999999")
    print(f"E1 sites loaded: {len(e1)}  (E2 subset: {e1['is_e2'].sum()})")
    for lc in LULC_ORDER:
        n = (e1["lc_display"] == lc).sum()
        if n:
            print(f"  {lc}: {n}")
    return e1


def load_e3_sites():
    """Return E3 GeoDataFrame in WGS84 (EPSG:4326) with point geometry."""
    import warnings

    e3 = gpd.read_file(E3_SHP, engine="fiona")
    if e3.crs is None or e3.crs.to_epsg() != 4326:
        e3 = e3.to_crs("EPSG:4326")
    # Use centroids if geometries are polygons (small fields, WGS84 centroid is fine)
    if not all(e3.geometry.geom_type == "Point"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            e3["geometry"] = e3.geometry.centroid
    print(f"E3 sites loaded: {len(e3)}")
    return e3


def load_states():
    """Return lower-48 state boundaries projected to CONUS Albers."""
    states = gpd.read_file(STATES_SHP, engine="fiona")
    exclude = {"Alaska", "Hawaii", "Puerto Rico"}
    states = states[~states["STATE_NAME"].isin(exclude)]
    return states.to_crs(CRS_CONUS)


def load_world():
    """Return world country boundaries in WGS84."""
    if not WORLD_SHP.exists():
        raise FileNotFoundError(
            f"Natural Earth shapefile not found at {WORLD_SHP}.\n"
            "Download from https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip and extract to data/cartographic/."
        )
    return gpd.read_file(WORLD_SHP, engine="fiona")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A — SITE GEOGRAPHY AND EXPERIMENT COHORTS
# ═══════════════════════════════════════════════════════════════════════════════
def draw_panel_a(out_dir):
    """CONUS map with E1/E2 + world inset with E3."""
    set_publication_style()
    e1 = load_e1_e2_sites()
    e3 = load_e3_sites()
    states = load_states()
    world = load_world()

    fig_w = mm_to_in(PANEL_A_W_MM)
    fig_h = mm_to_in(PANEL_A_H_MM)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── main CONUS axes ──────────────────────────────────────────────────────
    ax = fig.add_axes([0.0, 0.24, 1.0, 0.76])  # leave room at bottom
    states.plot(ax=ax, facecolor="#F2F2F2", edgecolor="#CCCCCC", linewidth=0.3)

    # E1 sites (non-E2)
    e1_only = e1[~e1["is_e2"]]
    ax.scatter(
        e1_only.geometry.x,
        e1_only.geometry.y,
        c=e1_only["lc_color"],
        s=18,
        linewidths=0.4,
        edgecolors="white",
        zorder=3,
    )

    # E2 sites (overlay with black ring)
    e2 = e1[e1["is_e2"]]
    ax.scatter(
        e2.geometry.x,
        e2.geometry.y,
        c=e2["lc_color"],
        s=18,
        linewidths=0.4,
        edgecolors="white",
        zorder=4,
    )
    ax.scatter(
        e2.geometry.x,
        e2.geometry.y,
        facecolors="none",
        s=42,
        linewidths=0.8,
        edgecolors="black",
        zorder=5,
    )

    # Clip to CONUS extent with margin
    xmin, ymin, xmax, ymax = states.total_bounds
    dx, dy = (xmax - xmin) * 0.03, (ymax - ymin) * 0.03
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "(a)")

    # ── count summary ────────────────────────────────────────────────────────
    count_text = (
        f"E1: {N_E1} CONUS sites, 6 LULC classes\n"
        f"E2: {N_E2} cropland sites (ringed)\n"
        f"E3: {N_E3} international sites (inset)"
    )
    ax.text(
        0.02,
        0.02,
        count_text,
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="#CCCCCC", linewidth=0.4, pad=2, alpha=0.9),
        zorder=10,
    )

    # ── LULC legend ──────────────────────────────────────────────────────────
    handles = []
    for lc in LULC_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=LULC_COLORS[lc],
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=5,
                label=lc,
            )
        )
    # E2 ring legend entry
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=6,
            label="E2 subset",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=5.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handletextpad=0.3,
        borderpad=0.4,
        labelspacing=0.3,
    )

    # ── world inset for E3 ───────────────────────────────────────────────────
    ax_inset = fig.add_axes([0.0, 0.0, 1.0, 0.24])
    world.plot(ax=ax_inset, facecolor="#EEEEEE", edgecolor="#CCCCCC", linewidth=0.2)
    ax_inset.scatter(
        e3.geometry.x,
        e3.geometry.y,
        c="#333333",
        s=8,
        marker="D",
        linewidths=0.3,
        edgecolors="white",
        zorder=3,
    )
    ax_inset.set_xlim(-140, 165)
    ax_inset.set_ylim(-50, 70)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")
    ax_inset.text(
        0.02,
        0.92,
        "E3 international sites",
        transform=ax_inset.transAxes,
        fontsize=5.5,
        va="top",
        fontweight="bold",
    )

    print("Panel A:")
    export_figure(fig, out_dir / "fig1a_site_scope")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B — SWIM-RS ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════


# Helper functions for architecture drawing
def _add_box(ax, cx, cy, w, h, text, fc="white", ec="black", fs=6.5, lw=1.0, zorder=3):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color="#222222",
        zorder=zorder + 1,
        linespacing=1.25,
    )
    return box


def _add_region(ax, cx, cy, w, h, label, fc, ec="#AAAAAA", fs=7, lw=1.0):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.03",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy + h / 2 - 0.08,
        label,
        ha="center",
        va="top",
        fontsize=fs,
        fontweight="bold",
        color="#444444",
        zorder=2,
    )


def _add_arrow(
    ax,
    start,
    end,
    color="#555555",
    lw=1.4,
    cs="arc3,rad=0",
    shrinkA=4,
    shrinkB=4,
    zorder=5,
    ms=14,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=ms,
        color=color,
        linewidth=lw,
        connectionstyle=cs,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _add_label(ax, x, y, text, fs=5.5, color="#444444", ha="center", va="center", fw="bold"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight=fw, color=color, zorder=8)


def draw_panel_b(out_dir):
    """SWIM-RS data / model / calibration / validation architecture."""
    set_publication_style()

    fig_w = mm_to_in(PANEL_B_W_MM)
    fig_h = mm_to_in(PANEL_B_H_MM)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    # Use coordinate space proportional to panel aspect ratio (100:82)
    xmax, ymax = 12.2, 10.0
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "(b)", x=0.0, y=1.01)

    # ── box dimensions ──────────────────────────────────────────────────────
    bw_in = 2.55  # input box width
    bh_in = 0.60  # input box height

    # ── REGION: Inputs ──────────────────────────────────────────────────────
    inp_cx = xmax / 2
    _add_region(ax, inp_cx, 9.15, 11.6, 1.4, "Inputs", REGION_COLORS["inputs"], ec="#6699CC")

    inp_y = 8.9
    inp_xs = [1.7, 4.5, 7.3, 10.1]
    _add_box(
        ax,
        inp_xs[0],
        inp_y,
        bw_in,
        bh_in,
        "Vegetation state\nLandsat + S2 NDVI",
        fc="#EEF4FB",
        ec="#4477AA",
        fs=5.5,
    )
    _add_box(
        ax,
        inp_xs[1],
        inp_y,
        bw_in,
        bh_in,
        "Satellite ETf\ncalibration targets",
        fc="#EEF4FB",
        ec="#4477AA",
        fs=5.5,
    )
    _add_box(
        ax,
        inp_xs[2],
        inp_y,
        bw_in,
        bh_in,
        "Meteorology & soils\nGridMET / ERA5, AWC",
        fc="#EEF4FB",
        ec="#4477AA",
        fs=5.5,
    )
    _add_box(
        ax,
        inp_xs[3],
        inp_y,
        bw_in,
        bh_in,
        "Management\nLULC, irrigation",
        fc="#EEF4FB",
        ec="#4477AA",
        fs=5.5,
    )

    # ── REGION: Model core ──────────────────────────────────────────────────
    _add_region(
        ax,
        inp_cx,
        6.55,
        11.6,
        2.7,
        "SWIM-RS daily water balance",
        REGION_COLORS["model"],
        ec="#CCAA44",
    )

    # NDVI -> Kcb
    _add_box(
        ax,
        2.5,
        6.95,
        3.0,
        0.55,
        "NDVI \u2192 sigmoid Kcb\nKcb = Kcmax / (1+exp[\u2212k(NDVI\u2212NDVI\u2080)])",
        fc="#FFFDE7",
        ec="#AA8833",
        fs=5.5,
    )

    # FAO-56 water balance
    _add_box(
        ax,
        6.5,
        6.1,
        3.4,
        1.3,
        "FAO-56 dual Kc\nsurface layer | root zone | deep\n"
        "P + snowmelt \u2212 runoff \u2212 DP\n"
        "\u00b1 irrigation \u00b1 GW subsidy",
        fc="#FFFDE7",
        ec="#AA8833",
        fs=5.5,
    )

    # Daily ET output
    _add_box(
        ax,
        10.5,
        6.95,
        2.2,
        0.55,
        "ETa = Kc_act \u00b7 ETo\nETf = ETa / ETo",
        fc="#FFFDE7",
        ec="#AA8833",
        fs=5.5,
    )

    # Internal model arrows
    _add_arrow(ax, (4.0, 6.95), (4.8, 6.55), color="#888888")
    _add_arrow(ax, (8.2, 6.5), (9.4, 6.95), color="#888888", cs="arc3,rad=-0.1")

    # Input arrows into model
    _add_arrow(ax, (inp_xs[0], inp_y - bh_in / 2), (2.5, 7.25), color="#4477AA", lw=1.2)
    _add_arrow(
        ax,
        (inp_xs[1], inp_y - bh_in / 2),
        (5.2, 6.78),
        color="#4477AA",
        lw=1.2,
        cs="arc3,rad=-0.05",
    )
    _add_arrow(
        ax, (inp_xs[2], inp_y - bh_in / 2), (7.0, 6.78), color="#4477AA", lw=1.2, cs="arc3,rad=0.05"
    )
    _add_arrow(
        ax, (inp_xs[3], inp_y - bh_in / 2), (9.5, 6.95), color="#4477AA", lw=1.2, cs="arc3,rad=0.1"
    )

    # ── REGION: Calibration loop ────────────────────────────────────────────
    _add_region(ax, 4.6, 2.65, 8.6, 3.0, "Calibration", REGION_COLORS["calibration"], ec="#CC6666")

    # Observed ETf
    _add_box(
        ax,
        2.0,
        3.0,
        2.6,
        0.7,
        "Observed ETf\nat satellite\noverpass dates",
        fc="#FFF0F0",
        ec="#CC4444",
        fs=5.5,
    )

    # Residuals
    _add_box(
        ax,
        4.8,
        1.85,
        2.3,
        0.55,
        "\u03a3 w\u1d62(ETf_sim \u2212 ETf_obs)\u00b2",
        fc="#FFF0F0",
        ec="#CC4444",
        fs=5.5,
    )

    # PEST++ IES
    _add_box(
        ax,
        7.8,
        3.0,
        2.6,
        0.7,
        "PEST++ IES\n200 realizations\n8 params/site",
        fc="#FFF0F0",
        ec="#CC4444",
        fs=6,
    )

    # Calibration internal arrows
    _add_arrow(ax, (3.3, 2.65), (3.65, 2.1), color="#CC4444", cs="arc3,rad=0.1")
    _add_arrow(ax, (5.95, 1.85), (6.9, 2.65), color="#CC4444", cs="arc3,rad=0.1")

    # Modeled ETf down from model -> residuals
    _add_arrow(ax, (6.5, 5.45), (5.3, 2.15), color="#CC4444", lw=1.2, cs="arc3,rad=-0.12")
    _add_label(ax, 5.1, 4.1, "simulated\nETf", fs=5, color="#CC4444")

    # Satellite ETf input -> observed ETf (route along left edge)
    _add_arrow(
        ax, (inp_xs[1] - bw_in / 2, inp_y), (2.0, 3.38), color="#CC4444", lw=1.2, cs="arc3,rad=0.4"
    )
    _add_label(ax, 1.0, 6.0, "clear-sky\noverpasses", fs=4.5, color="#CC4444", ha="left")

    # PEST++ -> updated params -> model (feedback loop on right)
    _add_arrow(ax, (7.8, 3.38), (6.5, 5.45), color="#CC4444", lw=1.5, cs="arc3,rad=-0.25")
    _add_label(ax, 8.3, 4.5, "updated\nparams", fs=5, color="#CC4444")

    # ── REGION: Validation (firewall) ───────────────────────────────────────
    _add_region(ax, 10.8, 2.65, 2.2, 3.0, "Validation", REGION_COLORS["validation"], ec="#666666")

    _add_box(
        ax,
        10.8,
        2.8,
        1.8,
        0.7,
        "Flux tower ET\nvalidation only",
        fc="#F0F0F0",
        ec="#555555",
        fs=5.5,
    )

    # Arrow from model to validation
    _add_arrow(ax, (10.5, 6.67), (10.8, 3.18), color="#555555", lw=1.2, cs="arc3,rad=0.06")
    _add_label(ax, 11.5, 5.1, "calibrated\ndaily ET", fs=5, color="#555555")

    # Dashed firewall line
    ax.plot([9.6, 9.6], [1.15, 4.15], color="#888888", linewidth=0.8, linestyle="--", zorder=6)
    _add_label(ax, 9.6, 1.0, "firewall", fs=4.5, color="#888888")

    # ── Outputs at bottom ───────────────────────────────────────────────────
    _add_box(
        ax,
        inp_cx,
        0.55,
        5.0,
        0.50,
        "Outputs: daily ET, ETf, monthly totals, irrigation partitioning",
        fc="white",
        ec="#333333",
        fs=5.5,
        lw=0.8,
    )
    _add_arrow(
        ax,
        (6.5, 5.45),
        (inp_cx, 0.82),
        color="#333333",
        lw=1.0,
        cs="arc3,rad=0.05",
        shrinkA=6,
        shrinkB=4,
    )

    print("Panel B:")
    export_figure(fig, out_dir / "fig1b_swim_architecture")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C — EXPERIMENT DESIGN MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def draw_panel_c(out_dir):
    """Compact experiment design matrix as a styled table."""
    set_publication_style()

    fig_w = mm_to_in(PANEL_C_W_MM)
    fig_h = mm_to_in(PANEL_C_H_MM)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.set_aspect("auto")
    ax.axis("off")
    panel_label(ax, "(c)", x=-0.005, y=1.02)

    # ── Column definitions ───────────────────────────────────────────────────
    col_x = [0.05, 0.65, 1.95, 3.1, 4.55, 6.15, 7.65]  # left edges
    col_w = [0.55, 1.25, 1.1, 1.4, 1.55, 1.45, 2.35]  # widths
    headers = [
        "Experiment",
        "Domain & period",
        "Met / soils",
        "ETf calibration target",
        "Validation\nbenchmark",
        "Sites (validated)",
        "Primary question",
    ]

    # ── Row data ─────────────────────────────────────────────────────────────
    rows = [
        {
            "exp": "E1",
            "domain": "CONUS, 6 LULC classes\n1987\u20132025",
            "met": "GridMET\nSSURGO",
            "etf": "SSEBop NHM ETf",
            "val": "Flux ET;\nSSEBop benchmark",
            "sites": f"{N_E1} CONUS",
            "question": "Generalizes across\nland covers?",
            "color": EXPERIMENT_COLORS["E1"],
        },
        {
            "exp": "E2",
            "domain": "CONUS cropland\n1995\u20132025",
            "met": "GridMET\nSSURGO",
            "etf": "OpenET 6-model\nMAD ensemble ETf",
            "val": "Flux ET;\nOpenET ensemble",
            "sites": f"{N_E2} paired\ncropland",
            "question": "Ensemble ETf improves\ncropland ET?",
            "color": EXPERIMENT_COLORS["E2"],
        },
        {
            "exp": "E3-LS",
            "domain": "International crop/grass\n2013\u20132025",
            "met": "ERA5-Land\nHWSD",
            "etf": "Landsat SSEBop\n+ PT-JPL ETf",
            "val": "Flux ET;\ninterpolated Landsat",
            "sites": f"{N_E3} intl.",
            "question": "Generalizes with\nglobal inputs?",
            "color": EXPERIMENT_COLORS["E3-LS"],
        },
        {
            "exp": "E3-Triple",
            "domain": "Same E3 sites\nw/ ECOSTRESS",
            "met": "ERA5-Land\nHWSD",
            "etf": "Landsat + ECOSTRESS\nPT-JPL ETf",
            "val": "Flux ET;\nE3-LS comparison",
            "sites": f"{N_E3} intl.",
            "question": "Higher sensor density\nimproves calibration?",
            "color": EXPERIMENT_COLORS["E3-Triple"],
        },
    ]

    # ── Draw table ───────────────────────────────────────────────────────────
    row_h = 0.58
    hdr_h = 0.36
    top_y = 3.0

    # Header row
    for i, hdr in enumerate(headers):
        x = col_x[i]
        w = col_w[i]
        rect = plt.Rectangle(
            (x, top_y - hdr_h),
            w,
            hdr_h,
            facecolor="#E0E0E0",
            edgecolor="#AAAAAA",
            linewidth=0.5,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            top_y - hdr_h / 2,
            hdr,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color="#333333",
            zorder=3,
            linespacing=1.15,
        )

    # Data rows
    for r_idx, row in enumerate(rows):
        y_top = top_y - hdr_h - r_idx * row_h
        # Accent stripe on left edge
        accent = plt.Rectangle(
            (col_x[0], y_top - row_h),
            0.04,
            row_h,
            facecolor=row["color"],
            edgecolor="none",
            zorder=4,
        )
        ax.add_patch(accent)

        fields = [
            row["exp"],
            row["domain"],
            row["met"],
            row["etf"],
            row["val"],
            row["sites"],
            row["question"],
        ]
        for i, txt in enumerate(fields):
            x = col_x[i]
            w = col_w[i]
            # Row background
            bg_color = "white" if r_idx % 2 == 0 else "#F8F8F8"
            rect = plt.Rectangle(
                (x, y_top - row_h),
                w,
                row_h,
                facecolor=bg_color,
                edgecolor="#DDDDDD",
                linewidth=0.3,
                zorder=1,
            )
            ax.add_patch(rect)
            # Text
            fs = 6 if i == 0 else 5
            fw = "bold" if i == 0 else "normal"
            ax.text(
                x + w / 2,
                y_top - row_h / 2,
                txt,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=fw,
                color="#222222",
                zorder=3,
                linespacing=1.15,
            )

    print("Panel C:")
    export_figure(fig, out_dir / "fig1c_experiment_matrix")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
def draw_assembly_reference(out_dir):
    """Combine panels into one reference layout (for review, not final art)."""
    set_publication_style()
    from matplotlib.image import imread

    # Read PNGs of individual panels
    img_a = imread(str(out_dir / "fig1a_site_scope.png"))
    img_b = imread(str(out_dir / "fig1b_swim_architecture.png"))
    img_c = imread(str(out_dir / "fig1c_experiment_matrix.png"))

    fig_w = mm_to_in(190)
    fig_h = mm_to_in(125)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Top-left: Panel A
    ax_a = fig.add_axes([0.0, 0.30, 0.46, 0.70])
    ax_a.imshow(img_a)
    ax_a.axis("off")

    # Top-right: Panel B
    ax_b = fig.add_axes([0.47, 0.30, 0.53, 0.70])
    ax_b.imshow(img_b)
    ax_b.axis("off")

    # Bottom: Panel C
    ax_c = fig.add_axes([0.0, 0.0, 1.0, 0.28])
    ax_c.imshow(img_c)
    ax_c.axis("off")

    print("Assembly reference:")
    export_figure(fig, out_dir / "fig1_assembly_reference")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIST NOTES
# ═══════════════════════════════════════════════════════════════════════════════
def write_artist_notes(out_dir):
    notes = """\
# Figure 1 Artist Handoff Notes

## Purpose
Opening figure for Remote Sensing of Environment manuscript. Introduces the
reader to experimental scope, SWIM-RS model architecture, and the three
experiments.

## Final target size
190 mm wide x 110-130 mm tall (full journal width)

## Panel order
(a) Top-left: site geography and experiment cohorts (86 x 82 mm)
(b) Top-right: SWIM-RS architecture diagram (100 x 82 mm)
(c) Bottom: experiment design matrix (190 x 34 mm)
4 mm gutter between panels, 3-5 mm outer margin.

## Color semantics
Land-cover fills (Panel A, reused in caption):
  Cropland:         #117733 (dark green)
  Grassland:        #88CCEE (light blue)
  Shrubland:        #DDCC77 (gold)
  Evergreen forest: #332288 (deep purple)
  Mixed forest:     #AA4499 (magenta)
  Wetland/riparian: #CC6677 (rose)

Experiment accents (Panel C row stripes):
  E1: #4477AA  E2: #228833  E3-LS: #EE6677  E3-Triple: #CCBB44

Architecture regions (Panel B):
  Inputs: light blue (#DCEAF7)
  Model:  light gold (#FFF9DB)
  Calibration: light pink (#FDECEC)
  Validation:  light gray (#E8E8E8)

## Font recommendation
Source Sans 3 or Arial, 7-9 pt body, 10 pt panel labels.

## Elements the artist can adjust freely
- Typography hierarchy, kerning, line spacing
- Exact panel spacing and alignment
- Arrow curvature and routing
- Minor box dimensions and corner radii
- Legend placement and formatting
- Stroke weights
- Visual grouping background tints
- Color refinement (keep semantic distinctions)

## Elements that should NOT change without scientific review
- Site locations on maps
- Site counts (E1: 151, E2: 58, E3: 71)
- Experiment labels and definitions
- Calibration target names (SSEBop NHM, OpenET MAD, etc.)
- Validation-only status of flux towers (flux tower ET is NOT used for calibration)
- Direction of calibration arrows (satellite ETf -> residuals -> PEST++ -> params -> model)
- Which products belong to E1/E2/E3
- E3-LS vs E3-Triple distinction (Landsat-only vs Landsat + ECOSTRESS)
- The dashed firewall separating calibration from validation

## Expected final deliverables
fig1_final.ai, fig1_final.pdf, fig1_final.svg,
fig1_final_300dpi.png, fig1_final_600dpi.tif, fig1_final_outlined_fonts.pdf
"""
    path = out_dir / "fig1_artist_notes.md"
    path.write_text(notes)
    print(f"  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1 panels")
    parser.add_argument(
        "--panel",
        choices=["a", "b", "c", "assembly", "notes", "all"],
        default="all",
        help="Which panel to generate (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HANDOFF_DIR,
        help=f"Output directory (default: {HANDOFF_DIR})",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.panel in ("a", "all"):
        draw_panel_a(args.out_dir)
    if args.panel in ("b", "all"):
        draw_panel_b(args.out_dir)
    if args.panel in ("c", "all"):
        draw_panel_c(args.out_dir)
    if args.panel in ("assembly", "all"):
        draw_assembly_reference(args.out_dir)
    if args.panel in ("notes", "all"):
        write_artist_notes(args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

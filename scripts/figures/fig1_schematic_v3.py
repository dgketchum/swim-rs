"""
Generate publication-quality schematic figure for SWIM-RS (Remote Sensing of Environment).
Portrait orientation, top-to-bottom flow.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(
    ax, x, y, w, h, text, facecolor="white", edgecolor="black", fontsize=9, linewidth=1.5, zorder=3
):
    """Add a rounded box with bold centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#222222",
        zorder=zorder + 1,
        linespacing=1.35,
    )
    return box


def add_group_box(
    ax, x, y, w, h, label, facecolor, edgecolor="#888888", fontsize=11, linewidth=1.8
):
    """Add a group background box with a label at the top."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + h / 2 - 0.12,
        label,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color="#222222",
        zorder=2,
    )
    return box


def add_arrow(
    ax,
    xy_start,
    xy_end,
    label="",
    color="#333333",
    linewidth=2.2,
    connectionstyle="arc3,rad=0",
    fontsize=8,
    label_pos=None,
    label_ha="center",
    shrinkA=6,
    shrinkB=6,
    zorder=5,
    mutation_scale=20,
):
    """Add an arrow between two points with optional label."""
    arrow = FancyArrowPatch(
        xy_start,
        xy_end,
        arrowstyle="->",
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label and label_pos:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha=label_ha,
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#444444",
            zorder=zorder + 1,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0F0F0", edgecolor="none", alpha=0.95),
        )
    return arrow


def main():
    fig = plt.figure(figsize=(8, 11))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#F0F0F0")
    ax.set_facecolor("#F0F0F0")
    ax.set_xlim(0.3, 7.7)
    ax.set_ylim(0.3, 10.9)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # SECTION 1: INPUTS (top)
    # =========================================================================
    add_group_box(ax, 4.0, 9.75, 6.8, 1.65, "Inputs", facecolor="#DCEAF7", edgecolor="#6699CC")

    input_y = 9.55
    bh_in = 0.9

    add_box(
        ax,
        1.85,
        input_y,
        1.95,
        bh_in,
        "Remote Sensing\nLandsat/Sentinel NDVI",
        facecolor="#EEF4FB",
        edgecolor="#4477AA",
        fontsize=8.5,
    )

    add_box(
        ax,
        4.0,
        input_y,
        1.95,
        bh_in,
        "Meteorology\nGridMET / ERA5-Land",
        facecolor="#EEF4FB",
        edgecolor="#4477AA",
        fontsize=8.5,
    )

    add_box(
        ax,
        6.15,
        input_y,
        1.95,
        bh_in,
        "Properties\nSoils, LULC,\nIrrigation",
        facecolor="#EEF4FB",
        edgecolor="#4477AA",
        fontsize=8.5,
    )

    # =========================================================================
    # SECTION 2: SWIM MODEL (middle)
    # =========================================================================
    model_cy = 7.0
    add_group_box(
        ax, 4.0, model_cy, 6.8, 2.8, "SWIM Model", facecolor="#FFF9DB", edgecolor="#CCAA44"
    )

    # NDVI -> Kcb
    ndvi_x, ndvi_y = 2.4, 7.45
    ndvi_w, ndvi_h = 2.1, 0.6
    add_box(
        ax,
        ndvi_x,
        ndvi_y,
        ndvi_w,
        ndvi_h,
        "NDVI \u2192 Kcb\nsigmoid function",
        facecolor="#E8F5E9",
        edgecolor="#558855",
        fontsize=9,
    )

    # Snow / Runoff / Irrigation / GW
    snow_x, snow_y = 2.4, 6.2
    snow_w, snow_h = 2.1, 0.8
    add_box(
        ax,
        snow_x,
        snow_y,
        snow_w,
        snow_h,
        "Snow Melt\nRunoff\nIrrigation\nGW Subsidy",
        facecolor="#E8F5E9",
        edgecolor="#558855",
        fontsize=9,
    )

    # FAO-56 (central model box)
    fao_x, fao_y = 5.6, 6.85
    fao_w, fao_h = 2.3, 0.85
    add_box(
        ax,
        fao_x,
        fao_y,
        fao_w,
        fao_h,
        "FAO-56 Dual Kc\nSoil Water Balance",
        facecolor="#FFFDE7",
        edgecolor="#AA8833",
        fontsize=10,
        linewidth=2.0,
    )

    # Internal model arrows
    add_arrow(
        ax,
        (ndvi_x + ndvi_w / 2, ndvi_y - 0.05),
        (fao_x - fao_w / 2, fao_y + 0.15),
        connectionstyle="arc3,rad=-0.08",
        color="#555555",
    )

    add_arrow(
        ax,
        (snow_x + snow_w / 2, snow_y + 0.1),
        (fao_x - fao_w / 2, fao_y - 0.15),
        connectionstyle="arc3,rad=0.08",
        color="#555555",
    )

    # Input arrows
    add_arrow(
        ax,
        (1.85, input_y - bh_in / 2),
        (ndvi_x, ndvi_y + ndvi_h / 2),
        color="#4477AA",
        linewidth=2.0,
    )

    add_arrow(
        ax,
        (4.0, input_y - bh_in / 2),
        (fao_x - 0.4, fao_y + fao_h / 2),
        color="#4477AA",
        linewidth=2.0,
    )

    add_arrow(
        ax,
        (6.15, input_y - bh_in / 2),
        (fao_x + 0.4, fao_y + fao_h / 2),
        color="#4477AA",
        linewidth=2.0,
        connectionstyle="arc3,rad=-0.05",
    )

    # =========================================================================
    # SECTION 3: CALIBRATION (bottom-left)
    # =========================================================================
    cal_cx, cal_cy = 3.05, 3.0
    cal_w, cal_h = 4.5, 2.55
    add_group_box(
        ax, cal_cx, cal_cy, cal_w, cal_h, "Calibration", facecolor="#FDECEC", edgecolor="#CC6666"
    )

    # PEST++ IES
    pest_x, pest_y = 3.7, 3.35
    pest_w, pest_h = 2.3, 0.75
    add_box(
        ax,
        pest_x,
        pest_y,
        pest_w,
        pest_h,
        "PEST++ IES\n200 realizations\n8 params/site",
        facecolor="#FFF0F0",
        edgecolor="#CC4444",
        fontsize=9,
    )

    # Observed ETf
    obs_x, obs_y = 2.0, 2.3
    obs_w, obs_h = 2.0, 0.6
    add_box(
        ax,
        obs_x,
        obs_y,
        obs_w,
        obs_h,
        "Observed ETf\nat Landsat dates",
        facecolor="#FFF0F0",
        edgecolor="#CC4444",
        fontsize=9,
    )

    # Arrow: FAO-56 -> PEST++ (modeled ETf going down)
    # Comes from right side of FAO-56 bottom, goes to right side of PEST++ top
    add_arrow(
        ax,
        (fao_x, fao_y - fao_h / 2),
        (pest_x + 0.3, pest_y + pest_h / 2),
        label="modeled ETf",
        color="#CC4444",
        connectionstyle="arc3,rad=-0.05",
        label_pos=(4.7, 4.95),
    )

    # Arrow: Observed ETf -> PEST++
    add_arrow(
        ax,
        (obs_x + obs_w / 2, obs_y + 0.12),
        (pest_x - pest_w / 2, pest_y - 0.12),
        color="#CC4444",
        connectionstyle="arc3,rad=0.15",
    )

    # Arrow: PEST++ -> FAO-56 (updated params feedback loop)
    # From left side of PEST++ top, curving left and up to left of FAO-56 bottom
    add_arrow(
        ax,
        (pest_x - pest_w / 2, pest_y + 0.15),
        (fao_x - fao_w / 2 + 0.15, fao_y - fao_h / 2),
        label="updated\nparams",
        color="#CC4444",
        connectionstyle="arc3,rad=0.2",
        label_pos=(2.9, 4.95),
        linewidth=2.2,
    )

    # =========================================================================
    # SECTION 4: VALIDATION (bottom-right)
    # =========================================================================
    val_cx, val_cy = 6.55, 3.0
    val_w, val_h = 1.7, 2.55
    add_group_box(
        ax, val_cx, val_cy, val_w, val_h, "Validation", facecolor="#E8F5E9", edgecolor="#66AA66"
    )

    # Flux Tower ET
    flux_x, flux_y = 6.55, 3.5
    flux_w, flux_h = 1.35, 0.5
    add_box(
        ax,
        flux_x,
        flux_y,
        flux_w,
        flux_h,
        "Flux Tower ET",
        facecolor="#F0FAF0",
        edgecolor="#448844",
        fontsize=9,
    )

    # OpenET
    openet_x, openet_y = 6.55, 2.5
    openet_w, openet_h = 1.35, 0.6
    add_box(
        ax,
        openet_x,
        openet_y,
        openet_w,
        openet_h,
        "OpenET\n6 models + ensemble",
        facecolor="#F0FAF0",
        edgecolor="#448844",
        fontsize=8.5,
    )

    # Arrow: FAO-56 -> Validation (calibrated daily ET)
    add_arrow(
        ax,
        (fao_x + 0.6, fao_y - fao_h / 2),
        (flux_x, flux_y + flux_h / 2),
        label="calibrated\ndaily ET",
        color="#448844",
        connectionstyle="arc3,rad=0.08",
        label_pos=(6.85, 4.95),
        linewidth=2.2,
    )

    # =========================================================================
    # Save
    # =========================================================================
    plt.tight_layout(pad=0.3)

    out_path = "/home/dgketchum/code/swim-rs/paper/figures/fig1_schematic_v3.png"
    fig.savefig(
        out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2
    )
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

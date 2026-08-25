"""Revision-5 SELECTED synthesis -- Study B basis with Study C's compact labels.

Gate A selected Study B's circular inverse-estimation ring (handoff Sec. 1).
This builder is the Sec. 16 Level-1 synthesis against the same frozen data:

1.  STUDY B PRESERVED. Six-row stack with weighted rows, conventional
    quantitative axes, tick numerals centred on their ticks, the circular
    Run/Compare/Update ring inside its restrained tinted field at Study B's
    full 7.9 mm radius and arc angles. The irrigation row rises from 4.8 to
    5.6 mm.

2.  STUDY C's LABEL TREATMENT, WITH UNITS. Two-line compact row labels in a
    15.8 mm gutter, but unlike C every dimensional row carries its unit
    (`mm d⁻¹` or `mm`) as a final label line, and the tick-numeral gutter
    stays distinct with numerals centred on their ticks (Sec. 5.2).

3.  COMPONENT BOX COLUMNS. Beneath the ring, the daily inputs sit in
    green-shaded boxes in a vertical column on the left (under Run, which
    they feed) and the calibrated parameters in blue-shaded boxes in a
    vertical column on the right (under Update, which emits them). Bold
    `Daily Drivers` / `Conditioned Parameters` headers cap each column.
    Parameter names use the FAO-56 symbols the manuscript defines -- Kcb
    (the NDVI-driven basal crop coefficient), Ks (water stress), Kr
    (evaporation reduction), AWC, MAD -- with the caption carrying the
    expansions. One role-level feed into Run, one role-level exit from
    Update; no component-level arrows.

4.  PANEL (b) RENDER OVERRIDES. The irrigated E1->E3 branch becomes the
    orthogonal up-and-over route, the E2 map uses symmetric latitude
    bounds +/-(max |site latitude| + 5 deg) computed from the frozen sites
    (Sec. 6.1/6.5), and the E3 map draws public-domain USGS NAIP
    orthoimagery (georegistered in EPSG:5070, provenance in `assets/`)
    beneath the frozen field marks in place of the flat land fill. All are
    render-level: sources, destinations and geometries are unchanged.

Run:
  uv run python paper/figures/proofs/fig01_evidence_190_r5/build_fig01_r5_selected.py
"""

from __future__ import annotations

from pathlib import Path

import fig01_r5_common as K
import numpy as np
import pandas as pd
from fig01_r5_common import (
    C_AXIS,
    C_DATUM,
    C_ETO,
    C_GUIDE,
    C_HELD,
    C_INV,
    C_MEMBER,
    C_MINMAX,
    C_PRECIP,
    C_SENSOR,
    C_SWIM,
    C_TARGET,
    C_TEXT,
    FS_LABEL,
    FS_PANEL,
    FS_PANEL_HEAD,
    FS_ROW,
    FS_STRUCT,
    FS_TICK,
    H_MM,
    HALO,
    LW_AXIS,
    LW_DATA,
    LW_DATUM,
    LW_GUIDE,
    LW_SPINE,
    MARGIN_MM,
    W_MM,
    arc_pts,
    mmtext,
    polyline,
    tag,
    ticktext,
)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

STUDY = "fig01_r5_selected"

# ===========================================================================
# LAYOUT
# ===========================================================================

A_HEAD_Y = 114.6
RECORD_Y = 110.4

LABEL_R = 15.8  # compact treatment (study C), Gate A wants ~15-18 mm
TICKNUM_R = 21.0
PA_X0 = 22.3
PA_X1 = 149.6
TICK_STUB = 0.9
LABEL_LEAD = 3.0  # mm between label lines (incl. the unit line)

ROW_GUTTER = 3.0
ROWS: dict[str, tuple[float, float]] = {
    "etf_ensemble": (100.6, 107.0),
    "ndvi_captures": (91.0, 97.6),
    "daily_forcing": (81.8, 88.0),
    "rz_depletion": (70.4, 78.8),
    "irrigation": (61.8, 67.4),  # raised from study B's 4.8 mm
    "et_comparison": (51.0, 58.8),
}
ROW_LABEL = {
    "etf_ensemble": ("ETf", "Ensemble"),
    "ndvi_captures": ("NDVI", "Captures"),
    "daily_forcing": ("Daily", "Forcing"),
    "rz_depletion": ("Root-Zone", "Depletion"),
    "irrigation": ("Irrigation",),
    "et_comparison": ("Daily ET",),
}
UNITS = {
    "daily_forcing": "mm d⁻¹",
    "rz_depletion": "mm",
    "irrigation": "mm",
    "et_comparison": "mm d⁻¹",
}
DOMAIN = {
    "etf_ensemble": (0.0, 1.6),
    "ndvi_captures": (0.2, 1.0),
    "daily_forcing": (0.0, 18.0),
    "rz_depletion": (0.0, 18.0),
    "irrigation": (0.0, 25.0),
    "et_comparison": (0.0, 12.0),
}
TICKS = {
    "etf_ensemble": [(0.0, "0.0"), (1.6, "1.6")],
    "ndvi_captures": [(0.2, "0.2"), (1.0, "1.0")],
    "daily_forcing": [(0.0, "0"), (18.0, "18")],
    "rz_depletion": [(0.0, "0"), (9.0, "9"), (18.0, "18")],
    "irrigation": [(0.0, "0"), (25.0, "25")],
    "et_comparison": [(0.0, "0"), (6.0, "6"), (12.0, "12")],
}

DATE_SPINE_Y = 49.2
DATE_LABEL_Y = 46.4
GUIDE_TOP = 107.0
GUIDE_BOT = 51.0

# ---- the ring column ------------------------------------------------------
LOOP_X0, LOOP_X1 = 152.0, 187.0
FIELD = (153.0, 81.6, 33.2, 20.6)  # deep enough for study B's full ring
RING_C = (169.6, 91.0)
RING_R = 7.9  # study B's radius, restored by the two-line inventories
TITLE_Y = 108.6  # 'Inverse Estimation'
# 'ETf + SWE' rides in a red-shaded box styled like the component boxes;
# ONE box, ONE label: the architecture forbids a separate swe edge into
# Compare, so the snow constraint stays inline with the ETf constraint.
SWE_BOX = (161.6, 103.2, 16.0, 3.8)  # x0, y0, w, h
SWE_FILL, SWE_EDGE = "#F8E4E2", "#D9A49E"
ANG = {"compare": 90.0, "update_parameters": -30.0, "run_balance": 210.0}
ARCS = {  # clockwise, with a word-width gap at each end (study B, r=7.9)
    "cycle-run-to-compare": (190.0, 126.0),
    "cycle-compare-to-update": (54.0, -7.0),
    "cycle-update-to-run": (-53.0, -127.0),
}
DRV_ARROW = ((160.6, 79.8), (162.3, 84.9))  # role-level feed into Run
EXIT_ARROW = ((177.6, 84.9), (178.6, 81.2))  # role-level exit from Update

# two component box columns beneath the ring: green daily inputs on the
# left (feeding Run), blue conditioned parameters on the right (from Update)
BOX_W = 16.0
BOX_H = 3.8
BOX_PITCH = 5.0
BOX_TOP = 73.8  # top edge of the first box in both columns
DRV_COL = {
    "x0": 152.6,
    "header": ("Daily Drivers",),
    "head_y": (77.1,),  # centred on the right column's two-line band
    "fill": "#E7F1E4",
    "edge": "#A8CBA4",
    "items": ("NDVI", "ETo", "Precip.", "Solar Rad.", "Air Temp."),
}
CP_COL = {
    "x0": 170.6,
    "header": ("Conditioned", "Parameters"),
    "head_y": (78.6, 75.6),
    "fill": "#E2ECF6",
    "edge": "#A9C6E0",
    "items": ("AWC", "MAD", "Kcb", "Ks", "Kr", "Snowmelt"),
}

ASSETS = Path(__file__).resolve().parent / "assets"
E3_BASEMAP = {
    "path": str(ASSETS / "e3_basemap_usgs_naip_5070.png"),
    "source": "USGS The National Map, USGSImageryOnly (NAIP), public domain",
    "provenance": "assets/e3_basemap_usgs_naip_5070.json",
    "wash": 0.10,
}

XLIM = (-1.5, 120.5)
MEMBER_MS = 2.6
DIAMOND_MS = 4.6


def xmm(d):
    return PA_X0 + (np.asarray(d, dtype=float) - XLIM[0]) / (XLIM[1] - XLIM[0]) * (PA_X1 - PA_X0)


def ring_xy(sid, r=None):
    a = np.radians(ANG[sid])
    r = RING_R if r is None else r
    return RING_C[0] + r * np.cos(a), RING_C[1] + r * np.sin(a)


def main() -> None:
    F = K.Frozen()
    fig, ov, family, faces = K.new_figure(STUDY)
    text_w_mm = K.measurer(fig)

    bg = fig.add_axes([0, 0, 1, 1], zorder=1)
    bg.set_xlim(0, W_MM)
    bg.set_ylim(0, H_MM)
    bg.set_facecolor("none")
    bg.axis("off")
    tag(bg, "background-guides-and-datums")

    ts, cap, day, cap_day = F.ts, F.cap, F.day, F.cap_day
    AXIS: dict[str, dict] = {}

    mmtext(
        ov,
        MARGIN_MM,
        A_HEAD_Y,
        "(a)",
        cls="title",
        pt=FS_PANEL,
        weight="bold",
        gid="label-panel-a-letter",
    )
    mmtext(
        ov,
        MARGIN_MM + 5.8,
        A_HEAD_Y,
        "Sparse Satellite Constraints to Daily State",
        cls="title",
        pt=FS_PANEL_HEAD,
        weight="semibold",
        gid="label-panel-a-heading",
    )
    mmtext(
        ov,
        MARGIN_MM,
        RECORD_Y,
        F.record_label,
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        gid="label-record-identification",
    )

    months = pd.date_range(ts["date"].min(), ts["date"].max(), freq="MS")
    for i, m in enumerate(months):
        gx = float(xmm((m - ts["date"].min()).days))
        bg.add_line(
            tag(
                Line2D([gx, gx], [GUIDE_BOT, GUIDE_TOP], color=C_GUIDE, lw=LW_GUIDE, zorder=1),
                f"guide-month-{i + 1:02d}",
            )
        )

    def row_axes(rid, zorder=3):
        y0, y1 = ROWS[rid]
        ax = fig.add_axes(
            [PA_X0 / W_MM, y0 / H_MM, (PA_X1 - PA_X0) / W_MM, (y1 - y0) / H_MM], zorder=zorder
        )
        ax.set_xlim(*XLIM)
        ax.set_ylim(*DOMAIN[rid])
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        tag(ax, f"row-{K._slug(rid)}-axes")
        return ax

    def row_frame(rid):
        y0, y1 = ROWS[rid]
        d0, d1 = DOMAIN[rid]
        mid = (y0 + y1) / 2.0

        # study C's compact multi-line label, plus the unit as a final line
        lines = list(ROW_LABEL[rid]) + ([UNITS[rid]] if rid in UNITS else [])
        top = mid + (len(lines) - 1) * LABEL_LEAD / 2.0
        for j, s in enumerate(lines):
            mmtext(
                ov,
                LABEL_R,
                top - j * LABEL_LEAD,
                s,
                cls="direct_label",
                pt=FS_ROW,
                color=C_TEXT,
                ha="right",
                va="center",
                gid=f"label-row-{K._slug(rid)}-{j + 1}",
            )
            w = text_w_mm(s, FS_ROW)
            assert LABEL_R - w >= MARGIN_MM - 0.05, (
                f"the {rid!r} label line {s!r} overruns the margin ({LABEL_R - w:.2f} mm)"
            )

        ln = Line2D([PA_X0, PA_X0], [y0, y1], color=C_AXIS, lw=LW_SPINE, zorder=6)
        ov.add_line(ln)
        tag(ln, f"axis-spine-{K._slug(rid)}")

        # study B: every numeral is CENTRED on its tick
        for v, s in TICKS[rid]:
            ty = y0 + (v - d0) / (d1 - d0) * (y1 - y0)
            ov.add_line(
                Line2D([PA_X0, PA_X0 - TICK_STUB], [ty, ty], color=C_AXIS, lw=LW_AXIS, zorder=6)
            )
            ticktext(ov, TICKNUM_R, ty, s, pt=FS_TICK, ha="right", va="center")

        bg.add_line(
            tag(
                Line2D([PA_X0, PA_X1], [y0, y0], color=C_DATUM, lw=LW_DATUM, zorder=2),
                f"datum-{K._slug(rid)}",
            )
        )

    def record_axis(rid, arrays, n_traces=1, shares=False):
        flat = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
        flat = flat[np.isfinite(flat)]
        y0, y1 = ROWS[rid]
        AXIS[rid] = {
            "display_domain": list(DOMAIN[rid]),
            "plotted_range": [round(float(flat.min()), 4), round(float(flat.max()), 4)],
            "band_mm": round(y1 - y0, 3),
            "ticks": [t[1] for t in TICKS[rid]],
            "n_traces": n_traces,
            "shares_date_mapping": shares,
            "shares_y_mapping": shares,
        }

    # ---- rows ----
    rid = "etf_ensemble"
    ax = row_axes(rid)
    mem = F.members
    lo, hi = np.nanmin(mem, axis=1), np.nanmax(mem, axis=1)
    tag(ax.vlines(cap_day, lo, hi, color=C_MINMAX, lw=0.9, zorder=3), "marks-etf-minmax")
    mx = np.repeat(cap_day, mem.shape[1])
    my = mem.ravel()
    ok = np.isfinite(my)
    tag(
        ax.plot(
            mx[ok],
            my[ok],
            ls="none",
            marker="o",
            ms=MEMBER_MS,
            mfc=C_MEMBER,
            mec="white",
            mew=0.35,
            zorder=4,
        )[0],
        "marks-etf-members",
    )
    tag(
        ax.plot(
            cap_day,
            cap["etf_target_mean"],
            ls="none",
            marker="D",
            ms=DIAMOND_MS,
            mfc="none",
            mec=C_TARGET,
            mew=0.9,
            zorder=5,
        )[0],
        "marks-etf-target-mean",
    )
    row_frame(rid)
    record_axis(rid, [mem])
    assert int(ok.sum()) == int(cap["etf_member_count"].sum())

    rid = "ndvi_captures"
    ax = row_axes(rid)
    mL = ts["ndvi_landsat_raw"].notna().to_numpy()
    mS = ts["ndvi_sentinel_raw"].notna().to_numpy()
    tag(
        ax.plot(
            day[mL],
            ts["ndvi_landsat_raw"].to_numpy()[mL],
            ls="none",
            marker="o",
            ms=2.8,
            mfc=C_SENSOR,
            mec="white",
            mew=0.4,
            zorder=4,
        )[0],
        "marks-ndvi-landsat",
    )
    tag(
        ax.plot(
            day[mS],
            ts["ndvi_sentinel_raw"].to_numpy()[mS],
            ls="none",
            marker="s",
            ms=2.8,
            mfc="none",
            mec=C_SENSOR,
            mew=0.7,
            zorder=4,
        )[0],
        "marks-ndvi-sentinel",
    )
    row_frame(rid)
    record_axis(
        rid, [ts["ndvi_landsat_raw"].to_numpy()[mL], ts["ndvi_sentinel_raw"].to_numpy()[mS]]
    )

    rid = "daily_forcing"
    ax = row_axes(rid)
    pr = ts["precip"].to_numpy()
    eto = ts["eto"].to_numpy()
    tag(
        ax.vlines(day[pr > 0], 0.0, pr[pr > 0], color=C_PRECIP, lw=1.5, zorder=3),
        "marks-forcing-precipitation",
    )
    tag(ax.plot(day, eto, color=C_ETO, lw=0.8, zorder=4)[0], "marks-forcing-eto")
    row_frame(rid)
    record_axis(rid, [pr, eto])
    mmtext(
        ov,
        xmm(12.0),
        ROWS[rid][1] - 1.2,
        "precipitation",
        cls="direct_label",
        pt=FS_ROW,
        color="#46545F",
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-precipitation",
    )
    mmtext(
        ov,
        xmm(96.0),
        ROWS[rid][0] + 3.6,
        "ETo",
        cls="direct_label",
        pt=FS_ROW,
        color=C_ETO,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-eto",
    )

    rid = "rz_depletion"
    ax = row_axes(rid)
    rz = ts["rz_depletion"].to_numpy()
    ax.fill_between(day, 0, rz, color=C_SWIM, alpha=0.10, linewidth=0, zorder=2)
    tag(
        ax.plot(day, rz, color=C_SWIM, lw=LW_DATA, zorder=4, clip_on=False)[0], "marks-rz-depletion"
    )
    row_frame(rid)
    record_axis(rid, [rz])

    rid = "irrigation"
    ax = row_axes(rid)
    irr = ts["irr_applied"].to_numpy()
    ev = irr > 0
    tag(ax.vlines(day[ev], 0.0, irr[ev], color=C_SWIM, lw=1.0, zorder=4), "marks-irrigation-stems")
    row_frame(rid)
    record_axis(rid, [irr])

    rid = "et_comparison"
    ax = row_axes(rid)
    swim = ts["swim_ET"].to_numpy()
    flux = ts["flux_ET"].to_numpy()
    tag(
        ax.plot(day, flux, color=C_HELD, lw=0.6, zorder=4, clip_on=False)[0],
        "marks-flux-et-held-out",
    )
    tag(ax.plot(day, swim, color=C_SWIM, lw=1.1, zorder=5, clip_on=False)[0], "marks-daily-et")
    row_frame(rid)
    record_axis(rid, [swim, flux], n_traces=2, shares=True)
    mmtext(
        ov,
        xmm(20.0),
        ROWS[rid][1] - 1.2,
        "Simulated",
        cls="direct_label",
        pt=FS_ROW,
        color=C_SWIM,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-lane-daily-et",
    )
    flux_x = xmm(60.0)
    mmtext(
        ov,
        flux_x,
        ROWS[rid][0] + 1.3,
        "Flux ET (Held Out)",
        cls="direct_label",
        pt=FS_ROW,
        color=C_HELD,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-lane-flux-et",
    )
    FLUX_INK = (
        flux_x,
        ROWS[rid][0] + 1.3 - 1.3,
        flux_x + text_w_mm("Flux ET (Held Out)", FS_ROW),
        ROWS[rid][0] + 1.3 + 1.3,
    )

    # ---- shared date axis ----
    ln = Line2D([PA_X0, PA_X1], [DATE_SPINE_Y, DATE_SPINE_Y], color=C_AXIS, lw=LW_AXIS, zorder=6)
    ov.add_line(ln)
    tag(ln, "axis-shared-date-spine")
    for m in months:
        gx = float(xmm((m - ts["date"].min()).days))
        ov.add_line(
            Line2D([gx, gx], [DATE_SPINE_Y, DATE_SPINE_Y - 0.9], color=C_AXIS, lw=LW_AXIS, zorder=6)
        )
        ticktext(ov, gx + 0.8, DATE_LABEL_Y, m.strftime("%b"), pt=FS_TICK, ha="left", va="baseline")

    # =====================================================================
    # the ring, in a restrained field (study B, raised to free room below)
    # =====================================================================
    fx, fy, fw, fh = FIELD
    box = FancyBboxPatch(
        (fx + 1.4, fy + 1.4),
        fw - 2.8,
        fh - 2.8,
        boxstyle="round,pad=1.4,rounding_size=2.4",
        facecolor="#F4F4F1",
        edgecolor="none",
        zorder=2,
    )
    ov.add_patch(box)
    tag(box, "field-inverse-estimation")

    mmtext(
        ov,
        RING_C[0],
        TITLE_Y,
        "Inverse Estimation",
        cls="title",
        pt=FS_LABEL,
        weight="bold",
        ha="center",
        gid="label-inverse-estimation",
    )
    sx, sy, sw, sh = SWE_BOX
    swe_rect = FancyBboxPatch(
        (sx, sy),
        sw,
        sh,
        boxstyle="round,pad=0,rounding_size=0.9",
        facecolor=SWE_FILL,
        edgecolor=SWE_EDGE,
        linewidth=0.5,
        zorder=4,
    )
    ov.add_patch(swe_rect)
    tag(swe_rect, "box-etf-swe")
    mmtext(
        ov,
        sx + sw / 2.0,
        sy + sh / 2.0,
        "ETf + SWE",
        cls="direct_label",
        pt=FS_ROW,
        ha="center",
        va="center",
        gid="label-etf-swe",
    )
    cx_top, cy_top = ring_xy("compare")
    polyline(
        ov,
        [(RING_C[0], sy - 0.2), (RING_C[0], cy_top + 1.9)],
        C_INV,
        0.8,
        head=1.3,
        zorder=6,
        rid="constraint-etf-swe-to-compare",
    )

    BOX = {}
    for sid, label in (
        ("run_balance", "Run"),
        ("compare", "Compare"),
        ("update_parameters", "Update"),
    ):
        px, py = ring_xy(sid)
        mmtext(
            ov,
            px,
            py,
            label,
            cls="direct_label",
            pt=FS_ROW,
            color=C_TEXT,
            weight="semibold",
            ha="center",
            va="center",
            gid=f"label-stage-{K._slug(sid)}",
        )
        w = text_w_mm(label, FS_ROW, "semibold")
        BOX[sid] = (px - w / 2, py - 1.4, px + w / 2, py + 1.4)

    for rid_, (a0, a1) in ARCS.items():
        polyline(
            ov, arc_pts(*RING_C, RING_R, a0, a1, 44), C_INV, 0.95, head=1.4, zorder=6, rid=rid_
        )

    # ---- the two component box columns beneath the ring (Sec. 5.3) ----
    def box_column(col, key):
        cx = col["x0"] + BOX_W / 2.0
        for j, (hs, hy) in enumerate(zip(col["header"], col["head_y"], strict=True)):
            mmtext(
                ov,
                cx,
                hy,
                hs,
                cls="title",
                pt=FS_STRUCT,
                weight="semibold",
                ha="center",
                gid=f"label-{key}-header-{j + 1}",
            )
        for j, s in enumerate(col["items"]):
            y_top = BOX_TOP - j * BOX_PITCH
            rect = FancyBboxPatch(
                (col["x0"], y_top - BOX_H),
                BOX_W,
                BOX_H,
                boxstyle="round,pad=0,rounding_size=0.9",
                facecolor=col["fill"],
                edgecolor=col["edge"],
                linewidth=0.5,
                zorder=4,
            )
            ov.add_patch(rect)
            tag(rect, f"box-{key}-{j + 1}")
            mmtext(
                ov,
                cx,
                y_top - BOX_H / 2.0,
                s,
                cls="direct_label",
                pt=FS_ROW,
                color=C_TEXT,
                ha="center",
                va="center",
                gid=f"label-{key}-{j + 1}",
            )
        return BOX_TOP - (len(col["items"]) - 1) * BOX_PITCH - BOX_H

    drv_bot = box_column(DRV_COL, "drivers")
    cp_bot = box_column(CP_COL, "parameters")
    polyline(ov, list(DRV_ARROW), C_INV, 0.8, head=1.3, zorder=6, rid="driver-to-run")
    polyline(
        ov, list(EXIT_ARROW), C_INV, 0.9, head=1.3, zorder=6, rid="exit-conditioned-parameters"
    )

    pb = K.draw_panel_b(
        fig,
        ov,
        F,
        text_w_mm,
        e2_mode="symmetric",
        e3_route="orthogonal",
        e3_basemap=E3_BASEMAP,
        e3_hull=False,
        e1_e3_locator=True,
    )

    # =====================================================================
    # composition guards
    # =====================================================================
    DATA_W = PA_X1 - PA_X0
    assert DATA_W >= 120.0, f"the temporal field is only {DATA_W:.1f} mm wide"
    assert 15.0 <= LABEL_R <= 18.0, "the label gutter left C's compact 15-18 mm treatment"
    assert TICKNUM_R + 0.4 <= PA_X0 - TICK_STUB
    widest_num = max(text_w_mm(t[1], FS_TICK) for r in TICKS for t in TICKS[r])
    assert TICKNUM_R - widest_num > LABEL_R + 1.0
    assert LOOP_X1 - LOOP_X0 <= 35.0
    assert LOOP_X0 > PA_X1 + 2.0
    # centred numerals: adjacent rows' ink must not meet across a gutter
    ink = K.ink_h_mm(FS_TICK)
    assert ROW_GUTTER - ink > 0.3, "centred numerals collide across the row gutter"
    ys = sorted(ROWS.values())
    for (a0_, a1_), (b0_, _b1) in zip(ys, ys[1:], strict=False):
        assert round(b0_ - a1_, 3) == ROW_GUTTER, f"uneven row gutter at {a1_}"
        _ = a0_
    # every arc is the SAME radius, so no edge of the cycle can read as weaker
    for rid_ in ARCS:
        rt = next(r for r in K.ROUTES if r["id"] == rid_)
        rr = [float(np.hypot(p[0] - RING_C[0], p[1] - RING_C[1])) for p in rt["pts"]]
        assert max(abs(np.array(rr) - RING_R)) < 1e-6, f"{rid_} is not on the ring"
    # ring and words inside the field, field clear of the data plots
    assert fx > PA_X1 + 2.0 and fx + fw <= W_MM - MARGIN_MM + 0.2
    for sid, (bx0, by0, bx1, by1) in BOX.items():
        assert fx <= bx0 and bx1 <= fx + fw, f"{sid} label leaves the field"
        assert fy <= by0 and by1 <= fy + fh, f"{sid} label leaves the field"

    # the box columns: labels inside their boxes, headers inside their
    # columns, columns inside the loop band, below (not inside) the field,
    # clear of each other and of panel (b)'s map headings
    for col, name in ((DRV_COL, "drivers"), (CP_COL, "parameters")):
        for s in col["items"]:
            assert text_w_mm(s, FS_ROW) <= BOX_W - 1.6, f"{name} box label {s!r} overfills its box"
        for hs in col["header"]:
            assert text_w_mm(hs, FS_STRUCT, "semibold") <= BOX_W + 0.4, (
                f"{name} header {hs!r} is wider than its column"
            )
        assert max(col["head_y"]) + K.ink_h_mm(FS_STRUCT) * 0.78 < fy, (
            f"the {name} header enters the field"
        )
        assert BOX_TOP <= min(col["head_y"]) - 0.7 - 1.0, (
            f"the {name} boxes collide with their header"
        )
        assert col["x0"] >= LOOP_X0 and col["x0"] + BOX_W <= LOOP_X1 - 0.35, (
            f"the {name} column leaves the loop band"
        )
    assert DRV_COL["x0"] + BOX_W + 1.5 <= CP_COL["x0"], "the two box columns collide"
    assert min(drv_bot, cp_bot) >= K.MAP_HEAD_Y + 2.5, (
        "a box column descends into panel (b)'s map headings"
    )
    # the feed rises into Run; the exit drops from Update onto its header
    assert BOX["run_balance"][1] - 0.9 < DRV_ARROW[1][1] < BOX["run_balance"][1], (
        "the driver feed does not terminate at Run"
    )
    assert BOX["update_parameters"][0] < EXIT_ARROW[0][0] < BOX["update_parameters"][2], (
        "the exit does not depart from beneath Update"
    )
    cp_head_ink_top = CP_COL["head_y"][0] + K.ink_h_mm(FS_STRUCT) * 0.78
    assert cp_head_ink_top < EXIT_ARROW[1][1] < cp_head_ink_top + 1.2, (
        "the exit arrow does not terminate at the Conditioned Parameters header"
    )
    assert abs(EXIT_ARROW[1][0] - (CP_COL["x0"] + BOX_W / 2.0)) < 0.5, (
        "the exit arrow does not land on the parameter column's centreline"
    )
    # the ETf + SWE constraint box: label inside, box between the title and
    # the field, arrow long enough to read as a connection into Compare
    assert text_w_mm("ETf + SWE", FS_ROW) <= SWE_BOX[2] - 1.6, "the SWE box label overfills it"
    assert SWE_BOX[1] > fy + fh + 0.6, "the ETf + SWE box touches the field"
    assert SWE_BOX[1] + SWE_BOX[3] < TITLE_Y - 0.9, "the ETf + SWE box collides with the title"
    assert SWE_BOX[1] - 0.2 - (cy_top + 1.9) >= 1.8, "the constraint arrow is too short to read"
    assert abs((SWE_BOX[0] + SWE_BOX[2] / 2.0) - RING_C[0]) < 1e-9, "the SWE box is off-centre"

    # panel (b) render overrides actually applied and recorded
    assert abs(pb["e2_lat_bound_deg"] - 56.0997) < 0.01, pb["e2_lat_bound_deg"]
    assert "e3_route_vertices_mm" in pb, "the orthogonal E1->E3 route was not drawn"
    assert "e3_basemap" in pb, "the E3 basemap was not drawn"
    assert "e1_e3_locator_epsg5070" in pb, "the E3 locator was not drawn on the E1 map"

    LOOP_BBOX = (
        DRV_COL["x0"],
        min(drv_bot, cp_bot) - 0.7,
        CP_COL["x0"] + BOX_W,
        TITLE_Y + K.ink_h_mm(FS_LABEL) * 0.75,
    )

    m = K.audit_scientific(
        F,
        AXIS,
        [FLUX_INK, pb["meter_ink"]],
        {
            "edges": [
                ("run_balance", "compare"),
                ("compare", "update_parameters"),
                ("update_parameters", "run_balance"),
            ],
            "exit": ("update_parameters", "daily_balance"),
            "inputs": [("etf_ensemble", "compare"), ("daily_drivers", "run_balance")],
        },
    )

    layout = {
        "data_field_mm": [PA_X0, PA_X1, round(DATA_W, 2)],
        "label_column_right_mm": LABEL_R,
        "label_lines": {k: list(v) for k, v in ROW_LABEL.items()},
        "units": dict(UNITS),
        "ticknum_right_mm": TICKNUM_R,
        "rows_mm": {k: list(v) for k, v in ROWS.items()},
        "row_gutter_mm": ROW_GUTTER,
        "tick_numerals": "centred on the tick",
        "domains": {k: list(v) for k, v in DOMAIN.items()},
        "ticks": {k: [t[1] for t in v] for k, v in TICKS.items()},
        "date_axis_mm": [DATE_SPINE_Y, DATE_LABEL_Y],
        "loop_column_mm": [LOOP_X0, LOOP_X1, round(LOOP_X1 - LOOP_X0, 2)],
        "loop_field_mm": list(FIELD),
        "ring_centre_mm": list(RING_C),
        "ring_radius_mm": RING_R,
        "drivers_block": {
            "header": list(DRV_COL["header"]),
            "header_y_mm": list(DRV_COL["head_y"]),
            "column_x0_mm": DRV_COL["x0"],
            "items": list(DRV_COL["items"]),
            "fill": DRV_COL["fill"],
        },
        "parameters_block": {
            "header": list(CP_COL["header"]),
            "header_y_mm": list(CP_COL["head_y"]),
            "column_x0_mm": CP_COL["x0"],
            "items": list(CP_COL["items"]),
            "fill": CP_COL["fill"],
        },
        "box_mm": {"w": BOX_W, "h": BOX_H, "pitch": BOX_PITCH, "top": BOX_TOP},
        "swe_box_mm": list(SWE_BOX),
        "loop_fonts": {
            "title": [FS_LABEL, "bold"],
            "column_headers": [K.FS_STRUCT, "semibold"],
        },
        "e3_key_x_mm": K.E3_KEY_X,
        "e3_key_baselines_mm": list(K.E3_KEY_LINES),
    }
    meas = {
        **m,
        "data_field_width_mm": round(DATA_W, 2),
        "label_gutter_mm": LABEL_R,
        "label_to_numeral_gap_mm": round(TICKNUM_R - widest_num - LABEL_R, 2),
        "numeral_to_spine_gap_mm": round(PA_X0 - TICK_STUB - TICKNUM_R, 2),
        "ring_circumference_mm": round(2 * np.pi * RING_R, 2),
        "box_label_max_mm": round(
            max(text_w_mm(s, FS_ROW) for s in DRV_COL["items"] + CP_COL["items"]), 2
        ),
        "box_label_min_margin_mm": round(
            BOX_W - max(text_w_mm(s, FS_ROW) for s in DRV_COL["items"] + CP_COL["items"]), 2
        ),
        "column_gap_mm": round(CP_COL["x0"] - (DRV_COL["x0"] + BOX_W), 2),
        "columns_bottom_mm": [round(drv_bot, 2), round(cp_bot, 2)],
        "columns_bottom_to_map_head_mm": round(min(drv_bot, cp_bot) - K.MAP_HEAD_Y, 2),
        "loop_footprint_mm": [
            round(LOOP_BBOX[2] - LOOP_BBOX[0], 2),
            round(LOOP_BBOX[3] - LOOP_BBOX[1], 2),
        ],
        **{f"panel_b_{k}": v for k, v in pb.items() if k != "meter_ink"},
    }
    out = K.export(
        fig,
        STUDY,
        {
            "family": family,
            "faces": faces,
            "layout": layout,
            "axis_audit": AXIS,
            "measured": meas,
            "architecture_sha256": F.arch_sha,
            "example_csv_sha256": F.csv_sha,
        },
    )
    print(
        f"[{STUDY}] data field {DATA_W:.1f} mm   label gutter {LABEL_R:.1f} mm   "
        f"ring r={RING_R} mm   E2 bounds ±{pb['e2_lat_bound_deg']:.2f}°"
    )
    print(f"[{STUDY}] E1->E3 route vertices: {pb['e3_route_vertices_mm']}")
    for k, v in AXIS.items():
        print(
            f"    {k:16s} {v['display_domain']}  data "
            f"[{v['plotted_range'][0]:.3f}, {v['plotted_range'][1]:.3f}]  "
            f"headroom {v['headroom_mm']:.2f} mm ({v['headroom_frac'] * 100:.1f}%)"
        )
    return out


if __name__ == "__main__":
    main()

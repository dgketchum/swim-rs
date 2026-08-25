"""Revision-5 panel-(a) design study B -- ring cycle, weighted rows.

Study B keeps study A's six-row stack, its four-part row grammar and its
120.0 mm data field, and changes three things that study A settled one way:

1.  ROW WEIGHTING. The two model-state rows that carry the paper's argument
    (Root-Zone Depletion, Daily ET) are given the most height; the three
    observation rows are tightened. Study A spread the height nearly evenly.

2.  TICK TREATMENT. Every tick numeral is centred on its tick, the way a
    conventional axis sets one, and the row gutters are opened to 3.2 mm so
    that adjacent rows' numerals cannot approach one another. Study A insets
    the two extreme numerals and buys back 0.8 mm of row height per gutter.

3.  THE INVERSE ELEMENT. A true circular ring inside a restrained tinted
    field, rather than study A's triangle of outward arcs on white. `Run`,
    `Compare` and `Update` sit ON the ring; three clockwise arcs with
    arrowheads connect them, and the Update-to-Run arc is the same radius as
    the other two, so the return cannot read as weaker than the forward path.
    Inputs cross the field boundary inward and the conditioned parameters
    cross it outward, which is what the enclosure is for.

Study B also opens the depletion domain to 0-18 mm so the seasonal maximum
clears the ceiling by nearly 2 mm rather than 1 mm.

Run:
  uv run python paper/figures/proofs/fig01_evidence_190_r5/build_fig01_r5_studyB.py
"""

from __future__ import annotations

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

STUDY = "fig01_r5_studyB"

# ===========================================================================
# LAYOUT
# ===========================================================================

A_HEAD_Y = 114.6
RECORD_Y = 110.4

LABEL_R = 26.2
TICKNUM_R = 31.3
PA_X0 = 32.6
PA_X1 = 152.6
TICK_STUB = 0.9

ROW_GUTTER = 3.2
ROWS: dict[str, tuple[float, float]] = {
    "etf_ensemble": (100.6, 107.0),
    "ndvi_captures": (90.8, 97.4),
    "daily_forcing": (81.4, 87.6),
    "rz_depletion": (69.8, 78.2),
    "irrigation": (61.8, 66.6),
    "et_comparison": (51.0, 58.6),
}
ROW_LABEL = {
    "etf_ensemble": "ETf Ensemble",
    "ndvi_captures": "NDVI Captures",
    "daily_forcing": "Daily Forcing",
    "rz_depletion": "Root-Zone Depletion",
    "irrigation": "Irrigation",
    "et_comparison": "Daily ET",
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
    "rz_depletion": [(0.0, "0"), (18.0, "18")],
    "irrigation": [(0.0, "0"), (25.0, "25")],
    "et_comparison": [(0.0, "0"), (12.0, "12")],
}
UNITS = {
    "daily_forcing": "mm/d",
    "rz_depletion": "mm",
    "irrigation": "mm",
    "et_comparison": "mm/d",
}

DATE_SPINE_Y = 49.2
DATE_LABEL_Y = 46.4
GUIDE_TOP = 107.0
GUIDE_BOT = 51.0

# ---- the ring, inside a restrained field --------------------------------
LOOP_X0, LOOP_X1 = 155.6, 187.0
FIELD = (156.4, 78.8, 29.8, 20.8)  # x, y, w, h -- light tint, no border
RING_C = (171.3, 88.2)
RING_R = 7.9
TITLE_Y = 105.6  # 'Inverse Estimation'
SWE_Y = 101.6  # 'ETf + SWE'
ANG = {"compare": 90.0, "update_parameters": -30.0, "run_balance": 210.0}
ARCS = {  # clockwise, with a word-width gap at each end
    "cycle-run-to-compare": (190.0, 126.0),
    "cycle-compare-to-update": (54.0, -7.0),
    "cycle-update-to-run": (-53.0, -127.0),
}
DRV_LABEL = (LOOP_X0 - 0.2, 75.2)  # 'Daily Drivers'
DRV_ARROW = ((158.0, 78.2), (161.0, 82.0))
EXIT_LABEL = (187.0, 70.6)  # 'Conditioned Parameters', right-aligned
EXIT_ARROW = ((178.1, 81.4), (178.1, 75.4))

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

        lab = ROW_LABEL[rid]
        lab_y = mid + 1.5 if rid in UNITS else mid
        mmtext(
            ov,
            LABEL_R,
            lab_y,
            lab,
            cls="direct_label",
            pt=FS_ROW,
            color=C_TEXT,
            ha="right",
            va="center",
            gid=f"label-row-{K._slug(rid)}",
        )
        w = text_w_mm(lab, FS_ROW)
        assert LABEL_R - w >= MARGIN_MM - 0.05, f"{rid!r} row label overruns the margin"
        if rid in UNITS:
            mmtext(
                ov,
                LABEL_R,
                mid - 2.2,
                UNITS[rid],
                cls="direct_label",
                pt=FS_ROW,
                color=C_TEXT,
                ha="right",
                va="center",
                gid=f"label-units-{K._slug(rid)}",
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
    # the ring, in a restrained field
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
        pt=FS_STRUCT,
        weight="semibold",
        ha="center",
        gid="label-inverse-estimation",
    )
    mmtext(
        ov,
        RING_C[0],
        SWE_Y,
        "ETf + SWE",
        cls="direct_label",
        pt=FS_ROW,
        ha="center",
        gid="label-etf-swe",
    )
    cx_top, cy_top = ring_xy("compare")
    polyline(
        ov,
        [(RING_C[0], SWE_Y - 1.0), (RING_C[0], cy_top + 1.9)],
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

    mmtext(
        ov,
        *DRV_LABEL,
        "Daily Drivers",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="left",
        va="baseline",
        gid="label-daily-drivers",
    )
    polyline(ov, list(DRV_ARROW), C_INV, 0.8, head=1.3, zorder=6, rid="driver-to-run")

    mmtext(
        ov,
        *EXIT_LABEL,
        "Conditioned Parameters",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="right",
        va="baseline",
        gid="label-conditioned-parameters",
    )
    polyline(
        ov, list(EXIT_ARROW), C_INV, 0.9, head=1.3, zorder=6, rid="exit-conditioned-parameters"
    )

    pb = K.draw_panel_b(fig, ov, F, text_w_mm)

    # =====================================================================
    # composition guards
    # =====================================================================
    DATA_W = PA_X1 - PA_X0
    assert DATA_W >= 118.0, f"the temporal field is only {DATA_W:.1f} mm wide"
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
    LOOP_BBOX = (fx, EXIT_LABEL[1] - 0.7, fx + fw, TITLE_Y + K.ink_h_mm(FS_STRUCT) * 0.75)

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
        "e3_key_x_mm": K.E3_KEY_X,
        "e3_key_baselines_mm": list(K.E3_KEY_LINES),
    }
    meas = {
        **m,
        "data_field_width_mm": round(DATA_W, 2),
        "loop_footprint_mm": [
            round(LOOP_BBOX[2] - LOOP_BBOX[0], 2),
            round(LOOP_BBOX[3] - LOOP_BBOX[1], 2),
        ],
        "ring_circumference_mm": round(2 * np.pi * RING_R, 2),
        "label_to_numeral_gap_mm": round(TICKNUM_R - widest_num - LABEL_R, 2),
        "numeral_to_spine_gap_mm": round(PA_X0 - TICK_STUB - TICKNUM_R, 2),
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
        f"[{STUDY}] data field {DATA_W:.1f} mm   loop field "
        f"{FIELD[2]:.1f} x {FIELD[3]:.1f} mm   ring r={RING_R} mm"
    )
    for k, v in AXIS.items():
        print(
            f"    {k:16s} {v['display_domain']}  data "
            f"[{v['plotted_range'][0]:.3f}, {v['plotted_range'][1]:.3f}]  "
            f"headroom {v['headroom_mm']:.2f} mm ({v['headroom_frac'] * 100:.1f}%)"
        )
    return out


if __name__ == "__main__":
    main()

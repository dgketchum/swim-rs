"""Revision-5 panel-(a) design study A -- the Section 4 wireframe, faithfully.

Study A implements the handoff's preferred treatment exactly as written:

* six aligned rows on one date axis -- ETf Ensemble, NDVI Captures, Daily
  Forcing, Root-Zone Depletion, Irrigation, Daily ET (with held-out flux ET);
* the four-part row grammar with genuinely distinct gutters -- compact row
  label | right-aligned tick-numeral gutter | visible y-spine | data field;
* 120.0 mm of temporal data field, reclaimed from the label column and the
  inverse element;
* two or three conventional labelled ticks per row, a faint lower-bound datum
  through the data field, and rounded domains with visible headroom;
* subtle April-July guides shared by the whole stack, tied to the single date
  axis below the last row; and
* a compact closed cycle in a ~31 mm right-hand column: `Run`, `Compare`,
  `Update` at the corners of a triangle whose three edges are outward-bulging
  arcs, so the loop reads as a ring before its words are read. The
  Update-to-Run return is the widest and lowest of the three edges.

Tick treatment: the two extreme numerals are set flush inside their row
(upper bound flush-top, lower bound flush-bottom) so that adjacent rows'
numerals never approach one another. Study B tests the conventional
centred-on-the-bound alternative with larger gutters, for direct comparison.

Run:
  uv run python paper/figures/proofs/fig01_evidence_190_r5/build_fig01_r5_studyA.py
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
    bezier_pts,
    mmtext,
    polyline,
    tag,
    ticktext,
)
from matplotlib.lines import Line2D

STUDY = "fig01_r5_studyA"

# ===========================================================================
# LAYOUT -- one block; a Level 1 edit is a one-line change here
# ===========================================================================

A_HEAD_Y = 114.6  # '(a)' + panel heading baseline
RECORD_Y = 110.4  # 'US-Bi1 (2017)'

# ---- the four-part row grammar, left to right ----------------------------
LABEL_R = 26.2  # right edge of the compact row-label column
TICKNUM_R = 31.3  # right edge of the tick-numeral gutter
PA_X0 = 32.6  # the y-spines AND the left edge of the data field
PA_X1 = 152.6  # right edge of the data field  -> 120.0 mm of temporal width
TICK_STUB = 0.9  # tick stubs, drawn leftward from each spine

# ---- six aligned rows, top to bottom, 2.4 mm gutters ---------------------
ROWS: dict[str, tuple[float, float]] = {
    "etf_ensemble": (99.6, 107.0),
    "ndvi_captures": (90.4, 97.2),
    "daily_forcing": (81.0, 88.2),
    "rz_depletion": (70.6, 78.6),
    "irrigation": (61.4, 68.2),
    "et_comparison": (51.0, 58.8),
}
ROW_LABEL = {
    "etf_ensemble": "ETf Ensemble",
    "ndvi_captures": "NDVI Captures",
    "daily_forcing": "Daily Forcing",
    "rz_depletion": "Root-Zone Depletion",
    "irrigation": "Irrigation",
    "et_comparison": "Daily ET",
}
# rounded domains with visible headroom (handoff Sec. 5.2 starting ranges)
DOMAIN = {
    "etf_ensemble": (0.0, 1.6),
    "ndvi_captures": (0.2, 1.0),
    "daily_forcing": (0.0, 18.0),
    "rz_depletion": (0.0, 16.0),
    "irrigation": (0.0, 25.0),
    "et_comparison": (0.0, 12.0),
}
TICKS = {
    "etf_ensemble": [(0.0, "0.0"), (1.6, "1.6")],
    "ndvi_captures": [(0.2, "0.2"), (1.0, "1.0")],
    "daily_forcing": [(0.0, "0"), (18.0, "18")],
    "rz_depletion": [(0.0, "0"), (8.0, "8"), (16.0, "16")],
    "irrigation": [(0.0, "0"), (25.0, "25")],
    "et_comparison": [(0.0, "0"), (6.0, "6"), (12.0, "12")],
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

# ---- the compact closed cycle, right-hand column 155.6 - 187.0 mm --------
LOOP_X0, LOOP_X1 = 155.6, 187.0
LOOP_CX = 171.3
TITLE_Y = 102.4  # 'Inverse Estimation'
SWE_Y = 97.4  # 'ETf + SWE', centred over Compare
SWE_TIP = (LOOP_CX, 94.3)
STAGE = {  # label centres
    "compare": (LOOP_CX, 92.2),
    "run_balance": (161.9, 83.2),
    "update_parameters": (180.5, 83.2),
}
EDGE_RUN_COMPARE = ((162.9, 85.3), (159.6, 90.2), (166.4, 90.6))
EDGE_COMPARE_UPDATE = ((176.2, 90.6), (183.1, 90.2), (179.6, 85.3))
EDGE_UPDATE_RUN = ((176.8, 81.2), (171.2, 76.9), (164.8, 81.2))
DRV_LABEL = (LOOP_X0 + 0.2, 75.6)  # 'Daily Drivers'
DRV_ARROW = ((157.6, 78.6), (159.4, 82.0))
EXIT_LABEL = (LOOP_X0 + 0.4, 70.6)  # 'Conditioned Parameters'
EXIT_ARROW = ((180.5, 80.7), (180.5, 75.2))

XLIM = (-1.5, 120.5)  # day index, padded so end marks are not on the spine

MEMBER_MS = 2.6
DIAMOND_MS = 4.6


def xmm(d):
    return PA_X0 + (np.asarray(d, dtype=float) - XLIM[0]) / (XLIM[1] - XLIM[0]) * (PA_X1 - PA_X0)


def main() -> None:
    F = K.Frozen()
    fig, ov, family, faces = K.new_figure(STUDY)
    text_w_mm = K.measurer(fig)

    # a background plane for the month guides and the row datums, so both sit
    # BEHIND the data rather than over it
    bg = fig.add_axes([0, 0, 1, 1], zorder=1)
    bg.set_xlim(0, W_MM)
    bg.set_ylim(0, H_MM)
    bg.set_facecolor("none")
    bg.axis("off")
    tag(bg, "background-guides-and-datums")

    ts, cap, day, cap_day = F.ts, F.cap, F.day, F.cap_day
    AXIS: dict[str, dict] = {}

    # ---------------- panel (a) header ----------------
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

    # ---------------- shared April-July guides ----------------
    months = pd.date_range(ts["date"].min(), ts["date"].max(), freq="MS")
    for i, m in enumerate(months):
        gx = float(xmm((m - ts["date"].min()).days))
        bg.add_line(
            tag(
                Line2D([gx, gx], [GUIDE_BOT, GUIDE_TOP], color=C_GUIDE, lw=LW_GUIDE, zorder=1),
                f"guide-month-{i + 1:02d}",
            )
        )

    # ---------------- row machinery ----------------
    def row_axes(rid, zorder=3, ylim=None):
        y0, y1 = ROWS[rid]
        ax = fig.add_axes(
            [PA_X0 / W_MM, y0 / H_MM, (PA_X1 - PA_X0) / W_MM, (y1 - y0) / H_MM], zorder=zorder
        )
        ax.set_xlim(*XLIM)
        ax.set_ylim(*(ylim or DOMAIN[rid]))
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        tag(ax, f"row-{K._slug(rid)}-axes")
        return ax

    def row_frame(rid):
        """Row label, y-spine, ticks, and the lower-bound datum."""
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
        assert LABEL_R - w >= MARGIN_MM - 0.05, (
            f"the {rid!r} row label overruns the {MARGIN_MM} mm margin ({LABEL_R - w:.2f} mm)"
        )
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

        for v, s in TICKS[rid]:
            ty = y0 + (v - d0) / (d1 - d0) * (y1 - y0)
            ov.add_line(
                Line2D([PA_X0, PA_X0 - TICK_STUB], [ty, ty], color=C_AXIS, lw=LW_AXIS, zorder=6)
            )
            va = "top" if abs(ty - y1) < 0.05 else ("bottom" if abs(ty - y0) < 0.05 else "center")
            ticktext(ov, TICKNUM_R, ty, s, pt=FS_TICK, ha="right", va=va)

        # faint lower-bound datum through the data field
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

    # ================= ROW 1 -- ETf Ensemble =================
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

    # ================= ROW 2 -- NDVI Captures =================
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

    # ================= ROW 3 -- Daily Forcing =================
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
        ROWS[rid][0] + 4.0,
        "ETo",
        cls="direct_label",
        pt=FS_ROW,
        color=C_ETO,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-eto",
    )

    # ================= ROW 4 -- Root-Zone Depletion =================
    rid = "rz_depletion"
    ax = row_axes(rid)
    rz = ts["rz_depletion"].to_numpy()
    ax.fill_between(day, 0, rz, color=C_SWIM, alpha=0.10, linewidth=0, zorder=2)
    tag(
        ax.plot(day, rz, color=C_SWIM, lw=LW_DATA, zorder=4, clip_on=False)[0], "marks-rz-depletion"
    )
    row_frame(rid)
    record_axis(rid, [rz])

    # ================= ROW 5 -- Irrigation =================
    rid = "irrigation"
    ax = row_axes(rid)
    irr = ts["irr_applied"].to_numpy()
    ev = irr > 0
    tag(ax.vlines(day[ev], 0.0, irr[ev], color=C_SWIM, lw=1.0, zorder=4), "marks-irrigation-stems")
    row_frame(rid)
    record_axis(rid, [irr])
    assert int(ev.sum()) > 0

    # ================= ROW 6 -- Daily ET + held-out flux ET =================
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

    # ---------------- one shared date axis ----------------
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
    # the compact closed cycle
    # =====================================================================
    mmtext(
        ov,
        LOOP_CX,
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
        LOOP_CX,
        SWE_Y,
        "ETf + SWE",
        cls="direct_label",
        pt=FS_ROW,
        ha="center",
        gid="label-etf-swe",
    )
    polyline(
        ov,
        [(LOOP_CX, SWE_Y - 1.0), SWE_TIP],
        C_INV,
        0.8,
        head=1.3,
        zorder=6,
        rid="constraint-etf-swe-to-compare",
    )

    BOX = {}
    for sid, (cx, cy) in STAGE.items():
        label = {"run_balance": "Run", "compare": "Compare", "update_parameters": "Update"}[sid]
        mmtext(
            ov,
            cx,
            cy,
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
        BOX[sid] = (cx - w / 2, cy - 1.4, cx + w / 2, cy + 1.4)

    for (p0, p1, p2), rid_, lw in (
        (EDGE_RUN_COMPARE, "cycle-run-to-compare", 0.9),
        (EDGE_COMPARE_UPDATE, "cycle-compare-to-update", 0.9),
        (EDGE_UPDATE_RUN, "cycle-update-to-run", 1.05),
    ):
        polyline(ov, bezier_pts(p0, p1, p2), C_INV, lw, head=1.4, zorder=6, rid=rid_)

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
        ha="left",
        va="baseline",
        gid="label-conditioned-parameters",
    )
    polyline(
        ov, list(EXIT_ARROW), C_INV, 0.9, head=1.3, zorder=6, rid="exit-conditioned-parameters"
    )

    # =====================================================================
    # panel (b) -- fixed, with the E3 key now beside the E3 map
    # =====================================================================
    pb = K.draw_panel_b(fig, ov, F, text_w_mm)

    # =====================================================================
    # composition guards that are geometric facts, not taste
    # =====================================================================
    DATA_W = PA_X1 - PA_X0
    assert DATA_W >= 118.0, f"the temporal field is only {DATA_W:.1f} mm wide"
    assert TICKNUM_R + 0.4 <= PA_X0 - TICK_STUB, "the numeral gutter runs into the tick stubs"
    widest_num = max(text_w_mm(t[1], FS_TICK) for r in TICKS for t in TICKS[r])
    assert TICKNUM_R - widest_num > LABEL_R + 1.0, (
        "the row-label and tick-numeral gutters are not separated"
    )
    assert LOOP_X1 - LOOP_X0 <= 35.0, "the inverse element exceeds its 35 mm allowance"
    assert LOOP_X0 > PA_X1 + 2.0, "the inverse element intrudes on the data field"
    for rt in K.ROUTES:
        x0b, y0b, x1b, y1b = rt["bbox"]
        if rt["id"].startswith(("cycle-", "constraint-", "driver-", "exit-")):
            assert x0b >= LOOP_X0 - 0.5, f"{rt['id']} leaves the inverse column"
            assert x1b <= LOOP_X1 + 0.5, f"{rt['id']} leaves the inverse column"
    # the loop's own bounding box, the Sec. 4 allowance
    loop_pts = [p for rt in K.ROUTES if rt["id"].startswith("cycle-") for p in rt["pts"]]
    lx = [p[0] for p in loop_pts]
    ly = [p[1] for p in loop_pts]
    LOOP_BBOX = (min(lx), min(ly), max(lx), max(ly))
    # the return edge must be at least as prominent as the forward edges
    ret = next(r for r in K.ROUTES if r["id"] == "cycle-update-to-run")
    ret_len = float(
        np.sum(np.hypot(np.diff([p[0] for p in ret["pts"]]), np.diff([p[1] for p in ret["pts"]])))
    )
    fwd = [
        float(
            np.sum(np.hypot(np.diff([p[0] for p in r["pts"]]), np.diff([p[1] for p in r["pts"]])))
        )
        for r in K.ROUTES
        if r["id"] in ("cycle-run-to-compare", "cycle-compare-to-update")
    ]
    assert ret_len >= max(fwd) * 0.9, (
        f"the Update->Run return ({ret_len:.1f} mm) is weaker than the forward edges"
    )
    # no cycle edge may touch a stage's ink box other than at its own endpoints
    for r in K.ROUTES:
        if not r["id"].startswith("cycle-"):
            continue
        a, b = {
            "cycle-run-to-compare": ("run_balance", "compare"),
            "cycle-compare-to-update": ("compare", "update_parameters"),
            "cycle-update-to-run": ("update_parameters", "run_balance"),
        }[r["id"]]
        for sid, (bx0, by0, bx1, by1) in BOX.items():
            if sid in (a, b):
                continue
            for px, py in r["pts"]:
                assert not (bx0 <= px <= bx1 and by0 <= py <= by1), (
                    f"{r['id']} runs through the {sid!r} label"
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
        "ticknum_right_mm": TICKNUM_R,
        "rows_mm": {k: list(v) for k, v in ROWS.items()},
        "row_gutter_mm": 2.4,
        "domains": {k: list(v) for k, v in DOMAIN.items()},
        "ticks": {k: [t[1] for t in v] for k, v in TICKS.items()},
        "date_axis_mm": [DATE_SPINE_Y, DATE_LABEL_Y],
        "loop_column_mm": [LOOP_X0, LOOP_X1, round(LOOP_X1 - LOOP_X0, 2)],
        "loop_bbox_mm": [round(v, 2) for v in LOOP_BBOX],
        "stage_centres_mm": {k: list(v) for k, v in STAGE.items()},
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
        "return_edge_mm": round(ret_len, 2),
        "forward_edges_mm": [round(v, 2) for v in fwd],
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
        f"[{STUDY}] data field {DATA_W:.1f} mm   loop "
        f"{meas['loop_footprint_mm'][0]:.1f} x {meas['loop_footprint_mm'][1]:.1f} mm"
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

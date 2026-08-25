"""Revision-5 panel-(a) design study C -- reclaimed width, regrouped rows.

Study C pushes the two levers that studies A and B left alone:

1.  TWO-LINE ROW NAMES. `Root-Zone` / `Depletion` instead of one long line.
    The label column falls from 26.2 mm to 15.4 mm and the temporal data field
    grows to 130.4 mm -- 10.4 mm more than A and B. The cost is stated
    plainly: with the label set on two lines there is no room for a unit line
    beside it, so study C carries no units in panel (a) and would need them in
    the caption. That trade is the point of the study.

2.  ROW ORDER. The two water inputs are made adjacent -- Daily Forcing then
    Irrigation -- so the reader sees everything that puts water into the
    profile before seeing the profile's response (Root-Zone Depletion) and
    then the flux that leaves it (Daily ET). A and B keep the handoff's
    order, in which depletion is separated from irrigation.

The inverse element is the Sec. 5.3 horizontal track without any enclosing
field: `Run` and `Compare` on one line, `Update` dropped below Compare, and a
single strong sweep from `Update` back out to `Run`. Both inputs descend into
the track in parallel from above and the conditioned parameters drop out of
`Update` below it. Every input and the exit is 3.3-4.6 mm long; the
only long stroke is the 10.2 mm return itself, which is the point.

Run:
  uv run python paper/figures/proofs/fig01_evidence_190_r5/build_fig01_r5_studyC.py
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

STUDY = "fig01_r5_studyC"

# ===========================================================================
# LAYOUT
# ===========================================================================

A_HEAD_Y = 114.6
RECORD_Y = 110.4

LABEL_R = 15.4  # two-line names: a much narrower label column
TICKNUM_R = 21.0
PA_X0 = 22.2
PA_X1 = 152.6  # -> 130.4 mm of temporal width
TICK_STUB = 0.9
LEADING = 3.3  # mm between the two label lines

ROW_GUTTER = 2.6
ROWS: dict[str, tuple[float, float]] = {
    "etf_ensemble": (100.0, 107.0),
    "ndvi_captures": (90.6, 97.4),
    "daily_forcing": (81.0, 88.0),
    "irrigation": (72.0, 78.4),
    "rz_depletion": (61.6, 69.4),
    "et_comparison": (51.0, 59.0),
}
ROW_LABEL = {
    "etf_ensemble": ("ETf", "Ensemble"),
    "ndvi_captures": ("NDVI", "Captures"),
    "daily_forcing": ("Daily", "Forcing"),
    "irrigation": ("Irrigation",),
    "rz_depletion": ("Root-Zone", "Depletion"),
    "et_comparison": ("Daily ET",),
}
DOMAIN = {
    "etf_ensemble": (0.0, 1.6),
    "ndvi_captures": (0.2, 1.0),
    "daily_forcing": (0.0, 18.0),
    "irrigation": (0.0, 25.0),
    "rz_depletion": (0.0, 16.0),
    "et_comparison": (0.0, 12.0),
}
TICKS = {
    "etf_ensemble": [(0.0, "0.0"), (1.6, "1.6")],
    "ndvi_captures": [(0.2, "0.2"), (1.0, "1.0")],
    "daily_forcing": [(0.0, "0"), (18.0, "18")],
    "irrigation": [(0.0, "0"), (25.0, "25")],
    "rz_depletion": [(0.0, "0"), (8.0, "8"), (16.0, "16")],
    "et_comparison": [(0.0, "0"), (6.0, "6"), (12.0, "12")],
}

DATE_SPINE_Y = 49.2
DATE_LABEL_Y = 46.4
GUIDE_TOP = 107.0
GUIDE_BOT = 51.0

# ---- the horizontal track, no enclosing field ---------------------------
LOOP_X0, LOOP_X1 = 155.6, 187.0
TITLE_Y = 103.0
IN_Y = 98.4  # both input labels, one line
STAGE = {
    "run_balance": (162.4, 92.0),
    "compare": (177.0, 92.0),
    "update_parameters": (171.3, 84.0),
}
EDGE_RUN_COMPARE = ((165.8, 92.0), (170.6, 92.0))
EDGE_COMPARE_UPDATE = ((177.6, 90.3), (180.6, 86.8), (176.4, 85.6))
EDGE_UPDATE_RUN = ((166.4, 84.7), (157.2, 86.6), (160.7, 90.3))
DRV_ARROW = ((162.4, 97.5), (162.4, 94.2))
SWE_ARROW = ((177.0, 97.5), (177.0, 94.2))
EXIT_ARROW = ((171.3, 82.4), (171.3, 77.8))
EXIT_LABEL_Y = 73.8

XLIM = (-1.5, 120.5)
MEMBER_MS = 2.6
DIAMOND_MS = 4.6


def xmm(d):
    return PA_X0 + (np.asarray(d, dtype=float) - XLIM[0]) / (XLIM[1] - XLIM[0]) * (PA_X1 - PA_X0)


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

        lines = ROW_LABEL[rid]
        top = mid + (len(lines) - 1) * LEADING / 2.0
        for j, s in enumerate(lines):
            mmtext(
                ov,
                LABEL_R,
                top - j * LEADING,
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

        for v, s in TICKS[rid]:
            ty = y0 + (v - d0) / (d1 - d0) * (y1 - y0)
            ov.add_line(
                Line2D([PA_X0, PA_X0 - TICK_STUB], [ty, ty], color=C_AXIS, lw=LW_AXIS, zorder=6)
            )
            va = "top" if abs(ty - y1) < 0.05 else ("bottom" if abs(ty - y0) < 0.05 else "center")
            ticktext(ov, TICKNUM_R, ty, s, pt=FS_TICK, ha="right", va=va)

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
        "precipitation, mm/d",
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

    rid = "irrigation"
    ax = row_axes(rid)
    irr = ts["irr_applied"].to_numpy()
    ev = irr > 0
    tag(ax.vlines(day[ev], 0.0, irr[ev], color=C_SWIM, lw=1.0, zorder=4), "marks-irrigation-stems")
    row_frame(rid)
    record_axis(rid, [irr])

    rid = "rz_depletion"
    ax = row_axes(rid)
    rz = ts["rz_depletion"].to_numpy()
    ax.fill_between(day, 0, rz, color=C_SWIM, alpha=0.10, linewidth=0, zorder=2)
    tag(
        ax.plot(day, rz, color=C_SWIM, lw=LW_DATA, zorder=4, clip_on=False)[0], "marks-rz-depletion"
    )
    row_frame(rid)
    record_axis(rid, [rz])

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
        "Simulated, mm/d",
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
    # the horizontal track
    # =====================================================================
    mmtext(
        ov,
        171.3,
        TITLE_Y,
        "Inverse Estimation",
        cls="title",
        pt=FS_STRUCT,
        weight="semibold",
        ha="center",
        gid="label-inverse-estimation",
    )

    BOX = {}
    for sid, label in (
        ("run_balance", "Run"),
        ("compare", "Compare"),
        ("update_parameters", "Update"),
    ):
        cx, cy = STAGE[sid]
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

    polyline(ov, list(EDGE_RUN_COMPARE), C_INV, 0.9, head=1.4, zorder=6, rid="cycle-run-to-compare")
    polyline(
        ov,
        bezier_pts(*EDGE_COMPARE_UPDATE),
        C_INV,
        0.9,
        head=1.4,
        zorder=6,
        rid="cycle-compare-to-update",
    )
    polyline(
        ov, bezier_pts(*EDGE_UPDATE_RUN), C_INV, 1.15, head=1.5, zorder=6, rid="cycle-update-to-run"
    )

    mmtext(
        ov,
        STAGE["run_balance"][0],
        IN_Y,
        "Daily Drivers",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="center",
        va="baseline",
        gid="label-daily-drivers",
    )
    polyline(ov, list(DRV_ARROW), C_INV, 0.8, head=1.3, zorder=6, rid="driver-to-run")
    mmtext(
        ov,
        STAGE["compare"][0],
        IN_Y,
        "ETf + SWE",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="center",
        va="baseline",
        gid="label-etf-swe",
    )
    polyline(
        ov, list(SWE_ARROW), C_INV, 0.8, head=1.3, zorder=6, rid="constraint-etf-swe-to-compare"
    )

    polyline(
        ov, list(EXIT_ARROW), C_INV, 0.9, head=1.3, zorder=6, rid="exit-conditioned-parameters"
    )
    mmtext(
        ov,
        171.3,
        EXIT_LABEL_Y,
        "Conditioned Parameters",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="center",
        va="baseline",
        gid="label-conditioned-parameters",
    )

    pb = K.draw_panel_b(fig, ov, F, text_w_mm)

    # =====================================================================
    # composition guards
    # =====================================================================
    DATA_W = PA_X1 - PA_X0
    assert DATA_W >= 128.0, f"study C must reclaim width; got {DATA_W:.1f} mm"
    assert TICKNUM_R + 0.3 <= PA_X0 - TICK_STUB
    widest_num = max(text_w_mm(t[1], FS_TICK) for r in TICKS for t in TICKS[r])
    assert TICKNUM_R - widest_num > LABEL_R + 1.0
    assert LOOP_X1 - LOOP_X0 <= 35.0
    assert LOOP_X0 > PA_X1 + 2.0
    # the two water inputs are adjacent, and depletion sits between irrigation
    # and the flux it produces
    order = list(ROWS)
    assert order.index("irrigation") == order.index("daily_forcing") + 1
    assert order.index("rz_depletion") == order.index("irrigation") + 1
    assert order.index("et_comparison") == order.index("rz_depletion") + 1
    ys = sorted(ROWS.values())
    for (_a0, a1), (b0, _b1) in zip(ys, ys[1:], strict=False):
        assert round(b0 - a1, 3) == ROW_GUTTER, f"uneven row gutter at {a1}"
    # every route in the inverse element is short
    for rt in K.ROUTES:
        if not rt["id"].startswith(("cycle-", "constraint-", "driver-", "exit-")):
            continue
        pts = np.array(rt["pts"])
        length = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
        assert length <= 16.0, f"{rt['id']} is {length:.1f} mm -- not a short route"
        x0b, _, x1b, _ = rt["bbox"]
        assert x0b >= LOOP_X0 - 0.5 and x1b <= LOOP_X1 + 0.5, f"{rt['id']} leaves the column"
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
    loop_pts = [
        p
        for rt in K.ROUTES
        if rt["id"].startswith(("cycle-", "constraint-", "driver-", "exit-"))
        for p in rt["pts"]
    ]
    lx = [p[0] for p in loop_pts]
    LOOP_BBOX = (min(lx), EXIT_LABEL_Y - 0.7, max(lx), TITLE_Y + K.ink_h_mm(FS_STRUCT) * 0.75)

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
        "ticknum_right_mm": TICKNUM_R,
        "rows_mm": {k: list(v) for k, v in ROWS.items()},
        "row_order": order,
        "row_gutter_mm": ROW_GUTTER,
        "units_in_panel_a": False,
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

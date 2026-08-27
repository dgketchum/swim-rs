"""Revision-6 recomposition of the r5 selected synthesis.

Three structural changes against the byte-identical frozen data; every mark,
colour, and datum otherwise inherits from `build_fig01_r5_selected.py`:

1.  CYCLE DIAGRAM MOVED TO THE SUPPLEMENT. The circular Run/Compare/Update
    inverse-estimation ring, its ETf + SWE constraint box, and the two
    component box columns leave the main figure entirely; they are rebuilt
    at their own page size by `build_figS_inverse_cycle.py` in this
    directory, which now carries the cycle-architecture audit contract.
    The figures plan flags the supplement diagram for a later rework at a
    much higher degree of flow fidelity.

2.  PANEL ORDER SWAPPED. The experiment-flow maps (E1 -> E2 -> E3 with the
    class-specific parameter relay) move to the TOP of the page as panel
    (a), drawn by `draw_panel_b` under the module-level `PANEL_DY = 75.0`
    shift in `fig01_r6_common.py`. The six-row time-series stack moves to
    the BOTTOM as panel (b), everything shifted down 40.8 mm.

3.  TIME SERIES EXPANDED INTO THE FREED SPACE. With the ring column gone,
    the temporal field widens from `PA_X1 = 149.6` to `187.0` mm (data
    field 127.3 -> 164.7 mm); the row bands, gutters, marker keys, and
    label treatment are unchanged from r5.

Panel (a)'s render overrides carry over verbatim: orthogonal E1->E3 route,
symmetric E2 latitude bounds, USGS NAIP orthoimagery under the E3 fields
(public domain, provenance in `assets/`), and the E1 locator.

Run:
  uv run python paper/figures/proofs/fig01_evidence_190_r6/build_fig01_r6.py
"""

from __future__ import annotations

from pathlib import Path

import fig01_r6_common as K
import numpy as np
import pandas as pd
from fig01_r6_common import (
    C_AXIS,
    C_DATUM,
    C_ETO,
    C_GUIDE,
    C_HELD,
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
    mmtext,
    tag,
    ticktext,
)
from matplotlib.lines import Line2D

STUDY = "fig01_r6"

# ===========================================================================
# LAYOUT
# ===========================================================================

# the time series is now the LOWER panel (b); the maps above are panel (a),
# drawn by draw_panel_b under the common module's PANEL_DY shift
B_TS_HEAD_Y = 73.6
RECORD_Y = 69.4

LABEL_R = 15.8  # compact treatment (study C), Gate A wants ~15-18 mm
TICKNUM_R = 21.0
PA_X0 = 22.3
PA_X1 = 187.0  # widened from r5's 149.6: the ring column is gone
TICK_STUB = 0.9
LABEL_LEAD = 3.0  # mm between label lines (incl. the unit line)

ROW_GUTTER = 3.0
ROWS: dict[str, tuple[float, float]] = {  # r5 bands shifted down 40.8 mm
    "etf_ensemble": (59.8, 66.2),
    "ndvi_captures": (50.2, 56.8),
    "daily_forcing": (41.0, 47.2),
    "rz_depletion": (29.6, 38.0),
    "irrigation": (21.0, 26.6),  # raised from study B's 4.8 mm
    "et_comparison": (10.2, 18.0),
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

DATE_SPINE_Y = 8.4
DATE_LABEL_Y = 5.6
GUIDE_TOP = 66.2
GUIDE_BOT = 10.2

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
        B_TS_HEAD_Y,
        "(b)",
        cls="title",
        pt=FS_PANEL,
        weight="bold",
        gid="label-panel-b-letter",
    )
    mmtext(
        ov,
        MARGIN_MM + 5.8,
        B_TS_HEAD_Y,
        "Sparse Satellite Constraints to Daily State",
        cls="title",
        pt=FS_PANEL_HEAD,
        weight="semibold",
        gid="label-panel-b-heading",
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

    # unboxed marker key (RSE convention: glyph + near-black definitional label,
    # no box) in the empty pre-capture corner — first capture is day 26. The
    # whisker-with-dots glyph defines members and their min-max span together.
    kgx, klx = 24.6, 26.4
    tag(
        ov.plot(
            [kgx],
            [64.6],
            ls="none",
            marker="D",
            ms=DIAMOND_MS,
            mfc="none",
            mec=C_TARGET,
            mew=0.9,
            zorder=6,
        )[0],
        "key-etf-mean-glyph",
    )
    mmtext(
        ov,
        klx,
        64.6,
        "Ensemble mean",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-key-etf-mean",
    )
    ov.add_line(
        tag(
            Line2D([kgx, kgx], [60.5, 62.7], color=C_MINMAX, lw=0.9, zorder=5),
            "key-etf-range-glyph",
        )
    )
    tag(
        ov.plot(
            [kgx, kgx],
            [61.0, 62.2],
            ls="none",
            marker="o",
            ms=MEMBER_MS,
            mfc=C_MEMBER,
            mec="white",
            mew=0.35,
            zorder=6,
        )[0],
        "key-etf-members-glyph",
    )
    mmtext(
        ov,
        klx,
        61.6,
        "Members",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-key-etf-members",
    )
    etf_key_right = klx + max(text_w_mm("Ensemble mean", FS_ROW), text_w_mm("Members", FS_ROW))
    first_cap_x = float(xmm(cap_day.min()))
    assert etf_key_right + 2.0 <= first_cap_x, (
        f"the ETf marker key ({etf_key_right:.2f} mm) crowds the first capture ({first_cap_x:.2f} mm)"
    )
    y0, y1 = ROWS[rid]
    assert y0 <= 60.5 and 64.6 + 1.1 <= y1, "the ETf marker key leaves its row band"

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

    # unboxed marker key in the clear lower-left corner (the day-19 Sentinel-2
    # capture sits at the top of the band; nearest low mark is day 26).
    tag(
        ov.plot(
            [kgx],
            [54.6],
            ls="none",
            marker="o",
            ms=2.8,
            mfc=C_SENSOR,
            mec="white",
            mew=0.4,
            zorder=6,
        )[0],
        "key-ndvi-landsat-glyph",
    )
    mmtext(
        ov,
        klx,
        54.6,
        "Landsat",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-key-ndvi-landsat",
    )
    tag(
        ov.plot(
            [kgx],
            [52.0],
            ls="none",
            marker="s",
            ms=2.8,
            mfc="none",
            mec=C_SENSOR,
            mew=0.7,
            zorder=6,
        )[0],
        "key-ndvi-sentinel-glyph",
    )
    mmtext(
        ov,
        klx,
        52.0,
        "Sentinel-2",
        cls="direct_label",
        pt=FS_ROW,
        color=C_TEXT,
        ha="left",
        va="center",
        bbox=HALO,
        gid="label-key-ndvi-sentinel",
    )
    ndvi_key_right = klx + max(text_w_mm("Landsat", FS_ROW), text_w_mm("Sentinel-2", FS_ROW))
    first_low_ndvi_x = float(xmm(day[mL].min()))
    assert ndvi_key_right + 2.0 <= first_low_ndvi_x, (
        f"the NDVI marker key ({ndvi_key_right:.2f} mm) crowds the first low capture "
        f"({first_low_ndvi_x:.2f} mm)"
    )
    y0, y1 = ROWS[rid]
    assert y0 <= 52.0 - 1.1 and 54.6 + 1.1 <= y1, "the NDVI marker key leaves its row band"

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
    assert PA_X1 <= W_MM - MARGIN_MM, "the temporal field overruns the right margin"
    assert 15.0 <= LABEL_R <= 18.0, "the label gutter left C's compact 15-18 mm treatment"
    assert TICKNUM_R + 0.4 <= PA_X0 - TICK_STUB
    widest_num = max(text_w_mm(t[1], FS_TICK) for r in TICKS for t in TICKS[r])
    assert TICKNUM_R - widest_num > LABEL_R + 1.0
    # centred numerals: adjacent rows' ink must not meet across a gutter
    ink = K.ink_h_mm(FS_TICK)
    assert ROW_GUTTER - ink > 0.3, "centred numerals collide across the row gutter"
    ys = sorted(ROWS.values())
    for (a0_, a1_), (b0_, _b1) in zip(ys, ys[1:], strict=False):
        assert round(b0_ - a1_, 3) == ROW_GUTTER, f"uneven row gutter at {a1_}"
        _ = a0_

    # panel stacking: the maps (panel a) sit above the time series (panel b);
    # the lowest map-block ink is the E3 key's bottom line (descenders reach
    # ~0.65 mm below a 7.5 pt baseline)
    assert B_TS_HEAD_Y + K.ink_h_mm(FS_PANEL) * 0.78 <= min(K.E3_KEY_LINES) - 0.65 - 0.8, (
        "the panel (b) heading rises into panel (a)'s E3 key"
    )
    assert RECORD_Y + K.ink_h_mm(FS_ROW) * 0.78 <= B_TS_HEAD_Y - 0.65 - 0.6, (
        "the record label collides with the panel (b) heading"
    )
    assert GUIDE_TOP <= RECORD_Y - 0.65 - 0.8, "the ETf row rises into the record label"
    assert DATE_LABEL_Y - 0.65 >= MARGIN_MM - 0.05, "the month labels drop below the margin"

    # panel (a) render overrides actually applied and recorded
    assert abs(pb["e2_lat_bound_deg"] - 56.0997) < 0.01, pb["e2_lat_bound_deg"]
    assert "e3_route_vertices_mm" in pb, "the orthogonal E1->E3 route was not drawn"
    assert "e3_basemap" in pb, "the E3 basemap was not drawn"
    assert "e1_e3_locator_epsg5070" in pb, "the E3 locator was not drawn on the E1 map"

    # the cycle-architecture contract now lives with the supplement figure
    # (build_figS_inverse_cycle.py); this page carries the axis contract only
    m = K.audit_scientific(F, AXIS, [FLUX_INK, pb["meter_ink"]], None)

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
        "panel_order": {
            "a": "experiment flow (E1 -> E2 -> E3 maps), top",
            "b": "time-series stack, bottom",
        },
        "panel_dy_mm": K.PANEL_DY,
        "inverse_cycle": "moved to the supplement (build_figS_inverse_cycle.py)",
        "e3_key_x_mm": K.E3_KEY_X,
        "e3_key_baselines_mm": list(K.E3_KEY_LINES),
        "panel_b_marker_keys": {
            "style": "unboxed glyph + near-black label (RSE convention)",
            "glyph_x_mm": 24.6,
            "label_x_mm": 26.4,
            "etf_ensemble": {"Ensemble mean": 64.6, "Members": 61.6},
            "ndvi_captures": {"Landsat": 54.6, "Sentinel-2": 52.0},
        },
    }
    meas = {
        **m,
        "data_field_width_mm": round(DATA_W, 2),
        "label_gutter_mm": LABEL_R,
        "label_to_numeral_gap_mm": round(TICKNUM_R - widest_num - LABEL_R, 2),
        "numeral_to_spine_gap_mm": round(PA_X0 - TICK_STUB - TICKNUM_R, 2),
        "panel_gap_mm": round(
            min(K.E3_KEY_LINES) - 0.65 - (B_TS_HEAD_Y + K.ink_h_mm(FS_PANEL) * 0.78), 2
        ),
        **{f"panel_a_{k}": v for k, v in pb.items() if k != "meter_ink"},
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
        f"E2 bounds ±{pb['e2_lat_bound_deg']:.2f}°"
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

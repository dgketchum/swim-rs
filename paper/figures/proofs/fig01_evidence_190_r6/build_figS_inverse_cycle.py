"""Supplementary figure: the inverse-estimation cycle, standalone.

Revision 6 removed the circular Run/Compare/Update ring from the main
figure (`build_fig01_r6.py`); this builder re-draws it byte-for-byte in
geometry -- Study B's 7.9 mm ring, arc angles, the ETf + SWE constraint
box, and the two component box columns -- translated onto its own
90 x 72 mm supplement page (dx = -124.6 mm, dy = -42.0 mm against the r5
main-figure coordinates). The cycle-architecture audit contract moves
here with it: this page, not the main figure, asserts the frozen
architecture's edges, exit, inputs, and forbidden connections.

The figures plan flags this diagram for a later rework at a much higher
degree of flow fidelity; this build preserves the r5 rendering so the
supplement stays truthful to what the main figure showed until then.

Run:
  uv run python paper/figures/proofs/fig01_evidence_190_r6/build_figS_inverse_cycle.py
"""

from __future__ import annotations

import fig01_r6_common as K
import numpy as np
from fig01_r6_common import (
    C_INV,
    C_TEXT,
    FS_LABEL,
    FS_ROW,
    FS_STRUCT,
    MARGIN_MM,
    arc_pts,
    mmtext,
    polyline,
    tag,
)
from matplotlib.patches import FancyBboxPatch

STUDY = "figS_inverse_cycle"

PAGE_W = 90.0
PAGE_H = 72.0

# ---- the ring column, translated from the r5 main figure -------------------
LOOP_X0, LOOP_X1 = 27.4, 62.4
FIELD = (28.4, 39.6, 33.2, 20.6)
RING_C = (45.0, 49.0)
RING_R = 7.9  # study B's radius, unchanged
TITLE_Y = 66.6  # 'Inverse Estimation'
# 'ETf + SWE' rides in a red-shaded box styled like the component boxes;
# ONE box, ONE label: the architecture forbids a separate swe edge into
# Compare, so the snow constraint stays inline with the ETf constraint.
SWE_BOX = (37.0, 61.2, 16.0, 3.8)  # x0, y0, w, h
SWE_FILL, SWE_EDGE = "#F8E4E2", "#D9A49E"
ANG = {"compare": 90.0, "update_parameters": -30.0, "run_balance": 210.0}
ARCS = {  # clockwise, with a word-width gap at each end (study B, r=7.9)
    "cycle-run-to-compare": (190.0, 126.0),
    "cycle-compare-to-update": (54.0, -7.0),
    "cycle-update-to-run": (-53.0, -127.0),
}
DRV_ARROW = ((36.0, 37.8), (37.7, 42.9))  # role-level feed into Run
EXIT_ARROW = ((53.0, 42.9), (54.0, 39.2))  # role-level exit from Update

# two component box columns beneath the ring: green daily inputs on the
# left (feeding Run), blue conditioned parameters on the right (from Update)
BOX_W = 16.0
BOX_H = 3.8
BOX_PITCH = 5.0
BOX_TOP = 31.8  # top edge of the first box in both columns
DRV_COL = {
    "x0": 28.0,
    "header": ("Daily Drivers",),
    "head_y": (35.1,),  # centred on the right column's two-line band
    "fill": "#E7F1E4",
    "edge": "#A8CBA4",
    "items": ("NDVI", "ETo", "Precip.", "Solar Rad.", "Air Temp."),
}
CP_COL = {
    "x0": 46.0,
    "header": ("Conditioned", "Parameters"),
    "head_y": (36.6, 33.6),
    "fill": "#E2ECF6",
    "edge": "#A9C6E0",
    "items": ("AWC", "MAD", "Kcb", "Ks", "Kr", "Snowmelt"),
}


def ring_xy(sid, r=None):
    a = np.radians(ANG[sid])
    r = RING_R if r is None else r
    return RING_C[0] + r * np.cos(a), RING_C[1] + r * np.sin(a)


def main() -> None:
    F = K.Frozen()
    fig, ov, family, faces = K.new_figure(STUDY, w_mm=PAGE_W, h_mm=PAGE_H)
    text_w_mm = K.measurer(fig)

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

    # ---- the two component box columns beneath the ring ----
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

    # =====================================================================
    # composition guards (r5's loop guards, on the supplement page)
    # =====================================================================
    assert LOOP_X1 - LOOP_X0 <= 35.0
    # every arc is the SAME radius, so no edge of the cycle can read as weaker
    for rid_ in ARCS:
        rt = next(r for r in K.ROUTES if r["id"] == rid_)
        rr = [float(np.hypot(p[0] - RING_C[0], p[1] - RING_C[1])) for p in rt["pts"]]
        assert max(abs(np.array(rr) - RING_R)) < 1e-6, f"{rid_} is not on the ring"
    # ring and words inside the field, field inside the page margins
    assert fx >= MARGIN_MM and fx + fw <= PAGE_W - MARGIN_MM + 0.2
    for sid, (bx0, by0, bx1, by1) in BOX.items():
        assert fx <= bx0 and bx1 <= fx + fw, f"{sid} label leaves the field"
        assert fy <= by0 and by1 <= fy + fh, f"{sid} label leaves the field"

    # the box columns: labels inside their boxes, headers inside their
    # columns, columns inside the loop band, below (not inside) the field,
    # clear of each other and of the bottom margin
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
    assert min(drv_bot, cp_bot) >= MARGIN_MM - 0.05, "a box column drops below the margin"
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
    assert TITLE_Y + K.ink_h_mm(FS_LABEL) * 0.78 <= PAGE_H - MARGIN_MM + 0.2, (
        "the title enters the top margin"
    )

    # the cycle-architecture contract lives here now, not on the main page
    m = K.audit_scientific(
        F,
        None,
        [],
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
        "page_mm": [PAGE_W, PAGE_H],
        "translated_from_r5_mm": {"dx": -124.6, "dy": -42.0},
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
            "column_headers": [FS_STRUCT, "semibold"],
        },
    }
    meas = {
        **m,
        "ring_circumference_mm": round(2 * np.pi * RING_R, 2),
        "box_label_max_mm": round(
            max(text_w_mm(s, FS_ROW) for s in DRV_COL["items"] + CP_COL["items"]), 2
        ),
        "box_label_min_margin_mm": round(
            BOX_W - max(text_w_mm(s, FS_ROW) for s in DRV_COL["items"] + CP_COL["items"]), 2
        ),
        "column_gap_mm": round(CP_COL["x0"] - (DRV_COL["x0"] + BOX_W), 2),
        "columns_bottom_mm": [round(drv_bot, 2), round(cp_bot, 2)],
    }
    out = K.export(
        fig,
        STUDY,
        {
            "family": family,
            "faces": faces,
            "layout": layout,
            "measured": meas,
            "architecture_sha256": F.arch_sha,
            "example_csv_sha256": F.csv_sha,
        },
        w_mm=PAGE_W,
        h_mm=PAGE_H,
    )
    print(f"[{STUDY}] page {PAGE_W:.0f} x {PAGE_H:.0f} mm   ring r={RING_R} mm")
    return out


if __name__ == "__main__":
    main()

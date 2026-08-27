"""Shared machinery for revision 6 of Figure 1 (and its supplementary figure).

Revision 6 (2026-08-26, user-directed) forks the r5 module with three changes,
still against the frozen revision-4 data (nothing re-extracts, rebuilds, or
re-contracts anything):

* the experiment-flow map block becomes panel **(a)** at the TOP of the page
  (one uniform ``PANEL_DY = +75.0`` mm shift of the r5 panel-b geometry);
* the inverse-estimation cycle leaves the main figure for a separate
  supplementary figure (`build_figS_inverse_cycle.py`), so `audit_scientific`
  now takes `axis_audit=None` (supplement) or `cycle=None` (main figure) and
  runs each check group only where its content is drawn;
* `new_figure` / `export` accept a page size so the supplement can render at
  single-column dimensions.

What this module owns
---------------------
* the canvas, colour and type system (handoff Sec. 9);
* the frozen-data load and every **scientific-integrity** check of Sec. 11;
* a drawn-string ledger, kept for review only -- handoff Sec. 10 now says the
  ledger "need not be identical to a pre-approved architecture string list", so
  there is no string-parity assertion anywhere in revision 5;
* stable SVG element ids (Sec. 15.2), the markup channel;
* mm-space primitives, the route ledger, and the map/route firewalls; and
* panel (b), which Sec. 16.3 fixes for this round apart from the E3
  aggregation key, which moves out of the inter-panel strip to sit adjacent to
  the E3 map (Sec. 5.4.2).

What each study owns
--------------------
Panel (a) entirely: row set, row heights, gutters, domains, ticks, datums,
month guides, the inverse-loop form, and all wording.

Assertions retained (Sec. 13: machine checks confirm data integrity, dimensions,
clipping and semantic edges -- they cannot certify composition):

* page is exactly 190 x 120 mm;
* cohort counts 60/66/50, classes 39/21 and 13/53, overlap 13, `MB_Pch` present;
* the example is the frozen site and window, 120 rows and 15 captures;
* every plotted column has recorded provenance and no audit-only column is drawn;
* ETf member marks reconcile to the frozen member count and target mean;
* no filled-NDVI column reaches the display;
* every row's display domain contains every mark **with visible headroom**;
* the ET/flux traces share one date mapping, one y mapping, one region;
* the inverse relationships form the closed semantic cycle with an exit;
* no held-out observation has an edge into the cycle, and no arrowhead points
  into one;
* the E1 applied-water series is never connected to the E3 meter key;
* the E3 key carries no data-like meter mark;
* the E3 transfer path branches at the irrigated E1 token, not at E2;
* no metric, member name, run label or `Held-Out Evaluation` heading is drawn;
* Source Sans 3 resolves (a DejaVu fallback aborts) and nothing is below 7.5 pt.

Dropped deliberately for revision 5: exact string parity, region-count, and
tick-count assertions. Gate A is a human visual decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from shapely.geometry import box as shp_box  # noqa: E402

# ===========================================================================
# canvas, type and colour  (handoff Sec. 9)
# ===========================================================================

MM = 1.0 / 25.4
W_MM, H_MM = 190.0, 120.0
MARGIN_MM = 3.0

FS_PANEL = 10.5  # panel labels (a) / (b), bold
FS_PANEL_HEAD = 9.0  # panel headings, semibold title case
FS_STRUCT = 8.5  # map headings and structural labels, semibold
FS_LABEL = 8.0  # direct labels on data
FS_ROW = 7.5  # row labels, units, stage labels
FS_TICK = 7.5  # tick numerals -- hard floor
FS_MIN = 7.5

C_TARGET = "#E69F00"  # satellite ET target
C_SWIM = "#0072B2"  # SWIM-RS state and output
C_INV = "#7B3294"  # inverse-estimation cycle
C_HELD = "#202124"  # held-out observations
C_E1, C_E2, C_E3 = "#4477AA", "#228833", "#AA3377"

C_TEXT = "#202124"  # near-black: ALL reader-facing text
C_AXIS = "#9A9DA1"  # y-spines, tick stubs, date spine
C_DATUM = "#C9CBCD"  # per-row lower-bound datum
C_GUIDE = "#E2E4E6"  # shared April-July month guides
C_MINMAX = "#6C7176"  # ETf min-max line
C_MEMBER = "#4F5358"  # ETf member marks
C_SENSOR = "#3F4246"  # NDVI capture marks
C_ETO = "#3F4246"  # ETo line
C_PRECIP = "#8FA0AC"  # precipitation stems
C_LAND = "#ECECE7"
C_BOUND = "#A9A9A3"
C_HAIR = "#B6B6B0"
C_CONTEXT = "#CBCBC5"

LW_AXIS = 0.55
LW_SPINE = 0.6
LW_DATUM = 0.35
LW_GUIDE = 0.4
LW_DATA = 1.0
LW_ROUTE = 0.8

# ===========================================================================
# the experiment-flow map block -- panel (a) in revision 6. The geometry is
# revision 5's panel (b) shifted UP by a uniform PANEL_DY = +75.0 mm so the
# maps lead the figure and the time series reads below them.
# ===========================================================================

PANEL_DY = 75.0  # r5 bottom-panel geometry -> r6 top panel, one uniform shift

B_HEAD_Y = 39.6 + PANEL_DY
MAP_HEAD_Y = 35.4 + PANEL_DY
MAP_COUNT_Y = 32.2 + PANEL_DY
MAP_BOT = 4.0 + PANEL_DY
MAP_TOP = 30.6 + PANEL_DY
E2_TOP = 25.0 + PANEL_DY

E1_FRAME = (3.0, MAP_BOT, 48.0, MAP_TOP - MAP_BOT)
E2_FRAME = (85.0, MAP_BOT, 65.0, E2_TOP - MAP_BOT)
# Sec. 5.4.2: the aggregation key leaves the inter-panel strip and sits with
# the E3 geography, so the E3 frame gives up its lower third to the key block.
E3_FRAME = (152.0, 13.6 + PANEL_DY, 33.0, MAP_TOP - (13.6 + PANEL_DY))
E3_KEY_X = 152.0
E3_KEY_LINES = (10.0 + PANEL_DY, 6.8 + PANEL_DY, 3.6 + PANEL_DY)  # baselines, top to bottom

TOKEN_X0 = 51.0
FORK_X = 76.4
PATH_IRRIG_Y = 21.0 + PANEL_DY
PATH_RAINFED_Y = 9.5 + PANEL_DY
FORK_CLEARANCE_MIN = 8.0
E3_ARC_PTS = [
    (FORK_X, PATH_IRRIG_Y),
    (85.0, 27.4 + PANEL_DY),
    (100.0, 30.0 + PANEL_DY),
    (125.0, 32.4 + PANEL_DY),
    (140.0, 31.4 + PANEL_DY),
    (E3_FRAME[0] - 0.4, 27.0 + PANEL_DY),
]
RETAIN_CONUS_STATES = True
RETAIN_SLV_BASIN = False

CRS_ALBERS = 5070
MEMBER_COLS = [
    "etf_ssebop",
    "etf_ptjpl",
    "etf_sims",
    "etf_geesebal",
    "etf_eemetric",
    "etf_disalexi",
]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PKG = REPO / "paper/data/final/figures"
GPKG = PKG / "fig01_scope.gpkg"
ARCH_PATH = PKG / "fig01_architecture.json"
TS_PATH = PKG / "fig01_example_timeseries.csv"
SEL_PATH = PKG / "fig01_example_selection.json"

# ===========================================================================
# drawn-string ledger -- review aid, NOT a parity gate (handoff Sec. 10)
# ===========================================================================

DRAWN: list[dict] = []
GIDS: list[str] = []
ROUTES: list[dict] = []

# Scientific guards that survive the revision-5 relaxation. `Inverse
# Estimation`, `Run`, `Compare` and `Update` are Level 1 wording (Sec. 15.1)
# and are no longer blocked; metrics, member names, run identifiers and the
# retired `Held-Out Evaluation` heading still are.
FORBIDDEN_PATTERNS = [
    r"\bKGE\b",
    r"\bRMSE\b",
    r"\bMBE\b",
    r"\bNSE\b",
    r"R²",
    r"\br²\b",
    r"\bbias\b",
    r"\bssebop\b",
    r"\bptjpl\b",
    r"\bsims\b",
    r"\bgeesebal\b",
    r"\beemetric\b",
    r"\bdisalexi\b",
    r"\bpt-?jpl\b",
    r"\bee-?metric\b",
    r"\bRun\s*\d+",
    r"\brealization",
    r"\biteration",
    r"interpolat",
    r"\bECOSTRESS\b",
    r"\bPEST",
    r"\bIES\b",
    r"\bsigma\b",
    r"σ",
    r"Φ",
    r"Held-Out Evaluation",
]


def _slug(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return out or "x"


def claim_gid(gid: str) -> str:
    assert gid not in GIDS, f"duplicate SVG element id {gid!r}"
    GIDS.append(gid)
    return gid


def tag(artist, gid: str):
    artist.set_gid(claim_gid(gid))
    return artist


def _register(text: str, cls: str, pt: float, color: str) -> None:
    DRAWN.append({"string": text, "class": cls, "font_pt": round(float(pt), 2), "color": color})


HALO = {"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.9}
_N = {"tick": 0, "glyph": 0}


def mmtext(
    ax,
    x,
    y,
    s,
    *,
    cls="direct_label",
    pt=FS_ROW,
    gid=None,
    ha="left",
    va="baseline",
    weight="normal",
    color=C_TEXT,
    rotation=0,
    zorder=7,
    **kw,
):
    """One reader-facing string: recorded, coloured, and given a stable id."""
    _register(s, cls, pt, color)
    t = ax.text(
        x,
        y,
        s,
        fontsize=pt,
        ha=ha,
        va=va,
        fontweight=weight,
        color=color,
        rotation=rotation,
        zorder=zorder,
        **kw,
    )
    return tag(t, gid or f"label-{_slug(s)}")


def ticktext(ax, x, y, s, *, pt=FS_TICK, ha="right", va="center", color=C_TEXT, zorder=7, **kw):
    """A tick numeral or month abbreviation generated from the frozen data."""
    _register(s, "generated_tick", pt, color)
    _N["tick"] += 1
    t = ax.text(x, y, s, fontsize=pt, ha=ha, va=va, color=color, zorder=zorder, **kw)
    return tag(t, f"tick-{_N['tick']:02d}-{_slug(s)}")


def glyphtext(
    ax, x, y, s, *, pt=FS_ROW, ha="left", va="baseline", color=C_TEXT, zorder=7, gid=None
):
    """A relation glyph drawn as a character rather than a stroke."""
    _register(s, "relation_glyph", pt, color)
    _N["glyph"] += 1
    t = ax.text(x, y, s, fontsize=pt, ha=ha, va=va, color=color, zorder=zorder)
    return tag(t, gid or f"glyph-{_N['glyph']:02d}")


FONT_DIRS = [
    Path.home() / ".fonts" / "source-sans",
    Path("/usr/share/fonts/opentype/source-sans"),
    Path("/usr/share/fonts/truetype/source-sans"),
]


def register_typeface(salt: str) -> tuple[str, list[str]]:
    faces: list[str] = []
    for d in FONT_DIRS:
        if d.is_dir():
            for f in sorted(d.glob("SourceSans3-*.[ot]tf")):
                fm.fontManager.addfont(str(f))
                faces.append(f.name)
    names = {e.name for e in fm.fontManager.ttflist}
    family = "Source Sans 3" if "Source Sans 3" in names else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",  # Sec. 15.2: the SVG must stay editable
            "svg.hashsalt": salt,
            "text.color": C_TEXT,
            "axes.unicode_minus": False,
        }
    )
    assert family == "Source Sans 3", (
        f"Source Sans 3 did not register (resolved {family!r}); refusing to "
        "render Figure 1 in a fallback face"
    )
    return family, faces


# ---------------------------------------------------------------------------
# mm-space primitives
# ---------------------------------------------------------------------------


def arrowhead(ax, p0, p1, color, head=1.35, zorder=6, rid=None):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    n = float(np.hypot(dx, dy))
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    p = mpatches.Polygon(
        [
            (x1, y1),
            (x1 - head * ux + 0.42 * head * px, y1 - head * uy + 0.42 * head * py),
            (x1 - head * ux - 0.42 * head * px, y1 - head * uy - 0.42 * head * py),
        ],
        closed=True,
        facecolor=color,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_patch(p)
    if rid is not None:
        tag(p, f"arrowhead-{_slug(rid)}")
    return p


def polyline(ax, pts, color, lw, *, ls="-", arrow=True, head=1.35, zorder=6, rid=None, alpha=1.0):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    if rid is not None:
        ROUTES.append(
            {
                "id": rid,
                "pts": list(zip(xs, ys, strict=True)),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "tail": (xs[0], ys[0]),
                "tip": (xs[-1], ys[-1]),
                "arrow": bool(arrow),
            }
        )
    ln = Line2D(
        xs,
        ys,
        color=color,
        lw=lw,
        ls=ls,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_line(ln)
    if rid is not None:
        tag(ln, f"route-{_slug(rid)}")
    if arrow:
        arrowhead(ax, (xs[-2], ys[-2]), (xs[-1], ys[-1]), color, head=head, zorder=zorder, rid=rid)
    return ln


def arc_pts(cx, cy, r, a0, a1, n=48):
    """Points along a circular arc, degrees, counter-clockwise positive."""
    a = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + r * np.cos(a), cy + r * np.sin(a), strict=True))


def bezier_pts(p0, p1, p2, n=40):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t**2 * np.array(p2)
    return [tuple(map(float, q)) for q in p]


def fit_extent(bounds, frame_w, frame_h, pad_frac=0.04):
    x0, y0, x1, y1 = bounds
    dx, dy = x1 - x0, y1 - y0
    x0 -= dx * pad_frac
    x1 += dx * pad_frac
    y0 -= dy * pad_frac
    y1 += dy * pad_frac
    dx, dy = x1 - x0, y1 - y0
    want = frame_w / frame_h
    have = dx / dy
    if have < want:
        grow = (dy * want - dx) / 2.0
        x0 -= grow
        x1 += grow
    else:
        grow = (dx / want - dy) / 2.0
        y0 -= grow
        y1 += grow
    return x0, y0, x1, y1


def ink_h_mm(pt: float) -> float:
    return pt * 25.4 / 72.0


# ===========================================================================
# frozen data + the Sec. 11 scientific checks
# ===========================================================================


class Frozen:
    """The frozen revision-4 display package, loaded and checked once."""

    def __init__(self):
        self.arch = json.loads(ARCH_PATH.read_text())
        self.sel = json.loads(SEL_PATH.read_text())
        self.arch_sha = hashlib.sha256(ARCH_PATH.read_bytes()).hexdigest()
        self.csv_sha = hashlib.sha256(TS_PATH.read_bytes()).hexdigest()
        assert self.csv_sha == (
            "7e8307c2c43b3991651660adaf60f0bbba7179a4f6e5a0a694fc8de9beb2f3da"
        ), "the frozen example record changed; revision 5 is presentation-only"

        rec = self.arch["example_record"]
        self.record_label = "US-Bi1 (2017)"
        assert rec["record_label"] == self.record_label
        assert rec["record_label_color"] == C_TEXT
        assert rec["record_label_rendered_as_muted_gray"] is False

        ts = pd.read_csv(TS_PATH, parse_dates=["date"])
        assert len(ts) == 120, f"expected 120 daily rows, found {len(ts)}"
        cap = ts[ts["is_calibration_capture"]].copy()
        assert len(cap) == 15, f"expected 15 calibration captures, found {len(cap)}"
        assert ts["site_id"].nunique() == 1 and ts["site_id"].iloc[0] == rec["site_id"]
        win = rec["window"]
        assert str(ts["date"].min().date()) == win[0]
        assert str(ts["date"].max().date()) == win[1]
        dw = self.sel["displayed_window"]
        assert (dw["start"], dw["end"]) == (win[0], win[1]), "display crop differs"
        assert dw["cropped"] is False and dw["n_days"] == 120
        assert not any(c.startswith("ndvi") and ("fill" in c or "kcb" in c) for c in ts.columns), (
            "a filled NDVI column reached the display file"
        )
        assert float(np.nanmax(np.abs(ts["swe_audit"].to_numpy()))) == 0.0

        prov = self.sel["column_provenance"]
        plotted = MEMBER_COLS + [
            "date",
            "ndvi_landsat_raw",
            "ndvi_sentinel_raw",
            "is_calibration_capture",
            "etf_target_mean",
            "eto",
            "precip",
            "rz_depletion",
            "irr_applied",
            "swim_ET",
            "flux_ET",
        ]
        for c in plotted:
            assert c in prov, f"plotted column {c!r} has no recorded provenance"
            assert not prov[c]["display_role"].startswith("audit_only"), (
                f"audit-only column {c!r} must not be plotted"
            )
        assert prov["swe_audit"]["display_role"] == "audit_only_not_plotted"
        assert prov["capture_sensor"]["display_role"] == "audit_only"

        # ETf member marks must reconcile to the frozen member count and mean
        mem = cap[MEMBER_COLS].to_numpy(dtype=float)
        n_mem = np.isfinite(mem).sum(axis=1)
        assert np.array_equal(n_mem, cap["etf_member_count"].to_numpy()), (
            "plotted ETf member marks do not reconcile to the frozen member count"
        )
        assert np.allclose(np.nanmean(mem, axis=1), cap["etf_target_mean"].to_numpy(), atol=1e-9), (
            "the plotted member marks do not reconcile to the frozen target mean"
        )

        self.ts = ts
        self.cap = cap
        self.day = (ts["date"] - ts["date"].min()).dt.days.to_numpy().astype(float)
        self.cap_day = (cap["date"] - ts["date"].min()).dt.days.to_numpy().astype(float)
        self.members = mem

        self.e1 = gpd.read_file(GPKG, layer="e1_sites", engine="fiona")
        self.e2 = gpd.read_file(GPKG, layer="e2_sites", engine="fiona")
        self.e3 = gpd.read_file(GPKG, layer="e3_display", engine="fiona")
        self.conus = gpd.read_file(GPKG, layer="conus_context", engine="fiona")
        self.world = gpd.read_file(GPKG, layer="world_context", engine="fiona")
        self.slv = gpd.read_file(GPKG, layer="slv_context", engine="fiona")
        self.states = gpd.read_file(GPKG, layer="conus_states_context", engine="fiona")

        assert (len(self.e1), len(self.e2), len(self.e3)) == (60, 66, 50)
        assert self.e1["irrigation_class"].value_counts().to_dict() == {
            "irrigated": 39,
            "rainfed": 21,
        }
        assert self.e2["irrigation_class"].value_counts().to_dict() == {
            "rainfed": 53,
            "irrigated": 13,
        }
        assert int(self.e2["in_e1"].sum()) == 13, "E1/E2 overlap is not 13"
        assert (self.e1["display_id"] == "MB_Pch").sum() == 1, "MB_Pch missing from E1 scope"
        assert set(self.e3.geom_type) == {"Point"}, "E3 public display must be points only"
        assert not any(
            k in self.e3.columns for k in ("acres", "acreage", "agency_id", "source_id")
        ), "a restricted E3 identifier reached the display layer"

        # the semantic cycle and its firewalls still come from the contract
        cyc = self.arch["inverse_cycle"]
        self.cycle_edges = [tuple(e) for e in self.arch["cycle_edges"]]
        assert self.cycle_edges == [
            ("run_balance", "compare"),
            ("compare", "update_parameters"),
            ("update_parameters", "run_balance"),
        ], self.cycle_edges
        assert cyc["exit"]["edge"] == ["update_parameters", "daily_balance"]
        self.forbidden_edges = {tuple(e) for e in self.arch["forbidden_edges"]}
        assert ("etf_ensemble", "run_balance") in self.forbidden_edges
        assert ("applied_water", "meters") in self.forbidden_edges
        assert ("daily_et", "flux_et") in self.forbidden_edges


def new_figure(salt: str, w_mm: float = W_MM, h_mm: float = H_MM):
    family, faces = register_typeface(salt)
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=600)
    fig.patch.set_facecolor("white")
    got = fig.get_size_inches()
    assert abs(got[0] - w_mm * MM) < 1e-9 and abs(got[1] - h_mm * MM) < 1e-9, got
    ov = fig.add_axes([0, 0, 1, 1], zorder=9)
    ov.set_xlim(0, w_mm)
    ov.set_ylim(0, h_mm)
    ov.set_facecolor("none")
    ov.axis("off")
    tag(ov, "overlay-mm-space")
    return fig, ov, family, faces


def measurer(fig):
    renderer = fig.canvas.get_renderer()

    def text_w_mm(s, pt, weight="normal"):
        t = fig.text(0, 0, s, fontsize=pt, fontweight=weight)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        return bb.width / fig.dpi * 25.4

    return text_w_mm


# ===========================================================================
# panel (b) -- fixed this round (Sec. 16.3); only the E3 key relocates
# ===========================================================================

MS_TRI, MS_CIR = 9.5, 7.0
HALO_LW = 1.0


def draw_panel_b(
    fig,
    ov,
    F: Frozen,
    text_w_mm,
    *,
    e2_mode="legacy",
    e3_route="spline",
    e3_basemap=None,
    e3_hull=True,
    e1_e3_locator=False,
) -> dict:
    """Draw panel (b) exactly as accepted in revision 4, plus the E3 key.

    Two-pass marker rendering (all halos, then all fills) so co-located marks
    are never knocked out. No jitter, thinning or resampling: co-location is
    data truth.

    The selected synthesis (handoff Sec. 6.1/6.5) passes two render-level
    overrides; the defaults reproduce Studies A-C byte-for-byte:

    * ``e2_mode="symmetric"``   E2 latitude bounds become +/-(max |site
      latitude| + 5 deg), computed from the frozen ``e2_sites`` layer, with the
      frame height derived so a degree of latitude and a degree of longitude
      get equal millimetres (no stretch).
    * ``e3_route="orthogonal"`` the irrigated E1->E3 branch becomes the
      geometric up-and-over route (rise, over, short terminal) in place of the
      revision-4 spline. Source and destination are unchanged.
    * ``e3_basemap={"path": ..., "wash": 0.18, ...}`` draws a georegistered
      raster (same EPSG:5070 extent as the frame) beneath the E3 field marks
      in place of the flat land fill, under a white wash that keeps it
      recessive; the hull becomes a white/gray outline. All other keys are
      provenance and are echoed into the returned dict. Display-asset only:
      the frozen point geometry is untouched.
    * ``e3_hull=False`` drops the SLV hull outline from the E3 map (the
      basemap or land fill carries the geography instead).
    * ``e1_e3_locator=True`` draws the E3 map's extent as a small locator
      rectangle on the E1 CONUS map, in the E3 accent colour, so the valley
      is placed in continental context.
    """
    out: dict = {}

    # E2 frame + graticule bounds, resolved before anything references them.
    if e2_mode == "symmetric":
        phi = float(F.e2.geometry.y.abs().max()) + 5.0
        lon_pad = 10.0  # buffered a bit past the site envelope on both sides
        lon0 = float(F.e2.geometry.x.min()) - lon_pad
        lon1 = float(F.e2.geometry.x.max()) + lon_pad
        e2_w = 60.0
        e2_h = e2_w * (2.0 * phi) / (lon1 - lon0)  # 1 deg lat == 1 deg lon in mm
        e2f = (86.0, MAP_BOT, e2_w, e2_h)
        out["e2_lon_bounds_deg"] = [round(lon0, 4), round(lon1, 4)]
        out["e2_lon_pad_deg"] = lon_pad
        lat_c, lat_half = 0.0, phi
        out["e2_max_abs_site_lat_deg"] = round(phi - 5.0, 4)
        out["e2_lat_bound_deg"] = round(phi, 4)
        out["e2_frame_mm"] = [round(v, 3) for v in e2f]
    else:
        assert e2_mode == "legacy", e2_mode
        lon0, lon1, lat_c, lat_span = -127.0, 157.0, 7.2, 91.0
        lat_half = lat_span / 2.0
        e2f = E2_FRAME

    mmtext(
        ov,
        MARGIN_MM,
        B_HEAD_Y,
        "(a)",
        cls="title",
        pt=FS_PANEL,
        weight="bold",
        gid="label-panel-a-letter",
    )
    mmtext(
        ov,
        MARGIN_MM + 5.8,
        B_HEAD_Y,
        "E1 Source Cohort and Parallel Transfer",
        cls="title",
        pt=FS_PANEL_HEAD,
        weight="semibold",
        gid="label-panel-a-heading",
    )

    def mapax(fr, gid):
        x, y, w, h = fr
        ax = fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM], zorder=2)
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_color(C_HAIR)
            ax.spines[s].set_linewidth(0.45)
        tag(ax, gid)
        return ax

    def maphead(fr, heading, count):
        mmtext(ov, fr[0], MAP_HEAD_Y, heading, cls="title", pt=FS_STRUCT, weight="semibold")
        mmtext(ov, fr[0], MAP_COUNT_Y, count, cls="direct_label", pt=FS_ROW)

    def halo_pass(ax, x, y, size, marker, gid):
        tag(
            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                facecolor="white",
                edgecolor="white",
                linewidths=HALO_LW,
                zorder=4,
            ),
            gid,
        )

    def fill_pass(ax, x, y, size, marker, color, gid):
        tag(
            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor=color,
                linewidths=0.2,
                zorder=5,
            ),
            gid,
        )

    def sites(ax, gdf, color, xcol, ycol, key):
        tri = gdf[gdf["irrigation_class"] == "irrigated"]
        cir = gdf[gdf["irrigation_class"] == "rainfed"]
        halo_pass(ax, tri[xcol], tri[ycol], MS_TRI, "^", f"halo-{key}-irrigated")
        halo_pass(ax, cir[xcol], cir[ycol], MS_CIR, "o", f"halo-{key}-rainfed")
        fill_pass(ax, tri[xcol], tri[ycol], MS_TRI, "^", color, f"marks-{key}-irrigated")
        fill_pass(ax, cir[xcol], cir[ycol], MS_CIR, "o", color, f"marks-{key}-rainfed")
        return len(tri), len(cir)

    # ------------------------------ E1 ------------------------------
    ax_e1 = mapax(E1_FRAME, "map-e1-axes")
    conus_a = F.conus.to_crs(CRS_ALBERS)
    e1_a = F.e1.to_crs(CRS_ALBERS)
    b, sb = conus_a.total_bounds, e1_a.total_bounds
    ext = fit_extent(
        (min(b[0], sb[0]), min(b[1], sb[1]), max(b[2], sb[2]), max(b[3], sb[3])),
        E1_FRAME[2],
        E1_FRAME[3],
        pad_frac=0.03,
    )
    conus_a.plot(ax=ax_e1, facecolor=C_LAND, edgecolor="none", zorder=1)
    if RETAIN_CONUS_STATES:
        F.states.to_crs(CRS_ALBERS).plot(
            ax=ax_e1, facecolor="none", edgecolor=C_CONTEXT, linewidth=0.3, zorder=2
        )
    conus_a.plot(ax=ax_e1, facecolor="none", edgecolor=C_BOUND, linewidth=0.4, zorder=3)
    ax_e1.set_xlim(ext[0], ext[2])
    ax_e1.set_ylim(ext[1], ext[3])
    e1_a = e1_a.assign(_x=e1_a.geometry.x, _y=e1_a.geometry.y)
    n1t, n1c = sites(ax_e1, e1_a, C_E1, "_x", "_y", "e1-sites")
    assert (n1t, n1c) == (39, 21), (n1t, n1c)
    if e1_e3_locator:
        slv_ext = fit_extent(
            F.slv.to_crs(CRS_ALBERS).total_bounds, E3_FRAME[2], E3_FRAME[3], pad_frac=0.05
        )
        loc = mpatches.Rectangle(
            (slv_ext[0], slv_ext[1]),
            slv_ext[2] - slv_ext[0],
            slv_ext[3] - slv_ext[1],
            facecolor="none",
            edgecolor=C_E3,
            linewidth=0.6,
            zorder=6,
        )
        ax_e1.add_patch(loc)
        tag(loc, "locator-e3-in-e1")
        out["e1_e3_locator_epsg5070"] = [round(v, 1) for v in slv_ext]
    maphead(E1_FRAME, "E1 · CONUS", "60 Cropland Sites")

    e0_x = E1_FRAME[0] + text_w_mm("E1 · CONUS", FS_STRUCT, "semibold") + 3.0
    mmtext(ov, e0_x, MAP_HEAD_Y, "E0 · Model-Form Selection", cls="direct_label", pt=FS_ROW)
    assert e0_x + text_w_mm("E0 · Model-Form Selection", FS_ROW) < e2f[0] - 2.0

    # ------------------------------ E2 ------------------------------
    ax_e2 = mapax(e2f, "map-e2-axes")
    k = (2.0 * lat_half) * e2f[2] / (e2f[3] * (lon1 - lon0))
    assert k <= 1.0 + 1e-9, f"the E2 frame would stretch longitude ({k:.3f} > 1)"
    assert F.e2.geometry.y.min() > lat_c - lat_half and F.e2.geometry.y.max() < lat_c + lat_half
    assert F.e2.geometry.x.min() > lon0 and F.e2.geometry.x.max() < lon1
    w2 = F.world.clip(shp_box(lon0, lat_c - lat_half, lon1, lat_c + lat_half)).copy()
    w2 = w2[w2.geometry.notna() & ~w2.geometry.is_empty]
    w2.plot(ax=ax_e2, facecolor=C_LAND, edgecolor=C_BOUND, linewidth=0.25, zorder=1)
    ax_e2.set_xlim(lon0, lon1)
    ax_e2.set_ylim(lat_c - lat_half, lat_c + lat_half)
    e2 = F.e2.assign(_x=F.e2.geometry.x, _y=F.e2.geometry.y)
    n2t, n2c = sites(ax_e2, e2, C_E2, "_x", "_y", "e2-sites")
    assert (n2t, n2c) == (13, 53), (n2t, n2c)
    assert e2["country"].nunique() == 10 and e2["continent"].nunique() == 4
    maphead(e2f, "E2 · 10 Countries", "66 Cropland Sites")

    # ------------------------------ E3 ------------------------------
    ax_e3 = mapax(E3_FRAME, "map-e3-axes")
    slv_a = F.slv.to_crs(CRS_ALBERS)
    e3_a = F.e3.to_crs(CRS_ALBERS)
    ext3 = fit_extent(slv_a.total_bounds, E3_FRAME[2], E3_FRAME[3], pad_frac=0.05)
    assert RETAIN_SLV_BASIN is False, "the HUC8 subdivisions stay out of the E3 map"
    if e3_basemap is not None:
        img = plt.imread(e3_basemap["path"])
        tag(
            ax_e3.imshow(
                img,
                extent=(ext3[0], ext3[2], ext3[1], ext3[3]),
                zorder=1,
                interpolation="bilinear",
            ),
            "raster-e3-basemap",
        )
        wash = float(e3_basemap.get("wash", 0.18))
        ax_e3.add_patch(
            tag(
                mpatches.Rectangle(
                    (ext3[0], ext3[1]),
                    ext3[2] - ext3[0],
                    ext3[3] - ext3[1],
                    facecolor="white",
                    edgecolor="none",
                    alpha=wash,
                    zorder=2,
                ),
                "wash-e3-basemap",
            )
        )
        if e3_hull:
            slv_a.plot(ax=ax_e3, facecolor="none", edgecolor="white", linewidth=1.0, zorder=3)
            slv_a.plot(ax=ax_e3, facecolor="none", edgecolor="#6E6E68", linewidth=0.4, zorder=3.1)
        out["e3_basemap"] = {
            **{k: v for k, v in e3_basemap.items() if k != "path"},
            "file": Path(e3_basemap["path"]).name,
            "wash": wash,
            "extent_epsg5070": [round(v, 1) for v in ext3],
        }
    else:
        slv_a.plot(ax=ax_e3, facecolor=C_LAND, edgecolor="none", zorder=1)
        if e3_hull:
            slv_a.plot(ax=ax_e3, facecolor="none", edgecolor=C_BOUND, linewidth=0.45, zorder=3)
    ax_e3.set_xlim(ext3[0], ext3[2])
    ax_e3.set_ylim(ext3[1], ext3[3])
    halo_pass(ax_e3, e3_a.geometry.x, e3_a.geometry.y, MS_TRI, "^", "halo-e3-fields")
    fill_pass(ax_e3, e3_a.geometry.x, e3_a.geometry.y, MS_TRI, "^", C_E3, "marks-e3-fields")
    maphead(E3_FRAME, "E3 · San Luis Valley", "50 Metered Fields")

    # ------------------- E3 aggregation key, ADJACENT to E3 -------------------
    # Sec. 5.4.2 wording, without the '· E3' suffix now that it sits at E3.
    # Typographic and schematic: no data-like meter mark, and nothing from the
    # US-Bi1 applied-water series reaches it.
    ky = E3_KEY_LINES
    mmtext(ov, E3_KEY_X, ky[0], "Daily Gross Applied Water", cls="direct_label", pt=FS_ROW)
    ind = 2.6
    polyline(
        ov,
        [(E3_KEY_X + 0.4, ky[1] + 0.9), (E3_KEY_X + 2.0, ky[1] + 0.9)],
        C_TEXT,
        0.55,
        head=1.05,
        zorder=7,
        rid="e3-key-aggregation",
    )
    mmtext(ov, E3_KEY_X + ind + 1.2, ky[1], "Annual Total", cls="direct_label", pt=FS_ROW)
    glyphtext(ov, E3_KEY_X + 0.2, ky[2], "—", pt=FS_ROW, gid="glyph-e3-key-relation")
    mmtext(
        ov,
        E3_KEY_X + ind + 1.2,
        ky[2],
        "Metered Water",
        cls="direct_label",
        pt=FS_ROW,
        color=C_HELD,
    )
    kw_max = max(text_w_mm(s, FS_ROW) for s in ("Daily Gross Applied Water",))
    assert E3_KEY_X + kw_max < W_MM - MARGIN_MM + 0.5, (
        f"the E3 key overruns the right margin ({E3_KEY_X + kw_max:.2f} mm)"
    )
    assert E3_KEY_LINES[-1] - 0.6 >= MARGIN_MM - 0.7, "the E3 key drops below the margin"
    meter_w = text_w_mm("Metered Water", FS_ROW)
    out["meter_ink"] = (
        E3_KEY_X + ind + 1.2,
        ky[2] - 0.6,
        E3_KEY_X + ind + 1.2 + meter_w,
        ky[2] + ink_h_mm(FS_ROW),
    )

    # --------------------- class-specific parameter relay ---------------------
    for y, label, marker, rid in (
        (PATH_IRRIG_Y, "Irrigated Parameters", "^", "irrigated-params-to-e2"),
        (PATH_RAINFED_Y, "Rainfed Parameters", "o", "rainfed-params-to-e2"),
    ):
        ov.add_line(Line2D([TOKEN_X0, FORK_X - 1.6], [y, y], color=C_E1, lw=0.7, zorder=5))
        tag(
            ov.add_patch(
                mpatches.RegularPolygon(
                    (FORK_X, y),
                    numVertices=3 if marker == "^" else 20,
                    radius=1.15,
                    orientation=0.0,
                    facecolor=C_E1,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=7,
                )
            ),
            f"token-{_slug(label)}",
        )
        polyline(
            ov, [(FORK_X + 1.6, y), (e2f[0] - 0.5, y)], C_E1, 0.7, head=1.35, zorder=5, rid=rid
        )
        mmtext(
            ov, FORK_X - 1.7, y + 1.4, label, cls="direct_label", pt=FS_ROW, color=C_E1, ha="right"
        )
        assert FORK_X - 1.7 - text_w_mm(label, FS_ROW) > TOKEN_X0 + 0.6, (
            f"the {label!r} token label overruns the E1 frame"
        )

    X0, Y0, Wf, Hf = e2f
    X1, Y1 = X0 + Wf, Y0 + Hf
    out["fork_clearance_mm"] = float(X0 - FORK_X)
    assert out["fork_clearance_mm"] >= FORK_CLEARANCE_MIN, (
        f"the fork stands only {out['fork_clearance_mm']:.2f} mm from the E2 frame"
    )

    if e3_route == "orthogonal":
        # Sec. 6.1: rise vertically from the irrigated branch point, turn 90 deg
        # into a horizontal segment above the E2 map, then a short terminal
        # (drop + entry) with the arrowhead into E3's left spine -- the same
        # landing point the revision-4 spline used.
        y_over = Y1 + 2.5
        x_drop = X1 + 3.0
        land_y = 27.0 + PANEL_DY
        verts = [
            (FORK_X, PATH_IRRIG_Y + 1.6),
            (FORK_X, y_over),
            (x_drop, y_over),
            (x_drop, land_y),
            (E3_FRAME[0] - 0.4, land_y),
        ]
        polyline(ov, verts, C_E1, 0.7, head=1.35, zorder=5, rid="irrigated-params-to-e3")
        out["e3_route_vertices_mm"] = [[round(a, 3), round(b, 3)] for a, b in verts]
        out["e3_route_over_clearance_mm"] = round(y_over - Y1, 3)
        # every vertex outside the E2 frame, with clearance above it
        for vx, vy in verts:
            assert not (X0 < vx < X1 and Y0 < vy < Y1), (
                "the E3 leg enters the E2 frame and could be read as E2 -> E3"
            )
        assert y_over - Y1 >= 1.5, "the E3 leg hugs the E2 frame"
        # the drop stands clear of both map frames
        assert x_drop - X1 >= 2.0 and E3_FRAME[0] - x_drop >= 2.0, (
            "the terminal drop crowds the E2 or E3 frame"
        )
        # the horizontal stays below the E2 count line's ink (descenders reach
        # ~0.65 mm below the baseline at 7.5 pt)
        assert MAP_COUNT_Y - 0.65 - y_over >= 1.0, "the E3 leg crowds the E2 count line"
        assert E3_FRAME[1] + 2.0 < land_y < MAP_TOP - 1.0, (
            "the E3 leg does not land inside the E3 frame's left spine"
        )
    else:
        assert e3_route == "spline", e3_route
        ax = np.array([p[0] for p in E3_ARC_PTS], dtype=float)
        ay = np.array([p[1] for p in E3_ARC_PTS], dtype=float)
        arc_x = np.linspace(ax[0], ax[-1], 320)
        arc_y = PchipInterpolator(ax, ay)(arc_x)
        polyline(
            ov,
            list(zip(arc_x, arc_y, strict=True)),
            C_E1,
            0.7,
            head=1.35,
            zorder=5,
            rid="irrigated-params-to-e3",
        )

        span = (arc_x >= X0) & (arc_x <= X1)
        over = arc_y[span]
        out["arc_clearance_mm"] = float(over.min() - Y1)
        out["arc_relief_mm"] = float(over.max() - over.min())
        out["arc_departure_slope"] = float((arc_y[8] - arc_y[0]) / (arc_x[8] - arc_x[0]))

        assert not np.any((arc_x > X0) & (arc_x < X1) & (arc_y > Y0) & (arc_y < Y1)), (
            "the E3 leg enters the E2 frame and could be read as E2 -> E3"
        )
        assert out["arc_clearance_mm"] >= 1.5, "the E3 leg hugs the E2 frame"
        assert out["arc_relief_mm"] >= 3.0, "the E3 leg crosses E2 at near-constant height"
        assert out["arc_departure_slope"] >= 0.5, "the E3 leg leaves the fork too shallowly"
        assert arc_y[-1] < MAP_TOP - 1.0 and arc_y[-1] > E3_FRAME[1] + 2.0, (
            "the E3 leg does not land inside the E3 frame's left spine"
        )
    return out


# ===========================================================================
# shared audits + export
# ===========================================================================


def audit_scientific(F: Frozen, axis_audit: dict | None, obs_ink: list, cycle: dict | None) -> dict:
    """Every check that Sec. 13 allows a machine to make.

    `axis_audit`   one entry per plotted row: domain, plotted range, headroom.
                   None for the supplementary cycle figure (no plotted rows,
                   no record identification).
    `obs_ink`      ink boxes of held-out observation labels; no arrowhead may
                   land in one.
    `cycle`        {'edges': [(a, b), ...], 'exit': (a, b), 'inputs': [...]}.
                   None for the r6 main figure: the inverse-estimation cycle
                   moved to the supplementary figure (decision 2026-08-26),
                   which runs these checks instead. Exactly one of the two
                   figures must carry the cycle contract.
    """
    m: dict = {}

    # --- type floor and colour policy ---
    pts = [d["font_pt"] for d in DRAWN]
    assert min(pts) >= FS_MIN - 1e-9, f"reader-facing text below {FS_MIN} pt: {min(pts)}"
    m["min_font_pt"] = min(pts)

    scaffolding = {C_AXIS, C_DATUM, C_GUIDE, C_HAIR, C_CONTEXT, C_LAND, C_BOUND}
    stray = [d for d in DRAWN if d["color"] in scaffolding]
    assert not stray, f"scaffolding gray used as text colour: {stray[:3]}"

    # --- forbidden copy ---
    for d in DRAWN:
        for pat in FORBIDDEN_PATTERNS:
            assert not re.search(pat, d["string"], flags=re.I), (
                f"forbidden copy {d['string']!r} matches {pat!r}"
            )
    caption_only = {
        v
        for k, v in F.arch["caption_facts"].items()
        if isinstance(v, str) and not k.startswith("_")
    }
    drawn_strings = {d["string"] for d in DRAWN}
    assert not (drawn_strings & caption_only), "a caption-owned sentence reached the artwork"
    m["drawn_strings"] = len(drawn_strings)

    # --- record identification + plotted rows (the main figure only) ---
    if axis_audit is not None:
        rec = [d for d in DRAWN if d["string"] == F.record_label]
        assert len(rec) == 1, "US-Bi1 (2017) must appear exactly once"
        assert rec[0]["color"] == C_TEXT, "the record identification is not near-black"
        assert "·" not in F.record_label

        # --- every plotted row: domain contains every mark WITH headroom ---
        for rid, a in axis_audit.items():
            lo, hi = a["plotted_range"]
            d0, d1 = a["display_domain"]
            span = d1 - d0
            assert d0 <= lo and hi <= d1, f"{rid}: {lo:.4f}..{hi:.4f} clipped at {d0}..{d1}"
            # Sec. 5.2: "a trace that clears the upper limit by only a line width
            # is not visually adequate even if it passes a numeric no-clip test"
            frac = (d1 - hi) / span
            mm_clear = frac * a["band_mm"]
            assert frac >= 0.05, f"{rid}: only {frac * 100:.1f}% headroom above the data"
            assert mm_clear >= 0.55, f"{rid}: only {mm_clear:.2f} mm of headroom above the data"
            a["headroom_frac"] = round(frac, 4)
            a["headroom_mm"] = round(mm_clear, 3)
        m["rows"] = len(axis_audit)

        # --- the ET comparison shares one date and one y mapping ---
        et = axis_audit["et_comparison"]
        assert et["shares_date_mapping"] is True and et["shares_y_mapping"] is True
        assert et["n_traces"] == 2, "the ET row must carry both traces"

    # --- the closed semantic cycle (the supplementary cycle figure only) ---
    if cycle is not None:
        edges = [tuple(e) for e in cycle["edges"]]
        assert edges == F.cycle_edges, f"the drawn cycle is not the contract cycle: {edges}"
        srcs = [a for a, _ in edges]
        dsts = [b for _, b in edges]
        for s in ("run_balance", "compare", "update_parameters"):
            assert srcs.count(s) == 1 and dsts.count(s) == 1, f"{s} is not on a closed cycle"
        assert tuple(cycle["exit"]) == ("update_parameters", "daily_balance"), (
            "the conditioned-parameter exit is absent or misdirected"
        )
        for e in cycle["inputs"]:
            assert tuple(e) not in F.forbidden_edges, f"forbidden edge drawn: {e}"
        assert ("etf_ensemble", "compare") in {tuple(e) for e in cycle["inputs"]}
        # The architecture forbids a SEPARATE swe edge into Compare: the snow
        # constraint is inline with the ETf constraint, on one route and one label.
        assert ("swe_inline", "compare") in F.forbidden_edges
        assert any("SWE" in d["string"] for d in DRAWN), (
            "the snow constraint is not named on the ETf comparison route"
        )

        # --- held-out observations never enter the cycle, in either direction ---
        all_edges = {tuple(e) for e in cycle["inputs"]} | set(edges) | {tuple(cycle["exit"])}
        for a, b in all_edges:
            assert a not in ("flux_et", "meters"), f"held-out {a!r} originates an edge"
            assert not (
                b in ("run_balance", "compare", "update_parameters") and a in ("flux_et", "meters")
            ), f"held-out {a!r} enters the cycle"
        assert ("applied_water", "meters") not in all_edges
        assert ("daily_et", "flux_et") not in all_edges

    # --- no arrowhead lands in, or aims at, a held-out observation's label ---
    # Direction-aware: an arrow whose tip merely passes near the label but
    # points elsewhere is not a claim about that observation; one that lands in
    # the ink, or whose heading would carry it into the ink, is.
    for rt in ROUTES:
        if not rt["arrow"]:
            continue
        tx, ty = rt["tip"]
        px, py = rt["pts"][-2]
        dx, dy = tx - px, ty - py
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        for x0, y0, x1, y1 in obs_ink:
            assert not (x0 - 1.0 <= tx <= x1 + 1.0 and y0 - 1.0 <= ty <= y1 + 1.0), (
                f"route {rt['id']!r} lands an arrowhead on a held-out observation"
            )
            for s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                ax_, ay_ = tx + dx * s, ty + dy * s
                assert not (x0 - 0.6 <= ax_ <= x1 + 0.6 and y0 - 0.6 <= ay_ <= y1 + 0.6), (
                    f"route {rt['id']!r} aims an arrowhead at a held-out observation"
                )
    m["routes"] = len(ROUTES)
    return m


def export(fig, stem: str, extra: dict, w_mm: float = W_MM, h_mm: float = H_MM) -> dict:
    """Write PDF / SVG / PNG, verify the page and the stable ids, ledger it."""
    pdf_p = HERE / f"{stem}.pdf"
    svg_p = HERE / f"{stem}.svg"
    png_p = HERE / f"{stem}.png"
    fig.savefig(pdf_p, format="pdf", facecolor="white")
    fig.savefig(svg_p, format="svg", facecolor="white")
    fig.savefig(png_p, format="png", dpi=600, facecolor="white")
    plt.close(fig)

    mb = re.search(
        rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", pdf_p.read_bytes()
    )
    assert mb, "no MediaBox in the written PDF"
    pw = (float(mb.group(3)) - float(mb.group(1))) * 25.4 / 72.0
    ph = (float(mb.group(4)) - float(mb.group(2))) * 25.4 / 72.0
    assert abs(pw - w_mm) < 0.02 and abs(ph - h_mm) < 0.02, (pw, ph)

    svg = svg_p.read_text(encoding="utf-8")
    pt = re.search(r'<svg([^>]*?)width="([\d.]+)pt" height="([\d.]+)pt"', svg)
    assert pt, "unexpected SVG root element"
    svg = svg.replace(
        f'width="{pt.group(2)}pt" height="{pt.group(3)}pt"',
        f'width="{w_mm:g}mm" height="{h_mm:g}mm"',
        1,
    )
    svg_p.write_text(svg, encoding="utf-8")
    missing = [g for g in GIDS if f'id="{g}"' not in svg]
    assert not missing, f"stable ids absent from the SVG: {missing[:10]}"
    n_text = svg.count("<text")
    assert n_text >= len(DRAWN), f"{n_text} <text> nodes for {len(DRAWN)} strings"
    assert "Source Sans 3" in svg, "the SVG does not name the typeface"

    led = {
        "study": stem,
        "page_mm": [round(pw, 4), round(ph, 4)],
        "svg_stable_ids": GIDS,
        "svg_text_nodes": n_text,
        "drawn": DRAWN,
        "routes": ROUTES,
        **extra,
    }
    json_p = HERE / f"{stem}_ledger.json"
    json_p.write_text(json.dumps(led, indent=2, default=str), encoding="utf-8")

    sizes = {p.name: round(p.stat().st_size / 1024, 1) for p in (pdf_p, svg_p, png_p, json_p)}
    print(f"[{stem}] page {pw:.4f} x {ph:.4f} mm")
    print(f"[{stem}] strings {len(DRAWN)}  svg text nodes {n_text}  stable ids {len(GIDS)}")
    for k, v in sizes.items():
        print(f"    {k:34s} {v:9.1f} kB")
    return {"page_mm": [pw, ph], "sizes_kb": sizes, "svg_text_nodes": n_text}

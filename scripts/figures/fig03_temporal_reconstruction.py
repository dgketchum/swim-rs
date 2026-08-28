"""Figure 3 -- daily ET agreement and temporal reconstruction (production).

Pooled-agreement composition per paper/notes/fig03_production_handoff.md
(2026-08-27), concept v3, with panel (a) rendered as hexbin density
(FIGURE_STYLE_GUIDE sec. 10). Despite the legacy filename, this script
renders the pooled-agreement design, not the superseded seasonal example.

Reads ONLY the frozen Figure 3 display package under
paper/data/final/figures/ (hash-verified against fig_manifest.json):

- fig03_pooled_daily_agreement.csv  -- panel (a) paired daily values
- fig03_scatter_metrics.csv         -- frozen facet statistics + display strings
- fig03_temporal_site_effects.csv   -- panel (c) paired site shifts + order key
- fig03_temporal_cohort_effects.csv -- panel (b) medians + bootstrap intervals
- fig03_metadata.json               -- construction record

The only transformations applied here are declared presentation steps:
hexbin count binning on a shared log scale (FIGURE_STYLE_GUIDE sec. 10
density threshold) and rasterization of the hexbin layers. All aggregation,
interpolation, metric, and bootstrap arithmetic lives in
scripts/figures/build_figure_data.py.

Usage::

    uv run python scripts/figures/fig03_temporal_reconstruction.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "paper" / "data" / "final" / "figures"
OUTDIR = REPO / "paper" / "figures" / "proofs" / "fig03_pooled_agreement_190"
STEM = "fig03_pooled_agreement"

PAGE_W, PAGE_H = 190.0, 125.0  # mm
RASTER_DPI = 600

# package palette
C_BLUE = "#0072B2"  # SWIM-RS (panel c cohort markers)
C_TEXT = "#000000"
C_CHARCOAL = "#50545A"
C_MID = "#7A7F85"
C_LIGHT = "#C9CDD1"

AX_LO, AX_HI = -2.0, 16.0
AX_TICKS = [0, 4, 8, 12, 16]

SUPPORT_HEADS = {
    "acquisition": ("ETf acquisition dates", "4,751 site-days"),
    "between_acquisitions": ("Between acquisitions", "55,584 site-days"),
}
METHODS = [("openet_et", "OpenET"), ("swim_et", "SWIM-RS")]

# panel (a) hexbin density (FIGURE_STYLE_GUIDE sec. 10: > ~5,000 points per
# panel is hexbin territory) -- one shared log count scale for all facets
HEX_GRIDSIZE = 40
CBAR_TICKS = [1, 10, 100, 1000]
B_ROWS = [
    ("acquisition", "Acquisition"),
    ("between_acquisitions", "Between"),
    ("all_dates", "All dates"),
]
EFFECT_FACETS = [
    ("kge", "ΔKGE", ""),
    ("rmse", "ΔRMSE", "mm d$^{-1}$"),
    ("mbe", "ΔMBE", "mm d$^{-1}$"),
]

# numeric effect axes frozen from the corrected package with a small margin
B_LIMS = {"kge": (-0.10, 0.10), "rmse": (-0.12, 0.15), "mbe": (-0.05, 0.45)}
B_TICKS = {"kge": [-0.1, 0.0, 0.1], "rmse": [0.0, 0.1], "mbe": [0.0, 0.2, 0.4]}
C_LIMS = {"kge": (-0.55, 0.42), "rmse": (-0.50, 1.45), "mbe": (-0.72, 2.45)}
C_TICKS = {"kge": [-0.4, 0.0, 0.4], "rmse": [0.0, 0.7, 1.4], "mbe": [0.0, 1.0, 2.0]}

# Section 5.3 frozen display strings, re-asserted against the package
EXPECT_DISPLAY = {
    ("OpenET", "acquisition"): ("0.90", "−0.32", "1.10"),
    ("OpenET", "between_acquisitions"): ("0.87", "−0.24", "1.13"),
    ("SWIM-RS", "acquisition"): ("0.88", "−0.05", "1.19"),
    ("SWIM-RS", "between_acquisitions"): ("0.87", "−0.01", "1.10"),
}

FORBIDDEN_STRINGS = [
    "run22",
    "non-overpass",
    "non_overpass",
    "no-satellite",
    "gap-filled",
    "NSE",
    "R²",
    "MAE",
    "|MBE|",
    "p =",
    "p<",
]

MIN_PT = 6.5  # Elsevier floor is 6 pt; hold 6.5 as the working floor


class ProofError(RuntimeError):
    pass


# ---------------------------------------------------------------- fonts


def register_fonts() -> None:
    # Arial, the guide-named family (FIGURE_STYLE_GUIDE.md section 3);
    # Microsoft core-fonts faces installed under ~/.fonts/arial
    candidates = [
        Path.home() / ".fonts" / "arial",
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    ]
    for d in candidates:
        if d.exists():
            for p in sorted(d.glob("[Aa]rial*.[TtOo][Tt][Ff]")):
                fm.fontManager.addfont(str(p))
    names = {f.name for f in fm.fontManager.ttflist}
    if "Arial" not in names:
        raise ProofError("Arial is not registered; no fallback is allowed")
    style = Path.home() / "code" / "style" / "journal_figures.mplstyle"
    if style.exists():
        plt.style.use(str(style))
    plt.rcParams.update(
        {
            "savefig.bbox": "standard",  # the mm layout is absolute; never tight-crop
            "font.family": "Arial",
            "font.size": 7.2,
            "text.color": C_TEXT,
            "axes.edgecolor": C_CHARCOAL,
            "axes.labelcolor": C_TEXT,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# ---------------------------------------------------------------- data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_package() -> dict:
    manifest = json.loads((PKG / "fig_manifest.json").read_text())["tables"]
    files = [
        "fig03_pooled_daily_agreement.csv",
        "fig03_scatter_metrics.csv",
        "fig03_temporal_site_effects.csv",
        "fig03_temporal_cohort_effects.csv",
        "fig03_metadata.json",
    ]
    for name in files:
        if name not in manifest:
            raise ProofError(f"{name} missing from fig_manifest.json")
        got = sha256(PKG / name)
        want = manifest[name]["output_sha256"]
        if got != want:
            raise ProofError(f"{name}: sha256 {got[:12]} != manifest {want[:12]}")

    pooled = pd.read_csv(PKG / "fig03_pooled_daily_agreement.csv")
    scatter = pd.read_csv(
        PKG / "fig03_scatter_metrics.csv",
        dtype={"display_r": str, "display_bias": str, "display_rmse": str},
    )
    effects = pd.read_csv(PKG / "fig03_temporal_site_effects.csv")
    cohort = pd.read_csv(PKG / "fig03_temporal_cohort_effects.csv")
    meta = json.loads((PKG / "fig03_metadata.json").read_text())

    # cohort/count gates
    if len(pooled) != 60335 or pooled["site_id"].nunique() != 43:
        raise ProofError("pooled table does not carry 60,335 rows over 43 sites")
    n_acq = int((pooled["temporal_support"] == "acquisition").sum())
    n_btw = int((pooled["temporal_support"] == "between_acquisitions").sum())
    if (n_acq, n_btw) != (4751, 55584):
        raise ProofError(f"temporal-support counts {n_acq}/{n_btw} do not reconcile")
    if pooled.duplicated(["site_id", "date"]).any():
        raise ProofError("duplicate site_id/date keys in the pooled table")
    if len(effects) != 129 or len(cohort) != 9:
        raise ProofError("effect table grain drifted from 129/9 rows")
    if not ((cohort["seed"] == 42).all() and (cohort["n_resamples"] == 10000).all()):
        raise ProofError("cohort effects do not carry the frozen bootstrap settings")

    # range gate
    vals = pooled[["flux_et", "swim_et", "openet_et"]].values
    if vals.min() < AX_LO or vals.max() > AX_HI:
        raise ProofError("a plotted value falls outside the fixed -2..16 axes")

    # metric gate: recompute the facet statistics from the exact plotted rows
    for _, row in scatter.iterrows():
        sub = pooled[pooled["temporal_support"] == row["temporal_support"]]
        col = "openet_et" if row["method"] == "OpenET" else "swim_et"
        resid = sub[col].to_numpy() - sub["flux_et"].to_numpy()
        r = float(np.corrcoef(sub["flux_et"], sub[col])[0, 1])
        bias = float(resid.mean())
        rmse = float(np.sqrt(np.mean(resid**2)))
        if (
            abs(r - row["pearson_r"]) > 1e-12
            or abs(bias - row["bias"]) > 1e-12
            or abs(rmse - row["rmse"]) > 1e-12
        ):
            raise ProofError(f"scatter stats fail to reproduce for {row['method']}")
        want = EXPECT_DISPLAY[(row["method"], row["temporal_support"])]
        if (row["display_r"], row["display_bias"], row["display_rmse"]) != want:
            raise ProofError(f"display strings drifted for {row['method']}")

    # panel (b) medians reproduce from the site effects
    for _, row in cohort.iterrows():
        sub = effects[effects["temporal_support"] == row["temporal_support"]]
        med = float(sub[f"d_{row['metric']}"].median())
        if abs(med - row["median_delta"]) > 1e-12:
            raise ProofError(f"cohort median fails to reproduce for {row['metric']}")
        if not (row["ci95_lo"] <= row["median_delta"] <= row["ci95_hi"]):
            raise ProofError("a bootstrap interval does not contain its median")

    # panel (c) ordering key
    order = effects.drop_duplicates("site_id")[["site_id", "site_order_between_kge"]]
    if sorted(order["site_order_between_kge"]) != list(range(1, 44)):
        raise ProofError("site_order_between_kge is not a 1..43 permutation")

    # effect axes cover every mark and interval without clipping
    for metric, (lo, hi) in B_LIMS.items():
        s = cohort[cohort["metric"] == metric]
        if s["ci95_lo"].min() < lo or s["ci95_hi"].max() > hi:
            raise ProofError(f"panel (b) {metric} interval clipped by frozen limits")
    for metric, (lo, hi) in C_LIMS.items():
        s = effects[effects["temporal_support"] != "all_dates"][f"d_{metric}"]
        if s.min() < lo or s.max() > hi:
            raise ProofError(f"panel (c) {metric} mark clipped by frozen limits")

    return {
        "pooled": pooled,
        "scatter": scatter,
        "effects": effects,
        "cohort": cohort,
        "meta": meta,
        "hashes": {n: sha256(PKG / n) for n in files},
    }


# ---------------------------------------------------------------- layout


def ax_mm(fig, x0: float, y0: float, w: float, h: float):
    return fig.add_axes([x0 / PAGE_W, y0 / PAGE_H, w / PAGE_W, h / PAGE_H])


def fig_text(fig, x_mm: float, y_mm: float, s: str, **kw):
    return fig.text(x_mm / PAGE_W, y_mm / PAGE_H, s, **kw)


# panel (a) geometry
A_S = 45.0  # square facet side, mm
A_X = [14.5, 63.0]
A_Y = [61.5, 13.0]  # row 0 = OpenET (top), row 1 = SWIM-RS (bottom)
A_HEAD_Y = 108.0

# right column
R_X0, R_X1 = 118.0, 187.0
B_FACET_W, B_FACET_GAP = 17.6, 2.4
B_LABEL_X = 129.0
B_FACET_X = [130.5, 150.5, 170.5]
B_Y0, B_H = 90.5, 21.5
C_FACET_W = 21.4
# facets start 1 mm right of panel (b)'s column so the panel (a) colorbar
# tick labels clear their frames (facet row ends at 188.2, matching b's 188.1)
C_FACET_X = [119.0, 142.9, 166.8]
C_Y0, C_H = 11.5, 59.5

TITLE_Y = 120.6


def strip_axis(ax, lo, hi, ticks, n_rows):
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.axvline(0, color=C_MID, lw=0.5, zorder=1)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}".replace("-", "−") for t in ticks], fontsize=6.5)
    ax.tick_params(axis="x", length=2.2, width=0.55, pad=1.2)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(C_CHARCOAL)
        side.set_linewidth(0.55)
    ax.set_facecolor("white")


def draw_panel_a(fig, pooled, scatter):
    disp = scatter.set_index(["method", "temporal_support"])
    hexes = []
    for irow, (col, method) in enumerate(METHODS):
        for icol, support in enumerate(["acquisition", "between_acquisitions"]):
            ax = ax_mm(fig, A_X[icol], A_Y[irow], A_S, A_S)
            sub = pooled[pooled["temporal_support"] == support]
            x = sub["flux_et"].to_numpy()
            y = sub[col].to_numpy()
            ax.set_xlim(AX_LO, AX_HI)
            ax.set_ylim(AX_LO, AX_HI)
            ax.set_aspect("equal", adjustable="box")
            hb = ax.hexbin(
                x,
                y,
                gridsize=HEX_GRIDSIZE,
                extent=(AX_LO, AX_HI, AX_LO, AX_HI),
                cmap="viridis",
                mincnt=1,
                linewidths=0.0,
                rasterized=True,
                zorder=2,
            )
            hexes.append(hb)
            # 1:1 guide drawn over the density so it stays readable
            ax.plot(
                [AX_LO, AX_HI],
                [AX_LO, AX_HI],
                color=C_CHARCOAL,
                lw=0.7,
                ls=(0, (4, 2)),
                zorder=3,
            )
            ax.set_xticks(AX_TICKS)
            ax.set_yticks(AX_TICKS)
            ax.tick_params(axis="both", labelsize=6.5, length=2.2, width=0.55, pad=1.4)
            if icol != 0:
                ax.tick_params(axis="y", labelleft=False)
            if irow != 1:
                ax.tick_params(axis="x", labelbottom=False)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(C_CHARCOAL)
                side.set_linewidth(0.55)
            # plain-weight facet identifier (FIGURE_STYLE_GUIDE §4/§8: identity
            # is never carried by colored text; each facet holds one series)
            ax.text(
                0.045,
                0.965,
                method,
                transform=ax.transAxes,
                fontsize=7.0,
                color=C_TEXT,
                ha="left",
                va="top",
                zorder=5,
            )
            row = disp.loc[(method, support)]
            ax.text(
                0.045,
                0.885,
                (
                    f"$r$ = {row['display_r']}\n"
                    f"Bias = {row['display_bias']}\n"
                    f"RMSE = {row['display_rmse']}"
                ),
                transform=ax.transAxes,
                fontsize=6.5,
                color=C_TEXT,
                linespacing=1.15,
                ha="left",
                va="top",
                zorder=5,
            )
    for icol, support in enumerate(["acquisition", "between_acquisitions"]):
        head, count = SUPPORT_HEADS[support]
        cx = A_X[icol] + A_S / 2
        fig_text(
            fig,
            cx,
            A_HEAD_Y + 3.1,
            head,
            fontsize=7.0,
            fontweight="semibold",
            ha="center",
            va="bottom",
        )
        fig_text(fig, cx, A_HEAD_Y, count, fontsize=6.5, color=C_TEXT, ha="center", va="bottom")
    fig_text(
        fig,
        A_X[0] + A_S + 1.75,
        6.0,
        "Flux ET (mm d$^{-1}$)",
        fontsize=7.0,
        ha="center",
        va="bottom",
    )
    fig_text(
        fig,
        6.0,
        A_Y[1] + A_S + 1.75,
        "Estimated ET (mm d$^{-1}$)",
        fontsize=7.0,
        ha="center",
        va="center",
        rotation=90,
    )
    # One shared log count scale across all four facets, keyed by one thin
    # vertical colorbar in the dead strip between panels (a) and (b)/(c).
    gmax = max(int(hb.get_array().max()) for hb in hexes)
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=gmax)
    for hb in hexes:
        hb.set_norm(norm)
    # the bar hugs panel (a); tick labels fill the strip to the right, ending
    # short of the panel (c) facets (x = 119); the title clears the facet
    # tops (y = 71) and panel (a)'s right column (x = 108)
    cax = ax_mm(fig, 108.8, 30.0, 2.6, 40.0)
    cb = fig.colorbar(hexes[0], cax=cax)
    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor(C_CHARCOAL)
    ticks = [t for t in CBAR_TICKS if t <= gmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:,}" for t in ticks])
    cb.minorticks_off()
    cax.tick_params(labelsize=6.5, length=2.2, width=0.55, pad=1.2)
    fig_text(fig, 108.8, 71.8, "Site-days", fontsize=7.0, ha="left", va="bottom")


def facet_heading(fig, x_center: float, y_top: float, head: str, unit: str):
    """Two-line facet heading: metric name plus optional 6.5 pt unit line."""
    fig_text(
        fig, x_center, y_top, head, fontsize=7.0, fontweight="semibold", ha="center", va="bottom"
    )
    if unit:
        # 3.4 mm keeps the mathtext superscript clear of the head above
        fig_text(fig, x_center, y_top - 3.4, f"({unit})", fontsize=6.5, ha="center", va="bottom")


def draw_panel_b(fig, cohort):
    n = len(B_ROWS)
    ypos = {sup: n - 1 - i for i, (sup, _) in enumerate(B_ROWS)}
    for j, (metric, head, unit) in enumerate(EFFECT_FACETS):
        ax = ax_mm(fig, B_FACET_X[j], B_Y0, B_FACET_W, B_H)
        lo, hi = B_LIMS[metric]
        strip_axis(ax, lo, hi, B_TICKS[metric], n)
        for sup, _ in B_ROWS:
            row = cohort[(cohort["metric"] == metric) & (cohort["temporal_support"] == sup)].iloc[0]
            y = ypos[sup]
            ax.hlines(y, row["ci95_lo"], row["ci95_hi"], color=C_CHARCOAL, lw=1.15, zorder=2)
            ax.plot(
                row["median_delta"],
                y,
                marker="o",
                ms=3.1,
                mfc=C_BLUE,
                mec=C_BLUE,
                mew=0,
                zorder=3,
            )
        facet_heading(fig, B_FACET_X[j] + B_FACET_W / 2, B_Y0 + B_H + 4.9, head, unit)
    for sup, label in B_ROWS:
        fig_text(
            fig,
            B_LABEL_X,
            B_Y0 + (ypos[sup] + 0.6) / (len(B_ROWS) + 0.2) * B_H,
            label,
            fontsize=6.5,
            ha="right",
            va="center",
        )


def draw_panel_c(fig, effects):
    eff = effects.set_index(["site_id", "temporal_support"])
    order = (
        effects.drop_duplicates("site_id").sort_values("site_order_between_kge")["site_id"].tolist()
    )
    n = len(order)
    for j, (metric, head, unit) in enumerate(EFFECT_FACETS):
        ax = ax_mm(fig, C_FACET_X[j], C_Y0, C_FACET_W, C_H)
        lo, hi = C_LIMS[metric]
        strip_axis(ax, lo, hi, C_TICKS[metric], n)
        for yi, fid in enumerate(order):
            a = float(eff.loc[(fid, "acquisition"), f"d_{metric}"])
            b = float(eff.loc[(fid, "between_acquisitions"), f"d_{metric}"])
            ax.plot([a, b], [yi, yi], color=C_LIGHT, lw=0.45, zorder=2)
            ax.plot(a, yi, marker="o", ms=1.9, mfc="white", mec=C_MID, mew=0.5, ls="none", zorder=3)
            ax.plot(b, yi, marker="D", ms=1.7, mfc=C_MID, mec=C_MID, mew=0.3, ls="none", zorder=4)
        facet_heading(fig, C_FACET_X[j] + C_FACET_W / 2, C_Y0 + C_H + 4.3, head, unit)
    # frameless two-entry marker key below the facets (FIGURE_STYLE_GUIDE §8:
    # every distinguishing symbol is defined inside the figure)
    kx0 = C_FACET_X[0]
    kw = C_FACET_X[2] + C_FACET_W - kx0
    axk = ax_mm(fig, kx0, 3.6, kw, 3.0)
    axk.set_xlim(0, kw)
    axk.set_ylim(0, 1)
    axk.set_axis_off()
    entries = [
        ("o", dict(mfc="white", mec=C_MID, mew=0.5, ms=1.9), "Acquisition dates"),
        ("D", dict(mfc=C_MID, mec=C_MID, mew=0.3, ms=1.7), "Between acquisitions"),
    ]
    xk = [kw * 0.16, kw * 0.52]
    for (marker, mkw, label), x in zip(entries, xk, strict=True):
        axk.plot([x], [0.5], marker=marker, ls="none", **mkw)
        axk.text(x + 1.7, 0.5, label, fontsize=6.5, color=C_TEXT, ha="left", va="center")


def add_panel_title(fig, x_mm: float, y_mm: float, label: str):
    """Elsevier panel label: plain-weight '(a)' fused with a sentence-case
    identifier, one text object on one baseline (FIGURE_STYLE_GUIDE §4-5)."""
    return fig_text(fig, x_mm, y_mm, label, fontsize=7.0, ha="left", va="bottom")


# ---------------------------------------------------------------- audit


def audit(fig) -> list[dict]:
    items = []
    for artist in fig.findobj(matplotlib.text.Text):
        s = artist.get_text().strip()
        if not s:
            continue
        size = float(artist.get_fontsize())
        items.append({"text": s, "fontsize_pt": round(size, 2)})
        if size < MIN_PT - 1e-6:
            raise ProofError(f"text below {MIN_PT}pt: {s!r} at {size}pt")
        for bad in FORBIDDEN_STRINGS:
            if bad.lower() in s.lower():
                raise ProofError(f"forbidden string {bad!r} rendered: {s!r}")
    return items


# ---------------------------------------------------------------- main


def main() -> None:
    register_fonts()
    pkg = load_package()

    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=300, facecolor="white")
    draw_panel_a(fig, pkg["pooled"], pkg["scatter"])
    draw_panel_b(fig, pkg["cohort"])
    draw_panel_c(fig, pkg["effects"])
    add_panel_title(fig, 4.5, TITLE_Y, "(a) Pooled daily ET agreement")
    add_panel_title(fig, R_X0 - 3.5, TITLE_Y, "(b) Cohort effects of temporal support")
    add_panel_title(fig, R_X0 - 3.5, C_Y0 + C_H + 8.0, "(c) Site-level effects")

    items = audit(fig)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUTDIR / f"{STEM}.{ext}", dpi=RASTER_DPI, facecolor="white")
    fig.savefig(OUTDIR / f"{STEM}.png", dpi=RASTER_DPI, facecolor="white")
    plt.close(fig)

    with open(OUTDIR / "fig03_pooled_agreement_textaudit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["text", "fontsize_pt"])
        w.writeheader()
        w.writerows(items)

    shutil.copy2(__file__, OUTDIR / Path(__file__).name)

    meta = {
        "figure": "Figure 3 -- daily ET agreement and temporal reconstruction",
        "contract": "paper/notes/fig03_production_handoff.md (2026-08-27)",
        "style": (
            "~/code/style/FIGURE_STYLE_GUIDE.md + journal_figures.mplstyle; "
            "Arial family (the guide-named face)"
        ),
        "composition_id": pkg["meta"]["composition_id"],
        "canvas_mm": [PAGE_W, PAGE_H],
        "raster_dpi": RASTER_DPI,
        "panel_a": {
            "axes_mm_day": [AX_LO, AX_HI],
            "ticks": AX_TICKS,
            "encoding": "hexbin density, viridis, shared log count scale",
            "hex_gridsize": HEX_GRIDSIZE,
            "rasterized": "hexbin layers only; axes, text, and 1:1 lines are vector",
        },
        "panel_b_limits": B_LIMS,
        "panel_c_limits": C_LIMS,
        "counts": {
            "sites": 43,
            "acquisition_site_days": 4751,
            "between_acquisition_site_days": 55584,
            "total_site_days": 60335,
        },
        "package_hashes": pkg["hashes"],
        "text_items_audited": len(items),
        "rendered_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "scripts/figures/fig03_temporal_reconstruction.py",
    }
    (OUTDIR / "fig03_pooled_agreement_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"rendered {STEM}.pdf/svg/png to {OUTDIR}")
    print(f"text items audited: {len(items)} (all >= {MIN_PT}pt)")


if __name__ == "__main__":
    main()

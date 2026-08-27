"""Figure 3 -- daily ET agreement and temporal reconstruction (production).

Scatter-first composition per paper/notes/fig03_production_handoff.md
(2026-08-27), concept v3. Despite the legacy filename, this script renders
the pooled-agreement design, not the superseded seasonal example.

Reads ONLY the frozen Figure 3 display package under
paper/data/final/figures/ (hash-verified against fig_manifest.json):

- fig03_pooled_daily_agreement.csv  -- panel (a) point clouds
- fig03_scatter_metrics.csv         -- frozen facet statistics + display strings
- fig03_temporal_site_effects.csv   -- panel (c) paired site shifts + order key
- fig03_temporal_cohort_effects.csv -- panel (b) medians + bootstrap intervals
- fig03_metadata.json               -- construction record

The only transformations applied here are declared presentation steps:
deterministic point draw-order shuffle (seed frozen in the metadata) and
rasterization of the point-cloud layers. All aggregation, interpolation,
metric, and bootstrap arithmetic lives in scripts/figures/build_figure_data.py.

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
C_BLUE = "#0072B2"  # SWIM-RS
C_ORANGE = "#E69F00"  # OpenET
C_TEXT = "#202124"
C_CHARCOAL = "#50545A"
C_MID = "#7A7F85"
C_LIGHT = "#C9CDD1"
C_GRID = "#E5E7E9"

AX_LO, AX_HI = -2.0, 16.0
AX_TICKS = [0, 4, 8, 12, 16]

SUPPORT_HEADS = {
    "acquisition": ("ETf Acquisition Dates", "4,751 site-days"),
    "between_acquisitions": ("Between Acquisitions", "55,584 site-days"),
}
METHODS = [("openet_et", "OpenET", C_ORANGE), ("swim_et", "SWIM-RS", C_BLUE)]
B_ROWS = [
    ("acquisition", "Acquisition"),
    ("between_acquisitions", "Between"),
    ("all_dates", "All Dates"),
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

MIN_PT = 7.0


class ProofError(RuntimeError):
    pass


# ---------------------------------------------------------------- fonts


def register_fonts() -> None:
    candidates = [
        Path.home() / ".fonts" / "source-sans",
        Path("/usr/share/fonts/opentype/source-sans"),
        Path("/usr/share/fonts/truetype/source-sans-pro"),
    ]
    for d in candidates:
        if d.exists():
            for p in sorted(d.glob("*.[ot]tf")):
                fm.fontManager.addfont(str(p))
    names = {f.name for f in fm.fontManager.ttflist}
    if "Source Sans 3" not in names:
        raise ProofError("Source Sans 3 is not registered; no fallback is allowed")
    plt.rcParams.update(
        {
            "font.family": "Source Sans 3",
            "font.size": 7.2,
            "text.color": C_TEXT,
            "axes.edgecolor": C_CHARCOAL,
            "axes.labelcolor": C_TEXT,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Source Sans 3",
            "mathtext.it": "Source Sans 3:italic",
            "mathtext.bf": "Source Sans 3:bold",
            "mathtext.cal": "Source Sans 3",
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
C_FACET_X = [118.0, 141.9, 165.8]
C_Y0, C_H = 15.0, 56.0

TITLE_Y = 120.6


def strip_axis(ax, lo, hi, ticks, n_rows):
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.axvline(0, color=C_CHARCOAL, lw=0.6, zorder=1)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7.0)
    ax.tick_params(axis="x", length=1.6, width=0.5, pad=1.2)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_CHARCOAL)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.set_facecolor("white")


def draw_panel_a(fig, pooled, scatter, draw_seed: int):
    rng = np.random.default_rng(draw_seed)
    disp = scatter.set_index(["method", "temporal_support"])
    for irow, (col, method, color) in enumerate(METHODS):
        for icol, support in enumerate(["acquisition", "between_acquisitions"]):
            ax = ax_mm(fig, A_X[icol], A_Y[irow], A_S, A_S)
            sub = pooled[pooled["temporal_support"] == support]
            x = sub["flux_et"].to_numpy()
            y = sub[col].to_numpy()
            order = rng.permutation(len(sub))
            size = 2.3 if support == "acquisition" else 1.15
            alpha = 0.16 if support == "acquisition" else 0.045
            ax.set_xlim(AX_LO, AX_HI)
            ax.set_ylim(AX_LO, AX_HI)
            ax.set_aspect("equal", adjustable="box")
            for t in AX_TICKS:
                ax.axvline(t, color=C_GRID, lw=0.4, zorder=0)
                ax.axhline(t, color=C_GRID, lw=0.4, zorder=0)
            ax.plot([AX_LO, AX_HI], [AX_LO, AX_HI], color=C_CHARCOAL, lw=0.7, zorder=1)
            ax.scatter(
                x[order],
                y[order],
                s=size,
                color=color,
                alpha=alpha,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            ax.set_xticks(AX_TICKS)
            ax.set_yticks(AX_TICKS)
            ax.tick_params(axis="both", labelsize=7.0, length=1.8, width=0.5, pad=1.4)
            if icol != 0:
                ax.tick_params(axis="y", labelleft=False)
            if irow != 1:
                ax.tick_params(axis="x", labelbottom=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(C_CHARCOAL)
                ax.spines[side].set_linewidth(0.55)
            ax.text(
                0.045,
                0.955,
                method,
                transform=ax.transAxes,
                fontsize=7.6,
                fontweight="semibold",
                color=color,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
                zorder=5,
            )
            row = disp.loc[(method, support)]
            ax.text(
                0.965,
                0.955,
                (
                    f"r = {row['display_r']}\n"
                    f"Bias = {row['display_bias']}\n"
                    f"RMSE = {row['display_rmse']}"
                ),
                transform=ax.transAxes,
                fontsize=7.0,
                color=C_TEXT,
                linespacing=1.15,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.9),
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
            fontsize=8.2,
            fontweight="semibold",
            ha="center",
            va="bottom",
        )
        fig_text(fig, cx, A_HEAD_Y, count, fontsize=7.0, color=C_TEXT, ha="center", va="bottom")
    fig_text(
        fig,
        A_X[0] + A_S + 1.75,
        6.0,
        "Flux ET (mm d$^{-1}$)",
        fontsize=8.0,
        ha="center",
        va="bottom",
    )
    fig_text(
        fig,
        6.0,
        A_Y[1] + A_S + 1.75,
        "Estimated ET (mm d$^{-1}$)",
        fontsize=8.0,
        ha="center",
        va="center",
        rotation=90,
    )


def facet_heading(fig, x_center: float, y_top: float, head: str, unit: str):
    """Two-line facet heading: metric name plus optional 7 pt unit line."""
    fig_text(
        fig, x_center, y_top, head, fontsize=8.0, fontweight="semibold", ha="center", va="bottom"
    )
    if unit:
        fig_text(fig, x_center, y_top - 2.7, f"({unit})", fontsize=7.0, ha="center", va="bottom")


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
        facet_heading(fig, B_FACET_X[j] + B_FACET_W / 2, B_Y0 + B_H + 4.2, head, unit)
    for sup, label in B_ROWS:
        fig_text(
            fig,
            B_LABEL_X,
            B_Y0 + (ypos[sup] + 0.6) / (len(B_ROWS) + 0.2) * B_H,
            label,
            fontsize=7.2,
            ha="right",
            va="center",
        )
    fig_text(
        fig,
        (B_FACET_X[0] + B_FACET_X[-1] + B_FACET_W) / 2,
        B_Y0 - 4.6,
        "Δ = SWIM-RS − OpenET; whole-site bootstrap 95% intervals",
        fontsize=7.0,
        color=C_TEXT,
        ha="center",
        va="top",
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
        facet_heading(fig, C_FACET_X[j] + C_FACET_W / 2, C_Y0 + C_H + 3.6, head, unit)
    # shared two-item legend
    lx = (C_FACET_X[0] + C_FACET_X[-1] + C_FACET_W) / 2
    ly = C_Y0 - 7.6
    fig_text(fig, lx - 24.0, ly, "○", fontsize=7.2, color=C_MID, ha="center", va="center")
    fig_text(fig, lx - 22.0, ly, "Acquisition dates", fontsize=7.2, ha="left", va="center")
    fig_text(fig, lx + 4.0, ly, "◆", fontsize=7.2, color=C_MID, ha="center", va="center")
    fig_text(fig, lx + 6.0, ly, "Between acquisitions", fontsize=7.2, ha="left", va="center")


def add_panel_title(fig, x_mm: float, label: str, title: str):
    t = fig_text(
        fig, x_mm, TITLE_Y, label, fontsize=10.5, fontweight="bold", ha="left", va="bottom"
    )
    fig_text(
        fig, x_mm + 7.2, TITLE_Y, title, fontsize=8.8, fontweight="semibold", ha="left", va="bottom"
    )
    return t


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
    draw_seed = int(pkg["meta"]["panel_a"]["draw_order_seed"])

    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=300, facecolor="white")
    draw_panel_a(fig, pkg["pooled"], pkg["scatter"], draw_seed)
    draw_panel_b(fig, pkg["cohort"])
    draw_panel_c(fig, pkg["effects"])
    add_panel_title(fig, 4.5, "(a)", "Pooled Daily ET Agreement (43 Sites)")
    add_panel_title(fig, R_X0 - 3.5, "(b)", "Accuracy Effects by Temporal Support")
    fig_text(
        fig,
        R_X0 - 3.5,
        C_Y0 + C_H + 8.0,
        "(c)",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig_text(
        fig,
        R_X0 + 3.7,
        C_Y0 + C_H + 8.0,
        "Site-Level Temporal Contrast",
        fontsize=8.8,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )

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
        "composition_id": pkg["meta"]["composition_id"],
        "canvas_mm": [PAGE_W, PAGE_H],
        "raster_dpi": RASTER_DPI,
        "panel_a": {
            "axes_mm_day": [AX_LO, AX_HI],
            "ticks": AX_TICKS,
            "point_area_pt2": {"acquisition": 2.3, "between_acquisitions": 1.15},
            "alpha": {"acquisition": 0.16, "between_acquisitions": 0.045},
            "draw_order_seed": draw_seed,
            "rasterized": "point clouds only; axes, text, and 1:1 lines are vector",
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

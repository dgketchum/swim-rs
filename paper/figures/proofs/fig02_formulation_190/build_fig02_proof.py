"""Figure 2 proof -- cover scaling makes the vegetation formulation coherent.

Final-size (190 x 125 mm) render of the redesigned Figure 2
(paper/notes/six_figure_plan.md section 6, 2026-08-27) from the frozen E0
display package under paper/data/final/figures/:

    fig02_formulation_response.csv   panel (a) fitted K_T distributions
    fig02_ndvi_support.csv           panel (a) site-equal NDVI support
    fig02_pooled_metrics.csv         panel (b) Table 3 pooled values
    fig02_site_rmse_effects.csv      panel (c) paired site RMSE effects
    fig02_metadata.json              labels, rules, provenance

Reads nothing else.  Follows the shared visual system (plan section 3) as
restyled to ~/code/style/FIGURE_STYLE_GUIDE.md (2026-08-27 revision):
Arial, plain-weight fused sentence-case panel labels, 8 pt semibold
facet headings, 8 pt axis titles, 7-7.5 pt ticks/legends, SWIM blue
#0072B2 = cover-scaled sigmoid, black square = unscaled linear,
vermillion triangle = unscaled sigmoid; identities survive grayscale via
marker shape and line pattern.  Machine checks assert the frozen-package
hashes, the manuscript-precision pooled values, the 43/45 and 27/31
isolated-cover win counts, the absence of internal run labels, and the
minimum type size before anything is exported.

Usage::

    uv run python paper/figures/proofs/fig02_formulation_190/build_fig02_proof.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PKG = REPO / "paper" / "data" / "final" / "figures"
STEM = "fig02_formulation"

PAGE_W, PAGE_H = 190.0, 125.0
MM = 1.0 / 25.4

# ---------------------------------------------------------------------------
# shared visual system
# ---------------------------------------------------------------------------

C_TEXT = "#000000"
C_RULE = "#B9BDC4"
C_SUPPORT = "#C7CCD4"

FORMS = ["cover_scaled_sigmoid", "unscaled_linear", "unscaled_sigmoid"]
# Okabe-Ito blue for the retained form, black + vermillion for the two
# rejected arms (guide sec. 7: data is saturated or black, never grey;
# orange #E69F00 stays reserved for OpenET in Figs 1 and 3).
STYLE = {
    "cover_scaled_sigmoid": dict(color="#0072B2", marker="o", ls="-"),
    "unscaled_linear": dict(color="#000000", marker="s", ls=(0, (4, 2))),
    "unscaled_sigmoid": dict(color="#D55E00", marker="^", ls=(0, (1.4, 1.6))),
}

# guide sec. 3 (2026-08-27 16:27 revision) cross-venue working ladder:
# titles and axis labels 7 pt, ticks/legend/annotations 6-7 pt
FS_PANEL = 7.0  # fused panel labels "(a) ...", plain weight, sentence case
FS_HEAD = 7.0
FS_AXIS = 7.0
FS_TICK = 6.5
FS_ANNO = 6.5
# Suite-standard interior framed key (figures_update_31082026.md section 2,
# journal_figures.mplstyle): 6 pt text so borderaxespad 0.5 yields the shared
# 3 pt (1.06 mm) frame-to-spine inset used by Figures 4-6.
FS_LEGEND = 6.0
FS_MIN = 6.0  # Elsevier floor; only the interior key sits below 6.5

FORBIDDEN_STRINGS = ("run22", "RunFAO56", "fao56_sig", "NSE", "|MBE|")

FONT_DIRS = [
    Path.home() / ".fonts" / "arial",
    Path("/usr/share/fonts/truetype/msttcorefonts"),
]


def register_typeface() -> None:
    # Arial, the guide-named family (FIGURE_STYLE_GUIDE.md section 3)
    for d in FONT_DIRS:
        if d.is_dir():
            for f in sorted(d.glob("[Aa]rial*.[TtOo][Tt][Ff]")):
                fm.fontManager.addfont(str(f))
    names = {e.name for e in fm.fontManager.ttflist}
    assert "Arial" in names, "Arial did not register; refusing fallback"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.cal": "Arial",
            # Guide section 8, "One inset distance, everywhere": interior
            # legends take their axes offset from this single shared value
            # (journal_figures.mplstyle), never a per-panel bbox_to_anchor
            # or borderaxespad override.
            "legend.borderaxespad": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": STEM,
            "text.color": C_TEXT,
            "axes.edgecolor": "#50545A",
            "axes.labelcolor": C_TEXT,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "axes.unicode_minus": True,
        }
    )


def ax_mm(fig, x0, y0, x1, y1):
    ax = fig.add_axes([x0 / PAGE_W, y0 / PAGE_H, (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H])
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6, pad=1.6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    return ax


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# frozen package
# ---------------------------------------------------------------------------


def load_package():
    manifest = json.loads((PKG / "fig_manifest.json").read_text())
    files = {
        "resp": "fig02_formulation_response.csv",
        "supp": "fig02_ndvi_support.csv",
        "pooled": "fig02_pooled_metrics.csv",
        "boot": "fig02_pooled_bootstrap.csv",
        "eff": "fig02_site_rmse_effects.csv",
        "meta": "fig02_metadata.json",
    }
    for name in files.values():
        want = manifest["output_files"][name]["sha256"]
        got = sha256(PKG / name)
        assert got == want, f"{name}: on-disk hash != manifest hash"
    resp = pd.read_csv(PKG / files["resp"])
    supp = pd.read_csv(PKG / files["supp"])
    pooled = pd.read_csv(PKG / files["pooled"], dtype={"manuscript_value": str})
    boot = pd.read_csv(PKG / files["boot"])
    eff = pd.read_csv(PKG / files["eff"])
    meta = json.loads((PKG / files["meta"]).read_text())

    assert len(resp) == 3 * 60 * 101, "response table size"
    assert len(supp) == 50, "support table size"
    assert len(pooled) == 18, "pooled table size"
    assert len(boot) == 18, "bootstrap table size"
    assert len(eff) == 2 * (45 + 31), "effects table size"

    # Whole-site bootstrap intervals (10,000 resamples, seed 42) must bracket
    # and exactly reproduce the frozen pooled point values before display.
    keys = ["formulation", "scale", "metric"]
    assert (boot["n_resamples"] == 10_000).all() and (boot["seed"] == 42).all()
    merged = pooled.merge(
        boot[keys + ["value", "ci95_lo", "ci95_hi"]], on=keys, suffixes=("", "_boot")
    )
    assert len(merged) == 18, "pooled/bootstrap key mismatch"
    assert (merged["value"] == merged["value_boot"]).all(), "bootstrap point-value drift"
    assert (merged["ci95_lo"] <= merged["value"]).all() and (
        merged["value"] <= merged["ci95_hi"]
    ).all()
    pooled = merged.drop(columns=["value_boot"])
    assert sorted(resp["formulation"].unique()) == sorted(FORMS)
    assert resp["site_id"].nunique() == 60

    # Manuscript-precision reproduction (Table 3) re-checked at proof time.
    for _, r in pooled.iterrows():
        nd = 3 if (r["scale"] == "daily" or r["metric"] == "kge") else 2
        s = f"{r['value']:.{nd}f}"
        if s == "-" + f"{0.0:.{nd}f}":
            s = s[1:]
        assert s == str(r["manuscript_value"]), f"pooled {r['formulation']} mismatch"

    iso = eff[eff["comparator"] == "isolated_cover"]
    for scale, n_want, w_want in (("daily", 45, 43), ("monthly", 31, 27)):
        sub = iso[iso["scale"] == scale]
        assert len(sub) == n_want, f"isolated-cover {scale} site count"
        assert int(sub["win_cover_scaled"].sum()) == w_want, f"isolated-cover {scale} wins"
    return resp, supp, pooled, eff, meta


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

LABELS = {
    "cover_scaled_sigmoid": "Cover-scaled sigmoid",
    "unscaled_linear": "Unscaled linear",
    "unscaled_sigmoid": "Unscaled sigmoid",
}

# panel (a) geometry (mm)
A_X0, A_X1 = 12.5, 65.0
A_RESP_Y0, A_RESP_Y1 = 79.0, 117.0
A_SUPP_Y0, A_SUPP_Y1 = 67.5, 76.0

# panel (b) geometry
B_COL_X0 = (82.0, 119.0, 156.0)
B_COL_W = 30.0
B_ROW_Y = {"daily": (95.0, 117.0), "monthly": (67.5, 89.5)}
B_METRICS = ["kge", "rmse", "mbe"]
B_HEADS = {"kge": "KGE", "rmse": "RMSE", "mbe": "MBE"}

# panel (c) geometry
C_COL_X = ((13.0, 96.0), (104.0, 187.0))
C_ROW_Y = {"daily": (34.5, 49.0), "monthly": (12.0, 26.5)}
C_COMPS = ["isolated_cover", "whole_formulation"]
C_HEADS = {
    "isolated_cover": "Isolated-cover contrast",
    "whole_formulation": "Whole-formulation contrast",
}
C_SUBS = {
    "isolated_cover": "cover-scaled sigmoid − unscaled sigmoid",
    "whole_formulation": "cover-scaled sigmoid − unscaled linear",
}
C_UNIT = {"daily": "mm d$^{-1}$", "monthly": "mm month$^{-1}$"}
C_ROW_NAME = {"daily": "Daily", "monthly": "Monthly"}


def draw_panel_a(fig, resp, supp, meta):
    ax = ax_mm(fig, A_X0, A_RESP_Y0, A_X1, A_RESP_Y1)
    grid = np.sort(resp["ndvi"].unique())
    band_max = 0.0
    end_val = {}
    for form in FORMS:
        sub = resp[resp["formulation"] == form].pivot(index="ndvi", columns="site_id", values="k_t")
        assert sub.shape == (101, 60), f"pivot shape for {form}"
        q25 = sub.quantile(0.25, axis=1).to_numpy()
        q50 = sub.quantile(0.50, axis=1).to_numpy()
        q75 = sub.quantile(0.75, axis=1).to_numpy()
        st = STYLE[form]
        ax.fill_between(grid, q25, q75, color=st["color"], alpha=0.16, lw=0)
        ax.plot(grid, q50, color=st["color"], ls=st["ls"], lw=1.1)
        band_max = max(band_max, float(q75.max()))
        end_val[form] = float(q50[-1])
    ymax = np.ceil(band_max * 10.0 + 0.5) / 10.0
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0.0, ymax + 1e-9, 0.3))
    ax.set_ylabel("$K_T$ (dimensionless)", fontsize=FS_AXIS, labelpad=2.0)

    # One figure legend (FIGURE_STYLE_GUIDE §8): framed — square corners,
    # 0.5 pt black rule, opaque white fill — because it sits inside the axes.
    # The line + marker samples key the panel (a) curves and the panel (b)
    # point markers together, in the empty interior upper-left, entered in
    # the curves' stacking order at their right edge.
    order = sorted(FORMS, key=lambda f: end_val[f], reverse=True)
    handles: list = [
        Line2D(
            [],
            [],
            color=STYLE[f]["color"],
            ls=STYLE[f]["ls"],
            lw=1.1,
            marker=STYLE[f]["marker"],
            ms=3.4,
            mew=0,
        )
        for f in order
    ]
    labels = [LABELS[f] for f in order]
    # Reviewer item 8.1-1: the band must be defined in the artwork key — it is
    # the across-site IQR around the across-site median fitted response, not a
    # fit confidence or prediction interval. One neutral swatch keys all three
    # per-form bands; the overlaid line keys the median curves.
    handles.append(
        (
            Patch(facecolor="#9AA0A8", alpha=0.30, lw=0),
            Line2D([], [], color="#50545A", lw=1.1),
        )
    )
    labels.append("Across-site median and IQR")
    leg = ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#000000",
        facecolor="white",
        fontsize=FS_LEGEND,
        handlelength=2.4,
        handletextpad=0.6,
        labelspacing=0.55,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
    )
    leg.get_frame().set_linewidth(0.5)
    for t in leg.get_texts():
        t.set_color(C_TEXT)

    # Site-equal observed-NDVI support strip on the shared axis.
    axs = ax_mm(fig, A_X0, A_SUPP_Y0, A_X1, A_SUPP_Y1)
    width = float(supp["bin_right"].iloc[0] - supp["bin_left"].iloc[0])
    axs.bar(
        supp["bin_left"].to_numpy() + width / 2.0,
        supp["density_site_equal"].to_numpy(),
        width=width * 0.92,
        color=C_SUPPORT,
        lw=0,
    )
    axs.set_xlim(0.0, 1.0)
    axs.set_ylim(0.0, float(supp["density_site_equal"].max()) * 1.12)
    axs.set_yticks([])
    axs.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axs.set_xlabel("NDVI", fontsize=FS_AXIS, labelpad=1.6)
    n_obs = int(supp["n_obs"].sum())
    assert n_obs == meta["ndvi_support"]["n_obs_total"]
    # Reviewer item 8.2-10 counts (76,758 obs., 60 sites) and the site-equal
    # weighting scheme live in the caption (guide section 9); the assert above
    # keeps the caption's number pinned to the frozen package.
    axs.text(
        0.015,
        0.86,
        "Observed NDVI",
        fontsize=FS_ANNO,
        color=C_TEXT,
        ha="left",
        va="top",
        transform=axs.transAxes,
    )


def draw_panel_b(fig, pooled):
    x_pos = {f: i + 1 for i, f in enumerate(FORMS)}
    for scale, (y0, y1) in B_ROW_Y.items():
        for j, metric in enumerate(B_METRICS):
            ax = ax_mm(fig, B_COL_X0[j], y0, B_COL_X0[j] + B_COL_W, y1)
            sub = pooled[(pooled["scale"] == scale) & (pooled["metric"] == metric)]
            assert len(sub) == 3, f"panel-b facet {scale}/{metric}"
            vals = {r["formulation"]: r for _, r in sub.iterrows()}
            vv = [vals[f]["value"] for f in FORMS]
            lo = min(vals[f]["ci95_lo"] for f in FORMS)
            hi = max(vals[f]["ci95_hi"] for f in FORMS)
            pad = 0.10 * (hi - lo)
            ylo, yhi = lo - pad, hi + pad
            if metric == "mbe":
                ylo = min(ylo, -0.05 * (hi - lo))
                ax.axhline(0.0, color=C_RULE, lw=0.7, zorder=1)
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(0.5, 3.5)
            ax.set_xticks([])
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            # Per-point numeric labels removed (2026-08-31 note item 6): the
            # y-axis carries magnitude and Table 3 holds the exact values.
            for f in FORMS:
                st = STYLE[f]
                v = vals[f]["value"]
                # 95% whole-site bootstrap interval (caption-defined).
                ax.plot(
                    [x_pos[f], x_pos[f]],
                    [vals[f]["ci95_lo"], vals[f]["ci95_hi"]],
                    color=st["color"],
                    lw=0.9,
                    solid_capstyle="butt",
                    zorder=2,
                )
                ax.plot(
                    [x_pos[f]],
                    [v],
                    marker=st["marker"],
                    ms=4.2,
                    color=st["color"],
                    mew=0,
                    zorder=3,
                )
            if metric == "kge":
                unit = ""
            else:
                unit = C_UNIT[scale]
            if unit:
                ax.text(
                    0.02,
                    1.015,
                    unit,
                    fontsize=FS_TICK,
                    color=C_TEXT,
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                )
            if scale == "daily":
                fig.text(
                    (B_COL_X0[j] + B_COL_W / 2.0) / PAGE_W,
                    119.3 / PAGE_H,
                    B_HEADS[metric],
                    fontsize=FS_HEAD,
                    fontweight="semibold",
                    ha="center",
                    va="bottom",
                )
        fig.text(
            73.8 / PAGE_W,
            ((y0 + y1) / 2.0) / PAGE_H,
            C_ROW_NAME[scale],
            fontsize=FS_AXIS,
            color=C_TEXT,
            ha="center",
            va="center",
            rotation=90,
        )


def draw_panel_c(fig, eff):
    rng = np.random.default_rng(20260827)
    for scale, (y0, y1) in C_ROW_Y.items():
        row = eff[eff["scale"] == scale]
        span = float(np.abs(row["d_rmse"]).max()) * 1.12
        for k, comp in enumerate(C_COMPS):
            cx0, cx1 = C_COL_X[k]
            ax = ax_mm(fig, cx0, y0, cx1, y1)
            sub = eff[(eff["scale"] == scale) & (eff["comparator"] == comp)]
            d = sub["d_rmse"].to_numpy(dtype=float)
            n = len(d)
            # Collision-scale jitter only (2026-08-31 note item 5): the strip
            # spans 1.0 y-unit over ~14.5 mm, so half a marker width (~0.46 mm)
            # is about 0.032 units.
            jitter = 0.52 + rng.uniform(-0.045, 0.045, size=n)
            ax.axvline(0.0, color=C_RULE, lw=0.7, zorder=1)
            ax.plot(
                d,
                jitter,
                marker="o",
                ms=2.6,
                ls="",
                color="#0072B2",
                alpha=0.55,
                mew=0,
                zorder=3,
            )
            q25, q50, q75 = np.percentile(d, [25, 50, 75])
            ax.plot([q25, q75], [0.16, 0.16], color=C_TEXT, lw=1.2, zorder=4)
            ax.plot(
                [q50],
                [0.16],
                marker="D",
                ms=3.6,
                color=C_TEXT,
                mew=0.8,
                mfc="white",
                zorder=5,
            )
            ax.set_xlim(-span, span)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([])
            # Win-count annotations removed (2026-08-31 note item 6): the
            # paired distribution and the median/IQR glyph carry the evidence;
            # the win counts move to the caption.
            ax.text(
                0.015,
                0.94,
                C_ROW_NAME[scale],
                fontsize=FS_ANNO,
                color=C_TEXT,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            if scale == "daily":
                fig.text(
                    (cx0 + cx1) / 2.0 / PAGE_W,
                    52.8 / PAGE_H,
                    C_HEADS[comp],
                    fontsize=FS_HEAD,
                    fontweight="semibold",
                    ha="center",
                    va="bottom",
                )
                fig.text(
                    (cx0 + cx1) / 2.0 / PAGE_W,
                    49.9 / PAGE_H,
                    C_SUBS[comp],
                    fontsize=FS_ANNO,
                    color=C_TEXT,
                    ha="center",
                    va="bottom",
                )
        # One shared x title per row (§6/§16: identical axis titles are never
        # repeated across a grid), centered on the two contrast columns.
        fig.text(
            (C_COL_X[0][0] + C_COL_X[1][1]) / 2.0 / PAGE_W,
            (y0 - 3.5) / PAGE_H,
            f"ΔRMSE ({C_UNIT[scale]})",
            fontsize=FS_AXIS,
            color=C_TEXT,
            ha="center",
            va="top",
        )


def draw_headers(fig, meta):
    # Elsevier panel labels: plain-weight "(a)" fused with a sentence-case
    # identifier, one text object each (FIGURE_STYLE_GUIDE.md sections 4-5).
    # The cohort counts and the reading direction of the RMSE effects are
    # caption material, not in-figure text.
    fig.text(
        0.5 / PAGE_W,
        122.0 / PAGE_H,
        "(a) Fitted vegetation response",
        fontsize=FS_PANEL,
        va="bottom",
    )
    fig.text(
        71.0 / PAGE_W,
        122.0 / PAGE_H,
        "(b) Held-out ET agreement",
        fontsize=FS_PANEL,
        va="bottom",
    )
    mask = meta["evaluation_mask"]
    assert mask["n_sites"] == 45 and mask["n_daily"] == 63681 and mask["n_monthly"] == 1435
    # Reviewer item 8.2-7 support counts are stated in the Figure 2 caption
    # (guide section 9: exact n lives in a stats block or the caption); the
    # assert above keeps the caption's numbers pinned to the frozen package.
    fig.text(
        0.5 / PAGE_W,
        57.8 / PAGE_H,  # clears the NDVI xlabel above at Arial's wider set
        "(c) Paired site RMSE effects",
        fontsize=FS_PANEL,
        va="bottom",
    )
    # Reviewer item 8.1-3: the summary glyph is an open diamond at the median
    # with a horizontal IQR rule over paired site effects — keyed in the
    # artwork, not relabeled as a confidence interval.
    site_handle = Line2D([], [], ls="", marker="o", ms=2.6, color="#0072B2", alpha=0.55, mew=0)
    glyph_handle = Line2D([], [], color=C_TEXT, lw=1.2, marker="D", ms=3.6, mew=0.8, mfc="white")
    fig.legend(
        [site_handle, glyph_handle],
        ["Site ΔRMSE", "Median and IQR of site effects"],
        loc="lower right",
        bbox_to_anchor=(187.0 / PAGE_W, 57.4 / PAGE_H),
        ncol=2,
        frameon=False,
        fontsize=FS_ANNO,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )


# ---------------------------------------------------------------------------
# audits and export
# ---------------------------------------------------------------------------


def collect_texts(fig):
    items = []
    for t in fig.texts:
        items.append(t)
    for leg in fig.legends:
        items.extend(leg.get_texts())
    for ax in fig.axes:
        items.extend(ax.texts)
        items.append(ax.xaxis.label)
        items.append(ax.yaxis.label)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            items.append(tick)
        leg = ax.get_legend()
        if leg is not None:
            items.extend(leg.get_texts())
    return [t for t in items if t.get_text().strip()]


def audit(fig):
    texts = collect_texts(fig)
    rows = []
    for t in texts:
        s = t.get_text()
        size = float(t.get_fontsize())
        assert size >= FS_MIN - 1e-6, f"type below {FS_MIN} pt: {s!r} at {size}"
        # Below the 6.5 pt working ladder only the suite-standard 6 pt
        # interior-key size is sanctioned.
        assert size >= FS_TICK - 1e-6 or abs(size - FS_LEGEND) < 1e-6, (
            f"unsanctioned type size {size} pt: {s!r}"
        )
        for bad in FORBIDDEN_STRINGS:
            assert bad not in s, f"forbidden string {bad!r} in rendered text {s!r}"
        rows.append({"text": s, "fontsize": size})
    with open(HERE / f"{STEM}_textaudit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["text", "fontsize"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _simulate_cvd(image: Image.Image, matrix: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    transformed = np.clip(rgb @ matrix.T, 0.0, 1.0)
    return Image.fromarray(np.round(transformed * 255).astype(np.uint8), mode="RGB")


def _write_review_rasters(png_path: Path) -> None:
    source = Image.open(png_path)
    ImageOps.grayscale(source).save(png_path.with_name(f"{png_path.stem}_grayscale.png"))
    source.resize(
        (source.width // 4, source.height // 4),
        resample=Image.Resampling.LANCZOS,
    ).save(png_path.with_name(f"{png_path.stem}_printcheck.png"))

    matrices = {
        "protanopia": np.array(
            [
                [0.56667, 0.43333, 0.00000],
                [0.55833, 0.44167, 0.00000],
                [0.00000, 0.24167, 0.75833],
            ]
        ),
        "deuteranopia": np.array(
            [
                [0.62500, 0.37500, 0.00000],
                [0.70000, 0.30000, 0.00000],
                [0.00000, 0.30000, 0.70000],
            ]
        ),
        "tritanopia": np.array(
            [
                [0.95000, 0.05000, 0.00000],
                [0.00000, 0.43333, 0.56667],
                [0.00000, 0.47500, 0.52500],
            ]
        ),
    }
    for label, matrix in matrices.items():
        _simulate_cvd(source, matrix).save(png_path.with_name(f"{png_path.stem}_cvd_{label}.png"))


def main():
    register_typeface()
    resp, supp, pooled, eff, meta = load_package()

    fig = plt.figure(figsize=(PAGE_W * MM, PAGE_H * MM))
    fig.patch.set_facecolor("white")

    draw_panel_a(fig, resp, supp, meta)
    draw_panel_b(fig, pooled)
    draw_panel_c(fig, eff)
    draw_headers(fig, meta)

    w_in, h_in = fig.get_size_inches()
    assert abs(w_in * 25.4 - PAGE_W) < 1e-6 and abs(h_in * 25.4 - PAGE_H) < 1e-6
    n_texts = audit(fig)

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(HERE / f"{STEM}.{ext}", facecolor="white", **kw)
    _write_review_rasters(HERE / f"{STEM}.png")
    print(f"{STEM}: rendered 190 x 125 mm, {n_texts} text items, all checks passed")
    print(f"  package: {PKG}")
    print("  review rasters: grayscale, printcheck, cvd_{protanopia,deuteranopia,tritanopia}")


if __name__ == "__main__":
    main()

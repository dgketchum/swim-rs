"""Render Figure 3 from the frozen E1 OpenET benchmark display package.

The renderer is deliberately presentation-only. All eligibility, metric,
aggregation, contrast, interaction, and bootstrap calculations are frozen by
``scripts/figures/build_figure_data.py --only fig03``. This script verifies the
display-package hashes, rechecks the plotted descriptive statistics, and draws
the publication proof.

Usage::

    uv run python scripts/figures/fig03_temporal_reconstruction.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
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
from PIL import Image, ImageOps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "paper" / "data" / "final" / "figures"
OUTDIR = REPO / "paper" / "figures" / "proofs" / "fig03_pooled_agreement_190_r2"
STEM = "fig03_pooled_agreement_r2"

PAGE_W, PAGE_H = 190.0, 125.0  # mm
RASTER_DPI = 600
MIN_PT = 6.5

C_BLUE = "#0072B2"
C_TEXT = "#000000"
C_CHARCOAL = "#4F5459"
C_MID = "#777D82"
C_LIGHT = "#C8CDD1"

AX_LO, AX_HI = -2.0, 16.0
AX_TICKS = [0, 4, 8, 12, 16]
HEX_GRIDSIZE = 40
CBAR_TICKS = [1, 10, 100, 1000]

SUPPORTS = [
    ("retrieval", "Retrieval dates", 4972),
    ("between_retrieval", "Between retrievals", 54300),
]
METHODS = [("openet_et", "OpenET"), ("swim_et", "SWIM-RS")]
METRICS = [
    ("kge", "ΔKGE", ""),
    ("rmse", "ΔRMSE", "mm d$^{-1}$"),
    ("mbe", "ΔMBE", "mm d$^{-1}$"),
]
AGGREGATIONS = [
    ("sqrt_n_weighted_site_metric", "Station-weighted"),
    ("pooled_observations", "Pooled"),
]

B_LIMS = {"kge": (-0.04, 0.07), "rmse": (-0.10, 0.20), "mbe": (-0.05, 0.40)}
B_TICKS = {
    "kge": [-0.04, 0.00, 0.04],
    "rmse": [-0.10, 0.00, 0.10, 0.20],
    "mbe": [0.00, 0.20, 0.40],
}
C_LIMS = {"kge": (-0.10, 0.14), "rmse": (-0.42, 0.34), "mbe": (-0.28, 0.42)}
C_TICKS = {
    "kge": [-0.10, 0.00, 0.10],
    "rmse": [-0.40, -0.20, 0.00, 0.20],
    "mbe": [-0.20, 0.00, 0.20, 0.40],
}

FILES = [
    "fig03_pooled_daily_agreement.csv",
    "fig03_scatter_metrics.csv",
    "fig03_grouped_contrasts.csv",
    "fig03_interactions.csv",
    "fig03_site_interactions.csv",
    "fig03_metadata.json",
]


def _ex5_canonical_run() -> str:
    """``CANONICAL_RUN`` from ``examples/5_Flux_Ensemble/ex5_paths.py``, loaded by path."""
    path = REPO / "examples" / "5_Flux_Ensemble" / "ex5_paths.py"
    spec = importlib.util.spec_from_file_location("_fig03_ex5_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CANONICAL_RUN


# Internal archive tags (the canonical Ex5 run and its predecessor) never reach a figure.
FORBIDDEN_STRINGS = [
    "run22",
    _ex5_canonical_run(),
    "non-overpass",
    "non_overpass",
    "acquisition",
    "gap-filled",
    "NSE",
    "Bias",
    "MAE",
    "|MBE|",
    "p =",
    "p<",
]


class ProofError(RuntimeError):
    """Raised when the frozen data or rendered contract has drifted."""


def register_fonts() -> None:
    for directory in [
        Path.home() / ".fonts" / "arial",
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    ]:
        if directory.exists():
            for path in sorted(directory.glob("[Aa]rial*.[TtOo][Tt][Ff]")):
                fm.fontManager.addfont(str(path))
    if "Arial" not in {font.name for font in fm.fontManager.ttflist}:
        raise ProofError("Arial is not registered; no fallback is allowed")

    style = Path.home() / "code" / "style" / "journal_figures.mplstyle"
    if not style.exists():
        raise ProofError(f"shared figure style is missing: {style}")
    plt.style.use(str(style))
    plt.rcParams.update(
        {
            "savefig.bbox": "standard",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    residual = sim - obs
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(np.std(sim, ddof=0) / np.std(obs, ddof=0))
    beta = float(np.mean(sim) / np.mean(obs))
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {
        "pearson_r": r,
        "kge": kge,
        "mbe": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def signed_display(value: float) -> str:
    rounded = round(value, 2)
    if rounded == 0:
        return "0.00"
    if rounded > 0:
        return f"+{rounded:.2f}"
    return f"−{abs(rounded):.2f}"


def load_package() -> dict[str, object]:
    manifest = json.loads((PKG / "fig_manifest.json").read_text())["tables"]
    hashes = {}
    for name in FILES:
        if name not in manifest:
            raise ProofError(f"{name} is missing from fig_manifest.json")
        hashes[name] = sha256(PKG / name)
        expected = manifest[name]["output_sha256"]
        if hashes[name] != expected:
            raise ProofError(f"{name}: sha256 {hashes[name][:12]} != manifest {expected[:12]}")

    pooled = pd.read_csv(PKG / "fig03_pooled_daily_agreement.csv")
    scatter = pd.read_csv(
        PKG / "fig03_scatter_metrics.csv",
        dtype={
            "display_r": str,
            "display_kge": str,
            "display_mbe": str,
            "display_rmse": str,
        },
    )
    contrasts = pd.read_csv(PKG / "fig03_grouped_contrasts.csv")
    interactions = pd.read_csv(PKG / "fig03_interactions.csv")
    sites = pd.read_csv(PKG / "fig03_site_interactions.csv")
    metadata = json.loads((PKG / "fig03_metadata.json").read_text())

    if metadata.get("source_package") != "paper/data/final/e1_openet_benchmark":
        raise ProofError("Figure 3 is not pointed at the promoted E1 package")
    if metadata.get("source_status") != "frozen_for_results_reporting":
        raise ProofError("promoted E1 package is not frozen for results reporting")
    if len(pooled) != 59272 or pooled["site_id"].nunique() != 43:
        raise ProofError("pooled table does not carry 59,272 rows over 43 sites")
    if pooled.duplicated(["site_id", "date"]).any():
        raise ProofError("duplicate site/date keys in the pooled table")
    for support, _label, expected_n in SUPPORTS:
        got = int((pooled["temporal_support"] == support).sum())
        if got != expected_n:
            raise ProofError(f"{support} count {got} != {expected_n}")
    values = pooled[["flux_et", "swim_et", "openet_et"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.min() < AX_LO or values.max() > AX_HI:
        raise ProofError("a pooled ET value is nonfinite or outside the frozen axes")

    if len(scatter) != 4:
        raise ProofError("scatter statistics must contain four facet rows")
    for row in scatter.itertuples(index=False):
        sub = pooled[pooled["temporal_support"] == row.temporal_support]
        column = "openet_et" if row.method == "OpenET" else "swim_et"
        reproduced = metrics(sub["flux_et"].to_numpy(), sub[column].to_numpy())
        for name in ["pearson_r", "kge", "mbe", "rmse"]:
            if abs(reproduced[name] - float(getattr(row, name))) > 1e-12:
                raise ProofError(f"scatter {name} fails to reproduce for {row.method}")
        expected_display = {
            "display_r": f"{reproduced['pearson_r']:.2f}",
            "display_kge": f"{reproduced['kge']:.2f}",
            "display_mbe": signed_display(reproduced["mbe"]),
            "display_rmse": f"{reproduced['rmse']:.2f}",
        }
        for name, expected in expected_display.items():
            if str(getattr(row, name)) != expected:
                raise ProofError(f"{name} drifted for {row.method}/{row.temporal_support}")

    expected_contrast_keys = {
        (support, aggregation, metric)
        for support, _label, _n in SUPPORTS
        for aggregation, _reader_label in AGGREGATIONS
        for metric, _head, _unit in METRICS
    }
    got_contrast_keys = set(
        contrasts[["temporal_class", "aggregation", "metric"]].itertuples(index=False, name=None)
    )
    if got_contrast_keys != expected_contrast_keys:
        raise ProofError("grouped contrast key set drifted")
    expected_interaction_keys = {
        (aggregation, metric)
        for aggregation, _reader_label in AGGREGATIONS
        for metric, _head, _unit in METRICS
    }
    got_interaction_keys = set(
        interactions[["aggregation", "metric"]].itertuples(index=False, name=None)
    )
    if got_interaction_keys != expected_interaction_keys:
        raise ProofError("interaction key set drifted")
    for frame in [contrasts, interactions]:
        if not (frame["n_sites"] == 43).all():
            raise ProofError("an interval row does not use the 43-site cohort")
        if not ((frame["bootstrap_reps"] == 10000) & (frame["bootstrap_seed"] == 42)).all():
            raise ProofError("bootstrap settings drifted")
        if not (
            (frame["ci95_low"] <= frame["estimate"]) & (frame["estimate"] <= frame["ci95_high"])
        ).all():
            raise ProofError("an estimate falls outside its 95% interval")

    if len(sites) != 43 or sites["site_id"].nunique() != 43:
        raise ProofError("site interactions do not contain 43 unique sites")
    if sorted(sites["site_order_interaction_kge"].astype(int)) != list(range(1, 44)):
        raise ProofError("site interaction order is not a 1..43 permutation")
    for metric, _head, _unit in METRICS:
        blo, bhi = B_LIMS[metric]
        b = contrasts[contrasts["metric"] == metric]
        if b["ci95_low"].min() < blo or b["ci95_high"].max() > bhi:
            raise ProofError(f"panel (b) {metric} interval is clipped")
        clo, chi = C_LIMS[metric]
        c = interactions[interactions["metric"] == metric]
        site_values = sites[f"interaction_{metric}"]
        if min(c["ci95_low"].min(), site_values.min()) < clo:
            raise ProofError(f"panel (c) {metric} lower range is clipped")
        if max(c["ci95_high"].max(), site_values.max()) > chi:
            raise ProofError(f"panel (c) {metric} upper range is clipped")

    return {
        "pooled": pooled,
        "scatter": scatter,
        "contrasts": contrasts,
        "interactions": interactions,
        "sites": sites,
        "metadata": metadata,
        "hashes": hashes,
    }


def ax_mm(fig, x0: float, y0: float, width: float, height: float):
    return fig.add_axes([x0 / PAGE_W, y0 / PAGE_H, width / PAGE_W, height / PAGE_H])


def fig_text(fig, x_mm: float, y_mm: float, text: str, **kwargs):
    return fig.text(x_mm / PAGE_W, y_mm / PAGE_H, text, **kwargs)


def style_box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_CHARCOAL)
        spine.set_linewidth(0.55)
    ax.tick_params(axis="both", labelsize=6.5, length=2.2, width=0.55, pad=1.4)
    ax.set_facecolor("white")


def draw_panel_a(fig, pooled: pd.DataFrame, scatter: pd.DataFrame) -> dict[str, int]:
    side = 41.5
    x_positions = [13.0, 60.5]
    y_positions = [58.0, 7.5]
    stat_top_y = [106.0, 56.0]
    stat_bottom_y = [102.8, 52.8]
    display = scatter.set_index(["method", "temporal_support"])
    hexbins = []

    for row_index, (column, method) in enumerate(METHODS):
        for column_index, (support, _label, _n) in enumerate(SUPPORTS):
            ax = ax_mm(fig, x_positions[column_index], y_positions[row_index], side, side)
            sub = pooled[pooled["temporal_support"] == support]
            hb = ax.hexbin(
                sub["flux_et"].to_numpy(),
                sub[column].to_numpy(),
                gridsize=HEX_GRIDSIZE,
                extent=(AX_LO, AX_HI, AX_LO, AX_HI),
                cmap="viridis",
                mincnt=1,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            hexbins.append(hb)
            ax.plot(
                [AX_LO, AX_HI],
                [AX_LO, AX_HI],
                color=C_CHARCOAL,
                lw=0.7,
                ls=(0, (4, 2)),
                zorder=3,
            )
            ax.set(xlim=(AX_LO, AX_HI), ylim=(AX_LO, AX_HI))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks(AX_TICKS)
            ax.set_yticks(AX_TICKS)
            if column_index:
                ax.tick_params(axis="y", labelleft=False)
            if not row_index:
                ax.tick_params(axis="x", labelbottom=False)
            style_box(ax)

            stat = display.loc[(method, support)]
            fig_text(
                fig,
                x_positions[column_index],
                stat_top_y[row_index],
                method,
                fontsize=7.0,
                fontweight="semibold",
                ha="left",
                va="top",
            )
            fig_text(
                fig,
                x_positions[column_index] + 12.0,
                stat_top_y[row_index],
                f"$r$ = {stat['display_r']}; KGE = {stat['display_kge']}",
                fontsize=6.5,
                ha="left",
                va="top",
            )
            fig_text(
                fig,
                x_positions[column_index],
                stat_bottom_y[row_index],
                (f"MBE = {stat['display_mbe']}; RMSE = {stat['display_rmse']} mm d$^{{-1}}$"),
                fontsize=6.5,
                ha="left",
                va="top",
            )

    for column_index, (_support, label, count) in enumerate(SUPPORTS):
        center = x_positions[column_index] + side / 2
        fig_text(
            fig,
            center,
            114.5,
            label,
            fontsize=7.0,
            fontweight="semibold",
            ha="center",
            va="bottom",
        )
        fig_text(
            fig,
            center,
            110.8,
            f"$n$ = {count:,} site-days",
            fontsize=6.5,
            ha="center",
            va="bottom",
        )

    fig_text(
        fig,
        57.5,
        0.2,
        "Flux ET (mm d$^{-1}$)",
        fontsize=7.0,
        ha="center",
        va="bottom",
    )
    fig_text(
        fig,
        4.3,
        53.5,
        "Estimated ET (mm d$^{-1}$)",
        fontsize=7.0,
        ha="center",
        va="center",
        rotation=90,
    )

    maximum = max(int(hb.get_array().max()) for hb in hexbins)
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=maximum)
    for hb in hexbins:
        hb.set_norm(norm)
    cax = ax_mm(fig, 104.5, 31.5, 2.3, 41.0)
    colorbar = fig.colorbar(hexbins[0], cax=cax)
    ticks = [tick for tick in CBAR_TICKS if tick <= maximum]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:,}" for tick in ticks])
    colorbar.minorticks_off()
    colorbar.outline.set_edgecolor(C_CHARCOAL)
    colorbar.outline.set_linewidth(0.55)
    cax.tick_params(labelsize=6.5, length=2.2, width=0.55, pad=1.0)
    fig_text(fig, 104.5, 73.5, "Site-days", fontsize=6.5, ha="left", va="bottom")
    return {"hexbin_max_count": maximum}


def contrast_axis(ax, metric: str, panel: str) -> None:
    limits = B_LIMS if panel == "b" else C_LIMS
    ticks = B_TICKS if panel == "b" else C_TICKS
    ax.set_xlim(*limits[metric])
    ax.set_xticks(ticks[metric])
    ax.set_xticklabels([f"{value:g}".replace("-", "−") for value in ticks[metric]])
    ax.axvline(0, color=C_MID, lw=0.55, zorder=1)
    style_box(ax)


def draw_panel_b(fig, contrasts: pd.DataFrame) -> None:
    x_positions = [131.5, 151.0, 170.5]
    width, y0, height = 17.0, 82.5, 22.0
    base_y = {"retrieval": 1.0, "between_retrieval": 0.0}
    offsets = {"sqrt_n_weighted_site_metric": 0.14, "pooled_observations": -0.14}

    for column_index, (metric, head, unit) in enumerate(METRICS):
        ax = ax_mm(fig, x_positions[column_index], y0, width, height)
        contrast_axis(ax, metric, "b")
        ax.set_ylim(-0.50, 1.50)
        ax.set_yticks([1.0, 0.0])
        if column_index == 0:
            ax.set_yticklabels(["Retrieval dates", "Between retrievals"], fontsize=6.5)
            ax.tick_params(axis="y", length=0, pad=3.0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        title = head if not unit else f"{head}\n({unit})"
        ax.set_title(title, fontsize=7.0, fontweight="semibold", pad=3.0, linespacing=1.0)

        for aggregation, _label in AGGREGATIONS:
            subset = contrasts[
                (contrasts["metric"] == metric) & (contrasts["aggregation"] == aggregation)
            ]
            for row in subset.itertuples(index=False):
                y = base_y[row.temporal_class] + offsets[aggregation]
                if aggregation == "sqrt_n_weighted_site_metric":
                    color, marker, face, size = C_BLUE, "o", C_BLUE, 3.1
                else:
                    color, marker, face, size = C_CHARCOAL, "s", "white", 3.0
                ax.hlines(y, row.ci95_low, row.ci95_high, color=color, lw=1.05, zorder=2)
                ax.plot(
                    row.estimate,
                    y,
                    marker=marker,
                    ms=size,
                    mfc=face,
                    mec=color,
                    mew=0.65,
                    ls="none",
                    zorder=3,
                )


def draw_shared_key(fig) -> None:
    ax = ax_mm(fig, 116.5, 72.0, 71.0, 4.5)
    ax.set(xlim=(0, 71), ylim=(0, 1))
    ax.set_axis_off()
    entries = [
        (8.0, C_BLUE, "o", C_BLUE, "Station-weighted"),
        (34.0, C_CHARCOAL, "s", "white", "Pooled"),
        (53.5, C_MID, "o", "white", "Individual site"),
    ]
    for x, color, marker, face, label in entries:
        if label == "Individual site":
            ax.plot(x, 0.50, marker=marker, ms=2.6, mfc=face, mec=color, mew=0.6, ls="none")
        else:
            ax.plot([x - 2.5, x + 2.5], [0.50, 0.50], color=color, lw=1.0)
            ax.plot(
                x,
                0.50,
                marker=marker,
                ms=3.0,
                mfc=face,
                mec=color,
                mew=0.65,
                ls="none",
            )
        ax.text(x + 3.2, 0.50, label, fontsize=6.5, ha="left", va="center")


def draw_panel_c(fig, interactions: pd.DataFrame, sites: pd.DataFrame) -> None:
    x_positions = [119.0, 142.3, 165.6]
    width, y0, height = 20.9, 11.5, 43.5
    ordered = sites.sort_values("site_order_interaction_kge")
    site_y = np.arange(len(ordered), dtype=float)
    summary_y = {"pooled_observations": 45.5, "sqrt_n_weighted_site_metric": 47.6}

    for column_index, (metric, head, unit) in enumerate(METRICS):
        ax = ax_mm(fig, x_positions[column_index], y0, width, height)
        contrast_axis(ax, metric, "c")
        ax.set_ylim(-1.0, 49.0)
        ax.set_yticks([])
        title = head if not unit else f"{head}\n({unit})"
        ax.set_title(title, fontsize=7.0, fontweight="semibold", pad=3.0, linespacing=1.0)
        ax.axhline(43.7, color=C_LIGHT, lw=0.55, zorder=1)
        ax.plot(
            ordered[f"interaction_{metric}"].to_numpy(),
            site_y,
            marker="o",
            ms=2.25,
            mfc="white",
            mec=C_MID,
            mew=0.55,
            ls="none",
            zorder=2,
        )

        for aggregation, _label in AGGREGATIONS:
            row = interactions[
                (interactions["metric"] == metric) & (interactions["aggregation"] == aggregation)
            ].iloc[0]
            y = summary_y[aggregation]
            if aggregation == "sqrt_n_weighted_site_metric":
                color, marker, face, size = C_BLUE, "o", C_BLUE, 3.1
            else:
                color, marker, face, size = C_CHARCOAL, "s", "white", 3.0
            ax.hlines(y, row["ci95_low"], row["ci95_high"], color=color, lw=1.05, zorder=3)
            ax.plot(
                row["estimate"],
                y,
                marker=marker,
                ms=size,
                mfc=face,
                mec=color,
                mew=0.65,
                ls="none",
                zorder=4,
            )


def audit(fig) -> list[dict[str, object]]:
    items = []
    for artist in fig.findobj(matplotlib.text.Text):
        text = artist.get_text().strip()
        if not text:
            continue
        size = float(artist.get_fontsize())
        items.append({"text": text, "fontsize_pt": round(size, 2)})
        if size < MIN_PT - 1e-6:
            raise ProofError(f"text below {MIN_PT} pt: {text!r} at {size} pt")
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden.lower() in text.lower():
                raise ProofError(f"forbidden string {forbidden!r} rendered: {text!r}")
    return items


def simulate_cvd(source: Image.Image, matrix: np.ndarray) -> Image.Image:
    rgb = np.asarray(source.convert("RGB"), dtype=float) / 255.0
    transformed = np.clip(rgb @ matrix.T, 0.0, 1.0)
    return Image.fromarray(np.round(transformed * 255).astype(np.uint8))


def write_review_rasters(png_path: Path) -> None:
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
        simulate_cvd(source, matrix).save(png_path.with_name(f"{png_path.stem}_cvd_{label}.png"))


def main() -> None:
    register_fonts()
    package = load_package()
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=300, facecolor="white")

    panel_a_meta = draw_panel_a(fig, package["pooled"], package["scatter"])
    draw_panel_b(fig, package["contrasts"])
    draw_shared_key(fig)
    draw_panel_c(fig, package["interactions"], package["sites"])

    fig_text(
        fig,
        4.5,
        120.5,
        "(a) Pooled daily ET agreement (43 sites)",
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    fig_text(
        fig,
        116.0,
        120.5,
        "(b) Relative performance by temporal support",
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    fig_text(
        fig,
        116.0,
        116.0,
        "Δ = SWIM-RS − OpenET; $n$ = 43 sites",
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    fig_text(
        fig,
        119.0,
        65.7,
        "(c) Temporal change in relative performance (43 sites)",
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    fig_text(
        fig,
        153.5,
        2.3,
        "Between-retrieval Δ − retrieval-date Δ; sites ordered by ΔKGE",
        fontsize=6.5,
        ha="center",
        va="bottom",
    )

    text_items = audit(fig)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(
            OUTDIR / f"{STEM}.{extension}",
            dpi=RASTER_DPI,
            facecolor="white",
            bbox_inches=None,
        )
    plt.close(fig)
    write_review_rasters(OUTDIR / f"{STEM}.png")

    with (OUTDIR / f"{STEM}_textaudit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "fontsize_pt"])
        writer.writeheader()
        writer.writerows(text_items)
    shutil.copy2(__file__, OUTDIR / Path(__file__).name)

    metadata = {
        "figure": "Figure 3 -- OpenET benchmark agreement by temporal support",
        "contract": "paper/notes/fig03_production_handoff.md (2026-09-01)",
        "style": "FIGURE_STYLE_GUIDE.md + journal_figures.mplstyle",
        "composition_id": package["metadata"]["composition_id"],
        "canvas_mm": [PAGE_W, PAGE_H],
        "raster_dpi": RASTER_DPI,
        "counts": package["metadata"]["counts"],
        "panel_a": {
            "axes_mm_day": [AX_LO, AX_HI],
            "ticks": AX_TICKS,
            "encoding": "hexbin density with a shared logarithmic count scale",
            "hex_gridsize": HEX_GRIDSIZE,
            **panel_a_meta,
        },
        "panel_b_limits": B_LIMS,
        "panel_c_limits": C_LIMS,
        "review_rasters": [
            f"{STEM}_grayscale.png",
            f"{STEM}_printcheck.png",
            f"{STEM}_cvd_protanopia.png",
            f"{STEM}_cvd_deuteranopia.png",
            f"{STEM}_cvd_tritanopia.png",
        ],
        "package_hashes": package["hashes"],
        "text_items_audited": len(text_items),
        "rendered_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "scripts/figures/fig03_temporal_reconstruction.py",
    }
    (OUTDIR / f"{STEM}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"rendered {STEM}.pdf/svg/png to {OUTDIR}")
    print(f"text items audited: {len(text_items)} (all >= {MIN_PT} pt)")
    print("review rasters: grayscale, printcheck, cvd_{protanopia,deuteranopia,tritanopia}")


if __name__ == "__main__":
    main()

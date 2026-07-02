"""
Publication-quality figures for SWIM-RS international flux tower evaluation.

Generates figures comparing Landsat PT-JPL vs ECOSTRESS PT-JPL as calibration
targets across ~64 cropland flux tower sites globally.

Usage:
    python publication_figures.py --results-dir /path/to/results [--format pdf]
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluate import find_flux_file, load_flux_sources
from sklearn.metrics import mean_squared_error, r2_score

# Journal-ready defaults
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
    }
)


def load_evaluation_summary(results_dir: str) -> pd.DataFrame:
    """Load evaluation_summary.csv from a results directory."""
    csv = os.path.join(results_dir, "evaluation_summary.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"No evaluation_summary.csv in {results_dir}")
    return pd.read_csv(csv)


def load_site_outputs(results_dir: str, sites: list[str]) -> dict[str, pd.DataFrame]:
    """Load per-site model output CSVs."""
    outputs = {}
    for site in sites:
        csv = os.path.join(results_dir, f"{site}.csv")
        if os.path.exists(csv):
            outputs[site] = pd.read_csv(csv, index_col=0, parse_dates=True)
    return outputs


def fig_site_map(shapefile: str, summary_df: pd.DataFrame, output_path: str):
    """Figure 1: Global map of flux stations colored by performance."""
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"), engine="fiona")
    except Exception:
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
            engine="fiona",
        )

    gdf = gpd.read_file(shapefile, engine="fiona")
    gdf = gdf.set_index("sid")

    # Merge with metrics if available
    if summary_df is not None and "site" in summary_df.columns:
        summary_df = summary_df.set_index("site")
        gdf = gdf.join(summary_df[["rmse", "r2", "kge"]], how="left")

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": None})
    world.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.5)

    color_col = "kge" if "kge" in gdf.columns else None
    if color_col and gdf[color_col].notna().any():
        gdf.plot(
            ax=ax,
            column=color_col,
            cmap="RdYlGn",
            vmin=-0.5,
            vmax=1.0,
            markersize=30,
            edgecolor="k",
            linewidth=0.3,
            legend=True,
            legend_kwds={"label": "KGE", "shrink": 0.6},
        )
    else:
        gdf.plot(ax=ax, color="royalblue", markersize=30, edgecolor="k", linewidth=0.3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"International Flux Tower Sites (n={len(gdf)})")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def fig_scatter(results_dirs: dict[str, str], output_path: str, flux_sources: dict | None = None):
    """Figure 2: Scatter plots of SWIM ET vs flux tower ET for each experiment."""
    flux_sources = flux_sources or {}
    experiments = list(results_dirs.keys())
    n_exp = len(experiments)

    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5), squeeze=False)

    for i, (exp_name, res_dir) in enumerate(results_dirs.items()):
        ax = axes[0, i]
        summary = load_evaluation_summary(res_dir)
        outputs = load_site_outputs(res_dir, summary["site"].tolist())

        all_model, all_flux = [], []

        for site in summary["site"]:
            if site not in outputs:
                continue
            network, et_col = flux_sources.get(site, (None, None))
            flux_file = find_flux_file(site, network)
            if flux_file is None:
                continue

            model_df = outputs[site]
            flux_df = pd.read_csv(flux_file, index_col="date", parse_dates=True)
            common = model_df.index.intersection(flux_df.index)

            if len(common) < 10:
                continue

            model_et = model_df.loc[common, "et_act"]
            for col in ([et_col] if et_col else []) + ["ET_corr", "ET", "LE_corr"]:
                if col in flux_df.columns:
                    flux_et = flux_df.loc[common, col]
                    break
            else:
                continue

            valid = ~(model_et.isna() | flux_et.isna())
            # Monthly aggregation for cleaner scatter
            m_monthly = model_et[valid].resample("MS").mean().dropna()
            f_monthly = flux_et[valid].resample("MS").mean().dropna()
            common_mo = m_monthly.index.intersection(f_monthly.index)

            all_model.extend(m_monthly.loc[common_mo].values)
            all_flux.extend(f_monthly.loc[common_mo].values)

        all_model = np.array(all_model)
        all_flux = np.array(all_flux)

        if len(all_model) > 0:
            ax.scatter(all_flux, all_model, s=5, alpha=0.3, c="steelblue")

            # 1:1 line
            lim = max(all_flux.max(), all_model.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)

            # Metrics
            rmse = np.sqrt(mean_squared_error(all_flux, all_model))
            r2 = r2_score(all_flux, all_model)
            bias = np.mean(all_model - all_flux)
            ax.text(
                0.05,
                0.95,
                f"RMSE={rmse:.2f}\nR$^2$={r2:.3f}\nBias={bias:.2f}\nn={len(all_model)}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel("Flux Tower ET (mm/day)")
        ax.set_ylabel("SWIM ET (mm/day)")
        ax.set_title(exp_name)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def fig_metric_distributions(results_dirs: dict[str, str], output_path: str):
    """Figure 3: Box/violin plots of per-site metrics across experiments."""
    metrics = ["rmse", "r2", "kge", "bias"]
    labels = ["RMSE (mm/day)", "R$^2$", "KGE", "Bias (mm/day)"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))

    for j, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[j]
        data = []
        names = []

        for exp_name, res_dir in results_dirs.items():
            try:
                summary = load_evaluation_summary(res_dir)
                if metric in summary.columns:
                    vals = summary[metric].dropna().values
                    data.append(vals)
                    names.append(exp_name)
            except FileNotFoundError:
                continue

        if data:
            bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.6)
            colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
            for patch, color in zip(bp["boxes"], colors[: len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def fig_head_to_head(results_a: str, results_c: str, output_path: str):
    """Figure 4: Head-to-head comparison of Experiment A vs C metrics."""
    try:
        df_a = load_evaluation_summary(results_a).set_index("site")
        df_c = load_evaluation_summary(results_c).set_index("site")
    except FileNotFoundError as e:
        print(f"  Skipping head-to-head: {e}")
        return

    common = df_a.index.intersection(df_c.index)
    if len(common) < 5:
        print("  Too few common sites for head-to-head")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric, label in zip(
        axes,
        ["rmse", "r2", "kge"],
        ["RMSE (mm/day)", "R$^2$", "KGE"],
    ):
        a_vals = df_a.loc[common, metric].values
        c_vals = df_c.loc[common, metric].values

        ax.scatter(a_vals, c_vals, s=20, alpha=0.6, c="steelblue", edgecolors="k", linewidths=0.3)

        lims = [
            min(np.nanmin(a_vals), np.nanmin(c_vals)),
            max(np.nanmax(a_vals), np.nanmax(c_vals)),
        ]
        margin = (lims[1] - lims[0]) * 0.1
        ax.plot(
            [lims[0] - margin, lims[1] + margin],
            [lims[0] - margin, lims[1] + margin],
            "k--",
            lw=0.8,
        )
        ax.set_xlim(lims[0] - margin, lims[1] + margin)
        ax.set_ylim(lims[0] - margin, lims[1] + margin)

        ax.set_xlabel(f"Landsat Only — {label}")
        ax.set_ylabel(f"Landsat + ECOSTRESS — {label}")
        ax.set_aspect("equal")

        # Count improved sites
        if metric == "rmse":
            improved = np.sum(c_vals < a_vals)
        else:
            improved = np.sum(c_vals > a_vals)
        ax.set_title(f"{improved}/{len(common)} sites improved")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def fig_time_series(
    results_dir: str, sites: list[str], output_path: str, flux_sources: dict | None = None
):
    """Figure 5: Time series panels for representative sites."""
    flux_sources = flux_sources or {}
    n_sites = len(sites)
    fig, axes = plt.subplots(n_sites, 1, figsize=(12, 3 * n_sites), sharex=False)
    if n_sites == 1:
        axes = [axes]

    for ax, site in zip(axes, sites):
        csv = os.path.join(results_dir, f"{site}.csv")
        if not os.path.exists(csv):
            ax.set_title(f"{site} — no model output")
            continue

        model_df = pd.read_csv(csv, index_col=0, parse_dates=True)

        # Plot model ET
        ax.plot(model_df.index, model_df["et_act"], color="steelblue", lw=0.8, label="SWIM ET")

        # Plot flux ET if available
        network, et_col = flux_sources.get(site, (None, None))
        flux_file = find_flux_file(site, network)
        if flux_file:
            flux_df = pd.read_csv(flux_file, index_col="date", parse_dates=True)
            col_order = ([et_col] if et_col else []) + ["ET_corr", "ET"]
            for col, clabel in [(c, f"Flux ET ({c})") for c in col_order]:
                if col in flux_df.columns:
                    common = model_df.index.intersection(flux_df.index)
                    ax.scatter(
                        common,
                        flux_df.loc[common, col],
                        s=3,
                        alpha=0.5,
                        c="orangered",
                        label=clabel,
                    )
                    break

        # Plot ETf observations if available
        if "etf" in model_df.columns:
            etf_obs = model_df["etf"].dropna()
            if not etf_obs.empty:
                etref = model_df.loc[etf_obs.index, "etref"]
                etf_et = etf_obs * etref
                ax.scatter(
                    etf_obs.index,
                    etf_et,
                    s=8,
                    marker="^",
                    c="green",
                    alpha=0.6,
                    label="Landsat ETf x ETo",
                )

        ax.set_ylabel("ET (mm/day)")
        ax.set_title(site)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def fig_summary_table(results_dirs: dict[str, str], output_path: str):
    """Figure 6: Summary table as a figure for journal submission."""
    rows = []
    for exp_name, res_dir in results_dirs.items():
        try:
            summary = load_evaluation_summary(res_dir)
        except FileNotFoundError:
            continue

        row = {
            "Experiment": exp_name,
            "N sites": len(summary),
            "Mean RMSE": f"{summary['rmse'].mean():.2f}",
            "Med RMSE": f"{summary['rmse'].median():.2f}",
            "Mean R2": f"{summary['r2'].mean():.3f}",
            "Med R2": f"{summary['r2'].median():.3f}",
            "Mean KGE": f"{summary['kge'].mean():.3f}" if "kge" in summary else "—",
            "Mean Bias": f"{summary['bias'].mean():.2f}",
        }
        rows.append(row)

    if not rows:
        print("  No data for summary table")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", weight="bold")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_figures(
    shapefile: str,
    results_dirs: dict[str, str],
    output_dir: str,
    fig_format: str = "pdf",
    representative_sites: list[str] | None = None,
):
    """Generate the complete publication figure set."""
    os.makedirs(output_dir, exist_ok=True)

    # Try to load any available summary for the map
    summary_df = None
    for res_dir in results_dirs.values():
        try:
            summary_df = load_evaluation_summary(res_dir)
            break
        except FileNotFoundError:
            continue

    # Per-site (network, et_col) truth sources from the cohort shapefile so
    # figures resolve the same flux files/columns as evaluate.py
    flux_sources = {}
    if os.path.exists(shapefile):
        flux_sources = load_flux_sources(shapefile)

    print("Generating figures...")

    # Figure 1: Site map
    print("  Fig 1: Site map")
    fig_site_map(shapefile, summary_df, os.path.join(output_dir, f"fig1_site_map.{fig_format}"))

    # Figure 2: Scatter plots
    print("  Fig 2: Scatter plots")
    fig_scatter(results_dirs, os.path.join(output_dir, f"fig2_scatter.{fig_format}"), flux_sources)

    # Figure 3: Metric distributions
    print("  Fig 3: Metric distributions")
    fig_metric_distributions(results_dirs, os.path.join(output_dir, f"fig3_metrics.{fig_format}"))

    # Figure 4: Head-to-head (A vs C)
    exp_names = list(results_dirs.keys())
    if len(exp_names) >= 2:
        print("  Fig 4: Head-to-head")
        fig_head_to_head(
            results_dirs[exp_names[0]],
            results_dirs[exp_names[-1]],
            os.path.join(output_dir, f"fig4_head_to_head.{fig_format}"),
        )

    # Figure 5: Time series
    if representative_sites:
        print("  Fig 5: Time series")
        first_res_dir = list(results_dirs.values())[0]
        fig_time_series(
            first_res_dir,
            representative_sites,
            os.path.join(output_dir, f"fig5_timeseries.{fig_format}"),
            flux_sources,
        )

    # Figure 6: Summary table
    print("  Fig 6: Summary table")
    fig_summary_table(results_dirs, os.path.join(output_dir, f"fig6_summary.{fig_format}"))

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument(
        "--results-a",
        type=str,
        default=None,
        help="Results directory for Experiment A (Landsat only)",
    )
    parser.add_argument(
        "--results-b",
        type=str,
        default=None,
        help="Results directory for Experiment B (ECOSTRESS only)",
    )
    parser.add_argument(
        "--results-c",
        type=str,
        default=None,
        help="Results directory for Experiment C (combined)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "svg", "png"],
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated representative site IDs for time series",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    project_data = "/data/ssd1/swim/6_Flux_International/data"
    shapefile_ = os.path.join(project_data, "gis/flux_intl_buffers_150m_06JAN2026.shp")

    # Build results directories dict
    results = {}
    if args.results_a:
        results["A: Landsat PT-JPL"] = args.results_a
    if args.results_b:
        results["B: ECOSTRESS PT-JPL"] = args.results_b
    if args.results_c:
        results["C: Combined"] = args.results_c

    if not results:
        # Default: look for results in standard location
        ws = "/data/ssd1/swim/6_Flux_International"
        default = os.path.join(ws, "results")
        if os.path.isdir(default):
            results["Baseline"] = default

    if not results:
        print("No results directories found. Use --results-a, --results-b, --results-c.")
        exit(1)

    out_dir = args.output_dir or os.path.join(str(project_dir), "figures")
    rep_sites = args.sites.split(",") if args.sites else None

    generate_all_figures(
        shapefile=shapefile_,
        results_dirs=results,
        output_dir=out_dir,
        fig_format=args.format,
        representative_sites=rep_sites,
    )

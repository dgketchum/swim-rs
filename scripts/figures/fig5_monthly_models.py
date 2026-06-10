"""
Figure 5: Monthly ET Performance by Model (KEY FIGURE)

Three-panel dot plot comparing SWIM against 6 OpenET models + ensemble
on the 33-site matched monthly cohort from Ex5 Run 13 (MAD ensemble).

(a) Median R²
(b) Mean RMSE (mm/month)
(c) Mean Bias (mm/month)

Usage:
    python paper/figures/fig5_monthly_models.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

MONTHLY_CSV = "/data/ssd1/swim/5_Flux_Ensemble/results/evaluation_monthly_metrics.csv"

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig5_monthly_models.png"
OUT_PDF = OUT_DIR / "fig5_monthly_models.pdf"

# Model display order (SWIM first, then ensemble, then individuals by performance)
MODELS = ["swim", "ensemble", "ptjpl", "sims", "disalexi", "eemetric", "ssebop", "geesebal"]

MODEL_LABELS = {
    "swim": "SWIM",
    "ensemble": "Ensemble",
    "ptjpl": "PT-JPL",
    "sims": "SIMS",
    "disalexi": "DisALEXI",
    "eemetric": "eeMETRIC",
    "ssebop": "SSEBop",
    "geesebal": "geeSEBAL",
}

MODEL_COLORS = {
    "swim": "#4C72B0",
    "ensemble": "#222222",
    "ptjpl": "#DD8452",
    "sims": "#55A868",
    "disalexi": "#C44E52",
    "eemetric": "#8172B2",
    "ssebop": "#CCB974",
    "geesebal": "#64B5CD",
}


def bootstrap_ci(data, stat_func, n_boot=2000, ci=95):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(42)
    boot = np.array(
        [stat_func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return lo, hi


def main():
    df = pd.read_csv(MONTHLY_CSV)

    # Filter to matched cohort (sites with ensemble data)
    matched = df.dropna(subset=["r2_ensemble"]).copy()
    n_sites = len(matched)
    print(f"Matched monthly cohort: {n_sites} sites")

    # Compute statistics per model
    stats = {}
    for model in MODELS:
        r2_col = f"r2_{model}"
        rmse_col = f"rmse_{model}"
        bias_col = f"bias_{model}"

        r2_vals = matched[r2_col].dropna().values
        rmse_vals = matched[rmse_col].dropna().values
        bias_vals = matched[bias_col].dropna().values

        n = len(r2_vals)
        if n < 5:
            continue

        med_r2 = np.median(r2_vals)
        mean_rmse = np.mean(rmse_vals)
        mean_bias = np.mean(bias_vals)

        r2_ci = bootstrap_ci(r2_vals, np.median)
        rmse_ci = bootstrap_ci(rmse_vals, np.mean)
        bias_ci = bootstrap_ci(bias_vals, np.mean)

        stats[model] = {
            "n": n,
            "med_r2": med_r2,
            "r2_ci": r2_ci,
            "mean_rmse": mean_rmse,
            "rmse_ci": rmse_ci,
            "mean_bias": mean_bias,
            "bias_ci": bias_ci,
        }

    models_present = [m for m in MODELS if m in stats]

    fig, (ax_r2, ax_rmse, ax_bias) = plt.subplots(1, 3, figsize=(14, 4.5))
    # workaround: use fig.axes
    ax_r2, ax_rmse, ax_bias = fig.axes[0], fig.axes[1], fig.axes[2]

    y_pos = np.arange(len(models_present))
    labels = [MODEL_LABELS[m] for m in models_present]
    colors = [MODEL_COLORS[m] for m in models_present]

    # --- Panel (a): Median R² ---
    vals = [stats[m]["med_r2"] for m in models_present]
    ci_lo = [stats[m]["med_r2"] - stats[m]["r2_ci"][0] for m in models_present]
    ci_hi = [stats[m]["r2_ci"][1] - stats[m]["med_r2"] for m in models_present]

    ax_r2.barh(
        y_pos,
        vals,
        xerr=[ci_lo, ci_hi],
        color=colors,
        height=0.6,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"lw": 1.0},
    )
    ax_r2.set_yticks(y_pos)
    ax_r2.set_yticklabels(labels)
    ax_r2.set_xlabel("Median R²")
    ax_r2.set_title("(a) Median R²", loc="left", fontweight="bold")
    ax_r2.invert_yaxis()
    ax_r2.set_xlim(0.4, 1.0)

    # Annotate values near bar base
    for i, v in enumerate(vals):
        ax_r2.text(
            0.42,
            i,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="white" if v > 0.6 else "black",
        )

    # --- Panel (b): Mean RMSE ---
    vals = [stats[m]["mean_rmse"] for m in models_present]
    ci_lo = [stats[m]["mean_rmse"] - stats[m]["rmse_ci"][0] for m in models_present]
    ci_hi = [stats[m]["rmse_ci"][1] - stats[m]["mean_rmse"] for m in models_present]

    ax_rmse.barh(
        y_pos,
        vals,
        xerr=[ci_lo, ci_hi],
        color=colors,
        height=0.6,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"lw": 1.0},
    )
    ax_rmse.set_yticks(y_pos)
    ax_rmse.set_yticklabels([])
    ax_rmse.set_xlabel("Mean RMSE (mm/month)")
    ax_rmse.set_title("(b) Mean RMSE", loc="left", fontweight="bold")
    ax_rmse.invert_yaxis()

    for i, v in enumerate(vals):
        ax_rmse.text(
            1.0, i, f"{v:.1f}", va="center", ha="left", fontsize=8, fontweight="bold", color="white"
        )

    # --- Panel (c): Mean Bias ---
    vals = [stats[m]["mean_bias"] for m in models_present]
    ci_lo = [stats[m]["mean_bias"] - stats[m]["bias_ci"][0] for m in models_present]
    ci_hi = [stats[m]["bias_ci"][1] - stats[m]["mean_bias"] for m in models_present]

    ax_bias.barh(
        y_pos,
        vals,
        xerr=[ci_lo, ci_hi],
        color=colors,
        height=0.6,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"lw": 1.0},
    )
    ax_bias.set_yticks(y_pos)
    ax_bias.set_yticklabels([])
    ax_bias.set_xlabel("Mean Bias (mm/month)")
    ax_bias.set_title("(c) Mean Bias", loc="left", fontweight="bold")
    ax_bias.invert_yaxis()
    ax_bias.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)

    # Compute actual whisker extents (bar end + upper CI) to set safe axis limits
    whisker_lo = min(v - lo for v, lo in zip(vals, ci_lo))
    whisker_hi = max(v + hi for v, hi in zip(vals, ci_hi))
    margin = (whisker_hi - whisker_lo) * 0.35
    ax_bias.set_xlim(whisker_lo - 2, whisker_hi + margin)
    x_right = whisker_hi + margin - 0.3

    for i, v in enumerate(vals):
        ax_bias.text(
            x_right, i, f"{v:+.1f}", va="center", ha="right", fontsize=8, fontweight="bold"
        )

    # Add n annotation
    ax_r2.text(
        0.98,
        0.02,
        f"n = {n_sites} sites",
        transform=ax_r2.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    # Print table for reference
    print(f"\n{'Model':12s} {'n':>3s} {'Med R²':>8s} {'RMSE':>8s} {'Bias':>8s}")
    for m in models_present:
        s = stats[m]
        print(
            f"{MODEL_LABELS[m]:12s} {s['n']:3d} {s['med_r2']:8.3f} {s['mean_rmse']:8.1f} {s['mean_bias']:+8.1f}"
        )


if __name__ == "__main__":
    main()

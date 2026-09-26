"""Field-specific accuracy of simulated vs metered applied water (Example 7).

The pooled field-year correlation mixes two different things: whether the model
ranks *different fields* correctly (between-field) and whether it tracks a *given
field* through time (within-field). For applied-water validation we care about the
latter, plus whether each field's total water budget lands on the 1:1 line.

This reads ``results/applied_<label>/per_field_year.csv`` and produces:

  * Scatter A -- one point per field, total applied VOLUME (acre-ft) summed over the
    record: does the model reproduce each field's total water budget?
  * Scatter B -- one point per field-year, annual applied DEPTH (mm).
  * Scatter C -- one point per field-year, DEPTH ANOMALY (field mean removed from
    both obs and sim): the pure within-field temporal signal.

plus a printed decomposition: between-field r, within-field r, per-field bias
distribution (fraction of fields within +/-10/20/30 %), and per-field temporal r.

    uv run python examples/7_Applied_Water/field_accuracy.py --label calibrated
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ex7_paths  # noqa: E402

MM_PER_FT = 304.8


def _fit(o, s):
    """Regression + skill stats for paired arrays."""
    o = np.asarray(o, float)
    s = np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    if len(o) < 3:
        return {"n": int(len(o))}
    r = float(np.corrcoef(o, s)[0, 1])
    slope, intercept = np.polyfit(o, s, 1)
    ss_res = float(np.sum((o - s) ** 2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    return {
        "n": int(len(o)),
        "r": round(r, 3),
        "r2": round(r * r, 3),
        "nse_1to1": round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else np.nan,
        "slope": round(float(slope), 3),
        "bias_pct": round(100 * (s.mean() - o.mean()) / o.mean(), 1),
        "rmse": round(float(np.sqrt(np.mean((o - s) ** 2))), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="project TOML (default: this example's)")
    ap.add_argument("--label", default=ex7_paths.LOCAL_LABEL)
    ap.add_argument("--min-years", type=int, default=4, help="min years for per-field temporal r")
    args = ap.parse_args()

    d = ex7_paths.eval_dir(args.label, args.config)
    p = ex7_paths.drop_excluded_field_years(pd.read_csv(d / "per_field_year.csv"))
    p = p[p.metered_depth_mm > 0].copy()  # irrigated paired field-years
    p["sim_vol_af"] = p.sim_applied_mm / MM_PER_FT * p.acres
    p["metered_vol_af"] = p.metered_volume_af

    # ---- per-field aggregates (each point a field) ----
    g = p.groupby("site_id")
    fields = pd.DataFrame(
        {
            "n_years": g.year.nunique(),
            "basin": g.basin.first(),
            "metered_mean_mm": g.metered_depth_mm.mean(),
            "sim_mean_mm": g.sim_applied_mm.mean(),
            "metered_vol_total_af": g.metered_vol_af.sum(),
            "sim_vol_total_af": g.sim_vol_af.sum(),
        }
    )
    fields["field_bias_pct"] = (
        100 * (fields.sim_mean_mm - fields.metered_mean_mm) / fields.metered_mean_mm
    )

    # ---- within-field temporal signal ----
    p["obs_anom"] = p.metered_depth_mm - g.metered_depth_mm.transform("mean")
    p["sim_anom"] = p.sim_applied_mm - g.sim_applied_mm.transform("mean")
    temporal_r = []
    for _, sub in g:
        if (
            sub.shape[0] >= args.min_years
            and sub.metered_depth_mm.std() > 0
            and sub.sim_applied_mm.std() > 0
        ):
            temporal_r.append(float(np.corrcoef(sub.metered_depth_mm, sub.sim_applied_mm)[0, 1]))
    temporal_r = np.array(temporal_r)

    stats = {
        "label": args.label,
        "n_fields": int(len(fields)),
        "n_field_years": int(len(p)),
        "per_field_volume_total": _fit(fields.metered_vol_total_af, fields.sim_vol_total_af),
        "per_field_mean_depth": _fit(fields.metered_mean_mm, fields.sim_mean_mm),
        "pooled_field_year_depth": _fit(p.metered_depth_mm, p.sim_applied_mm),
        "within_field_anomaly_depth": _fit(p.obs_anom, p.sim_anom),
        "per_field_bias_pct": {
            "median": round(float(fields.field_bias_pct.median()), 1),
            "within_10pct": round(float((fields.field_bias_pct.abs() <= 10).mean()), 3),
            "within_20pct": round(float((fields.field_bias_pct.abs() <= 20).mean()), 3),
            "within_30pct": round(float((fields.field_bias_pct.abs() <= 30).mean()), 3),
        },
        "per_field_temporal_r": {
            "n_fields": int(len(temporal_r)),
            "median": round(float(np.median(temporal_r)), 3),
            "q25": round(float(np.percentile(temporal_r, 25)), 3),
            "q75": round(float(np.percentile(temporal_r, 75)), 3),
            "frac_positive": round(float((temporal_r > 0).mean()), 3),
            "frac_gt_0.5": round(float((temporal_r > 0.5).mean()), 3),
        },
    }
    (d / "field_accuracy_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------ figure
    colors = {"SLV": "#1f77b4", "ESPA": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A: per-field total volume
    ax = axes[0]
    for b, sub in fields.groupby("basin"):
        ax.scatter(
            sub.metered_vol_total_af,
            sub.sim_vol_total_af,
            s=30,
            alpha=0.7,
            color=colors.get(b, "gray"),
            label=b,
            edgecolor="k",
            linewidth=0.3,
        )
    lim = [
        0,
        float(np.nanmax([fields.metered_vol_total_af.max(), fields.sim_vol_total_af.max()])) * 1.05,
    ]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(
        xlim=lim,
        ylim=lim,
        xlabel="Metered total volume (acre-ft)",
        ylabel="Simulated total volume (acre-ft)",
        title="A. Per field (record total)",
    )
    s = stats["per_field_volume_total"]
    ax.text(
        0.05,
        0.95,
        f"n={s['n']} fields\nr={s['r']}  r$^2$={s['r2']}\nbias={s['bias_pct']}%",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")

    # B: per-field-year depth
    ax = axes[1]
    for b, sub in p.groupby("basin"):
        ax.scatter(
            sub.metered_depth_mm,
            sub.sim_applied_mm,
            s=12,
            alpha=0.4,
            color=colors.get(b, "gray"),
            label=b,
        )
    lim = [0, float(np.nanmax([p.metered_depth_mm.max(), p.sim_applied_mm.max()])) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(
        xlim=lim,
        ylim=lim,
        xlabel="Metered depth (mm/yr)",
        ylabel="Simulated depth (mm/yr)",
        title="B. Per field-year (annual)",
    )
    s = stats["pooled_field_year_depth"]
    ax.text(
        0.05,
        0.95,
        f"n={s['n']} field-yrs\nr={s['r']}  r$^2$={s['r2']}\nbias={s['bias_pct']}%",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")

    # C: within-field anomaly
    ax = axes[2]
    for b, sub in p.groupby("basin"):
        ax.scatter(
            sub.obs_anom, sub.sim_anom, s=12, alpha=0.4, color=colors.get(b, "gray"), label=b
        )
    lo = float(np.nanmin([p.obs_anom.min(), p.sim_anom.min()])) * 1.05
    hi = float(np.nanmax([p.obs_anom.max(), p.sim_anom.max()])) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set(
        xlim=[lo, hi],
        ylim=[lo, hi],
        xlabel="Metered anomaly (mm/yr)",
        ylabel="Simulated anomaly (mm/yr)",
        title="C. Within-field (field mean removed)",
    )
    s = stats["within_field_anomaly_depth"]
    ax.text(
        0.05,
        0.95,
        f"n={s['n']} field-yrs\nr={s['r']}  r$^2$={s['r2']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"SWIM-RS applied-water field accuracy -- {args.label}", fontsize=13)
    fig.tight_layout()
    out = d / "field_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Figure 6: metered applied water and the whole-field bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cohorts import EXPECTED
from .manifest import (
    MANIFEST,
    BuildError,
    require_columns,
    require_count,
    require_unique,
    sha256,
    write_table,
)
from .paths import E3_LOCAL, E3_TRANSFER, ex7_paths

# --------------------------------------------------------------------------
# Figure 6 -- metered applied water
# --------------------------------------------------------------------------

E3_TREATMENTS = {
    "local_calibration": (E3_LOCAL, "E3 local satellite calibration (e7cal)"),
    "e1_irrigated_transfer": (
        E3_TRANSFER,
        "fixed E1-derived irrigated parameter set, forward run, no local calibration",
    ),
}


def _load_e3_paths() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, (d, _) in E3_TREATMENTS.items():
        p = d / "per_field_year.csv"
        if not p.exists():
            raise BuildError(f"fig06 source missing: {p}")
        df = pd.read_csv(p)
        require_columns(
            df,
            [
                "site_id",
                "year",
                "metered_depth_mm",
                "sim_applied_mm",
                "sim_et_mm",
                "acres",
                "crop",
                "basin",
            ],
            f"E3 {name}",
        )
        df = df[df["site_id"].astype(str).str.startswith("SLV_")].copy()
        # Explicit exclusion (not a silent filter): the evaluator drops these keys at
        # scoring, so an input that carries them was scored on all 408 field-years.
        stale = len(df) - len(ex7_paths.drop_excluded_field_years(df))
        if stale:
            raise BuildError(
                f"fig06: {p} still carries {stale} of ex7_paths.EXCLUDED_FIELD_YEARS; "
                "re-score it with evaluate_applied_water.py --rescore"
            )
        require_unique(df, ["site_id", "year"], f"E3 {name} SLV")
        df = df.sort_values(["site_id", "year"]).reset_index(drop=True)
        frames[name] = df
    keys = [set(zip(f["site_id"], f["year"])) for f in frames.values()]
    if keys[0] != keys[1]:
        raise BuildError("fig06: (site_id, year) keys differ between local and transfer paths")
    for name, f in frames.items():
        require_count(len(f), EXPECTED["E3_field_years"], f"fig06 {name} field-years")
        require_count(f["site_id"].nunique(), EXPECTED["E3_fields"], f"fig06 {name} fields")
    a, b = frames["local_calibration"], frames["e1_irrigated_transfer"]
    if not np.allclose(a["metered_depth_mm"].values, b["metered_depth_mm"].values):
        raise BuildError("fig06: metered truth differs between the two evaluator outputs")
    return frames


def build_fig06() -> pd.DataFrame:
    frames = _load_e3_paths()
    blocks = []
    for name, df in frames.items():
        d = df.copy()
        d["field_mean_metered_mm"] = d.groupby("site_id")["metered_depth_mm"].transform("mean")
        d["field_mean_sim_mm"] = d.groupby("site_id")["sim_applied_mm"].transform("mean")
        d["anomaly_metered_mm"] = d["metered_depth_mm"] - d["field_mean_metered_mm"]
        d["anomaly_sim_mm"] = d["sim_applied_mm"] - d["field_mean_sim_mm"]
        d["treatment"] = name
        d["treatment_provenance"] = E3_TREATMENTS[name][1]
        d["experiment"] = "E3"
        d["legacy_prefix"] = "e4_"
        blocks.append(d)
    fy = pd.concat(blocks, ignore_index=True)
    cols = [
        "experiment",
        "legacy_prefix",
        "site_id",
        "year",
        "treatment",
        "treatment_provenance",
        "crop",
        "basin",
        "acres",
        "metered_depth_mm",
        "sim_applied_mm",
        "sim_et_mm",
        "field_mean_metered_mm",
        "field_mean_sim_mm",
        "anomaly_metered_mm",
        "anomaly_sim_mm",
    ]
    fy = fy[cols].sort_values(["treatment", "site_id", "year"])
    n_fy = write_table(fy, "fig06_field_years.csv")
    require_count(n_fy, EXPECTED["E3_field_years"] * 2, "fig06 field-year rows")

    summaries = []
    for name, df in frames.items():
        g = df.groupby("site_id")
        s = pd.DataFrame(
            {
                "n_years": g.size(),
                "mean_metered_mm": g["metered_depth_mm"].mean(),
                "mean_sim_mm": g["sim_applied_mm"].mean(),
                "total_metered_mm": g["metered_depth_mm"].sum(),
                "total_sim_mm": g["sim_applied_mm"].sum(),
                "acres": g["acres"].first(),
                "crop": g["crop"].first(),
            }
        )
        s["field_bias_pct"] = (
            (s["total_sim_mm"] - s["total_metered_mm"]) / s["total_metered_mm"] * 100.0
        )
        s["within_20pct"] = s["field_bias_pct"].abs() <= 20.0
        s["field_temporal_r"] = g.apply(
            lambda x: np.corrcoef(x["metered_depth_mm"], x["sim_applied_mm"])[0, 1]
            if len(x) > 2
            else np.nan,
            include_groups=False,
        )
        s = s.reset_index()
        s["treatment"] = name
        s["treatment_provenance"] = E3_TREATMENTS[name][1]
        s["experiment"] = "E3"
        s["legacy_prefix"] = "e4_"
        summaries.append(s)
    fs = pd.concat(summaries, ignore_index=True)
    n_fs = write_table(fs, "fig06_field_summaries.csv")
    require_count(n_fs, EXPECTED["E3_fields"] * 2, "fig06 field-summary rows")

    srcs = {
        "applied_calibrated_per_field_year": E3_LOCAL / "per_field_year.csv",
        f"applied_{ex7_paths.TRANSFER_LABEL}_per_field_year": E3_TRANSFER / "per_field_year.csv",
    }
    base = {
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in srcs.items()},
        "experiment_mapping": {"E3": "legacy e4_*"},
        "cohort_key": "(site_id, year)",
        "inclusion_rule": (
            f"{EXPECTED['E3_fields']} San Luis Valley metered fields (site_id prefix SLV_) and "
            f"their {EXPECTED['E3_field_years']} paired field-years with a positive recorded "
            "pumping volume, 2011-2021, excluding the field-years in "
            "ex7_paths.EXCLUDED_FIELD_YEARS (2019 RGDSS parcel-well links that resolve to "
            "another cohort field's well). The local and "
            "transfer evaluator outputs were asserted to carry identical (site_id, year) "
            "keys and identical metered truth before joining. metered_truth.csv is never "
            "read directly. The excluded ESPA basin and the 10 rainfed control fields are "
            "not part of this package."
        ),
        "temporal_support_rule": "Annual: model-generated daily irrigation summed by calendar year and paired with positive meter observations on identical field-year keys.",
        "units": {
            "metered_depth_mm": "mm year-1",
            "sim_applied_mm": "mm year-1 gross applied water",
            "sim_et_mm": "mm year-1",
            "acres": "acres",
            "field_bias_pct": "percent",
        },
        "display_transformations": [
            "restricted to SLV_ fields",
            "within-field anomalies computed by removing each field's record mean from both the metered and simulated annual depths",
            "field_bias_pct is a record-total bias, (sum(sim) - sum(metered)) / sum(metered) * 100",
        ],
        "deterministic_seed": None,
        "configured_counts": {"E3": EXPECTED["E3_fields"]},
        "evaluated_counts": {
            "fields": EXPECTED["E3_fields"],
            "field_years": EXPECTED["E3_field_years"],
        },
        "excluded_field_years": [
            {"site_id": k[0], "year": k[1], "reason": v}
            for k, v in sorted(ex7_paths.EXCLUDED_FIELD_YEARS.items())
        ],
        "independent_unit": "field (not field-year)",
    }
    MANIFEST.add("fig06_field_years.csv", rows=n_fy, **base)
    MANIFEST.add("fig06_field_summaries.csv", rows=n_fs, **base)
    print(
        f"  fig06: field-years {n_fy} rows, field summaries {n_fs} rows "
        f"({EXPECTED['E3_fields']} fields, {EXPECTED['E3_field_years']} field-years)"
    )
    return fy


# ---- panel (d): prespecified whole-field bootstrap -------------------------

BOOTSTRAP_STATS = {
    "between_field_mean_depth_r": {
        "definition": "Pearson r between per-field record-mean metered depth and per-field record-mean simulated depth across the resampled fields",
        "group": "cross-field magnitude",
        "favorable_direction": "positive favours the transfer path",
    },
    "pooled_annual_depth_r": {
        "definition": "Pearson r between metered and simulated annual depths pooled over all field-years in the resample",
        "group": "cross-field magnitude",
        "favorable_direction": "positive favours the transfer path",
    },
    "pooled_rmse_mm": {
        "definition": "sqrt(mean((sim - metered)^2)) over pooled field-years, mm year-1",
        "group": "cross-field magnitude",
        "favorable_direction": "negative favours the transfer path",
    },
    "within_field_anomaly_r": {
        "definition": "Pearson r between metered and simulated within-field anomalies (each field's record mean removed within the resample) pooled over field-years",
        "group": "within-field response",
        "favorable_direction": "positive favours the transfer path",
    },
    "within_field_anomaly_slope": {
        "definition": "ordinary least-squares slope of simulated anomaly on metered anomaly, pooled over field-years",
        "group": "within-field response",
        "favorable_direction": "positive moves the slope toward one; sign retained",
    },
    "pooled_abs_bias_pct": {
        "definition": "absolute value of pooled percent bias, |sum(sim) - sum(metered)| / sum(metered) * 100",
        "group": "bias and coverage",
        "favorable_direction": "negative favours the transfer path",
    },
    "median_abs_field_bias_pct": {
        "definition": "median across resampled fields of |record-total field bias| in percent",
        "group": "bias and coverage",
        "favorable_direction": "negative favours the transfer path",
    },
    "fraction_within_20pct": {
        "definition": "fraction of resampled fields whose record-total percent bias lies within +/-20%",
        "group": "bias and coverage",
        "favorable_direction": "positive favours the transfer path",
    },
    "median_field_temporal_r": {
        "definition": "median across resampled fields of the within-field Pearson r between annual metered and simulated depths (fields with >2 years only)",
        "group": "within-field response",
        "favorable_direction": "positive favours the transfer path",
    },
}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float((xc * yc).sum() / den) if den > 0 else np.nan


def _e3_statistics(obs: np.ndarray, sim: np.ndarray, field: np.ndarray) -> dict:
    """Nine prespecified applied-water statistics.

    ``field`` may be labels of any dtype; it is factorised to contiguous
    integers so that a bootstrap resample can relabel a repeated field as two
    distinct inferential units.
    """
    lab, _ = pd.factorize(field, sort=False)
    k = lab.max() + 1
    n = np.bincount(lab, minlength=k).astype(float)
    so = np.bincount(lab, weights=obs, minlength=k)
    ss = np.bincount(lab, weights=sim, minlength=k)
    soo = np.bincount(lab, weights=obs * obs, minlength=k)
    sss = np.bincount(lab, weights=sim * sim, minlength=k)
    sos = np.bincount(lab, weights=obs * sim, minlength=k)
    mo, ms = so / n, ss / n

    out = {
        "between_field_mean_depth_r": _pearson(mo, ms),
        "pooled_annual_depth_r": _pearson(obs, sim),
        "pooled_rmse_mm": float(np.sqrt(np.mean((sim - obs) ** 2))),
    }
    ao = obs - mo[lab]
    as_ = sim - ms[lab]
    out["within_field_anomaly_r"] = _pearson(ao, as_)
    den = (ao * ao).sum()
    out["within_field_anomaly_slope"] = float((ao * as_).sum() / den) if den > 0 else np.nan

    fb = (ss - so) / so * 100.0
    out["pooled_abs_bias_pct"] = float(abs((sim.sum() - obs.sum()) / obs.sum() * 100.0))
    out["median_abs_field_bias_pct"] = float(np.median(np.abs(fb)))
    out["fraction_within_20pct"] = float(np.mean(np.abs(fb) <= 20.0))

    num = n * sos - so * ss
    dsq = (n * soo - so * so) * (n * sss - ss * ss)
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = np.where(dsq > 0, num / np.sqrt(dsq), np.nan)
    tr = np.where(n > 2, tr, np.nan)
    out["median_field_temporal_r"] = float(np.nanmedian(tr))
    return out


def build_fig06_bootstrap() -> None:
    frames = _load_e3_paths()
    a = frames["local_calibration"].set_index(["site_id", "year"]).sort_index()
    b = frames["e1_irrigated_transfer"].set_index(["site_id", "year"]).sort_index()
    if not a.index.equals(b.index):
        raise BuildError("fig06 bootstrap: index mismatch after alignment")

    obs = a["metered_depth_mm"].to_numpy(float)
    sim_local = a["sim_applied_mm"].to_numpy(float)
    sim_transfer = b["sim_applied_mm"].to_numpy(float)
    fields = a.index.get_level_values(0).to_numpy()
    uniq = np.array(sorted(set(fields)))
    require_count(len(uniq), EXPECTED["E3_fields"], "fig06 bootstrap fields")
    require_count(len(obs), EXPECTED["E3_field_years"], "fig06 bootstrap field-years")

    idx_by_field = {f: np.flatnonzero(fields == f) for f in uniq}

    point_local = _e3_statistics(obs, sim_local, fields)
    point_transfer = _e3_statistics(obs, sim_transfer, fields)
    point_delta = {k: point_transfer[k] - point_local[k] for k in BOOTSTRAP_STATS}

    n_boot, seed = 10000, 42
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot) for k in BOOTSTRAP_STATS}
    for i in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_field[f] for f in draw])
        # relabel so a field drawn twice contributes as two distinct fields
        lab = np.concatenate([np.full(len(idx_by_field[f]), j) for j, f in enumerate(draw)])
        sl = _e3_statistics(obs[take], sim_local[take], lab)
        st = _e3_statistics(obs[take], sim_transfer[take], lab)
        for k in BOOTSTRAP_STATS:
            reps[k][i] = st[k] - sl[k]

    rows = []
    for k, meta in BOOTSTRAP_STATS.items():
        r = reps[k]
        rows.append(
            {
                "experiment": "E3",
                "legacy_prefix": "e4_",
                "statistic": k,
                "statistic_group": meta["group"],
                "statistic_definition": meta["definition"],
                "favorable_direction": meta["favorable_direction"],
                "value_local_calibration": point_local[k],
                "value_e1_irrigated_transfer": point_transfer[k],
                "delta_transfer_minus_local": point_delta[k],
                "bootstrap_median_delta": float(np.median(r)),
                "ci95_lo": float(np.percentile(r, 2.5)),
                "ci95_hi": float(np.percentile(r, 97.5)),
                "excludes_zero": bool(np.percentile(r, 2.5) > 0 or np.percentile(r, 97.5) < 0),
                "n_resamples": n_boot,
                "seed": seed,
                "resample_unit": "whole field (all years retained per sampled field)",
                "n_fields": len(uniq),
                "n_field_years": len(obs),
            }
        )
    out = pd.DataFrame(rows)
    n = write_table(out, "fig06_bootstrap_effects.csv")

    MANIFEST.add(
        "fig06_bootstrap_effects.csv",
        rows=n,
        sources={
            "applied_calibrated_per_field_year": {
                "path": str(E3_LOCAL / "per_field_year.csv"),
                "sha256": sha256(E3_LOCAL / "per_field_year.csv"),
            },
            f"applied_{ex7_paths.TRANSFER_LABEL}_per_field_year": {
                "path": str(E3_TRANSFER / "per_field_year.csv"),
                "sha256": sha256(E3_TRANSFER / "per_field_year.csv"),
            },
        },
        experiment_mapping={"E3": "legacy e4_*"},
        cohort_key="(site_id, year); resample unit = site_id",
        inclusion_rule=(
            f"Identical {EXPECTED['E3_fields']}-field / {EXPECTED['E3_field_years']}-field-year "
            "SLV keys under both treatments (ex7_paths.EXCLUDED_FIELD_YEARS removed). This is a "
            "NEW comparison of local satellite calibration against the current E1-derived "
            "IRRIGATED parameter set. The archived applied_local_vs_transfer_run22 "
            "bootstrap compares local calibration with the SUPERSEDED pooled transfer "
            "vector and is deliberately not used."
        ),
        temporal_support_rule="Annual gross applied water, 2011-2021, positive meter records only.",
        units={
            "pooled_rmse_mm": "mm year-1",
            "pooled_abs_bias_pct": "percent",
            "median_abs_field_bias_pct": "percent",
            "fraction_within_20pct": "fraction",
            "correlations_and_slope": "dimensionless",
        },
        display_transformations=[
            "all statistics reported as transfer minus local with natural signs retained",
            "no metric was sign-flipped to make favourable results point the same way",
        ],
        deterministic_seed=seed,
        bootstrap=(
            "Whole-field resampling with replacement, all years retained for each sampled "
            "field, 10,000 resamples, numpy default_rng(42). A field drawn more than once "
            "is relabelled so that it contributes as separate fields to the field-level "
            "statistics. Design frozen before any interval was inspected."
        ),
        configured_counts={"E3": EXPECTED["E3_fields"]},
        evaluated_counts={
            "fields": EXPECTED["E3_fields"],
            "field_years": EXPECTED["E3_field_years"],
        },
        excluded_field_years=[
            {"site_id": k[0], "year": k[1], "reason": v}
            for k, v in sorted(ex7_paths.EXCLUDED_FIELD_YEARS.items())
        ],
        independent_unit="field",
    )
    print(f"  fig06 bootstrap: {n} statistics, 10,000 whole-field resamples, seed 42")
    for _, r in out.iterrows():
        print(
            f"    {r['statistic']:<32} delta {r['delta_transfer_minus_local']:+.4f} "
            f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]"
            f"{'  *' if r['excludes_zero'] else ''}"
        )

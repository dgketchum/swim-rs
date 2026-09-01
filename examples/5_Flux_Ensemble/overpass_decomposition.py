"""E1 temporal decomposition — strict consumer of the evaluator's paired record.

Decomposes the canonical daily flux evaluation into retrieval days (dates with
a finite raw May v2.1 ensemble_mean_3x3 capture before interpolation) and
between-retrieval days (paired dates whose OpenET value exists only through
the ETf-first temporal reconstruction). All observations, support classes, and
temporal classes come from the evaluator-owned paired-record artifact
(``evaluation_paired_daily_records.csv``, schema ``e1_openet_paired_daily/v1``)
written by ``evaluate.py``. This script performs no raw-source I/O, no ETo
loading, no benchmark reconstruction, and no cohort import from separate CSVs;
a missing, stale, malformed, or hash-mismatched parent bundle is a hard error
(rerun ``evaluate.py``), never a fallback.

Before any analysis, the parent-artifact identity gate proves the child starts
from the exact observations underlying the evaluator headline: every parent
artifact named in the evaluator's ``output_hashes`` is re-hashed, the record
counts are checked against the metadata contract, the all-days grouped point
estimates are recomputed from the records and required to match
``evaluation_grouped_daily_metrics.csv`` within 1e-12, and the evaluator
contrast rows are required to equal SWIM minus OpenET.

The primary temporal estimand is the cross-model support interaction

    (SWIM - OpenET | between_retrieval) - (SWIM - OpenET | retrieval)

computed on the common temporal cohort (sites with >= 10 paired dates in BOTH
temporal classes) from ONE shared whole-site bootstrap multiplicity matrix
across both models, every metric, and all three temporal partitions, so the
interaction is paired at the site-draw level.

Usage:
    uv run python overpass_decomposition.py \
        --evaluator-output-dir <dir with the evaluate.py daily volk bundle> \
        --output-dir <staging dir> \
        [--bootstrap-reps 10000] [--seed 42] [--legacy-site-products]
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    BENCHMARK_SOURCE_MACHINE_TOKENS,
    CONSTRUCTION_TOKENS,
    GROUPED_MODEL_ORDER,
    MIN_OBS_FOR_METRICS,
    PAIRED_RECORD_FILENAME,
    PAIRED_RECORD_SCHEMA_VERSION,
    POOLED_METRICS,
    TEMPORAL_CLASS_BETWEEN,
    TEMPORAL_CLASS_DEFINITION,
    TEMPORAL_CLASS_RETRIEVAL,
    TEMPORAL_INTERACTION_FORMULA,
    WEIGHTED_METRICS,
    GroupedEstimationError,
    grouped_point_estimates,
    paired_records_from_frame,
    read_paired_record_frame,
    site_secondary_metrics,
    temporal_decomposition,
    validate_paired_record_frame,
)

IDENTITY_TOL = 1e-12
MIN_PAIRED = MIN_OBS_FOR_METRICS

# Fixed canonical parent-bundle filenames (never the DIY-suffixed variants)
PARENT_METRICS = "evaluation_grouped_daily_metrics.csv"
PARENT_CONTRASTS = "evaluation_grouped_daily_contrasts.csv"
PARENT_METADATA = "evaluation_grouped_daily_metadata.json"

# Default temporal outputs
OUT_METRICS = "evaluation_temporal_grouped_metrics.csv"
OUT_CONTRASTS = "evaluation_temporal_grouped_contrasts.csv"
OUT_INTERACTIONS = "evaluation_temporal_interactions.csv"
OUT_ELIGIBILITY = "evaluation_temporal_site_eligibility.csv"
OUT_METADATA = "evaluation_temporal_metadata.json"

# Legacy compatibility products (transition-only, from the record, never
# from raw-source reconstruction). Subset names keep the historical labels:
# overpass == retrieval, non_overpass == between_retrieval.
LEGACY_SUBSETS = ["all_days", "overpass", "non_overpass"]
LEGACY_SUBSET_CLASS = {
    "all_days": None,
    "overpass": TEMPORAL_CLASS_RETRIEVAL,
    "non_overpass": TEMPORAL_CLASS_BETWEEN,
}
METRIC_KEYS = ["nse", "kge", "r", "rmse", "mbe"]
DELTA_METRICS = ["nse", "kge", "rmse", "abs_mbe"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo_dir):
    def run(args):
        return subprocess.run(
            args, cwd=repo_dir, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_paths": len(run(["git", "status", "--porcelain"]).splitlines()),
    }


def _write_atomic(write_fn, path):
    tmp = f"{path}.tmp"
    write_fn(tmp)
    os.replace(tmp, path)


class ParentBundleError(ValueError):
    """The evaluator bundle is absent, stale, malformed, or hash-mismatched."""


def _require(condition, message):
    if not condition:
        raise ParentBundleError(f"{message} — rerun evaluate.py (daily, openet_source=volk)")


def validate_parent_bundle(evaluator_output_dir):
    """Parent-artifact identity gate (never a reconstruction gate).

    Returns ``(record_frame, evaluator_metadata, gate_report)``. Any failure
    is a hard error naming the defect; there is no fallback input path.
    """
    parent = Path(evaluator_output_dir)
    paths = {
        "grouped_metrics": parent / PARENT_METRICS,
        "grouped_contrasts": parent / PARENT_CONTRASTS,
        "metadata": parent / PARENT_METADATA,
        "paired_records": parent / PAIRED_RECORD_FILENAME,
    }
    for name, p in paths.items():
        _require(p.is_file(), f"missing canonical parent artifact {name}: {p}")

    with open(paths["metadata"]) as f:
        meta = json.load(f)

    _require(meta.get("scale") == "daily", f"parent scale {meta.get('scale')!r} is not daily")
    _require(
        meta.get("openet_source") == "volk",
        f"parent openet_source {meta.get('openet_source')!r} is not the canonical volk source",
    )
    _require(
        meta.get("benchmark_source") == BENCHMARK_SOURCE_MACHINE_TOKENS["volk"],
        f"parent benchmark_source {meta.get('benchmark_source')!r} is not "
        f"{BENCHMARK_SOURCE_MACHINE_TOKENS['volk']}",
    )
    _require(
        meta.get("benchmark_construction") == CONSTRUCTION_TOKENS["daily"],
        f"parent benchmark_construction {meta.get('benchmark_construction')!r} is not "
        f"{CONSTRUCTION_TOKENS['daily']}",
    )
    _require(
        meta.get("bootstrap", {}).get("unit") == "site",
        "parent bootstrap unit is not whole-site",
    )
    for k in ("kge", "rmse", "mbe"):
        _require(k in meta.get("formulas", {}), f"parent metadata lacks the {k} formula")

    contract = meta.get("paired_record_contract")
    _require(contract is not None, "parent metadata has no paired_record_contract")
    _require(
        contract.get("schema_version") == PAIRED_RECORD_SCHEMA_VERSION,
        f"record schema {contract.get('schema_version')!r} is not {PAIRED_RECORD_SCHEMA_VERSION}",
    )
    _require(
        contract.get("filename") == PAIRED_RECORD_FILENAME,
        f"record filename {contract.get('filename')!r} is not canonical ({PAIRED_RECORD_FILENAME})",
    )

    # Re-hash every parent artifact named in output_hashes
    output_hashes = meta.get("output_hashes", {})
    _require(bool(output_hashes), "parent metadata has no output_hashes map")
    rehashed = {}
    for basename, expected in output_hashes.items():
        p = parent / basename
        _require(p.is_file(), f"artifact named in output_hashes is missing: {p}")
        actual = sha256_file(p)
        _require(
            actual == expected,
            f"hash mismatch for {basename}: metadata {expected} vs on-disk {actual}",
        )
        rehashed[basename] = actual
    _require(
        PAIRED_RECORD_FILENAME in output_hashes,
        "paired record is not covered by the parent output_hashes",
    )
    _require(
        contract.get("sha256") == output_hashes[PAIRED_RECORD_FILENAME],
        "paired_record_contract sha256 disagrees with output_hashes",
    )

    # Load, validate, and count the paired frame against the contract
    frame = read_paired_record_frame(paths["paired_records"])
    counts = validate_paired_record_frame(frame)
    for key in ("n_sites", "n_rows", "n_retrieval", "n_between_retrieval"):
        _require(
            counts[key] == contract.get(key),
            f"record {key}={counts[key]} != contract {contract.get(key)}",
        )
    _require(
        counts["support_class_counts"] == contract.get("support_class_counts"),
        f"record support_class_counts {counts['support_class_counts']} != "
        f"contract {contract.get('support_class_counts')}",
    )

    # Recompute the all-days grouped point estimates from the records and
    # require identity with the evaluator's grouped metrics table
    records = paired_records_from_frame(frame)
    est = grouped_point_estimates(records)
    gmetrics = pd.read_csv(paths["grouped_metrics"], float_precision="round_trip")
    gref = gmetrics.set_index(["aggregation", "model", "metric"])["estimate"]
    point_diffs = {}
    for key, val in est.items():
        _require(key in gref.index, f"evaluator grouped metrics lack estimand {key}")
        diff = abs(val - float(gref[key]))
        point_diffs["/".join(key)] = diff
        _require(
            diff <= IDENTITY_TOL,
            f"grouped point {key} recomputed from records differs by {diff!r} (> {IDENTITY_TOL})",
        )

    # Evaluator contrast rows must equal SWIM minus OpenET model rows
    gcontrasts = pd.read_csv(paths["grouped_contrasts"], float_precision="round_trip")
    contrast_diffs = {}
    for _, row in gcontrasts.iterrows():
        agg, k = row["aggregation"], row["metric"]
        expected = float(gref[(agg, "swim", k)]) - float(gref[(agg, "openet_ensemble", k)])
        diff = abs(float(row["estimate"]) - expected)
        contrast_diffs[f"{agg}/{k}"] = diff
        _require(
            diff <= IDENTITY_TOL,
            f"evaluator contrast {agg}/{k} is not SWIM minus OpenET (diff {diff!r})",
        )

    gate_report = {
        "parent_dir": str(parent),
        "rehashed_artifacts": rehashed,
        "metadata_sha256": sha256_file(paths["metadata"]),
        "record_counts": counts,
        "grouped_point_identity_max_abs_diff": max(point_diffs.values()),
        "contrast_identity_max_abs_diff": max(contrast_diffs.values()),
        "identity_tolerance": IDENTITY_TOL,
    }
    return frame, meta, gate_report


# ---------------------------------------------------------------------------
# Legacy compatibility products (transition-only; record-derived, secondary)
# ---------------------------------------------------------------------------


def bootstrap_median_ci(deltas, reps, seed):
    """95% site-bootstrap CI of the median paired delta. Resamples sites."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    medians = np.median(deltas[idx], axis=1)
    return (
        float(np.median(deltas)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def _iqr(values):
    v = pd.Series(values).dropna()
    return float(v.quantile(0.25)), float(v.quantile(0.75))


def build_summary_row(subset, cohort, rows):
    """Cohort summary for one subset from eligible per-site metric rows."""
    days = rows["n_paired"].values
    q25, q75 = _iqr(days)
    out = {
        "subset": subset,
        "cohort": cohort,
        "n_sites": len(rows),
        "total_paired_site_days": int(days.sum()),
        "median_paired_days_per_site": float(np.median(days)),
        "iqr25_paired_days": q25,
        "iqr75_paired_days": q75,
    }
    for model in ["swim", "openet"]:
        for k in METRIC_KEYS:
            col = f"{k}_{model}"
            vals = rows[col]
            lo, hi = _iqr(vals)
            out[f"median_{col}"] = float(vals.median())
            out[f"iqr25_{col}"] = lo
            out[f"iqr75_{col}"] = hi
    return out


def legacy_site_metrics(frame):
    """Per-site per-subset legacy metric rows from the paired record only."""
    rows = []
    for fid, grp in frame.groupby("fid", sort=True):
        for subset in LEGACY_SUBSETS:
            cls = LEGACY_SUBSET_CLASS[subset]
            sdf = grp if cls is None else grp[grp["temporal_class"] == cls]
            n = len(sdf)
            row = {
                "fid": fid,
                "subset": subset,
                "n_paired": n,
                "first_date": sdf["date"].min().date().isoformat() if n else "",
                "last_date": sdf["date"].max().date().isoformat() if n else "",
                "eligible": n >= MIN_PAIRED,
                "exclusion_reason": "" if n >= MIN_PAIRED else f"n_paired={n} < {MIN_PAIRED}",
            }
            obs = sdf["flux_et_mm_d"].to_numpy()
            for model, col in [("swim", "swim_et_mm_d"), ("openet", "openet_et_mm_d")]:
                m = site_secondary_metrics(obs, sdf[col].to_numpy(), min_n=MIN_PAIRED)
                for k in METRIC_KEYS:
                    row[f"{k}_{model}"] = m[k]
            rows.append(row)
    return pd.DataFrame(rows)


def legacy_date_audit(frame):
    """Record-derivable audit counts per site.

    Calibration-capture auditing and raw support spans are deliberately gone:
    they need raw sources, and calibration flags never participate in the
    temporal classification (see the paired-record contract).
    """
    rows = []
    for fid, grp in frame.groupby("fid", sort=True):
        support = grp["openet_support_class"]
        n_over = int((support == "capture").sum())
        n_interp = int((support == "interpolated").sum())
        n_flat = int((support == "flat_fill").sum())
        rows.append(
            {
                "fid": fid,
                "n_paired_all_days": len(grp),
                "n_paired_overpass": n_over,
                "n_paired_non_overpass": n_interp + n_flat,
                "n_paired_interpolated": n_interp,
                "n_paired_flat_fill": n_flat,
                "first_paired_date": grp["date"].min().date().isoformat(),
                "last_paired_date": grp["date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def write_legacy_products(frame, out_dir, reps, seed):
    """Emit the former secondary site-level products from the record.

    Historical subset labels are kept (overpass == retrieval, non_overpass ==
    between_retrieval). Metric definitions are preserved exactly: per-site
    metrics carry signed MBE; only the paired-delta product uses |MBE|
    (labeled abs_mbe). Everything derives from the paired record — no
    reconstruction code exists in this script.
    """
    metrics_df = legacy_site_metrics(frame)
    audit_df = legacy_date_audit(frame)

    # Gate C equivalent: per-site union identity from record counts
    bad = audit_df[
        audit_df["n_paired_overpass"] + audit_df["n_paired_non_overpass"]
        != audit_df["n_paired_all_days"]
    ]
    if not bad.empty:
        raise GroupedEstimationError(f"legacy union violation at {list(bad['fid'])}")

    eligible = metrics_df[metrics_df["eligible"]]
    eligible_by_subset = {
        s: set(eligible.loc[eligible["subset"] == s, "fid"]) for s in LEGACY_SUBSETS
    }
    common_split = sorted(eligible_by_subset["overpass"] & eligible_by_subset["non_overpass"])

    summary_rows = []
    for subset in LEGACY_SUBSETS:
        sub_rows = eligible[eligible["subset"] == subset]
        summary_rows.append(build_summary_row(subset, "subset_eligible", sub_rows))
        summary_rows.append(
            build_summary_row(subset, "common_split", sub_rows[sub_rows["fid"].isin(common_split)])
        )
    summary_df = pd.DataFrame(summary_rows)

    fractions = {}
    for cohort, cohort_fids in [
        ("subset_eligible", sorted(eligible_by_subset["non_overpass"])),
        ("common_split", common_split),
    ]:
        a = audit_df[audit_df["fid"].isin(cohort_fids)]
        site_frac = a["n_paired_non_overpass"] / a["n_paired_all_days"]
        fractions[cohort] = {
            "site_median_non_overpass_fraction": float(site_frac.median()),
            "pooled_non_overpass_fraction": float(
                a["n_paired_non_overpass"].sum() / a["n_paired_all_days"].sum()
            ),
        }
    for cohort, vals in fractions.items():
        mask = summary_df["cohort"] == cohort
        for k, v in vals.items():
            summary_df.loc[mask, k] = v

    delta_rows = []
    for cohort, cohort_fids in [
        ("common_split", common_split),
        ("subset_eligible", None),
    ]:
        for subset in LEGACY_SUBSETS:
            sub = eligible[eligible["subset"] == subset]
            if cohort_fids is not None:
                sub = sub[sub["fid"].isin(cohort_fids)]
            deltas = {
                "nse": sub["nse_swim"] - sub["nse_openet"],
                "kge": sub["kge_swim"] - sub["kge_openet"],
                "rmse": sub["rmse_swim"] - sub["rmse_openet"],
                "abs_mbe": sub["mbe_swim"].abs() - sub["mbe_openet"].abs(),
            }
            for metric in DELTA_METRICS:
                med, lo, hi = bootstrap_median_ci(deltas[metric].values, reps, seed)
                delta_rows.append(
                    {
                        "metric": metric,
                        "subset": subset,
                        "cohort": cohort,
                        "n_sites": len(sub),
                        "median_delta_swim_minus_openet": med,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "seed": seed,
                        "n_resamples": reps,
                    }
                )
    deltas_df = pd.DataFrame(delta_rows)

    # Support contrast: per-model metric change non_overpass - overpass,
    # all five metrics, common_split cohort only
    by = metrics_df.set_index(["fid", "subset"])
    persite_rows, contrast_rows = [], []
    for fid in common_split:
        for model in ["swim", "openet"]:
            for k in METRIC_KEYS:
                o = by.loc[(fid, "overpass"), f"{k}_{model}"]
                nv = by.loc[(fid, "non_overpass"), f"{k}_{model}"]
                persite_rows.append(
                    {
                        "fid": fid,
                        "model": model,
                        "metric": k,
                        "overpass": o,
                        "non_overpass": nv,
                        "delta_non_minus_overpass": nv - o,
                    }
                )
    contrast_persite = pd.DataFrame(persite_rows)
    for model in ["swim", "openet"]:
        for k in METRIC_KEYS:
            sub = contrast_persite[
                (contrast_persite["model"] == model) & (contrast_persite["metric"] == k)
            ]
            med, lo, hi = bootstrap_median_ci(sub["delta_non_minus_overpass"].values, reps, seed)
            contrast_rows.append(
                {
                    "model": model,
                    "metric": k,
                    "cohort": "common_split",
                    "n_sites": len(sub),
                    "median_delta_non_minus_overpass": med,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "seed": seed,
                    "n_resamples": reps,
                }
            )
    contrast_df = pd.DataFrame(contrast_rows)

    metrics_out = metrics_df[
        ["fid", "subset", "n_paired", "first_date", "last_date", "eligible", "exclusion_reason"]
        + [f"{k}_{m}" for m in ["swim", "openet"] for k in METRIC_KEYS]
    ]
    written = {}
    for name, df in [
        ("overpass_split_metrics.csv", metrics_out),
        ("overpass_split_summary.csv", summary_df),
        ("overpass_split_paired_deltas.csv", deltas_df),
        ("overpass_date_audit.csv", audit_df),
        ("e2_temporal_support_contrast.csv", contrast_df),
        ("e2_temporal_support_contrast_persite.csv", contrast_persite),
    ]:
        path = os.path.join(out_dir, name)
        _write_atomic(lambda p, d=df: d.to_csv(p, index=False), path)
        written[name] = path

    legacy_meta = {
        "role": (
            "LEGACY/SECONDARY transition products derived from the paired-record "
            "artifact; the default temporal grouped outputs are the primary "
            "products. No raw-source reconstruction was performed."
        ),
        "record_schema": PAIRED_RECORD_SCHEMA_VERSION,
        "subset_labels": {
            "overpass": TEMPORAL_CLASS_RETRIEVAL,
            "non_overpass": TEMPORAL_CLASS_BETWEEN,
        },
        "definitions": {
            "min_paired_dates": MIN_PAIRED,
            "nse": "1 - SSE/SST; labeled NSE, never r2",
            "kge": "Gupta et al. 2009: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), "
            "alpha = std ratio (ddof=0), beta = mean ratio",
            "mbe": "mean(model - flux), mm/day, signed in per-site metrics",
            "abs_mbe": "|MBE_swim| - |MBE_openet| in the paired-delta product only",
            "bootstrap": "site resampling with replacement; per-(subset,cohort) "
            "index matrix drawn once from default_rng(seed) and shared across metrics",
        },
        "date_audit_scope": (
            "record-derivable counts only; calibration-capture counts, raw benchmark "
            "capture totals, support spans, and reconstruction identity errors are "
            "no longer audited here (they require raw sources, which this consumer "
            "does not read)"
        ),
        "cohort_fractions": fractions,
        "common_split_cohort": common_split,
        "n_common_split_sites": len(common_split),
        "seed": seed,
        "bootstrap_reps": reps,
    }
    path = os.path.join(out_dir, "overpass_split_metadata.json")
    _write_atomic(lambda p: Path(p).write_text(json.dumps(legacy_meta, indent=2)), path)
    written["overpass_split_metadata.json"] = path
    return written


def main():
    parser = argparse.ArgumentParser(
        description="E1 temporal decomposition from the evaluator paired-record bundle"
    )
    parser.add_argument(
        "--evaluator-output-dir",
        required=True,
        help="Directory holding the canonical daily volk evaluator bundle "
        "(grouped metrics/contrasts/metadata + paired daily records)",
    )
    parser.add_argument("--output-dir", required=True, help="Working output directory")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy-site-products",
        action="store_true",
        help="Also emit the former secondary site-level products "
        "(overpass_split_*, e2_temporal_support_contrast*) from the record — "
        "transition/compatibility only",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Validating parent bundle: {args.evaluator_output_dir}")
    frame, parent_meta, gate_report = validate_parent_bundle(args.evaluator_output_dir)
    print(
        f"Parent identity gate passed: {gate_report['record_counts']['n_sites']} sites, "
        f"{gate_report['record_counts']['n_rows']:,} rows; grouped-point max |diff| "
        f"{gate_report['grouped_point_identity_max_abs_diff']:.2e} "
        f"(tolerance {IDENTITY_TOL:.0e})"
    )

    decomp = temporal_decomposition(frame, reps=args.bootstrap_reps, seed=args.seed)
    n_common = len(decomp.common_cohort)
    excluded = sorted(set(frame["fid"].unique()) - set(decomp.common_cohort))
    print(f"Common temporal cohort: {n_common} sites (excluded: {excluded or 'none'})")
    print(
        "Class rows (common cohort): "
        + ", ".join(f"{k}={v:,}" for k, v in decomp.class_counts.items())
    )

    written = {}
    for name, df in [
        (OUT_METRICS, decomp.grouped_metrics),
        (OUT_CONTRASTS, decomp.grouped_contrasts),
        (OUT_INTERACTIONS, decomp.interactions),
        (OUT_ELIGIBILITY, decomp.site_eligibility),
    ]:
        path = os.path.join(out_dir, name)
        _write_atomic(lambda p, d=df: d.to_csv(p, index=False), path)
        written[name] = path

    legacy_written = None
    if args.legacy_site_products:
        legacy_written = write_legacy_products(frame, str(out_dir), args.bootstrap_reps, args.seed)

    metadata = {
        "run_name": "e1_temporal_decomposition",
        "analysis_date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(str(Path(__file__).resolve().parents[2])),
        "parent_bundle": gate_report,
        "record_schema": PAIRED_RECORD_SCHEMA_VERSION,
        "temporal_class_definition": TEMPORAL_CLASS_DEFINITION,
        "cohort": {
            "rule": f"sites with >= {MIN_OBS_FOR_METRICS} paired dates in BOTH "
            "retrieval and between_retrieval",
            "common_cohort": list(decomp.common_cohort),
            "n_common_sites": n_common,
            "excluded_from_common": excluded,
            "class_row_counts": decomp.class_counts,
        },
        "estimands": {
            "within_class_model_contrast": "SWIM - OpenET within each temporal class",
            "within_model_support_change": (
                "model between_retrieval - model retrieval (secondary; does not by "
                "itself test comparative advantage; derivable from the metrics table)"
            ),
            "cross_model_support_interaction": TEMPORAL_INTERACTION_FORMULA,
            "primary_conclusion_estimand": "cross_model_support_interaction",
            "interaction_favorable_directions": {
                "kge": "positive favors a relative SWIM gain between retrievals",
                "rmse": "negative favors a relative SWIM gain between retrievals",
                "mbe": "signed, directional only — never a bias-magnitude claim",
            },
        },
        "aggregations": [AGG_POOLED, AGG_WEIGHTED],
        "models": list(GROUPED_MODEL_ORDER),
        "pooled_metrics": list(POOLED_METRICS),
        "weighted_metrics": list(WEIGHTED_METRICS),
        "bootstrap": {
            "unit": "site",
            "reps": args.bootstrap_reps,
            "seed": args.seed,
            "interval": "percentile_2.5_97.5",
            "shared_draws": (
                "one multiplicity matrix shared across both models, every metric, "
                "and all three temporal partitions; the interaction is computed "
                "from the same replicate arrays"
            ),
        },
        "parent_evaluator_metadata": {
            "benchmark_source": parent_meta.get("benchmark_source"),
            "benchmark_construction": parent_meta.get("benchmark_construction"),
            "bootstrap": parent_meta.get("bootstrap"),
        },
        "legacy_site_products": (
            {"emitted": True, "files": sorted(legacy_written)}
            if legacy_written
            else {"emitted": False}
        ),
        "output_hashes": {
            name: sha256_file(path)
            for name, path in {**written, **(legacy_written or {})}.items()
            if path.endswith(".csv")
        },
    }
    meta_path = os.path.join(out_dir, OUT_METADATA)
    _write_atomic(lambda p: Path(p).write_text(json.dumps(metadata, indent=2)), meta_path)
    written[OUT_METADATA] = meta_path

    print(f"\nOutputs written to {out_dir}")
    inter = decomp.interactions
    print(
        "\nCross-model support interaction "
        "((SWIM-OpenET | between_retrieval) - (SWIM-OpenET | retrieval)):"
    )
    cols = ["aggregation", "metric", "estimate", "ci95_low", "ci95_high", "unit"]
    print(inter[cols].to_string(index=False))
    if legacy_written:
        print(f"\nLegacy site products (secondary/transition): {len(legacy_written)} files")


if __name__ == "__main__":
    main()

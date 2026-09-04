"""Table S10: paired treatment-minus-control bootstrap for the JET additional dates.

Reproduces the frozen additional-date sensitivity from the hash-verified archived
site-level paired metrics. This is a strict downstream *consumer*, in the same
sense as ``examples/5_Flux_Ensemble/overpass_decomposition.py``: it does not
recompute per-site metrics, it re-bootstraps the ones the evaluator already wrote
and then asserts row-for-row equality against the published summary.

Why a consumer and not a rebuild
--------------------------------
The per-site metrics were produced by ``evaluate.py`` @ 162a35a scoring the
treatment run against the control arm ``archive_recal20260702_classifier`` -- a
*frozen archive subdirectory*, not the live results root of that run, which later
work has since overwritten. A rebuild that points at the live root silently scores
a different control: it yields 66 daily / 59 monthly sites instead of 63 / 50, and
changes paired-day counts at 24 of the 63 common sites. The archived CSVs are the
only surviving record of the pairing behind Table S10, so they are the input, and
their SHA-256 digests are checked on every run.

Provenance chain: tracked ``evaluate.py`` -> archived paired CSVs (hashed in
``paper/data/final/e3_additional_date_paired_bootstrap_metadata.json``) -> this
script -> Table S10.

The estimand
------------
Table S10 reports the **median of the site-level paired differences**, not the
difference between the treatment and control medians. On these data the two are
not interchangeable: for daily KGE the median paired delta is +0.0020, whereas the
difference of medians is +0.0063 -- three times larger. Metric columns are read as
recorded (``r2_delta`` carries NSE under the pre-rename column name), except
absolute MBE, which is rebuilt as ``abs(bias_trt) - abs(bias_ctl)`` because the
archived ``bias_delta`` is a signed difference.

Bootstrap conventions are taken from the metadata and must be preserved to
reproduce the published intervals: 10,000 site resamples, seed 42, the RNG
reinitialized **once per scale** and then drawn sequentially across metrics in the
order ``nse, kge, rmse, absolute_mbe``, with the finite filter applied separately
per metric. Reseeding per metric is not harmless but is easy to miss: it still
reproduces 5 of the 8 published rows exactly, and shifts three monthly bounds. A
bootstrap median is always one of the resampled site values, so these intervals
are coarse and often survive a changed draw -- which is why the reproduction is
asserted against the frozen artifact rather than eyeballed.

Subset rows (``ecostress_active`` / ``no_ecostress``) support the supplement's
statement that the 13 no-ECOSTRESS controls moved by comparable magnitudes. They
are drawn from their own RNG streams so that adding or removing them cannot
perturb the reproduced all-sites rows.

This bounds the sensitivity of the calibration to *uniformly weighted* auxiliary
dates (TOML ``etf_auxiliary_fixed_sd``). It is not an assessment of JET retrieval
quality, and a null here does not establish that the observations carry no
information -- the product distributes a per-pixel intermodel spread that this
design does not use.

Usage:
    uv run python examples/6_Flux_International/expc_paired_deltas.py
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_METADATA = REPO_ROOT / "paper/data/final/e3_additional_date_paired_bootstrap_metadata.json"
FROZEN_SUMMARY = REPO_ROOT / "paper/data/final/e3_additional_date_paired_bootstrap_summary.csv"

# Bootstrap draws follow this order within a scale; changing it changes every interval.
METRIC_ORDER = ("nse", "kge", "rmse", "absolute_mbe")

# Archived column stems. ``r2`` predates the R^2 -> NSE rename; the values are NSE.
METRIC_STEM = {"nse": "r2", "kge": "kge", "rmse": "rmse", "absolute_mbe": "bias"}

SCALE_SOURCE = {
    "daily": "daily_paired_site_metrics",
    "monthly": "monthly_paired_site_metrics",
}

SUBSETS = ("ecostress_active", "no_ecostress")

SUMMARY_CSV = "expc_paired_delta_summary.csv"
SUBSET_CSV = "expc_paired_delta_subsets.csv"

REPRO_ATOL = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(metadata_path=FROZEN_METADATA):
    with open(metadata_path) as handle:
        return json.load(handle)


def resolve_archive(metadata, archive_dir=None):
    """Directory holding the archived paired CSVs, defaulting to the recorded root."""
    if archive_dir:
        return Path(archive_dir)
    rel = metadata["source_files"]["daily_paired_site_metrics"]["path"]
    return Path(metadata["source_root"]) / Path(rel).parent


def verify_sources(archive_dir, metadata):
    """Re-hash every recorded source file; hard-fail on any drift.

    A mismatch means the archived pairing is not the one Table S10 was computed
    from, so the reproduction below would be meaningless.
    """
    checked = {}
    for key, entry in metadata["source_files"].items():
        path = archive_dir / Path(entry["path"]).name
        if not path.exists():
            raise SystemExit(f"missing archived source {key}: {path}")
        got = sha256(path)
        if got != entry["sha256"]:
            raise SystemExit(
                f"SHA-256 mismatch for {key}\n  path     {path}\n"
                f"  expected {entry['sha256']}\n  found    {got}"
            )
        checked[key] = path
    return checked


def paired_deltas(frame, metric):
    """Site-level paired treatment-minus-control differences, finite only.

    Absolute MBE is rebuilt from the two arms because the archived ``bias_delta``
    is signed; every other metric is read from its recorded delta column.
    """
    stem = METRIC_STEM[metric]
    if metric == "absolute_mbe":
        delta = frame[f"{stem}_trt"].abs() - frame[f"{stem}_ctl"].abs()
    else:
        delta = frame[f"{stem}_delta"]
    finite = np.isfinite(delta)
    return frame.loc[finite, "fid"].tolist(), delta[finite].to_numpy()


def bootstrap_median_delta(deltas, rng, reps):
    """Median paired delta with a 95% site-bootstrap interval.

    Resamples the site-level differences themselves, which is the Table S10
    estimand. ``rng`` is advanced in place so a caller can preserve the recorded
    sequential-draw order across metrics.
    """
    n = len(deltas)
    if n == 0:
        return dict.fromkeys(("n_sites", "median_delta", "ci_lower_95", "ci_upper_95"), np.nan)
    medians = np.median(deltas[rng.integers(0, n, size=(reps, n))], axis=1)
    return {
        "n_sites": n,
        "median_delta": float(np.median(deltas)),
        "ci_lower_95": float(np.percentile(medians, 2.5)),
        "ci_upper_95": float(np.percentile(medians, 97.5)),
    }


def reproduce_frozen(frames, reps, seed):
    """All-sites rows in the recorded RNG order: one stream per scale."""
    rows = []
    for scale, frame in frames.items():
        rng = np.random.default_rng(seed)
        for metric in METRIC_ORDER:
            _, deltas = paired_deltas(frame, metric)
            rows.append(
                {
                    "scale": scale,
                    "metric": metric,
                    **bootstrap_median_delta(deltas, rng, reps),
                    "n_bootstrap": reps,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def subset_rows(frames, reps, seed):
    """Treated / stochastic-control rows, each on its own RNG stream.

    Isolated streams keep the reproduced all-sites rows bit-identical regardless
    of whether these are computed.
    """
    rows = []
    for scale, frame in frames.items():
        for subset in SUBSETS:
            sub = frame[frame["subset"] == subset]
            rng = np.random.default_rng(seed)
            for metric in METRIC_ORDER:
                _, deltas = paired_deltas(sub, metric)
                rows.append(
                    {
                        "scale": scale,
                        "subset": subset,
                        "metric": metric,
                        **bootstrap_median_delta(deltas, rng, reps),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["spans_zero"] = (frame["ci_lower_95"] <= 0) & (frame["ci_upper_95"] >= 0)
    return frame


def compare_to_frozen(reproduced, frozen_path=FROZEN_SUMMARY, atol=REPRO_ATOL):
    """Row-for-row comparison against the published summary.

    Returns the joined table with absolute deviations. The caller decides whether
    a deviation is fatal; nothing here rewrites the frozen artifact.
    """
    frozen = pd.read_csv(frozen_path)
    fields = ["n_sites", "median_delta", "ci_lower_95", "ci_upper_95"]
    merged = reproduced.merge(frozen, on=["scale", "metric"], suffixes=("", "_frozen"))
    if len(merged) != len(frozen):
        raise SystemExit(f"reproduced {len(merged)} of {len(frozen)} frozen rows")
    for field in fields:
        merged[f"{field}_dev"] = (merged[field] - merged[f"{field}_frozen"]).abs()
    merged["matches"] = merged[[f"{f}_dev" for f in fields]].max(axis=1) <= atol
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", default=None, help="dir holding the archived paired CSVs")
    parser.add_argument("--metadata", default=str(FROZEN_METADATA))
    parser.add_argument("--frozen-summary", default=str(FROZEN_SUMMARY))
    # Outputs never land in the archive: it is hash-referenced and stays immutable.
    parser.add_argument("--out", default=None, help="output dir (default: the run root)")
    parser.add_argument("--reps", type=int, default=None, help="default: the recorded replicates")
    parser.add_argument("--seed", type=int, default=None, help="default: the recorded seed")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    reps = args.reps if args.reps is not None else metadata["bootstrap"]["replicates"]
    seed = args.seed if args.seed is not None else metadata["bootstrap"]["seed"]

    archive_dir = resolve_archive(metadata, args.archive)
    sources = verify_sources(archive_dir, metadata)
    print(f"archive        : {archive_dir}")
    print(f"control arm    : {metadata['scientific_contrast']}")
    print(f"sha256 verified: {len(sources)} archived source files")
    print(f"bootstrap      : {reps} reps, seed {seed}, {metadata['bootstrap']['rng_scope']}")

    frames = {
        scale: pd.read_csv(sources[key]) for scale, key in SCALE_SOURCE.items() if key in sources
    }

    reproduced = reproduce_frozen(frames, reps, seed)
    merged = compare_to_frozen(reproduced, args.frozen_summary)
    print("\n--- Table S10 reproduction (median paired delta, 95% site bootstrap) ---")
    show = ["scale", "metric", "n_sites", "median_delta", "ci_lower_95", "ci_upper_95", "matches"]
    print(merged[show].to_string(index=False))

    if not merged["matches"].all():
        failed = merged.loc[~merged["matches"], ["scale", "metric"]].to_dict("records")
        raise SystemExit(f"reproduction failed for {failed}; refusing to write outputs")
    print(f"\nall {len(merged)} frozen rows reproduced within {REPRO_ATOL:g}")

    subsets = subset_rows(frames, reps, seed)
    print("\n--- by ECOSTRESS activation (independent RNG streams) ---")
    print(subsets.round(6).to_string(index=False))

    out_dir = Path(args.out) if args.out else Path(metadata["source_root"])
    os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(out_dir / SUMMARY_CSV, index=False)
    subsets.to_csv(out_dir / SUBSET_CSV, index=False)
    print(f"\nwrote {out_dir / SUMMARY_CSV}")
    print(f"wrote {out_dir / SUBSET_CSV}")


if __name__ == "__main__":
    main()

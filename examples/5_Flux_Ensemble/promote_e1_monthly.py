"""Promote a Volk-protocol E1 monthly bundle into ``paper/data/final/e1_openet_benchmark/``.

The monthly half of the frozen E1 benchmark package was first promoted on
2026-09-01 under the repo's own monthly rule (28 valid flux days, at least ten
paired months per site; 30 sites, 1,301 months). The evaluator now follows the
Volk et al. (2024) monthly protocol (flux-data-qaqc gap fill against raw gridMET
ETo, a month total when more than 80% of its days carry ET, admission with at
most five filled days, no site minimum; pooled rows on every paired site and
station-weighted rows on sites with at least three paired months). This script
is the tracked producer of the re-frozen monthly package:

    <source-dir>/evaluation_grouped_monthly_{metrics,contrasts}.csv
    <source-dir>/evaluation_grouped_monthly_metadata.json
    <source-dir>/evaluation_monthly_metrics.csv
    <source-dir>/evaluation_sites_excluded.csv
        -> e1_openet_benchmark/monthly/*                    (byte copy)
    e1_openet_benchmark/monthly/* (28-day package)
        -> superseded_e1_monthly_28day/* + README.md        (moved, never deleted)
    <run>/archive/6_evaluation/monthly_paired_metrics.csv
        -> monthly_paired_metrics_28day_superseded.csv; the new per-site table
           takes its name, and the whole bundle lands in
           archive/6_evaluation/monthly_volk2024/ with a PROMOTION.json sidecar

Gates before anything is written:

  1. the evaluator sidecar names the Volk construction token, was produced on a
     clean worktree (``--allow-dirty`` overrides), and its recorded output
     hashes match the files;
  2. the 18 grouped point estimates are recomputed independently from the run's
     archive daily series (``swim_ET``, raw gridMET ETo), the v2.1 flux files
     (``ET_corr``) and the OpenET v2.1 monthly totals with the shared
     ``flux_utils`` helpers, and must agree with the bundle to REPLICATION_TOL
     with identical pooled and station-weighted cohorts.

``MANIFEST.json`` keeps every daily and temporal field; the monthly cohort,
artifact hashes, analysis-code shas, validation notes and a ``promotion_history``
entry are rewritten. The default mode runs the gates and reports whether the
promoted package already matches the source; ``--write`` performs the promotion.

Usage:
    uv run python examples/5_Flux_Ensemble/promote_e1_monthly.py --source-dir <dir>
    uv run python examples/5_Flux_Ensemble/promote_e1_monthly.py --source-dir <dir> --write
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.calibrate.flux_utils import (
    VOLK_MAX_FILLED_DAYS,
    VOLK_MONTH_COMPLETENESS,
    volk_full_month_paired_sums,
)
from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    CONSTRUCTION_TOKENS,
    PairedSiteSeries,
    grouped_point_estimates,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ex5_paths  # noqa: E402

PACKAGE = "e1_openet_benchmark"
SUPERSEDED_DIRNAME = "superseded_e1_monthly_28day"
ARCHIVE_SUBDIR = "monthly_volk2024"
DEFAULT_SOURCE_SUBDIR = "monthly_volk2024"
MONTHLY_FILES = (
    "evaluation_grouped_monthly_metrics.csv",
    "evaluation_grouped_monthly_contrasts.csv",
    "evaluation_grouped_monthly_metadata.json",
    "evaluation_monthly_metrics.csv",
    "evaluation_sites_excluded.csv",
)
HASHED_IN_SIDECAR = MONTHLY_FILES[:2]
ANALYSIS_FILES = (
    "examples/5_Flux_Ensemble/evaluate.py",
    "src/swimrs/evaluation/benchmark.py",
    "src/swimrs/calibrate/flux_utils.py",
)
MONTHLY_TOKEN = CONSTRUCTION_TOKENS["monthly"]
POOLED_MIN_MONTHS = 1
WEIGHTED_MIN_MONTHS = 3
REPLICATION_TOL = 1e-6
N_GROUPED_ESTIMATES = 18
MONTHLY_PROTOCOL = (
    "Volk et al. (2024) monthly protocol: tower ET gap-filled as flux-data-qaqc "
    "(EToF against raw gridMET ETo, IQR filter, 7-day centered mean, linear "
    f"interpolation); a month is a total when more than {VOLK_MONTH_COMPLETENESS:.0%} of "
    f"its days carry ET and is admitted with at most {VOLK_MAX_FILLED_DAYS} filled days; "
    "SWIM summed over the full calendar month against native OpenET v2.1 monthly ET; "
    "no site-minimum gate"
)


class PromotionError(RuntimeError):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def _json_bytes(obj):
    return (json.dumps(obj, indent=2) + "\n").encode()


def _head_sha(repo):
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# source bundle


def read_source(source_dir):
    """Return ``{name: Path}`` for the five monthly files; every file must exist."""
    source_dir = Path(source_dir)
    files = {name: source_dir / name for name in MONTHLY_FILES}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise PromotionError("missing source files:\n  " + "\n  ".join(missing))
    return files


def check_sidecar(files):
    """Validate the evaluator sidecar; returns the parsed metadata."""
    meta = json.loads(files["evaluation_grouped_monthly_metadata.json"].read_text())
    token = meta.get("benchmark_construction")
    if token != MONTHLY_TOKEN:
        raise PromotionError(f"sidecar benchmark_construction {token!r} != {MONTHLY_TOKEN!r}")
    if not meta.get("git", {}).get("sha"):
        raise PromotionError("sidecar records no git sha")
    recorded = meta.get("output_hashes", {})
    for name in HASHED_IN_SIDECAR:
        if name not in recorded:
            raise PromotionError(f"sidecar has no output hash for {name}")
        actual = sha256_file(files[name])
        if recorded[name] != actual:
            raise PromotionError(
                f"{name}: sidecar sha256 {recorded[name][:16]} != file {actual[:16]}"
            )
    cohorts = meta.get("cohorts", {})
    for agg in (AGG_POOLED, AGG_WEIGHTED):
        if agg not in cohorts:
            raise PromotionError(f"sidecar has no {agg} cohort (split-cohort evaluator required)")
    return meta


def analysis_files_dirty(repo):
    """Analysis files with uncommitted changes (``git status --porcelain`` on those paths)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", *ANALYSIS_FILES],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line[3:].strip() for line in out.stdout.splitlines() if line.strip())


def check_generation_tree(meta, head_sha, dirty_files, allow_dirty=False):
    """The bundle must come from the committed analysis code.

    The evaluator sidecar's whole-worktree ``dirty`` flag counts untracked files
    and is informational only; the gate is that the sidecar sha is HEAD and no
    analysis file (ANALYSIS_FILES) has uncommitted changes. Returns True when
    both hold; raises unless ``allow_dirty``.
    """
    problems = []
    sha = meta["git"]["sha"]
    if sha != head_sha:
        problems.append(f"sidecar sha {sha[:12]} != HEAD {head_sha[:12]}")
    if dirty_files:
        problems.append("uncommitted analysis files: " + ", ".join(dirty_files))
    if problems and not allow_dirty:
        raise PromotionError(
            "bundle is not from the committed analysis code: "
            + "; ".join(problems)
            + " (re-run evaluate.py on the committed tree or pass --allow-dirty)"
        )
    return not problems


def cohort_from_meta(meta, aggregation):
    return tuple(sorted((s["fid"], int(s["n"])) for s in meta["cohorts"][aggregation]["sites"]))


# ---------------------------------------------------------------------------
# independent replication from the run archive


def _archive_site_ids(ts_dir):
    return sorted(p.stem for p in Path(ts_dir).glob("*.csv"))


def read_archive_series(path):
    """Archived daily series with ``swim_ET`` and the raw gridMET ETo as ``eto_raw``.

    The Volk et al. (2024) flux gap fill scales the tower's fraction of raw
    gridMET ETo, so the replication must use the raw series. archive_run.py
    writes the reference ET the model consumed under ``eto``; when that is the
    bias-corrected series it also exports the raw one as ``eto_raw``. Older
    archives (Run 22) wrote only the raw series, as ``eto``.
    """
    ts = pd.read_csv(path, index_col="date", parse_dates=True)
    if "eto_raw" not in ts.columns:
        ts["eto_raw"] = ts["eto"]
    return ts[["swim_ET", "eto_raw"]]


def monthly_records_from_archive(ts_dir, flux_dir, monthly_dir, static_exclusions=()):
    """Per-site Volk-protocol monthly ``PairedSiteSeries`` from archived series.

    ``ts_dir``: ``archive/6_evaluation/site_daily_timeseries`` (``swim_ET`` and
    the raw gridMET ETo, ``eto_raw`` when the archive also carries a corrected
    ``eto``); ``flux_dir``: the Volk v2.1 daily flux
    files (``ET_corr``); ``monthly_dir``: the OpenET v2.1 monthly totals
    (``ensemble_mean_3x3``). Returns the records in ascending fid order; every
    site with at least POOLED_MIN_MONTHS paired months is kept.
    """
    ts_dir, flux_dir, monthly_dir = Path(ts_dir), Path(flux_dir), Path(monthly_dir)
    records = []
    for fid in _archive_site_ids(ts_dir):
        if fid in static_exclusions:
            continue
        flux_path = flux_dir / f"{fid}_daily_data.csv"
        volk_path = monthly_dir / f"{fid}.csv"
        if not flux_path.exists() or not volk_path.exists():
            continue
        flux = pd.read_csv(flux_path, index_col="date", parse_dates=True)
        if "ET_corr" not in flux.columns:
            continue
        ts = read_archive_series(ts_dir / f"{fid}.csv")
        volk = pd.read_csv(volk_path, index_col="DATE", parse_dates=True)
        if "ensemble_mean_3x3" not in volk.columns:
            continue
        swim_m, flux_m, _filled = volk_full_month_paired_sums(
            ts["swim_ET"].astype(float), flux["ET_corr"].astype(float), ts["eto_raw"].astype(float)
        )
        ens = volk["ensemble_mean_3x3"].astype(float).reindex(flux_m.index)
        swim_m = swim_m.reindex(flux_m.index)
        mask = flux_m.notna() & swim_m.notna() & ens.notna()
        months = flux_m.index[mask]
        if len(months) < POOLED_MIN_MONTHS:
            continue
        records.append(
            PairedSiteSeries(
                fid=fid,
                index=pd.DatetimeIndex(months),
                observed=flux_m.loc[months].values,
                swim=swim_m.loc[months].values,
                openet=ens.loc[months].values,
                min_obs=POOLED_MIN_MONTHS,
            )
        )
    if not records:
        raise PromotionError(f"no paired sites replicated from {ts_dir}")
    return tuple(records)


def replicate_from_archive(ts_dir, flux_dir, monthly_dir, static_exclusions=()):
    """Recompute the grouped monthly estimates from archived series.

    See ``monthly_records_from_archive`` for the inputs. Returns
    ``(estimates, cohorts)`` where ``cohorts`` maps each aggregation to a
    sorted tuple of ``(fid, n)``.
    """
    records = monthly_records_from_archive(
        ts_dir, flux_dir, monthly_dir, static_exclusions=static_exclusions
    )
    weighted = tuple(r for r in records if r.n >= WEIGHTED_MIN_MONTHS)
    if not weighted:
        raise PromotionError("no site reaches the station-weighted month floor")
    estimates = grouped_point_estimates(records, aggregations=(AGG_POOLED,))
    estimates.update(grouped_point_estimates(weighted, aggregations=(AGG_WEIGHTED,)))
    cohorts = {
        AGG_POOLED: tuple(sorted((r.fid, r.n) for r in records)),
        AGG_WEIGHTED: tuple(sorted((r.fid, r.n) for r in weighted)),
    }
    return estimates, cohorts


def compare_replication(estimates, cohorts, files, meta, tol=REPLICATION_TOL):
    """Gate the bundle against the replication; returns the max abs difference."""
    df = pd.read_csv(files["evaluation_grouped_monthly_metrics.csv"])
    got = df.set_index(["aggregation", "model", "metric"])["estimate"]
    if len(got) != N_GROUPED_ESTIMATES or len(estimates) != N_GROUPED_ESTIMATES:
        raise PromotionError(
            f"expected {N_GROUPED_ESTIMATES} grouped estimates; bundle {len(got)}, "
            f"replication {len(estimates)}"
        )
    diffs = {}
    for key, value in estimates.items():
        if key not in got.index:
            raise PromotionError(f"bundle lacks the grouped estimate {key}")
        diffs[key] = abs(float(got[key]) - float(value))
    worst = max(diffs.values())
    if not np.isfinite(worst) or worst > tol:
        bad = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
        raise PromotionError(
            f"replication differs from the bundle (max {worst:.3e} > {tol}): {bad}"
        )
    for agg in (AGG_POOLED, AGG_WEIGHTED):
        if cohorts[agg] != cohort_from_meta(meta, agg):
            raise PromotionError(f"{agg}: replicated cohort differs from the sidecar cohort")
        n_sites, n_pairs = len(cohorts[agg]), sum(n for _, n in cohorts[agg])
        rows = df[df["aggregation"] == agg]
        if not ((rows["n_sites"] == n_sites).all() and (rows["n_pairs"] == n_pairs).all()):
            raise PromotionError(f"{agg}: bundle n_sites/n_pairs != {n_sites}/{n_pairs}")
    return worst


# ---------------------------------------------------------------------------
# manifest and README


def _cohort_entry(meta, aggregation, min_months):
    c = meta["cohorts"][aggregation]
    return {
        "n_sites": int(c["n_sites"]),
        "n_paired_site_months": int(c["n_pairs"]),
        "min_paired_months": int(min_months),
    }


def build_manifest(
    prior,
    meta,
    files,
    repo,
    replication_max_diff,
    now=None,
    head_sha=None,
    analysis_clean=True,
):
    """The prior MANIFEST with its monthly fields rewritten for the promoted bundle."""
    now = now or datetime.now().astimezone().isoformat(timespec="seconds")
    head_sha = head_sha or _head_sha(repo)
    manifest = json.loads(json.dumps(prior))
    prior_monthly = prior.get("cohorts", {}).get("monthly", {})
    manifest["promoted_at"] = now
    manifest["promoted_at_git_sha"] = head_sha
    manifest.setdefault("reporting_contract", {})["monthly_cohorts"] = (
        "Pooled monthly rows use every site with at least one paired month; "
        "station-weighted monthly rows and per-site monthly metrics require at least "
        f"{WEIGHTED_MIN_MONTHS} paired months (Volk et al. 2024 monthly protocol)"
    )
    manifest.setdefault("cohorts", {})["monthly"] = {
        "protocol": MONTHLY_PROTOCOL,
        AGG_POOLED: _cohort_entry(meta, AGG_POOLED, POOLED_MIN_MONTHS),
        AGG_WEIGHTED: _cohort_entry(meta, AGG_WEIGHTED, WEIGHTED_MIN_MONTHS),
        "pooled_only_sites": list(meta.get("pooled_only_sites", [])),
    }
    hashes = manifest.setdefault("artifact_sha256", {})
    for name, path in files.items():
        hashes[f"monthly/{name}"] = sha256_file(path)
    manifest["artifact_sha256"] = dict(sorted(hashes.items()))
    code = manifest.setdefault("analysis_code", {})
    code["monthly_generation_git_sha"] = meta["git"]["sha"]
    code["monthly_generation_worktree_dirty"] = bool(meta["git"].get("dirty", False))
    code["monthly_analysis_files_clean_at_promotion"] = bool(analysis_clean)
    current = code.setdefault("current_file_sha256", {})
    for rel in ANALYSIS_FILES:
        current[rel] = sha256_file(Path(repo) / rel)
    code["monthly_promotion_note"] = (
        "Monthly re-promoted on the Volk et al. (2024) protocol; flux_utils.py added to the "
        "listed analysis files because it holds the gap-fill and month-total helpers. The "
        "daily and temporal products are untouched. The sidecar's whole-worktree dirty flag "
        "counts untracked files; monthly_analysis_files_clean_at_promotion records whether "
        "the sidecar sha was HEAD with no uncommitted change to the analysis files."
    )
    validation = manifest.setdefault("validation", {})
    validation["monthly_promoted_vs_validated_bundle"] = (
        "SUPERSEDED with the 28-day package; see monthly_replication_from_archive"
    )
    validation["monthly_sidecar_hashes"] = (
        "PASS: grouped metrics and contrasts sha256 match the evaluator sidecar"
    )
    validation["monthly_replication_from_archive"] = (
        f"PASS: {N_GROUPED_ESTIMATES} grouped estimates recomputed from "
        "archive/6_evaluation/site_daily_timeseries (swim_ET, raw gridMET eto), the v2.1 flux files "
        "(ET_corr) and the OpenET v2.1 monthly totals with the flux_utils helpers; "
        f"max abs diff {replication_max_diff:.3e} at tolerance {REPLICATION_TOL:g}; "
        "identical pooled and station-weighted cohorts"
    )
    legacy = manifest.setdefault("legacy_and_exclusions", {})
    legacy["superseded_monthly_28day"] = (
        f"paper/data/final/{SUPERSEDED_DIRNAME}/ holds the monthly package frozen "
        f"2026-09-01 on the repo's 28-valid-day / >= 10-month rule "
        f"({prior_monthly.get('n_sites')} sites, {prior_monthly.get('n_paired_site_months')} "
        "months); it must not be used for current monthly benchmark reporting. The within-E1 "
        "transfer (Table S7) and weighting-ablation (Table S6, Fig. 4d) monthly rows still "
        "use the 28-day rule through their own producers."
    )
    history = list(manifest.get("promotion_history", []))
    history.append(
        {
            "promoted_at": now,
            "git_sha": head_sha,
            "scope": "monthly",
            "change": (
                "28-valid-day / >= 10-month monthly rule replaced by the Volk et al. (2024) "
                "protocol with split pooled / station-weighted cohorts"
            ),
            "from": {
                "n_sites": prior_monthly.get("n_sites"),
                "n_paired_site_months": prior_monthly.get("n_paired_site_months"),
                "moved_to": f"paper/data/final/{SUPERSEDED_DIRNAME}/",
            },
            "to": {
                AGG_POOLED: _cohort_entry(meta, AGG_POOLED, POOLED_MIN_MONTHS),
                AGG_WEIGHTED: _cohort_entry(meta, AGG_WEIGHTED, WEIGHTED_MIN_MONTHS),
            },
            "source_dir": str(Path(next(iter(files.values()))).parent),
            "generation_git_sha": meta["git"]["sha"],
        }
    )
    manifest["promotion_history"] = history
    return manifest


def superseded_readme(prior, moved, now):
    """README text for the superseded 28-day monthly package."""
    prior_monthly = prior.get("cohorts", {}).get("monthly", {})
    lines = [
        "# Superseded E1 monthly package (28-valid-day rule)",
        "",
        f"Moved here on {now} from `paper/data/final/{PACKAGE}/monthly/`.",
        "",
        "These files are the monthly half of the E1 benchmark package frozen on",
        f"{prior.get('promoted_at', '2026-09-01')} (promotion sha "
        f"{prior.get('promoted_at_git_sha', '?')}, monthly generation sha "
        f"{prior.get('analysis_code', {}).get('monthly_generation_git_sha', '?')}).",
        "They were produced under the repo's own monthly rule: a calendar month entered",
        "when it had at least 28 valid flux days, flux ET was summed over the valid days",
        "only, SWIM and native OpenET totals covered the full month, and a site needed at",
        f"least ten paired months ({prior_monthly.get('n_sites')} sites, "
        f"{prior_monthly.get('n_paired_site_months')} site-months).",
        "",
        "Superseded by the Volk et al. (2024) monthly protocol (flux-data-qaqc gap fill",
        "against raw gridMET ETo, month total when more than 80% of days carry ET,",
        "admission with at most five filled days, no site minimum; pooled rows on every",
        "paired site and station-weighted rows on sites with at least three paired months).",
        f"The current monthly package lives in `paper/data/final/{PACKAGE}/monthly/` and",
        "is described in that package's `MANIFEST.json`. **Do not use these files for",
        "current monthly benchmark reporting.** The within-E1 transfer (Table S7) and the",
        "weighting ablation (Table S6, Fig. 4d) monthly rows still use the 28-day rule",
        "through `within_e1_transfer.py` and `rebuild_e1_benchmark_evidence.py`.",
        "",
        "| file | sha256 |",
        "| --- | --- |",
    ]
    for name, digest in sorted(moved.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# write


def compare_promoted(files, monthly_dir):
    rows = []
    for name, src in files.items():
        target = Path(monthly_dir) / name
        data = src.read_bytes()
        if not target.exists():
            status = "missing"
        else:
            status = "match" if target.read_bytes() == data else "differs"
        rows.append((name, status, sha256_bytes(data)))
    return rows


def _refuse_if_exists(path, what):
    if Path(path).exists():
        raise PromotionError(f"{what} already exists, refusing to overwrite: {path}")


def promote(
    files,
    meta,
    final_dir,
    archive_dir,
    repo,
    replication_max_diff,
    now=None,
    head_sha=None,
    analysis_clean=True,
):
    """Move the 28-day package aside, copy the bundle in, rewrite MANIFEST, refresh Cat 6."""
    final_dir = Path(final_dir)
    package = final_dir / PACKAGE
    monthly_dir = package / "monthly"
    manifest_path = package / "MANIFEST.json"
    superseded = final_dir / SUPERSEDED_DIRNAME
    now = now or datetime.now().astimezone().isoformat(timespec="seconds")
    prior = json.loads(manifest_path.read_text())

    # every target that would be overwritten is checked before the first write
    existing = (
        sorted(p for p in monthly_dir.glob("*") if p.is_file()) if monthly_dir.exists() else []
    )
    if existing:
        _refuse_if_exists(superseded, "superseded monthly directory")
    archive_dir = Path(archive_dir) if archive_dir else None
    if archive_dir is not None:
        old_paired = archive_dir / "monthly_paired_metrics.csv"
        if old_paired.exists():
            _refuse_if_exists(
                archive_dir / "monthly_paired_metrics_28day_superseded.csv",
                "archive superseded per-site table",
            )
        _refuse_if_exists(archive_dir / ARCHIVE_SUBDIR, "archive monthly bundle")

    manifest = build_manifest(
        prior,
        meta,
        files,
        repo,
        replication_max_diff,
        now=now,
        head_sha=head_sha,
        analysis_clean=analysis_clean,
    )

    if existing:
        superseded.mkdir(parents=True)
        moved = {}
        for src in existing:
            moved[src.name] = sha256_file(src)
            shutil.move(str(src), str(superseded / src.name))
        (superseded / "README.md").write_text(superseded_readme(prior, moved, now))
    monthly_dir.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copyfile(src, monthly_dir / name)
    manifest_path.write_bytes(_json_bytes(manifest))

    if archive_dir is not None:
        if old_paired.exists():
            shutil.move(
                str(old_paired), str(archive_dir / "monthly_paired_metrics_28day_superseded.csv")
            )
        shutil.copyfile(files["evaluation_monthly_metrics.csv"], old_paired)
        bundle_dir = archive_dir / ARCHIVE_SUBDIR
        bundle_dir.mkdir(parents=True)
        for name, src in files.items():
            shutil.copyfile(src, bundle_dir / name)
        (bundle_dir / "PROMOTION.json").write_bytes(
            _json_bytes(
                {
                    "promoted_at": now,
                    "promoted_to": str(monthly_dir),
                    "source_dir": str(next(iter(files.values())).parent),
                    "generation_git_sha": meta["git"]["sha"],
                    "protocol": MONTHLY_PROTOCOL,
                    "replication_max_abs_diff": replication_max_diff,
                    "note": (
                        "monthly_paired_metrics.csv in the parent directory is now this "
                        "bundle's per-site table; the 28-day table it replaced is kept as "
                        "monthly_paired_metrics_28day_superseded.csv"
                    ),
                }
            )
        )
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source-dir", default=None, help=f"default: <run>/{DEFAULT_SOURCE_SUBDIR}")
    ap.add_argument(
        "--run-dir",
        default=None,
        help=f"default: the {ex5_paths.CANONICAL_RUN} results dir (ex5_paths.CANONICAL_RUN)",
    )
    ap.add_argument("--archive-dir", default=None, help="default: <run-dir>/archive/6_evaluation")
    ap.add_argument("--no-archive", action="store_true", help="leave the run archive untouched")
    ap.add_argument("--final-dir", default=str(ex5_paths.FINAL_DIR))
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="accept a bundle whose sidecar sha is not HEAD or whose analysis files are modified",
    )
    ap.add_argument(
        "--skip-replication", action="store_true", help="skip the archive replication gate"
    )
    ap.add_argument("--write", action="store_true", help="perform the promotion")
    args = ap.parse_args(argv)

    if args.run_dir is None:
        args.run_dir = ex5_paths.run_dir(cfg=ex5_paths.load_config())
    run_dir = Path(args.run_dir)
    source_dir = Path(args.source_dir or run_dir / DEFAULT_SOURCE_SUBDIR)
    archive_dir = (
        None if args.no_archive else Path(args.archive_dir or run_dir / "archive" / "6_evaluation")
    )
    final_dir = Path(args.final_dir)

    files = read_source(source_dir)
    meta = check_sidecar(files)
    head = _head_sha(ex5_paths.REPO)
    dirty = analysis_files_dirty(ex5_paths.REPO)
    clean = check_generation_tree(meta, head, dirty, allow_dirty=args.allow_dirty)
    print(
        f"sidecar OK: {MONTHLY_TOKEN} at {meta['git']['sha'][:12]}; analysis code "
        f"{'committed at HEAD' if clean else 'NOT the committed tree (--allow-dirty)'}"
    )

    max_diff = float("nan")
    if not args.skip_replication:
        ts_dir = (archive_dir or run_dir / "archive" / "6_evaluation") / "site_daily_timeseries"
        estimates, cohorts = replicate_from_archive(
            ts_dir,
            meta["paths"]["flux_dir"],
            meta["paths"]["openet_monthly_dir"],
            static_exclusions=set(meta.get("static_exclusions", [])),
        )
        max_diff = compare_replication(estimates, cohorts, files, meta)
        print(
            f"replication OK: max |diff| {max_diff:.3e} over {N_GROUPED_ESTIMATES} estimates; "
            f"pooled {len(cohorts[AGG_POOLED])} sites, weighted {len(cohorts[AGG_WEIGHTED])} sites"
        )

    monthly_dir = final_dir / PACKAGE / "monthly"
    rows = compare_promoted(files, monthly_dir)
    for name, status, digest in rows:
        print(f"{status:8s} {digest[:16]}  monthly/{name}")

    if args.write:
        if args.skip_replication:
            raise PromotionError("--write requires the replication gate")
        manifest = promote(
            files,
            meta,
            final_dir,
            archive_dir,
            ex5_paths.REPO,
            max_diff,
            head_sha=head,
            analysis_clean=clean,
        )
        c = manifest["cohorts"]["monthly"]
        print(
            f"promoted: pooled {c[AGG_POOLED]['n_sites']} sites / "
            f"{c[AGG_POOLED]['n_paired_site_months']} months, station-weighted "
            f"{c[AGG_WEIGHTED]['n_sites']} sites / {c[AGG_WEIGHTED]['n_paired_site_months']} months"
        )
        print(f"28-day package moved to {final_dir / SUPERSEDED_DIRNAME}")
        if archive_dir is not None:
            print(f"archive refreshed: {archive_dir / ARCHIVE_SUBDIR}")
        return 0
    if any(status != "match" for _, status, _ in rows):
        print("promoted monthly package differs from the source bundle (run with --write)")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PromotionError as exc:
        print(f"PROMOTION REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)

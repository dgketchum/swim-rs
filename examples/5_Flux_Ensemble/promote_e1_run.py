"""Promote a whole Example 5 run's E1 benchmark record into ``paper/data/final/``.

Tracked producer for the run-dependent E1 packages. When a recalibration
replaces the canonical run (``ex5_paths.CANONICAL_RUN``), the frozen packages
below are moved, never deleted, to ``paper/data/final/<superseded-name>/`` with
a README, and the new run's products are byte-copied in:

    <run>/evaluation_daily_volk/{evaluation_grouped_daily_{metrics,contrasts}.csv,
        evaluation_grouped_daily_metadata.json, evaluation_metrics.csv,
        evaluation_paired_daily_records.csv, evaluation_sites_excluded.csv}
        -> e1_openet_benchmark/daily/*              (evaluate.py --output-dir, daily volk)
    <run>/monthly_volk2024/*
        -> e1_openet_benchmark/monthly/*            (evaluate.py --monthly --output-dir)
    <run>/temporal_decomposition/*
        -> e1_openet_benchmark/temporal/*           (overpass_decomposition.py on the
                                                     daily bundle above)
    <results>/ablation_<run>_summary/{paired_site_deltas_daily,paired_site_deltas_monthly,
        paired_delta_summary,ablation_summary}.csv
        -> e2_weighting_ablation_{daily_site_deltas,monthly_site_deltas,
           paired_deltas,summary}.csv               (run_weighting_ablation.py, renamed)
    e1_openet_benchmark/MANIFEST.json               rewritten (cohorts, hashes, git sha,
                                                     promotion_history)
    e2_weighting_ablation_metadata.json             written (source paths, sha256, run tag)

``archive/6_evaluation`` (archive_run.py Cat 6) is not a complete daily source:
it names the per-site table ``daily_paired_metrics.csv`` and writes no daily
exclusion ledger, so the daily bundle comes from an ``evaluate.py --output-dir``
run. When Cat 6 carries a grouped daily bundle it must be byte-identical to the
daily source (gate). ``e1_openet_benchmark/reconstruction_fidelity/`` is
run-independent (OpenET native monthly vs the ETf-first reconstruction; no SWIM
input) and is never touched.

Promotion order for a new canonical run:

  (a) ``promote_e1_run.py --write`` (this script): benchmark record + ablation;
  (b) ``rebuild_e1_benchmark_evidence.py --run-dir <run> --output-dir <scratch>``
      then its own freeze of the ``e2_primary_*``/``e2_benchmark_*``/
      ``e2_temporal_*``/``e2_evidence_metadata.json`` files (its G-ABLATION gate
      reads the ablation files promoted in (a));
  (c) ``promote_final.py --write`` (``e2_spread_error_*``, ``e2_within_transfer_*``,
      ``e2_irrigation_stratified_*``).

This script owns only the files listed above; the superseded directory it
creates holds nothing else, so it cannot collide with
``superseded_e2_direct_interpolation/`` or any move made by (b) or (c).

Gates (fail, never warn; ``--write`` refuses on any failure):
  G-GIT       analysis files have no uncommitted change and none changed between
              each sidecar's generation sha and HEAD (``--allow-dirty`` overrides)
  G-EXIST     every source file exists
  G-DAILY     sidecar names ``etf_first_volk_window`` / ``openet_flux_2pt1`` / volk,
              production bootstrap, recorded output hashes match, run tag in the
              par/container paths, par.csv sha256 matches the recorded input hash,
              exclusion ledger and per-site table agree with the sidecar; the
              paired record re-derives the grouped estimates to 1e-12
              (overpass_decomposition.validate_parent_bundle); archive Cat 6 agrees
  G-MONTHLY   promote_e1_monthly sidecar gate (Volk token, hashes, split cohorts),
              run tag, ledger/table agreement, and the 18-estimate replication from
              the run's archive daily series
  G-TEMPORAL  output hashes match; the recorded parent hashes equal the daily
              source files (the decomposition was computed on this daily bundle)
  G-ABLATION  summary dir and spread-arm container carry the run tag; the spread
              arm equals the primary daily/monthly site metrics within 1e-9
              (rebuild_e1_benchmark_evidence.check_rescored_ablation)
  G-SUPERSEDE the superseded directory does not already exist

The default is check mode: all gates run and every package is compared with the
frozen copy byte-for-byte (sha256). The last line is ``PROMOTION_STATE: MATCH``
or ``PROMOTION_STATE: DIFFERS``.

Usage:
    uv run python examples/5_Flux_Ensemble/promote_e1_run.py                  # check
    uv run python examples/5_Flux_Ensemble/promote_e1_run.py --write
    uv run python examples/5_Flux_Ensemble/promote_e1_run.py --run run22 --allow-dirty
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import pandas as pd

from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    BENCHMARK_SOURCE_MACHINE_TOKENS,
    BOOTSTRAP_REPS_DEFAULT,
    CONSTRUCTION_TOKENS,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ex5_paths  # noqa: E402
import promote_e1_monthly as pm  # noqa: E402
from overpass_decomposition import validate_parent_bundle  # noqa: E402
from rebuild_e1_benchmark_evidence import check_rescored_ablation  # noqa: E402

PromotionError = pm.PromotionError
sha256_file = pm.sha256_file

PACKAGE = "e1_openet_benchmark"
MANIFEST = "MANIFEST.json"
UNTOUCHED_SUBPACKAGES = ("reconstruction_fidelity",)

DAILY_SUBDIR = "evaluation_daily_volk"
MONTHLY_SUBDIR = pm.DEFAULT_SOURCE_SUBDIR
TEMPORAL_SUBDIR = "temporal_decomposition"

DAILY_METADATA = "evaluation_grouped_daily_metadata.json"
DAILY_FILES = (
    "evaluation_grouped_daily_metrics.csv",
    "evaluation_grouped_daily_contrasts.csv",
    DAILY_METADATA,
    "evaluation_metrics.csv",
    "evaluation_paired_daily_records.csv",
    "evaluation_sites_excluded.csv",
)
DAILY_HASHED = (DAILY_FILES[0], DAILY_FILES[1], DAILY_FILES[4])
MONTHLY_METADATA = "evaluation_grouped_monthly_metadata.json"
MONTHLY_FILES = pm.MONTHLY_FILES
TEMPORAL_METADATA = "evaluation_temporal_metadata.json"
TEMPORAL_FILES = (
    "evaluation_temporal_grouped_metrics.csv",
    "evaluation_temporal_grouped_contrasts.csv",
    "evaluation_temporal_interactions.csv",
    "evaluation_temporal_site_eligibility.csv",
    TEMPORAL_METADATA,
)
# run_weighting_ablation.py summary name -> frozen name
ABLATION_RENAMES = {
    "paired_site_deltas_daily.csv": "e2_weighting_ablation_daily_site_deltas.csv",
    "paired_site_deltas_monthly.csv": "e2_weighting_ablation_monthly_site_deltas.csv",
    "paired_delta_summary.csv": "e2_weighting_ablation_paired_deltas.csv",
    "ablation_summary.csv": "e2_weighting_ablation_summary.csv",
}
ABLATION_METADATA = "e2_weighting_ablation_metadata.json"
ABLATION_SPREAD_ARM = "e1_spread"
ABLATION_TOL = 1e-9

ANALYSIS_FILES = (
    "examples/5_Flux_Ensemble/evaluate.py",
    "examples/5_Flux_Ensemble/overpass_decomposition.py",
    "examples/5_Flux_Ensemble/5_Flux_Ensemble.toml",
    "src/swimrs/evaluation/benchmark.py",
    "src/swimrs/calibrate/benchmark.py",
    "src/swimrs/calibrate/flux_utils.py",
)
SUPERSEDED_TEMPLATE = "superseded_e1_{prev}_nextday_eto"
DEFAULT_REASON = (
    "The run's ETf calibration targets for PT-JPL, geeSEBAL and DisALEXi were formed "
    "against the next day's reference ETo (ET(d)/ETo(d+1)); the replacing run was "
    "calibrated on targets rebuilt with the scene's calendar-day ETo "
    "(paper/notes/eto_offset_defect_20260925.md)."
)

Entry = namedtuple("Entry", "package final_rel source")
Gate = namedtuple("Gate", "name ok message")


# ---------------------------------------------------------------------------
# sources


def default_sources(run, results_root):
    """Default source dirs for ``run`` under the Ex5 ``results`` root."""
    results_root = Path(results_root)
    run_dir = results_root / run
    return {
        "run_dir": run_dir,
        "daily": run_dir / DAILY_SUBDIR,
        "monthly": run_dir / MONTHLY_SUBDIR,
        "temporal": run_dir / TEMPORAL_SUBDIR,
        "ablation": results_root / f"ablation_{run}_summary",
        "ablation_spread": results_root / f"ablation_{run}_{ABLATION_SPREAD_ARM}",
        "archive": run_dir / "archive" / "6_evaluation",
    }


def package_entries(src):
    """Every promoted file: ``Entry(package, path relative to final dir, source path)``."""
    entries = []
    for scale, names in (
        ("daily", DAILY_FILES),
        ("monthly", MONTHLY_FILES),
        ("temporal", TEMPORAL_FILES),
    ):
        for name in names:
            entries.append(Entry(scale, f"{PACKAGE}/{scale}/{name}", Path(src[scale]) / name))
    for name, frozen in ABLATION_RENAMES.items():
        entries.append(Entry("ablation", frozen, Path(src["ablation"]) / name))
    return entries


def compare_entries(entries, final_dir):
    """``[(entry, status, source_sha, final_sha)]``; status MATCH / DIFFERS / MISSING.

    MISSING means the frozen copy is absent; NO-SOURCE means the source is absent.
    """
    rows = []
    for e in entries:
        target = Path(final_dir) / e.final_rel
        src_sha = sha256_file(e.source) if e.source.is_file() else None
        dst_sha = sha256_file(target) if target.is_file() else None
        if src_sha is None:
            status = "NO-SOURCE"
        elif dst_sha is None:
            status = "MISSING"
        else:
            status = "MATCH" if src_sha == dst_sha else "DIFFERS"
        rows.append((e, status, src_sha, dst_sha))
    return rows


def _load_json(path):
    return json.loads(Path(path).read_text())


def previous_run_tag(final_dir):
    """The run tag of the frozen package, from its MANIFEST and sidecars (must agree)."""
    final_dir = Path(final_dir)
    package = final_dir / PACKAGE
    found = {}
    manifest = package / MANIFEST
    if manifest.is_file():
        tag = _load_json(manifest).get("internal_archive_id")
        if tag:
            found["MANIFEST.json internal_archive_id"] = tag
    for scale, meta_name in (("daily", DAILY_METADATA), ("monthly", MONTHLY_METADATA)):
        path = package / scale / meta_name
        if path.is_file():
            par = _load_json(path).get("paths", {}).get("par_csv")
            if par:
                found[f"{scale} sidecar paths.par_csv"] = Path(par).parent.name
    ablation = final_dir / ABLATION_METADATA
    if ablation.is_file():
        tag = _load_json(ablation).get("internal_archive_id")
        if tag:
            found[f"{ABLATION_METADATA} internal_archive_id"] = tag
    if not found:
        raise PromotionError(f"cannot determine the frozen run tag under {final_dir}")
    if len(set(found.values())) != 1:
        raise PromotionError(f"frozen records disagree on the run tag: {found}")
    return next(iter(found.values()))


# ---------------------------------------------------------------------------
# gates


def _require(cond, message):
    if not cond:
        raise PromotionError(message)


def _check_output_hashes(meta, directory, names, label):
    recorded = meta.get("output_hashes", {})
    for name in names:
        _require(name in recorded, f"{label}: sidecar has no output hash for {name}")
        actual = sha256_file(Path(directory) / name)
        _require(
            recorded[name] == actual,
            f"{label}: {name} sidecar sha256 {recorded[name][:16]} != file {actual[:16]}",
        )


def _check_run_tag(meta, run, label):
    paths = meta.get("paths", {})
    par, container = paths.get("par_csv"), paths.get("container")
    _require(par and container, f"{label}: sidecar lacks paths.par_csv / paths.container")
    _require(
        Path(par).parent.name == run,
        f"{label}: sidecar par_csv {par} is not under the {run} results dir",
    )
    _require(
        Path(container).name.endswith(f"_{run}.swim"),
        f"{label}: sidecar container {container} is not the {run} container",
    )
    recorded = meta.get("input_hashes", {}).get("par_csv")
    if recorded is not None:
        _require(Path(par).is_file(), f"{label}: posterior {par} is gone")
        actual = sha256_file(par)
        _require(
            recorded.get("sha256") == actual,
            f"{label}: posterior {par} sha256 {actual[:16]} != recorded "
            f"{str(recorded.get('sha256'))[:16]}",
        )


def _check_bootstrap(meta, label):
    reps = meta.get("bootstrap", {}).get("reps")
    _require(
        reps == BOOTSTRAP_REPS_DEFAULT,
        f"{label}: bootstrap reps {reps} != production {BOOTSTRAP_REPS_DEFAULT}",
    )


def _check_ledger_and_table(meta, directory, table_name, label):
    ledger = pd.read_csv(Path(directory) / "evaluation_sites_excluded.csv")
    got = sorted(zip(ledger["site"], ledger["reason"], strict=True))
    want = sorted((r["site"], r["reason"]) for r in meta.get("excluded_sites", []))
    _require(got == want, f"{label}: exclusion ledger != sidecar excluded_sites")
    table = pd.read_csv(Path(directory) / table_name)
    got = sorted(zip(table["fid"], table["n"].astype(int), strict=True))
    want = sorted((s["fid"], int(s["n"])) for s in meta.get("sites", []))
    _require(got == want, f"{label}: {table_name} (fid, n) != sidecar sites")


def _require_files(directory, names, label):
    missing = [n for n in names if not (Path(directory) / n).is_file()]
    _require(not missing, f"{label}: missing in {directory}: {', '.join(missing)}")


def check_daily(src, run, heavy=True):
    d = Path(src["daily"])
    _require_files(d, DAILY_FILES, "daily")
    meta = _load_json(d / DAILY_METADATA)
    _require(meta.get("scale") == "daily", f"daily: sidecar scale {meta.get('scale')!r}")
    _require(meta.get("openet_source") == "volk", "daily: sidecar openet_source is not volk")
    _require(
        meta.get("benchmark_source") == BENCHMARK_SOURCE_MACHINE_TOKENS["volk"],
        f"daily: benchmark_source {meta.get('benchmark_source')!r}",
    )
    _require(
        meta.get("benchmark_construction") == CONSTRUCTION_TOKENS["daily"],
        f"daily: benchmark_construction {meta.get('benchmark_construction')!r} != "
        f"{CONSTRUCTION_TOKENS['daily']!r}",
    )
    _require(bool(meta.get("git", {}).get("sha")), "daily: sidecar records no git sha")
    _check_bootstrap(meta, "daily")
    _check_output_hashes(meta, d, DAILY_HASHED, "daily")
    _check_run_tag(meta, run, "daily")
    _check_ledger_and_table(meta, d, "evaluation_metrics.csv", "daily")
    detail = {}
    archive = Path(src["archive"]) if src.get("archive") else None
    if archive is not None and (archive / DAILY_FILES[0]).is_file():
        for name in DAILY_HASHED:
            _require(
                (archive / name).is_file() and sha256_file(archive / name) == sha256_file(d / name),
                f"daily: archive Cat 6 {name} differs from the daily source",
            )
        detail["archive_cat6"] = "byte-identical grouped metrics, contrasts, paired record"
    if heavy:
        _frame, _meta, report = validate_parent_bundle(d)
        detail["record_identity_max_abs_diff"] = report["grouped_point_identity_max_abs_diff"]
    return meta, detail


def check_monthly(src, run, heavy=True):
    d = Path(src["monthly"])
    files = pm.read_source(d)
    meta = pm.check_sidecar(files)
    _check_bootstrap(meta, "monthly")
    _check_run_tag(meta, run, "monthly")
    _check_ledger_and_table(meta, d, "evaluation_monthly_metrics.csv", "monthly")
    detail = {}
    if heavy:
        estimates, cohorts = pm.replicate_from_archive(
            Path(src["archive"]) / "site_daily_timeseries",
            meta["paths"]["flux_dir"],
            meta["paths"]["openet_monthly_dir"],
            static_exclusions=set(meta.get("static_exclusions", [])),
        )
        detail["replication_max_abs_diff"] = pm.compare_replication(estimates, cohorts, files, meta)
    return meta, detail


def check_temporal(src):
    d, daily = Path(src["temporal"]), Path(src["daily"])
    _require_files(d, TEMPORAL_FILES, "temporal")
    meta = _load_json(d / TEMPORAL_METADATA)
    _require(bool(meta.get("git", {}).get("sha")), "temporal: sidecar records no git sha")
    _check_bootstrap(meta, "temporal")
    _check_output_hashes(meta, d, TEMPORAL_FILES[:-1], "temporal")
    parent_meta = meta.get("parent_evaluator_metadata", {})
    _require(
        parent_meta.get("benchmark_construction") == CONSTRUCTION_TOKENS["daily"],
        f"temporal: parent construction {parent_meta.get('benchmark_construction')!r}",
    )
    parent = meta.get("parent_bundle", {})
    rehashed = parent.get("rehashed_artifacts", {})
    _require(bool(rehashed), "temporal: sidecar records no parent artifact hashes")
    for name, digest in rehashed.items():
        _require(
            (daily / name).is_file() and sha256_file(daily / name) == digest,
            f"temporal: parent {name} is not the daily source file (decomposition was "
            "computed on another daily bundle)",
        )
    _require(
        parent.get("metadata_sha256") == sha256_file(daily / DAILY_METADATA),
        "temporal: parent sidecar hash != the daily source sidecar",
    )
    return meta, {}


def _ablation_container_ok(container, run):
    return re.search(rf"_{re.escape(run)}(ablation)?\.swim$", str(container)) is not None


def check_ablation(src, run, monthly_primary=None):
    a = Path(src["ablation"])
    _require_files(a, tuple(ABLATION_RENAMES), "ablation")
    _require(
        a.name.startswith(f"ablation_{run}_"),
        f"ablation: summary dir {a.name} does not carry the {run} tag",
    )
    spread = Path(src["ablation_spread"])
    runtime = spread / "runtime.json"
    _require(runtime.is_file(), f"ablation: spread arm runtime.json missing: {runtime}")
    rt = _load_json(runtime)
    _require(
        rt.get("experiment_id") == ABLATION_SPREAD_ARM,
        f"ablation: spread arm experiment_id {rt.get('experiment_id')!r}",
    )
    _require(
        _ablation_container_ok(rt.get("container_path", ""), run),
        f"ablation: spread arm container {rt.get('container_path')} is not {run}-seeded",
    )
    daily_primary = Path(src["daily"]) / "evaluation_metrics.csv"
    monthly_primary = Path(
        monthly_primary or Path(src["monthly"]) / "evaluation_monthly_metrics.csv"
    )
    with tempfile.TemporaryDirectory() as tmp:
        for name, frozen in ABLATION_RENAMES.items():
            (Path(tmp) / frozen).symlink_to((a / name).resolve())
        block = check_rescored_ablation(
            tmp, pd.read_csv(daily_primary), pd.read_csv(monthly_primary), tol=ABLATION_TOL
        )
    return rt, {
        "gate": block["gate"],
        "daily_primary": str(daily_primary),
        "daily_primary_sha256": sha256_file(daily_primary),
        "monthly_primary": str(monthly_primary),
        "monthly_primary_sha256": sha256_file(monthly_primary),
    }


GATE_ERRORS = (PromotionError, ValueError, KeyError, OSError)


def run_gates(src, run, heavy=True, monthly_primary=None):
    """Run every source gate; returns ``(gates, metas, details)`` without raising."""
    gates, metas, details = [], {}, {}
    checks = (
        ("G-DAILY", "daily", lambda: check_daily(src, run, heavy=heavy)),
        ("G-MONTHLY", "monthly", lambda: check_monthly(src, run, heavy=heavy)),
        ("G-TEMPORAL", "temporal", lambda: check_temporal(src)),
        ("G-ABLATION", "ablation", lambda: check_ablation(src, run, monthly_primary)),
    )
    for name, key, fn in checks:
        try:
            metas[key], details[key] = fn()
            extra = {k: v for k, v in details[key].items() if "sha256" not in k}
            gates.append(Gate(name, True, "PASS" + (f" {extra}" if extra else "")))
        except GATE_ERRORS as exc:
            gates.append(Gate(name, False, f"{type(exc).__name__}: {exc}"))
    return gates, metas, details


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def check_git(repo, metas, allow_dirty=False):
    """G-GIT: returns ``(Gate, {scale: unchanged_since_generation})``."""
    tracked = (*ANALYSIS_FILES, "examples/5_Flux_Ensemble/promote_e1_run.py")
    out = _git(repo, "status", "--porcelain", "--", *tracked)
    dirty = sorted(line[3:].strip() for line in out.stdout.splitlines() if line.strip())
    problems = []
    if dirty:
        problems.append("uncommitted: " + ", ".join(dirty))
    unchanged = {}
    for scale in ("daily", "monthly", "temporal"):
        sha = metas.get(scale, {}).get("git", {}).get("sha")
        if not sha:
            continue
        res = _git(repo, "diff", "--quiet", sha, "HEAD", "--", *ANALYSIS_FILES)
        unchanged[scale] = res.returncode == 0
        if res.returncode == 1:
            problems.append(f"{scale}: analysis files changed since {sha[:12]}")
        elif res.returncode != 0:
            problems.append(f"{scale}: generation sha {sha[:12]} unknown to the repo")
    if problems and not allow_dirty:
        return Gate("G-GIT", False, "; ".join(problems) + " (or pass --allow-dirty)"), unchanged
    msg = "PASS" if not problems else "OVERRIDDEN (--allow-dirty): " + "; ".join(problems)
    return Gate("G-GIT", True, msg), unchanged


# ---------------------------------------------------------------------------
# provenance records


def manifest_status(final_dir, rows, run):
    """MATCH when the frozen MANIFEST hashes and run tag describe the sources."""
    path = Path(final_dir) / PACKAGE / MANIFEST
    if not path.is_file():
        return "MISSING", "no frozen MANIFEST"
    manifest = _load_json(path)
    recorded = manifest.get("artifact_sha256", {})
    prefix = f"{PACKAGE}/"
    bad = [
        e.final_rel
        for e, _s, src_sha, _d in rows
        if e.final_rel.startswith(prefix) and recorded.get(e.final_rel[len(prefix) :]) != src_sha
    ]
    tag = manifest.get("internal_archive_id") or previous_run_tag(final_dir)
    if tag != run:
        return "DIFFERS", f"frozen run tag {tag} != {run}"
    if bad:
        return "DIFFERS", f"artifact_sha256 disagrees for {len(bad)} file(s)"
    return "MATCH", "artifact_sha256 and run tag describe the sources"


def ablation_record_status(final_dir, rows, run):
    path = Path(final_dir) / ABLATION_METADATA
    if not path.is_file():
        return "MISSING", "sidecar not yet written (written on --write)"
    meta = _load_json(path)
    files = meta.get("files", {})
    bad = [
        e.final_rel
        for e, _s, src_sha, _d in rows
        if e.package == "ablation" and files.get(e.final_rel, {}).get("sha256") != src_sha
    ]
    if meta.get("internal_archive_id") != run:
        return "DIFFERS", f"sidecar run tag {meta.get('internal_archive_id')} != {run}"
    if bad:
        return "DIFFERS", f"sidecar hashes disagree for {', '.join(bad)}"
    return "MATCH", "sidecar hashes and run tag describe the sources"


def ablation_metadata(src, run, rows, detail, now, head_sha):
    spread_par = Path(src["ablation_spread"]) / f"{ex5_paths.CANONICAL_CONFIG.stem}.3.par.csv"
    run_par = Path(src["run_dir"]) / spread_par.name
    record = {
        "schema_version": "e2_weighting_ablation_provenance/v1",
        "experiment": "E1 (paper numbering) observation-weighting ablation (Table S6, Fig. 4d)",
        "internal_archive_id": run,
        "promoted_at": now,
        "promoted_at_git_sha": head_sha,
        "producer": "examples/5_Flux_Ensemble/run_weighting_ablation.py",
        "promoted_by": "examples/5_Flux_Ensemble/promote_e1_run.py",
        "source_dir": str(src["ablation"]),
        "spread_arm_dir": str(src["ablation_spread"]),
        "files": {
            e.final_rel: {"source": str(e.source), "sha256": src_sha}
            for e, _s, src_sha, _d in rows
            if e.package == "ablation"
        },
        "g_ablation": detail,
        "consumers": [
            "examples/5_Flux_Ensemble/rebuild_e1_benchmark_evidence.py (G-ABLATION)",
            "scripts/figures/build_figure_data.py",
        ],
    }
    if spread_par.is_file() and run_par.is_file():
        a, b = sha256_file(spread_par), sha256_file(run_par)
        record["spread_arm_posterior"] = {
            "spread_arm_par_csv_sha256": a,
            "run_par_csv_sha256": b,
            "identical": a == b,
        }
    return record


def _temporal_cohort(meta):
    c = meta.get("cohort", {})
    counts = c.get("class_row_counts", {})
    return {
        "n_sites": int(c.get("n_common_sites")),
        "n_retrieval_days": int(counts.get("retrieval")),
        "n_between_retrieval_days": int(counts.get("between_retrieval")),
    }


def _cohorts(metas):
    daily, monthly = metas["daily"], metas["monthly"]
    return {
        "daily": {"n_sites": int(daily["n_sites"]), "n_paired_site_days": int(daily["n_pairs"])},
        "monthly": {
            "protocol": pm.MONTHLY_PROTOCOL,
            AGG_POOLED: pm._cohort_entry(monthly, AGG_POOLED, pm.POOLED_MIN_MONTHS),
            AGG_WEIGHTED: pm._cohort_entry(monthly, AGG_WEIGHTED, pm.WEIGHTED_MIN_MONTHS),
            "pooled_only_sites": list(monthly.get("pooled_only_sites", [])),
        },
        "temporal_common": _temporal_cohort(metas["temporal"]),
    }


def build_manifest(
    prior,
    run,
    prev,
    superseded_name,
    src,
    rows,
    metas,
    details,
    repo,
    now,
    head_sha,
    unchanged,
    analysis_clean,
):
    """The prior MANIFEST with every run-dependent field rewritten for ``run``."""
    manifest = json.loads(json.dumps(prior))
    manifest["internal_archive_id"] = run
    manifest["promoted_at"] = now
    manifest["promoted_at_git_sha"] = head_sha
    prior_cohorts = prior.get("cohorts", {})
    manifest["cohorts"] = _cohorts(metas)
    prefix = f"{PACKAGE}/"
    manifest["artifact_sha256"] = dict(
        sorted(
            (e.final_rel[len(prefix) :], src_sha)
            for e, _s, src_sha, _d in rows
            if e.final_rel.startswith(prefix)
        )
    )
    manifest["analysis_code"] = {
        "daily_generation_git_sha": metas["daily"]["git"]["sha"],
        "monthly_generation_git_sha": metas["monthly"]["git"]["sha"],
        "temporal_generation_git_sha": metas["temporal"]["git"]["sha"],
        "current_file_sha256": {rel: sha256_file(Path(repo) / rel) for rel in ANALYSIS_FILES},
        "analysis_files_unchanged_since_generation": unchanged,
        "analysis_files_clean_at_promotion": bool(analysis_clean),
        "promotion_note": (
            f"Whole-run promotion of {run} by promote_e1_run.py. The listed analysis files "
            "are compared between each sidecar's generation sha and HEAD; sidecar "
            "whole-worktree dirty flags count unrelated files."
        ),
    }
    d, m = details["daily"], details["monthly"]
    validation = {
        "promoted_vs_sources": f"PASS: all {len(rows)} files byte-copied; sha256 recorded above",
        "daily_sidecar": (
            "PASS: etf_first_volk_window / openet_flux_2pt1 / volk; grouped metrics, "
            "contrasts and paired record sha256 match; exclusion ledger and per-site table "
            f"agree with the sidecar; posterior par.csv sha256 matches ({run})"
        ),
        "daily_record_grouped_identity": (
            "PASS: maximum absolute difference "
            f"{d.get('record_identity_max_abs_diff')} at tolerance 1e-12"
        ),
        "monthly_sidecar_hashes": (
            "PASS: grouped metrics and contrasts sha256 match the evaluator sidecar"
        ),
        "monthly_replication_from_archive": (
            f"PASS: {pm.N_GROUPED_ESTIMATES} grouped estimates recomputed from "
            f"results/{run}/archive/6_evaluation/site_daily_timeseries; max abs diff "
            f"{m.get('replication_max_abs_diff', float('nan')):.3e} at tolerance "
            f"{pm.REPLICATION_TOL:g}; identical pooled and station-weighted cohorts"
        ),
        "temporal_parent_hash_gate": (
            "PASS: recorded parent artifact and sidecar hashes equal the promoted daily bundle"
        ),
        "weighting_ablation": (
            "PASS: spread arm equals the primary daily and monthly site metrics within "
            f"{ABLATION_TOL:g}; see ../{ABLATION_METADATA}"
        ),
        "bootstrap": prior.get("validation", {}).get(
            "bootstrap", "10,000 whole-site resamples, seed 42, 95% percentile intervals"
        ),
    }
    if "archive_cat6" in d:
        validation["daily_archive_cat6"] = f"PASS: {d['archive_cat6']}"
    manifest["validation"] = validation
    legacy = manifest.setdefault("legacy_and_exclusions", {})
    legacy[superseded_name] = (
        f"paper/data/final/{superseded_name}/ holds the {prev} benchmark record "
        f"(daily, monthly, temporal, MANIFEST) and weighting-ablation files; it must not "
        "be used for current reporting."
    )
    history = list(manifest.get("promotion_history", []))
    history.append(
        {
            "promoted_at": now,
            "git_sha": head_sha,
            "scope": "run",
            "change": f"canonical run {prev} replaced by {run}",
            "from": {
                "internal_archive_id": prev,
                "cohorts": prior_cohorts,
                "moved_to": f"paper/data/final/{superseded_name}/",
            },
            "to": {"internal_archive_id": run, "cohorts": manifest["cohorts"]},
            "source_dirs": {k: str(src[k]) for k in ("daily", "monthly", "temporal", "ablation")},
            "generation_git_sha": {
                k: metas[k]["git"]["sha"] for k in ("daily", "monthly", "temporal")
            },
        }
    )
    manifest["promotion_history"] = history
    return manifest


def superseded_readme(prior, prev, run, moved, now, reason):
    cohorts = prior.get("cohorts", {})
    daily = cohorts.get("daily", {})
    monthly = cohorts.get("monthly", {}).get(AGG_POOLED, {})
    lines = [
        f"# Superseded E1 benchmark record ({prev})",
        "",
        f"Moved here on {now} by `examples/5_Flux_Ensemble/promote_e1_run.py` when the",
        f"canonical Example 5 run changed from `{prev}` to `{run}`.",
        "",
        reason,
        "",
        f"`{PACKAGE}/` holds the {prev} daily ({daily.get('n_sites')} sites, "
        f"{daily.get('n_paired_site_days')} site-days), monthly (pooled "
        f"{monthly.get('n_sites')} sites, {monthly.get('n_paired_site_months')} months) and",
        "temporal packages with their MANIFEST (promoted at "
        f"{prior.get('promoted_at', '?')}, sha {prior.get('promoted_at_git_sha', '?')}). The",
        "`e2_weighting_ablation_*` files are the observation-weighting ablation scored on",
        f"{prev}. The current record lives in `paper/data/final/{PACKAGE}/` and",
        "`paper/data/final/e2_weighting_ablation_*`. **Do not use these files for current",
        "reporting.** `reconstruction_fidelity/` is run-independent and was not moved.",
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


def _owned_existing(final_dir, entries):
    """Frozen files this script owns and would move: package files, MANIFEST, sidecar."""
    final_dir = Path(final_dir)
    owned = [e.final_rel for e in entries if (final_dir / e.final_rel).is_file()]
    for rel in (f"{PACKAGE}/{MANIFEST}", ABLATION_METADATA):
        if (final_dir / rel).is_file():
            owned.append(rel)
    return owned


def promote(
    final_dir,
    src,
    run,
    rows,
    metas,
    details,
    superseded_name,
    repo,
    now,
    head_sha,
    unchanged,
    analysis_clean,
    reason=DEFAULT_REASON,
):
    """Move the frozen record aside, copy the sources in, write MANIFEST + sidecar + README."""
    final_dir = Path(final_dir).resolve()
    superseded = final_dir / superseded_name
    pm._refuse_if_exists(superseded, "superseded directory")
    for e, *_ in rows:
        if final_dir in Path(e.source).resolve().parents:
            raise PromotionError(f"source {e.source} lies inside {final_dir}")
    entries = [e for e, *_ in rows]
    prior_path = final_dir / PACKAGE / MANIFEST
    prior = _load_json(prior_path) if prior_path.is_file() else {}
    prev = previous_run_tag(final_dir) if prior else None
    manifest = build_manifest(
        prior,
        run,
        prev,
        superseded_name,
        src,
        rows,
        metas,
        details,
        repo,
        now,
        head_sha,
        unchanged,
        analysis_clean,
    )
    sidecar = ablation_metadata(src, run, rows, details["ablation"], now, head_sha)

    moved = {}
    for rel in _owned_existing(final_dir, entries):
        target = superseded / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        moved[rel] = sha256_file(final_dir / rel)
        shutil.move(str(final_dir / rel), str(target))
    if moved:
        (superseded / "README.md").write_text(
            superseded_readme(prior, prev, run, moved, now, reason)
        )
    for e in entries:
        (final_dir / e.final_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.source, final_dir / e.final_rel)
    prior_path.write_bytes(pm._json_bytes(manifest))
    (final_dir / ABLATION_METADATA).write_bytes(pm._json_bytes(sidecar))
    return manifest, moved


# ---------------------------------------------------------------------------
# report


def report(rows, gates, manifest_row, sidecar_row):
    for g in gates:
        msg = g.message[4:].strip() if g.message.startswith("PASS") else g.message
        print(f"{g.name:12s} {'PASS' if g.ok else 'FAIL'}  {msg}")
    print()
    print(f"{'package':9s} {'status':9s} {'source':12s} {'final':12s}  file <- source")
    for e, status, src_sha, dst_sha in rows:
        print(
            f"{e.package:9s} {status:9s} {(src_sha or '-')[:12]:12s} {(dst_sha or '-')[:12]:12s}"
            f"  {e.final_rel} <- {e.source}"
        )
    print(
        f"{'manifest':9s} {manifest_row[0]:9s} {'':12s} {'':12s}  {PACKAGE}/{MANIFEST}: {manifest_row[1]}"
    )
    print(
        f"{'ablation':9s} {sidecar_row[0]:9s} {'':12s} {'':12s}  {ABLATION_METADATA}: {sidecar_row[1]}"
    )
    for sub in UNTOUCHED_SUBPACKAGES:
        print(f"{'-':9s} {'UNTOUCHED':9s} {'':12s} {'':12s}  {PACKAGE}/{sub}/ (run-independent)")


def promotion_state(rows, gates, manifest_row, sidecar_row):
    ok = all(g.ok for g in gates)
    data = all(status == "MATCH" for _, status, _, _ in rows)
    return "MATCH" if ok and data and manifest_row[0] == sidecar_row[0] == "MATCH" else "DIFFERS"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", default=ex5_paths.CANONICAL_RUN, help="run tag (default: %(default)s)")
    ap.add_argument("--final-dir", default=str(ex5_paths.FINAL_DIR))
    ap.add_argument("--daily-source-dir", default=None, help=f"default: <run>/{DAILY_SUBDIR}")
    ap.add_argument("--monthly-source-dir", default=None, help=f"default: <run>/{MONTHLY_SUBDIR}")
    ap.add_argument("--temporal-source-dir", default=None, help=f"default: <run>/{TEMPORAL_SUBDIR}")
    ap.add_argument(
        "--ablation-source-dir", default=None, help="default: <results>/ablation_<run>_summary"
    )
    ap.add_argument(
        "--ablation-spread-dir", default=None, help="default: <results>/ablation_<run>_e1_spread"
    )
    ap.add_argument(
        "--ablation-monthly-primary",
        default=None,
        help="per-site monthly table the spread arm must equal (default: the monthly source's "
        "evaluation_monthly_metrics.csv; Run 22's ablation is on the 28-day rule, so pass "
        "superseded_e1_monthly_28day/evaluation_monthly_metrics.csv for it)",
    )
    ap.add_argument("--archive-dir", default=None, help="default: <run>/archive/6_evaluation")
    ap.add_argument(
        "--superseded-name",
        default=None,
        help=f"default: {SUPERSEDED_TEMPLATE.format(prev='<previous run tag>')}",
    )
    ap.add_argument("--supersede-reason", default=DEFAULT_REASON, help="README reason text")
    ap.add_argument("--allow-dirty", action="store_true", help="override G-GIT")
    ap.add_argument("--write", action="store_true", help="perform the promotion")
    args = ap.parse_args(argv)

    run, final_dir = args.run, Path(args.final_dir)
    src = default_sources(run, ex5_paths.results_root(ex5_paths.load_config()))
    for key, override in (
        ("daily", args.daily_source_dir),
        ("monthly", args.monthly_source_dir),
        ("temporal", args.temporal_source_dir),
        ("ablation", args.ablation_source_dir),
        ("ablation_spread", args.ablation_spread_dir),
        ("archive", args.archive_dir),
    ):
        if override:
            src[key] = Path(override)

    prev = previous_run_tag(final_dir)
    superseded_name = args.superseded_name
    if superseded_name is None:
        if prev == run:
            superseded_name = None
        else:
            superseded_name = SUPERSEDED_TEMPLATE.format(prev=prev)
    print(f"run {run}; frozen record {prev}; superseded dir {superseded_name or '(none)'}")

    entries = package_entries(src)
    gates, metas, details = run_gates(
        src, run, heavy=True, monthly_primary=args.ablation_monthly_primary
    )
    missing = [str(e.source) for e in entries if not e.source.is_file()]
    gates.insert(
        0,
        Gate("G-EXIST", not missing, "PASS" if not missing else f"{len(missing)} missing"),
    )
    git_gate, unchanged = check_git(ex5_paths.REPO, metas, allow_dirty=args.allow_dirty)
    gates.insert(0, git_gate)
    if superseded_name is not None:
        exists = (final_dir / superseded_name).exists()
        gates.append(
            Gate(
                "G-SUPERSEDE",
                not exists,
                "PASS" if not exists else f"{final_dir / superseded_name} already exists",
            )
        )

    rows = compare_entries(entries, final_dir)
    manifest_row = manifest_status(final_dir, rows, run)
    sidecar_row = ablation_record_status(final_dir, rows, run)
    report(rows, gates, manifest_row, sidecar_row)
    state = promotion_state(rows, gates, manifest_row, sidecar_row)

    if args.write:
        failed = [g for g in gates if not g.ok]
        if failed:
            raise PromotionError("gates failed: " + "; ".join(f"{g.name}" for g in failed))
        if all(status == "MATCH" for _, status, _, _ in rows):
            if sidecar_row[0] != "MATCH" and manifest_row[0] == "MATCH":
                sidecar = ablation_metadata(
                    src, run, rows, details["ablation"], _now(), pm._head_sha(ex5_paths.REPO)
                )
                (final_dir / ABLATION_METADATA).write_bytes(pm._json_bytes(sidecar))
                print(f"data unchanged; wrote {ABLATION_METADATA}")
                print("PROMOTION_STATE: MATCH")
                return 0
            if manifest_row[0] != "MATCH":
                raise PromotionError(
                    "frozen files match the sources but the MANIFEST does not describe them"
                )
            print("nothing to promote")
            print(f"PROMOTION_STATE: {state}")
            return 0
        if superseded_name is None:
            raise PromotionError(
                f"the frozen record is already {run}; pass --superseded-name to re-promote"
            )
        clean = git_gate.message == "PASS"
        manifest, moved = promote(
            final_dir,
            src,
            run,
            rows,
            metas,
            details,
            superseded_name,
            ex5_paths.REPO,
            _now(),
            pm._head_sha(ex5_paths.REPO),
            unchanged,
            clean,
            reason=args.supersede_reason,
        )
        print(f"moved {len(moved)} files to {final_dir / superseded_name}")
        print(
            f"promoted {len(rows)} files; MANIFEST internal_archive_id {manifest['internal_archive_id']}"
        )
        state = "MATCH"
    print(f"PROMOTION_STATE: {state}")
    return 0 if state == "MATCH" else 1


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PromotionError as exc:
        print(f"PROMOTION REFUSED: {exc}", file=sys.stderr)
        print("PROMOTION_STATE: DIFFERS")
        sys.exit(2)

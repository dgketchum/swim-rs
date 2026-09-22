"""Promote the Example 6 results into ``paper/data/final/e2_closure_pool/`` (paper E2 + E0).

The package is the paper's source of record for Tables 3, 5 and S8, Figs 2 and 5b,
and §3.3/S9.1. It was first frozen by hand on 2026-09-22 (HWSD AWC recal); this
script is its tracked producer. Every file is a byte copy of a results-root output:

    <canonical>/archive/6_evaluation/closure_pool/*      -> closure_pool/*
        closure_pool_summary.py + monthly_28day_sensitivity.py outputs (Table 5, S8, §3.3, S9.1)
    <results>/e2_run22_transfer_by_irrigation_to_grassbasis/
        pooled_metrics_{daily,monthly}[_strat].csv, transfer_comparison_*.csv,
        transfer_winrates.csv, run_metadata.json           -> transfer/*
        transfer_ex5_params.py outputs (Fig 5b; the uncalibrated row)
    <canonical>/transfer_refresh/e3_irrigation_stratified_param_mapping*.json
                                                           -> transfer/*
        transfer/build_e2_irrigation_mapping.py output (the by-irrigation class map)
    <results>/e0_disjoint/<pair>/<disjoint37|pool47>/*     -> e0_disjoint/<pair>/<tag>/*
        pooled_arm_compare.py gates for the three cover-formulation arms (Table 3, Fig 2)

``MANIFEST.json`` (schema ``e2_closure_pool_reporting/v1``) records the sha256 and
source of every file, the run SHAs from ``archive/1_provenance/git_sha.txt``, and the
headline values re-read from the promoted CSV/JSON. Its prose fields (``reason``,
``e0_gate_note``, ``irrigation_mapping_change``, ``retired``, ...) were written at
freeze time and are carried over from the existing manifest; edit them there.

The ``e3_*`` filenames are the legacy namespace the figure builder keys on and are
kept. The default mode compares every source against the promoted copy byte-for-byte
and the regenerated manifest against the frozen one field by field; ``--write``
replaces the files and rewrites the manifest.

Usage:
    uv run python examples/6_Flux_International/promote_final.py            # check
    uv run python examples/6_Flux_International/promote_final.py --write
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ex6_paths  # noqa: E402

PACKAGE = "e2_closure_pool"
SCHEMA = "e2_closure_pool_reporting/v1"
TRANSFER_RUN = "e2_run22_transfer_by_irrigation_to_grassbasis"
E0_ARMS = {"e0_arm_fao56_sig": "fao56_sig", "e0_arm_fao56": "fao56"}
E0_PAIRS = ("grassbasis_vs_fao56_sig", "grassbasis_vs_fao56", "fao56_sig_vs_fao56")
E0_TAGS = ("disjoint37", "pool47")
TRANSFER_FILES = (
    "transfer_comparison_persite.csv",
    "transfer_comparison_summary.csv",
    "transfer_winrates.csv",
    "pooled_metrics_daily.csv",
    "pooled_metrics_daily_strat.csv",
    "pooled_metrics_monthly.csv",
    "pooled_metrics_monthly_strat.csv",
    "run_metadata.json",
)
MAPPING_FILES = (
    "e3_irrigation_stratified_param_mapping.json",
    "e3_irrigation_stratified_param_mapping_metadata.json",
)
POOL_KEYS = (
    "decision",
    "pool_definition",
    "n_daily_sites_all",
    "n_daily_sites_pool",
    "n_daily_sites_raw_excluded",
    "n_monthly_rows_pool",
    "n_monthly_finite_pool",
    "n_paired_days_pool",
    "irrigation_class_pool",
    "configured_pool_note",
    "bootstrap",
)
HEADLINE_COLS = (
    "n_rows",
    "n_sites",
    "r2_median",
    "r2_mean",
    "kge_median",
    "kge_mean",
    "rmse_median",
    "rmse_mean",
    "bias_median",
    "bias_mean",
)
GATE_KEYS = ("arm_a", "arm_b", "n_sites", "n_daily", "n_monthly", "wins_a", "gate_rule", "passed")
# Prose written at freeze time; carried over from the existing manifest verbatim.
CARRIED_NOTES = ("e0_gate_note", "irrigation_mapping_change", "closure_pool_problems")
CARRIED_TAIL = ("retired", "pending_downstream_consumers")
# Sibling frozen files whose sha256 the manifest records for the audit.
RELATED_PACKAGES = {
    "e2_evidence_metadata.json": (
        "paper E1 (legacy e2_* namespace) — independent of this package and "
        "unchanged by the recalibration; sha256 recorded below for the audit"
    ),
    "e2_run22_transfer_vector.json": None,
    "e2_run22_transfer_vectors_by_irrigation.json": None,
}
VOLATILE_KEYS = ("promoted_at", "promoted_at_git_sha")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def collect_sources(run_dir, transfer_dir, e0_dir):
    """Return {package-relative path: source Path} for every promoted file."""
    run_dir, transfer_dir, e0_dir = Path(run_dir), Path(transfer_dir), Path(e0_dir)
    sources = {}
    closure = run_dir / "archive" / "6_evaluation" / "closure_pool"
    for src in sorted(p for p in closure.iterdir() if p.is_file()):
        sources[f"closure_pool/{src.name}"] = src
    for name in TRANSFER_FILES:
        sources[f"transfer/{name}"] = transfer_dir / name
    for name in MAPPING_FILES:
        sources[f"transfer/{name}"] = run_dir / "transfer_refresh" / name
    for pair in sorted(E0_PAIRS):
        for tag in E0_TAGS:
            d = e0_dir / pair / tag
            for src in sorted(p for p in d.iterdir() if p.is_file()):
                sources[f"e0_disjoint/{pair}/{tag}/{src.name}"] = src
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing source files:\n  " + "\n  ".join(missing))
    return sources


def compare(sources, final_dir):
    """Return [(name, status, sha256)] where status is 'match', 'differs', or 'missing'."""
    rows = []
    for name, src in sources.items():
        data = src.read_bytes()
        target = Path(final_dir) / name
        if not target.exists():
            status = "missing"
        else:
            status = "match" if target.read_bytes() == data else "differs"
        rows.append((name, status, sha256_bytes(data)))
    return rows


def _git_sha(run_dir):
    p = Path(run_dir) / "archive" / "1_provenance" / "git_sha.txt"
    return p.read_text().strip() if p.exists() else None


def _head_sha(repo):
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _source_runs(prior, results_root, run_dir, transfer_dir, e0_dir):
    runs = {"canonical": run_dir}
    for key, suffix in E0_ARMS.items():
        runs[key] = Path(results_root) / f"{ex6_paths.CANONICAL_RUN}_{suffix}"
    runs["transfer"] = transfer_dir
    runs["e0_disjoint"] = e0_dir
    out = {}
    for key, path in runs.items():
        entry = {"path": str(path)}
        sha = _git_sha(path)
        if sha:
            entry["git_sha"] = sha
        for carried in ("supersedes", "note"):
            if carried in prior.get(key, {}):
                entry[carried] = prior[key][carried]
        out[key] = entry
    return out


def _headline(sources):
    df = pd.read_csv(sources["closure_pool/headline_aggregates.csv"])
    df = df[df["tier"] == "closure_corrected"]
    out = {}
    for basis, grp in df.groupby("basis", sort=False):
        out[f"{basis}_closure_corrected"] = {
            row["model"]: {c: float(row[c]) for c in HEADLINE_COLS} for _, row in grp.iterrows()
        }
    return out


def _e0_gates(sources):
    out = {}
    for pair in E0_PAIRS:
        for tag in E0_TAGS:
            gate = json.loads(sources[f"e0_disjoint/{pair}/{tag}/pooled_gate.json"].read_text())
            out[f"{pair}/{tag}"] = {k: gate[k] for k in GATE_KEYS}
    return out


def build_manifest(sources, prior, results_root, run_dir, transfer_dir, e0_dir, final_dir):
    """Regenerate MANIFEST.json; ``prior`` supplies the carried prose fields."""
    pool_meta = json.loads(sources["closure_pool/closure_pool_metadata.json"].read_text())
    uncal = dict(pool_meta["uncalibrated_baseline"])
    uncal["canonical_row"] = pd.read_csv(
        sources["closure_pool/uncalibrated_baseline_summary.csv"]
    ).to_dict(orient="records")
    if "note" in prior.get("uncalibrated_baseline", {}):
        uncal["note"] = prior["uncalibrated_baseline"]["note"]
    related = {}
    for name, note in RELATED_PACKAGES.items():
        digest = sha256_bytes((Path(final_dir).parent / name).read_bytes())
        if note:
            related[name] = note
            related[f"{Path(name).stem}_sha256"] = digest
        else:
            related[name] = digest
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_for_results_reporting",
        "experiment": "E2 (paper numbering); internal Example 6 / legacy e3_* namespace",
        "scope": prior.get("scope"),
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "promoted_at_git_sha": _head_sha(ex6_paths.REPO),
        "reason": prior.get("reason"),
        "source_runs": _source_runs(
            prior.get("source_runs", {}), results_root, run_dir, transfer_dir, e0_dir
        ),
        "pool": {k: pool_meta[k] for k in POOL_KEYS},
        "uncalibrated_baseline": uncal,
        "transfer_refresh_summary": pd.read_csv(
            sources["closure_pool/transfer_refresh_summary.csv"]
        ).to_dict(orient="records"),
        "headline_closure_corrected": _headline(sources),
        "e0_pooled_gates": _e0_gates(sources),
    }
    manifest.update({k: prior[k] for k in CARRIED_NOTES if k in prior})
    manifest["related_frozen_packages"] = related
    manifest.update({k: prior[k] for k in CARRIED_TAIL if k in prior})
    manifest["files"] = {
        name: {
            "source": str(src),
            "sha256": sha256_bytes(src.read_bytes()),
            "bytes": src.stat().st_size,
        }
        for name, src in sources.items()
    }
    return manifest


def _json_bytes(obj):
    return (json.dumps(obj, indent=2) + "\n").encode()


def manifest_diff(new, prior):
    """Top-level keys whose regenerated value differs from the frozen manifest."""
    keys = [k for k in dict.fromkeys([*prior, *new]) if k not in VOLATILE_KEYS]
    return [k for k in keys if json.loads(_json_bytes(new.get(k))) != prior.get(k)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="default: the canonical Ex6 TOML")
    ap.add_argument("--results-root", default=None, help="default: {project_ws}/results")
    ap.add_argument("--run-dir", default=None, help="default: <results-root>/<canonical run>")
    ap.add_argument("--transfer-dir", default=None, help=f"default: <results-root>/{TRANSFER_RUN}")
    ap.add_argument("--e0-dir", default=None, help="default: <results-root>/e0_disjoint")
    ap.add_argument("--final-dir", default=str(ex6_paths.FINAL_DIR / PACKAGE))
    ap.add_argument("--write", action="store_true", help="replace the promoted files + manifest")
    args = ap.parse_args()
    if args.results_root is None:
        args.results_root = ex6_paths.results_root(ex6_paths.load_config(args.config))
    results_root = Path(args.results_root)
    run_dir = Path(args.run_dir or results_root / ex6_paths.CANONICAL_RUN)
    transfer_dir = Path(args.transfer_dir or results_root / TRANSFER_RUN)
    e0_dir = Path(args.e0_dir or results_root / "e0_disjoint")
    final_dir = Path(args.final_dir)

    sources = collect_sources(run_dir, transfer_dir, e0_dir)
    rows = compare(sources, final_dir)
    for name, status, digest in rows:
        print(f"{status:8s} {digest[:16]}  {name}")

    manifest_path = final_dir / "MANIFEST.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = build_manifest(
        sources, prior, results_root, run_dir, transfer_dir, e0_dir, final_dir
    )
    changed = manifest_diff(manifest, prior)
    print(f"MANIFEST.json: {'unchanged' if not changed else 'differs in ' + ', '.join(changed)}")

    if args.write:
        for name, src in sources.items():
            target = final_dir / name
            os.makedirs(target.parent, exist_ok=True)
            target.write_bytes(src.read_bytes())
        manifest_path.write_bytes(_json_bytes(manifest))
        print(f"wrote {len(sources)} files + MANIFEST.json to {final_dir}")
    elif changed or any(status != "match" for _, status, _ in rows):
        print("promoted package differs from the results root (run with --write to replace)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

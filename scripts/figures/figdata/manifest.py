"""BuildError, hashing and gate helpers, git stamps, and the package manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import paths
from .cohorts import EXPECTED, EXPERIMENT_MAP, METRIC_DEFS
from .paths import REPO, SCRIPT_VERSION


class BuildError(RuntimeError):
    """Raised when a frozen table cannot be built to specification."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha256(path: Path, pattern: str = "*") -> str:
    """Stable hash over a directory's file names + contents (sorted)."""
    h = hashlib.sha256()
    for p in sorted(path.glob(pattern)):
        if p.is_file():
            h.update(p.name.encode())
            h.update(sha256(p).encode())
    return h.hexdigest()


def require_columns(df: pd.DataFrame, cols, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise BuildError(f"{label}: missing required columns {missing}")


def require_unique(df: pd.DataFrame, keys, label: str) -> None:
    dup = df.duplicated(subset=list(keys)).sum()
    if dup:
        raise BuildError(f"{label}: {dup} duplicate rows on key {list(keys)}")


def require_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise BuildError(f"{label}: expected {expected}, got {actual}")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(REPO), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:  # pragma: no cover
        return True


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


class Manifest:
    def __init__(self) -> None:
        self.tables: dict[str, dict] = {}
        self.status: list[dict] = []
        # Set by main() on an --only run: the builder names that ran. A partial
        # run merges the prior on-disk manifest so records for builders that did
        # not run are carried forward instead of dropped.
        self.partial_targets: list[str] | None = None

    def add(self, name: str, **meta) -> None:
        meta.setdefault("generator_script", "scripts/figures/build_figure_data.py")
        meta.setdefault("generator_version", SCRIPT_VERSION)
        meta.setdefault("frozen_utc", datetime.now(UTC).isoformat())
        self.tables[name] = meta

    def block(self, item: str, reason: str, detail: str = "") -> None:
        self.status.append({"item": item, "reason": reason, "detail": detail})

    # Files the manifest self-hashes (everything the builders emit into paths.OUT).
    OUTPUT_GLOBS = ("fig*.csv", "fig*.json", "fig*.gpkg")
    MANIFEST_NAME = "fig_manifest.json"

    def _output_hashes(self) -> dict[str, dict]:
        """SHA256 + size for every file produced into ``OUT``, computed after write.

        The manifest cannot hash itself, so ``fig_manifest.json`` is excluded;
        every other emitted artifact -- including ``fig01_scope.gpkg`` -- gets an
        independently verifiable digest (bug D4).
        """
        found: dict[str, Path] = {}
        for pattern in self.OUTPUT_GLOBS:
            for p in paths.OUT.glob(pattern):
                if p.is_file() and p.name != self.MANIFEST_NAME:
                    found[p.name] = p
        return {
            name: {"sha256": sha256(found[name]), "bytes": found[name].stat().st_size}
            for name in sorted(found)
        }

    def write(self) -> Path:
        outputs = self._output_hashes()
        tables = self.tables
        status = self.status
        if self.partial_targets is not None:
            prior_path = paths.OUT / self.MANIFEST_NAME
            if not prior_path.exists():
                raise RuntimeError(
                    "--only run requires an existing fig_manifest.json to merge into; "
                    "run --all first to establish the full package manifest"
                )
            prior = json.loads(prior_path.read_text())
            # Carry forward table records for builders that did not run, but only
            # while their output file is still present on disk.
            tables = {
                name: meta
                for name, meta in prior.get("tables", {}).items()
                if name not in self.tables and name in outputs
            }
            tables.update(self.tables)
            # Carry forward blocked/incomplete records unless this run replaced
            # them or successfully re-ran the builder they describe.
            current_items = {s["item"] for s in self.status}
            ran = set(self.partial_targets)
            status = [
                s
                for s in prior.get("blocked_or_incomplete", [])
                if s["item"] not in current_items and s["item"] not in ran
            ] + self.status
        for name, meta in tables.items():
            if name in outputs:
                meta["output_sha256"] = outputs[name]["sha256"]
                meta["output_bytes"] = outputs[name]["bytes"]
            else:
                meta["output_sha256"] = None
                meta["output_bytes"] = None
        payload = {
            "package": "SWIM-RS six-figure display package",
            "phase": "Phase 1 - frozen figure data contract",
            "generator_script": "scripts/figures/build_figure_data.py",
            "generator_version": SCRIPT_VERSION,
            "frozen_utc": datetime.now(UTC).isoformat(),
            "repo_git_sha": git_sha(),
            "repo_worktree_dirty": git_dirty(),
            "governing_documents": [
                "paper/notes/figure_production_handoff.md",
                "paper/notes/six_figure_plan.md",
                "paper/notes/fig01_production_handoff.md",
                "paper/text/main.md",
                "paper/text/supp.md",
            ],
            "legacy_to_current_experiment_map": EXPERIMENT_MAP,
            "removed_experiment_note": (
                "The legacy e1_* artifacts under paper/data/final/ describe the removed "
                "broad-land-cover experiment. They are never read by this builder and must "
                "not appear in any reader-facing figure."
            ),
            "metric_definitions": METRIC_DEFS,
            "nse_vs_r2_guard": (
                "Every legacy r2_* column mapped to NSE in this package was verified "
                "against the evaluator implementation before renaming; see "
                "metric_definitions.nse.verification. No column was renamed on the basis "
                "of its name alone."
            ),
            "expected_cohort_counts": EXPECTED,
            "output_files": outputs,
            "output_hash_note": (
                "sha256 of every file this builder emitted into the package directory, "
                "computed after the file was written. fig_manifest.json itself is excluded "
                "because it cannot contain its own digest; hash it directly to verify. Each "
                "tables[*] entry repeats its own digest as output_sha256."
            ),
            "last_run": {
                "partial": self.partial_targets is not None,
                "targets": self.partial_targets or "all",
            },
            "tables": tables,
            "blocked_or_incomplete": status,
        }
        p = paths.OUT / "fig_manifest.json"
        p.write_text(json.dumps(payload, indent=2, sort_keys=False))
        return p


MANIFEST = Manifest()


def write_table(df: pd.DataFrame, name: str) -> int:
    path = paths.OUT / name
    df.to_csv(path, index=False)
    return len(df)

"""Verify that the sources recorded in ``fig_manifest.json`` are still the files on disk.

Every table record names the artifacts it was built from, in one of three shapes:
a ``{"path": ..., "sha256": ...}`` record, a ``sources[label]`` path paired with a
``source_hashes[label]`` digest, or a ``source_package`` whose ``MANIFEST.json``
digest is ``source_manifest_sha256``.  ``check_sources`` re-hashes each one, so a
package whose inputs have moved on since it was frozen is detectable without a
rebuild.  Recorded absolute paths from the authoring machine are re-rooted with
``swimrs.swim.reproduce.rebase``; recorded relative paths are below the repository.

This module imports nothing that needs the data root, so it runs anywhere the
manifest does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from swimrs.swim import reproduce


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_records(obj, label: str) -> list[tuple[str, str, str]]:
    """Every ``{"path", "sha256"}`` record below ``obj`` as ``(label, path, sha256)``."""
    out: list[tuple[str, str, str]] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str):
            out.append((label, obj["path"], obj["sha256"]))
        else:
            for k, v in obj.items():
                out.extend(_path_records(v, f"{label}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_path_records(v, f"{label}[{i}]"))
    return out


def table_sources(record: dict) -> list[tuple[str, str, str]]:
    """The ``(label, path, sha256)`` triples one manifest table record declares."""
    triples = _path_records(record.get("sources", {}), "sources")
    triples += _path_records(record.get("container_source", {}), "container_source")
    sources, hashes = record.get("sources", {}), record.get("source_hashes", {})
    if isinstance(sources, dict) and isinstance(hashes, dict):
        for label, digest in hashes.items():
            path = sources.get(label)
            if isinstance(path, str) and isinstance(digest, str):
                triples.append((f"source_hashes.{label}", path, digest))
    package, digest = record.get("source_package"), record.get("source_manifest_sha256")
    if isinstance(package, str) and isinstance(digest, str):
        triples.append(("source_manifest_sha256", f"{package}/MANIFEST.json", digest))
    return triples


def resolve(path: str, repo: Path, root: str | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return repo / p
    return reproduce.rebase(p, root=root, repo=repo)


def check_sources(manifest_path: Path, repo: Path, root: str | None = None):
    """``(n_checked, problems)`` for every source the manifest records with a digest."""
    manifest = json.loads(Path(manifest_path).read_text())
    problems: list[str] = []
    n = 0
    for table, record in manifest.get("tables", {}).items():
        for label, path, digest in table_sources(record):
            n += 1
            p = resolve(path, repo, root)
            if not p.exists():
                problems.append(f"{table} {label}: missing {p}")
            elif _sha256(p) != digest:
                problems.append(f"{table} {label}: digest changed {p}")
    return n, problems

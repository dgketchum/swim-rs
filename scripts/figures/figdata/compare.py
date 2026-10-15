"""Compare a rebuilt figure-data package with the frozen one.

The builder is deterministic: every table it emits is byte-identical between two
builds of the same sources on the same machine.  On other hardware the last digit
of a float can differ, so a table that is not byte-identical is accepted when it is
equal within floating-point round-off (``swimrs.swim.reproduce.equivalent``).  Beyond
that, only three things legitimately differ, and they are the only things this
comparison ignores:

* build stamps: ``frozen_utc`` (in ``fig_manifest.json`` and the ``fig0*_metadata.json``
  sidecars) and the manifest's ``repo_git_sha`` / ``repo_worktree_dirty``;
* absolute paths below the output directory (the sidecars record where superseded
  packages were moved); they are compared relative to each package's directory;
* the GeoPackage's internal write metadata; ``fig01_scope.gpkg`` is compared layer by
  layer (attributes and geometry) instead of by bytes.

The manifest repeats the digest and size of every output file; those records are
dropped for files that compare equal under the rules above and keep their byte-level
meaning for every other file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from swimrs.swim import reproduce

OUTPUT_GLOBS = ("fig*.csv", "fig*.json", "fig*.gpkg")
MANIFEST_NAME = "fig_manifest.json"
VOLATILE_KEYS = frozenset({"frozen_utc", "repo_git_sha", "repo_worktree_dirty"})
DIGEST_KEYS = ("output_sha256", "output_bytes", "sha256", "bytes")


def _normalize(obj, out_dir: str):
    """``obj`` with build stamps dropped, output-dir paths made relative, NaN made comparable."""
    if isinstance(obj, dict):
        return {k: _normalize(v, out_dir) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_normalize(v, out_dir) for v in obj]
    if isinstance(obj, str) and obj.startswith(out_dir):
        return "<OUT>" + obj[len(out_dir) :]
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    return obj


def _json_equal(a: Path, b: Path, drop_digests: set[str]) -> bool:
    da = _normalize(json.loads(a.read_text()), str(a.parent.resolve()))
    db = _normalize(json.loads(b.read_text()), str(b.parent.resolve()))
    for d in (da, db):
        for section in ("tables", "output_files"):
            for name in drop_digests:
                rec = d.get(section, {}).get(name)
                if isinstance(rec, dict):
                    for key in DIGEST_KEYS:
                        rec.pop(key, None)
    if da == db:
        return True
    return reproduce.equivalent(
        json.dumps(da, sort_keys=True), json.dumps(db, sort_keys=True), a.name
    )


def _gpkg_equal(a: Path, b: Path) -> bool:
    import fiona
    import geopandas as gpd

    layers = fiona.listlayers(a)
    if layers != fiona.listlayers(b):
        return False
    for layer in layers:
        ga = gpd.read_file(a, layer=layer, engine="fiona")
        gb = gpd.read_file(b, layer=layer, engine="fiona")
        if list(ga.columns) != list(gb.columns) or len(ga) != len(gb) or ga.crs != gb.crs:
            return False
        if not ga.drop(columns="geometry").equals(gb.drop(columns="geometry")):
            return False
        if not ga.geometry.geom_equals_exact(gb.geometry, tolerance=0).all():
            return False
    return True


def package_differences(built: Path, frozen: Path) -> list[str]:
    """Every builder output in ``frozen`` that ``built`` does not reproduce (empty = match).

    Files the builder does not emit (``BUILD_STATUS.md``, superseded subdirectories,
    renderer-side tables) are not compared.
    """
    built, frozen = Path(built), Path(frozen)
    names = sorted({p.name for g in OUTPUT_GLOBS for p in frozen.glob(g) if p.is_file()})
    extra = sorted(
        {p.name for g in OUTPUT_GLOBS for p in built.glob(g) if p.is_file()} - set(names)
    )
    diffs = [f"{n}: not in the frozen package" for n in extra]
    stamped_equal: set[str] = set()
    manifest_pending = False
    for name in names:
        a, b = built / name, frozen / name
        if not a.exists():
            diffs.append(f"{name}: not built")
            continue
        if a.read_bytes() == b.read_bytes():
            continue
        if name == MANIFEST_NAME:
            manifest_pending = True
            continue
        if name.endswith(".json") and _json_equal(a, b, set()):
            stamped_equal.add(name)
        elif name.endswith(".gpkg") and _gpkg_equal(a, b):
            stamped_equal.add(name)
        elif name.endswith(".csv") and reproduce.equivalent(a.read_bytes(), b.read_bytes(), name):
            stamped_equal.add(name)
        else:
            diffs.append(f"{name}: content differs")
    if manifest_pending and not _json_equal(
        built / MANIFEST_NAME, frozen / MANIFEST_NAME, stamped_equal
    ):
        diffs.append(f"{MANIFEST_NAME}: content differs beyond build stamps")
    return diffs

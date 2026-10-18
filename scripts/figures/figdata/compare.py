"""Compare a rebuilt figure-data package with the frozen one.

The builder is deterministic: every table it emits is byte-identical between two
builds of the same sources on the same machine.  On other hardware the last digit
of a float can differ, so a table that is not byte-identical is accepted when it is
equal within floating-point round-off (``swimrs.swim.reproduce.equivalent``).  Beyond
that, only four things legitimately differ, and they are the only things this
comparison ignores:

* build stamps: ``frozen_utc`` (in ``fig_manifest.json`` and the ``fig0*_metadata.json``
  sidecars) and the manifest's ``repo_git_sha`` / ``repo_worktree_dirty``;
* recorded absolute paths: below the output directory they are compared as if
  written at the frozen location (``paper/data/final/figures/...``, which is what
  the frozen package records on the authoring machine), elsewhere from the
  data-root or repository marker on (``swimrs.swim.reproduce.portable_key``), so a
  scratch ``--out`` on another machine's clone and data root compares equal;
* digests and sizes (``*sha256``, ``*bytes`` keys and columns, 64-hex values;
  ``swimrs.swim.reproduce.digest``) of the inputs the sidecars hash and of the
  outputs the manifest hashes: this comparison judges the table content itself,
  and ``build_figure_data.py --check`` re-hashes the recorded inputs;
* the GeoPackage's internal write metadata; ``fig01_scope.gpkg`` is compared layer by
  layer (attributes and geometry) instead of by bytes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from swimrs.swim import reproduce

OUTPUT_GLOBS = ("fig*.csv", "fig*.json", "fig*.gpkg")
MANIFEST_NAME = "fig_manifest.json"
FROZEN_KEY = "paper/data/final/figures"
VOLATILE_KEYS = frozenset({"frozen_utc", "repo_git_sha", "repo_worktree_dirty"})


def _normalize(obj, out_dir: str):
    """``obj`` with build stamps and digests dropped, paths made machine-independent,
    NaN made comparable."""
    if isinstance(obj, dict):
        return {
            _normalize(k, out_dir): _normalize(v, out_dir)
            for k, v in obj.items()
            if k not in VOLATILE_KEYS and not reproduce.digest(k, v)
        }
    if isinstance(obj, list):
        return [_normalize(v, out_dir) for v in obj]
    if isinstance(obj, str) and obj.startswith(out_dir):
        return FROZEN_KEY + obj[len(out_dir) :]
    if isinstance(obj, str) and obj.startswith("/"):
        return reproduce.portable_key(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    return obj


def _json_equal(a: Path, b: Path) -> bool:
    da = _normalize(json.loads(a.read_text()), str(a.parent.resolve()))
    db = _normalize(json.loads(b.read_text()), str(b.parent.resolve()))
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
    for name in names:
        a, b = built / name, frozen / name
        if not a.exists():
            diffs.append(f"{name}: not built")
            continue
        if a.read_bytes() == b.read_bytes():
            continue
        if name.endswith(".json") and _json_equal(a, b):
            continue
        if name.endswith(".gpkg") and _gpkg_equal(a, b):
            continue
        if name.endswith(".csv") and reproduce.equivalent(a.read_bytes(), b.read_bytes(), name):
            continue
        diffs.append(f"{name}: content differs")
    return diffs

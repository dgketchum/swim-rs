"""Helpers for reproducing frozen evidence on a machine other than the one that made it.

Frozen sidecars, run metadata, pooled-gate JSONs, and container attributes record
absolute paths from the authoring machine. On a clone with a relocated data root
(``SWIM_ROOT``) those strings do not resolve, and gates that compare them as
strings fail although the data is present. Two helpers fix that:

``rebase(path, root, repo)``
    A recorded path mapped onto this machine: unchanged when it exists, otherwise
    re-rooted below ``root`` (for ``<root>/<project>/...`` paths) or ``repo``
    (for ``<clone>/examples|paper|scripts/...`` paths).
``same_path(a, b)`` / ``portable_key(path)``
    Machine-independent comparison: the suffix from the project or repository
    marker onward.

``equivalent(a, b, name)`` compares two serialized files (CSV or JSON) allowing
floating-point round-off, for check modes that would otherwise fail on 1-ULP
differences between platforms. Build stamps (``volatile_key``), digests of inputs
and outputs (``digest``: a rebuild within round-off changes every hash), and the
machine part of recorded paths, whole or embedded in text (``portable_text``),
are ignored. ``json_equivalent`` is the same comparison on parsed JSON.
"""

import io
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECTS = (
    "1_Boulder",
    "2_Fort_Peck",
    "3_Crane",
    "4_Flux_Network",
    "5_Flux_Ensemble",
    "6_Flux_International",
    "7_Applied_Water",
    "8_Soil_Moisture",
    "9_Bushland",
    "10_Oregon",
)
# clone-relative markers; ``data`` covers the repo-local inputs (``data/cartographic``)
REPO_MARKERS = ("examples", "paper", "scripts", "data")
RTOL = 1e-9
ATOL = 1e-12
# sidecar stamps that a rebuild always changes (compared nowhere; the content is):
# these keys, and any key ending in one of the suffixes (``date_generated_utc``,
# ``promoted_at``, ``promoted_at_git_sha``, ``build_timestamp``)
VOLATILE_KEYS = frozenset({"created_utc", "created", "generated_utc", "git_sha", "hostname"})
VOLATILE_SUFFIXES = ("_utc", "_at", "git_sha", "timestamp")
# hashes and sizes of inputs and outputs: a rebuild equal within round-off changes them
DIGEST_SUFFIXES = ("sha256", "bytes")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# an absolute path, whole or inside a sentence ("... median of /data/ssd1/swim/x.csv")
_PATH_IN_TEXT = re.compile(r"(?<![\w/])/[\w.-]+(?:/[\w.-]+)+")


def volatile_key(key):
    """True for a JSON key that holds a build stamp rather than content."""
    return isinstance(key, str) and (key in VOLATILE_KEYS or key.endswith(VOLATILE_SUFFIXES))


def digest(key, value=None):
    """True for a JSON entry or CSV column that holds a hash or a byte count."""
    return (isinstance(key, str) and key.lower().endswith(DIGEST_SUFFIXES)) or (
        isinstance(value, str) and _HEX64.fullmatch(value) is not None
    )


def portable_text(text):
    """``text`` with every absolute path below a project or repository marker
    replaced by its machine-independent suffix."""

    def swap(m):
        token = m.group(0)
        trail = len(token) - len(token.rstrip("."))
        body = token[: len(token) - trail] if trail else token
        kind, key = split(body)
        return (key.as_posix() if kind else body) + token[len(body) :]

    return _PATH_IN_TEXT.sub(swap, text)


def split(path):
    """``(kind, key)``: ``kind`` is ``"root"``, ``"repo"``, or ``None``; ``key`` is the
    machine-independent suffix (``Path``)."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in PROJECTS:
            if i > 0 and parts[i - 1] == "examples":
                return "repo", Path(*parts[i - 1 :])
            return "root", Path(*parts[i:])
    for i, part in enumerate(parts):
        # a clone root is never the filesystem root, so skip "/data/..." itself
        if part in REPO_MARKERS and not (i == 1 and parts[0] == "/"):
            return "repo", Path(*parts[i:])
    return None, Path(path)


def portable_key(path):
    return split(path)[1].as_posix()


def same_path(a, b):
    return portable_key(a) == portable_key(b)


def rebase(path, root=None, repo=None):
    """A path recorded on another machine, resolved on this one (``Path``)."""
    p = Path(path)
    if p.exists():
        return p
    kind, key = split(p)
    if kind == "root":
        base = root or os.environ.get("SWIM_ROOT")
        return Path(base) / key if base else p
    if kind == "repo":
        return Path(repo) / key if repo else p
    return p


def _json_key(k):
    return portable_key(k) if isinstance(k, str) and k.startswith("/") else k


def json_equivalent(a, b):
    """True when two parsed JSON values agree apart from build stamps, digests,
    floating-point round-off, and the machine part of recorded paths."""
    if isinstance(a, dict) and isinstance(b, dict):
        da = {_json_key(k): v for k, v in a.items() if not (volatile_key(k) or digest(k, v))}
        db = {_json_key(k): v for k, v in b.items() if not (volatile_key(k) or digest(k, v))}
        return da.keys() == db.keys() and all(json_equivalent(da[k], db[k]) for k in da)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(a, str) and isinstance(b, str):
        # recorded paths: absolute on one machine, relative or re-rooted on another
        return a == b or portable_text(a) == portable_text(b)
    return a == b


def _csv_equivalent(a, b):
    try:
        da = pd.read_csv(io.BytesIO(a))
        db = pd.read_csv(io.BytesIO(b))
    except Exception:
        return False
    if da.shape != db.shape or list(da.columns) != list(db.columns):
        return False
    for col in da.columns:
        if digest(col):
            continue
        x, y = da[col], db[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(float), y.to_numpy(float), RTOL, ATOL, equal_nan=True):
                return False
        elif not x.astype(str).equals(y.astype(str)):
            # a mixed column (key,value files): numbers within round-off, text equal
            # apart from the machine part of any path it carries
            nx, ny = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
            num = nx.notna() & ny.notna()
            if not np.allclose(nx[num], ny[num], RTOL, ATOL):
                return False
            tx, ty = x[~num].astype(str).map(portable_text), y[~num].astype(str).map(portable_text)
            if not tx.equals(ty):
                return False
    return True


def equivalent(a, b, name):
    """True when two serialized files are byte-equal or equal within floating-point
    round-off (CSV numeric columns, JSON numbers) with machine-independent paths and
    ignoring the build stamps in ``VOLATILE_KEYS`` (JSON keys and values alike)."""
    if a == b:
        return True
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return _csv_equivalent(a, b)
    if suffix == ".json":
        try:
            return json_equivalent(json.loads(a), json.loads(b))
        except ValueError:
            return False
    return False

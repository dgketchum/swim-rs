"""Seed a scratch figure-data output directory from the frozen package.

Two builders read the previous package (``fig03`` re-registers Figure 1 provenance
from the prior ``fig_manifest.json``; ``fig01`` reads the prior
``fig03_example_selection.json``), so an ``--out`` directory that holds no manifest
yet is first filled with a copy of the frozen ``paper/data/final/figures/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SEED_IGNORE = ("superseded_*", "BUILD_STATUS.md", "__pycache__")


def seed_output(out: Path, frozen: Path) -> bool:
    """Copy the frozen package into an ``--out`` directory that holds no manifest yet.

    Returns True when a copy was made.  The frozen directory itself and any directory
    that already holds ``fig_manifest.json`` are left alone.
    """
    out, frozen = Path(out), Path(frozen)
    if out.resolve() == frozen.resolve() or (out / "fig_manifest.json").exists():
        return False
    if not (frozen / "fig_manifest.json").exists():
        raise SystemExit(
            f"{out} holds no fig_manifest.json and the frozen package {frozen} is not "
            "on this machine to seed it from (unpack swim_reference_final.tar.gz)"
        )
    shutil.copytree(frozen, out, ignore=shutil.ignore_patterns(*SEED_IGNORE), dirs_exist_ok=True)
    return True

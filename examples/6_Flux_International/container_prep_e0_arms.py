"""Seed a clean container copy for one E0 vegetation-formulation arm on the E2 footing.

The E0 disjoint confirmation re-runs the Ex5 formulation trio on the Example 6 cohort:
arm A is the canonical cover-scaled run (``6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr``,
no rerun), arms B/C are ``..._fao56_sig`` (unscaled sigmoid) and ``..._fao56`` (unscaled
linear). Every arm must calibrate against bit-identical inputs, so each arm container is a
``copytree`` of the canonical GrassBasis container with the canonical run's calibration
state removed:

* ``calibration/`` (posterior parameters, uncertainty, metadata, ``batches`` ingest log) —
  otherwise ``_build_swim_input`` bakes the arm-A posterior into the PEST base and
  ``batch_runner --resume`` skips both batches as already ingested;
* ``simulation/runs/*`` plus the ``last_run`` / ``default_restart_run_id`` attributes — the
  resolved-state run written after ingest would otherwise be picked up as a restart state;
* ``simulate`` provenance events and the stale ``last_health_check`` — so the copy is in the
  same pre-launch state the canonical container was in (phase8 ``check_no_calibration_state``).

The input arrays are hash-compared with the source after cleaning.

Usage:
    uv run python examples/6_Flux_International/container_prep_e0_arms.py \
        --config examples/6_Flux_International/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56.toml
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import zarr

from swimrs.swim.config import ProjectConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "e2_refooting"))
from phase8_container_health import check_no_calibration_state  # noqa: E402
from phase9_archive_prelaunch import compare_hashes, container_manifest  # noqa: E402

SOURCE = (
    "/data/ssd1/swim/6_Flux_International/data/"
    "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
)
RUN_ATTRS = ("last_run", "default_restart_run_id", "last_health_check")


def clean_calibration_state(root: zarr.Group) -> list[str]:
    removed = []
    if "calibration" in root:
        del root["calibration"]
        removed.append("calibration/")
    if "simulation/runs" in root:
        for run_id in list(root["simulation/runs"].keys()):
            del root[f"simulation/runs/{run_id}"]
            removed.append(f"simulation/runs/{run_id}")
    for key in RUN_ATTRS:
        if key in root.attrs:
            attrs = dict(root.attrs)
            attrs.pop(key)
            root.attrs.clear()
            root.attrs.update(attrs)
            removed.append(f"attrs.{key}")
    prov = dict(root.attrs.get("provenance", {}))
    events = prov.get("events", [])
    kept = [e for e in events if e.get("operation") != "simulate"]
    if len(kept) != len(events):
        prov["events"] = kept
        root.attrs["provenance"] = prov
        removed.append(f"provenance simulate events ({len(events) - len(kept)})")
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="arm TOML; its [paths] container is the target")
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = ProjectConfig()
    cfg.read_config(args.config, calibrate=True)
    dst = cfg.container_path
    if os.path.abspath(dst) == os.path.abspath(args.source):
        raise SystemExit("config container is the source container; refusing")
    if os.path.exists(dst):
        if not args.overwrite:
            raise SystemExit(f"{dst} exists (pass --overwrite to replace)")
        shutil.rmtree(dst)
    print(f"copy {args.source}\n  -> {dst}")
    shutil.copytree(args.source, dst)

    root = zarr.open_group(dst, mode="r+")
    for item in clean_calibration_state(root):
        print(f"  removed {item}")
    state = check_no_calibration_state(root)
    print(f"  no_calibration_state: {'PASS' if state['pass'] else state['problems']}")

    src_root = zarr.open_group(args.source, mode="r")
    rec = {
        k: v["content_sha256"]
        for k, v in container_manifest(src_root).items()
        if "content_sha256" in v
    }
    cur = {
        k: v["content_sha256"] for k, v in container_manifest(root).items() if "content_sha256" in v
    }
    diffs = compare_hashes(rec, cur)
    print(f"  input arrays hashed: {len(cur)}; differences vs source: {diffs or 'none'}")
    if not state["pass"] or diffs:
        raise SystemExit(1)
    print(f"ready: {dst}")


if __name__ == "__main__":
    main()

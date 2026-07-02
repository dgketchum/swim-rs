"""Container prep for Experiment B: Triple ETf POR on the 66-site annual_2yr cohort.

Copies the canonical Experiment A container
(`6_Flux_International_ls_ensemble_por_annual2yr.swim`, 66 sites, which already
holds Landsat SSEBop + PT-JPL ETf, met, NDVI, properties, and the two-stage
`annual_2yr` Landsat-based irrigation classification / dynamics), adds ECOSTRESS
PT-JPL ETf, and merges all three sources via nanmean into
`remote_sensing/etf/merged/triple/no_mask`.

Unlike the predecessor 75-cohort `container_prep_triple_etf_por.py`, this script
**does not recompute dynamics**: the irrigation classification is left exactly as
Experiment A computed it (on the Landsat ETf). The only variable that differs
from Experiment A is the calibration target, so the ECOSTRESS ablation is a clean
one-variable comparison.

Because the source is the *calibrated* Experiment A container, the copy also
inherits Experiment A's ``calibration/`` group (posterior parameters + the
``batches`` ingest log). That state is dropped here: it would otherwise (1) make
``_build_swim_input`` bake Experiment A's posterior into the PEST base
``swim_input.h5`` — anchoring Experiment B's calibration on Experiment A's
solution instead of the default base Experiment A itself started from — and
(2) make ``batch_runner --resume`` treat both batches as already ingested and
skip the run entirely. Experiment B must calibrate fresh against the triple
target, so the container is delivered with inputs + classification but no
calibration state.

Usage:
    python container_prep_triple_etf_por_annual2yr.py [--config PATH] [--overwrite]
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from swimrs.container import SwimContainer, open_container
from swimrs.swim.config import ProjectConfig

SRC_CONTAINER = (
    "/data/ssd1/swim/6_Flux_International/data/6_Flux_International_ls_ensemble_por_annual2yr.swim"
)
TOML = Path(__file__).resolve().parent / "6_Flux_International_TripleETf_POR_annual2yr.toml"


def _load_config(config_path: str | None = None) -> ProjectConfig:
    conf = Path(config_path) if config_path else TOML
    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def copy_container(cfg: ProjectConfig, overwrite: bool = False) -> SwimContainer:
    dst = cfg.container_path or os.path.join(
        cfg.data_dir, f"{cfg.project_name}_triple_etf_por_annual2yr.swim"
    )

    if os.path.exists(dst) and not overwrite:
        print(f"Opening existing container: {dst}")
        return open_container(dst, mode="r+")

    if os.path.exists(dst):
        print(f"Removing existing: {dst}")
        shutil.rmtree(dst)

    if not os.path.isdir(SRC_CONTAINER):
        raise FileNotFoundError(f"Source (Experiment A) container not found: {SRC_CONTAINER}")

    print(f"Copying {SRC_CONTAINER} -> {dst}")
    shutil.copytree(SRC_CONTAINER, dst)
    return open_container(dst, mode="r+")


def ingest_ecostress_etf(container: SwimContainer, cfg: ProjectConfig):
    eco_dir = os.path.join(cfg.ecostress_dir, "extracts", "etf", "no_mask")
    if not os.path.isdir(eco_dir):
        raise FileNotFoundError(f"ECOSTRESS ETf directory not found: {eco_dir}")

    print(f"\n=== Ingesting ECOSTRESS PT-JPL ETf from {eco_dir} ===")
    container.ingest.etf(
        source_dir=eco_dir,
        uid_column=cfg.feature_id_col,
        model="ptjpl",
        instrument="ecostress",
        mask="no_mask",
    )


def merge_etf(container: SwimContainer):
    """Merge Landsat SSEBop + Landsat PT-JPL + ECOSTRESS PT-JPL via nanmean."""
    print("\n=== Merging ETf (nanmean of 3 sources) ===")
    root = container._root

    ls_ssebop = np.array(root["remote_sensing/etf/landsat/ssebop/no_mask"][:])
    ls_ptjpl = np.array(root["remote_sensing/etf/landsat/ptjpl/no_mask"][:])
    eco_ptjpl = np.array(root["remote_sensing/etf/ecostress/ptjpl/no_mask"][:])

    print(f"  Landsat SSEBop: {int((~np.isnan(ls_ssebop)).sum())} valid")
    print(f"  Landsat PT-JPL: {int((~np.isnan(ls_ptjpl)).sum())} valid")
    print(f"  ECOSTRESS PT-JPL: {int((~np.isnan(eco_ptjpl)).sum())} valid")

    stacked = np.stack([ls_ssebop, ls_ptjpl, eco_ptjpl], axis=0)
    merged = np.nanmean(stacked, axis=0)
    all_nan = np.all(np.isnan(stacked), axis=0)
    merged[all_nan] = np.nan

    n_valid = int((~np.isnan(merged)).sum())
    n_ls_only = int((np.isnan(eco_ptjpl) & (~np.isnan(ls_ssebop) | ~np.isnan(ls_ptjpl))).sum())
    n_eco_only = int((np.isnan(ls_ssebop) & np.isnan(ls_ptjpl) & ~np.isnan(eco_ptjpl)).sum())
    n_both = int((~np.isnan(eco_ptjpl) & (~np.isnan(ls_ssebop) | ~np.isnan(ls_ptjpl))).sum())
    print(f"  Merged (triple): {n_valid} valid values")
    print(
        f"  Landsat-only dates: {n_ls_only}, ECOSTRESS-only dates: {n_eco_only}, overlap: {n_both}"
    )

    rs_grp = root["remote_sensing"]
    etf_grp = rs_grp["etf"] if "etf" in rs_grp else rs_grp.create_group("etf")
    merged_grp = etf_grp["merged"] if "merged" in etf_grp else etf_grp.create_group("merged")
    triple_grp = (
        merged_grp["triple"] if "triple" in merged_grp else merged_grp.create_group("triple")
    )

    if "no_mask" in triple_grp:
        del triple_grp["no_mask"]

    triple_grp.create_array("no_mask", data=merged.astype("float32"))
    print(f"  Written to: remote_sensing/etf/merged/triple/no_mask ({merged.shape})")


def drop_inherited_calibration(container: SwimContainer):
    """Remove the Experiment A calibration state inherited via the container copy.

    The source is the calibrated Experiment A container, so the copy carries its
    ``calibration/`` group (posterior parameters, uncertainty, metadata, and the
    ``batches`` ingest log). Experiment B must calibrate fresh against the triple
    target from the same default parameter base Experiment A started from, so this
    group is deleted. See the module docstring for the two failure modes it
    otherwise causes (PEST base-param contamination and ``--resume`` skip).
    """
    root = container._root
    if "calibration" in root:
        print("\n=== Dropping inherited Experiment A calibration group ===")
        del root["calibration"]
        print("  Removed: calibration/ (parameters, uncertainty, metadata, batches attr)")
    else:
        print("\n=== No inherited calibration group present (already clean) ===")


def main(config_path: str | None = None, overwrite: bool = False):
    cfg = _load_config(config_path)
    container = copy_container(cfg, overwrite=overwrite)

    ingest_ecostress_etf(container, cfg)
    merge_etf(container)
    drop_inherited_calibration(container)

    # NB: dynamics are intentionally NOT recomputed. The Experiment A
    # Landsat-based annual_2yr irrigation classification is preserved verbatim,
    # so the ECOSTRESS ablation differs from Experiment A only in the target.

    print("\n=== Container Preparation Complete (Experiment A classification preserved) ===")
    root = container._root
    for p in [
        "remote_sensing/etf/landsat/ssebop/no_mask",
        "remote_sensing/etf/landsat/ptjpl/no_mask",
        "remote_sensing/etf/ecostress/ptjpl/no_mask",
        "remote_sensing/etf/merged/triple/no_mask",
    ]:
        arr = np.array(root[p][:])
        print(f"  {p}: shape={arr.shape}, valid={int((~np.isnan(arr)).sum())}")

    has_cal = "calibration" in root
    print(f"  calibration group present: {has_cal} (expected False — fresh calibration)")

    container.close()
    print(f"\nContainer saved to: {container.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Experiment B (Triple ETf POR) container from the 66-cohort annual_2yr container"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(config_path=args.config, overwrite=args.overwrite)

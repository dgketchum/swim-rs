"""Container prep: E3 Landsat + ECOSTRESS additional-date treatment.

Copies the canonical Landsat-only control container
(`6_Flux_International_ls_ensemble_por_annual2yr.swim`, 66-site annual_2yr
cohort: Landsat SSEBop + PT-JPL ETf, ERA5-Land met, HWSD soils, fused NDVI,
two-stage irrigation classification, and dynamics), drops the inherited
``calibration/`` group, and ingests ECOSTRESS PT-JPL ETf from existing
extracts. Nothing else is touched: no dynamics recompute, no properties, no
NDVI, no classifier, no gwsub, and no ``merged/triple`` array — the obsolete
triple-ETf design is provenance only.

The treatment differs from the control in exactly one way: the container
additionally holds `remote_sensing/etf/ecostress/ptjpl/no_mask`, which the
calibration consumes only on dates where neither Landsat member retrieved
(TOML `etf_auxiliary_*` keys; overlap dates keep Landsat and exclude
ECOSTRESS).

The inherited ``calibration/`` group must be dropped because the source is the
*calibrated* control container: keeping it would (1) bake the control posterior
into the PEST base ``swim_input.h5``, anchoring the treatment on the control's
solution, and (2) make ``batch_runner --resume`` treat batches as already
ingested and skip the run.

Usage:
    uv run python container_prep_ls_ecostress_additional_dates.py \
        --config 6_Flux_International_LSEnsemble_ECOSTRESSAddDates_POR_annual2yr.toml \
        [--overwrite]
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
TOML = (
    Path(__file__).resolve().parent
    / "6_Flux_International_LSEnsemble_ECOSTRESSAddDates_POR_annual2yr.toml"
)


def _load_config(config_path: str | None = None) -> ProjectConfig:
    conf = Path(config_path) if config_path else TOML
    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def copy_container(cfg: ProjectConfig, overwrite: bool = False) -> SwimContainer:
    dst = cfg.container_path
    if dst is None:
        raise ValueError("Config must set a container path")
    if os.path.abspath(dst) == os.path.abspath(SRC_CONTAINER):
        raise ValueError(f"Treatment container path equals the control container: {dst}")

    if os.path.exists(dst) and not overwrite:
        print(f"Opening existing container: {dst}")
        return open_container(dst, mode="r+")

    if os.path.exists(dst):
        print(f"Removing existing: {dst}")
        shutil.rmtree(dst)

    if not os.path.isdir(SRC_CONTAINER):
        raise FileNotFoundError(f"Source (control) container not found: {SRC_CONTAINER}")

    print(f"Copying {SRC_CONTAINER} -> {dst}")
    shutil.copytree(SRC_CONTAINER, dst)
    return open_container(dst, mode="r+")


def drop_inherited_calibration(container: SwimContainer):
    """Remove the control calibration state inherited via the container copy."""
    root = container._root
    if "calibration" in root:
        print("\n=== Dropping inherited control calibration group ===")
        del root["calibration"]
        print("  Removed: calibration/ (parameters, uncertainty, metadata, batches attr)")
    else:
        print("\n=== No inherited calibration group present (already clean) ===")


def ingest_ecostress_etf(container: SwimContainer, cfg: ProjectConfig):
    eco_dir = os.path.join(cfg.ecostress_dir, "extracts", "etf", "no_mask")
    if not os.path.isdir(eco_dir):
        raise FileNotFoundError(f"ECOSTRESS ETf directory not found: {eco_dir}")

    print(f"\n=== Ingesting ECOSTRESS PT-JPL ETf from {eco_dir} ===")
    # overwrite=True makes a re-run on a partially-built container idempotent
    # (the ingest re-reads the same extracts either way).
    container.ingest.etf(
        source_dir=eco_dir,
        uid_column=cfg.feature_id_col,
        model="ptjpl",
        instrument="ecostress",
        mask="no_mask",
        overwrite=True,
    )


def report_source_date_counts(container: SwimContainer):
    """Recomputed ECOSTRESS-only / overlap counts over all stored fields.

    The cohort-restricted (66-site) counts are recomputed by the pre-calibration
    validation step against the batch field list; this reports the full stored
    population.
    """
    root = container._root
    ls_ssebop = np.array(root["remote_sensing/etf/landsat/ssebop/no_mask"][:])
    ls_ptjpl = np.array(root["remote_sensing/etf/landsat/ptjpl/no_mask"][:])
    eco = np.array(root["remote_sensing/etf/ecostress/ptjpl/no_mask"][:])

    ls_any = ~np.isnan(ls_ssebop) | ~np.isnan(ls_ptjpl)
    eco_valid = ~np.isnan(eco)
    eco_only = eco_valid & ~ls_any
    overlap = eco_valid & ls_any

    # Container timeseries arrays are (time, fields): reduce over time (axis 0)
    # to get a per-field any-capture vector, then count fields.
    n_sites_eco = int(eco_valid.any(axis=0).sum())
    print("\n=== ECOSTRESS source/date counts (all stored fields) ===")
    print(f"  ECOSTRESS total valid: {int(eco_valid.sum())}")
    print(f"  ECOSTRESS-only (no Landsat member): {int(eco_only.sum())}")
    print(f"  Overlap with either Landsat member: {int(overlap.sum())}")
    print(f"  Sites with any ECOSTRESS: {n_sites_eco}")


def main(config_path: str | None = None, overwrite: bool = False):
    cfg = _load_config(config_path)
    container = copy_container(cfg, overwrite=overwrite)

    drop_inherited_calibration(container)
    ingest_ecostress_etf(container, cfg)

    # NB: dynamics, properties, NDVI, classifier, and gwsub are intentionally
    # NOT recomputed — the control state is preserved verbatim so the treatment
    # differs only in the added auxiliary ETf source.

    print("\n=== Container Preparation Complete (control state preserved) ===")
    root = container._root
    for p in [
        "remote_sensing/etf/landsat/ssebop/no_mask",
        "remote_sensing/etf/landsat/ptjpl/no_mask",
        "remote_sensing/etf/ecostress/ptjpl/no_mask",
    ]:
        arr = np.array(root[p][:])
        print(f"  {p}: shape={arr.shape}, valid={int((~np.isnan(arr)).sum())}")

    if "remote_sensing/etf/merged" in root:
        raise RuntimeError(
            "merged/ ETf group present — the control container should not carry "
            "the obsolete triple design. Investigate before proceeding."
        )

    has_cal = "calibration" in root
    print(f"  calibration group present: {has_cal} (expected False — fresh calibration)")

    report_source_date_counts(container)

    container.close()
    print(f"\nContainer saved to: {container.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build the E3 Landsat + ECOSTRESS additional-date treatment container "
            "from the 66-cohort Landsat-only control"
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(config_path=args.config, overwrite=args.overwrite)

"""Build a SWIM container for one NWI extraction partition (county/chunk).

Data layout (gs://wudr/nwi/data/{label} pulled locally; see the scripts/swim_nwi.py
header and notes/nwi_ingest_plan.md):

    etf/{mask}/{model}_etf_{mask}_{year}.csv
    ndvi/{mask}/ndvi_{mask}_{year}.csv
    ndvi/sentinel/{mask}/ndvi_sentinel_{mask}_{year}.csv
    met/eto/eto_{year}.csv                    OpenET bias-corrected ETo (mm/day)
    snow/snodas/extracts/swe_{year}.csv       SNODAS SWE (meters)
    properties/{landcover,irr,ssurgo,cdl}_{label}.csv

GridMET parquets ({GFID}.parquet) live at [paths.conus] met; the UID->GFID
mapping comes from [paths.conus] gridmet_mapping.

The pre-computed OpenET "ensemble" CSVs are QC references and are NOT
ingested: dynamics and the obs builder average every model present in the
container, so ingesting it would double-count it inside the computed
6-member mean.

An existing container is never deleted: without --overwrite the build
refuses to touch it; with --overwrite it is moved aside to
{container}.bak-<timestamp> first.

Usage:
    uv run python scripts/nwi_build_container.py --config /path/to/32009.toml
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.container import SwimContainer, open_container
from swimrs.container.schema import is_cropland
from swimrs.swim.config import ProjectConfig

MASKS = ("irr", "inv_irr")
MEMBERS = ("ssebop", "sims", "geesebal", "eemetric", "ptjpl", "disalexi")
MET_VARS = ("eto", "prcp", "tmin", "tmax", "srad", "u2", "ea")


def stage_etf_symlinks(data_dir: Path) -> Path:
    """Stage per-model ETf directories the ingestor can consume.

    The ingestor globs *.csv in a directory and filters by mask only, so a
    directory holding several models' files would cross-contaminate a single
    model's ingest. Symlink each member's files into
    {data}/ingest_staging/etf/{model}/{mask}/.
    """
    staging = data_dir / "ingest_staging" / "etf"
    if staging.exists():
        shutil.rmtree(staging)
    for model in MEMBERS:
        for mask in MASKS:
            dst = staging / model / mask
            dst.mkdir(parents=True)
            for src in sorted((data_dir / "etf" / mask).glob(f"{model}_etf_{mask}_*.csv")):
                os.symlink(src.resolve(), dst / src.name)
    return staging


def ingest_openet_eto(container_path: str, eto_dir: Path) -> None:
    """Write OpenET bias-corrected ETo to meteorology/gridmet/eto_corr.

    Unlike Ex5, the NWI export covers the full container period (1979+
    source collection), so no raw-GridMET spinup backfill should be needed;
    any backfilled cells are reported and indicate an export gap.
    """
    c = open_container(container_path, mode="r+")
    fids = c.field_uids
    dates = pd.date_range(c.start_date, c.end_date, freq="D")

    files = sorted(eto_dir.glob("eto_*.csv"))
    if not files:
        c.close()
        raise FileNotFoundError(f"no eto_*.csv under {eto_dir}")

    frames = []
    for f in files:
        raw = pd.read_csv(f, index_col=0)
        date_cols = [col for col in raw.columns if len(col) == 8 and col.isdigit()]
        raw = raw[date_cols]
        raw.columns = pd.to_datetime(raw.columns, format="%Y%m%d")
        frames.append(raw)

    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    df = wide.T.reindex(index=dates, columns=fids)
    arr = df.values.astype(np.float32)
    print(f"  OpenET eto: {df.notna().sum().sum():,} values from {len(files)} files")

    openet_path = "meteorology/gridmet/eto_openet"
    if openet_path in c._root:
        c._root[openet_path][:] = arr
    else:
        c._root.create_array(openet_path, data=arr, overwrite=True)

    raw_gridmet = np.array(c._root["meteorology/gridmet/eto"], dtype=np.float32)
    nan_mask = np.isnan(arr)
    n_backfilled = int(nan_mask.sum())
    arr = np.where(nan_mask, raw_gridmet, arr)
    if n_backfilled > 0:
        print(f"  WARNING: backfilled {n_backfilled:,} NaN cells from raw GridMET eto")

    corr_path = "meteorology/gridmet/eto_corr"
    if corr_path in c._root:
        c._root[corr_path][:] = arr
    else:
        c._root.create_array(corr_path, data=arr, overwrite=True)
    print(f"  wrote {corr_path}: mean={np.nanmean(arr):.3f} mm/day")

    c.close()


def completeness_sweep(container_path: str) -> list[str]:
    """Per-field completeness checks per notes/nwi_calibration_plan.md.

    Mask semantics make all-null single-mask years expected; coverage is
    judged on the irr/inv_irr union per field.
    """
    c = open_container(container_path, mode="r")
    fids = c.field_uids
    failures = []

    print(f"\n--- Completeness sweep: {len(fids)} fields ---")

    # The calibration target is the computed ensemble (nanmean of members
    # present), so a field missing one member still has a valid target — a
    # per-model gap is a documented waiver, not a failure. A field with zero
    # coverage across ALL members has no target at all and fails.
    member_gaps = []
    target_union = None
    for model in MEMBERS:
        union = None
        for mask in MASKS:
            path = f"remote_sensing/etf/landsat/{model}/{mask}"
            if path not in c._root:
                failures.append(f"missing array: {path}")
                continue
            valid = ~np.isnan(np.array(c._root[path]))
            union = valid if union is None else (union | valid)
        if union is None:
            continue
        target_union = union if target_union is None else (target_union | union)
        counts = union.sum(axis=0)
        zero = [fids[i] for i in np.nonzero(counts == 0)[0]]
        print(
            f"  etf/{model}: union obs/field min={counts.min()} "
            f"median={int(np.median(counts))} zero-coverage={len(zero)}"
        )
        member_gaps.extend((fid, model) for fid in zero)
    if member_gaps:
        gaps_csv = Path(container_path).parent / "health" / "etf_member_gaps.csv"
        gaps_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(gaps_csv, "w") as f:
            f.write("fid,model\n")
            f.writelines(f"{fid},{model}\n" for fid, model in member_gaps)
        print(
            f"  WAIVER: {len(member_gaps)} field-model gaps (ensemble target "
            f"degrades to fewer members there) -> {gaps_csv}"
        )
    if target_union is not None:
        counts = target_union.sum(axis=0)
        zero = [fids[i] for i in np.nonzero(counts == 0)[0]]
        if zero:
            failures.append(f"etf ensemble: zero coverage in every member for {zero}")

    union = None
    for mask in MASKS:
        path = f"remote_sensing/ndvi/landsat/{mask}"
        if path in c._root:
            valid = ~np.isnan(np.array(c._root[path]))
            union = valid if union is None else (union | valid)
    if union is None:
        failures.append("missing landsat NDVI arrays")
    else:
        counts = union.sum(axis=0)
        zero = [fids[i] for i in np.nonzero(counts == 0)[0]]
        print(
            f"  ndvi/landsat: union obs/field min={counts.min()} "
            f"median={int(np.median(counts))} zero-coverage={len(zero)}"
        )
        if zero:
            failures.append(f"ndvi/landsat: zero union coverage for {zero}")

    # NWI exports include Sentinel-2 NDVI (2017+) for both masks; an absent
    # group means the ingest silently dropped it (all-NaN-row parse failures
    # were invisible before the astype fix) — fail loudly.
    for mask in MASKS:
        path = f"remote_sensing/ndvi/sentinel/{mask}"
        if path not in c._root:
            failures.append(f"missing array: {path}")
            continue
        nonnull = int((~np.isnan(np.array(c._root[path]))).sum())
        print(f"  ndvi/sentinel/{mask}: non-null={nonnull:,}")
        if nonnull == 0:
            failures.append(f"ndvi/sentinel/{mask}: empty")

    # Dynamics irrigation flags must not contradict IrrMapper on non-cropland
    # fields: the gwsub fallback is a cropland-only rescue, and a
    # contradiction means groundwater discharge is being misread as
    # irrigation (which also mask-switches the ETf obs into the empty irr
    # series for never-irrigated fields).
    irr_yearly = c._root["properties/irrigation/irr_yearly"]
    glc10 = np.array(c._root["properties/land_cover/glc10"]).astype(int)
    dyn_irr = c.export._get_dynamics_dict(fids).get("irr", {})
    noncrop_flips, crop_rescues = [], 0
    for i, fid in enumerate(fids):
        props = json.loads(str(irr_yearly[i]))
        n_flip = sum(
            1
            for yr, v in dyn_irr.get(fid, {}).items()
            if yr != "fallow_years"
            and isinstance(v, dict)
            and v.get("f_irr", 0.0) > 0.3
            and props.get(str(yr), 0.0) <= 0.3
        )
        if n_flip:
            if is_cropland(int(glc10[i]), "glc10"):
                crop_rescues += n_flip
            else:
                noncrop_flips.append((fid, n_flip))
    print(
        f"  irrigation flags: cropland gwsub rescues={crop_rescues} field-years; "
        f"non-cropland contradictions={len(noncrop_flips)} fields"
    )
    if noncrop_flips:
        failures.append(
            f"dynamics irrigation contradicts IrrMapper on non-cropland fields: {noncrop_flips[:10]}"
        )

    for var in MET_VARS + ("eto_corr",):
        path = f"meteorology/gridmet/{var}"
        if path not in c._root:
            failures.append(f"missing array: {path}")
            continue
        arr = np.array(c._root[path])
        nan_frac = np.isnan(arr).mean(axis=0)
        all_nan = [fids[i] for i in np.nonzero(nan_frac == 1.0)[0]]
        holes = int(np.isnan(arr).sum())
        print(f"  met/{var}: NaN cells={holes:,} all-NaN fields={len(all_nan)}")
        if all_nan:
            failures.append(f"met/{var}: all-NaN for {all_nan}")
        if var == "eto_corr" and holes:
            failures.append(f"met/eto_corr: {holes} NaN cells (export gap)")

    path = "snow/snodas/swe"
    if path in c._root:
        arr = np.array(c._root[path])
        counts = (~np.isnan(arr)).sum(axis=0)
        zero = [fids[i] for i in np.nonzero(counts == 0)[0]]
        print(f"  swe ({path}): obs/field min={counts.min()} zero-coverage={len(zero)}")
        if zero:
            failures.append(f"swe: zero coverage for {zero}")
    else:
        failures.append("missing SWE array")

    c.close()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Project TOML")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Move an existing container aside to *.bak-<timestamp> and rebuild",
    )
    ap.add_argument("--workers", type=int, default=1, help="CSV parse workers")
    ap.add_argument("--skip-health", action="store_true")
    args = ap.parse_args()

    config = ProjectConfig()
    config.read_config(args.config)

    data_dir = Path(config.data_dir)
    container_path = config.container_path
    uid = config.feature_id_col

    if os.path.exists(container_path):
        if not args.overwrite:
            print(f"Container exists: {container_path}\nRefusing to touch it (no clobber). ")
            print("Pass --overwrite to move it aside and rebuild.")
            return 1
        bak = f"{container_path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(container_path, bak)
        lock = f"{container_path}.lock"
        if os.path.exists(lock):
            os.remove(lock)
        print(f"Moved existing container aside: {bak}")

    print(f"Creating container {container_path}")
    container = SwimContainer.create(
        container_path,
        fields_shapefile=config.fields_shapefile,
        uid_column=uid,
        start_date=str(config.start_dt.date()),
        end_date=str(config.end_dt.date()),
        project_name=config.project_name,
    )

    try:
        print("Ingesting properties")
        container.ingest.properties(
            lulc_csv=config.lulc_csv,
            soils_csv=config.ssurgo_csv,
            irr_csv=config.irr_csv,
            # Optional: units built before swim_nwi exported CDL have no such
            # file, and config.cdl_csv is then None, which the ingestor skips.
            cdl_csv=config.cdl_csv,
            uid_column=uid,
        )

        for mask in MASKS:
            print(f"Ingesting Landsat NDVI ({mask})")
            container.ingest.ndvi(
                data_dir / "ndvi" / mask,
                uid_column=uid,
                instrument="landsat",
                mask=mask,
                workers=args.workers,
            )
            print(f"Ingesting Sentinel NDVI ({mask})")
            container.ingest.ndvi(
                data_dir / "ndvi" / "sentinel" / mask,
                uid_column=uid,
                instrument="sentinel",
                mask=mask,
                workers=args.workers,
            )

        staging = stage_etf_symlinks(data_dir)
        for model in MEMBERS:
            for mask in MASKS:
                print(f"Ingesting ETf {model} ({mask})")
                container.ingest.etf(
                    staging / model / mask,
                    uid_column=uid,
                    model=model,
                    mask=mask,
                    instrument="landsat",
                    workers=args.workers,
                )

        print("Ingesting GridMET parquets")
        container.ingest.gridmet(
            config.met_dir,
            grid_shapefile=config.gridmet_mapping_shp,
            uid_column=uid,
            grid_column=config.gridmet_id_col or "GFID",
        )

        print("Ingesting SNODAS")
        container.ingest.snodas(config.snodas_in_dir, uid_column=uid)

        print("Computing merged NDVI")
        container.compute.merged_ndvi(masks=MASKS, instruments=("landsat", "sentinel"))
    finally:
        container.close()

    print("Ingesting OpenET corrected ETo")
    ingest_openet_eto(container_path, data_dir / "met" / "eto")

    container = SwimContainer.open(container_path, mode="a")
    try:
        print("Computing dynamics")
        container.compute.dynamics(
            etf_model=config.etf_target_model,
            irr_threshold=config.irrigation_threshold,
            masks=MASKS,
            instruments=("landsat", "sentinel"),
            use_mask=True,
            use_lulc=False,
            met_source="gridmet",
        )

        if not args.skip_health:
            print("\n--- Health report ---")
            health_dir = data_dir / "health"
            container.report(
                config={
                    "mask_mode": "irrigation",
                    "etf_target_model": config.etf_target_model,
                    "etf_ensemble_members": config.etf_ensemble_members,
                    "met_source": "gridmet",
                    "snow_source": "snodas",
                },
                output_dir=str(health_dir),
                health_profile="calibration",
            )
    finally:
        container.close()

    failures = completeness_sweep(container_path)
    if failures:
        print("\nCOMPLETENESS FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nContainer build complete and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

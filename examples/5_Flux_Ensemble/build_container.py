"""Build an Example 5 calibration container from the base container.

Copies the base container, re-ingests ETf (OpenET ETo denominator, MAX dedup,
2016-2025), ingests OpenET reference ETo/ETr, computes simple mean ensemble,
and recomputes dynamics.

Dynamics settings (Example 5 constraints):
  - Irrigation status from IrrMapper/LANID (use_mask=True)
  - No gwsub irrigation fallback (gwsub_irr_fallback=False)
  - ETf and NDVI from no_mask only (masks=("no_mask",))
  - ndvi_min_start=0.25 filters dormant-season irrigation windows
  - Group bridging disabled, decay-threshold forward extension

Usage:
    python build_container.py --run run17
    python build_container.py --run run16  # reproduce Run 16
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.container import open_container
from swimrs.swim.config import ProjectConfig

MODELS = ("ssebop", "sims", "eemetric", "ptjpl", "geesebal", "disalexi")
MASKS = ("no_mask",)
MAX_VALID_ETF = 2.0

ETF_START = "2016-01-01"
ETF_END = "2025-12-31"


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"
    cfg = ProjectConfig()
    cfg.read_config(str(conf))
    return cfg


def _read_etf_csv_max(csv_path):
    """Read ETf CSV, resolve duplicate dates with MAX (old ingestor behavior)."""
    raw = pd.read_csv(csv_path, index_col=0)
    date_cols = pd.to_datetime(raw.columns, format="%Y%m%d")
    raw.columns = date_cols
    df = raw.T
    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        df = df.groupby(df.index).max()
        print(f"    Resolved {n_dups} duplicate dates with MAX")
    return df


def ingest_new_etf(container_path, etf_csv_dir):
    """Replace ETf arrays, MAX dedup, cut to 2016-2025."""
    c = open_container(container_path, mode="r+")
    fids = c.field_uids
    dates = pd.date_range(c.start_date, c.end_date, freq="D")

    for model in MODELS:
        csv_path = os.path.join(etf_csv_dir, f"{model}_etf_no_mask.csv")
        if not os.path.exists(csv_path):
            print(f"  {model}: CSV not found, skipping")
            continue

        print(f"  {model}:")
        df = _read_etf_csv_max(csv_path)

        df = df.where(df >= 0.05)
        df = df.where(df <= MAX_VALID_ETF)

        # Cut to 2016-2025 only
        df = df.loc[(df.index >= ETF_START) & (df.index <= ETF_END)]
        print(f"    Cut to {ETF_START}..{ETF_END}")

        # Align to container time index and fields
        df = df.reindex(index=dates)
        df = df.reindex(columns=fids)

        n_valid = df.notna().sum().sum()
        per_field = df.notna().sum(axis=0)
        print(
            f"    {n_valid:,} valid obs, per-field: min={per_field.min()}, "
            f"max={per_field.max()}, mean={per_field.mean():.0f}"
        )

        arr_data = df.values.astype(np.float32)

        for mask in MASKS:
            path = f"remote_sensing/etf/landsat/{model}/{mask}"
            if path in c._root:
                c._root[path][:] = arr_data
            else:
                c._root.create_array(path, data=arr_data, overwrite=True)

    c.close()
    print("  ETf ingestion complete.")


def ingest_openet_eto(container_path, refet_dir):
    """Write OpenET reference ETo as eto_corr, backfilling pre-1999 from existing eto_corr."""
    c = open_container(container_path, mode="r+")
    fids = c.field_uids
    dates = pd.date_range(c.start_date, c.end_date, freq="D")

    for var in ("eto", "etr"):
        csv_path = os.path.join(refet_dir, f"openet_{var}.csv")
        if not os.path.exists(csv_path):
            print(f"  OpenET {var}: CSV not found, skipping")
            continue

        raw = pd.read_csv(csv_path, index_col=0)
        raw.columns = pd.to_datetime(raw.columns, format="%Y%m%d")
        df = raw.T.reindex(index=dates, columns=fids)

        n_valid = df.notna().sum().sum()
        print(f"  OpenET {var}: {n_valid:,} values")

        arr = df.values.astype(np.float32)

        openet_path = f"meteorology/gridmet/{var}_openet"
        if openet_path in c._root:
            c._root[openet_path][:] = arr
        else:
            c._root.create_array(openet_path, data=arr, overwrite=True)

        corr_path = f"meteorology/gridmet/{var}_corr"
        if corr_path in c._root:
            old = np.array(c._root[corr_path], dtype=np.float32)
            old_mean = np.nanmean(old)
            nan_mask = np.isnan(arr)
            n_backfilled = int(nan_mask.sum())
            arr = np.where(nan_mask, old, arr)
            if n_backfilled > 0:
                print(
                    f"    Backfilled {n_backfilled:,} NaN cells in {var}_corr from existing {var}_corr"
                )
            new_mean = np.nanmean(arr)
            print(f"    Overwriting {corr_path}: old mean={old_mean:.3f}, new mean={new_mean:.3f}")
            c._root[corr_path][:] = arr
        else:
            c._root.create_array(corr_path, data=arr, overwrite=True)
            print(f"    Created {corr_path}")

    c.close()
    print("  OpenET reference ET ingestion complete.")


def compute_mean_ensemble(container_path):
    """Compute simple nanmean ensemble."""
    c = open_container(container_path, mode="r+")

    for mask in MASKS:
        arrays = []
        present = []
        for m in MODELS:
            path = f"remote_sensing/etf/landsat/{m}/{mask}"
            if path not in c._root:
                continue
            arrays.append(np.array(c._root[path]))
            present.append(m)

        if len(arrays) < 2:
            print(f"  {mask}: only {len(arrays)} models, skipping")
            continue

        stack = np.stack(arrays, axis=2)
        mean_result = np.nanmean(stack, axis=2).astype(np.float32)

        out_path = f"remote_sensing/etf/landsat/ensemble/{mask}"
        if out_path in c._root:
            c._root[out_path][:] = mean_result
        else:
            c._root.create_array(out_path, data=mean_result, overwrite=True)

        n_valid = np.count_nonzero(~np.isnan(mean_result))
        print(f"  {mask}: {len(present)} models -> mean ensemble, {n_valid:,} valid cells")

    c.close()


def compute_mad_ensemble(container_path):
    """Compute MAD-filtered ensemble mean (Volk et al. methodology).

    For each cell (day × field), removes up to 2 models that fall outside
    median ± 2·MAD, then takes the mean of the remaining models.
    Writes to remote_sensing/etf/landsat/ensemble/{mask}.
    """
    MAD_SCALE = 1.483
    MAD_THRESHOLD = 2.0
    MAX_REMOVALS = 2

    c = open_container(container_path, mode="r+")

    for mask in MASKS:
        arrays = []
        present = []
        for m in MODELS:
            path = f"remote_sensing/etf/landsat/{m}/{mask}"
            if path not in c._root:
                continue
            arrays.append(np.array(c._root[path]))
            present.append(m)

        if len(arrays) < 3:
            print(f"  {mask}: only {len(arrays)} models, need ≥3 for MAD, skipping")
            continue

        stack = np.stack(arrays, axis=2)  # (n_days, n_fields, n_models)

        # Simple mean for comparison
        simple_mean = np.nanmean(stack, axis=2)

        # MAD filter
        median_val = np.nanmedian(stack, axis=2, keepdims=True)
        deviations = np.abs(stack - median_val)
        mad = MAD_SCALE * np.nanmedian(deviations, axis=2, keepdims=True)

        lower = median_val - MAD_THRESHOLD * mad
        upper = median_val + MAD_THRESHOLD * mad
        outlier = (stack < lower) | (stack > upper)

        # Cap removals
        n_outliers = np.sum(outlier, axis=2)
        excess_mask = n_outliers > MAX_REMOVALS
        if excess_mask.any():
            idx = np.argwhere(excess_mask)
            for i, j in idx:
                devs = deviations[i, j, :]
                sorted_idx = np.argsort(devs)[::-1]
                outlier[i, j, :] = False
                for k in range(MAX_REMOVALS):
                    if (
                        not np.isnan(devs[sorted_idx[k]])
                        and devs[sorted_idx[k]] > MAD_THRESHOLD * mad[i, j, 0]
                    ):
                        outlier[i, j, sorted_idx[k]] = True

        filtered = stack.copy()
        filtered[outlier] = np.nan
        mad_mean = np.nanmean(filtered, axis=2).astype(np.float32)

        out_path = f"remote_sensing/etf/landsat/ensemble/{mask}"
        if out_path in c._root:
            c._root[out_path][:] = mad_mean
        else:
            c._root.create_array(out_path, data=mad_mean, overwrite=True)

        valid = ~np.isnan(mad_mean) & ~np.isnan(simple_mean)
        diff = mad_mean[valid] - simple_mean[valid]
        any_outlier = np.any(outlier, axis=2)
        frac_filtered = any_outlier[valid].sum() / valid.sum()

        n_valid = np.count_nonzero(~np.isnan(mad_mean))
        print(f"  {mask}: {len(present)} models -> MAD ensemble, {n_valid:,} valid cells")
        print(f"    Cells with outlier removal: {100 * frac_filtered:.1f}%")
        print(
            f"    MAD - simple mean: {diff.mean():+.4f} ({100 * diff.mean() / simple_mean[valid].mean():+.1f}%)"
        )

        # Per-model outlier rates
        for mi, m in enumerate(present):
            n_rem = outlier[:, :, mi][valid].sum()
            print(f"    {m:<12}: {n_rem:,} removed ({100 * n_rem / valid.sum():.1f}%)")

    c.close()


def compute_dynamics(container_path, cfg):
    """Recompute dynamics: IrrMapper/LANID status, no_mask ETf/NDVI, no gwsub fallback."""
    c = open_container(container_path, mode="r+")
    c.compute.dynamics(
        etf_model="ensemble",
        masks=MASKS,
        irr_threshold=cfg.irrigation_threshold or 0.3,
        use_mask=True,
        use_lulc=False,
        lookback=5,
        ndvi_min_start=0.25,
        overwrite=True,
        gwsub_irr_fallback=False,
    )
    c.close()


def validate(container_path):
    """Print container validation summary."""
    c = open_container(container_path, mode="r")
    fids = c.field_uids
    dates = pd.date_range(c.start_date, c.end_date, freq="D")

    for model in list(MODELS) + ["ensemble"]:
        path = f"remote_sensing/etf/landsat/{model}/no_mask"
        if path in c._root:
            data = np.array(c._root[path])
            per_field = np.count_nonzero(~np.isnan(data), axis=0)
            any_valid = ~np.isnan(data).all(axis=1)
            first_day = np.argmax(any_valid) if any_valid.any() else -1
            last_day = len(any_valid) - 1 - np.argmax(any_valid[::-1]) if any_valid.any() else -1
            print(
                f"  {model:<15}: {per_field.sum():>8,} total, per-field min={per_field.min():>4}, "
                f"max={per_field.max():>4}, range={dates[first_day].date()} to {dates[last_day].date()}"
            )

    eto_corr = np.array(c._root["meteorology/gridmet/eto_corr"])
    n_nan = np.count_nonzero(np.isnan(eto_corr))
    print(
        f"  eto_corr:        mean={np.nanmean(eto_corr):.3f}, "
        f"valid={np.count_nonzero(~np.isnan(eto_corr)):,}, nan={n_nan}"
    )

    # Irrigation summary
    import json

    irr_raw = np.array(c._root["derived/dynamics/irr_data"])
    n_irr = 0
    doy_lengths = []
    for i, fid in enumerate(fids):
        data = json.loads(irr_raw[i])
        for k, v in data.items():
            if k == "fallow_years" or not isinstance(v, dict):
                continue
            if v.get("irrigated", 0) == 1:
                n_irr += 1
                doy_lengths.append(len(v.get("irr_doys", [])))
                break
    print(f"  Irrigated sites: {n_irr}/{len(fids)}")
    if doy_lengths:
        print(
            f"  Irr DOY window: median={np.median(doy_lengths):.0f}, "
            f"min={min(doy_lengths)}, max={max(doy_lengths)} days/yr"
        )

    c.close()


def main():
    parser = argparse.ArgumentParser(description="Build Example 5 calibration container")
    parser.add_argument("--run", required=True, help="Run name (e.g. run17)")
    parser.add_argument(
        "--source", default=None, help="Source container (default: base container from config)"
    )
    parser.add_argument(
        "--mad",
        action="store_true",
        help="Use MAD-filtered ensemble (Volk methodology) instead of simple mean",
    )
    args = parser.parse_args()

    cfg = _load_config()
    project_dir = Path(__file__).resolve().parent

    source = args.source or os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    dest = os.path.join(cfg.data_dir, f"5_Flux_Ensemble_{args.run}.swim")
    etf_dir = str(project_dir / "data" / "etf_v21_openet_eto")
    refet_dir = str(project_dir / "data" / "openet_refet")

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source container not found: {source}")

    if os.path.exists(dest):
        raise FileExistsError(
            f"Destination already exists (non-clobbering): {dest}\n"
            f"Remove manually if you want to rebuild."
        )

    print(f"Building {args.run}")
    print(f"Copying {source}\n    -> {dest}")
    shutil.copytree(source, dest)

    print("\n=== Step 1: Ingest ETf (6 models, 2016-2025, MAX dedup) ===")
    ingest_new_etf(dest, etf_dir)

    print("\n=== Step 2: Ingest OpenET reference ETo/ETr ===")
    ingest_openet_eto(dest, refet_dir)

    if args.mad:
        print("\n=== Step 3: Compute MAD-filtered ensemble ===")
        compute_mad_ensemble(dest)
    else:
        print("\n=== Step 3: Compute simple mean ensemble ===")
        compute_mean_ensemble(dest)

    print("\n=== Step 4: Recompute dynamics ===")
    compute_dynamics(dest, cfg)

    print("\n=== Validation Summary ===")
    validate(dest)

    print(f"\n{args.run} container ready: {dest}")


if __name__ == "__main__":
    main()

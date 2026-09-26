"""Quantify the revisit gain from merging Sentinel-2 into the Landsat NDVI record.

For each paper experiment container (E1/Ex5, E2/Ex6, E3/Ex7) this script reads the
per-instrument field-mean NDVI time series stored in the Zarr container and compares
the sampling density of

  * the Landsat-only record  (remote_sensing/ndvi/landsat/{mask}), and
  * the merged Landsat + Sentinel-2 record (derived/merged_ndvi/{mask}).

A *valid observation* is a date on which the field-mean NDVI for that record is
non-null (not NaN). The container time axis is daily and complete; non-acquisition
dates are stored as NaN, so the set of valid dates is the set of acquisition dates
that survived cloud/quality masking in the Earth Engine extraction.

The merged record is a chronological merge (SwimContainer.compute.merged_ndvi) in
which Landsat is preferred on dates observed by both sensors; the set of merged dates
is therefore the union of Landsat and Sentinel-2 dates. The script verifies this
union identity and reports any mismatch.

Outputs (CSV) go to paper/data/derived/ndvi_revisit/.

Read-only with respect to the containers. No Earth Engine, no model runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.container import SwimContainer

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper" / "data" / "derived" / "ndvi_revisit"

EXPERIMENTS = {
    "E1": {
        "label": "E1 / Ex5 Flux Ensemble (CONUS cropland)",
        "container": "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim",
        "pool_csv": REPO / "paper/data/final/e1_openet_benchmark/daily/evaluation_metrics.csv",
        "pool_col": "fid",
        "pool_name": "45-site E1 evaluation pool",
    },
    "E2": {
        "label": "E2 / Ex6 Flux International (cropland cohort)",
        "container": (
            "/data/ssd1/swim/6_Flux_International/data/"
            "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
        ),
        "pool_csv": REPO / "paper/data/final/e2_closure_pool/closure_pool/closure_pool_sites.csv",
        "pool_col": "fid",
        "pool_name": "47-site E2 closure pool",
    },
    "E3": {
        "label": "E3 / Ex7 Applied Water (SLV + ESPA fields)",
        "container": "/data/ssd1/swim/7_Applied_Water/data/7_Applied_Water_e7cal.swim",
        "pool_csv": None,
        "pool_col": None,
        "pool_name": "all 110 calibrated fields",
    },
}

PERIODS = {
    "pre2015": (None, "2014-12-31"),
    "2017+": ("2017-01-01", None),
    "2019+": ("2019-01-01", None),
}


def stamp(container_path: str) -> dict:
    """sha256 of the container root zarr.json plus a size/mtime stamp of the store."""
    root = Path(container_path)
    zj = root / "zarr.json"
    h = hashlib.sha256(zj.read_bytes()).hexdigest() if zj.exists() else None
    total = 0
    nfiles = 0
    newest = 0.0
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            total += st.st_size
            nfiles += 1
            newest = max(newest, st.st_mtime)
    return {
        "path": str(root),
        "root_zarr_json_sha256": h,
        "bytes": total,
        "n_files": nfiles,
        "newest_mtime_utc": pd.Timestamp(newest, unit="s", tz="UTC").isoformat(),
    }


def pick_mask(root, group: str) -> str | None:
    for mask in ("no_mask", "irr", "inv_irr"):
        if f"{group}/{mask}" in root:
            return mask
    return None


def gap_stats(dates: pd.DatetimeIndex, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    n = len(dates)
    span_days = (t1 - t0).days + 1
    years = span_days / 365.25
    out = {
        "n_obs": n,
        "span_days": span_days,
        "obs_per_year": n / years if years > 0 else np.nan,
    }
    if n < 2:
        out.update(
            {
                "gap_median": np.nan,
                "gap_p25": np.nan,
                "gap_p75": np.nan,
                "gap_p90": np.nan,
                "gap_mean": np.nan,
                "frac_gap_gt16": np.nan,
                "frac_gap_gt32": np.nan,
                "n_gaps": 0,
            }
        )
        return out
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    out.update(
        {
            "gap_median": float(np.median(gaps)),
            "gap_p25": float(np.percentile(gaps, 25)),
            "gap_p75": float(np.percentile(gaps, 75)),
            "gap_p90": float(np.percentile(gaps, 90)),
            "gap_mean": float(np.mean(gaps)),
            "frac_gap_gt16": float(np.mean(gaps > 16)),
            "frac_gap_gt32": float(np.mean(gaps > 32)),
            "n_gaps": int(gaps.size),
        }
    )
    return out


def run_experiment(key: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c = SwimContainer(cfg["container"])
    root = c._root

    ndvi_mask = pick_mask(root, "remote_sensing/ndvi/landsat")
    merged_mask = pick_mask(root, "derived/merged_ndvi")
    has_sentinel = "remote_sensing/ndvi/sentinel" in root and (
        pick_mask(root, "remote_sensing/ndvi/sentinel") is not None
    )
    meta = {
        "experiment": key,
        "label": cfg["label"],
        "container": stamp(cfg["container"]),
        "ndvi_mask_used": ndvi_mask,
        "merged_mask_used": merged_mask,
        "has_sentinel_group": bool(has_sentinel),
    }
    if not has_sentinel:
        meta["note"] = "container has NO sentinel NDVI group"

    ls = c.to_xarray(f"remote_sensing/ndvi/landsat/{ndvi_mask}").to_pandas()
    mg = c.to_xarray(f"derived/merged_ndvi/{merged_mask}").to_pandas()
    se = (
        c.to_xarray(
            f"remote_sensing/ndvi/sentinel/{pick_mask(root, 'remote_sensing/ndvi/sentinel')}"
        ).to_pandas()
        if has_sentinel
        else None
    )

    # verify the merged record is the union of the two instrument date sets
    if se is not None:
        union = ls.notna() | se.notna()
        merged = mg.notna()
        meta["merged_equals_union_cells"] = int((union == merged).values.sum())
        meta["merged_union_mismatch_cells"] = int((union != merged).values.sum())
        meta["merged_union_mismatch_frac"] = float((union != merged).values.mean())

    pool = None
    if cfg["pool_csv"] is not None:
        pool = set(pd.read_csv(cfg["pool_csv"])[cfg["pool_col"]].astype(str))
    sites = [s for s in ls.columns if pool is None or str(s) in pool]
    meta["n_container_fields"] = int(ls.shape[1])
    meta["n_sites_analyzed"] = len(sites)
    meta["pool_name"] = cfg["pool_name"]
    if pool is not None:
        meta["pool_sites_not_in_container"] = sorted(pool - set(map(str, ls.columns)))

    rows = []
    for pname, (a, b) in PERIODS.items():
        t0 = pd.Timestamp(a) if a else ls.index[0]
        t1 = pd.Timestamp(b) if b else ls.index[-1]
        if t1 < ls.index[0] or t0 > ls.index[-1]:
            continue
        t0 = max(t0, ls.index[0])
        t1 = min(t1, ls.index[-1])
        for record, frame in (("landsat", ls), ("merged", mg), ("sentinel", se)):
            if frame is None:
                continue
            sub = frame.loc[t0:t1]
            for s in sites:
                d = sub.index[sub[s].notna().values]
                r = gap_stats(pd.DatetimeIndex(d), t0, t1)
                r.update({"experiment": key, "period": pname, "record": record, "site": str(s)})
                rows.append(r)

    per_site = pd.DataFrame(rows)

    # pooled distribution: all gaps from all sites, per period/record
    pooled_rows = []
    for pname, (a, b) in PERIODS.items():
        t0 = pd.Timestamp(a) if a else ls.index[0]
        t1 = pd.Timestamp(b) if b else ls.index[-1]
        if t1 < ls.index[0] or t0 > ls.index[-1]:
            continue
        t0 = max(t0, ls.index[0])
        t1 = min(t1, ls.index[-1])
        for record, frame in (("landsat", ls), ("merged", mg), ("sentinel", se)):
            if frame is None:
                continue
            sub = frame.loc[t0:t1]
            allg = []
            for s in sites:
                d = sub.index[sub[s].notna().values]
                if len(d) >= 2:
                    allg.append(np.diff(d.values).astype("timedelta64[D]").astype(int))
            if not allg:
                continue
            g = np.concatenate(allg)
            sd = per_site[
                (per_site.period == pname)
                & (per_site.record == record)
                & (per_site.experiment == key)
            ]
            pooled_rows.append(
                {
                    "experiment": key,
                    "period": pname,
                    "record": record,
                    "n_sites": len(sites),
                    "site_median_obs_per_year": sd.obs_per_year.median(),
                    "site_p25_obs_per_year": sd.obs_per_year.quantile(0.25),
                    "site_p75_obs_per_year": sd.obs_per_year.quantile(0.75),
                    "site_median_gap_median": sd.gap_median.median(),
                    "site_p25_gap_median": sd.gap_median.quantile(0.25),
                    "site_p75_gap_median": sd.gap_median.quantile(0.75),
                    "site_median_gap_p90": sd.gap_p90.median(),
                    "site_median_frac_gt16": sd.frac_gap_gt16.median(),
                    "site_median_frac_gt32": sd.frac_gap_gt32.median(),
                    "pooled_n_gaps": int(g.size),
                    "pooled_gap_median": float(np.median(g)),
                    "pooled_gap_p25": float(np.percentile(g, 25)),
                    "pooled_gap_p75": float(np.percentile(g, 75)),
                    "pooled_gap_p90": float(np.percentile(g, 90)),
                    "pooled_gap_mean": float(np.mean(g)),
                    "pooled_frac_gt16": float(np.mean(g > 16)),
                    "pooled_frac_gt32": float(np.mean(g > 32)),
                }
            )
    summary = pd.DataFrame(pooled_rows)
    return per_site, summary, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_site, all_sum, metas = [], [], []
    for key, cfg in EXPERIMENTS.items():
        ps, sm, meta = run_experiment(key, cfg)
        all_site.append(ps)
        all_sum.append(sm)
        metas.append(meta)
        print(f"[{key}] {meta['n_sites_analyzed']} sites; sentinel={meta['has_sentinel_group']}")

    site_df = pd.concat(all_site, ignore_index=True)
    sum_df = pd.concat(all_sum, ignore_index=True)
    site_df.to_csv(out / "ndvi_revisit_per_site.csv", index=False)
    sum_df.to_csv(out / "ndvi_revisit_summary.csv", index=False)
    (out / "ndvi_revisit_metadata.json").write_text(json.dumps(metas, indent=2))

    cols = [
        "experiment",
        "period",
        "record",
        "n_sites",
        "site_median_obs_per_year",
        "site_median_gap_median",
        "site_p25_gap_median",
        "site_p75_gap_median",
        "pooled_gap_median",
        "pooled_frac_gt16",
        "pooled_frac_gt32",
        "site_median_frac_gt16",
        "site_median_frac_gt32",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(sum_df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

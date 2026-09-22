"""Post-calibration RUN_POLICY archive builder for Example 7 (batch calibration).

Assembles Category 4 (PEST++ outputs) and Category 5 (posterior summaries) for a
completed batch calibration, reading the per-batch PEST archives written by
``batch_runner`` (``pestrun/pest_archive/batch_NNN/``) plus the ingested
calibration group in the container.

Produces, under ``results/<run>/archive/``:

  4_pest_outputs/
    batch_NNN/                      copied per-batch PEST artifacts (authoritative)
    merged/merged_posterior.csv     final .{noptmax}.par.csv concatenated, +batch col
    merged/merged_phi_history.csv   per-batch phi histories, +batch col
    merged/batch_site_map.csv       site_id -> batch index
    merged/merged_posterior.json    nested {fid: {param: median}} (evaluator input)
  5_posterior_summaries/
    posterior_site_summary.csv      per-site,per-param median/mean/std/q25/q75/IQR/CV
    boundary_hit_rates.csv          per-param x group (+ALL) lower/upper bound hit rate
    lulc_grouped_summary.csv        per-crop,per-param stats across sites
    irrigated_grouped_summary.csv   irrigated vs control per-param stats

    uv run python examples/7_Applied_Water/archive_postcalibration.py
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from swimrs.swim.config import ProjectConfig  # noqa: E402

CALIBRATION_PARAMS = [
    "aw",
    "mad",
    "ndvi_k",
    "ndvi_0",
    "swe_alpha",
    "swe_beta",
    "ks_damp",
    "kr_damp",
]
BOUND_TOL = 0.01  # within 1% of a bound counts as "at bound"


def _load_config() -> ProjectConfig:
    conf = HERE / "7_Applied_Water.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


def _col_to_fid_param(col: str, fids: list[str]) -> tuple[str | None, str | None]:
    """Map a PEST parameter column name to (fid, param), else (None, None)."""
    parts = col.split("_ptype:")[0]
    parts = parts.replace("pname:p_", "")
    parts = parts.rsplit("_:0", 1)[0]
    for fid in fids:
        if parts.lower().endswith(f"_{fid.lower()}"):
            return fid, parts[: -(len(fid) + 1)]
    return None, None


def _par_ensemble_long(par_csv: Path, fids: list[str]) -> pd.DataFrame:
    """Long-form ensemble: rows = (realization, fid, param, value), 'base' dropped."""
    df = pd.read_csv(par_csv, index_col=0)
    df = df.loc[df.index != "base"]
    recs = []
    for col in df.columns:
        fid, param = _col_to_fid_param(col, fids)
        if fid is None:
            continue
        vals = df[col].to_numpy(float)
        for r_idx, v in zip(df.index, vals):
            recs.append((r_idx, fid, param, v))
    return pd.DataFrame(recs, columns=["realization", "site_id", "param", "value"])


def _param_bounds(par_data_csv: Path) -> dict[str, tuple[float, float]]:
    """Per-param (lower, upper) from a batch par_data.csv (pargp = param name)."""
    pd_df = pd.read_csv(par_data_csv)
    bounds = {}
    for _, row in pd_df.iterrows():
        g = str(row["pargp"])
        bounds[g] = (float(row["parlbnd"]), float(row["parubnd"]))
    return bounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", default=None)
    ap.add_argument(
        "--pestrun", default=None, help="pestrun dir (holds pest_archive/, batch_manifest.csv)"
    )
    ap.add_argument("--run-name", default="e7cal")
    ap.add_argument("--noptmax", type=int, default=3)
    args = ap.parse_args()

    cfg = _load_config()
    container_path = args.container or os.path.join(
        cfg.data_dir, f"{cfg.project_name}_{args.run_name}.swim"
    )
    pestrun = Path(args.pestrun or (Path(cfg.project_ws) / "pestrun"))
    archive = Path(cfg.project_ws) / "results" / args.run_name / "archive"
    cat4 = archive / "4_pest_outputs"
    cat5 = archive / "5_posterior_summaries"
    (cat4 / "merged").mkdir(parents=True, exist_ok=True)
    cat5.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(pestrun / "batch_manifest.csv")
    manifest.columns = [c.lower() for c in manifest.columns]
    site_col = "site_id" if "site_id" in manifest.columns else manifest.columns[-1]
    batch_ids = sorted(manifest["batch_id"].unique())

    # ---------- Category 4 ----------
    manifest.rename(columns={site_col: "site_id"}).to_csv(
        cat4 / "merged" / "batch_site_map.csv", index=False
    )

    merged_par, phi_rows = [], []
    site_long = []  # ensemble long-form across all batches
    bounds = None
    for bid in batch_ids:
        src = pestrun / "pest_archive" / f"batch_{bid:03d}"
        dst = cat4 / f"batch_{bid:03d}"
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
        batch_fids = manifest.loc[manifest["batch_id"] == bid, "site_id"].astype(str).tolist()

        if bounds is None:
            pdc = src / "7_applied_water.par_data.csv"
            if pdc.exists():
                bounds = _param_bounds(pdc)

        # merged posterior: final iteration par.csv, row-concat + batch column
        par_csv = src / f"7_Applied_Water.{args.noptmax}.par.csv"
        if par_csv.exists():
            p = pd.read_csv(par_csv, index_col=0)
            p = pd.concat([pd.Series(bid, index=p.index, name="batch"), p], axis=1)
            merged_par.append(p)
            site_long.append(_par_ensemble_long(par_csv, batch_fids))

        # phi history
        summ = src / "calibration_summary.json"
        if summ.exists():
            hist = json.loads(summ.read_text()).get("phi_history", [])
            for it, phi in enumerate(hist):
                phi_rows.append({"batch": bid, "iteration": it, "phi": phi})

    if merged_par:
        pd.concat(merged_par, axis=0).to_csv(cat4 / "merged" / "merged_posterior.csv")
    pd.DataFrame(phi_rows).to_csv(cat4 / "merged" / "merged_phi_history.csv", index=False)

    ens = pd.concat(site_long, axis=0, ignore_index=True)

    # ---------- Category 5 ----------
    # per-site, per-param distribution stats from the posterior ensemble
    def _stats(v):
        v = v.to_numpy(float)
        q25, q75 = np.percentile(v, [25, 75])
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1))
        return pd.Series(
            {
                "median": float(np.median(v)),
                "mean": mean,
                "std": std,
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(q75 - q25),
                "cv": float(std / mean) if mean != 0 else np.nan,
            }
        )

    site_summary = ens.groupby(["site_id", "param"]).value.apply(_stats).unstack()
    site_summary = site_summary.reset_index()
    site_summary.to_csv(cat5 / "posterior_site_summary.csv", index=False)

    # site-level posterior median (wide) joined to crop / control status
    med = ens.groupby(["site_id", "param"]).value.median().unstack()
    gdf = (
        gpd.read_file(cfg.fields_shapefile, engine="fiona")
        .drop_duplicates("site_id")
        .set_index("site_id")
    )
    med["crop"] = med.index.map(gdf["crop"]) if "crop" in gdf else "UNKNOWN"
    med["irrigated"] = np.where(med["crop"].eq("RAINFED_CONTROL"), "control", "irrigated")

    # boundary hit rates (per-param x group + ALL)
    def _hit_rates(sub: pd.DataFrame, label: str, group: str, n: int) -> list[dict]:
        rows = []
        for param in CALIBRATION_PARAMS:
            if param not in sub or bounds is None or param not in bounds:
                continue
            lo, hi = bounds[param]
            span = hi - lo
            vals = sub[param].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            lower = float(np.mean(np.abs(vals - lo) <= BOUND_TOL * span))
            upper = float(np.mean(np.abs(vals - hi) <= BOUND_TOL * span))
            rows.append(
                {
                    "run_name": args.run_name,
                    "parameter": param,
                    "lulc_group": label,
                    "n_sites": int(n),
                    "lower_hit_rate": round(lower, 3),
                    "upper_hit_rate": round(upper, 3),
                    "bound_tolerance": BOUND_TOL,
                }
            )
        return rows

    hit_rows = _hit_rates(med, "ALL", "ALL", len(med))
    for crop, sub in med.groupby("crop"):
        hit_rows += _hit_rates(sub, str(crop), "crop", len(sub))
    pd.DataFrame(hit_rows).to_csv(cat5 / "boundary_hit_rates.csv", index=False)

    # lulc-grouped + irrigated-grouped summaries (median across sites)
    def _grouped(key: str, path: Path):
        recs = []
        for gval, sub in med.groupby(key):
            for param in CALIBRATION_PARAMS:
                if param not in sub:
                    continue
                v = sub[param].dropna().to_numpy(float)
                if len(v) == 0:
                    continue
                recs.append(
                    {
                        key: gval,
                        "parameter": param,
                        "n_sites": int(len(v)),
                        "median": round(float(np.median(v)), 5),
                        "mean": round(float(np.mean(v)), 5),
                        "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 5),
                    }
                )
        pd.DataFrame(recs).to_csv(path, index=False)

    _grouped("crop", cat5 / "lulc_grouped_summary.csv")
    _grouped("irrigated", cat5 / "irrigated_grouped_summary.csv")

    # nested {fid: {param: median}} for the evaluator / convenience
    nested = {
        fid: {
            p: float(med.loc[fid, p])
            for p in CALIBRATION_PARAMS
            if p in med and pd.notna(med.loc[fid, p])
        }
        for fid in med.index
    }
    (cat4 / "merged" / "merged_posterior.json").write_text(json.dumps(nested, indent=2))

    print(f"[{args.run_name}] archived {len(batch_ids)} batches, {len(med)} sites")
    print(f"  Cat 4 -> {cat4}")
    print(f"  Cat 5 -> {cat5}")
    print(f"  merged_posterior.json: {len(nested)} fields")


if __name__ == "__main__":
    main()

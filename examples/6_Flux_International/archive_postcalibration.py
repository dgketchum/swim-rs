"""Post-calibration RUN_POLICY capture for the E2 GrassBasis (re-footed) run: Categories 4 and 5
plus the plan §16 completion checks.

Category 4 copies every ``pest_archive/batch_NNN`` produced by ``batch_runner`` into
``results/<run_name>/archive/4_pest_outputs/batch_NNN`` and writes the merged run-level tables:

* ``merged/batch_site_map.csv``      batch_id, site_id
* ``merged/merged_phi_history.csv``  per batch x iteration: best-mean phi (calibration_summary
                                     ``phi_history``), phi.meas mean/sd/min/max, realizations
* ``merged/merged_posterior.csv``    the posterior ensemble at the declared final iteration,
                                     column-concatenated across batches (realizations x all
                                     site-parameters, ``base`` row kept). This is the explicit
                                     parameter source for ``evaluate.py --par-csv``.
* ``merged/merged_posterior.json``   ``{site: {pest_param: median}}`` of the same ensemble

Category 5 summarises the posterior: ``posterior_site_summary.csv`` (median/mean/std/IQR/CV per
site x parameter), ``boundary_hit_rates.csv`` (ALL, land cover, irrigation class, region; PEST
bounds from ``par_data.csv``; 1 % tolerance), ``lulc_grouped_summary.csv``,
``irrigated_grouped_summary.csv`` and ``posterior_comparison_baseline.csv`` (GrassBasis
iteration-3 medians against the frozen baseline's iteration 3 and iteration 4, read-only).

Completion checks (plan §16) are written to ``4_pest_outputs/completion_checks.json`` and
printed; any failure exits non-zero:

* every iteration 0..noptmax has par/obs/rei files in every batch
* phi is finite and strictly decreasing from iteration 0 to the final iteration
* the declared posterior (iteration ``noptmax``) is what the batch runner ingested: container
  ``calibration/parameters`` equal the merged-posterior medians
* the ingestion touched only the new container (baseline container ``calibration/`` files
  predate the launch)
* dropped realizations / dropped fids are listed (from ``batch_log.json`` and ensemble sizes)

The run log is gzipped into ``1_provenance/run_stdout.log.gz`` (original left in place).

    uv run python examples/6_Flux_International/archive_postcalibration.py
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ex6_paths  # noqa: E402

DEFAULT_CONFIG = ex6_paths.CANONICAL_CONFIG
DEFAULT_RUN_NAME = ex6_paths.CANONICAL_RUN

BOUND_TOL = 0.01  # fraction of the bound span (RUN_POLICY / calibration_guidance)
BOUNDARY_FLAG_RATE = 0.5
# PEST parameter-group name -> container / model name (mirrors ingestor._PEST_NAME_MAP)
PEST_TO_INTERNAL = {"ks_alpha": "ks_damp", "kr_alpha": "kr_damp"}
INGEST_REL_TOL = 1e-6


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_par_column(col: str) -> tuple[str, str] | None:
    """``pname:p_<param>_<fid>_:0_ptype:...`` -> (fid as written, pest param), else None.

    Site ids contain hyphens and digits but never ``_``; the parameter name may contain ``_``
    (``ndvi_0``, ``ks_alpha``), so the fid is the last ``_``-separated token before ``_:0``.
    """
    if not col.startswith("pname:p_"):
        return None
    body = col.split("_ptype:")[0][len("pname:p_") :]
    body = body.rsplit("_:0", 1)[0]
    if "_" not in body:
        return None
    param, fid = body.rsplit("_", 1)
    return fid, param


def merge_par_csvs(par_paths: list[Path]) -> pd.DataFrame:
    """Column-concatenate per-batch ``.par.csv`` ensembles into one realization x parameter frame.

    Each batch estimates a disjoint set of site-parameters for the same realization ids
    (``0..N-1`` plus ``base``), so the merge is a column join on ``real_name``; the per-column
    medians are identical to the per-batch medians the batch runner ingested.
    """
    frames = []
    for p in par_paths:
        df = pd.read_csv(p, index_col=0)
        df.index = df.index.astype(str)
        frames.append(df)
    if not frames:
        raise ValueError("no par.csv files to merge")
    merged = pd.concat(frames, axis=1)
    dupes = merged.columns[merged.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"site-parameters appear in more than one batch: {dupes[:5]}")
    if merged.isna().any().any():
        raise ValueError("realization ids differ between batches; the column merge left gaps")
    return merged


def posterior_medians(merged: pd.DataFrame, fids: list[str]) -> pd.DataFrame:
    """Median over non-``base`` realizations, as a site x pest-parameter frame."""
    fid_by_lower = {f.lower(): f for f in fids}
    med = merged.loc[merged.index != "base"].median()
    recs = {}
    for col, v in med.items():
        dec = decode_par_column(col)
        if dec is None:
            continue
        fid_l, param = dec
        fid = fid_by_lower.get(fid_l.lower())
        if fid is None:
            raise KeyError(f"parameter column for unknown site {fid_l!r}: {col}")
        recs.setdefault(fid, {})[param] = float(v)
    return pd.DataFrame.from_dict(recs, orient="index").sort_index()


def ensemble_long(merged: pd.DataFrame, fids: list[str]) -> pd.DataFrame:
    """Long-form posterior ensemble (realization, site_id, param, value), ``base`` dropped."""
    fid_by_lower = {f.lower(): f for f in fids}
    df = merged.loc[merged.index != "base"]
    recs = []
    for col in df.columns:
        dec = decode_par_column(col)
        if dec is None:
            continue
        fid = fid_by_lower[dec[0].lower()]
        vals = df[col].to_numpy(float)
        recs.extend((r, fid, dec[1], v) for r, v in zip(df.index, vals))
    return pd.DataFrame(recs, columns=["realization", "site_id", "param", "value"])


def param_bounds(par_data_csvs: list[Path], fids: list[str]) -> pd.DataFrame:
    """Per (site_id, pest param) ``lower``/``upper`` from the batches' ``par_data.csv``.

    Bounds are per parameter *column*, not per group: ``mad`` is bounded 0.3-0.8 on irrigated
    sites and 0.1-0.3 on rainfed sites, so a group-level bound would misreport hits.
    """
    fid_by_lower = {f.lower(): f for f in fids}
    recs = []
    for path in par_data_csvs:
        pdf = pd.read_csv(path)
        for name, lo, hi in zip(pdf["parnme"], pdf["parlbnd"], pdf["parubnd"]):
            dec = decode_par_column(str(name))
            if dec is None:
                continue
            fid = fid_by_lower.get(dec[0].lower())
            if fid is None:
                continue
            recs.append((fid, dec[1], float(lo), float(hi)))
    out = pd.DataFrame(recs, columns=["site_id", "param", "lower", "upper"])
    if out.duplicated(["site_id", "param"]).any():
        raise ValueError("a site-parameter has bounds in more than one batch")
    return out.set_index(["site_id", "param"]).sort_index()


def boundary_hit_rows(
    med: pd.DataFrame, bounds: pd.DataFrame, group_label: str, group_kind: str, run_name: str
) -> list[dict]:
    """RUN_POLICY ``boundary_hit_rates.csv`` rows for one group of sites (posterior medians).

    A site's median is "at" a bound when within ``BOUND_TOL`` of that site's own bound span.
    """
    rows = []
    for param in sorted(med.columns):
        vals, lows, highs = [], [], []
        for fid in med.index:
            v = med.loc[fid, param]
            if (fid, param) not in bounds.index or not np.isfinite(v):
                continue
            lo, hi = bounds.loc[(fid, param), ["lower", "upper"]].to_numpy(float)
            vals.append(float(v))
            lows.append(lo)
            highs.append(hi)
        if not vals:
            continue
        vals, lows, highs = (np.asarray(x, dtype=float) for x in (vals, lows, highs))
        span = highs - lows
        lower = float(np.mean(np.abs(vals - lows) <= BOUND_TOL * span))
        upper = float(np.mean(np.abs(vals - highs) <= BOUND_TOL * span))
        pairs = sorted({(float(a), float(b)) for a, b in zip(lows, highs)})
        rows.append(
            {
                "run_name": run_name,
                "parameter": param,
                "internal_name": PEST_TO_INTERNAL.get(param, param),
                "lulc_group": group_label,
                "group_kind": group_kind,
                "n_sites": int(len(vals)),
                "lower_hit_rate": round(lower, 4),
                "upper_hit_rate": round(upper, 4),
                "bound_tolerance": BOUND_TOL,
                "bounds": ";".join(f"{a:g}-{b:g}" for a, b in pairs),
                "boundary_seeking": bool(max(lower, upper) > BOUNDARY_FLAG_RATE),
            }
        )
    return rows


def phi_history_checks(phi_history: list[float]) -> list[str]:
    """Problems with a batch's best-mean phi sequence (finite, strictly decreasing to the end)."""
    problems = []
    arr = np.asarray(phi_history, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        problems.append("phi history empty or non-finite")
        return problems
    if arr[-1] >= arr[0]:
        problems.append(f"final phi {arr[-1]} not below initial {arr[0]}")
    if np.any(np.diff(arr) >= 0):
        problems.append(f"phi not strictly decreasing: {arr.tolist()}")
    return problems


def compare_ingested(container_params: pd.DataFrame, medians: pd.DataFrame) -> pd.DataFrame:
    """Per site x parameter: container value vs merged-posterior median and relative error."""
    recs = []
    for fid in medians.index:
        for pest_param in medians.columns:
            internal = PEST_TO_INTERNAL.get(pest_param, pest_param)
            if fid not in container_params.index or internal not in container_params.columns:
                recs.append(
                    (fid, pest_param, internal, np.nan, medians.loc[fid, pest_param], np.nan)
                )
                continue
            c = float(container_params.loc[fid, internal])
            m = float(medians.loc[fid, pest_param])
            rel = abs(c - m) / max(abs(m), 1e-12)
            recs.append((fid, pest_param, internal, c, m, rel))
    return pd.DataFrame(
        recs,
        columns=["site_id", "pest_param", "internal", "container", "posterior_median", "rel_err"],
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_irrigation_class(root, fids: list[str]) -> pd.Series:
    """``equipped`` (>= 1 irrigated year in derived/dynamics/irr_data) -> 'irrigated'/'rainfed'."""
    uids = [str(u) for u in root["geometry/uid"][:]]
    raw = root["derived/dynamics/irr_data"][:]
    n_years = {}
    for uid, blob in zip(uids, raw):
        per_year = json.loads(blob) if isinstance(blob, str) and blob else {}
        n = sum(
            int(v.get("irrigated", 0))
            for k, v in per_year.items()
            if k != "fallow_years" and isinstance(v, dict)
        )
        n_years[uid] = n
    return pd.Series({f: "irrigated" if n_years.get(f, 0) > 0 else "rainfed" for f in fids})


def container_params_frame(root, fids: list[str]) -> pd.DataFrame:
    uids = [str(u) for u in root["geometry/uid"][:]]
    idx = {u: i for i, u in enumerate(uids)}
    cols = {}
    for name in sorted(root["calibration/parameters"].keys()):
        arr = np.asarray(root[f"calibration/parameters/{name}"][:], dtype=float)
        cols[name] = [arr[idx[f]] if f in idx else np.nan for f in fids]
    return pd.DataFrame(cols, index=fids)


def newest_mtime(path: Path) -> float:
    return max((p.stat().st_mtime for p in path.rglob("*") if p.is_file()), default=0.0)


def _stats(v: np.ndarray) -> dict:
    q25, q75 = np.percentile(v, [25, 75])
    mean, std = float(np.mean(v)), float(np.std(v, ddof=1))
    return {
        "median": float(np.median(v)),
        "mean": mean,
        "std": std,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "cv": float(std / mean) if mean != 0 else np.nan,
        "n_realizations": int(len(v)),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--results-root", default=None, help="default {project_ws}/results")
    p.add_argument("--noptmax", type=int, default=3, help="declared posterior iteration")
    p.add_argument(
        "--log",
        default=None,
        help="batch_runner stdout to gzip into Cat 1 (default {project_ws}/nohup_calibrate_<run>.out)",
    )
    p.add_argument(
        "--baseline-config",
        default=str(ex6_paths.BASELINE_CONFIG),
        help="TOML of the superseded run the posterior is compared against",
    )
    p.add_argument("--baseline-pestrun", default=None, help="default the baseline pest_run_dir")
    p.add_argument("--baseline-container", default=None, help="default the baseline container")
    p.add_argument("--baseline-iterations", default="3,4")
    args = p.parse_args()

    import geopandas as gpd
    import zarr

    cfg = ex6_paths.load_config(args.config)
    results_root = Path(args.results_root) if args.results_root else ex6_paths.results_root(cfg)
    if args.log is None:
        args.log = str(ex6_paths.calibration_log(cfg))
    if args.baseline_pestrun is None or args.baseline_container is None:
        bcfg = ex6_paths.load_config(args.baseline_config)
        args.baseline_pestrun = args.baseline_pestrun or bcfg.pest_run_dir
        args.baseline_container = args.baseline_container or bcfg.container_path
    pestrun = Path(cfg.pest_run_dir)
    archive = results_root / args.run_name / "archive"
    cat4, cat5 = archive / "4_pest_outputs", archive / "5_posterior_summaries"
    (cat4 / "merged").mkdir(parents=True, exist_ok=True)
    cat5.mkdir(parents=True, exist_ok=True)
    project = cfg.project_name

    manifest = pd.read_csv(pestrun / "batch_manifest.csv")
    id_col = [c for c in manifest.columns if c != "batch_id"][0]
    manifest = manifest.rename(columns={id_col: "site_id"})
    manifest.to_csv(cat4 / "merged" / "batch_site_map.csv", index=False)
    batch_ids = sorted(int(b) for b in manifest["batch_id"].unique())
    fids = [str(s) for s in manifest["site_id"]]
    batch_log = json.loads((pestrun / "batch_log.json").read_text())

    problems: list[str] = []
    checks: dict = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_name": args.run_name,
        "pest_run_dir": str(pestrun),
        "declared_posterior_iteration": args.noptmax,
        "batches": {},
    }

    # ---------------- Category 4 ----------------
    phi_rows, par_paths, par_data_paths = [], [], []
    for bid in batch_ids:
        tag = f"batch_{bid:03d}"
        src, dst = pestrun / "pest_archive" / tag, cat4 / tag
        if not src.is_dir():
            problems.append(f"{tag}: pest_archive missing")
            continue
        if dst.exists():
            diff = [
                f.name
                for f in src.iterdir()
                if f.is_file()
                and (not (dst / f.name).exists() or sha256_file(f) != sha256_file(dst / f.name))
            ]
            if diff:
                problems.append(f"{tag}: archive copy differs from pest_archive: {diff[:5]}")
        else:
            shutil.copytree(src, dst)
        n_files = sum(1 for f in dst.iterdir() if f.is_file())

        missing = []
        for it in range(args.noptmax + 1):
            for suffix in ("par.csv", "obs.csv", "base.rei"):
                if not (src / f"{project}.{it}.{suffix}").exists():
                    missing.append(f"{project}.{it}.{suffix}")
        if missing:
            problems.append(f"{tag}: missing iteration files {missing}")

        summ = json.loads((src / "calibration_summary.json").read_text())
        hist = [float(x) for x in summ.get("phi_history", [])]
        for q in phi_history_checks(hist):
            problems.append(f"{tag}: {q}")
        if len(hist) != args.noptmax + 1:
            problems.append(
                f"{tag}: phi history has {len(hist)} entries, expected {args.noptmax + 1}"
            )
        if int(summ.get("iterations_completed", -1)) != args.noptmax:
            problems.append(f"{tag}: iterations_completed={summ.get('iterations_completed')}")

        phi_meas = pd.read_csv(src / f"{project}.phi.meas.csv")
        reals_by_iter = {}
        for it in range(args.noptmax + 1):
            pp = src / f"{project}.{it}.par.csv"
            if pp.exists():
                idx = pd.read_csv(pp, index_col=0).index.astype(str)
                reals_by_iter[it] = int((idx != "base").sum())
        for it, phi in enumerate(hist):
            pm = phi_meas.loc[phi_meas["iteration"] == it]
            row = {
                "batch": bid,
                "iteration": it,
                "phi_best_mean": phi,
                "phi_meas_mean": float(pm["mean"].iloc[0]) if len(pm) else np.nan,
                "phi_meas_sd": float(pm["standard_deviation"].iloc[0]) if len(pm) else np.nan,
                "phi_meas_min": float(pm["min"].iloc[0]) if len(pm) else np.nan,
                "phi_meas_max": float(pm["max"].iloc[0]) if len(pm) else np.nan,
                "n_realizations": reals_by_iter.get(it, np.nan),
            }
            if len(pm) and not np.isclose(row["phi_meas_mean"], phi, rtol=1e-3):
                problems.append(
                    f"{tag} it{it}: phi.meas mean {row['phi_meas_mean']} != summary {phi}"
                )
            phi_rows.append(row)

        n_reals = sorted(set(reals_by_iter.values()))
        log_entry = batch_log.get(str(bid), {})
        if log_entry.get("status") != "ingested":
            problems.append(f"{tag}: batch_log status {log_entry.get('status')!r}")
        checks["batches"][tag] = {
            "n_sites": int((manifest["batch_id"] == bid).sum()),
            "archive_files": n_files,
            "phi_history": hist,
            "phi_reduction_pct": summ.get("phi_reduction_pct"),
            "realizations_by_iteration": reals_by_iter,
            "realizations_dropped": (int(max(n_reals) - min(n_reals)) if n_reals else None),
            "dropped_fids": log_entry.get("dropped_fids", []),
            "batch_log_status": log_entry.get("status"),
        }

        par_paths.append(src / f"{project}.{args.noptmax}.par.csv")
        pdc = src / f"{project.lower()}.par_data.csv"
        if pdc.exists():
            par_data_paths.append(pdc)
        else:
            problems.append(f"{tag}: par_data.csv missing (no bounds)")

    pd.DataFrame(phi_rows).to_csv(cat4 / "merged" / "merged_phi_history.csv", index=False)
    merged = merge_par_csvs([pp for pp in par_paths if pp.exists()])
    merged_csv = cat4 / "merged" / "merged_posterior.csv"
    merged.to_csv(merged_csv, index_label="real_name")
    med = posterior_medians(merged, fids)
    nested = {fid: {k: float(v) for k, v in row.items()} for fid, row in med.iterrows()}
    (cat4 / "merged" / "merged_posterior.json").write_text(json.dumps(nested, indent=2))
    (cat4 / "merged" / "README.md").write_text(
        f"merged_posterior.csv: iteration-{args.noptmax} `.par.csv` ensembles of all batches, column-joined "
        "on real_name (`base` row kept). Site-parameters are disjoint across batches so the "
        "per-column median equals the per-batch median the batch runner ingested. Use it as "
        "`evaluate.py --par-csv` for an explicit parameter source. merged_posterior.json: "
        "median over realizations, PEST parameter names (ks_alpha/kr_alpha = ks_damp/kr_damp).\n"
        "merged_phi_history.csv: best-mean phi per batch x iteration with phi.meas statistics.\n"
        "batch_site_map.csv: batch_id -> site_id.\n"
    )
    if set(med.index) != set(fids):
        problems.append(f"posterior covers {len(med)} sites, manifest has {len(fids)}")

    # ---------------- completion checks vs the container ----------------
    root = zarr.open_group(cfg.container_path, mode="r")
    cparams = container_params_frame(root, fids)
    comp = compare_ingested(cparams, med)
    comp.to_csv(cat4 / "merged" / "ingested_vs_posterior.csv", index=False)
    max_rel = float(np.nanmax(comp["rel_err"])) if comp["rel_err"].notna().any() else np.nan
    if not np.isfinite(max_rel) or max_rel > INGEST_REL_TOL or comp["rel_err"].isna().any():
        problems.append(
            f"container calibration != merged iteration-{args.noptmax} medians "
            f"(max rel err {max_rel}, {int(comp['rel_err'].isna().sum())} unmatched)"
        )
    calibrated = np.asarray(root["calibration/metadata/calibrated"][:]).astype(bool)
    cal_attrs = dict(root["calibration"].attrs)
    checks["container"] = {
        "path": str(cfg.container_path),
        "n_calibrated_flag": int(calibrated.sum()),
        "n_fields": int(len(calibrated)),
        "summary_stat": cal_attrs.get("summary_stat"),
        "n_batches_completed": cal_attrs.get("n_batches_completed"),
        "max_rel_err_vs_posterior_median": max_rel,
    }
    if int(calibrated.sum()) != len(fids):
        problems.append(f"container calibrated flag set on {int(calibrated.sum())}/{len(fids)}")

    launch_ts = json.loads((pestrun / "run_manifest.json").read_text()).get("timestamp")
    launch_epoch = datetime.fromisoformat(launch_ts).timestamp() if launch_ts else None
    base_cal = Path(args.baseline_container) / "calibration"
    base_newest = newest_mtime(base_cal) if base_cal.exists() else None
    checks["baseline_container_untouched"] = {
        "baseline_calibration_dir": str(base_cal),
        "newest_file_mtime": (
            datetime.fromtimestamp(base_newest).isoformat() if base_newest else None
        ),
        "grassbasis_launch": launch_ts,
        "ok": bool(base_newest is not None and launch_epoch and base_newest < launch_epoch),
    }
    if not checks["baseline_container_untouched"]["ok"]:
        problems.append("baseline container calibration/ modified after the GrassBasis launch")

    # ---------------- Category 5 ----------------
    ens = ensemble_long(merged, fids)
    site_summary = (
        ens.groupby(["site_id", "param"])["value"]
        .apply(lambda s: pd.Series(_stats(s.to_numpy(float))))
        .unstack()
        .reset_index()
    )
    site_summary["internal_name"] = site_summary["param"].map(lambda q: PEST_TO_INTERNAL.get(q, q))
    site_summary.to_csv(cat5 / "posterior_site_summary.csv", index=False)

    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    gid = cfg.feature_id_col if cfg.feature_id_col in gdf.columns else "sid"
    gdf = gdf.drop_duplicates(gid).set_index(gdf[gid].astype(str))
    groups = pd.DataFrame(index=med.index)
    groups["lulc"] = [
        f"glc10_{int(gdf.loc[f, 'glc10_lulc'])}" if f in gdf.index else "unknown" for f in med.index
    ]
    groups["irrigation_class"] = read_irrigation_class(root, list(med.index))
    groups["region"] = ["CONUS" if f.startswith("US-") else "ex-CONUS" for f in med.index]
    groups.to_csv(cat5 / "site_groups.csv", index_label="site_id")

    bounds = param_bounds(par_data_paths, fids)
    bounds.to_csv(cat5 / "parameter_bounds_by_site.csv")
    hit_rows = boundary_hit_rows(med, bounds, "ALL", "ALL", args.run_name)
    for kind in ("lulc", "irrigation_class", "region"):
        for label, sub in groups.groupby(kind):
            hit_rows += boundary_hit_rows(
                med.loc[sub.index], bounds, str(label), kind, args.run_name
            )
    hits = pd.DataFrame(hit_rows)
    hits.to_csv(cat5 / "boundary_hit_rates.csv", index=False)

    def _grouped(kind: str, path: Path):
        recs = []
        for label, sub in groups.groupby(kind):
            for param in sorted(med.columns):
                v = med.loc[sub.index, param].dropna().to_numpy(float)
                if len(v) == 0:
                    continue
                recs.append(
                    {
                        kind: label,
                        "parameter": param,
                        "internal_name": PEST_TO_INTERNAL.get(param, param),
                        "n_sites": int(len(v)),
                        "median": float(np.median(v)),
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    }
                )
        pd.DataFrame(recs).to_csv(path, index=False)

    _grouped("lulc", cat5 / "lulc_grouped_summary.csv")
    _grouped("irrigation_class", cat5 / "irrigated_grouped_summary.csv")
    _grouped("region", cat5 / "region_grouped_summary.csv")

    # baseline posterior comparison (read-only on the frozen baseline pest_archive)
    comp_frames = [med.add_suffix(f"__grassbasis_it{args.noptmax}")]
    base_arch = Path(args.baseline_pestrun) / "pest_archive"
    for it in [int(x) for x in args.baseline_iterations.split(",") if x.strip()]:
        paths = sorted(base_arch.glob(f"batch_*/{project}.{it}.par.csv"))
        if not paths:
            continue
        bmed = posterior_medians(merge_par_csvs(paths), fids)
        comp_frames.append(bmed.add_suffix(f"__baseline_it{it}"))
        # baseline bounds are identical by construction (same pest_builder, same sites); use
        # the GrassBasis per-site bounds for a like-for-like hit rate
        hits_b = pd.DataFrame(boundary_hit_rows(bmed, bounds, "ALL", "ALL", f"baseline_it{it}"))
        for kind in ("irrigation_class",):
            for label, sub in groups.groupby(kind):
                hits_b = pd.concat(
                    [
                        hits_b,
                        pd.DataFrame(
                            boundary_hit_rows(
                                bmed.loc[sub.index], bounds, str(label), kind, f"baseline_it{it}"
                            )
                        ),
                    ],
                    ignore_index=True,
                )
        hits = pd.concat([hits, hits_b], ignore_index=True)
    pd.concat(comp_frames, axis=1).to_csv(
        cat5 / "posterior_comparison_baseline.csv", index_label="site_id"
    )
    hits.to_csv(cat5 / "boundary_hit_rates.csv", index=False)

    # ---------------- Category 1: run log ----------------
    log = Path(args.log)
    if log.exists():
        gz = archive / "1_provenance" / "run_stdout.log.gz"
        with open(log, "rb") as fi, gzip.open(gz, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        checks["run_log"] = {"source": str(log), "archived": str(gz), "bytes": log.stat().st_size}
    else:
        problems.append(f"run log not found: {log}")

    checks["merged_posterior_sha256"] = sha256_file(merged_csv)
    checks["problems"] = problems
    checks["pass"] = not problems
    (cat4 / "completion_checks.json").write_text(json.dumps(checks, indent=2, default=str))

    flagged = hits.loc[hits["boundary_seeking"] & (hits["run_name"] == args.run_name)]
    print(f"Cat 4 -> {cat4}  ({len(batch_ids)} batches, {len(med)} sites)")
    print(f"Cat 5 -> {cat5}")
    print(
        f"merged posterior: {merged.shape[0]} rows x {merged.shape[1]} columns; sha256 {checks['merged_posterior_sha256'][:16]}"
    )
    print(f"ingested vs posterior median: max rel err {max_rel:.3e}")
    print("boundary-seeking (>50 % of group at a bound):")
    print(
        flagged[
            ["parameter", "lulc_group", "n_sites", "lower_hit_rate", "upper_hit_rate"]
        ].to_string(index=False)
        if len(flagged)
        else "  none"
    )
    if problems:
        print("COMPLETION CHECK FAILURES:")
        for q in problems:
            print("  -", q)
        sys.exit(1)
    print("completion checks: PASS")


if __name__ == "__main__":
    main()

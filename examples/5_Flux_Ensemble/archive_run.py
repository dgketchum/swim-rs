"""Materialize the RUN_POLICY 7-category archive for an Example 5 run.

This is a post-hoc tool. Category 1 provenance basics (git SHA/status/diff,
config copy, SHAs, container path, command) are captured by the launch script
(commands.sh) *before* calibration. Category 4 (full PEST trajectory + raw
problem definition: .pst, loc.mat) is captured by ``calibrate.py`` immediately
after PEST++ finishes, *before* any cleanup (archive-before-cleanup rule).

This script fills in everything else and rounds out Category 1:

  Cat 1  enrich:   environment.json, run_metadata.json, container_manifest.json,
                   run_stdout.log.gz
  Cat 2  input:    container_health.json, etf_capture_counts.csv (is_active_target),
                   ndvi_coverage.csv, met_completeness.csv, *_sites_excluded.csv
  Cat 3  decoded:  observation_table.csv, observation_metadata.csv (weight
                   decomposition), parameter_bounds.csv, localizer_summary.json
  Cat 5  post:     posterior_site_summary.csv, boundary_hit_rates.csv (per-LULC),
                   lulc_grouped_summary.csv, irrigated_grouped_summary.csv
  Cat 6  eval:     daily_paired_metrics.csv, monthly_paired_metrics.csv,
                   site_daily_timeseries/, evaluation_metadata.json

Category 7 (figure audit) is deferred until manuscript figures are defined; the
gap is recorded in archive/GAPS.md.

Each category runs under its own try/except so one failure does not lose the
rest; failures and known gaps are written to archive/GAPS.md.

Usage:
    uv run python examples/5_Flux_Ensemble/archive_run.py --results-tag run21 \
        --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run20.swim
"""

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import traceback
from datetime import UTC, datetime

import evaluate as ev
import numpy as np
import pandas as pd

from swimrs.container import SwimContainer

NOPTMAX = 3

# GLC10 (FROM-GLC10, 10 m) code -> label
GLC10_LABELS = {
    10: "Cropland",
    20: "Forest",
    30: "Grassland",
    40: "Shrubland",
    50: "Wetland",
    60: "Water",
    70: "Tundra",
    80: "Impervious",
    90: "Barren",
    100: "SnowIce",
}

MET_EVAL_VARS = ["eto", "eto_corr", "srad", "tmax", "tmin", "prcp"]


def _utcnow():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _as_str(x):
    """Coerce a zarr StringDType scalar/0-d array to a python str."""
    a = np.asarray(x)
    if a.ndim == 0:
        x = a.item()
    return "" if x is None else str(x)


def _glc10_label(code):
    try:
        return GLC10_LABELS.get(int(code), f"GLC{int(code)}")
    except (ValueError, TypeError):
        return "Unknown"


# --------------------------------------------------------------------------- #
# Category 1: provenance enrichment
# --------------------------------------------------------------------------- #
def cat1_enrich(cfg, container, container_path, prov_dir, log_path, gaps):
    os.makedirs(prov_dir, exist_ok=True)

    # config.toml + config_sha256.txt — snapshot the TOML that was actually
    # loaded (cfg.config_path), not a hardcoded default. Runs on a variant
    # config are otherwise indistinguishable from the canonical one in the
    # archive, which is exactly the provenance failure this category exists
    # to prevent.
    src_cfg = getattr(cfg, "config_path", None)
    if src_cfg and os.path.exists(src_cfg):
        shutil.copyfile(src_cfg, os.path.join(prov_dir, "config.toml"))
        with open(src_cfg, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        with open(os.path.join(prov_dir, "config_sha256.txt"), "w") as fh:
            fh.write(f"{digest}  {os.path.basename(src_cfg)}\n")
    else:
        gaps.append(f"Cat1: config.toml not snapshotted (config_path={src_cfg!r})")

    # environment.json
    def _cmd(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<error: {exc}>"

    env = {
        "captured": _utcnow(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pestpp_ies_version": _cmd(["pestpp-ies", "--version"]),
    }
    for mod in ("numpy", "scipy", "pandas", "pyemu", "osgeo.gdal", "pyproj", "shapely"):
        try:
            if mod == "osgeo.gdal":
                try:
                    from osgeo import gdal

                    env["gdal_version"] = gdal.__version__
                except Exception:  # noqa: BLE001
                    import rasterio

                    env["gdal_version"] = rasterio.__gdal_version__
            else:
                m = __import__(mod)
                env[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            env[f"{mod}_version"] = f"<unavailable: {exc}>"
    try:
        import pyproj

        env["proj_version"] = pyproj.proj_version_str
        env["geos_version"] = __import__("shapely").geos_version_string
    except Exception:  # noqa: BLE001
        pass
    env["uv_pip_freeze"] = _cmd(["uv", "pip", "freeze"]).splitlines()
    with open(os.path.join(prov_dir, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)

    # run_metadata.json
    log_mtime = None
    elapsed = None
    if log_path and os.path.exists(log_path):
        log_mtime = datetime.fromtimestamp(os.path.getmtime(log_path), tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            with open(log_path) as fh:
                for line in fh:
                    if line.startswith("Total elapsed:"):
                        elapsed = line.strip()
        except Exception:  # noqa: BLE001
            pass
    try:
        swim_version = __import__("swimrs").__version__
    except Exception:  # noqa: BLE001
        swim_version = "unknown"
    meta = {
        "captured": _utcnow(),
        "hostname": socket.gethostname(),
        "project_name": cfg.project_name,
        "etf_target_model_effective": "ensemble",
        "etf_target_model_toml": getattr(cfg, "etf_target_model", None),
        "ensemble_source": getattr(cfg, "ensemble_source", None),
        "etf_ensemble_members": list(getattr(cfg, "etf_ensemble_members", []) or []),
        "etf_weighting_mode": getattr(cfg, "etf_weighting_mode", None),
        "mask_mode": getattr(cfg, "mask_mode", None),
        "realizations": getattr(cfg, "realizations", None),
        "workers": getattr(cfg, "workers", None),
        "noptmax": NOPTMAX,
        "swim_version": swim_version,
        "container_path": container_path,
        "n_fields": int(container.n_fields),
        "log_last_modified": log_mtime,
        "elapsed": elapsed,
    }
    with open(os.path.join(prov_dir, "run_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # container_manifest.json
    root = container._root
    manifest = {"captured": _utcnow(), "container_path": container_path, "arrays": []}
    members = list(getattr(cfg, "etf_ensemble_members", []) or [])
    mandatory_prefixes = [f"remote_sensing/etf/landsat/{m}/" for m in members] + [
        "remote_sensing/etf/landsat/ensemble/",
        "remote_sensing/ndvi/landsat/",
        "meteorology/gridmet/eto",
        "meteorology/gridmet/prcp",
        "properties/soils/",
        "properties/land_cover/",
        "properties/irrigation/",
    ]

    def _walk(group, prefix=""):
        for key in sorted(group.keys()):
            child = group[key]
            path = f"{prefix}/{key}" if prefix else key
            if hasattr(child, "shape"):
                arr = child
                entry = {
                    "path": path,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "zattrs_sha256": _sha256_bytes(
                        json.dumps(dict(arr.attrs), sort_keys=True, default=str).encode()
                    ),
                }
                try:
                    data = arr[:]
                    if np.issubdtype(np.asarray(data).dtype, np.number):
                        entry["nonnull_count"] = int(np.sum(~np.isnan(data)))
                    else:
                        entry["nonnull_count"] = int(
                            np.sum([1 for v in np.asarray(data).ravel() if _as_str(v)])
                        )
                    if any(path.startswith(p) for p in mandatory_prefixes):
                        entry["content_sha256"] = _sha256_bytes(
                            np.ascontiguousarray(data).tobytes()
                        )
                except Exception as exc:  # noqa: BLE001
                    entry["read_error"] = str(exc)
                manifest["arrays"].append(entry)
            else:
                _walk(child, path)

    _walk(root)
    with open(os.path.join(prov_dir, "container_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # run_stdout.log.gz
    if log_path and os.path.exists(log_path):
        with (
            open(log_path, "rb") as src,
            gzip.open(os.path.join(prov_dir, "run_stdout.log.gz"), "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
    else:
        gaps.append("Cat1: run stdout log not found; run_stdout.log.gz not archived")

    print(f"  Cat1: enriched provenance -> {prov_dir}")


# --------------------------------------------------------------------------- #
# Category 2: input data audit
# --------------------------------------------------------------------------- #
def cat2_input_audit(cfg, container, cat2_dir, gaps):
    os.makedirs(cat2_dir, exist_ok=True)
    root = container._root
    fids = list(container.field_uids)
    n = len(fids)
    members = list(getattr(cfg, "etf_ensemble_members", []) or [])

    # container health (best-effort)
    try:
        report = container.report(
            config={
                "mask_mode": getattr(cfg, "mask_mode", "none"),
                "etf_target_model": "ensemble",
                "etf_ensemble_members": members,
                "met_source": "gridmet",
            },
            raise_on_fail=False,
            health_profile="calibration",
        )
        with open(os.path.join(cat2_dir, "container_health.json"), "w") as fh:
            json.dump(report.to_json(), fh, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"Cat2: container health check unavailable ({exc})")

    # etf_capture_counts.csv
    active_members = set(members)
    rows = []
    etf_group = "remote_sensing/etf/landsat"
    if etf_group in root:
        for model in sorted(root[etf_group].keys()):
            for mask in sorted(root[f"{etf_group}/{model}"].keys()):
                path = f"{etf_group}/{model}/{mask}"
                data = np.asarray(root[path][:])
                counts = np.sum(~np.isnan(data), axis=0)
                is_active = (model in active_members or model == "ensemble") and (mask == "no_mask")
                for i, fid in enumerate(fids):
                    rows.append(
                        {
                            "site": fid,
                            "instrument": "landsat",
                            "model": model,
                            "mask": mask,
                            "n_nonnull": int(counts[i]),
                            "is_active_target": bool(is_active),
                        }
                    )
    etf_counts = pd.DataFrame(rows)
    etf_counts.to_csv(os.path.join(cat2_dir, "etf_capture_counts.csv"), index=False)

    # ndvi_coverage.csv
    time_index = container._time_index
    ndvi_rows = []
    ndvi_path = "remote_sensing/ndvi/landsat/no_mask"
    if ndvi_path in root:
        ndvi = np.asarray(root[ndvi_path][:])
        for i, fid in enumerate(fids):
            valid = np.where(~np.isnan(ndvi[:, i]))[0]
            first = time_index[valid[0]] if len(valid) else None
            last = time_index[valid[-1]] if len(valid) else None
            months = set(time_index[valid].month) if len(valid) else set()
            ndvi_rows.append(
                {
                    "site": fid,
                    "n_nonnull": int(len(valid)),
                    "first_date": first.strftime("%Y-%m-%d") if first is not None else "",
                    "last_date": last.strftime("%Y-%m-%d") if last is not None else "",
                    "seasonal_coverage": bool({4, 5, 6, 7, 8, 9}.issubset(months)),
                }
            )
    pd.DataFrame(ndvi_rows).to_csv(os.path.join(cat2_dir, "ndvi_coverage.csv"), index=False)

    # met_completeness.csv
    met_rows = []
    for i, fid in enumerate(fids):
        row = {"site": fid}
        for var in MET_EVAL_VARS:
            p = f"meteorology/gridmet/{var}"
            if p in root:
                row[f"{var}_nonnull"] = int(np.sum(~np.isnan(np.asarray(root[p][:, i]))))
            else:
                row[f"{var}_nonnull"] = 0
        met_rows.append(row)
    met_df = pd.DataFrame(met_rows)
    met_df.to_csv(os.path.join(cat2_dir, "met_completeness.csv"), index=False)

    # exclusion CSVs
    # Calibration: sites with null AWC or no active-target ETf capture.
    awc = np.asarray(root["properties/soils/awc"][:]) if "properties/soils/awc" in root else None
    active_total = (
        etf_counts[etf_counts["is_active_target"]].groupby("site")["n_nonnull"].sum()
        if not etf_counts.empty
        else pd.Series(dtype=int)
    )
    cal_excl = []
    for i, fid in enumerate(fids):
        reasons = []
        if awc is not None and (np.isnan(awc[i]) or awc[i] <= 0):
            reasons.append("null_or_zero_awc")
        if active_total.get(fid, 0) == 0:
            reasons.append("no_active_target_etf")
        if reasons:
            cal_excl.append({"site": fid, "reason": ";".join(reasons)})
    pd.DataFrame(cal_excl, columns=["site", "reason"]).to_csv(
        os.path.join(cat2_dir, "calibration_sites_excluded.csv"), index=False
    )

    # Evaluation: canonical exclusion policy + sites without flux files.
    flux_dir = ev.resolve_flux_dir(cfg)
    eval_excl = []
    for fid in fids:
        reasons = []
        if fid in ev.EXCLUDED_SITES:
            reasons.append("exclusion_policy")
        if not os.path.exists(os.path.join(flux_dir, f"{fid}_daily_data.csv")):
            reasons.append("no_flux_file")
        if reasons:
            eval_excl.append({"site": fid, "reason": ";".join(reasons)})
    pd.DataFrame(eval_excl, columns=["site", "reason"]).to_csv(
        os.path.join(cat2_dir, "evaluation_sites_excluded.csv"), index=False
    )

    print(f"  Cat2: input audit -> {cat2_dir} ({n} sites)")


# --------------------------------------------------------------------------- #
# Category 3: problem definition (decoded)
# --------------------------------------------------------------------------- #
def _parse_param_col(col):
    """pname:p_{param}_{fid}_:0_ptype:... -> (param, fid) using fid suffix match.

    Returns (raw_token,) so the caller can match fids; here we strip the
    pname wrapper and the trailing _:0, leaving '{param}_{fid}'.
    """
    token = col.split("_ptype:")[0]
    token = token.replace("pname:p_", "").replace("pname:", "")
    token = token.rsplit("_:0", 1)[0]
    return token


def _split_param_fid(token, fids_lower):
    for fid_l in fids_lower:
        if token.lower().endswith("_" + fid_l):
            return token[: -(len(fid_l) + 1)], fid_l
    return token, None


def _find_sidecar(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def cat3_problem_definition(cfg, container, cat3_dir, pst_path, weight_audit_csv, gaps):
    """Decode the inverse problem from the .pst external sidecar CSVs.

    The pcf-v2 .pst references par_data/obs_data CSVs by relative name; parsing
    those directly is more robust than pyemu.Pst() (which would also need the
    600 .tpl/.ins files referenced by the model input/output sections).
    """
    os.makedirs(cat3_dir, exist_ok=True)
    fids = list(container.field_uids)
    fid_by_lower = {f.lower(): f for f in fids}
    fids_lower = list(fid_by_lower.keys())
    proj_l = cfg.project_name.lower()
    src_dirs = [cat3_dir]
    pest_run_dir = getattr(cfg, "pest_run_dir", None)
    if pest_run_dir:
        src_dirs.append(os.path.join(pest_run_dir, "pest"))

    par_path = _find_sidecar(f"{proj_l}.par_data.csv", src_dirs)
    obs_path = _find_sidecar(f"{proj_l}.obs_data.csv", src_dirs)
    if par_path is None or obs_path is None:
        gaps.append(
            "Cat3: par_data/obs_data sidecar CSVs not found; observation_table, "
            "parameter_bounds, observation_metadata not produced"
        )
        return

    # parameter_bounds.csv (pargp is the parameter group; fid parsed from parnme)
    par = pd.read_csv(par_path)
    sites, params = [], []
    for pn, pg in zip(par["parnme"], par["pargp"]):
        _, fid_l = _split_param_fid(_parse_param_col(pn), fids_lower)
        sites.append(fid_by_lower.get(fid_l))
        params.append(pg)
    pb = pd.DataFrame(
        {
            "parnme": par["parnme"],
            "param": params,
            "site": sites,
            "initial_value": par["parval1"].astype(float),
            "lower_bound": par["parlbnd"].astype(float),
            "upper_bound": par["parubnd"].astype(float),
            "transform": par["partrans"],
            "status": par["partrans"].apply(
                lambda t: "fixed" if t == "fixed" else ("tied" if t == "tied" else "adjustable")
            ),
        }
    )
    pb.to_csv(os.path.join(cat3_dir, "parameter_bounds.csv"), index=False)

    # observation_table.csv
    obs = pd.read_csv(obs_path)
    obs[["obsnme", "obsval", "weight", "obgnme"]].to_csv(
        os.path.join(cat3_dir, "observation_table.csv"), index=False
    )

    # observation_metadata.csv (decoded + weight decomposition)
    time_index = container._time_index
    n_days = len(time_index)
    audit = None
    audit_key = None
    if weight_audit_csv and os.path.exists(weight_audit_csv):
        audit = pd.read_csv(weight_audit_csv, parse_dates=["date"])
        audit_key = audit.set_index(["fid", "date"])

    nonzero = obs[obs["weight"].astype(float) > 0]
    meta_rows = []
    for obsnme, obsval, weight, arr_i in zip(
        nonzero["obsnme"], nonzero["obsval"], nonzero["weight"], nonzero["i"]
    ):
        is_etf = "obs_etf_" in obsnme
        is_swe = "obs_swe_" in obsnme
        if is_etf:
            fid_token = obsnme.split("obs_etf_")[1].split("_otype:")[0]
        elif is_swe:
            fid_token = obsnme.split("obs_swe_")[1].split("_otype:")[0]
        else:
            fid_token = ""
        fid = fid_by_lower.get(fid_token, fid_token)
        date = None
        if not pd.isna(arr_i) and 0 <= int(arr_i) < n_days:
            date = time_index[int(arr_i)]
        row = {
            "obsnme": obsnme,
            "site": fid,
            "date": date.strftime("%Y-%m-%d") if date is not None else "",
            "sensor": "landsat" if is_etf else "",
            "model": "ensemble" if is_etf else ("swe" if is_swe else ""),
            "mask_mode": getattr(cfg, "mask_mode", "none"),
            "target_etf": float(obsval),
            "member_count": "",
            "ensemble_std": "",
            "mad_included": "",
            "eto_correction_factor": 1.0,
            "raw_weight": "",
            "final_weight": float(weight),
            "weight_formula": "obsval_over_std_plus_floor" if is_etf else "fixed",
        }
        if is_etf and audit_key is not None and date is not None:
            try:
                a = audit_key.loc[(fid, pd.Timestamp(date))]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                row["member_count"] = int(a["member_count"])
                row["ensemble_std"] = float(a["member_std"])
                row["raw_weight"] = float(a["weight_pre_pdc"])
                row["mad_included"] = bool(a["eligible"])
            except KeyError:
                pass
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(cat3_dir, "observation_metadata.csv"), index=False)
    if audit is None:
        gaps.append(
            "Cat3: etf_weight_audit.csv not found; observation_metadata weight "
            "decomposition columns left blank"
        )

    # localizer_summary.json (prefer precomputed; the 415 MB ASCII loc.mat is
    # not auto-summarized to avoid loading it into memory)
    dst = os.path.join(cat3_dir, "localizer_summary.json")
    loc_summary = _find_sidecar("localizer_summary.json", src_dirs)
    if loc_summary and os.path.abspath(loc_summary) != os.path.abspath(dst):
        shutil.copyfile(loc_summary, dst)
    elif loc_summary is None:
        gaps.append("Cat3: localizer_summary.json not found; not produced")

    print(
        f"  Cat3: decoded problem definition -> {cat3_dir} "
        f"({len(meta_df)} active obs, {len(pb)} params)"
    )


# --------------------------------------------------------------------------- #
# Category 5: posterior summaries
# --------------------------------------------------------------------------- #
def cat5_posterior(cfg, container, cat5_dir, par_csv, cat3_dir, gaps):
    os.makedirs(cat5_dir, exist_ok=True)
    fids = list(container.field_uids)
    fid_by_lower = {f.lower(): f for f in fids}
    fids_lower = list(fid_by_lower.keys())

    df = pd.read_csv(par_csv, index_col=0)
    df = df.loc[df.index != "base"]

    rows = []
    median_by_sp = {}
    for col in df.columns:
        token = _parse_param_col(col)
        param, fid_l = _split_param_fid(token, fids_lower)
        if fid_l is None:
            continue
        fid = fid_by_lower[fid_l]
        vals = df[col].astype(float).values
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        rows.append(
            {
                "site": fid,
                "parameter": param,
                "median": float(med),
                "mean": mean,
                "std": std,
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(q75 - q25),
                "cv": float(std / mean) if mean else np.nan,
            }
        )
        median_by_sp[(fid, param)] = float(med)
    post = pd.DataFrame(rows)
    post.to_csv(os.path.join(cat5_dir, "posterior_site_summary.csv"), index=False)

    # LULC + irrigation grouping
    root = container._root
    glc = (
        np.asarray(root["properties/land_cover/glc10"][:])
        if "properties/land_cover/glc10" in root
        else np.full(len(fids), -1)
    )
    irr_frac = (
        np.asarray(root["properties/irrigation/irr"][:])
        if "properties/irrigation/irr" in root
        else np.full(len(fids), np.nan)
    )
    lulc_by_fid = {fids[i]: _glc10_label(glc[i]) for i in range(len(fids))}
    irr_by_fid = {
        fids[i]: ("irrigated" if irr_frac[i] > 0.5 else "rainfed") for i in range(len(fids))
    }
    post["lulc"] = post["site"].map(lulc_by_fid)
    post["irrigation"] = post["site"].map(irr_by_fid)

    # grouped summaries
    def _grouped(group_col, out_name):
        g = (
            post.groupby([group_col, "parameter"])["median"]
            .agg(["median", "mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n_sites"})
        )
        g.to_csv(os.path.join(cat5_dir, out_name), index=False)

    _grouped("lulc", "lulc_grouped_summary.csv")
    _grouped("irrigation", "irrigated_grouped_summary.csv")

    # boundary_hit_rates.csv (per-LULC + ALL), needs bounds from cat3
    bounds_path = os.path.join(cat3_dir, "parameter_bounds.csv")
    if os.path.exists(bounds_path):
        bnds = pd.read_csv(bounds_path)
        # one bound per (param, site); collapse to (param)->(lb, ub) assuming
        # shared bounds per parameter group, but key by site when available.
        bnd_by_sp = {}
        for _, r in bnds.iterrows():
            if pd.notna(r.get("site")):
                bnd_by_sp[(r["site"], r["param"])] = (r["lower_bound"], r["upper_bound"])
        tol = 0.01  # within 1% of the bound range
        hit_rows = []
        post_sp = post.copy()
        for param in sorted(post_sp["parameter"].unique()):
            sub = post_sp[post_sp["parameter"] == param]
            for grp_label, grp in [("ALL", sub)] + [
                (lab, sub[sub["lulc"] == lab]) for lab in sorted(sub["lulc"].unique())
            ]:
                lower_hits = upper_hits = counted = 0
                for _, r in grp.iterrows():
                    key = (r["site"], param)
                    if key not in bnd_by_sp:
                        continue
                    lb, ub = bnd_by_sp[key]
                    rng = ub - lb
                    if rng <= 0:
                        continue
                    counted += 1
                    if abs(r["median"] - lb) <= tol * rng:
                        lower_hits += 1
                    if abs(r["median"] - ub) <= tol * rng:
                        upper_hits += 1
                if counted:
                    hit_rows.append(
                        {
                            "run_name": os.path.basename(os.path.dirname(cat5_dir)),
                            "parameter": param,
                            "lulc_group": grp_label,
                            "n_sites": counted,
                            "lower_hit_rate": round(lower_hits / counted, 4),
                            "upper_hit_rate": round(upper_hits / counted, 4),
                            "bound_tolerance": tol,
                        }
                    )
        pd.DataFrame(hit_rows).to_csv(os.path.join(cat5_dir, "boundary_hit_rates.csv"), index=False)
    else:
        gaps.append("Cat5: parameter_bounds.csv missing; boundary_hit_rates not produced")

    print(f"  Cat5: posterior summaries -> {cat5_dir} ({len(post)} site-params)")


# --------------------------------------------------------------------------- #
# Category 6: evaluation products
# --------------------------------------------------------------------------- #
def _full_model_run(cfg, container, fids, par_csv):
    """Run the calibrated model keeping the full DailyOutput for time series."""
    import json as _json
    import tempfile

    from swimrs.process.input import build_swim_input
    from swimrs.process.loop_fast import run_daily_loop_fast

    params = ev.parse_pest_params(par_csv, fids)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        _json.dump(params, tmp)
        params_json = tmp.name
    try:
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
        )
        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        out_fids = list(swim_input.fids)
        swim_input.close()
        return output, dates, out_fids
    finally:
        for p in (temp_h5, params_json):
            if os.path.exists(p):
                os.remove(p)


def active_refet_arrays(root, met_source="gridmet", refet_type="eto"):
    """Reference ET the model actually used, plus the raw series for provenance.

    Mirrors the model-input preference (swimrs.process.input): the corrected
    series (``{refet_type}_corr``) when present, else the raw series. Returns
    ``(active, raw)`` where ``raw`` is None unless the corrected series is the
    active one — the archive then exports it under ``eto_raw`` so the ``eto``
    column always carries what SWIM consumed.
    """
    corr_path = f"meteorology/{met_source}/{refet_type}_corr"
    raw_path = f"meteorology/{met_source}/{refet_type}"
    if corr_path in root:
        active = np.asarray(root[corr_path][:])
        raw = np.asarray(root[raw_path][:]) if raw_path in root else None
        return active, raw
    return np.asarray(root[raw_path][:]), None


def cat6_evaluation(cfg, container, cat6_dir, par_csv, cat3_dir, gaps):
    os.makedirs(cat6_dir, exist_ok=True)
    flux_dir = ev.resolve_flux_dir(cfg)
    fids = ev.apply_exclusions(list(container.field_uids))

    # paired metrics (reuse evaluate.py's authoritative logic); the grouped
    # bundle comes from the same single forward run as the per-site table
    try:
        daily_bundle = ev.evaluate_benchmark_daily(
            cfg, container, par_csv, list(container.field_uids), flux_dir, openet_source="volk"
        )
        daily_bundle.site_metrics.to_csv(os.path.join(cat6_dir, "daily_paired_metrics.csv"))
        ev.write_grouped_outputs(daily_bundle, cat6_dir, "daily", openet_source="volk")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"Cat6: daily_paired_metrics failed ({exc})")
    try:
        monthly_bundle = ev.evaluate_benchmark_monthly(
            cfg, container, par_csv, list(container.field_uids), flux_dir
        )
        monthly_bundle.site_metrics.to_csv(os.path.join(cat6_dir, "monthly_paired_metrics.csv"))
        ev.write_grouped_outputs(monthly_bundle, cat6_dir, "monthly", openet_source="volk")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"Cat6: monthly_paired_metrics failed ({exc})")

    # site_daily_timeseries/
    ts_dir = os.path.join(cat6_dir, "site_daily_timeseries")
    os.makedirs(ts_dir, exist_ok=True)
    try:
        output, dates, out_fids = _full_model_run(cfg, container, fids, par_csv)
        idx = {f: i for i, f in enumerate(out_fids)}
        # observed ETf + weights from decoded metadata (if available)
        obs_meta_path = os.path.join(cat3_dir, "observation_metadata.csv")
        obs_meta = pd.read_csv(obs_meta_path) if os.path.exists(obs_meta_path) else pd.DataFrame()
        root = container._root
        eto, eto_raw = active_refet_arrays(
            root, refet_type=getattr(cfg, "refet_type", "eto") or "eto"
        )
        prcp = np.asarray(root["meteorology/gridmet/prcp"][:])
        cfids = list(container.field_uids)
        cidx = {f: i for i, f in enumerate(cfids)}
        n_written = 0
        for fid in out_fids:
            i = idx[fid]
            ci = cidx.get(fid)
            data = {
                "date": dates,
                "swim_ET": output.eta[:, i],
                "etf_model": output.etf[:, i],
                "precip": prcp[:, ci] if ci is not None else np.nan,
                "eto": eto[:, ci] if ci is not None else np.nan,
                "ndvi_kcb": output.kcb[:, i],
                "ks": output.ks[:, i],
                "rz_depletion": output.depl_root[:, i],
                "irr_applied": output.irr_sim[:, i],
                "swe": output.swe[:, i],
            }
            if eto_raw is not None:
                data["eto_raw"] = eto_raw[:, ci] if ci is not None else np.nan
            df = pd.DataFrame(data)
            # flux ET
            flux = ev.load_flux_et(fid, flux_dir)
            df = df.merge(flux.rename("flux_ET"), left_on="date", right_index=True, how="left")
            # observed etf + weight on overpass days
            if not obs_meta.empty:
                sub = obs_meta[(obs_meta["site"] == fid) & (obs_meta["model"] == "ensemble")]
                if not sub.empty:
                    sub = sub.assign(date=pd.to_datetime(sub["date"]))
                    om = sub.set_index("date")
                    df["observed_etf"] = df["date"].map(om["target_etf"])
                    df["obs_weight"] = df["date"].map(om["final_weight"])
            if "observed_etf" not in df.columns:
                df["observed_etf"] = np.nan
                df["obs_weight"] = np.nan
            df["is_overpass"] = df["observed_etf"].notna()
            df["sensor"] = np.where(df["is_overpass"], "landsat", "")
            df.to_csv(os.path.join(ts_dir, f"{fid}.csv"), index=False)
            n_written += 1
        print(f"  Cat6: wrote {n_written} site daily time series")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"Cat6: site_daily_timeseries failed ({exc})")
        traceback.print_exc()

    # evaluation_metadata.json
    meta = {
        "captured": _utcnow(),
        "evaluation_date": _utcnow(),
        "swim_version": getattr(__import__("swimrs"), "__version__", "unknown"),
        "flux_dir": flux_dir,
        "flux_et_column": "ET_corr",
        "n_sites_container": int(container.n_fields),
        "n_sites_evaluated": len(fids),
        "period_start": str(cfg.start_dt.date()),
        "period_end": str(cfg.end_dt.date()),
        "par_csv": par_csv,
        "openet_source": "volk",
        "excluded_sites": sorted(ev.EXCLUDED_SITES),
    }
    with open(os.path.join(cat6_dir, "evaluation_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Cat6: evaluation products -> {cat6_dir}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-tag", required=True, help="e.g. run21")
    ap.add_argument("--container", default=None, help="Override container path")
    ap.add_argument("--par-csv", default=None, help="Override posterior par.csv")
    ap.add_argument("--log", default=None, help="calibration stdout log to archive")
    ap.add_argument(
        "--config",
        default=None,
        help="Path to the project TOML the run was calibrated with (default: "
        "5_Flux_Ensemble.toml). REQUIRED for any run using a variant config — "
        "Cat 1 snapshots this file, and Cat 6 rebuilds swim_input from it, so "
        "the wrong config silently evaluates posterior parameters under the "
        "wrong physics.",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated category numbers to run (e.g. 2,5). Default: all.",
    )
    args = ap.parse_args()

    cfg = ev.load_config(args.config)
    project = cfg.project_name
    results_dir = os.path.join(cfg.project_ws, "results", args.results_tag)
    archive = os.path.join(results_dir, "archive")
    os.makedirs(archive, exist_ok=True)

    container_path = args.container or os.path.join(cfg.data_dir, f"{project}.swim")
    par_csv = args.par_csv or os.path.join(results_dir, f"{project}.{NOPTMAX}.par.csv")
    log_path = args.log or f"/data/ssd1/swim/5_Flux_Ensemble/nohup_{args.results_tag}_calibrate.out"
    pst_path = os.path.join(archive, "3_problem_definition", f"{project}.pst")
    weight_audit = os.path.join(results_dir, "etf_weight_audit.csv")

    only = set(args.only.split(",")) if args.only else None

    def _run(num):
        return only is None or num in only

    print(f"Archiving {args.results_tag} -> {archive}")
    print(f"  container: {container_path}")
    print(f"  par_csv:   {par_csv}")

    container = SwimContainer.open(container_path, mode="r")
    gaps = []
    try:
        if _run("1"):
            try:
                cat1_enrich(
                    cfg,
                    container,
                    container_path,
                    os.path.join(archive, "1_provenance"),
                    log_path,
                    gaps,
                )
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"Cat1 FAILED: {exc}")
                traceback.print_exc()
        if _run("2"):
            try:
                cat2_input_audit(cfg, container, os.path.join(archive, "2_input_audit"), gaps)
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"Cat2 FAILED: {exc}")
                traceback.print_exc()
        if _run("3"):
            if os.path.exists(pst_path):
                try:
                    cat3_problem_definition(
                        cfg,
                        container,
                        os.path.join(archive, "3_problem_definition"),
                        pst_path,
                        weight_audit,
                        gaps,
                    )
                except Exception as exc:  # noqa: BLE001
                    gaps.append(f"Cat3 FAILED: {exc}")
                    traceback.print_exc()
            else:
                gaps.append(f"Cat3: {pst_path} missing (PEST .pst not archived)")
        if _run("5"):
            try:
                cat5_posterior(
                    cfg,
                    container,
                    os.path.join(archive, "5_posterior_summaries"),
                    par_csv,
                    os.path.join(archive, "3_problem_definition"),
                    gaps,
                )
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"Cat5 FAILED: {exc}")
                traceback.print_exc()
        if _run("6"):
            try:
                cat6_evaluation(
                    cfg,
                    container,
                    os.path.join(archive, "6_evaluation"),
                    par_csv,
                    os.path.join(archive, "3_problem_definition"),
                    gaps,
                )
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"Cat6 FAILED: {exc}")
                traceback.print_exc()
    finally:
        container.close()

    # GAPS.md
    gaps.append(
        "Cat7 (figure audit): deferred until manuscript figures/tables are "
        "defined; no fig_*_audit tables produced."
    )
    with open(os.path.join(archive, "GAPS.md"), "w") as fh:
        fh.write(f"# Archive gaps for {args.results_tag}\n\n")
        fh.write(f"Generated: {_utcnow()}\n\n")
        if gaps:
            for g in gaps:
                fh.write(f"- {g}\n")
        else:
            fh.write("No gaps recorded.\n")

    print(f"\nDone. {len(gaps)} gap note(s) -> {os.path.join(archive, 'GAPS.md')}")


if __name__ == "__main__":
    main()

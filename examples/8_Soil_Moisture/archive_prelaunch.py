"""Pre-launch RUN_POLICY capture for the Example 8 (SCAN soil-moisture) calibration.

Writes Category 1 (provenance) and Category 2 (input audit) into
``results/<run_name>/archive/{1_provenance,2_input_audit}/`` *before* calibration
launches, per ``examples/RUN_POLICY.md``. Categories 3-4 are produced by
``calibrate.py`` (which reuses Example 5's ``_archive_pest_outputs``); Categories
5-7 are filled post-hoc after the run.

Self-contained (reads the Zarr container directly) so it can run as the first step
of ``commands.sh`` with no calibration dependencies. Adapted from the Example 7
prelaunch archiver: Category 1 is identical (fully generic), Category 2 uses the
SCAN container's properties (state / glc10 in place of Ex7's crop / basin). The
input-audit gate is the CLAUDE.md container-validation check: every site must have
non-null ensemble ETf, seasonal NDVI, and no all-NaN meteorology, or the script
HALTS before calibration.

In-situ soil moisture is validation-only and is NOT part of this audit or the
calibration targets — only the satellite ETf ensemble is.

    uv run python examples/8_Soil_Moisture/archive_prelaunch.py \
        --config    examples/8_Soil_Moisture/8_Soil_Moisture.toml \
        --container /data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8cal.swim \
        --run-name  e8cal \
        --command   "uv run python examples/8_Soil_Moisture/calibrate.py ..." \
        --output-root /data/ssd1/swim/8_Soil_Moisture/pestrun
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]

# Active calibration target = the computed ensemble; its members are the streams
# that form it (TOML etf_ensemble_members). Everything else is diagnostic.
ETF_MODELS = ["ssebop", "sims", "eemetric", "geesebal", "ptjpl", "disalexi", "ensemble"]
ACTIVE_TARGET = {"ssebop", "sims", "eemetric", "geesebal", "ptjpl", "disalexi", "ensemble"}
MET_EVAL = ["eto", "eto_corr", "etr", "etr_corr", "prcp", "srad", "tmax", "tmin", "u2", "ea"]
# Arrays whose undetected change would silently invalidate results -> content hash.
MANDATORY_HASH_PREFIXES = (
    "remote_sensing/etf/",
    "remote_sensing/ndvi/",
    "derived/merged_ndvi/",
    "meteorology/gridmet/",
    "properties/",
)


def _run(cmd):
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:  # noqa: BLE001
        return f"<error running {' '.join(cmd)}: {exc}>\n"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strs(arr):
    return [str(x) for x in np.asarray(arr[:]).tolist()]


def _nn(arr):
    """Non-null count per field (axis 0 = time)."""
    return np.sum(~np.isnan(np.asarray(arr[:])), axis=0)


def capture_provenance(prov, container, run_name, command, ctx_meta):
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "command.txt").write_text(command.rstrip() + "\n")
    (prov / "git_sha.txt").write_text(_run(["git", "rev-parse", "HEAD"]))
    (prov / "git_status.txt").write_text(_run(["git", "status", "--short"]))
    (prov / "git_diff.patch").write_text(_run(["git", "diff"]))
    (prov / "git_diff_cached.patch").write_text(_run(["git", "diff", "--cached"]))
    (prov / "container_path.txt").write_text(str(container) + "\n")

    cfg_src = Path(ctx_meta["config_path"])
    (prov / "config.toml").write_text(cfg_src.read_text())
    (prov / "config_sha256.txt").write_text(_sha256_file(cfg_src) + "\n")

    lock = REPO / "uv.lock"
    if lock.exists():
        (prov / "uv_lock_sha256.txt").write_text(_sha256_file(lock) + "\n")

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pestpp_ies_version": _run(["pestpp-ies", "--version"]).strip() or "<not found>",
        "uv_pip_freeze": _run(["uv", "run", "pip", "freeze"]),
    }
    for lib in ("numpy", "scipy", "pandas", "geopandas", "rasterio", "pyproj", "shapely", "zarr"):
        try:
            env[f"{lib}_version"] = __import__(lib).__version__
        except Exception:  # noqa: BLE001
            env[f"{lib}_version"] = "<unavailable>"
    (prov / "environment.json").write_text(json.dumps(env, indent=2))

    meta = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        **ctx_meta,
    }
    (prov / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # Container manifest: every array's path/shape/dtype/non-null + .zattrs hash;
    # content SHA-256 for figure-critical inputs.
    root = zarr.open_group(str(container), mode="r")
    manifest = {}
    for name, node in sorted(root.members(max_depth=None)):
        if not isinstance(node, zarr.Array):
            continue
        x = np.asarray(node[:])
        if np.issubdtype(x.dtype, np.floating):
            nn = int(np.sum(~np.isnan(x)))
        else:
            nn = int(x.size)
        entry = {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "non_null": nn,
            "zattrs_sha256": hashlib.sha256(
                json.dumps(dict(node.attrs), sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        if name.startswith(MANDATORY_HASH_PREFIXES):
            entry["content_sha256"] = hashlib.sha256(np.ascontiguousarray(x)).hexdigest()
        manifest[name] = entry
    (prov / "container_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Cat 1: wrote {len(list(prov.iterdir()))} provenance artifacts")


def capture_input_audit(audit, container, output_root):
    import pandas as pd

    audit.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(container), mode="r")
    uid = _strs(root["geometry/uid"])
    state = _strs(root["geometry/properties/state"])
    glc10 = _strs(root["geometry/properties/glc10"])
    irrig_label = _strs(
        root["geometry/properties/irrig"]
    )  # provisional climate proxy, not a target
    days = np.asarray(root["time/daily"][:])
    n = len(uid)

    etf = {m: _nn(root[f"remote_sensing/etf/landsat/{m}/no_mask"]) for m in ETF_MODELS}
    ndvi_m = np.asarray(root["derived/merged_ndvi/no_mask"][:])
    ndvi_nn = np.sum(~np.isnan(ndvi_m), axis=0)
    met = {v: _nn(root[f"meteorology/gridmet/{v}"]) for v in MET_EVAL}

    irr_json = _strs(root["derived/dynamics/irr_data"])

    def ever_irr(js):
        d = json.loads(js)
        return any(
            isinstance(v, dict) and (v.get("irrigated") or len(v.get("irr_doys", [])) > 0)
            for v in d.values()
        )

    # irrigation status is the internal water-balance algorithm's determination,
    # NOT the provisional climate label (irrig_label) carried for reference only.
    irrig = np.array([ever_irr(j) for j in irr_json])

    # etf_capture_counts.csv (per site, per model, per mask, is_active_target)
    rows = []
    for i, u in enumerate(uid):
        for m in ETF_MODELS:
            rows.append(
                {
                    "site": u,
                    "sensor": "landsat",
                    "model": m,
                    "mask_mode": "no_mask",
                    "non_null": int(etf[m][i]),
                    "is_active_target": m in ACTIVE_TARGET,
                }
            )
    pd.DataFrame(rows).to_csv(audit / "etf_capture_counts.csv", index=False)

    # ndvi_coverage.csv (per site: non-null, first/last obs date, seasonal flag)
    nd_rows = []
    for i, u in enumerate(uid):
        obs = ~np.isnan(ndvi_m[:, i])
        dts = days[obs]
        months = sorted({int(str(np.datetime64(d, "M")).split("-")[1]) for d in dts})
        nd_rows.append(
            {
                "site": u,
                "non_null": int(ndvi_nn[i]),
                "first_date": str(np.datetime64(dts.min(), "D")) if dts.size else "",
                "last_date": str(np.datetime64(dts.max(), "D")) if dts.size else "",
                "months_covered": len(months),
                "seasonal_ok": len(months) >= 6,
            }
        )
    pd.DataFrame(nd_rows).to_csv(audit / "ndvi_coverage.csv", index=False)

    # met_completeness.csv (per site non-null for the eval vars)
    met_cols = ["eto", "eto_corr", "srad", "tmax", "tmin", "prcp"]
    pd.DataFrame(
        [{"site": uid[i], **{v: int(met[v][i]) for v in met_cols}} for i in range(n)]
    ).to_csv(audit / "met_completeness.csv", index=False)

    # per_field_audit.csv (compact one-row-per-field roll-up)
    pd.DataFrame(
        [
            {
                "site": uid[i],
                "state": state[i],
                "glc10": glc10[i],
                "irrig_label": irrig_label[i],
                "irrigated_algo": bool(irrig[i]),
                "etf_ensemble_nn": int(etf["ensemble"][i]),
                "etf_min_member_nn": int(min(etf[m][i] for m in ETF_MODELS if m != "ensemble")),
                "ndvi_nn": int(ndvi_nn[i]),
                "met_prcp_nn": int(met["prcp"][i]),
                "met_eto_nn": int(met["eto"][i]),
            }
            for i in range(n)
        ]
    ).to_csv(audit / "per_field_audit.csv", index=False)

    # exclusions (from prep's manifest/excluded_fids.json — none expected)
    excluded = []
    exc_json = Path(output_root) / "excluded_fids.json"
    if exc_json.exists():
        excluded = json.loads(exc_json.read_text()).get("fids", [])
    pd.DataFrame([{"site": s, "reason": "zero RS coverage"} for s in excluded]).to_csv(
        audit / "calibration_sites_excluded.csv", index=False
    )

    # gate: PASS iff every field has ensemble ETf, seasonal NDVI, and no all-NaN met
    ens0 = int((etf["ensemble"] == 0).sum())
    ndvi_bad = int((ndvi_nn == 0).sum())
    met_bad = int(sum((met[v] == 0).any() for v in MET_EVAL))
    gate = "PASS" if (ens0 == 0 and ndvi_bad == 0 and met_bad == 0) else "HALT"
    summary = {
        "n_fields": n,
        "n_days": int(len(days)),
        "etf_ensemble_zero_sites": ens0,
        "ndvi_zero_sites": ndvi_bad,
        "met_vars_with_allnan_site": met_bad,
        "irrigated": int(irrig.sum()),
        "non_irrigated": int((~irrig).sum()),
        "excluded": len(excluded),
        "gate": gate,
    }
    (audit / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    (audit / "gate.txt").write_text(
        f"GATE={gate}  fields={n}  ens_zero={ens0}  ndvi_zero={ndvi_bad}  "
        f"met_allnan_vars={met_bad}  irrigated={int(irrig.sum())}/{n}\n"
    )
    print(f"  Cat 2: gate={gate}  (ens_zero={ens0}, ndvi_zero={ndvi_bad}, met_allnan={met_bad})")
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--command", required=True, help="Exact launch command being archived")
    p.add_argument("--output-root", required=True, help="pest_run_dir (for exclusions/manifest)")
    p.add_argument("--results-root", default="/data/ssd1/swim/8_Soil_Moisture/results")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--reals", type=int, default=200)
    p.add_argument("--noptmax", type=int, default=3)
    args = p.parse_args()

    archive = Path(args.results_root) / args.run_name / "archive"
    ctx_meta = {
        "config_path": os.path.abspath(args.config),
        "container_path": os.path.abspath(args.container),
        "output_root": os.path.abspath(args.output_root),
        "workers": args.workers,
        "realizations": args.reals,
        "noptmax": args.noptmax,
    }
    print(f"Pre-launch archive -> {archive}")
    capture_provenance(
        archive / "1_provenance", args.container, args.run_name, args.command, ctx_meta
    )
    gate = capture_input_audit(archive / "2_input_audit", args.container, args.output_root)
    if gate != "PASS":
        raise SystemExit(f"Input-audit gate = {gate}; refusing to proceed to calibration.")
    print("Pre-launch archive complete; gate PASS.")


if __name__ == "__main__":
    main()

"""Pre-launch RUN_POLICY capture for the E2 GrassBasis (re-footed) calibration, plus launch-time
hash verification.

``capture`` writes Categories 1 (provenance), 2 (input audit) and 3 (problem definition) into
``results/<run_name>/archive/{1_provenance,2_input_audit,3_problem_definition}/`` *before* PEST++
launches (examples/RUN_POLICY.md). Categories 4-7 are produced after the run by the batch runner
and the evaluation step. Category 3 is copied from the batches built by
``batch_runner --action build-all`` and decoded with the Phase 9 audit rows
(``objective_weight_rows.csv``) into ``observation_metadata.csv``.

``verify`` recomputes the config, uv.lock, control-file and mandatory container-array hashes and
compares them with the archive; a non-zero exit blocks the launch (plan §16 "config, container and
source hashes are checked again at launch").

    uv run python examples/6_Flux_International/e2_refooting/phase9_archive_prelaunch.py capture \
        --command "<exact launch command>"
    uv run python examples/6_Flux_International/e2_refooting/phase9_archive_prelaunch.py verify
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EX6 = REPO / "examples" / "6_Flux_International"
QA_ROOT = Path("/data/ssd1/swim/6_Flux_International/data/e2_etf_refooting")
DEFAULT_CONFIG = EX6 / "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml"
DEFAULT_RUN_NAME = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
DEFAULT_RESULTS_ROOT = "/data/ssd1/swim/6_Flux_International/results"

ACTIVE_TARGET_MODELS = ("ssebop", "ptjpl")
# Arrays whose undetected change would silently invalidate results -> content SHA-256
# (RUN_POLICY Category 1 "mandatory content checksums"): ETf targets, NDVI, meteorology used by
# the model and the exports, properties, snow, dynamics, geometry.
MANDATORY_HASH_PREFIXES = (
    "remote_sensing/etf/",
    "remote_sensing/ndvi/",
    "derived/",
    "meteorology/",
    "properties/",
    "snow/",
    "geometry/",
)
# QA evidence produced by the re-footing phases, copied verbatim into 2_input_audit/e2_refooting_qa
QA_EVIDENCE = (
    "container_health.json",
    "etf_capture_counts.csv",
    "ndvi_coverage.csv",
    "met_completeness.csv",
    "irrigation_classifier_transition.csv",
    "daily_basis_gate_summary.json",
    "daily_basis_gate_by_site.csv",
    "objective_weight_audit.json",
    "objective_weight_losses.csv",
    "objective_weight_counts.csv",
    "ssebop_conversion_summary.json",
    "ssebop_native_consolidation_summary.json",
    "le07_delivery_summary.json",
    "sentinel_glob_order_asbuilt_20260907.json",
    "sentinel_family_overlap_20260907.csv",
    "ingestor_sorted_glob_not_applied_20260907.patch",
)
PROBLEM_FILES = ("*.pst", "params.csv", "loc.mat", "localizer_summary.json", "weight_audit.csv")
WEIGHT_FORMULA = (
    "target_over_(member_sd_ddof1+spread_floor)_if_members>=min_members_and_eto>=eto_floor_else_0"
)


# ---------------------------------------------------------------------------
# Helpers (unit-tested)
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_content_sha256(x: np.ndarray) -> str:
    """Content hash of an array's bytes (dtype/shape-stable through ascontiguousarray)."""
    x = np.ascontiguousarray(x)
    if x.dtype.kind in ("O", "U", "S"):
        payload = json.dumps([str(v) for v in x.ravel().tolist()]).encode()
    else:
        payload = x.tobytes()
    return hashlib.sha256(payload).hexdigest()


def container_manifest(root) -> dict:
    """Per-array path/shape/dtype/non-null + .zattrs hash; content hash for mandatory arrays."""
    import zarr

    manifest = {}
    for name, node in sorted(root.members(max_depth=None)):
        if not isinstance(node, zarr.Array):
            continue
        x = np.asarray(node[:])
        nn = int(np.sum(~np.isnan(x))) if np.issubdtype(x.dtype, np.floating) else int(x.size)
        entry = {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "non_null": nn,
            "zattrs_sha256": hashlib.sha256(
                json.dumps(dict(node.attrs), sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        if name.startswith(MANDATORY_HASH_PREFIXES):
            entry["content_sha256"] = array_content_sha256(x)
        manifest[name] = entry
    return manifest


def compare_hashes(recorded: dict, current: dict) -> list[str]:
    """Names whose hash differs or is missing on either side."""
    problems = []
    for k, v in recorded.items():
        if k not in current:
            problems.append(f"missing now: {k}")
        elif current[k] != v:
            problems.append(f"changed: {k}")
    for k in current:
        if k not in recorded:
            problems.append(f"new (not recorded): {k}")
    return problems


def observation_metadata(rows: pd.DataFrame, members: list[str], resolved: dict) -> pd.DataFrame:
    """RUN_POLICY Category 3 decoded observation table from the Phase 9 audit rows."""
    dates = pd.to_datetime(rows["date"])
    start = pd.Timestamp(resolved["start_date"])
    day_index = (dates - start).dt.days.astype(int)
    fid_lower = rows["fid"].str.lower()
    member_vals = [
        json.dumps([None if not np.isfinite(v) else round(float(v), 6) for v in vals])
        for vals in rows[members].to_numpy(dtype=float)
    ]
    return pd.DataFrame(
        {
            "obsnme": [
                f"oname:obs_etf_{f}_otype:arr_i:{i}_j:0" for f, i in zip(fid_lower, day_index)
            ],
            "site": rows["fid"],
            "date": dates.dt.strftime("%Y-%m-%d"),
            "sensor": resolved["etf_target_instrument"],
            "model": resolved["etf_target_model"],
            "mask_mode": resolved["mask"],
            "target_etf": rows["target"],
            "raw_member_values": member_vals,
            "member_count": rows["member_count"],
            "ensemble_std": rows["member_std"],
            "mad_included": "not_applied",
            "eto_correction_factor": 1.0,
            "daily_eto": rows["eto"],
            "eto_floor_excluded": rows["eto_floor_excluded"],
            "raw_weight": rows["weight"],
            "final_weight": rows["weight_pst"],
            "standard_deviation": rows["sd_pst"],
            "weight_formula": WEIGHT_FORMULA,
            "batch": rows["batch"],
        }
    )


def _run(cmd, cwd=REPO):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300).stdout
    except Exception as exc:  # noqa: BLE001
        return f"<error running {' '.join(cmd)}: {exc}>\n"


def _strs(arr):
    return [str(x) for x in np.asarray(arr[:]).tolist()]


# ---------------------------------------------------------------------------
# Category 1
# ---------------------------------------------------------------------------


def capture_provenance(prov: Path, cfg, config_path: Path, run_name, command, params, root):
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "command.txt").write_text(command.rstrip() + "\n")
    (prov / "git_sha.txt").write_text(_run(["git", "rev-parse", "HEAD"]))
    (prov / "git_status.txt").write_text(_run(["git", "status", "--short"]))
    (prov / "git_diff.patch").write_text(_run(["git", "diff"]))
    (prov / "git_diff_cached.patch").write_text(_run(["git", "diff", "--cached"]))
    (prov / "container_path.txt").write_text(str(cfg.container_path) + "\n")
    (prov / "config.toml").write_text(config_path.read_text())
    (prov / "config_sha256.txt").write_text(sha256_file(config_path) + "\n")
    lock = REPO / "uv.lock"
    (prov / "uv_lock_sha256.txt").write_text(sha256_file(lock) + "\n")

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pestpp_ies_version": _run(["pestpp-ies", "--version"]).strip() or "<not found>",
        "uv_pip_freeze": _run(["uv", "run", "pip", "freeze"]),
        "gdal_version": _run(["gdalinfo", "--version"]).strip() or "<not found>",
    }
    for lib in (
        "numpy",
        "scipy",
        "pandas",
        "geopandas",
        "rasterio",
        "pyproj",
        "shapely",
        "zarr",
        "pyemu",
        "swimrs",
    ):
        try:
            env[f"{lib}_version"] = __import__(lib).__version__
        except Exception:  # noqa: BLE001
            env[f"{lib}_version"] = "<unavailable>"
    try:
        import pyproj

        env["proj_version"] = pyproj.proj_version_str
    except Exception:  # noqa: BLE001
        env["proj_version"] = "<unavailable>"
    try:
        import shapely

        env["geos_version"] = shapely.geos_version_string
    except Exception:  # noqa: BLE001
        env["geos_version"] = "<unavailable>"
    (prov / "environment.json").write_text(json.dumps(env, indent=2))

    meta = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "container_path": cfg.container_path,
        "pest_run_dir": cfg.pest_run_dir,
        "etf_target_model": cfg.etf_target_model,
        "etf_ensemble_members": list(cfg.etf_ensemble_members),
        "etf_weighting_spread_floor": getattr(cfg, "etf_weighting_spread_floor", None),
        "etf_weighting_min_members": getattr(cfg, "etf_weighting_min_members", None),
        "etf_weighting_eto_floor": getattr(cfg, "etf_weighting_eto_floor", None),
        "swe_weighting": {
            "sd_frac": getattr(cfg, "swe_weighting_sd_frac", None),
            "sd_floor": getattr(cfg, "swe_weighting_sd_floor", None),
            "phi_share": getattr(cfg, "swe_weighting_phi_share", None),
        },
        "prior_regularization_fraction": getattr(cfg, "prior_regularization_fraction", None),
        "date_range": [cfg.start_dt.date().isoformat(), cfg.end_dt.date().isoformat()],
        **params,
    }
    (prov / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    (prov / "container_manifest.json").write_text(json.dumps(container_manifest(root), indent=2))
    for name in ("batch_manifest.csv", "excluded_fids.json"):
        src = Path(cfg.pest_run_dir) / name
        if src.exists():
            shutil.copy2(src, prov / name)
    print(f"  Cat 1: wrote {len(list(prov.iterdir()))} provenance artifacts")


# ---------------------------------------------------------------------------
# Category 2
# ---------------------------------------------------------------------------


def capture_input_audit(audit: Path, cfg, root, qa_root: Path) -> str:
    audit.mkdir(parents=True, exist_ok=True)
    uid = _strs(root["geometry/uid"])
    days = pd.to_datetime(np.asarray(root["time/daily"][:]))
    n = len(uid)
    inst = cfg.etf_target_instrument or "landsat"
    members = list(cfg.etf_ensemble_members)

    etf_rows, etf_nn = [], {}
    for model in ACTIVE_TARGET_MODELS:
        arr = np.asarray(root[f"remote_sensing/etf/{inst}/{model}/no_mask"][:])
        etf_nn[model] = np.sum(~np.isnan(arr), axis=0)
        for j, u in enumerate(uid):
            valid = ~np.isnan(arr[:, j])
            etf_rows.append(
                {
                    "site": u,
                    "sensor": inst,
                    "model": model,
                    "mask_mode": "no_mask",
                    "non_null": int(valid.sum()),
                    "first_date": days[valid].min().date().isoformat() if valid.any() else "",
                    "last_date": days[valid].max().date().isoformat() if valid.any() else "",
                    "is_active_target": model in members,
                }
            )
    pd.DataFrame(etf_rows).to_csv(audit / "etf_capture_counts.csv", index=False)

    nd_rows = []
    for path, label in (
        ("remote_sensing/ndvi/landsat/no_mask", "landsat"),
        ("remote_sensing/ndvi/sentinel/no_mask", "sentinel"),
        ("derived/merged_ndvi/no_mask", "merged"),
    ):
        arr = np.asarray(root[path][:])
        for j, u in enumerate(uid):
            valid = ~np.isnan(arr[:, j])
            months = set(days[valid].month) if valid.any() else set()
            nd_rows.append(
                {
                    "site": u,
                    "source": label,
                    "non_null": int(valid.sum()),
                    "first_date": days[valid].min().date().isoformat() if valid.any() else "",
                    "last_date": days[valid].max().date().isoformat() if valid.any() else "",
                    "months_covered": len(months),
                    "seasonal_ok": len(months & {4, 5, 6, 7, 8, 9, 10}) >= 5,
                }
            )
    ndvi = pd.DataFrame(nd_rows)
    ndvi.to_csv(audit / "ndvi_coverage.csv", index=False)
    merged_nn = ndvi[ndvi.source == "merged"].set_index("site").loc[uid, "non_null"].to_numpy()

    met_vars = sorted(root[f"meteorology/{cfg.met_source}"].array_keys())
    met = {
        v: np.sum(~np.isnan(np.asarray(root[f"meteorology/{cfg.met_source}/{v}"][:])), axis=0)
        for v in met_vars
    }
    pd.DataFrame(
        [{"site": uid[i], **{v: int(met[v][i]) for v in met_vars}} for i in range(n)]
    ).to_csv(audit / "met_completeness.csv", index=False)

    excluded = []
    exc_json = Path(cfg.pest_run_dir) / "excluded_fids.json"
    if exc_json.exists():
        excluded = json.loads(exc_json.read_text()).get("fids", [])
    pd.DataFrame(
        [{"site": s, "reason": "zero RS coverage (--exclude-uncovered)"} for s in excluded],
        columns=["site", "reason"],
    ).to_csv(audit / "calibration_sites_excluded.csv", index=False)
    pd.DataFrame(columns=["site", "reason"]).to_csv(
        audit / "evaluation_sites_excluded.csv", index=False
    )

    qa_dir = audit / "e2_refooting_qa"
    qa_dir.mkdir(exist_ok=True)
    copied = []
    for name in QA_EVIDENCE:
        src = qa_root / name
        if src.exists():
            shutil.copy2(src, qa_dir / name)
            copied.append(name)
    health_src = qa_root / "container_health.json"
    if health_src.exists():
        shutil.copy2(health_src, audit / "container_health.json")

    target_zero = int(sum(int((etf_nn[m] == 0).sum()) for m in members))
    ndvi_zero = int((merged_nn == 0).sum())
    met_allnan = int(sum(int((met[v] == 0).any()) for v in met_vars))
    gate = "PASS" if (target_zero == 0 and ndvi_zero == 0 and met_allnan == 0) else "HALT"
    summary = {
        "n_fields": n,
        "n_days": int(len(days)),
        "active_target_zero_site_streams": target_zero,
        "ndvi_zero_sites": ndvi_zero,
        "met_vars_with_allnan_site": met_allnan,
        "excluded": excluded,
        "qa_evidence_copied": copied,
        "gate": gate,
    }
    (audit / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  Cat 2: gate={gate} (target_zero={target_zero}, ndvi_zero={ndvi_zero}, met_allnan={met_allnan}); QA files copied {len(copied)}"
    )
    return gate


# ---------------------------------------------------------------------------
# Category 3
# ---------------------------------------------------------------------------


def capture_problem_definition(prob: Path, cfg, qa_root: Path) -> dict:
    from pyemu import Pst

    prob.mkdir(parents=True, exist_ok=True)
    rows_path = qa_root / "objective_weight_rows.csv"
    audit_json = qa_root / "objective_weight_audit.json"
    if not rows_path.exists() or not audit_json.exists():
        raise FileNotFoundError(
            "run phase9_objective_audit.py first (objective_weight_rows.csv / objective_weight_audit.json)"
        )
    audit = json.loads(audit_json.read_text())
    if not audit.get("pass"):
        raise RuntimeError(
            "objective_weight_audit.json does not pass; refusing to stage the problem definition"
        )
    rows = pd.read_csv(rows_path)
    resolved = dict(audit["resolved"])
    resolved["start_date"] = cfg.start_dt.date().isoformat()
    members = list(cfg.etf_ensemble_members)

    hashes = {}
    for batch_dir in sorted(p for p in Path(cfg.pest_run_dir).glob("batch_*") if p.is_dir()):
        dest = prob / batch_dir.name
        dest.mkdir(exist_ok=True)
        pest_dir = batch_dir / "pest"
        for pattern in PROBLEM_FILES:
            for src in list(pest_dir.glob(pattern)) + list(batch_dir.glob(pattern)):
                shutil.copy2(src, dest / src.name)
                hashes[str(src)] = sha256_file(src)
        pst_files = sorted(pest_dir.glob("*.pst"))
        if not pst_files:
            raise FileNotFoundError(f"no .pst in {pest_dir}")
        pst = Pst(str(pst_files[0]))
        obs = pst.observation_data[["obsnme", "obsval", "weight", "obgnme", "standard_deviation"]]
        obs.to_csv(dest / "observation_table.csv", index=False)
        par = pst.parameter_data
        par[["parnme", "parlbnd", "parubnd", "parval1", "partrans", "pargp"]].rename(
            columns={
                "parlbnd": "lower_bound",
                "parubnd": "upper_bound",
                "parval1": "initial_value",
                "partrans": "transform",
                "pargp": "group",
            }
        ).assign(tied_or_fixed=par.partrans.isin(["tied", "fixed"])).to_csv(
            dest / "parameter_bounds.csv", index=False
        )
        if pst.prior_information is not None and len(pst.prior_information):
            pst.prior_information.to_csv(dest / "prior_info.csv", index=False)
        meta = observation_metadata(rows[rows.batch == batch_dir.name], members, resolved)
        meta.to_csv(dest / "observation_metadata.csv", index=False)
        (dest / "pestpp_options.json").write_text(
            json.dumps(
                {
                    "noptmax": int(pst.control_data.noptmax),
                    **{k: str(v) for k, v in pst.pestpp_options.items()},
                },
                indent=2,
            )
        )
    (prob / "problem_definition_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(
        f"  Cat 3: staged {len(hashes)} problem-definition files for {sum(1 for p in prob.glob('batch_*') if p.is_dir())} batches"
    )
    return hashes


# ---------------------------------------------------------------------------
# Verify (launch time)
# ---------------------------------------------------------------------------


def verify(archive: Path, cfg, config_path: Path, root) -> list[str]:
    prov, prob = archive / "1_provenance", archive / "3_problem_definition"
    problems = []
    if (prov / "config_sha256.txt").read_text().strip() != sha256_file(config_path):
        problems.append("config.toml hash changed since capture")
    if (prov / "uv_lock_sha256.txt").read_text().strip() != sha256_file(REPO / "uv.lock"):
        problems.append("uv.lock hash changed since capture")
    if (prov / "container_path.txt").read_text().strip() != str(cfg.container_path):
        problems.append("container path differs from capture")
    recorded = json.loads((prov / "container_manifest.json").read_text())
    current = container_manifest(root)
    rec_h = {k: v["content_sha256"] for k, v in recorded.items() if "content_sha256" in v}
    cur_h = {k: v["content_sha256"] for k, v in current.items() if "content_sha256" in v}
    problems += [f"container {p}" for p in compare_hashes(rec_h, cur_h)]
    rec_shape = {k: (v["shape"], v["non_null"]) for k, v in recorded.items()}
    cur_shape = {k: (v["shape"], v["non_null"]) for k, v in current.items()}
    problems += [f"container shape/non-null {p}" for p in compare_hashes(rec_shape, cur_shape)]
    rec_files = json.loads((prob / "problem_definition_sha256.json").read_text())
    cur_files = {p: sha256_file(p) if os.path.exists(p) else "<missing>" for p in rec_files}
    problems += [f"problem definition {p}" for p in compare_hashes(rec_files, cur_files)]
    git_sha_now = _run(["git", "rev-parse", "HEAD"]).strip()
    if (prov / "git_sha.txt").read_text().strip() != git_sha_now:
        problems.append("git HEAD moved since capture")
    if (prov / "git_diff.patch").read_text() != _run(["git", "diff"]):
        problems.append("working-tree diff changed since capture")
    return problems


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["capture", "verify"])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--qa-root", default=str(QA_ROOT))
    p.add_argument("--command", default=None, help="Exact launch command being archived (capture)")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--reals", type=int, default=200)
    p.add_argument("--noptmax", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=50)
    args = p.parse_args()

    import zarr

    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(args.config, calibrate=True)
    config_path = Path(args.config).resolve()
    root = zarr.open_group(cfg.container_path, mode="r")
    archive = Path(args.results_root) / args.run_name / "archive"

    if args.mode == "capture":
        if not args.command:
            p.error("--command is required for capture")
        params = {
            "workers": args.workers,
            "realizations": args.reals,
            "noptmax": args.noptmax,
            "batch_size": args.batch_size,
        }
        print(f"Pre-launch archive -> {archive}")
        capture_provenance(
            archive / "1_provenance", cfg, config_path, args.run_name, args.command, params, root
        )
        gate = capture_input_audit(archive / "2_input_audit", cfg, root, Path(args.qa_root))
        capture_problem_definition(archive / "3_problem_definition", cfg, Path(args.qa_root))
        if gate != "PASS":
            raise SystemExit(f"Input-audit gate = {gate}; refusing to proceed to calibration.")
        print("Pre-launch archive complete; gate PASS.")
    else:
        problems = verify(archive, cfg, config_path, root)
        if problems:
            for q in problems:
                print("HASH MISMATCH:", q)
            raise SystemExit(1)
        print(
            f"verify OK: config, uv.lock, git state, container arrays and problem-definition files match {archive}"
        )


if __name__ == "__main__":
    main()

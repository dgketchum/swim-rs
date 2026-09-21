"""Phase 0 of the E2 ETf re-footing: freeze and verify the baseline.

Read-only against every baseline artifact. Writes only into the working QA root:

    baseline_inventory.json   sources, cohorts, repo state, proposed-path reservations
    baseline_hashes.sha256    sha256 of every baseline file and container target array
    baseline_objective_counts.json
                              the archived .pst objective cross-classified against member
                              availability in the container (must reproduce 16,102 / 7,270 / 1,957)

Plan: examples/6_Flux_International/notes/e2_etf_refooting_plan.md, Phase 0 / Gate G0.

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase0_freeze_baseline.py
"""

from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

import fiona
import numpy as np
import pandas as pd
import zarr

REPO = "/home/dgketchum/code/swim-rs"
E6 = os.path.join(REPO, "examples", "6_Flux_International")
DATA = "/data/ssd1/swim/6_Flux_International/data"
RESULTS = (
    "/data/ssd1/swim/6_Flux_International/results/6_Flux_International_LSEnsemble_POR_annual2yr"
)
ARCHIVE = os.path.join(RESULTS, "archive_recal20260702_classifier")
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")

BASELINE_TOML = os.path.join(E6, "6_Flux_International_LSEnsemble_POR_annual2yr.toml")
CONTAINER = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")
COHORT_66 = os.path.join(DATA, "gis", "flux_crop_pub_66_150m.shp")
COHORT_75 = os.path.join(DATA, "gis", "flux_crop_pub_75_150m.shp")
ESPA_MANIFEST = os.path.join(DATA, "remote_sensing", "espa", "espa_manifest.csv")
NHM_DIR = (
    "/data/ssd1/swim/4_Flux_Network/data/remote_sensing/landsat/extracts/ssebop_nhm_etf/no_mask"
)

CONTAINER_ARRAYS = [
    "time/daily",
    "geometry/uid",
    "remote_sensing/etf/landsat/ssebop/no_mask",
    "remote_sensing/etf/landsat/ptjpl/no_mask",
    "meteorology/era5/eto",
]

ARCHIVE_FILES = [
    "1_provenance/config.toml",
    "1_provenance/config_sha256.txt",
    "1_provenance/git_sha.txt",
    "1_provenance/container_path.txt",
    "1_provenance/run_metadata.json",
    "6_evaluation/evaluation_metrics.csv",
    "6_evaluation/evaluation_monthly_metrics.csv",
    "6_evaluation/evaluation_summary.csv",
    "6_evaluation/evaluation_sites_excluded.csv",
]
ARCHIVE_BATCH_FILES = [
    "6_Flux_International.pst",
    "6_flux_international.obs_data.csv",
    "6_Flux_International.3.par.csv",
    "6_Flux_International.phi.meas.csv",
]

# Reserved corrected-artifact paths from plan §4. None may exist yet.
PROPOSED_PATHS = [
    os.path.join(DATA, "remote_sensing", "espa", "refet_ratio_era5land"),
    os.path.join(DATA, "remote_sensing", "landsat", "extracts", "ssebop_etf_grass"),
    os.path.join(E6, "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml"),
    os.path.join(DATA, "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"),
    os.path.join(
        "/data/ssd1/swim/6_Flux_International/results",
        "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr",
    ),
]

EXPECTED_COUNTS = {"both": 16102, "ptjpl_only": 7270, "ssebop_only": 1957, "neither": 288039}

OBS_RE = re.compile(r"oname:obs_etf_(?P<fid>.+?)_otype:arr_i:(?P<i>\d+)_j:0$")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", REPO, *args], text=True).strip()


def _shapefile_ids(path: str) -> list[str]:
    with fiona.open(path) as src:
        return [f["properties"]["sid"] for f in src]


def objective_counts(root: zarr.Group) -> dict:
    """Cross-classify every archived ETf observation against member availability."""
    uid = [str(u) for u in root["geometry/uid"][:]]
    site_idx = {u.lower(): k for k, u in enumerate(uid)}
    ss = np.asarray(root["remote_sensing/etf/landsat/ssebop/no_mask"][:])
    pj = np.asarray(root["remote_sensing/etf/landsat/ptjpl/no_mask"][:])

    frames = []
    for batch in sorted(glob.glob(os.path.join(ARCHIVE, "4_pest_outputs", "batch_*"))):
        obs = pd.read_csv(os.path.join(batch, "6_flux_international.obs_data.csv"))
        obs = obs[obs["obsnme"].str.contains("obs_etf_")].copy()
        obs["batch"] = os.path.basename(batch)
        frames.append(obs)
    obs = pd.concat(frames, ignore_index=True)

    parsed = obs["obsnme"].str.extract(OBS_RE)
    if parsed.isna().any().any():
        bad = obs.loc[parsed.isna().any(axis=1), "obsnme"].head().tolist()
        raise ValueError(f"unparseable ETf observation names, e.g. {bad}")
    obs["fid"] = parsed["fid"]
    obs["i"] = parsed["i"].astype(int)
    unknown = sorted(set(obs["fid"]) - set(site_idx))
    if unknown:
        raise ValueError(f"observation fids not in container: {unknown}")
    obs["site_idx"] = obs["fid"].map(site_idx)

    has_ss = ~np.isnan(ss[obs["i"].values, obs["site_idx"].values])
    has_pj = ~np.isnan(pj[obs["i"].values, obs["site_idx"].values])
    obs["avail"] = np.select(
        [has_ss & has_pj, has_pj & ~has_ss, has_ss & ~has_pj],
        ["both", "ptjpl_only", "ssebop_only"],
        default="neither",
    )
    obs["weighted"] = obs["weight"] > 0

    table = (
        obs.groupby("avail")
        .agg(n_obs=("obsnme", "size"), n_weighted=("weighted", "sum"), sum_weight=("weight", "sum"))
        .reindex(["both", "ptjpl_only", "ssebop_only", "neither"])
    )
    weighted_per_year = (
        obs[obs["weighted"]]
        .assign(
            year=pd.to_datetime(root["time/daily"][:]).year[obs.loc[obs["weighted"], "i"].values]
        )
        .groupby("year")
        .size()
    )
    result = {
        "n_fids": int(obs["fid"].nunique()),
        "n_obs": int(len(obs)),
        "expected": EXPECTED_COUNTS,
        "observed": {k: int(v) for k, v in table["n_obs"].items()},
        "observed_weighted": {k: int(v) for k, v in table["n_weighted"].items()},
        "observed_sum_weight": {k: float(v) for k, v in table["sum_weight"].items()},
        "weighted_per_year": {int(k): int(v) for k, v in weighted_per_year.items()},
        "min_members_2_holds": bool(
            table.loc["ptjpl_only", "n_weighted"] == 0
            and table.loc["ssebop_only", "n_weighted"] == 0
        ),
    }
    result["reproduces"] = result["observed"] == EXPECTED_COUNTS
    return result


def main() -> int:
    os.makedirs(QA_ROOT, exist_ok=True)
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    hashes: dict[str, str] = {}
    for path in [BASELINE_TOML, ESPA_MANIFEST, COHORT_66, COHORT_75]:
        hashes[path] = _sha256_file(path)
    for rel in ARCHIVE_FILES:
        hashes[os.path.join(ARCHIVE, rel)] = _sha256_file(os.path.join(ARCHIVE, rel))
    for batch in sorted(glob.glob(os.path.join(ARCHIVE, "4_pest_outputs", "batch_*"))):
        for rel in ARCHIVE_BATCH_FILES:
            hashes[os.path.join(batch, rel)] = _sha256_file(os.path.join(batch, rel))

    root = zarr.open(CONTAINER, mode="r")
    array_meta = {}
    for key in CONTAINER_ARRAYS:
        arr = np.asarray(root[key][:])
        hashes[f"{CONTAINER}::{key}"] = _sha256_array(arr)
        array_meta[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if "calibration" in root:
        cal_groups = list(root["calibration"].group_keys()) + list(root["calibration"].array_keys())
    else:
        cal_groups = []

    sids_66 = _shapefile_ids(COHORT_66)
    sids_75 = _shapefile_ids(COHORT_75)
    uid = [str(u) for u in root["geometry/uid"][:]]
    nhm_sites = sorted(
        {
            m.group(1)
            for p in glob.glob(os.path.join(NHM_DIR, "*.csv"))
            if (m := re.match(r"ssebop_etf_(.+)_no_mask_\d{4}\.csv", os.path.basename(p)))
        }
    )

    counts = objective_counts(root)

    existing = [p for p in PROPOSED_PATHS if os.path.exists(p)]
    nonempty = [p for p in existing if os.path.isdir(p) and os.listdir(p)]

    inventory = {
        "generated_utc": now,
        "repo": {
            "sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_paths": int(len(_git("status", "--porcelain").splitlines())),
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "baseline": {
            "toml": BASELINE_TOML,
            "container": CONTAINER,
            "results_root": RESULTS,
            "archive": ARCHIVE,
            "toml_matches_archived_config": hashes[BASELINE_TOML]
            == hashes[os.path.join(ARCHIVE, "1_provenance/config.toml")],
            "container_arrays": array_meta,
            "container_calibration_groups": cal_groups,
            "met_source_dir": os.path.join(DATA, "meteorology", "era5_crop"),
        },
        "cohorts": {
            "publication_66": {"path": COHORT_66, "n": len(sids_66), "sids": sorted(sids_66)},
            "container_75": {
                "path": COHORT_75,
                "n": len(sids_75),
                "matches_container_uid": set(sids_75) == set(uid),
            },
            "cohort_66_subset_of_container": set(sids_66) <= set(uid),
            "nhm_sites_total": len(nhm_sites),
            "nhm_in_container": sorted(set(nhm_sites) & set(uid)),
            "nhm_in_66": sorted(set(nhm_sites) & set(sids_66)),
        },
        "denominators": {
            "archived_objective_fids": counts["n_fids"],
            "archived_objective_slots": counts["n_obs"],
            "container_slots_75": int(
                np.prod(array_meta["remote_sensing/etf/landsat/ptjpl/no_mask"]["shape"])
            ),
            "note": "66-site archived-objective counts and 75-site container counts are never combined.",
        },
        "proposed_paths": {
            "reserved": PROPOSED_PATHS,
            "already_existing": existing,
            "already_nonempty": nonempty,
        },
        "instruction_reconciliation": {
            "monthly_metric_floor": ">= 6 paired months (supplement '10 qualifying months' is a manuscript bug)",
            "espa_etm_supported": True,
            "oli_only_diagnosis": "false; not to be narrated in reader-facing text",
            "pre_repair_weight_distribution": "internal provenance, not a final weighting result",
        },
    }

    g0 = {
        "toml_matches_archive": inventory["baseline"]["toml_matches_archived_config"],
        "objective_counts_reproduce": counts["reproduces"],
        "min_members_2_holds": counts["min_members_2_holds"],
        "cohort_66_in_container": inventory["cohorts"]["cohort_66_subset_of_container"],
        "shapefile_75_matches_container": inventory["cohorts"]["container_75"][
            "matches_container_uid"
        ],
        "no_proposed_path_populated": not nonempty,
    }
    g0["pass"] = all(g0.values())
    inventory["gate_g0"] = g0

    with open(os.path.join(QA_ROOT, "baseline_inventory.json"), "w") as fh:
        json.dump(inventory, fh, indent=2)
    with open(os.path.join(QA_ROOT, "baseline_objective_counts.json"), "w") as fh:
        json.dump(counts, fh, indent=2)
    with open(os.path.join(QA_ROOT, "baseline_hashes.sha256"), "w") as fh:
        for path in sorted(hashes):
            fh.write(f"{hashes[path]}  {path}\n")

    print(
        json.dumps(
            {
                "gate_g0": g0,
                "objective": counts["observed"],
                "weighted": counts["observed_weighted"],
                "nhm_in_66": inventory["cohorts"]["nhm_in_66"],
                "nhm_in_container": inventory["cohorts"]["nhm_in_container"],
                "existing_proposed_paths": existing,
            },
            indent=2,
        )
    )
    return 0 if g0["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

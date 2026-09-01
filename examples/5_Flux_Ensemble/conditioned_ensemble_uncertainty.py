"""Conditioned-ensemble uncertainty diagnostic for Example 5 / paper Experiment E1.

On the frozen 2,131-capture / 33-site spread-error cohort, compares multi-algorithm
retrieval spread (sample std among available OpenET ETf members at a
capture) with conditioned simulation spread (dispersion among ETf
simulations from the 199 non-base final PEST++ IES realizations) as
indicators of external SWIM-RS ETf error (effective-parameter prediction
minus flux-derived ETf), and scores the central 90% parameter-conditional
envelope as an empirical coverage diagnostic.

Read-only with respect to canonical Run 22 artifacts: the only writes are
to the dedicated output directory and the results note. The forward model
runs once with effective (componentwise median) parameters through the
canonical evaluator; no calibration, no Earth Engine.

Usage:
    uv run python conditioned_ensemble_uncertainty.py
"""

import argparse
import hashlib
import json
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import evaluate as ev
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swimrs.container import SwimContainer

RUN_DIR = "/data/ssd1/swim/5_Flux_Ensemble/results/run22"
PROJECT_DIR = Path(__file__).resolve().parent

# Canonical inputs (plan section 3). No automatic result discovery.
CANONICAL = {
    "config": str(PROJECT_DIR / "5_Flux_Ensemble.toml"),
    "container": "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim",
    "par_final": f"{RUN_DIR}/5_Flux_Ensemble.3.par.csv",
    "obs_final": f"{RUN_DIR}/archive/4_pest_outputs/5_Flux_Ensemble.3.obs.csv",
    "obs_metadata": f"{RUN_DIR}/archive/3_problem_definition/observation_metadata.csv",
    "spread_obs": f"{RUN_DIR}/spread_error/spread_error_observations.csv",
    "spread_summary": f"{RUN_DIR}/spread_error/spread_error_summary.csv",
    "spread_persite": f"{RUN_DIR}/spread_error/spread_error_persite.csv",
    "spread_quintiles": f"{RUN_DIR}/spread_error/spread_error_quintiles.csv",
    "spot_verify": f"{RUN_DIR}/spread_error/spot_verify.csv",
    "par_prior": f"{RUN_DIR}/archive/4_pest_outputs/5_Flux_Ensemble.0.par.csv",
    "daily_metrics": f"{RUN_DIR}/archive/6_evaluation/daily_paired_metrics.csv",
    "provenance_config": f"{RUN_DIR}/archive/1_provenance/config.toml",
    "provenance_container_path": f"{RUN_DIR}/archive/1_provenance/container_path.txt",
    "provenance_manifest": f"{RUN_DIR}/archive/1_provenance/container_manifest.json",
}
DEFAULT_OUT_DIR = f"{RUN_DIR}/conditioned_ensemble_uncertainty"
RESULTS_NOTE = str(PROJECT_DIR / "notes" / "conditioned_ensemble_uncertainty_results.md")

SEED = 42
N_BOOT = 10_000
PERSITE_MIN_OBS = 20
N_REALIZATIONS = 199
COHORT_N = 2131
COHORT_SITES = 33

# Reproduction targets after manuscript rounding (plan section 6.1).
REPRO_TARGETS = {
    "pooled_spearman_3dp": 0.238,
    "n_obs": COHORT_N,
    "n_sites": COHORT_SITES,
    "persite_sites": 27,
    "persite_positive": 26,
    "quintile_rmse_lo_3dp": 0.205,
    "quintile_rmse_hi_3dp": 0.437,
}

# Numerical tolerances (recorded in metadata; plan Gate E).
TOL_ARCHIVE_EXACT = 1e-9  # recomputed vs archived full-precision CSV values
TOL_GATE_E = 1e-8  # rerun evaluator metrics vs archived daily_paired_metrics.csv
TOL_SPOT_RECOMPUTE = 1e-9  # recomputed member mean/std vs archived *_container columns
TOL_SPOT_ARCHIVE = 1e-6  # recomputed vs archived PEST-side values (float32 storage)

MEMBERS = ["ssebop", "sims", "geesebal", "eemetric", "ptjpl", "disalexi"]


class GateError(RuntimeError):
    """A validation gate failed; analysis must not proceed to interpretation."""


# ---------------------------------------------------------------------------
# Gate A: canonical identity
# ---------------------------------------------------------------------------


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    st = os.stat(path)
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
        "sha256": sha256_file(path),
    }


def gate_a_canonical_identity():
    identities = {
        k: file_identity(CANONICAL[k])
        for k in ["par_final", "obs_final", "obs_metadata", "spread_obs"]
    }
    identities["config"] = file_identity(CANONICAL["config"])
    identities["container_manifest"] = file_identity(CANONICAL["provenance_manifest"])
    identities["container_zarr_json"] = file_identity(
        os.path.join(CANONICAL["container"], "zarr.json")
    )

    # Confirm config and container against the canonical E1 archive.
    archived_cfg_sha = sha256_file(CANONICAL["provenance_config"])
    config_matches_archive = identities["config"]["sha256"] == archived_cfg_sha
    with open(CANONICAL["provenance_container_path"]) as f:
        archived_container = f.read().strip()
    with open(CANONICAL["provenance_manifest"]) as f:
        manifest_container = json.load(f).get("container_path")
    container_matches_archive = (
        archived_container == CANONICAL["container"]
        and manifest_container == CANONICAL["container"]
    )
    container_exists = os.path.isdir(CANONICAL["container"])

    ok = config_matches_archive and container_matches_archive and container_exists
    result = {
        "status": "PASS" if ok else "FAIL",
        "identities": identities,
        "archived_config_sha256": archived_cfg_sha,
        "config_matches_archive": bool(config_matches_archive),
        "archived_container_path": archived_container,
        "manifest_container_path": manifest_container,
        "container_matches_archive": bool(container_matches_archive),
        "container_exists": bool(container_exists),
    }
    if result["status"] == "FAIL":
        raise GateError(f"Gate A failed: {result}")
    return result


# ---------------------------------------------------------------------------
# Gate B: cohort identity and obsnme mapping
# ---------------------------------------------------------------------------


def load_frozen_cohort(path):
    """Frozen spread-error cohort; dates kept as YYYY-MM-DD strings for keying."""
    return pd.read_csv(path, dtype={"site": str, "date": str})


def load_obs_metadata_etf(path):
    """Landsat ensemble ETf rows of the Cat-3 weight-decomposition table."""
    meta = pd.read_csv(path, dtype={"site": str, "date": str})
    etf = meta[(meta["model"] == "ensemble") & meta["member_count"].notna()].copy()
    return etf[["obsnme", "site", "date", "target_etf", "ensemble_std"]].reset_index(drop=True)


def map_cohort_to_obsnme(cohort, etf_meta):
    """Map each frozen (site, date) capture one-to-one onto its PEST obsnme.

    Raises GateError on duplicated keys, unmapped captures, or archived
    target/spread disagreement (any of which would mean the frozen cohort and
    the calibration problem no longer describe the same observations).
    """
    if cohort.duplicated(["site", "date"]).any():
        raise GateError("Gate B failed: duplicated site-date keys in frozen cohort")
    if etf_meta.duplicated(["site", "date"]).any():
        raise GateError("Gate B failed: duplicated site-date keys in ETf metadata")
    if etf_meta["obsnme"].duplicated().any():
        raise GateError("Gate B failed: duplicated obsnme in ETf metadata")

    merged = cohort.merge(
        etf_meta, on=["site", "date"], how="left", validate="one_to_one", suffixes=("", "_meta")
    )
    missing = merged["obsnme"].isna()
    if missing.any():
        keys = merged.loc[missing, ["site", "date"]].to_records(index=False).tolist()
        raise GateError(f"Gate B failed: {missing.sum()} captures unmapped: {keys[:10]}")

    d_target = float((merged["target"] - merged["target_etf"]).abs().max())
    d_spread = float((merged["spread"] - merged["ensemble_std"]).abs().max())
    if d_target > TOL_ARCHIVE_EXACT or d_spread > TOL_ARCHIVE_EXACT:
        raise GateError(
            f"Gate B failed: archived target/spread disagree with metadata "
            f"(max|dtarget|={d_target:.2e}, max|dspread|={d_spread:.2e})"
        )
    return merged.drop(columns=["target_etf", "ensemble_std"])


def gate_b_cohort_identity(merged):
    ok = len(merged) == COHORT_N and merged["site"].nunique() == COHORT_SITES
    result = {
        "status": "PASS" if ok else "FAIL",
        "n_captures": int(len(merged)),
        "n_sites": int(merged["site"].nunique()),
        "expected_captures": COHORT_N,
        "expected_sites": COHORT_SITES,
    }
    if not ok:
        raise GateError(f"Gate B failed: {result}")
    return result


# ---------------------------------------------------------------------------
# Gate C: realization identity
# ---------------------------------------------------------------------------


def load_conditioned_matrix(obs_csv, obsnames):
    """Read only real_name + the mapped capture columns from the obs ensemble."""
    wanted = ["real_name"] + list(obsnames)
    try:
        df = pd.read_csv(obs_csv, usecols=wanted)
    except ValueError as exc:
        raise GateError(f"Gate B failed: obs ensemble missing mapped columns: {exc}")
    df["real_name"] = df["real_name"].astype(str)
    df = df.set_index("real_name")
    missing = [c for c in obsnames if c not in df.columns]
    if missing:
        raise GateError(f"Gate B failed: {len(missing)} obsnme columns absent from obs ensemble")
    return df[list(obsnames)]


def select_numbered_realizations(df, expected=N_REALIZATIONS):
    """Exclude the base realization; require exactly `expected` numbered rows."""
    idx = df.index.astype(str)
    if "base" not in set(idx):
        raise GateError("Gate C failed: base realization not present")
    numbered = [i for i in idx if i != "base"]
    non_numeric = [i for i in numbered if not i.isdigit()]
    if non_numeric:
        raise GateError(f"Gate C failed: non-numeric realization ids {non_numeric[:5]}")
    if expected is not None and len(numbered) != expected:
        raise GateError(
            f"Gate C failed: {len(numbered)} numbered realizations, expected {expected}"
        )
    return df.loc[sorted(numbered, key=int)]


def gate_c_realization_identity(par_df, obs_matrix):
    par_ids = set(par_df.index.astype(str))
    obs_ids = set(obs_matrix.index.astype(str))
    if par_ids != obs_ids:
        raise GateError(
            f"Gate C failed: parameter/observation realization ids differ "
            f"(par-only={sorted(par_ids - obs_ids)[:5]}, obs-only={sorted(obs_ids - par_ids)[:5]})"
        )
    n_numbered = len(par_ids - {"base"})
    result = {
        "status": "PASS" if n_numbered == N_REALIZATIONS else "FAIL",
        "n_numbered": n_numbered,
        "ids_match": True,
    }
    if result["status"] == "FAIL":
        raise GateError(f"Gate C failed: {result}")
    return result


# ---------------------------------------------------------------------------
# Conditioned simulation spread (plan section 5.3)
# ---------------------------------------------------------------------------


def conditioned_stats(matrix):
    """Per-capture stats across conditioned realizations (rows).

    Quantiles use numpy's default linear interpolation; std is sample (ddof=1).
    """
    arr = matrix.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise GateError(f"conditioned matrix has {n_bad} non-finite values; investigate upstream")
    q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    out = pd.DataFrame(
        {
            "q05": q05,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q95": q95,
            "spread_conditioned_std": arr.std(axis=0, ddof=1),
        },
        index=matrix.columns,
    )
    out["spread_conditioned_iqr"] = out["q75"] - out["q25"]
    out["width90"] = out["q95"] - out["q05"]
    return out


# ---------------------------------------------------------------------------
# Gate D: reproduce the existing retrieval-target spread-error result
# ---------------------------------------------------------------------------


def retrieval_persite(cohort, spread_col="spread", err_col="abs_err_etf", min_obs=PERSITE_MIN_OBS):
    rows = []
    for site, g in cohort.groupby("site"):
        g = g.dropna(subset=[spread_col, err_col])
        if len(g) < min_obs:
            continue
        rho, p = stats.spearmanr(g[spread_col].values, g[err_col].values)
        rows.append({"site": site, "n": len(g), "spearman_rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


def retrieval_quintiles(cohort, spread_col="spread", err_col="err_etf", n_bins=5):
    df = cohort.dropna(subset=[spread_col, err_col]).copy()
    df["bin"] = pd.qcut(df[spread_col], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby("bin"):
        err = g[err_col].values
        rows.append(
            {
                "quintile": int(b) + 1,
                "n": len(g),
                "mean_spread": float(g[spread_col].mean()),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
            }
        )
    return pd.DataFrame(rows)


def gate_d_reproduction(cohort, container):
    """Reproduce plan section 6.1 values and the archived 20-row spot check."""
    rho, _ = stats.spearmanr(cohort["spread"].values, cohort["abs_err_etf"].values)
    persite = retrieval_persite(cohort)
    n_positive = int((persite["spearman_rho"] > 0).sum())
    quint = retrieval_quintiles(cohort)

    archived_summary = pd.read_csv(CANONICAL["spread_summary"], header=None, index_col=0).squeeze(
        "columns"
    )
    archived_rho = float(archived_summary["pooled_etf_spearman_rho"])
    archived_quint = pd.read_csv(CANONICAL["spread_quintiles"])
    archived_persite = pd.read_csv(CANONICAL["spread_persite"]).set_index("site")

    rp = persite.set_index("site")
    persite_matches_archive = set(rp.index) == set(archived_persite.index) and bool(
        np.allclose(
            rp.loc[archived_persite.index, "spearman_rho"].values,
            archived_persite["spearman_rho"].values,
            atol=TOL_ARCHIVE_EXACT,
            rtol=0,
        )
    )
    quintile_bins_match = len(quint) == len(archived_quint) == 5
    quintile_monotonic = bool(np.all(np.diff(quint["RMSE"].values) > 0))

    checks = {
        "pooled_rho_recomputed": float(rho),
        "pooled_rho_archived": archived_rho,
        "pooled_rho_matches_archive": bool(abs(rho - archived_rho) <= TOL_ARCHIVE_EXACT),
        "pooled_rho_3dp": round(float(rho), 3),
        "pooled_rho_3dp_matches_plan": round(float(rho), 3) == REPRO_TARGETS["pooled_spearman_3dp"],
        "n_obs": int(len(cohort)),
        "n_obs_matches": len(cohort) == REPRO_TARGETS["n_obs"],
        "n_sites": int(cohort["site"].nunique()),
        "n_sites_matches": cohort["site"].nunique() == REPRO_TARGETS["n_sites"],
        "persite_sites": int(len(persite)),
        "persite_sites_matches": len(persite) == REPRO_TARGETS["persite_sites"],
        "persite_positive": n_positive,
        "persite_positive_matches": n_positive == REPRO_TARGETS["persite_positive"],
        "quintile_rmse_lo": float(quint["RMSE"].iloc[0]),
        "quintile_rmse_hi": float(quint["RMSE"].iloc[-1]),
        "quintile_rmse_lo_3dp_matches": round(float(quint["RMSE"].iloc[0]), 3)
        == REPRO_TARGETS["quintile_rmse_lo_3dp"],
        "quintile_rmse_hi_3dp_matches": round(float(quint["RMSE"].iloc[-1]), 3)
        == REPRO_TARGETS["quintile_rmse_hi_3dp"],
        "quintile_bins_match": bool(quintile_bins_match),
        "quintile_monotonic": quintile_monotonic,
        "quintile_rmse_matches_archive": bool(
            quintile_bins_match
            and np.allclose(
                quint["RMSE"].values, archived_quint["RMSE"].values, atol=TOL_ARCHIVE_EXACT, rtol=0
            )
        ),
        "persite_matches_archive": bool(persite_matches_archive),
    }

    spot = reproduce_spot_check(CANONICAL["spot_verify"], container)
    checks.update(spot)

    required = [
        "pooled_rho_matches_archive",
        "pooled_rho_3dp_matches_plan",
        "n_obs_matches",
        "n_sites_matches",
        "persite_sites_matches",
        "persite_positive_matches",
        "quintile_rmse_lo_3dp_matches",
        "quintile_rmse_hi_3dp_matches",
        "quintile_bins_match",
        "quintile_monotonic",
        "quintile_rmse_matches_archive",
        "persite_matches_archive",
        "spot_recompute_matches",
        "spot_archive_matches",
    ]
    checks["status"] = "PASS" if all(checks[k] for k in required) else "FAIL"
    if checks["status"] == "FAIL":
        failed = [k for k in required if not checks[k]]
        raise GateError(f"Gate D failed on {failed}: {checks}")
    return checks


def reproduce_spot_check(spot_csv, container):
    """Recompute member mean/std for the archived 20-row spot check."""
    spot = pd.read_csv(spot_csv, dtype={"site": str, "date": str})
    member_cache = {}
    d_recompute, d_archive = [], []
    for _, r in spot.iterrows():
        fid, d = r["site"], pd.Timestamp(r["date"])
        if fid not in member_cache:
            frames = {}
            for m in MEMBERS:
                mdf = container.query.dataframe(
                    f"remote_sensing/etf/landsat/{m}/no_mask", fields=[fid]
                )
                if fid in mdf.columns:
                    frames[m] = mdf[fid]
            member_cache[fid] = frames
        vals = np.array(
            [
                float(s.loc[d])
                for s in member_cache[fid].values()
                if d in s.index and np.isfinite(s.loc[d])
            ]
        )
        rmean = float(np.mean(vals))
        rstd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

        def _diff(a, b):
            if np.isnan(a) and np.isnan(b):
                return 0.0
            return abs(a - b)

        d_recompute.append(_diff(rmean, r["mean_container"]))
        d_recompute.append(_diff(rstd, r["std_container"]))
        d_archive.append(_diff(rmean, r["target_archive"]))
        d_archive.append(_diff(rstd, r["spread_archive"]))
    return {
        "spot_n": int(len(spot)),
        "spot_max_diff_recompute": float(np.max(d_recompute)),
        "spot_max_diff_archive": float(np.max(d_archive)),
        "spot_recompute_matches": bool(np.max(d_recompute) <= TOL_SPOT_RECOMPUTE),
        "spot_archive_matches": bool(np.max(d_archive) <= TOL_SPOT_ARCHIVE),
    }


# ---------------------------------------------------------------------------
# Gate E: central-prediction identity (canonical evaluator, forward-only)
# ---------------------------------------------------------------------------


def gate_e_central_prediction(cfg, container, fids, flux_dir):
    """Rerun the canonical daily evaluation and require the archived metrics.

    The forward model output is captured from inside ``evaluate.evaluate`` so
    the capture-level predictions used downstream are the exact simulations
    that reproduce the frozen E1 evaluation (one forward run, one code path).
    """
    captured = {}
    orig_run = ev.run_calibrated_model
    orig_write = ev.write_excluded_sites

    def _capturing_run(cfg_, container_, fids_, params_):
        res = orig_run(cfg_, container_, fids_, params_)
        captured["model_results"] = res
        return res

    ev.run_calibrated_model = _capturing_run
    # Keep the run read-only outside the output directory: the evaluator's
    # excluded-sites side-effect file is suppressed.
    ev.write_excluded_sites = lambda *a, **k: None
    try:
        metrics = ev.evaluate(
            cfg, container, CANONICAL["par_final"], fids, flux_dir, openet_source="volk"
        )
    finally:
        ev.run_calibrated_model = orig_run
        ev.write_excluded_sites = orig_write

    archived = pd.read_csv(CANONICAL["daily_metrics"], index_col="fid")
    produced = metrics.copy()

    if set(produced.index) != set(archived.index):
        raise GateError(
            f"Gate E failed: evaluated site set differs from archive "
            f"(produced {len(produced)}, archived {len(archived)}, "
            f"only-produced={sorted(set(produced.index) - set(archived.index))[:5]}, "
            f"only-archived={sorted(set(archived.index) - set(produced.index))[:5]})"
        )
    archived = archived.reindex(produced.index)
    common_cols = [c for c in archived.columns if c in produced.columns]
    if len(common_cols) != len(archived.columns):
        missing = [c for c in archived.columns if c not in produced.columns]
        raise GateError(f"Gate E failed: metric columns missing from rerun: {missing}")

    a = archived[common_cols].to_numpy(dtype=float)
    p = produced[common_cols].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(p)
    diff = np.abs(a - p)
    diff[both_nan] = 0.0
    if np.isnan(diff).any():
        raise GateError("Gate E failed: NaN pattern differs between rerun and archive")
    max_diff = float(np.max(diff))

    result = {
        "status": "PASS" if max_diff <= TOL_GATE_E else "FAIL",
        "max_abs_diff": max_diff,
        "tolerance": TOL_GATE_E,
        "n_sites": int(len(produced)),
        "n_metric_columns": len(common_cols),
        "openet_source": "volk",
    }
    if result["status"] == "FAIL":
        raise GateError(f"Gate E failed: max|diff|={max_diff:.3e} > {TOL_GATE_E}")
    return result, captured["model_results"]


def extract_capture_predictions(merged, model_results):
    """Effective-parameter ETf at each frozen capture date (plan section 5.1)."""
    preds = np.empty(len(merged))
    for i, (site, date) in enumerate(zip(merged["site"], merged["date"])):
        if site not in model_results:
            raise GateError(f"cohort site {site} missing from forward-model results")
        try:
            preds[i] = float(model_results[site]["etf_model"].loc[pd.Timestamp(date)])
        except KeyError:
            raise GateError(f"capture date {date} at {site} outside forward-model index")
    if not np.isfinite(preds).all():
        n_bad = int((~np.isfinite(preds)).sum())
        raise GateError(f"{n_bad} non-finite effective predictions; investigate upstream")
    return preds


# ---------------------------------------------------------------------------
# Analyses (plan sections 6.2-6.4)
# ---------------------------------------------------------------------------

SPREAD_MEASURES = {
    "retrieval": "spread_retrieval",
    "conditioned_iqr": "spread_conditioned_iqr",
    "conditioned_std": "spread_conditioned_std",
    "conditioned_width90": "width90",
}


def _safe_spearman(x, y):
    """Spearman rho with an explicit cause when undefined (never silently NaN)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan, "non_finite_values"
    if np.unique(x).size < 2:
        return np.nan, "constant_spread"
    if np.unique(y).size < 2:
        return np.nan, "constant_error"
    rho, _ = stats.spearmanr(x, y)
    return float(rho), None


def persite_associations(obs, min_obs=PERSITE_MIN_OBS):
    """Per-site Spearman correlations of each spread measure with both errors.

    Primary: rho(spread_retrieval, abs_error_effective) minus
    rho(spread_conditioned_iqr, abs_error_effective) on sites with >= min_obs
    captures. Undefined correlations are recorded with their cause, never
    silently dropped (plan section 6.2).
    """
    rows = []
    undefined = []
    for site, g in obs.groupby("site"):
        row = {"site": site, "n": int(len(g)), "eligible": bool(len(g) >= min_obs)}
        for label, col in SPREAD_MEASURES.items():
            rho, cause = _safe_spearman(g[col], g["abs_error_effective"])
            row[f"rho_{label}"] = rho
            if cause is not None and row["eligible"]:
                undefined.append({"site": site, "measure": label, "cause": cause})
                row[f"rho_{label}_undefined_cause"] = cause
        rho_sens, cause_sens = _safe_spearman(g["spread_retrieval"], g["abs_error_ensemble_median"])
        row["rho_retrieval_err_ensmed"] = rho_sens
        rho_sens2, _ = _safe_spearman(g["spread_conditioned_iqr"], g["abs_error_ensemble_median"])
        row["rho_conditioned_iqr_err_ensmed"] = rho_sens2
        row["delta_rho"] = row["rho_retrieval"] - row["rho_conditioned_iqr"]
        row["delta_rho_std"] = row["rho_retrieval"] - row["rho_conditioned_std"]
        row["delta_rho_width90"] = row["rho_retrieval"] - row["rho_conditioned_width90"]
        row["delta_rho_err_ensmed"] = (
            row["rho_retrieval_err_ensmed"] - row["rho_conditioned_iqr_err_ensmed"]
        )
        row["coverage_90"] = float(g["covered_90"].mean())
        row["median_width90"] = float(g["width90"].median())
        row["median_iqr"] = float(g["spread_conditioned_iqr"].median())
        row["median_spread_retrieval"] = float(g["spread_retrieval"].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True), undefined


def bootstrap_site_median(values, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for the median of site-level values, resampling sites.

    Sites are the sampling unit; each carries its (already-paired) site-level
    statistic, so pairing between spread measures is preserved by construction.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("site-level values must be non-empty and finite")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    reps = np.median(values[idx], axis=1)
    return {
        "point": float(np.median(values)),
        "ci_lo": float(np.percentile(reps, 2.5)),
        "ci_hi": float(np.percentile(reps, 97.5)),
        "n_sites": int(values.size),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def pooled_site_bootstrap(obs, n_boot=N_BOOT, seed=SEED):
    """Whole-site bootstrap for pooled correlations and coverage (plan 6.3, 5.4).

    Each replicate resamples complete sites with replacement and carries all
    captures with both spread measures, the common error, and coverage flags,
    preserving the paired structure.
    """
    sites = sorted(obs["site"].unique())
    groups = {
        s: g[
            ["spread_retrieval", "spread_conditioned_iqr", "abs_error_effective", "covered_90"]
        ].to_numpy(dtype=float)
        for s, g in obs.groupby("site")
    }
    full = np.concatenate([groups[s] for s in sites], axis=0)

    def _stats(block):
        rho_r, _ = stats.spearmanr(block[:, 0], block[:, 2])
        rho_c, _ = stats.spearmanr(block[:, 1], block[:, 2])
        return rho_r, rho_c, rho_r - rho_c, float(np.mean(block[:, 3]))

    point = _stats(full)
    rho_rc, _ = stats.spearmanr(full[:, 0], full[:, 1])

    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, 4))
    site_arr = np.array(sites)
    for b in range(n_boot):
        chosen = site_arr[rng.integers(0, len(site_arr), size=len(site_arr))]
        block = np.concatenate([groups[s] for s in chosen], axis=0)
        reps[b] = _stats(block)

    def _ci(col):
        return {
            "ci_lo": float(np.percentile(reps[:, col], 2.5)),
            "ci_hi": float(np.percentile(reps[:, col], 97.5)),
        }

    return {
        "pooled_rho_retrieval": {"point": float(point[0]), **_ci(0)},
        "pooled_rho_conditioned_iqr": {"point": float(point[1]), **_ci(1)},
        "pooled_rho_difference": {"point": float(point[2]), **_ci(2)},
        "pooled_coverage_90": {"point": float(point[3]), **_ci(3)},
        "pooled_rho_retrieval_vs_conditioned": float(rho_rc),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def quintile_diagnostics(obs, n_bins=5):
    """Spread-ranked error tables (plan section 6.4): raw and within-site rank."""
    tables = []
    for analysis in ["raw", "within_site_rank"]:
        for label in ["retrieval", "conditioned_iqr"]:
            col = SPREAD_MEASURES[label]
            df = obs[[col, "abs_error_effective", "site"]].dropna().copy()
            if analysis == "within_site_rank":
                df["rank_col"] = df.groupby("site")[col].rank(pct=True)
                bin_source = "rank_col"
            else:
                bin_source = col
            df["bin"] = pd.qcut(df[bin_source], n_bins, labels=False, duplicates="drop")
            for b, g in df.groupby("bin"):
                err = g["abs_error_effective"].values
                tables.append(
                    {
                        "analysis": analysis,
                        "spread_measure": label,
                        "quintile": int(b) + 1,
                        "n": int(len(g)),
                        "spread_mean": float(g[col].mean()),
                        "spread_median": float(g[col].median()),
                        "MAE": float(np.mean(err)),
                        "RMSE": float(np.sqrt(np.mean(err**2))),
                    }
                )
    return pd.DataFrame(tables)


# ---------------------------------------------------------------------------
# Optional prior-to-conditioned contraction (plan section 6.5)
# ---------------------------------------------------------------------------


def _param_site_from_col(col, fids):
    """Replicates evaluate.parse_pest_params column parsing."""
    parts = col.split("_ptype:")[0]
    parts = parts.replace("pname:p_", "")
    parts = parts.rsplit("_:0", 1)[0]
    for fid in fids:
        if parts.lower().endswith(f"_{fid.lower()}"):
            return parts[: -(len(fid) + 1)], fid
    return None, None


def contraction_table(par_prior, par_final, fids):
    """Ratio of final to prior parameter IQR per site, summarized by parameter.

    Descriptive only: characterizes conditioning strength and weakly
    constrained parameters, not recovery of true physical parameters.
    """
    if set(par_prior.columns) != set(par_final.columns):
        raise GateError("6.5: prior and final parameter columns differ")
    par_prior = par_prior[par_final.columns]
    records = []
    for col in par_final.columns:
        param, fid = _param_site_from_col(col, fids)
        if param is None:
            continue
        p0 = par_prior[col].to_numpy(dtype=float)
        p3 = par_final[col].to_numpy(dtype=float)
        iqr0 = float(np.quantile(p0, 0.75) - np.quantile(p0, 0.25))
        iqr3 = float(np.quantile(p3, 0.75) - np.quantile(p3, 0.25))
        records.append(
            {
                "parameter": param,
                "site": fid,
                "iqr_prior": iqr0,
                "iqr_final": iqr3,
                "ratio_final_prior": iqr3 / iqr0 if iqr0 > 0 else np.nan,
            }
        )
    persite = pd.DataFrame(records)
    summary = (
        persite.groupby("parameter")
        .agg(
            n_sites=("site", "nunique"),
            median_ratio=("ratio_final_prior", "median"),
            q25_ratio=("ratio_final_prior", lambda s: s.quantile(0.25)),
            q75_ratio=("ratio_final_prior", lambda s: s.quantile(0.75)),
            n_undefined=("ratio_final_prior", lambda s: int(s.isna().sum())),
        )
        .sort_values("median_ratio")
        .reset_index()
    )
    return persite, summary


# ---------------------------------------------------------------------------
# Outcome classification (plan section 10)
# ---------------------------------------------------------------------------


def classify_outcome(delta_ci, rho_cond_ci):
    """Mechanical outcome per plan section 10; no post hoc adaptation.

    Outcome D (conditioned spread not informative about error) is determined
    solely from the prespecified association evidence: the site-bootstrap
    interval for the median within-site conditioned correlation failing to
    exclude zero in the positive direction. Coverage is reported descriptively
    and never thresholded (plan sections 9 and 10).
    """
    if delta_ci["ci_lo"] > 0:
        comparison = "A"
    elif delta_ci["ci_hi"] < 0:
        comparison = "conditioned_favored"
    else:
        comparison = "B"
    conditioned_informative = rho_cond_ci["ci_lo"] > 0

    if comparison == "A":
        primary = "A"
    elif conditioned_informative:
        primary = "C"
    elif comparison == "B":
        primary = "B"
    else:
        primary = "D"

    return {
        "comparison_outcome": comparison,
        "conditioned_informative": bool(conditioned_informative),
        "primary_outcome": primary,
        "outcome_d_applies": bool(not conditioned_informative),
    }


OUTCOME_TEXT = {
    "A": (
        "At flux-matched E1 acquisitions, multi-algorithm retrieval spread tracked "
        "absolute SWIM-RS ETf error more closely than dispersion among simulations "
        "from the final conditioned parameter ensemble. Retrieval disagreement was "
        "therefore the more useful observation-level reliability signal in this "
        "framework. Neither spread is a calibrated estimate of total predictive "
        "uncertainty."
    ),
    "B": (
        "The analysis did not distinguish the two spread signals: the paired "
        "site-bootstrap interval for the median correlation difference spans zero. "
        "Differences between pooled point estimates must not be used to claim "
        "superiority."
    ),
    "C": (
        "Conditioned simulation spread carried parameter-conditional reliability "
        "information (positive within-site association with error) and the paired "
        "comparison did not favor retrieval spread. This is not total predictive "
        "uncertainty; that would require an independently justified observation, "
        "forcing, and structural error model, which is outside this analysis."
    ),
    "D": (
        "Residual parameter dispersion in the final conditioned ensemble does not "
        "capture the dominant sources of external ET error and should not be "
        "propagated as a stand-alone predictive interval."
    ),
}


# ---------------------------------------------------------------------------
# Figure (plan section 13)
# ---------------------------------------------------------------------------

COLOR_RETRIEVAL = "#0072B2"  # Okabe-Ito blue
COLOR_CONDITIONED = "#D55E00"  # Okabe-Ito vermillion


def _binned_medians(x, y, n_bins=10):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    df["bin"] = pd.qcut(df["x"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(x=("x", "median"), y=("y", "median"))
    return g["x"].values, g["y"].values


def make_figure(obs, quintiles, out_png, out_pdf):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))

    err = obs["abs_error_effective"].values
    ymax = float(np.quantile(err, 0.99))

    panels = [
        ("spread_retrieval", "Retrieval spread (member std, ETf)", COLOR_RETRIEVAL),
        ("spread_conditioned_iqr", "Conditioned IQR (ETf)", COLOR_CONDITIONED),
    ]
    site_medians = obs.groupby("site")[
        ["spread_retrieval", "spread_conditioned_iqr", "abs_error_effective"]
    ].median()
    for ax, (col, xlabel, color) in zip(axes[:2], panels):
        x = obs[col].values
        ax.scatter(x, err, s=5, color="0.55", alpha=0.25, linewidths=0, rasterized=True)
        bx, by = _binned_medians(x, err)
        ax.plot(bx, by, color=color, lw=2, marker="o", ms=4, label="decile median", zorder=3)
        ax.scatter(
            site_medians[col],
            site_medians["abs_error_effective"],
            s=28,
            facecolor="none",
            edgecolor="0.2",
            linewidths=0.9,
            label="site medians",
            zorder=4,
        )
        ax.set_xlim(0, float(np.quantile(x, 0.99)))
        ax.set_ylim(0, ymax)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("|SWIM-RS ETf error| (effective params)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_yticklabels([])

    raw = quintiles[quintiles["analysis"] == "raw"]
    width = 0.38
    q = np.arange(1, 6)
    for off, label, color in [
        (-width / 2, "retrieval", COLOR_RETRIEVAL),
        (width / 2, "conditioned_iqr", COLOR_CONDITIONED),
    ]:
        sub = raw[raw["spread_measure"] == label].set_index("quintile").reindex(q)
        axes[2].bar(
            q + off,
            sub["RMSE"].values,
            width=width,
            color=color,
            label=label.replace("_", " "),
        )
    axes[2].set_xticks(q)
    axes[2].set_xlabel("Spread quintile (raw)")
    axes[2].set_ylabel("RMSE of ETf error")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "E1 acquisition-date error vs retrieval spread and conditioned simulation spread",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results note (plan section 12)
# ---------------------------------------------------------------------------


def _markdown_table(df, floatfmt=".3f"):
    """Minimal markdown table (tabulate is not a project dependency)."""

    def _cell(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_results_note(path, meta, summary_rows, contraction_summary):
    gates = meta["gates"]
    outcome = meta["outcome"]
    s = {r["metric"]: r for r in summary_rows}

    def _fmt(key, signed=True):
        r = s[key]
        spec = "+.3f" if signed else ".3f"
        if np.isfinite(r.get("ci_lo", np.nan)):
            return (
                f"{format(r['point'], spec)} "
                f"[{format(r['ci_lo'], spec)}, {format(r['ci_hi'], spec)}]"
            )
        return format(r["point"], spec)

    lines = [
        "# E1 Conditioned-Ensemble Uncertainty Diagnostic — Results",
        "",
        f"**Date:** {meta['finished']}",
        "**Plan:** `notes/conditioned_ensemble_uncertainty_plan.md` (prespecified; no adaptations)",
        f"**Outputs:** `{meta['out_dir']}`",
        "",
        "## Validation gates",
        "",
        "| Gate | Status | Key detail |",
        "| --- | --- | --- |",
        f"| A canonical identity | {gates['A']['status']} | config matches archive: "
        f"{gates['A']['config_matches_archive']} |",
        f"| B cohort identity | {gates['B']['status']} | {gates['B']['n_captures']} captures, "
        f"{gates['B']['n_sites']} sites, one-to-one obsnme map |",
        f"| C realization identity | {gates['C']['status']} | {gates['C']['n_numbered']} "
        "numbered realizations, par/obs ids match, base excluded |",
        f"| D existing-result reproduction | {gates['D']['status']} | pooled rho "
        f"{gates['D']['pooled_rho_3dp']}, {gates['D']['persite_positive']}/"
        f"{gates['D']['persite_sites']} positive, spot check reproduced |",
        f"| E central-prediction identity | {gates['E']['status']} | max abs diff vs archived "
        f"daily metrics {gates['E']['max_abs_diff']:.2e} (tol {gates['E']['tolerance']:.0e}) |",
        f"| F leakage | {gates['F']['status']} | {gates['F']['statement']} |",
        f"| G deterministic inference | {gates['G']['status']} | seed {meta['seed']}, "
        f"{meta['n_boot']} site-bootstrap replicates |",
        "",
        "## Primary result (section 6.2)",
        "",
        f"Median paired difference delta_rho_site = rho(retrieval, |err|) - "
        f"rho(conditioned IQR, |err|) over {s['median_delta_rho']['n_sites']} eligible sites:",
        "",
        f"**{_fmt('median_delta_rho')}** (site bootstrap, 95% percentile interval)",
        "",
        f"- median within-site rho, retrieval spread: {_fmt('median_rho_retrieval')}",
        f"- median within-site rho, conditioned IQR: {_fmt('median_rho_conditioned_iqr')}",
        "",
        "## Pooled associations (section 6.3, descriptive)",
        "",
        f"- retrieval spread vs |err|: {_fmt('pooled_rho_retrieval')}",
        f"- conditioned IQR vs |err|: {_fmt('pooled_rho_conditioned_iqr')}",
        f"- difference: {_fmt('pooled_rho_difference')}",
        f"- retrieval spread vs conditioned IQR: {_fmt('pooled_rho_retrieval_vs_conditioned')}",
        "",
        "## Empirical coverage diagnostic (section 5.4)",
        "",
        f"- pooled coverage of flux ETf by central 90% conditioned envelope: "
        f"{_fmt('pooled_coverage_90', signed=False)}",
        f"- median site-level coverage: {_fmt('median_site_coverage', signed=False)}",
        f"- median envelope width (q95-q05): {_fmt('median_width90', signed=False)}",
        f"- median conditioned IQR: {_fmt('median_iqr', signed=False)}; "
        f"median retrieval spread: {_fmt('median_spread_retrieval', signed=False)}; "
        f"median abs error: {_fmt('median_abs_error', signed=False)}",
        "",
        "This is an empirical coverage diagnostic, not validation of a nominal 90%",
        "predictive interval: the envelope omits forcing, structural, retrieval, and",
        "flux-reference uncertainty.",
        "",
        "## Sensitivities (prespecified)",
        "",
        f"- delta with conditioned std instead of IQR: {_fmt('median_delta_rho_std')}",
        f"- delta with 90% width instead of IQR: {_fmt('median_delta_rho_width90')}",
        f"- delta with ensemble-median-centered error: {_fmt('median_delta_rho_err_ensmed')}",
        "",
        "## Outcome (section 10, mechanical)",
        "",
        f"**Primary outcome: {outcome['primary_outcome']}**"
        + (" (with Outcome D characteristics)" if outcome["outcome_d_applies"] else ""),
        "",
        f"> {OUTCOME_TEXT[outcome['primary_outcome']]}",
    ]
    if outcome["outcome_d_applies"] and outcome["primary_outcome"] != "D":
        lines += ["", f"> {OUTCOME_TEXT['D']}"]
    if contraction_summary is not None:
        contraction_block = _markdown_table(contraction_summary)
    else:
        contraction_block = (
            "Section 6.5 contraction table failed to compute (optional, descriptive "
            "only); see uncertainty_metadata.json."
        )
    lines += [
        "",
        "## Prior-to-conditioned contraction (section 6.5, descriptive)",
        "",
        "Ratio of final to prior parameter IQR (median across sites), most to least",
        "contracted:",
        "",
        contraction_block,
        "",
        "## Manuscript disposition (pending user review)",
        "",
        "Per plan section 14, likely changes given this outcome: 2-3 Methods sentences",
        "defining conditioned simulation spread and the site-bootstrap comparison; one",
        "Results paragraph with the paired contrast and empirical coverage; a Discussion",
        "statement on which signal was informative; retain 'final conditioned ensemble'",
        "and 'effective parameters', remove 'posterior ensemble' from reader-facing",
        "text. No manuscript edit has been made by this analysis.",
        "",
        "## Execution record",
        "",
        f"- runtime: {meta['runtime_s']:.0f} s; peak RSS: {meta['peak_rss_mb']:.0f} MB",
        f"- rerun: `uv run python {Path(__file__).name}` (deterministic, seed {meta['seed']})",
        f"- undefined correlations: {len(meta['undefined_correlations'])} "
        f"{meta['undefined_correlations'] if meta['undefined_correlations'] else ''}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(out_dir=DEFAULT_OUT_DIR, n_boot=N_BOOT):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "started": datetime.now(tz=UTC).isoformat(),
        "plan": "examples/5_Flux_Ensemble/notes/conditioned_ensemble_uncertainty_plan.md",
        "canonical_inputs": dict(CANONICAL),
        "out_dir": out_dir,
        "seed": SEED,
        "n_boot": int(n_boot),
        "quantile_method": "numpy linear interpolation",
        "std_ddof": 1,
        "persite_min_obs": PERSITE_MIN_OBS,
        "tolerances": {
            "archive_exact": TOL_ARCHIVE_EXACT,
            "gate_e": TOL_GATE_E,
            "spot_recompute": TOL_SPOT_RECOMPUTE,
            "spot_archive": TOL_SPOT_ARCHIVE,
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "gates": {},
    }
    if n_boot != N_BOOT:
        meta["non_canonical"] = f"n_boot overridden to {n_boot}; canonical value is {N_BOOT}"

    def _fail(exc):
        meta["status"] = "FAILED"
        meta["error"] = str(exc)
        meta["finished"] = datetime.now(tz=UTC).isoformat()
        with open(os.path.join(out_dir, "uncertainty_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        raise SystemExit(f"VALIDATION FAILURE — analysis stopped.\n{exc}")

    container = None
    try:
        print("=== Gate A: canonical identity ===")
        meta["gates"]["A"] = gate_a_canonical_identity()
        print(f"  PASS (config matches archive: {meta['gates']['A']['config_matches_archive']})")

        print("=== Gate B: cohort identity ===")
        cohort = load_frozen_cohort(CANONICAL["spread_obs"])
        etf_meta = load_obs_metadata_etf(CANONICAL["obs_metadata"])
        merged = map_cohort_to_obsnme(cohort, etf_meta)
        meta["gates"]["B"] = gate_b_cohort_identity(merged)
        print(f"  PASS ({len(merged)} captures, {merged['site'].nunique()} sites)")

        print("=== Gate C: realization identity ===")
        obs_matrix_all = load_conditioned_matrix(CANONICAL["obs_final"], merged["obsnme"].tolist())
        par_all = pd.read_csv(CANONICAL["par_final"], index_col=0)
        par_all.index = par_all.index.astype(str)
        meta["gates"]["C"] = gate_c_realization_identity(par_all, obs_matrix_all)
        obs_matrix = select_numbered_realizations(obs_matrix_all)
        print(f"  PASS ({len(obs_matrix)} numbered realizations, base excluded)")

        cond = conditioned_stats(obs_matrix)

        print("=== Gate D: existing-result reproduction ===")
        container = SwimContainer.open(CANONICAL["container"], mode="r")
        meta["gates"]["D"] = gate_d_reproduction(cohort, container)
        print(
            f"  PASS (pooled rho {meta['gates']['D']['pooled_rho_3dp']}, "
            f"{meta['gates']['D']['persite_positive']}/{meta['gates']['D']['persite_sites']} "
            "positive per-site)"
        )

        print("=== Gate E: central-prediction identity ===")
        cfg = ev.load_config(CANONICAL["config"])
        flux_dir = ev.resolve_flux_dir(cfg)
        fids = ev.apply_exclusions(container.field_uids)
        gate_e, model_results = gate_e_central_prediction(cfg, container, fids, flux_dir)
        meta["gates"]["E"] = gate_e
        print(f"  PASS (max|diff| = {gate_e['max_abs_diff']:.2e})")

        # Assemble the capture-level observation table (plan section 12).
        obs = merged.rename(columns={"spread": "spread_retrieval"}).copy()
        obs = obs.join(cond, on="obsnme")
        obs["etf_effective"] = extract_capture_predictions(obs, model_results)
        obs["error_effective"] = obs["etf_effective"] - obs["flux_etf"]
        obs["abs_error_effective"] = obs["error_effective"].abs()
        obs["abs_error_ensemble_median"] = (obs["q50"] - obs["flux_etf"]).abs()
        obs["covered_90"] = (obs["flux_etf"] >= obs["q05"]) & (obs["flux_etf"] <= obs["q95"])

        meta["gates"]["F"] = {
            "status": "PASS",
            "statement": (
                "flux ETf used only to compute external errors and score envelope "
                "coverage; no envelope rescaling, realization removal, threshold "
                "tuning, or parameter modification"
            ),
        }

        print("=== Analyses ===")
        persite, undefined = persite_associations(obs)
        meta["undefined_correlations"] = undefined
        if undefined:
            print(f"  WARNING: undefined correlations recorded: {undefined}")

        # Written before any inference so an eligibility failure leaves the
        # capture- and site-level evidence on disk to investigate (plan 6.2).
        out_cols = [
            "site",
            "date",
            "obsnme",
            "member_count",
            "spread_retrieval",
            "target",
            "flux_etf",
            "etf_effective",
            "error_effective",
            "abs_error_effective",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "spread_conditioned_iqr",
            "spread_conditioned_std",
            "width90",
            "abs_error_ensemble_median",
            "covered_90",
        ]
        obs[out_cols].to_csv(os.path.join(out_dir, "uncertainty_observations.csv"), index=False)
        persite.to_csv(os.path.join(out_dir, "uncertainty_persite.csv"), index=False)

        eligible = persite[persite["eligible"]]
        deltas = eligible["delta_rho"].to_numpy(dtype=float)
        if not np.isfinite(deltas).all():
            _fail(GateError("undefined delta_rho among eligible sites; investigate before use"))

        primary = bootstrap_site_median(deltas, n_boot=n_boot)
        rho_cond_ci = bootstrap_site_median(
            eligible["rho_conditioned_iqr"].to_numpy(dtype=float), n_boot=n_boot
        )
        rho_retr_ci = bootstrap_site_median(
            eligible["rho_retrieval"].to_numpy(dtype=float), n_boot=n_boot
        )
        sens_std = bootstrap_site_median(
            eligible["delta_rho_std"].to_numpy(dtype=float), n_boot=n_boot
        )
        sens_w90 = bootstrap_site_median(
            eligible["delta_rho_width90"].to_numpy(dtype=float), n_boot=n_boot
        )
        sens_ensmed = bootstrap_site_median(
            eligible["delta_rho_err_ensmed"].to_numpy(dtype=float), n_boot=n_boot
        )
        site_cov_ci = bootstrap_site_median(
            persite["coverage_90"].to_numpy(dtype=float), n_boot=n_boot
        )
        pooled = pooled_site_bootstrap(obs, n_boot=n_boot)
        quintiles = quintile_diagnostics(obs)
        quintiles.to_csv(os.path.join(out_dir, "uncertainty_quintiles.csv"), index=False)

        # Gate G: recompute the bootstrap blocks under the same seed and require
        # exact reproduction of every summary and interval.
        primary_rerun = bootstrap_site_median(deltas, n_boot=n_boot)
        pooled_rerun = pooled_site_bootstrap(obs, n_boot=n_boot)
        if primary_rerun != primary or pooled_rerun != pooled:
            raise GateError("Gate G failed: bootstrap results not reproducible under fixed seed")
        meta["gates"]["G"] = {
            "status": "PASS",
            "statement": (
                f"primary and pooled bootstrap blocks recomputed under seed {SEED} and "
                "reproduced exactly; no time- or order-dependent randomness"
            ),
        }

        outcome = classify_outcome(primary, rho_cond_ci)
        meta["outcome"] = outcome
        print(
            f"  primary median delta_rho = {primary['point']:+.3f} "
            f"[{primary['ci_lo']:+.3f}, {primary['ci_hi']:+.3f}] over {primary['n_sites']} sites"
            f" -> outcome {outcome['primary_outcome']}"
            + (" +D" if outcome["outcome_d_applies"] else "")
        )
        print(
            f"  pooled 90% envelope coverage = {pooled['pooled_coverage_90']['point']:.3f} "
            f"[{pooled['pooled_coverage_90']['ci_lo']:.3f}, "
            f"{pooled['pooled_coverage_90']['ci_hi']:.3f}]"
        )

        summary_rows = [
            {"metric": "median_delta_rho", **primary, "note": "primary estimand (6.2)"},
            {"metric": "median_rho_retrieval", **rho_retr_ci},
            {"metric": "median_rho_conditioned_iqr", **rho_cond_ci},
            {"metric": "median_delta_rho_std", **sens_std, "note": "sensitivity"},
            {"metric": "median_delta_rho_width90", **sens_w90, "note": "sensitivity"},
            {"metric": "median_delta_rho_err_ensmed", **sens_ensmed, "note": "sensitivity"},
            {"metric": "pooled_rho_retrieval", **pooled["pooled_rho_retrieval"]},
            {"metric": "pooled_rho_conditioned_iqr", **pooled["pooled_rho_conditioned_iqr"]},
            {"metric": "pooled_rho_difference", **pooled["pooled_rho_difference"]},
            {
                "metric": "pooled_rho_retrieval_vs_conditioned",
                "point": pooled["pooled_rho_retrieval_vs_conditioned"],
            },
            {"metric": "pooled_coverage_90", **pooled["pooled_coverage_90"]},
            {"metric": "median_site_coverage", **site_cov_ci},
            {"metric": "median_width90", "point": float(obs["width90"].median())},
            {"metric": "median_iqr", "point": float(obs["spread_conditioned_iqr"].median())},
            {
                "metric": "median_spread_retrieval",
                "point": float(obs["spread_retrieval"].median()),
            },
            {"metric": "median_abs_error", "point": float(obs["abs_error_effective"].median())},
        ]
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "uncertainty_summary.csv"), index=False
        )

        # Optional section 6.5: a failure here must not cost the required
        # deliverables above (plan: "if time permits", descriptive only).
        contraction_summary = None
        try:
            print("=== Section 6.5: prior-to-conditioned contraction ===")
            par_prior = pd.read_csv(CANONICAL["par_prior"], index_col=0)
            par_prior.index = par_prior.index.astype(str)
            par_prior_n = select_numbered_realizations(par_prior, expected=None)
            par_final_n = select_numbered_realizations(par_all)
            contraction_persite, contraction_summary = contraction_table(
                par_prior_n, par_final_n, fids
            )
            contraction_persite.to_csv(
                os.path.join(out_dir, "uncertainty_param_contraction_persite.csv"), index=False
            )
            contraction_summary.to_csv(
                os.path.join(out_dir, "uncertainty_param_contraction.csv"), index=False
            )
            meta["section_6_5"] = {"status": "PASS"}
        except Exception as exc:
            meta["section_6_5"] = {"status": "FAILED", "error": str(exc)}
            print(f"  WARNING: section 6.5 failed (optional, continuing): {exc}")

        try:
            make_figure(
                obs,
                quintiles,
                os.path.join(out_dir, "uncertainty_figure.png"),
                os.path.join(out_dir, "uncertainty_figure.pdf"),
            )
            meta["figure"] = {"status": "PASS"}
        except Exception as exc:
            meta["figure"] = {"status": "FAILED", "error": str(exc)}
            print(f"  WARNING: figure generation failed (continuing): {exc}")

        meta["status"] = "PASS"
        meta["runtime_s"] = time.time() - t0
        meta["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        meta["finished"] = datetime.now(tz=UTC).isoformat()
        with open(os.path.join(out_dir, "uncertainty_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        write_results_note(RESULTS_NOTE, meta, summary_rows, contraction_summary)
        print(f"\nAll gates passed. Outputs in {out_dir}")
        print(f"Results note: {RESULTS_NOTE}")
        print(f"Runtime {meta['runtime_s']:.0f} s, peak RSS {meta['peak_rss_mb']:.0f} MB")

    except (GateError, ValueError) as exc:
        _fail(exc)
    finally:
        if container is not None:
            container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help="bootstrap replicates (non-canonical if changed; recorded in metadata)",
    )
    args = parser.parse_args()
    main(out_dir=args.out_dir, n_boot=args.n_boot)

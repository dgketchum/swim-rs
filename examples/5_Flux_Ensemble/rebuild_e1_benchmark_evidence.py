"""Rebuild the E1 OpenET benchmark evidence from May v2.1, ETf-first.

Repairs two compounding defects in the frozen E1 evidence: (1) the daily
OpenET benchmark was built by directly interpolating sparse capture-date ET
in time (invalid — interp(ETf·ETo) != interp(ETf)·ETo under varying ETo);
(2) the evaluators read the January capture set (``openet_flux/``) instead
of May v2.1 (``openet_flux_2pt1/`` + masters in ``flux_2pt1/``).

Evaluation-only. Reads ONLY: the archived Run 22
``site_daily_timeseries/{fid}.csv`` (``date, swim_ET, flux_ET`` — the
archived ``eto`` column is raw gridMET, ancillary, and FORBIDDEN for
reconstruction), the May v2.1 per-site files and masters, the pinned
extracted ETo ``data/openet_refet/openet_eto.csv`` (the sole benchmark ETo
basis), and — as read-only provenance — the frozen run container
(``meteorology/gridmet/eto_corr`` for the 0a identity gate) and its
manifest. Never imports or re-runs the calibrated model.

Construction (binding): ETf_i = ET_i / ETo_i at captures; daily ET =
interp(ETf) × ETo under the Volk et al. (2024) ±32-day temporal-support
rule with openet-core semantics (swimrs.calibrate.benchmark). Pairing
happens only AFTER reconstruction. Daily/monthly pairing and metric
semantics replicate evaluate.py exactly so the outputs keep the frozen
``e2_*`` schemas; the monthly evidence is an independent product comparison
(May full-month totals), NOT sums of the daily reconstruction.

Gates (fail-not-warn): G-SOURCE (May allowlist by path, per-file sha256,
per-site values verified against the May master), G-FLUX (archived flux_ET
≡ master Closed on the frozen calendar), G-EPOCH (captures and scored dates
inside the extracted-ETo support), and per site × series G-ANCHORS /
G-IDENT / G-SUPPORT / G-PARTITION / G-PAIR. ``--verify`` recomputes and
compares against the pinned metadata (G-VALUES).

Usage:
    uv run python rebuild_e1_benchmark_evidence.py \
        --output-dir /data/ssd1/swim/5_Flux_Ensemble/results/run22/e1_rebuild_scratch
    uv run python rebuild_e1_benchmark_evidence.py --output-dir ... --verify
    uv run python rebuild_e1_benchmark_evidence.py --output-dir ... --emit-test-fixture
"""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from swimrs.calibrate.benchmark import (
    VOLK_WINDOW_DAYS,
    BenchmarkConstructionError,
    reconstruct_daily_benchmark,
)
from swimrs.calibrate.flux_utils import full_month_paired_sums, passes_site_minimum

# Mirrors evaluate.py (kept local so this script never imports the model stack)
OPEN_SOURCE_MODELS = ["geesebal", "ptjpl", "ssebop", "sims", "eemetric", "disalexi"]
VOLK_COLUMN_MAP = {
    "GEESEBAL_3x3": "geesebal",
    "PTJPL_3x3": "ptjpl",
    "SSEBOP_3x3": "ssebop",
    "SIMS_3x3": "sims",
    "EEMETRIC_3x3": "eemetric",
    "DISALEXI_3x3": "disalexi",
}
EXCLUDED_SITES = {"MB_Pch"}
# May master column names (bare) paired with the split-file `_3x3` columns
MASTER_SERIES = ["EEMETRIC", "GEESEBAL", "PTJPL", "SSEBOP", "SIMS", "DISALEXI", "ensemble_mean"]

ETO_SOURCE = "openet_refet/openet_eto.csv"
ETO_EE_ASSET = "projects/openet/assets/reference_et/conus/gridmet/daily/v1"
IDENTITY_TOL = 1e-10
VALUE_TOL = 1e-12
ETO_IDENTITY_TOL = 1e-9
SCHEMA_VERSION = "1.1"

DATA_DIR_DEFAULT = "/data/ssd1/swim/5_Flux_Ensemble/data"
RUN_DIR_DEFAULT = "/data/ssd1/swim/5_Flux_Ensemble/results/run22"
REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERSEDED_FINAL_DIR = REPO_ROOT / "paper" / "data" / "final"
SUPERSEDED_FILES = [
    "e2_primary_daily_site_metrics.csv",
    "e2_primary_monthly_site_metrics.csv",
    "e2_primary_performance_summary.csv",
    "e2_primary_daily_exclusion_ledger.csv",
    "e2_primary_monthly_exclusion_ledger.csv",
    "e2_temporal_summary.csv",
    "e2_temporal_paired_deltas.csv",
    "e2_evidence_metadata.json",
]

BENCHMARK_DESIGN = (
    "OpenET-method temporal benchmark using a common OpenET bias-corrected gridMET ETo basis"
)
TEMPORAL_RULE = (
    "site-series ETf reconstruction using the Volk et al. temporal-support rule "
    "(openet-core semantics; NOT a native-product reproduction — the native daily "
    "product interpolates at pixel level and uses CIMIS reference ET in California)"
)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def _jsonable(v):
    """Row value → JSON-safe (NaN → null so expected.json is strict JSON)."""
    if isinstance(v, np.floating | float):
        v = float(v)
        return None if np.isnan(v) else v
    if isinstance(v, np.integer | int):
        return int(v)
    return v


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo_dir):
    def run(args):
        return subprocess.run(
            args, cwd=repo_dir, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_paths": len(run(["git", "status", "--porcelain"]).splitlines()),
    }


class SourceLedger:
    """G-SOURCE: May-source allowlist by resolved path, with per-file sha256.

    Any read outside the allowlist is a hard failure — the January
    ``openet_flux/`` directories are excluded by construction, and are
    additionally rejected explicitly by path component.
    """

    def __init__(self):
        self.allowed = set()
        self.hashes = {}

    def allow_dir(self, directory, pattern="*.csv"):
        for p in sorted(Path(directory).glob(pattern)):
            self.allowed.add(p.resolve())

    def allow_file(self, path):
        self.allowed.add(Path(path).resolve())

    def read_csv(self, path, **kwargs):
        p = Path(path).resolve()
        parts = p.parts
        if "openet_flux" in parts:
            raise BenchmarkConstructionError(f"G-SOURCE: January OpenET source rejected: {p}")
        if p not in self.allowed:
            raise BenchmarkConstructionError(f"G-SOURCE: {p} is not on the May-source allowlist")
        if str(p) not in self.hashes:
            self.hashes[str(p)] = sha256_file(p)
        return pd.read_csv(p, **kwargs)


def calc_metrics(obs, mod):
    """R2 (NSE), Pearson r, RMSE, bias, KGE — verbatim evaluate.py semantics."""
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[mask], mod[mask]
    if len(obs) < 10:
        return {
            "n": len(obs),
            "r2": np.nan,
            "r": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "kge": np.nan,
        }
    r, _ = stats.pearsonr(obs, mod)
    r2 = r2_score(obs, mod)
    rmse = np.sqrt(mean_squared_error(obs, mod))
    bias = float((mod - obs).mean())
    alpha = np.std(mod) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(mod) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return {"n": len(obs), "r2": r2, "r": r, "rmse": rmse, "bias": bias, "kge": kge}


def eto_wide_to_frame(wide):
    """site_id-wide ETo CSV frame → dates × sites, dates ascending."""
    wide = wide.copy()
    wide.columns = pd.to_datetime(wide.columns, format="%Y%m%d")
    return wide.T.sort_index()


def load_openet_eto_wide(path):
    """Pinned extracted ETo (dates × sites). The sole benchmark ETo basis."""
    return eto_wide_to_frame(pd.read_csv(path, index_col="site_id"))


def audit_0a_eto_identity(container_path, openet_eto):
    """0a: openet_eto.csv must equal the frozen container's eto_corr.

    The archived site_daily_timeseries ``eto`` column is raw gridMET
    (archive_run.py exported meteorology/gridmet/eto while the model consumed
    eto_corr) and is forbidden for reconstruction; this gate pins the valid
    basis against the container the run actually used.
    """
    z = zarr.open(str(container_path), mode="r")
    uids = list(z["geometry/uid"][:])
    dates = pd.DatetimeIndex(z["time/daily"][:])
    eto_corr = pd.DataFrame(z["meteorology/gridmet/eto_corr"][:], index=dates, columns=uids)
    sites = sorted(set(uids) & set(openet_eto.columns))
    common = dates.intersection(openet_eto.index)
    a = eto_corr.loc[common, sites].astype("float64")
    b = openet_eto.loc[common, sites].astype("float64")
    both = a.notna() & b.notna()
    max_abs = float(np.nanmax((a - b).where(both).abs().values))
    mismatch = int((a.notna() ^ b.notna()).values.sum())
    if max_abs > ETO_IDENTITY_TOL or mismatch:
        raise BenchmarkConstructionError(
            f"0a: openet_eto.csv vs container eto_corr max |diff| {max_abs:.3e}, "
            f"{mismatch} NaN-mask mismatches — stop and report"
        )
    return {
        "container": str(container_path),
        "n_sites_compared": len(sites),
        "n_values_compared": int(both.values.sum()),
        "max_abs_diff": max_abs,
        "tolerance": ETO_IDENTITY_TOL,
    }


def validate_site_against_master(fid, split_df, master):
    """0c/G-SOURCE per-site value check against a May master (daily or monthly).

    Master-only dates are permitted ONLY when every OpenET series is NaN
    there (flux-only rows / calendar padding); any finite OpenET value on a
    master-only date, any split-only date, any NaN-mask mismatch, or any
    value difference above 1e-12 is a hard failure.
    """
    ms = master[master["SITE_ID"] == fid].set_index("DATE").sort_index()
    if ms.empty:
        raise BenchmarkConstructionError(f"G-SOURCE {fid}: absent from May master")
    sp = split_df.sort_index()
    extra = sp.index.difference(ms.index)
    if len(extra):
        raise BenchmarkConstructionError(f"G-SOURCE {fid}: {len(extra)} split dates not in master")
    missing = ms.index.difference(sp.index)
    if len(missing):
        gone = ms.loc[missing, [c for c in MASTER_SERIES if c in ms.columns]]
        informative = gone.notna().any(axis=1)
        if informative.any():
            raise BenchmarkConstructionError(
                f"G-SOURCE {fid}: {int(informative.sum())} master-only dates "
                "with finite OpenET values"
            )
        ms = ms.drop(index=missing)
    max_d = 0.0
    for series in MASTER_SERIES:
        split_col = f"{series}_3x3"
        if split_col not in sp.columns or series not in ms.columns:
            raise BenchmarkConstructionError(
                f"G-SOURCE {fid}: column missing: {split_col}/{series}"
            )
        sv = sp[split_col].astype("float64")
        mv = ms[series].astype("float64")
        if not (sv.isna() == mv.isna()).all():
            raise BenchmarkConstructionError(f"G-SOURCE {fid}: NaN mask mismatch on {split_col}")
        d = (sv - mv).abs().max()
        if pd.notna(d):
            max_d = max(max_d, float(d))
            if d > VALUE_TOL:
                raise BenchmarkConstructionError(
                    f"G-SOURCE {fid}: {split_col} max |diff| {d:.3e} > {VALUE_TOL}"
                )
    return max_d


def gate_g_flux(fid, flux_archived, master_daily):
    """G-FLUX: archived flux_ET ≡ May master Closed on the frozen calendar."""
    ms = master_daily[master_daily["SITE_ID"] == fid].set_index("DATE")["Closed"].sort_index()
    if ms.empty:
        raise BenchmarkConstructionError(f"G-FLUX {fid}: absent from May daily master")
    aligned = ms.reindex(flux_archived.index)
    mismatch = int((aligned.notna() ^ flux_archived.notna()).sum())
    if mismatch:
        raise BenchmarkConstructionError(f"G-FLUX {fid}: missingness mismatch on {mismatch} dates")
    both = aligned.notna() & flux_archived.notna()
    max_d = 0.0
    if both.any():
        max_d = float((aligned[both] - flux_archived[both]).abs().max())
        if max_d > VALUE_TOL:
            raise BenchmarkConstructionError(f"G-FLUX {fid}: max |diff| {max_d:.3e} > {VALUE_TOL}")
    return max_d


def load_may_series(split_df):
    """Per-series sparse ET from a May daily/monthly per-site frame."""
    out = {}
    for raw_col, model_name in VOLK_COLUMN_MAP.items():
        if raw_col in split_df.columns:
            out[model_name] = split_df[raw_col].astype(float)
    if "ensemble_mean_3x3" in split_df.columns:
        out["ensemble"] = split_df["ensemble_mean_3x3"].astype(float)
    return out


def reconstruct_site_series(may_daily_df, site_eto, fid):
    """ETf-first reconstruction for ensemble + every member (per-series support)."""
    recons = {}
    for name, sparse in load_may_series(may_daily_df).items():
        if not sparse.notna().any():
            continue
        recons[name] = reconstruct_daily_benchmark(
            capture_series=sparse,
            capture_space="et",
            eto=site_eto,
            eto_name=ETO_SOURCE,
            label=f"{fid}:{name}",
        )
    return recons


def daily_site_row(fid, swim_et, flux_et, recons, support_rows=None, member_days=None):
    """One daily metrics row, evaluate.py openet_source='volk' semantics.

    Pairing happens only after reconstruction: the ensemble defines the
    primary paired mask (flux + SWIM + ensemble finite); each member is
    re-paired on its own support with SWIM re-scored on the same days.
    Also enforces the per-series G-SUPPORT / G-PARTITION / G-PAIR gates and
    (optionally) appends support-ledger rows.
    """
    common = swim_et.index.intersection(flux_et.index)
    if len(common) < 10:
        return None
    obs = flux_et.loc[common].values
    swim_vals = swim_et.loc[common].values

    ens_vals = np.full(len(common), np.nan)
    if "ensemble" in recons:
        ens_vals = recons["ensemble"].daily_et.reindex(common).values

    paired_mask = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(ens_vals)
    n_paired = int(paired_mask.sum())

    row = {"fid": fid, "n": n_paired}
    if n_paired >= 10:
        m = calc_metrics(obs[paired_mask], swim_vals[paired_mask])
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_swim"] = m[k]
        m = calc_metrics(obs[paired_mask], ens_vals[paired_mask])
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_ensemble"] = m[k]
    else:
        # evaluate.py sets these interleaved in one loop — keep key order identical
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_swim"] = np.nan
            row[f"{k}_ensemble"] = np.nan

    scored_by_series = {"ensemble": common[paired_mask]}
    for model_name in OPEN_SOURCE_MODELS:
        # An all-NaN May series has no reconstruction; evaluate.py scores it as
        # zero paired days — same values, same key order (r2_swim_vs first)
        model_vals = (
            recons[model_name].daily_et.reindex(common).values
            if model_name in recons
            else np.full(len(common), np.nan)
        )
        model_paired = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(model_vals)
        scored_by_series[model_name] = common[model_paired]
        if member_days is not None:
            member_days.setdefault(model_name, {})[fid] = int(model_paired.sum())
        if model_paired.sum() >= 10:
            m = calc_metrics(obs[model_paired], model_vals[model_paired])
            row[f"r2_swim_vs_{model_name}"] = r2_score(obs[model_paired], swim_vals[model_paired])
        else:
            m = {"r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan, "kge": np.nan}
            row[f"r2_swim_vs_{model_name}"] = np.nan
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_{model_name}"] = m[k]

    # Per-series gates on the scored record (G-SUPPORT, G-PARTITION, G-PAIR,
    # G-EPOCH scored side); support ledger rows per site × series
    for name, recon in recons.items():
        scored = scored_by_series.get(name, pd.DatetimeIndex([]))
        bench = recon.daily_et.reindex(scored)
        n_outside = int((~np.isfinite(bench.values)).sum())
        if n_outside:
            raise BenchmarkConstructionError(
                f"G-SUPPORT {fid}:{name}: {n_outside} scored dates outside support"
            )
        classes = recon.support_class.reindex(scored)
        n_cap = int((classes == "capture").sum())
        n_int = int((classes == "interpolated").sum())
        n_flat = int((classes == "flat_fill").sum())
        if n_cap + n_int + n_flat != len(scored):
            raise BenchmarkConstructionError(
                f"G-PARTITION {fid}:{name}: capture+interpolated+flat_fill != scored"
            )
        if len(scored) and scored.min() < recon.daily_et.index.min():
            raise BenchmarkConstructionError(
                f"G-EPOCH {fid}:{name}: scored date before ETo support"
            )
        if support_rows is not None:
            cap_eto = recon.capture_et / recon.capture_etf
            support_rows.append(
                {
                    "fid": fid,
                    "series": name,
                    "n_captures": recon.n_captures,
                    "first_capture": recon.capture_dates.min().date().isoformat(),
                    "last_capture": recon.capture_dates.max().date().isoformat(),
                    "support_start": recon.support_start.date().isoformat(),
                    "support_end": recon.support_end.date().isoformat(),
                    "n_interpolated_days": recon.n_interpolated_days,
                    "n_flat_filled_days": recon.n_flat_filled_days,
                    "n_unsupported_days": recon.n_unsupported_days,
                    "max_capture_gap_days": recon.max_capture_gap_days,
                    "identity_max_abs_err": recon.identity_max_abs_err,
                    "min_capture_eto": float(cap_eto.min()),
                    "n_captures_without_flux": int(
                        (~np.isfinite(flux_et.reindex(recon.capture_dates).values)).sum()
                    ),
                    "n_scored": len(scored),
                    "n_scored_capture": n_cap,
                    "n_scored_interpolated": n_int,
                    "n_scored_flat_fill": n_flat,
                    "n_scored_outside_support": n_outside,
                    "eto_source": ETO_SOURCE,
                    "window_days": recon.window_days,
                }
            )
    return row


def monthly_site_row(fid, swim_et, flux_et, may_monthly_df):
    """One monthly metrics row, evaluate_monthly semantics (May full-month totals)."""
    daily_common = swim_et.index.intersection(flux_et.index)
    if len(daily_common) < 30:
        return None
    flux_daily = flux_et.loc[daily_common]
    swim_monthly, flux_monthly = full_month_paired_sums(swim_et, flux_daily)

    volk_monthly = load_may_series(may_monthly_df)
    ens_monthly = volk_monthly.get("ensemble")
    if ens_monthly is not None:
        all_idx = flux_monthly.index
        ens_on_idx = ens_monthly.reindex(all_idx)
        paired_mask = (
            flux_monthly.notna() & swim_monthly.reindex(all_idx).notna() & ens_on_idx.notna()
        )
        paired_months = all_idx[paired_mask]
    else:
        paired_months = swim_monthly.index.intersection(flux_monthly.index)

    n_paired = len(paired_months)
    if n_paired < 6:
        return None

    obs = flux_monthly.loc[paired_months].values
    row = {"fid": fid, "n": n_paired}
    m = calc_metrics(obs, swim_monthly.reindex(paired_months).values)
    for k in ["r2", "r", "rmse", "bias", "kge"]:
        row[f"{k}_swim"] = m[k]

    swim_on_paired = swim_monthly.reindex(paired_months).values
    for model_name in OPEN_SOURCE_MODELS + ["ensemble"]:
        if model_name not in volk_monthly:
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_{model_name}"] = np.nan
            continue
        model_vals = volk_monthly[model_name].reindex(paired_months).values
        model_valid = np.isfinite(model_vals) & np.isfinite(obs) & np.isfinite(swim_on_paired)
        if model_valid.sum() >= 6:
            m = calc_metrics(obs[model_valid], model_vals[model_valid])
        else:
            m = {"r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan, "kge": np.nan}
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_{model_name}"] = m[k]
    return row


def build_performance_summary(daily_df, monthly_df):
    """Existing e2_primary_performance_summary schema + benchmark_construction."""
    rows = []
    d_both = daily_df[daily_df["r2_swim"].notna() & daily_df["r2_ensemble"].notna()]
    for model in ["swim", "ensemble"]:
        rows.append(
            {
                "scale": "daily",
                "model": model,
                "n_sites": len(d_both),
                "n_paired_observations": int(d_both["n"].sum()),
                "nse_median": float(d_both[f"r2_{model}"].median()),
                "kge_median": float(d_both[f"kge_{model}"].median()),
                "rmse_median": float(d_both[f"rmse_{model}"].median()),
                "mbe_median": float(d_both[f"bias_{model}"].median()),
                "rmse_unit": "mm/d",
                "mbe_unit": "mm/d",
                "benchmark_construction": "etf_first_volk_window",
            }
        )
    m_both = monthly_df[monthly_df["r2_swim"].notna() & monthly_df["r2_ensemble"].notna()]
    for model in ["swim", "ensemble"]:
        rows.append(
            {
                "scale": "monthly",
                "model": model,
                "n_sites": len(m_both),
                "n_paired_observations": int(m_both["n"].sum()),
                "nse_median": float(m_both[f"r2_{model}"].median()),
                "kge_median": float(m_both[f"kge_{model}"].median()),
                "rmse_median": float(m_both[f"rmse_{model}"].median()),
                "mbe_median": float(m_both[f"bias_{model}"].median()),
                "rmse_unit": "mm/month",
                "mbe_unit": "mm/month",
                "benchmark_construction": "independent_full_month_openet_totals_v2pt1",
            }
        )
    return pd.DataFrame(rows), d_both, m_both


def build_member_summary(daily_df, member_days):
    """e2_benchmark_member_daily_nse.csv: per-series daily medians + matched SWIM."""
    rows = []
    d_both = daily_df[daily_df["r2_swim"].notna() & daily_df["r2_ensemble"].notna()]
    rows.append(
        {
            "series": "ensemble_mean",
            "n_sites": len(d_both),
            "n_paired_site_days": int(d_both["n"].sum()),
            "median_nse": float(d_both["r2_ensemble"].median()),
            "median_kge": float(d_both["kge_ensemble"].median()),
            "median_rmse": float(d_both["rmse_ensemble"].median()),
            "median_mbe": float(d_both["bias_ensemble"].median()),
            "matched_swim_median_nse": float(d_both["r2_swim"].median()),
        }
    )
    for member in OPEN_SOURCE_MODELS:
        sub = daily_df[daily_df[f"r2_{member}"].notna()]
        days = member_days.get(member, {})
        rows.append(
            {
                "series": member,
                "n_sites": len(sub),
                "n_paired_site_days": int(sum(days.get(f, 0) for f in sub["fid"])),
                "median_nse": float(sub[f"r2_{member}"].median()),
                "median_kge": float(sub[f"kge_{member}"].median()),
                "median_rmse": float(sub[f"rmse_{member}"].median()),
                "median_mbe": float(sub[f"bias_{member}"].median()),
                "matched_swim_median_nse": float(sub[f"r2_swim_vs_{member}"].median()),
            }
        )
    return pd.DataFrame(rows)


def rebuild(args):
    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    ts_dir = run_dir / "archive" / "6_evaluation" / "site_daily_timeseries"
    may_daily_dir = data_dir / "openet_flux_2pt1" / "daily_data"
    may_monthly_dir = data_dir / "openet_flux_2pt1" / "monthly_data"
    daily_master_csv = data_dir / "flux_2pt1" / "daily_2pt1_paired_data.csv"
    monthly_master_csv = data_dir / "flux_2pt1" / "monthly_2pt1_paired_data.csv"
    openet_eto_csv = Path(args.openet_eto_csv)
    container_path = Path(args.container)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = SourceLedger()
    ledger.allow_dir(ts_dir)
    ledger.allow_dir(may_daily_dir)
    ledger.allow_dir(may_monthly_dir)
    for f in (daily_master_csv, monthly_master_csv, openet_eto_csv):
        ledger.allow_file(f)

    openet_eto = eto_wide_to_frame(ledger.read_csv(openet_eto_csv, index_col="site_id"))

    print("0a: ETo identity gate (openet_eto.csv vs container eto_corr)...")
    audit_0a = audit_0a_eto_identity(container_path, openet_eto)
    audit_0a["eto_start"] = str(openet_eto.index.min().date())
    audit_0a["eto_end"] = str(openet_eto.index.max().date())
    print(
        f"  PASS: max |diff| {audit_0a['max_abs_diff']:.3e} over "
        f"{audit_0a['n_values_compared']} values"
    )

    master_daily = ledger.read_csv(daily_master_csv, parse_dates=["DATE"])
    master_monthly = ledger.read_csv(monthly_master_csv, parse_dates=["DATE"])

    # Site universe: archived evaluation record ∩ May daily data, minus policy
    # exclusions — cohorts are then reconstructed by the prespecified gates
    archived = {p.stem for p in sorted(ts_dir.glob("*.csv"))}
    may_sites = {p.stem for p in sorted(may_daily_dir.glob("*.csv"))}
    universe = sorted((archived & may_sites) - EXCLUDED_SITES)
    print(
        f"Site universe: {len(universe)} sites "
        f"(archived {len(archived)}, May daily {len(may_sites)})"
    )

    daily_rows, monthly_rows, support_rows = [], [], []
    excluded_daily, excluded_monthly = [], []
    member_days = {}
    gflux_max = 0.0
    gsource_max = 0.0

    for fid in universe:
        ts = ledger.read_csv(
            ts_dir / f"{fid}.csv",
            usecols=["date", "swim_ET", "flux_ET"],
            index_col="date",
            parse_dates=True,
        )
        swim_et = ts["swim_ET"].astype(float)
        flux_et = ts["flux_ET"].astype(float)

        gflux_max = max(gflux_max, gate_g_flux(fid, flux_et, master_daily))

        may_daily = ledger.read_csv(
            may_daily_dir / f"{fid}.csv", index_col="DATE", parse_dates=True
        )
        gsource_max = max(gsource_max, validate_site_against_master(fid, may_daily, master_daily))

        if not flux_et.notna().any():
            excluded_daily.append({"site": fid, "reason": "no_flux_data"})
            excluded_monthly.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            excluded_daily.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            excluded_monthly.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue

        if fid not in openet_eto.columns:
            raise BenchmarkConstructionError(f"{fid}: absent from {openet_eto_csv}")
        site_eto = openet_eto[fid].astype("float64")

        # G-ANCHORS: every finite May capture anchors (helper hard-fails on a
        # capture without finite positive ETo); count identity checked here
        recons = reconstruct_site_series(may_daily, site_eto, fid)
        for name, recon in recons.items():
            sparse = load_may_series(may_daily)[name]
            n_finite = int(np.isfinite(sparse.values).sum())
            if recon.n_captures != n_finite:
                raise BenchmarkConstructionError(
                    f"G-ANCHORS {fid}:{name}: {n_finite} finite captures, "
                    f"{recon.n_captures} anchored"
                )

        row = daily_site_row(
            fid, swim_et, flux_et, recons, support_rows=support_rows, member_days=member_days
        )
        if row is None:
            excluded_daily.append({"site": fid, "reason": "insufficient_daily_overlap"})
        else:
            daily_rows.append(row)

        may_monthly_path = may_monthly_dir / f"{fid}.csv"
        if may_monthly_path.exists():
            may_monthly = ledger.read_csv(may_monthly_path, index_col="DATE", parse_dates=True)
            gsource_max = max(
                gsource_max, validate_site_against_master(fid, may_monthly, master_monthly)
            )
        else:
            may_monthly = pd.DataFrame()
        mrow = monthly_site_row(fid, swim_et, flux_et, may_monthly)
        if mrow is None:
            excluded_monthly.append({"site": fid, "reason": "below_monthly_output_floor_6mo"})
        else:
            monthly_rows.append(mrow)

    print(f"G-FLUX: PASS ({len(universe)} sites, max |diff| {gflux_max:.3e})")
    print(f"G-SOURCE per-site values: PASS (max |diff| vs master {gsource_max:.3e})")

    daily_df = pd.DataFrame(daily_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    support_df = pd.DataFrame(support_rows)

    # G-EPOCH (capture side): every capture inside the extracted-ETo support
    eto_start = openet_eto.index.min()
    bad_epoch = support_df[pd.to_datetime(support_df["first_capture"]) < eto_start]
    if len(bad_epoch):
        raise BenchmarkConstructionError(
            f"G-EPOCH: {len(bad_epoch)} site×series with captures before {eto_start.date()}"
        )
    if int(support_df["n_scored_outside_support"].sum()) != 0:
        raise BenchmarkConstructionError("G-SUPPORT: scored rows outside support")
    ident_max = float(support_df["identity_max_abs_err"].max())
    print(
        f"G-EPOCH: PASS (earliest capture "
        f"{support_df['first_capture'].min()}, ETo starts {eto_start.date()})"
    )
    print(f"G-IDENT: PASS (max capture-identity error {ident_max:.3e})")

    summary_df, d_both, m_both = build_performance_summary(daily_df, monthly_df)
    member_df = build_member_summary(daily_df, member_days)

    outputs = {
        "e2_primary_daily_site_metrics.csv": daily_df.set_index("fid"),
        "e2_primary_monthly_site_metrics.csv": monthly_df.set_index("fid"),
        "e2_primary_performance_summary.csv": summary_df,
        "e2_primary_daily_exclusion_ledger.csv": pd.DataFrame(
            excluded_daily, columns=["site", "reason"]
        ),
        "e2_primary_monthly_exclusion_ledger.csv": pd.DataFrame(
            excluded_monthly, columns=["site", "reason"]
        ),
        "e2_benchmark_member_daily_nse.csv": member_df,
        "e2_benchmark_site_support.csv": support_df,
    }

    metadata = build_metadata(
        args,
        audit_0a,
        ledger,
        daily_df,
        monthly_df,
        summary_df,
        d_both,
        m_both,
        support_df,
        excluded_daily,
        excluded_monthly,
        gflux_max,
        gsource_max,
        openet_eto_csv,
        container_path,
        run_dir,
        universe,
    )

    print("\nHeadline (reconstructed May v2.1 footing):")
    print(summary_df.to_string(index=False))
    print(
        f"\nDaily cohort: {len(d_both)} sites, {int(d_both['n'].sum())} paired days; "
        f"monthly cohort: {len(m_both)} sites, {int(m_both['n'].sum())} paired months"
    )
    return outputs, metadata, daily_df, monthly_df


def build_metadata(
    args,
    audit_0a,
    ledger,
    daily_df,
    monthly_df,
    summary_df,
    d_both,
    m_both,
    support_df,
    excluded_daily,
    excluded_monthly,
    gflux_max,
    gsource_max,
    openet_eto_csv,
    container_path,
    run_dir,
    universe,
):
    superseded = {}
    for name in SUPERSEDED_FILES:
        p = SUPERSEDED_FINAL_DIR / name
        if p.exists():
            superseded[name] = sha256_file(p)
    manifest_path = run_dir / "archive" / "1_provenance" / "container_manifest.json"
    per_series_counts = support_df.groupby("series")["n_scored"].sum().astype(int).to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E2",
        "status": "rebuilt_awaiting_review",
        "rebuilt_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "internal_archive_id": "run22",
        "git": git_state(str(REPO_ROOT)),
        "benchmark_construction": {
            "design": BENCHMARK_DESIGN,
            "temporal_rule": TEMPORAL_RULE,
            "equations": [
                "ETf_i = ET_i / ETo_i (May v2.1 capture dates)",
                "ET_t = interp(ETf)_t * ETo_t (Volk +/-32-day window, "
                "openet-core semantics: linear with both anchors, one-sided "
                "flat fill within the window, NaN otherwise)",
            ],
            "citations": [
                "https://etdata.org/methods/",
                "Volk et al. 2024, Nature Water, "
                "https://www.nature.com/articles/s44221-023-00181-7",
            ],
            "eto_basis": {
                "path": str(openet_eto_csv),
                "sha256": ledger.hashes[str(openet_eto_csv.resolve())],
                "ee_asset": ETO_EE_ASSET,
                "identity_vs_container_eto_corr": audit_0a,
                "archived_eto_column": "raw gridMET, ancillary only — FORBIDDEN "
                "for benchmark reconstruction",
            },
            "window_days": VOLK_WINDOW_DAYS,
            "identity_tolerance": IDENTITY_TOL,
            "identity_max_abs_err_observed": float(support_df["identity_max_abs_err"].max()),
            "epoch": {
                "eto_start": audit_0a["eto_start"],
                "eto_end": audit_0a["eto_end"],
                "earliest_capture_any_series": support_df["first_capture"].min(),
                "rule": "captures and scored dates inside the extracted-ETo support; "
                "no raw-gridMET backfill",
            },
            "daily_scope_note": "the common-gridMET design applies to the "
            "reconstructed DAILY benchmark only",
        },
        "monthly_independence": (
            "May monthly totals are an independent product comparison "
            "(independent_full_month_openet_totals_v2pt1), NOT sums of the "
            "daily reconstruction"
        ),
        "source_data": {
            "may_daily_dir": str(Path(args.data_dir) / "openet_flux_2pt1" / "daily_data"),
            "may_monthly_dir": str(Path(args.data_dir) / "openet_flux_2pt1" / "monthly_data"),
            "may_masters": [
                str(Path(args.data_dir) / "flux_2pt1" / "daily_2pt1_paired_data.csv"),
                str(Path(args.data_dir) / "flux_2pt1" / "monthly_2pt1_paired_data.csv"),
            ],
            "rejected_sources": [
                str(Path(args.data_dir) / "openet_flux" / "daily_data"),
                str(Path(args.data_dir) / "openet_flux" / "monthly_data"),
            ],
            "rejected_reason": "January capture set — superseded by May v2.1",
            "input_sha256": dict(sorted(ledger.hashes.items())),
        },
        "provenance_inputs": {
            "frozen_container": str(container_path),
            "container_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
            },
            "archived_timeseries_columns_read": ["date", "swim_ET", "flux_ET"],
        },
        "scientific_configuration": {
            "site_universe": len(universe),
            "sites_passing_flux_site_minimum": len(universe)
            - sum(
                1
                for e in excluded_daily
                if e["reason"] in ("no_flux_data", "below_site_minimum_90d_3mo")
            ),
            "daily_evaluation_sites": int(len(d_both)),
            "daily_paired_observations": int(d_both["n"].sum()),
            "median_daily_observations_per_site": float(d_both["n"].median()),
            "primary_monthly_output_sites": int(len(monthly_df)),
            "primary_monthly_finite_metric_sites": int(len(m_both)),
            "primary_monthly_paired_site_months": int(m_both["n"].sum()),
        },
        "evaluation_conventions": {
            "replication_unit": "site",
            "daily_support": "SWIM-RS and the ETf-first reconstructed OpenET "
            "benchmark scored on identical flux-supported days within each site "
            "(per-member re-pairing on each member's own support)",
            "monthly_output_floor": "at least 6 paired months to retain an evaluator row",
            "monthly_metric_floor": "at least 10 paired months for finite site-level metrics",
            "aggregate_statistic": "median across sites",
            "nse_note": "Source columns named r2 are Nash-Sutcliffe efficiency "
            "(1 - SSE/SST), not squared Pearson correlation",
            "bias_sign": "modeled minus observed",
        },
        "support_reconciliation": {
            "n_scored_outside_support_total": int(support_df["n_scored_outside_support"].sum()),
            "per_series_scored_days": per_series_counts,
            "series": ["ensemble"] + OPEN_SOURCE_MODELS,
        },
        "gates": {
            "g_source": "PASS (May allowlist; per-site values vs master <= 1e-12; "
            f"observed max |diff| {gsource_max:.3e})",
            "g_flux": "PASS (archived flux_ET == May master Closed on the frozen "
            f"calendar, <= 1e-12; observed max |diff| {gflux_max:.3e})",
            "g_epoch": "PASS",
            "g_anchors": "PASS (every finite May capture anchored, per series)",
            "g_ident": "PASS (capture identity <= 1e-10, per site x series)",
            "g_support": "PASS (zero scored rows outside support, per series)",
            "g_partition": "PASS (capture+interpolated+flat_fill == scored, per series)",
            "g_pair": "PASS (single paired-date mask per site x series; flux "
            "closure-corrected, evaluation-only)",
            "g_values": "run with --verify",
            "g_boot": "overpass_decomposition.py run twice on the rebuilt cohort "
            "(seed 42, 10000 resamples)",
            "gate_a_replacement": "overpass_decomposition.py Gate A now targets "
            "the rebuilt e2_primary_daily_site_metrics.csv; the January-based "
            "daily_paired_metrics.csv identity check is superseded",
        },
        "bootstrap": {"replication_unit": "site", "resamples": 10000, "seed": 42},
        "headline": {
            r["scale"] + "_" + r["model"]: {
                "n_sites": r["n_sites"],
                "n_paired_observations": r["n_paired_observations"],
                "nse_median": r["nse_median"],
                "kge_median": r["kge_median"],
                "rmse_median": r["rmse_median"],
                "mbe_median": r["mbe_median"],
            }
            for r in summary_df.to_dict("records")
        },
        "superseded": {
            "reason": "direct-ET interpolation (construction) AND January capture "
            "source (source-version) — both defects; see benchmark_construction",
            "files_sha256_at_supersession": superseded,
        },
        "pending_downstream_consumers": [
            "paper Figure 3 package (build_figure_data.py fig03 — also consumed "
            "the erroneous archived raw-gridMET eto)",
            "fig01 helper",
            "manuscript main.md/supp.md E1 numbers",
            "LaTeX tables",
            "VALIDATION_POLICY / example READMEs",
            "Ex6 (paper E2/E3) endpoint audit — separate plan",
        ],
    }


def write_outputs(out_dir, outputs, metadata):
    out_dir = Path(out_dir)
    frozen = {}
    for name, df in outputs.items():
        path = out_dir / name
        index = name in ("e2_primary_daily_site_metrics.csv", "e2_primary_monthly_site_metrics.csv")
        df.to_csv(path, index=index)
        frozen[name] = sha256_file(path)
    metadata["frozen_artifacts"] = frozen
    with open(out_dir / "e2_evidence_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)
    print(f"\nOutputs written to {out_dir}")


def verify(out_dir, metadata):
    """G-VALUES: recomputed medians/counts must equal the pinned metadata."""
    pinned_path = Path(out_dir) / "e2_evidence_metadata.json"
    with open(pinned_path) as f:
        pinned = json.load(f)
    failures = []
    for key, block in pinned["headline"].items():
        new = metadata["headline"].get(key)
        if new is None:
            failures.append(f"{key}: missing from recomputation")
            continue
        for stat, val in block.items():
            nv = new[stat]
            if isinstance(val, float):
                if not (abs(nv - val) <= 1e-9):
                    failures.append(f"{key}.{stat}: {nv!r} != pinned {val!r}")
            elif nv != val:
                failures.append(f"{key}.{stat}: {nv!r} != pinned {val!r}")
    for block in ("scientific_configuration", "support_reconciliation"):
        for k, val in pinned[block].items():
            nv = metadata[block].get(k)
            if isinstance(val, int | float) and not isinstance(val, bool):
                if not (abs(nv - val) <= 1e-9):
                    failures.append(f"{block}.{k}: {nv!r} != pinned {val!r}")
    if failures:
        for msg in failures:
            print(f"G-VALUES FAIL: {msg}")
        raise BenchmarkConstructionError(f"G-VALUES: {len(failures)} mismatches")
    print("G-VALUES: PASS (recomputed values match pinned metadata to 1e-9)")


def emit_test_fixture(args, daily_df, monthly_df, openet_eto_csv):
    """Freeze 3 representative sites + expected values for the regression test."""
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "e1_benchmark"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    cohort = sorted(daily_df.loc[daily_df["r2_swim"].notna(), "fid"])
    fids = [cohort[0], cohort[len(cohort) // 2], cohort[-1]]

    data_dir = Path(args.data_dir)
    ts_dir = Path(args.run_dir) / "archive" / "6_evaluation" / "site_daily_timeseries"
    eto = load_openet_eto_wide(openet_eto_csv)

    def gz_write(df, name, index=True):
        with gzip.open(fixture_dir / name, "wt") as f:
            df.to_csv(f, index=index)

    expected = {"fids": fids, "eto_source": ETO_SOURCE, "window_days": VOLK_WINDOW_DAYS}
    for fid in fids:
        ts = pd.read_csv(
            ts_dir / f"{fid}.csv",
            usecols=["date", "swim_ET", "flux_ET"],
            index_col="date",
            parse_dates=True,
        )
        gz_write(ts, f"{fid}_timeseries.csv.gz")
        may_d = pd.read_csv(
            data_dir / "openet_flux_2pt1" / "daily_data" / f"{fid}.csv",
            index_col="DATE",
            parse_dates=True,
        )
        gz_write(may_d, f"{fid}_openet_daily.csv.gz")
        may_m = pd.read_csv(
            data_dir / "openet_flux_2pt1" / "monthly_data" / f"{fid}.csv",
            index_col="DATE",
            parse_dates=True,
        )
        gz_write(may_m, f"{fid}_openet_monthly.csv.gz")

        row = daily_df.loc[daily_df["fid"] == fid].iloc[0].to_dict()
        expected[fid] = {"daily": {k: _jsonable(v) for k, v in row.items()}}
        msub = monthly_df.loc[monthly_df["fid"] == fid]
        expected[fid]["monthly"] = (
            {k: _jsonable(v) for k, v in msub.iloc[0].to_dict().items()} if len(msub) else None
        )
    gz_write(eto[fids], "openet_eto_subset.csv.gz")
    with open(fixture_dir / "expected.json", "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Test fixture written to {fixture_dir} ({fids})")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the E1 OpenET benchmark evidence from May v2.1, ETf-first"
    )
    parser.add_argument("--run-dir", default=RUN_DIR_DEFAULT)
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    parser.add_argument(
        "--openet-eto-csv",
        default=str(Path(__file__).resolve().parent / "data" / ETO_SOURCE),
    )
    parser.add_argument(
        "--container",
        default=os.path.join(DATA_DIR_DEFAULT, "5_Flux_Ensemble_run22.swim"),
        help="Frozen run container (read-only provenance; 0a identity gate)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="G-VALUES: recompute and compare against the pinned "
        "metadata in --output-dir instead of writing",
    )
    parser.add_argument(
        "--emit-test-fixture",
        action="store_true",
        help="Write tests/fixtures/e1_benchmark/ for the committed regression test",
    )
    args = parser.parse_args()

    outputs, metadata, daily_df, monthly_df = rebuild(args)

    if args.verify:
        verify(args.output_dir, metadata)
        return
    write_outputs(args.output_dir, outputs, metadata)
    if args.emit_test_fixture:
        emit_test_fixture(args, daily_df, monthly_df, Path(args.openet_eto_csv))


if __name__ == "__main__":
    main()

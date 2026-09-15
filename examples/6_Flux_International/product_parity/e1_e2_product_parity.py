"""E1 (OpenET v2.1) vs E2 (globally portable) SSEBop and PT-JPL: same-date paired benchmark.

Read-only, no Earth Engine, no calibration, no forward model. Tests whether the E2
satellite products (USGS ESPA SSEBop rescaled to the grass basis; ERA5-Land-forced
``openet-ptjpl``) lose accuracy against closure-corrected flux ET relative to their
OpenET v2.1 counterparts, scored on exactly identical site-Landsat acquisition dates.

Cohort
------
* E1: the paper's Experiment 1 daily validation cohort (Ex5 Run 22; the frozen
  ``paper/data/final/e2_primary_daily_site_metrics.csv`` -- the ``e2_`` file namespace is the
  legacy label for the paper's E1, see ``e2_evidence_metadata.json``), 45 sites.
* E2: the 47-site closure-corrected pool (``archive/6_evaluation/closure_pool/
  closure_pool_sites.csv``), restricted to ``region == CONUS`` (29 sites).
* Intersection by exact site id.

Values
------
* E1: ``SSEBOP`` / ``PTJPL`` capture-date ET (mm/d) and ``Closed`` flux ET from the frozen Volk
  May v2.1 master ``daily_2pt1_paired_data.csv`` (sha256 verified against the E1 evidence
  package). Values are used as delivered; no interpolation.
* E2: capture-date EToF from the final grass-basis container
  (``remote_sensing/etf/landsat/{ssebop,ptjpl}/no_mask``) multiplied by the same-day stored
  ERA5-Land ETo (``meteorology/era5/eto``, raw ASCE ETo, no station correction). No
  interpolation.
* Flux reference (primary): Volk v2.1 ``Closed`` (energy-balance-closed ET) -- identical for
  both product versions. Sensitivity arm: the E2 archive's own truth, ``ET_corr`` from the
  flux-data-qaqc archive (``/nas/climate/flux_stations/qaqc/<network>/<site>_daily_data.csv``,
  network from the closure-pool table). The two references are byte-identical at 5 of the 9
  sites and differ at US-Bi1, US-Bi2, US-Mj1, US-Tw2.

Scoring window and support
--------------------------
Per site, the comparison window is the E1 master's date range for that site clipped to the
container period (2013-01-01..2025-12-31). Inside the window, for each member:
``common`` = dates with an E1 capture AND an E2 capture; ``e1_only`` / ``e2_only`` otherwise.
Metrics use the ``common`` dates with finite flux (site minimum 10 paired days, the E2
evaluator's rule). E2 captures outside the E1 window are counted but never scored.

Metrics (E2 evaluator definitions, ``examples/6_Flux_International/evaluate.py::calc_metrics``)
NSE = sklearn r2_score; KGE-2009 with alpha = sd(model)/sd(obs) (population sd), beta =
mean(model)/mean(obs); RMSE; MBE = mean(model - obs); |MBE|; Pearson r.

Uncertainty: 95 % percentile whole-site bootstrap (sites resampled with replacement,
``N_BOOT`` replicates, one shared index array reused for every member, metric, arm and the
pooled statistics, so intervals are comparable across rows).

Usage::

    uv run python examples/6_Flux_International/product_parity/e1_e2_product_parity.py \
        [--out-dir DIR] [--n-boot 10000] [--seed 20260909]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

REPO = Path(__file__).resolve().parents[3]

E1_PAIRED = Path("/data/ssd1/swim/5_Flux_Ensemble/data/flux_2pt1/daily_2pt1_paired_data.csv")
E1_PAIRED_SHA256_EXPECTED = "bc553977782090367ab13861bcfcb369d07c8fad9a794305b8491f178766ad29"
E1_COHORT = REPO / "paper" / "data" / "final" / "e2_primary_daily_site_metrics.csv"
E1_EVIDENCE_META = REPO / "paper" / "data" / "final" / "e2_evidence_metadata.json"

E2_RESULTS = Path(
    "/data/ssd1/swim/6_Flux_International/results/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
)
E2_ARCHIVE = E2_RESULTS / "archive"
E2_POOL = E2_ARCHIVE / "6_evaluation" / "closure_pool" / "closure_pool_sites.csv"
E2_CONTAINER = Path(
    "/data/ssd1/swim/6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
)
E2_ETF_PATHS = {
    "ssebop": "remote_sensing/etf/landsat/ssebop/no_mask",
    "ptjpl": "remote_sensing/etf/landsat/ptjpl/no_mask",
}
E2_ETO_PATH = "meteorology/era5/eto"
QAQC_ROOT = Path("/nas/climate/flux_stations/qaqc")
OPENET_ETO = REPO / "examples" / "5_Flux_Ensemble" / "data" / "openet_refet" / "openet_eto.csv"
OPENET_ETO_SHA256_EXPECTED = "f7917756e81e07cd0c8d828ca271a81fb4e767657147d9a321e1310a633ccfef"

E1_COLS = {"ssebop": "SSEBOP", "ptjpl": "PTJPL"}
MEMBERS = ["ssebop", "ptjpl"]
FLUX_ARMS = {"closed_v2pt1": "flux_closed", "etcorr_qaqc": "flux_etcorr"}
PRIMARY_ARM = "closed_v2pt1"
METRICS = ["kge", "nse", "rmse", "mbe", "abs_mbe", "r", "alpha", "beta"]
HIGHER_IS_BETTER = {
    "kge": True,
    "nse": True,
    "rmse": False,
    "mbe": None,
    "abs_mbe": False,
    "r": True,
    "alpha": None,
    "beta": None,
}
MIN_PAIRED_DAYS = 10
MIN_SUPPORT_DAYS = 5  # per-site minimum for the e2_only / e1_only error contrast
DEFAULT_OUT = E2_ARCHIVE / "6_evaluation" / "e1_e2_product_parity"


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calc_metrics(obs: np.ndarray, mod: np.ndarray) -> dict:
    """E2 evaluator metric set (evaluate.py::calc_metrics) plus |MBE|; NaN below the minimum."""
    obs = np.asarray(obs, float)
    mod = np.asarray(mod, float)
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[mask], mod[mask]
    out = {"n": int(len(obs))}
    if len(obs) < MIN_PAIRED_DAYS:
        out.update({k: np.nan for k in METRICS})
        return out
    r, _ = stats.pearsonr(obs, mod)
    alpha = np.std(mod) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(mod) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    mbe = float((mod - obs).mean())
    out.update(
        {
            "kge": 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2),
            "nse": r2_score(obs, mod),
            "rmse": float(np.sqrt(mean_squared_error(obs, mod))),
            "mbe": mbe,
            "abs_mbe": abs(mbe),
            "r": float(r),
            "alpha": float(alpha),
            "beta": float(beta),
        }
    )
    return out


def site_windows(e1: pd.DataFrame, sites: list[str], t0: pd.Timestamp, t1: pd.Timestamp):
    """Per-site comparison window: E1 master date range clipped to the container period."""
    rows = []
    for fid in sites:
        d = e1.loc[e1["SITE_ID"] == fid, "DATE"]
        rows.append(
            {
                "fid": fid,
                "e1_master_start": d.min(),
                "e1_master_end": d.max(),
                "window_start": max(d.min(), t0),
                "window_end": min(d.max(), t1),
            }
        )
    return pd.DataFrame(rows).set_index("fid")


def build_capture_table(
    fid: str,
    member: str,
    e1_site: pd.DataFrame,
    e2_etf: pd.Series,
    e2_eto: pd.Series,
    flux_closed: pd.Series,
    flux_etcorr: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp],
    eto_openet: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per capture date (either version) for a site/member, with category labels.

    ``in_window`` marks dates inside the site's comparison window; E2 captures outside it are
    kept (``category == e2_outside_e1_window``) for the ledger but never scored.
    """
    w0, w1 = window
    e1_col = E1_COLS[member]
    e1_caps = e1_site.loc[e1_site[e1_col].notna(), e1_col]
    e2_caps = e2_etf.dropna()
    idx = e1_caps.index.union(e2_caps.index).sort_values()
    df = pd.DataFrame(index=idx)
    df.index.name = "date"
    df["fid"] = fid
    df["member"] = member
    df["e1_et"] = e1_caps.reindex(idx)
    df["e2_etf"] = e2_caps.reindex(idx)
    df["eto"] = e2_eto.reindex(idx)
    df["e2_et"] = df["e2_etf"] * df["eto"]
    df["flux_closed"] = flux_closed.reindex(idx)
    df["flux_etcorr"] = flux_etcorr.reindex(idx)
    # decomposition columns: E1 ETf on the OpenET (bias-corrected gridMET) ETo basis and the
    # two counterfactual products that swap only the ETo basis or only the retrieval fraction
    df["eto_openet"] = eto_openet.reindex(idx) if eto_openet is not None else np.nan
    df["e1_etf"] = df["e1_et"] / df["eto_openet"]
    df["e2etf_x_openet_eto"] = df["e2_etf"] * df["eto_openet"]
    df["e1etf_x_era5_eto"] = df["e1_etf"] * df["eto"]
    has1 = df["e1_et"].notna()
    has2 = df["e2_etf"].notna()
    df["in_window"] = (df.index >= w0) & (df.index <= w1)
    cat = np.where(has1 & has2, "common", np.where(has1, "e1_only", "e2_only"))
    cat = np.where(df["in_window"], cat, "e2_outside_e1_window")
    df["category"] = cat
    df["month"] = df.index.month
    df["eto_below_1"] = df["eto"] < 1.0
    return df.reset_index()


def score_sites(days: pd.DataFrame, flux_col: str) -> pd.DataFrame:
    """Per-site E1 and E2 metrics on identical (common, flux-finite) dates, plus E2 - E1."""
    rows = []
    sc = days[(days["category"] == "common") & days[flux_col].notna()]
    for (fid, member), g in sc.groupby(["fid", "member"], sort=True):
        obs = g[flux_col].to_numpy()
        m1 = calc_metrics(obs, g["e1_et"].to_numpy())
        m2 = calc_metrics(obs, g["e2_et"].to_numpy())
        row = {
            "fid": fid,
            "member": member,
            "flux_col": flux_col,
            "n_days": m1["n"],
            "n_days_eto_below_1": int(g["eto_below_1"].sum()),
            "first_date": g["date"].min().date().isoformat(),
            "last_date": g["date"].max().date().isoformat(),
            "scored": m1["n"] >= MIN_PAIRED_DAYS,
        }
        for k in METRICS:
            row[f"{k}_e1"] = m1[k]
            row[f"{k}_e2"] = m2[k]
            row[f"{k}_delta"] = m2[k] - m1[k]
        rows.append(row)
    return pd.DataFrame(rows)


def make_draws(n_sites: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_sites, size=(n_boot, n_sites))


def boot_median(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    """Median of a per-site vector with a percentile CI from shared site draws (NaN skipped)."""
    v = np.asarray(values, float)
    n_fin = int(np.isfinite(v).sum())
    if n_fin == 0:
        return np.nan, np.nan, np.nan
    med = float(np.nanmedian(v))
    if n_fin < 3:
        return med, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reps = np.nanmedian(v[draws], axis=1)
    return med, float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))


def pooled_replicates(obs_by_site: list, mod_by_site: list, draws: np.ndarray) -> dict:
    """All metrics of the concatenated resampled sites, one calc_metrics call per replicate."""
    reps = {k: np.empty(len(draws)) for k in METRICS}
    for j, idx in enumerate(draws):
        m = calc_metrics(
            np.concatenate([obs_by_site[i] for i in idx]),
            np.concatenate([mod_by_site[i] for i in idx]),
        )
        for k in METRICS:
            reps[k][j] = m[k]
    return reps


def pooled_summary(
    days: pd.DataFrame, flux_col: str, sites: list[str], draws: np.ndarray
) -> pd.DataFrame:
    """Concatenated-day metrics for E1 and E2 with site-cluster bootstrap CIs (shared draws)."""
    rows = []
    for member in MEMBERS:
        sc = days[
            (days["category"] == "common") & (days["member"] == member) & days[flux_col].notna()
        ]
        obs_l, e1_l, e2_l = [], [], []
        for fid in sites:
            g = sc[sc["fid"] == fid]
            obs_l.append(g[flux_col].to_numpy())
            e1_l.append(g["e1_et"].to_numpy())
            e2_l.append(g["e2_et"].to_numpy())
        obs = np.concatenate(obs_l)
        e1 = np.concatenate(e1_l)
        e2 = np.concatenate(e2_l)
        m1 = calc_metrics(obs, e1)
        m2 = calc_metrics(obs, e2)
        rep1 = pooled_replicates(obs_l, e1_l, draws)
        rep2 = pooled_replicates(obs_l, e2_l, draws)
        for k in METRICS:
            dd = rep2[k] - rep1[k]
            rows.append(
                {
                    "member": member,
                    "flux_col": flux_col,
                    "metric": k,
                    "n_sites": len(sites),
                    "n_days": int(len(obs)),
                    "e1": m1[k],
                    "e1_ci_low": float(np.nanpercentile(rep1[k], 2.5)),
                    "e1_ci_high": float(np.nanpercentile(rep1[k], 97.5)),
                    "e2": m2[k],
                    "e2_ci_low": float(np.nanpercentile(rep2[k], 2.5)),
                    "e2_ci_high": float(np.nanpercentile(rep2[k], 97.5)),
                    "delta_e2_minus_e1": m2[k] - m1[k],
                    "delta_ci_low": float(np.nanpercentile(dd, 2.5)),
                    "delta_ci_high": float(np.nanpercentile(dd, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def paired_delta_summary(
    site_metrics: pd.DataFrame, sites: list[str], draws: np.ndarray
) -> pd.DataFrame:
    """Median within-site E2 - E1 per metric with shared-draw site-bootstrap CI and sign test."""
    rows = []
    for member in MEMBERS:
        sm = site_metrics[(site_metrics["member"] == member) & site_metrics["scored"]].set_index(
            "fid"
        )
        sm = sm.reindex(sites)
        for k in METRICS:
            d = sm[f"{k}_delta"].to_numpy(float)
            med, lo, hi = boot_median(d, draws)
            fin = d[np.isfinite(d)]
            n = len(fin)
            better = HIGHER_IS_BETTER[k]
            if better is None:
                frac_better = np.nan
            elif better:
                frac_better = float(np.mean(fin > 0)) if n else np.nan
            else:
                frac_better = float(np.mean(fin < 0)) if n else np.nan
            try:
                p_wilcoxon = (
                    float(stats.wilcoxon(fin).pvalue) if n >= 5 and np.any(fin != 0) else np.nan
                )
            except ValueError:
                p_wilcoxon = np.nan
            rows.append(
                {
                    "member": member,
                    "flux_col": sm["flux_col"].dropna().iloc[0]
                    if sm["flux_col"].notna().any()
                    else None,
                    "metric": k,
                    "n_sites": n,
                    "median_e1": float(np.nanmedian(sm[f"{k}_e1"])),
                    "median_e2": float(np.nanmedian(sm[f"{k}_e2"])),
                    "median_delta_e2_minus_e1": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "frac_sites_e2_better": frac_better,
                    "p_wilcoxon_two_sided": p_wilcoxon,
                    "ci_excludes_zero": bool(
                        np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def decomposition(days: pd.DataFrame, flux_col: str, sites: list[str], draws: np.ndarray):
    """Split the E2 - E1 change into an ETo-basis part and a retrieval-fraction part.

    On the common scored dates: ``e1`` = E1 ETf x OpenET ETo (as delivered), ``e2`` = E2 ETf x
    ERA5-Land ETo (as delivered), ``e1etf_x_era5_eto`` swaps only the ETo basis (forcing
    effect), ``e2etf_x_openet_eto`` swaps only the retrieval fraction (retrieval effect).
    Per-site metrics for the four products, medians of the ETo and ETf ratios, within-site
    median deltas relative to E1 with shared-draw CIs, and pooled values.
    """
    arms = {
        "e1": "e1_et",
        "e2": "e2_et",
        "e1etf_x_era5_eto": "e1etf_x_era5_eto",
        "e2etf_x_openet_eto": "e2etf_x_openet_eto",
    }
    site_rows, summ_rows, pooled_rows = [], [], []
    for member in MEMBERS:
        sc = days[
            (days["category"] == "common")
            & (days["member"] == member)
            & days[flux_col].notna()
            & days["eto_openet"].notna()
        ]
        per = {}
        for fid in sites:
            g = sc[sc["fid"] == fid]
            obs = g[flux_col].to_numpy()
            row = {
                "fid": fid,
                "member": member,
                "flux_col": flux_col,
                "n_days": int(len(g)),
                "median_eto_ratio_era5_over_openet": float((g["eto"] / g["eto_openet"]).median())
                if len(g)
                else np.nan,
                "median_etf_ratio_e2_over_e1": float((g["e2_etf"] / g["e1_etf"]).median())
                if len(g)
                else np.nan,
                "median_etf_diff_e2_minus_e1": float((g["e2_etf"] - g["e1_etf"]).median())
                if len(g)
                else np.nan,
                "median_et_ratio_e2_over_e1": float((g["e2_et"] / g["e1_et"]).median())
                if len(g)
                else np.nan,
            }
            for arm, col in arms.items():
                m = calc_metrics(obs, g[col].to_numpy())
                for k in METRICS:
                    row[f"{k}_{arm}"] = m[k]
            per[fid] = row
            site_rows.append(row)
        for arm in ["e2", "e1etf_x_era5_eto", "e2etf_x_openet_eto"]:
            for k in METRICS:
                d = np.array([per[f][f"{k}_{arm}"] - per[f][f"{k}_e1"] for f in sites], float)
                med, lo, hi = boot_median(d, draws)
                summ_rows.append(
                    {
                        "member": member,
                        "flux_col": flux_col,
                        "arm": arm,
                        "metric": k,
                        "n_sites": int(np.isfinite(d).sum()),
                        "median_delta_vs_e1": med,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
        for ratio in [
            "median_eto_ratio_era5_over_openet",
            "median_etf_ratio_e2_over_e1",
            "median_et_ratio_e2_over_e1",
        ]:
            d = np.array([per[f][ratio] for f in sites], float)
            med, lo, hi = boot_median(d, draws)
            summ_rows.append(
                {
                    "member": member,
                    "flux_col": flux_col,
                    "arm": "ratio",
                    "metric": ratio,
                    "n_sites": int(np.isfinite(d).sum()),
                    "median_delta_vs_e1": med,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        # pooled
        obs_l = [sc.loc[sc["fid"] == f, flux_col].to_numpy() for f in sites]
        obs = np.concatenate(obs_l)
        base = calc_metrics(
            obs, np.concatenate([sc.loc[sc["fid"] == f, "e1_et"].to_numpy() for f in sites])
        )
        rep_base = pooled_replicates(
            obs_l, [sc.loc[sc["fid"] == f, "e1_et"].to_numpy() for f in sites], draws
        )
        for arm, col in arms.items():
            mod_l = [sc.loc[sc["fid"] == f, col].to_numpy() for f in sites]
            m = calc_metrics(obs, np.concatenate(mod_l))
            rep = pooled_replicates(obs_l, mod_l, draws)
            for k in METRICS:
                dd = rep[k] - rep_base[k]
                pooled_rows.append(
                    {
                        "member": member,
                        "flux_col": flux_col,
                        "arm": arm,
                        "metric": k,
                        "n_days": int(len(obs)),
                        "value": m[k],
                        "ci_low": float(np.nanpercentile(rep[k], 2.5)),
                        "ci_high": float(np.nanpercentile(rep[k], 97.5)),
                        "delta_vs_e1": m[k] - base[k],
                        "delta_ci_low": float(np.nanpercentile(dd, 2.5)),
                        "delta_ci_high": float(np.nanpercentile(dd, 97.5)),
                    }
                )
    return pd.DataFrame(site_rows), pd.DataFrame(summ_rows), pd.DataFrame(pooled_rows)


def capture_ledger(days: pd.DataFrame, flux_col: str) -> pd.DataFrame:
    rows = []
    for (fid, member), g in days.groupby(["fid", "member"], sort=True):
        win = g[g["in_window"]]
        fv = win[win[flux_col].notna()]
        row = {"fid": fid, "member": member, "flux_col": flux_col}
        for cat in ["common", "e1_only", "e2_only"]:
            row[f"n_{cat}"] = int((win["category"] == cat).sum())
            row[f"n_{cat}_flux_valid"] = int((fv["category"] == cat).sum())
        row["n_e2_outside_e1_window"] = int((g["category"] == "e2_outside_e1_window").sum())
        n_e1 = row["n_common"] + row["n_e1_only"]
        n_e2 = row["n_common"] + row["n_e2_only"]
        row["n_e1_captures_in_window"] = n_e1
        row["n_e2_captures_in_window"] = n_e2
        row["frac_e1_captures_shared"] = row["n_common"] / n_e1 if n_e1 else np.nan
        row["frac_e2_captures_shared"] = row["n_common"] / n_e2 if n_e2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def support_error_contrast(
    days: pd.DataFrame, flux_col: str, draws_sites: list[str], draws: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do the dates only one version captured carry larger error than the shared dates?

    For each member and each version: absolute error of that version's ET against flux on its
    own exclusive dates vs the common dates, within site (median |err| difference, sites with at
    least ``MIN_SUPPORT_DAYS`` exclusive flux-valid dates), with a shared-draw site bootstrap and
    a Wilcoxon test; and pooled (all days), with a Mann-Whitney U test and a site-cluster
    bootstrap of the mean |err| difference. Seasonality covariates (median month, median ETo)
    are reported so a curation claim can be separated from a season-mix claim.
    """
    persite_rows, summary_rows = [], []
    for member in MEMBERS:
        for version, excl_cat, et_col in [("e2", "e2_only", "e2_et"), ("e1", "e1_only", "e1_et")]:
            g = days[(days["member"] == member) & days["in_window"] & days[flux_col].notna()].copy()
            g["abs_err"] = (g[et_col] - g[flux_col]).abs()
            g["err"] = g[et_col] - g[flux_col]
            per = {}
            for fid, gs in g.groupby("fid"):
                c = gs[gs["category"] == "common"]
                x = gs[gs["category"] == excl_cat]
                rec = {
                    "member": member,
                    "version": version,
                    "flux_col": flux_col,
                    "fid": fid,
                    "n_common": int(len(c)),
                    "n_exclusive": int(len(x)),
                    "median_abs_err_common": float(c["abs_err"].median()) if len(c) else np.nan,
                    "median_abs_err_exclusive": float(x["abs_err"].median()) if len(x) else np.nan,
                    "mean_err_common": float(c["err"].mean()) if len(c) else np.nan,
                    "mean_err_exclusive": float(x["err"].mean()) if len(x) else np.nan,
                    "median_eto_common": float(c["eto"].median()) if len(c) else np.nan,
                    "median_eto_exclusive": float(x["eto"].median()) if len(x) else np.nan,
                    "median_flux_common": float(c[flux_col].median()) if len(c) else np.nan,
                    "median_flux_exclusive": float(x[flux_col].median()) if len(x) else np.nan,
                    "frac_nov_feb_common": float(c["month"].isin([11, 12, 1, 2]).mean())
                    if len(c)
                    else np.nan,
                    "frac_nov_feb_exclusive": float(x["month"].isin([11, 12, 1, 2]).mean())
                    if len(x)
                    else np.nan,
                }
                rec["eligible"] = (
                    rec["n_exclusive"] >= MIN_SUPPORT_DAYS and rec["n_common"] >= MIN_SUPPORT_DAYS
                )
                rec["delta_median_abs_err_excl_minus_common"] = (
                    rec["median_abs_err_exclusive"] - rec["median_abs_err_common"]
                    if rec["eligible"]
                    else np.nan
                )
                per[fid] = rec
                persite_rows.append(rec)
            # within-site
            d = np.array(
                [
                    per[f]["delta_median_abs_err_excl_minus_common"] if f in per else np.nan
                    for f in draws_sites
                ]
            )
            med, lo, hi = boot_median(d, draws)
            fin = d[np.isfinite(d)]
            try:
                p_w = (
                    float(stats.wilcoxon(fin).pvalue)
                    if len(fin) >= 5 and np.any(fin != 0)
                    else np.nan
                )
            except ValueError:
                p_w = np.nan
            # pooled
            c_all = g[g["category"] == "common"]
            x_all = g[g["category"] == excl_cat]
            if len(x_all) >= MIN_SUPPORT_DAYS and len(c_all) >= MIN_SUPPORT_DAYS:
                p_mw = float(
                    stats.mannwhitneyu(
                        x_all["abs_err"], c_all["abs_err"], alternative="two-sided"
                    ).pvalue
                )
            else:
                p_mw = np.nan
            c_by = [c_all.loc[c_all["fid"] == f, "abs_err"].to_numpy() for f in draws_sites]
            x_by = [x_all.loc[x_all["fid"] == f, "abs_err"].to_numpy() for f in draws_sites]
            reps = []
            for dr in draws:
                cc = np.concatenate([c_by[i] for i in dr])
                xx = np.concatenate([x_by[i] for i in dr])
                reps.append(np.mean(xx) - np.mean(cc) if len(xx) and len(cc) else np.nan)
            reps = np.array(reps, float)
            summary_rows.append(
                {
                    "member": member,
                    "version": version,
                    "exclusive_category": excl_cat,
                    "flux_col": flux_col,
                    "n_sites_eligible": int(len(fin)),
                    "n_days_common": int(len(c_all)),
                    "n_days_exclusive": int(len(x_all)),
                    "within_site_median_delta_abs_err": med,
                    "within_site_ci_low": lo,
                    "within_site_ci_high": hi,
                    "within_site_frac_sites_exclusive_worse": float(np.mean(fin > 0))
                    if len(fin)
                    else np.nan,
                    "within_site_p_wilcoxon": p_w,
                    "pooled_mean_abs_err_common": float(c_all["abs_err"].mean())
                    if len(c_all)
                    else np.nan,
                    "pooled_mean_abs_err_exclusive": float(x_all["abs_err"].mean())
                    if len(x_all)
                    else np.nan,
                    "pooled_delta_mean_abs_err": (
                        float(x_all["abs_err"].mean() - c_all["abs_err"].mean())
                        if len(x_all) and len(c_all)
                        else np.nan
                    ),
                    "pooled_delta_ci_low": float(np.nanpercentile(reps, 2.5))
                    if np.isfinite(reps).any()
                    else np.nan,
                    "pooled_delta_ci_high": float(np.nanpercentile(reps, 97.5))
                    if np.isfinite(reps).any()
                    else np.nan,
                    "pooled_median_abs_err_common": float(c_all["abs_err"].median())
                    if len(c_all)
                    else np.nan,
                    "pooled_median_abs_err_exclusive": float(x_all["abs_err"].median())
                    if len(x_all)
                    else np.nan,
                    "pooled_mean_err_common": float(c_all["err"].mean()) if len(c_all) else np.nan,
                    "pooled_mean_err_exclusive": float(x_all["err"].mean())
                    if len(x_all)
                    else np.nan,
                    "pooled_p_mannwhitney": p_mw,
                    "pooled_median_eto_common": float(c_all["eto"].median())
                    if len(c_all)
                    else np.nan,
                    "pooled_median_eto_exclusive": float(x_all["eto"].median())
                    if len(x_all)
                    else np.nan,
                    "pooled_frac_nov_feb_common": float(c_all["month"].isin([11, 12, 1, 2]).mean())
                    if len(c_all)
                    else np.nan,
                    "pooled_frac_nov_feb_exclusive": float(
                        x_all["month"].isin([11, 12, 1, 2]).mean()
                    )
                    if len(x_all)
                    else np.nan,
                }
            )
    return pd.DataFrame(persite_rows), pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# loaders (read-only)
# ---------------------------------------------------------------------------


def load_e1_cohort() -> list[str]:
    return sorted(pd.read_csv(E1_COHORT)["fid"].astype(str).tolist())


def load_e2_pool() -> pd.DataFrame:
    pool = pd.read_csv(E2_POOL)
    if len(pool) != 47 or (pool["closure_tier"] != "closure_corrected").any():
        raise ValueError("closure_pool_sites.csv is not the 47-site closure-corrected pool")
    return pool


def load_e1_master() -> pd.DataFrame:
    sha = sha256_file(E1_PAIRED)
    if sha != E1_PAIRED_SHA256_EXPECTED:
        raise ValueError(f"{E1_PAIRED} sha256 {sha} != frozen {E1_PAIRED_SHA256_EXPECTED}")
    df = pd.read_csv(E1_PAIRED, parse_dates=["DATE"])
    df["DATE"] = df["DATE"].dt.normalize()
    dup = df.duplicated(["SITE_ID", "DATE"]).sum()
    if dup:
        raise ValueError(f"E1 master has {dup} duplicated site-dates")
    return df


def load_container_series(sites: list[str]):
    from swimrs.container.container import SwimContainer

    c = SwimContainer(str(E2_CONTAINER), mode="r")
    try:
        etf = {m: c.query.dataframe(p, fields=sites) for m, p in E2_ETF_PATHS.items()}
        eto = c.query.dataframe(E2_ETO_PATH, fields=sites)
        attrs = {
            "project_name": c.project_name,
            "created_at": str(c._root.attrs.get("created_at")),
            "start_date": c.start_date.date().isoformat(),
            "end_date": c.end_date.date().isoformat(),
            "n_fields": c.n_fields,
            "source_shapefile": c._root.attrs.get("source_shapefile"),
        }
    finally:
        c.close()
    for df in list(etf.values()) + [eto]:
        df.index = pd.DatetimeIndex(df.index).normalize()
    return etf, eto, attrs


def load_qaqc_etcorr(fid: str, network: str) -> tuple[pd.Series, Path]:
    p = QAQC_ROOT / network / f"{fid}_daily_data.csv"
    df = pd.read_csv(p, index_col="date", parse_dates=True)
    s = df["ET_corr"]
    s.index = s.index.normalize()
    return s, p


def load_openet_eto(sites: list[str]) -> pd.DataFrame:
    """OpenET bias-corrected gridMET ETo (E1 benchmark basis), wide site x YYYYMMDD -> daily frame."""
    sha = sha256_file(OPENET_ETO)
    if sha != OPENET_ETO_SHA256_EXPECTED:
        raise ValueError(f"{OPENET_ETO} sha256 {sha} != frozen {OPENET_ETO_SHA256_EXPECTED}")
    wide = pd.read_csv(OPENET_ETO, index_col="site_id")
    missing = [s for s in sites if s not in wide.index]
    if missing:
        raise KeyError(f"OpenET ETo missing sites: {missing}")
    df = wide.loc[sites].T
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.astype(float)


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": len(run("status", "--short").splitlines()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    e1_sites = load_e1_cohort()
    pool = load_e2_pool()
    e2_conus = sorted(pool.loc[pool["region"] == "CONUS", "fid"].astype(str).tolist())
    sites = sorted(set(e1_sites) & set(e2_conus))
    print(
        f"E1 cohort {len(e1_sites)} | E2 pool {len(pool)} (CONUS {len(e2_conus)}) | intersection {len(sites)}: {sites}"
    )
    if not sites:
        raise SystemExit("empty intersection")
    e1_not_fluxnet_style = [s for s in e1_sites if "-" not in s]

    e1 = load_e1_master()
    e1 = e1[e1["SITE_ID"].isin(sites)].copy()
    etf, eto, cattrs = load_container_series(sites)
    eto_openet = load_openet_eto(sites)
    t0, t1 = (
        pd.Timestamp(cattrs.get("start_date", "2013-01-01")),
        pd.Timestamp(cattrs.get("end_date", "2025-12-31")),
    )
    windows = site_windows(e1, sites, t0, t1)

    pool_i = pool.set_index("fid")
    flux_files = {}
    day_frames = []
    for fid in sites:
        e1_site = e1[e1["SITE_ID"] == fid].set_index("DATE").sort_index()
        etcorr, fpath = load_qaqc_etcorr(fid, pool_i.loc[fid, "flux_network"])
        if pool_i.loc[fid, "flux_et_col"] != "ET_corr":
            raise ValueError(
                f"{fid}: pool flux_et_col is {pool_i.loc[fid, 'flux_et_col']}, expected ET_corr"
            )
        flux_files[fid] = {
            "path": str(fpath),
            "sha256": sha256_file(fpath),
            "network": pool_i.loc[fid, "flux_network"],
        }
        w = (windows.loc[fid, "window_start"], windows.loc[fid, "window_end"])
        for m in MEMBERS:
            day_frames.append(
                build_capture_table(
                    fid,
                    m,
                    e1_site,
                    etf[m][fid],
                    eto[fid],
                    e1_site["Closed"],
                    etcorr,
                    w,
                    eto_openet[fid],
                )
            )
    days = pd.concat(day_frames, ignore_index=True)

    # flux-reference agreement (documentation only)
    ref_rows = []
    for fid in sites:
        g = days[(days["fid"] == fid) & (days["member"] == "ssebop") & days["in_window"]]
        both = g[g["flux_closed"].notna() & g["flux_etcorr"].notna()]
        dd = (both["flux_closed"] - both["flux_etcorr"]).abs()
        ref_rows.append(
            {
                "fid": fid,
                "n_capture_dates_with_both_refs": int(len(both)),
                "n_closed_only": int((g["flux_closed"].notna() & g["flux_etcorr"].isna()).sum()),
                "n_etcorr_only": int((g["flux_closed"].isna() & g["flux_etcorr"].notna()).sum()),
                "max_abs_diff_mm": float(dd.max()) if len(dd) else np.nan,
                "n_diff_gt_1e-3": int((dd > 1e-3).sum()),
                "identical": bool(len(dd) and (dd <= 1e-3).all()),
            }
        )
    ref_agreement = pd.DataFrame(ref_rows)

    draws = make_draws(len(sites), args.n_boot, args.seed)
    results = {}
    for arm, flux_col in FLUX_ARMS.items():
        sm = score_sites(days, flux_col)
        sm["arm"] = arm
        sm = sm.merge(
            windows.reset_index()[["fid", "window_start", "window_end"]], on="fid", how="left"
        )
        pd_sum = paired_delta_summary(sm, sites, draws)
        pd_sum["arm"] = arm
        pooled = pooled_summary(days, flux_col, sites, draws)
        pooled["arm"] = arm
        ledger = capture_ledger(days, flux_col)
        ledger["arm"] = arm
        per_sup, sup = support_error_contrast(days, flux_col, sites, draws)
        per_sup["arm"] = arm
        sup["arm"] = arm
        results[arm] = dict(
            site_metrics=sm,
            paired=pd_sum,
            pooled=pooled,
            ledger=ledger,
            support_persite=per_sup,
            support=sup,
        )

    dec_site, dec_summary, dec_pooled = decomposition(days, FLUX_ARMS[PRIMARY_ARM], sites, draws)

    # ETo-floor sensitivity on the primary arm: drop scored days with ETo < 1.0 mm/d
    prim = FLUX_ARMS[PRIMARY_ARM]
    days_floor = days[~days["eto_below_1"].fillna(False)]
    sm_floor = score_sites(days_floor, prim)
    sm_floor["arm"] = f"{PRIMARY_ARM}_eto_ge_1"
    pd_floor = paired_delta_summary(sm_floor, sites, draws)
    pd_floor["arm"] = f"{PRIMARY_ARM}_eto_ge_1"

    # ------------------------------------------------------------------ write
    def cat(key):
        return pd.concat([results[a][key] for a in results], ignore_index=True)

    days_out = days.copy()
    days_out["date"] = days_out["date"].dt.date
    days_out.to_csv(out / "capture_days.csv", index=False)
    site_metrics_all = pd.concat([cat("site_metrics"), sm_floor], ignore_index=True)
    site_metrics_all.to_csv(out / "site_metrics.csv", index=False)
    paired_all = pd.concat([cat("paired"), pd_floor], ignore_index=True)
    paired_all.to_csv(out / "paired_delta_summary.csv", index=False)
    cat("pooled").to_csv(out / "pooled_summary.csv", index=False)
    cat("ledger").to_csv(out / "capture_support_ledger.csv", index=False)
    cat("support").to_csv(out / "support_error_contrast.csv", index=False)
    cat("support_persite").to_csv(out / "support_error_contrast_persite.csv", index=False)
    ref_agreement.to_csv(out / "flux_reference_agreement.csv", index=False)
    dec_site.to_csv(out / "decomposition_site.csv", index=False)
    dec_summary.to_csv(out / "decomposition_within_site_summary.csv", index=False)
    dec_pooled.to_csv(out / "decomposition_pooled.csv", index=False)
    windows.reset_index().to_csv(out / "site_windows.csv", index=False)

    ledger_prim = results[PRIMARY_ARM]["ledger"]
    ledger_tot = ledger_prim.groupby("member")[
        [c for c in ledger_prim.columns if c.startswith("n_")]
    ].sum()
    ledger_tot.to_csv(out / "capture_support_ledger_totals.csv")

    e1_meta = json.load(open(E1_EVIDENCE_META))
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(REPO)),
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "read_only": "no Earth Engine, no calibration, no forward model; container opened mode='r'",
        "cohorts": {
            "e1_source": {
                "path": str(E1_COHORT),
                "sha256": sha256_file(E1_COHORT),
                "n_sites": len(e1_sites),
                "note": "paper Experiment 1 daily cohort (Ex5 Run 22); the e2_ file namespace is the legacy label for the paper's E1 (see e2_evidence_metadata.json 'file_namespace')",
                "n_sites_without_fluxnet_style_id": len(e1_not_fluxnet_style),
            },
            "e2_source": {
                "path": str(E2_POOL),
                "sha256": sha256_file(E2_POOL),
                "n_sites": int(len(pool)),
                "n_conus": len(e2_conus),
            },
            "intersection": sites,
            "intersection_rule": "exact site-id match; E1 sites with OpenET-internal ids (no ISO-prefix FLUXNET id) have no E2 counterpart",
        },
        "e1_values": {
            "path": str(E1_PAIRED),
            "sha256": E1_PAIRED_SHA256_EXPECTED,
            "matches_e1_evidence_package_hash": e1_meta["source_data"]["input_sha256"].get(
                str(E1_PAIRED)
            )
            == E1_PAIRED_SHA256_EXPECTED,
            "columns": {"ssebop": "SSEBOP", "ptjpl": "PTJPL", "flux_primary": "Closed"},
            "definition": "OpenET v2.1 capture-date ET (mm/d), Volk May v2.1 master; used as delivered, no interpolation",
        },
        "e2_values": {
            "container": str(E2_CONTAINER),
            "container_attrs": {
                k: cattrs.get(k)
                for k in (
                    "project_name",
                    "created_at",
                    "start_date",
                    "end_date",
                    "n_fields",
                    "source_shapefile",
                )
            },
            "etf_paths": E2_ETF_PATHS,
            "eto_path": E2_ETO_PATH,
            "definition": "ET_e2 = EToF(capture date) x ERA5-Land ETo(same day, stored in container, raw ASCE grass reference, no station correction); no interpolation",
            "ssebop_provenance": "USGS ESPA Collection 2 Level-3 provisional ET (ETrF), rescaled per site-day by ERA5-Land ETr/ETo (e2_refooting/phase2_convert_ssebop_basis.py), config [paths.etf_sources] ssebop = {landsat}/extracts/ssebop_etf_grass/no_mask",
            "ptjpl_provenance": "openet-ptjpl run in Earth Engine with ERA5-Land forcing and ERA5-Land grass ETo (not an OpenET v2.1 production asset)",
            "e2_archive_evaluation_metadata": str(
                E2_ARCHIVE / "6_evaluation" / "evaluation_metadata.json"
            ),
        },
        "decomposition": {
            "openet_eto": {
                "path": str(OPENET_ETO),
                "sha256": OPENET_ETO_SHA256_EXPECTED,
                "identity": "OpenET bias-corrected gridMET ETo asset; identical to the Ex5 container eto_corr (e2_evidence_metadata.json)",
            },
            "arms": {
                "e1": "E1 ET as delivered (= E1 ETf x OpenET ETo)",
                "e2": "E2 ETf x ERA5-Land ETo (as delivered)",
                "e1etf_x_era5_eto": "E1 ETf (E1 ET / OpenET ETo) x ERA5-Land ETo: swaps only the ETo basis (forcing effect)",
                "e2etf_x_openet_eto": "E2 ETf x OpenET ETo: swaps only the retrieval fraction (retrieval effect)",
            },
        },
        "flux_reference": {
            "primary_arm": PRIMARY_ARM,
            "primary": "Volk v2.1 'Closed' from the E1 master (energy-balance-closed ET), identical for both product versions",
            "sensitivity_arm": "etcorr_qaqc",
            "sensitivity": "flux-data-qaqc ET_corr (EBR-corrected), the E2 archive truth; per-site file + sha256 below",
            "files": flux_files,
            "agreement": ref_agreement.to_dict(orient="records"),
        },
        "design": {
            "window": "per site, E1 master date range clipped to the container period",
            "categories": "common = E1 capture AND E2 capture on the same date; e1_only / e2_only otherwise; e2_outside_e1_window counted, never scored",
            "scoring_set": "common dates with finite flux reference; site minimum 10 paired days (E2 evaluator rule)",
            "metrics": "calc_metrics from examples/6_Flux_International/evaluate.py plus |MBE|: NSE = sklearn r2_score, KGE-2009 with population sd, MBE = mean(model - obs)",
            "bootstrap": {
                "n_boot": args.n_boot,
                "seed": args.seed,
                "unit": "site",
                "shared_draws": "one (n_boot x n_sites) index array reused for every member, metric, arm, within-site median and pooled statistic",
            },
            "support_contrast": f"|ET - flux| on a version's exclusive dates vs the common dates; within-site median difference (sites with >= {MIN_SUPPORT_DAYS} exclusive and >= {MIN_SUPPORT_DAYS} common flux-valid dates) with shared-draw bootstrap + Wilcoxon; pooled mean difference with site-cluster bootstrap + Mann-Whitney; season/ETo covariates reported",
            "eto_floor_sensitivity": "arm closed_v2pt1_eto_ge_1 drops scored days with ERA5-Land ETo < 1.0 mm/d (the E2 calibration weight floor); evaluation itself applies no floor",
        },
        "attribution_scope": {
            "can_evaluate": [
                "meteorological forcing of the retrieval and of the ETo used to scale EToF (ERA5-Land vs bias-corrected GridMET/NLDAS-2)",
                "ancillary retrieval inputs and model implementation (ESPA SSEBop vs OpenET SSEBop; ERA5-Land-forced openet-ptjpl vs production PT-JPL)",
                "processing and scene treatment (cloud/QA screening, scene selection, reference-basis conversion)",
                "capture-date support (which acquisitions each version delivers)",
            ],
            "cannot_evaluate": [
                "HWSD vs SSURGO soils: soils do not enter either satellite product, only the SWIM-RS water balance",
                "irrigation classifier / parameter transfer: not part of either product",
            ],
        },
        "counts": {
            "n_sites": len(sites),
            "site_windows": {
                f: {
                    "start": windows.loc[f, "window_start"].date().isoformat(),
                    "end": windows.loc[f, "window_end"].date().isoformat(),
                }
                for f in sites
            },
            "ledger_totals_primary": ledger_tot.to_dict(orient="index"),
            "scored_sites": {
                m: int(
                    results[PRIMARY_ARM]["site_metrics"].query("member == @m and scored").shape[0]
                )
                for m in MEMBERS
            },
            "scored_days": {
                m: int(
                    results[PRIMARY_ARM]["site_metrics"]
                    .query("member == @m and scored")["n_days"]
                    .sum()
                )
                for m in MEMBERS
            },
        },
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    manifest = {p.name: sha256_file(p) for p in sorted(out.glob("*.csv"))}
    manifest["metadata.json"] = sha256_file(out / "metadata.json")
    with open(out / "sha256_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------ console
    pd.set_option("display.width", 220)
    print("\n== capture support ledger totals (primary arm) ==")
    print(ledger_tot.to_string())
    for arm in list(results) + [f"{PRIMARY_ARM}_eto_ge_1"]:
        print(f"\n== paired within-site E2 - E1, arm={arm} ==")
        pdf = paired_all[paired_all["arm"] == arm]
        print(
            pdf[
                [
                    "member",
                    "metric",
                    "n_sites",
                    "median_e1",
                    "median_e2",
                    "median_delta_e2_minus_e1",
                    "ci_low",
                    "ci_high",
                    "frac_sites_e2_better",
                    "p_wilcoxon_two_sided",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )
    print("\n== pooled (concatenated days), site-cluster bootstrap ==")
    print(
        cat("pooled")[
            [
                "arm",
                "member",
                "metric",
                "n_days",
                "e1",
                "e2",
                "delta_e2_minus_e1",
                "delta_ci_low",
                "delta_ci_high",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print("\n== support-error contrast ==")
    print(
        cat("support")[
            [
                "arm",
                "member",
                "version",
                "n_sites_eligible",
                "n_days_common",
                "n_days_exclusive",
                "within_site_median_delta_abs_err",
                "within_site_ci_low",
                "within_site_ci_high",
                "within_site_p_wilcoxon",
                "pooled_delta_mean_abs_err",
                "pooled_delta_ci_low",
                "pooled_delta_ci_high",
                "pooled_p_mannwhitney",
                "pooled_median_eto_common",
                "pooled_median_eto_exclusive",
                "pooled_frac_nov_feb_common",
                "pooled_frac_nov_feb_exclusive",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print("\n== decomposition, within-site median delta vs E1 (primary arm) ==")
    print(dec_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== decomposition, pooled (primary arm) ==")
    print(
        dec_pooled[
            [
                "member",
                "arm",
                "metric",
                "n_days",
                "value",
                "delta_vs_e1",
                "delta_ci_low",
                "delta_ci_high",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

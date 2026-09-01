"""Within-E1 held-out spatial transfer (leave-region-out + leave-one-site-out).

The filename retains the former paper-numbering label for archive continuity.

Supports the manuscript claim (``paper/text/main.md`` Table 1 and the Parameter
Transfer methods paragraph) that within Experiment 1 a fixed parameter vector
formed from the across-site median of *training-site* posterior medians, applied
to *held-out* sites without local recalibration, retains most of the
site-calibrated skill. This is the within-domain spatial cross-validation that
removes the E1->E2 domain-shift confound (single forcing GridMET, single soils
SSURGO, single OpenET target; only the site set changes between train and test).

Design (see ``paper/final_implementations.md`` WP6.2 and
``paper/POINTS_TO_MAKE.md`` section 7):

  1. Folds are defined by geography ONLY (fixed agroclimatic regions from the
     cohort shapefile ``state`` field). Flux-tower ET never enters fold
     definition, vector aggregation, or selection.
  2. For each held-out fold, the transfer vector is the median across the
     TRAINING sites of each training site's posterior median (median-of-site-
     medians), for the eight calibrated parameters. It is applied forward to the
     held-out site(s) with no local recalibration.
  3. Held-out transfer is scored against flux ET on identical paired days/months
     alongside two reference configurations run on the same container:
        local   - the canonical Run 22 per-site calibration (upper bound)
        default - the model's generic/initial parameters (lower bound)

Two pooled held-out schemes are produced:
    loro  - leave-region-out (PRIMARY): the vector excludes every site in the
            held-out site's region.
    loso  - leave-one-site-out (SECONDARY): the vector excludes only the held-out
            site itself.

Irrigation-stratified arms
--------------------------
The pooled cropland vector is dominated by irrigated sites (39 of the 60 container
fields satisfy ``properties/irrigation/irr > 0.5``), and its ``mad = 0.136917`` is
plausible for the irrigated prior (0.10-0.30) but sits *outside* the configured
rainfed prior (0.30-0.80). A single pooled vector therefore hands every rainfed
held-out site a management-allowable-depletion value its own prior forbids. Two
extra arms test whether conditioning the transfer on an independently inferred
irrigation class preserves or improves held-out skill while keeping rainfed ``mad``
inside the rainfed domain (see ``paper/notes/irrigation_stratified_transfer_handoff.md``,
"Target-class policy > Within E1"):

    loro_strat - stratified leave-region-out (PRIMARY stratified comparison): the
                 vector is the median over training sites in the SAME irrigation
                 class as the held-out site AND outside its region.
    loso_strat - stratified leave-one-site-out: the vector is the median over the
                 other sites in the same irrigation class.

The class is read from the container's ``properties/irrigation/irr`` (the same
remote-sensing irrigation fraction ``archive_run.py`` uses to group the Run 22
posterior), never from flux ET. Vectors stay fixed per site and never vary by year;
annual irrigation status continues to drive scheduler activation exactly as in the
canonical model. Class-specific training counts are recorded for every fold and a
fold with inadequate same-class support RAISES - there is deliberately no silent
fallback to the pooled vector.

All six configurations (``loro``, ``loro_strat``, ``loso``, ``loso_strat``,
``local``, ``default``) are scored in one invocation on identical paired
days/months so every paired delta is internally consistent. The primary reported
delta is ``loro_strat`` minus ``loro``; results are reported for all sites, for
irrigated and rainfed sites separately, and per LORO region.

This is a FORWARD run with fixed parameters against existing container inputs. It
does NOT calibrate and does NOT call Earth Engine. It reuses the Run 22 posterior
ensemble (``*.par.csv``) already on disk and opens the container read-only.

Usage:
    uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/within_e2_transfer.py
    uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/within_e2_transfer.py \
        --out /data/ssd1/swim/5_Flux_Ensemble/results/within_e2_transfer_irrigation_stratified \
        --n-boot 2000 --seed 1234 --irr-threshold 0.5 --min-class-train 5
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate as ev  # noqa: E402  (sibling module; needs HERE on sys.path)

import swimrs.process.input as swim_input_mod  # noqa: E402
from swimrs.calibrate.flux_utils import (  # noqa: E402
    full_month_paired_sums,
    passes_site_minimum,
)
from swimrs.container import SwimContainer  # noqa: E402
from swimrs.process.input import build_swim_input  # noqa: E402
from swimrs.process.loop_fast import run_daily_loop_fast  # noqa: E402

# Run 22 canonical publication basis: par.csv + the run22 container it was
# calibrated and evaluated against (seeded from run21 with the calibration group
# dropped, then recalibrated under source-exclusive physics + gw gate).
DEFAULT_PAR_CSV = "/data/ssd1/swim/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv"
DEFAULT_CONTAINER = "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim"

PARAM_FAMILIES = [
    "aw",
    "ndvi_k",
    "ndvi_0",
    "mad",
    "ks_alpha",
    "kr_alpha",
    "swe_alpha",
    "swe_beta",
]

# Fixed agroclimatic regions (Koppen-informed groupings of the CONUS states
# present in the cohort). Defined a priori from geography, never from skill.
#   West_Coast          - CA Central Valley / delta + PNW (Mediterranean Csa/Csb)
#   Desert_Southwest    - AZ + NV (hot/cold desert BWh/BWk/BSk, fully irrigated)
#   Great_Plains_Rockies- MT, CO, SD, KS (semi-arid cold steppe BSk / Dfb)
#   Corn_Belt           - MN, IA, IL, OH, NE (humid continental Dfa, W. Corn Belt)
#   South_Central       - TX, OK, AR, LA, MS, NC (humid subtropical Cfa)
STATE_TO_REGION = {
    "CA": "West_Coast",
    "OR": "West_Coast",
    "AZ": "Desert_Southwest",
    "NV": "Desert_Southwest",
    "MT": "Great_Plains_Rockies",
    "CO": "Great_Plains_Rockies",
    "SD": "Great_Plains_Rockies",
    "KS": "Great_Plains_Rockies",
    "MN": "Corn_Belt",
    "IA": "Corn_Belt",
    "IL": "Corn_Belt",
    "OH": "Corn_Belt",
    "NE": "Corn_Belt",
    "TX": "South_Central",
    "OK": "South_Central",
    "AR": "South_Central",
    "LA": "South_Central",
    "MS": "South_Central",
    "NC": "South_Central",
}

CONFIGS = ["loro", "loro_strat", "loso", "loso_strat", "local", "default"]
CONFIG_LABELS = {
    "loro": "Held-out transfer (leave-region-out)",
    "loro_strat": "Held-out transfer (LORO, irrigation-stratified)",
    "loso": "Held-out transfer (leave-one-site-out)",
    "loso_strat": "Held-out transfer (LOSO, irrigation-stratified)",
    "local": "Local site calibration (Run 22)",
    "default": "Generic defaults (uncalibrated)",
}

# Paired-bootstrap deltas, (a, b) meaning "a minus b". The PRIMARY comparison for
# the stratified experiment is loro_strat - loro (same fold geometry, same scoring
# support; the only thing that changes is whether the training pool is restricted
# to the held-out site's irrigation class). The *_minus_local pairs are retained
# unchanged from the pooled-only version of this script.
DELTA_PAIRS = [
    ("loro_strat", "loro"),
    ("loso_strat", "loso"),
    ("loro", "local"),
    ("loso", "local"),
    ("loro_strat", "local"),
]

# Win-rate pairs, (a, b) meaning "fraction of sites where a beats b".
WIN_PAIRS = [
    ("loro", "default"),
    ("loro", "local"),
    ("loso", "default"),
    ("loso", "local"),
    ("loro_strat", "default"),
    ("loro_strat", "local"),
    ("loro_strat", "loro"),
    ("loso_strat", "default"),
    ("loso_strat", "local"),
    ("loso_strat", "loso"),
]

# `r2` in this codebase IS Nash-Sutcliffe efficiency (verified pure relabel); the
# key name is kept for continuity with the legacy ``e2_*`` artifacts. `abs_bias` is the
# absolute mean bias error, lower-is-better, added for the stratified comparison.
METRIC_KEYS = ["r2", "kge", "rmse", "bias", "r", "mae", "alpha", "beta", "abs_bias"]
SUMMARY_METRICS = ["r2", "kge", "rmse", "bias", "abs_bias"]

# Source irrigation class rule (identical to archive_run.py's posterior grouping).
IRR_CLASS_RULE = "properties/irrigation/irr > {threshold} -> irrigated, else rainfed"
IRR_CLASSES = ["irrigated", "rainfed"]

# Daily/monthly pairing gates (match evaluate.py / VALIDATION_POLICY).
MIN_DAILY_OBS = 10
MIN_DAILY_FOR_MONTHLY = 30
MIN_MONTHLY_OBS = 6


# --------------------------------------------------------------------------- #
# Parameter aggregation
# --------------------------------------------------------------------------- #
def _family_columns(columns):
    """Map each parameter family to its per-site .par.csv columns."""
    by_family = {fam: [] for fam in PARAM_FAMILIES}
    for col in columns:
        for fam in PARAM_FAMILIES:
            if col.startswith(f"pname:p_{fam}_"):
                by_family[fam].append(col)
                break
    return by_family


def _site_from_column(col, fam):
    remainder = col[len(f"pname:p_{fam}_") :]
    return remainder.split("_:")[0]


def per_site_median_table(par_csv):
    """Return a DataFrame (index=site, columns=families) of posterior medians.

    Each cell is a site's median realization value for that family (the ``base``
    realization is excluded), i.e. the per-site posterior median. This is the
    single source of truth for both the local comparator (a site's own row) and
    every held-out transfer vector (median across a training subset of rows).
    """
    df = pd.read_csv(par_csv, index_col=0)
    n_base = sum(1 for i in df.index if str(i) == "base")
    if n_base != 1:
        raise ValueError(f"Expected exactly one 'base' realization, found {n_base} in {par_csv}")
    realizations = df.loc[[i for i in df.index if str(i) != "base"]]
    by_family = _family_columns(df.columns)

    per_site = {}
    site_sets = {}
    for fam in PARAM_FAMILIES:
        cols = by_family[fam]
        if not cols:
            raise ValueError(f"No columns found for parameter family {fam!r} in {par_csv}")
        sites = [_site_from_column(c, fam) for c in cols]
        if len(set(sites)) != len(sites):
            raise ValueError(f"Duplicate site columns for family {fam!r} in {par_csv}")
        med = realizations[cols].median()
        med.index = sites
        if not med.map(lambda v: pd.notna(v) and abs(v) != float("inf")).all():
            raise ValueError(f"Non-finite per-site median for family {fam!r} in {par_csv}")
        per_site[fam] = med
        site_sets[fam] = frozenset(sites)
    if len(set(site_sets.values())) != 1:
        raise ValueError(f"Site sets differ across parameter families in {par_csv}")
    table = pd.DataFrame(per_site).sort_index()
    table.index.name = "site"
    return table[PARAM_FAMILIES]


# --------------------------------------------------------------------------- #
# Irrigation class (remote sensing only; flux never enters)
# --------------------------------------------------------------------------- #
def read_irrigation_class(container_path, threshold):
    """Read the per-field irrigation fraction and class from a container.

    ``properties/irrigation/irr`` is the remote-sensing irrigated fraction of each
    field, stored in ``geometry/uid`` order. The class rule is the same one
    ``archive_run.py`` uses to group the Run 22 posterior (``irr > threshold`` is
    irrigated), which keeps the source classification here identical to the
    classification the frozen posterior summary was built on. Opened read-only;
    nothing is written back. Returns ``(irr_by_fid, class_by_fid)``.
    """
    root = zarr.open(str(container_path), mode="r")
    for key in ("properties/irrigation/irr", "geometry/uid"):
        if key not in root:
            raise ValueError(
                f"Container {container_path} has no {key!r} array; irrigation-stratified "
                "transfer cannot assign a source class without it"
            )
    uids = [str(u) for u in np.asarray(root["geometry/uid"][:])]
    irr = np.asarray(root["properties/irrigation/irr"][:], dtype=float)
    if len(uids) != len(irr):
        raise ValueError(
            f"geometry/uid ({len(uids)}) and properties/irrigation/irr ({len(irr)}) "
            f"length mismatch in {container_path}"
        )
    if not np.isfinite(irr).all():
        bad = [uids[i] for i in np.flatnonzero(~np.isfinite(irr))]
        raise ValueError(
            f"Non-finite properties/irrigation/irr for {bad} in {container_path}; "
            "the class rule cannot be applied to a missing irrigation fraction"
        )
    irr_by_fid = {u: float(v) for u, v in zip(uids, irr)}
    class_by_fid = {u: ("irrigated" if v > threshold else "rainfed") for u, v in irr_by_fid.items()}
    return irr_by_fid, class_by_fid


def check_class_support(support_rows, min_class_train, out_of=None):
    """Raise if any stratified fold has fewer same-class training sites than allowed.

    The handoff forbids a silent fallback to the pooled vector, so an under-supported
    fold is a hard stop naming the fold, its class, the scheme, and the count.
    """
    for row in support_rows:
        for scheme in ("loro_strat", "loso_strat"):
            n = row[f"n_train_{scheme}"]
            if n < min_class_train:
                raise ValueError(
                    f"Inadequate same-class training support for fold {row['fid']!r} "
                    f"(scheme={scheme}, class={row['irr_class']}, region={row['region']}, "
                    f"n_train={n} < --min-class-train={min_class_train}"
                    + (f", cohort class size={out_of[row['irr_class']]}" if out_of else "")
                    + "). Refusing to fall back to the pooled vector; drop the fold from "
                    "the cohort or lower --min-class-train deliberately."
                )


# --------------------------------------------------------------------------- #
# Forward runs
# --------------------------------------------------------------------------- #
def run_fixed_params(cfg, container, params_by_fid):
    """Forward run with an explicit per-site parameter dict. {fid: DataFrame}."""
    fids = list(params_by_fid.keys())
    return ev.run_calibrated_model(cfg, container, fids, params_by_fid)


def run_default_params(cfg, container, fids):
    """Forward run with the model's generic/initial parameters. {fid: Series}.

    The run22 container carries an ingested calibration, so build_swim_input
    would normally load it even with calibrated_params_path=None. Force the
    no-calibration branch so every site runs with defaults (mirrors
    examples/6_Flux_International/derived_metrics.run_uncalibrated_model).
    """
    orig = swim_input_mod._container_has_calibration
    swim_input_mod._container_has_calibration = lambda c: False
    temp_h5 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            temp_h5 = tmp.name
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=None,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            stress_depletion_fraction=getattr(cfg, "stress_depletion_fraction", None),
        )
        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        results = {
            fid: pd.Series(output.eta[:, i], index=dates) for i, fid in enumerate(swim_input.fids)
        }
        swim_input.close()
    finally:
        swim_input_mod._container_has_calibration = orig
        if temp_h5 and os.path.exists(temp_h5):
            os.remove(temp_h5)
    return results


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(obs, mod):
    """n, r2, r, rmse, bias, abs_bias, kge, mae, alpha, beta on finite paired values."""
    mask = np.isfinite(obs) & np.isfinite(mod)
    o, m = obs[mask], mod[mask]
    out = {k: np.nan for k in METRIC_KEYS}
    out["n"] = int(len(o))
    if len(o) < MIN_DAILY_OBS:
        return out
    base = ev.calc_metrics(o, m)  # n, r2, r, rmse, bias
    out.update({k: base[k] for k in ["r2", "r", "rmse", "bias"]})
    # abs_bias = |MBE|; a sign-blind magnitude so over- and under-prediction are
    # penalized identically when comparing pooled against stratified vectors.
    out["abs_bias"] = float(abs(base["bias"])) if np.isfinite(base["bias"]) else np.nan
    so, mo = np.std(o), np.mean(o)
    alpha = float(np.std(m) / so) if so > 0 else np.nan
    beta = float(np.mean(m) / mo) if mo != 0 else np.nan
    out["mae"] = float(np.mean(np.abs(m - o)))
    out["alpha"] = alpha
    out["beta"] = beta
    if np.isfinite(base["r"]) and np.isfinite(alpha) and np.isfinite(beta):
        out["kge"] = float(1.0 - np.sqrt((base["r"] - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return out


def daily_paired_mask(flux, series_by_config):
    """Shared finite mask over dates common to flux and every config series."""
    idx = flux.index
    for s in series_by_config.values():
        idx = idx.intersection(s.index)
    if len(idx) < MIN_DAILY_OBS:
        return None, None
    finite = np.isfinite(flux.reindex(idx).values)
    for s in series_by_config.values():
        finite &= np.isfinite(s.reindex(idx).values)
    if int(finite.sum()) < MIN_DAILY_OBS:
        return None, None
    return idx, finite


def monthly_paired(flux, series_by_config):
    """Monthly per-config metrics on identical paired months (>=28 valid days)."""
    idx = flux.index
    for s in series_by_config.values():
        idx = idx.intersection(s.index)
    if len(idx) < MIN_DAILY_FOR_MONTHLY:
        return None
    flux_d = flux.loc[idx]
    monthly = {}
    flux_m_ref = None
    for cfg_name, s in series_by_config.items():
        s_m, flux_m = full_month_paired_sums(s.loc[idx], flux_d)
        monthly[cfg_name] = s_m
        flux_m_ref = flux_m
    all_idx = flux_m_ref.index
    paired = flux_m_ref.notna()
    for cfg_name in series_by_config:
        paired &= monthly[cfg_name].reindex(all_idx).notna()
    months = all_idx[paired]
    if len(months) < MIN_MONTHLY_OBS:
        return None
    obs = flux_m_ref.loc[months].values
    return {
        cfg_name: {**metrics(obs, monthly[cfg_name].reindex(months).values), "n": len(months)}
        for cfg_name in series_by_config
    }


# --------------------------------------------------------------------------- #
# Aggregation / bootstrap
# --------------------------------------------------------------------------- #
def stratum_frames(persite_df, regions):
    """(stratum, sub-frame) pairs for the reporting strata.

    ``all`` reproduces the pooled-only version of this script; the irrigation
    classes and the LORO regions are the stratified experiment's reporting axes.
    Each stratum is bootstrapped over its OWN common-support sites, so a stratum's
    n_common is not a subset count of the all-site n_common.
    """
    out = [("all", persite_df)]
    for cls in IRR_CLASSES:
        sub = persite_df[persite_df["irr_class"] == cls]
        if not sub.empty:
            out.append((cls, sub))
    for r in regions:
        sub = persite_df[persite_df["region"] == r]
        if not sub.empty:
            out.append((f"region:{r}", sub))
    return out


def bootstrap_medians(persite_df, configs, metric, n_boot, seed, delta_pairs=DELTA_PAIRS):
    """Paired site bootstrap of the per-config median and the paired config deltas.

    Resamples the common-support sites (rows finite for every config) with
    replacement; each draw is applied to all configs so the CIs are comparable and
    every delta is paired. Returns {config: (med, lo, hi)} plus one
    {'<a>_minus_<b>': (med, lo, hi)} entry per requested delta pair.
    """
    cols = [f"{c}_{metric}" for c in configs]
    common = persite_df[cols].dropna()
    rng = np.random.default_rng(seed)
    n = len(common)
    out = {}
    if n == 0:
        for c in configs:
            out[c] = (np.nan, np.nan, np.nan)
        return out, common.index.tolist()
    draws = rng.integers(0, n, size=(n_boot, n))
    arr = {c: common[f"{c}_{metric}"].values for c in configs}
    for c in configs:
        meds = np.median(arr[c][draws], axis=1)
        out[c] = (
            float(np.median(arr[c])),
            float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)),
        )
    for a, b in delta_pairs:
        if a in configs and b in configs:
            diff = np.median(arr[a][draws], axis=1) - np.median(arr[b][draws], axis=1)
            out[f"{a}_minus_{b}"] = (
                float(np.median(arr[a]) - np.median(arr[b])),
                float(np.percentile(diff, 2.5)),
                float(np.percentile(diff, 97.5)),
            )
    return out, common.index.tolist()


def win_rate(persite_df, a, b, metric, higher_better=True):
    """Fraction of common sites where config `a` beats config `b` on `metric`."""
    sub = persite_df[[f"{a}_{metric}", f"{b}_{metric}"]].dropna()
    if len(sub) == 0:
        return np.nan, 0
    av, bv = sub[f"{a}_{metric}"].values, sub[f"{b}_{metric}"].values
    wins = (av > bv).sum() if higher_better else (av < bv).sum()
    return float(wins) / len(sub), len(sub)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE), text=True
        ).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--par-csv", default=DEFAULT_PAR_CSV)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument(
        "--config", default=None, help="Config TOML (default: 5_Flux_Ensemble.toml)"
    )
    parser.add_argument("--shapefile", default=None, help="Cohort shapefile (default: cfg)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: {project_ws}/results/within_e2_transfer_irrigation_stratified; "
        "the pooled-only within_e2_transfer dir is never written by this script)",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--irr-threshold",
        type=float,
        default=0.5,
        help="Irrigated if properties/irrigation/irr exceeds this fraction (default: 0.5)",
    )
    parser.add_argument(
        "--min-class-train",
        type=int,
        default=5,
        help="Minimum same-class training sites per stratified fold; a fold below this "
        "raises rather than falling back to the pooled vector (default: 5)",
    )
    args = parser.parse_args()

    cfg = ev.load_config(args.config)
    flux_dir = ev.resolve_flux_dir(cfg)
    shp = args.shapefile or cfg.fields_shapefile
    # Non-clobbering default: the pooled-only artifacts under
    # results/within_e2_transfer/ stay exactly as they were produced.
    out_dir = (
        Path(args.out)
        if args.out
        else Path(cfg.project_ws) / "results" / "within_e2_transfer_irrigation_stratified"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Cohort + regions (geography only) ------------------------------------
    gdf = gpd.read_file(shp, engine="fiona")
    id_col = "site_id" if "site_id" in gdf.columns else cfg.feature_id_col
    gdf = gdf[[id_col, "state"]].copy()
    gdf["region"] = gdf["state"].map(STATE_TO_REGION)
    if gdf["region"].isna().any():
        missing = sorted(gdf.loc[gdf["region"].isna(), "state"].unique())
        raise ValueError(f"States without a region mapping: {missing}")
    region_by_fid = dict(zip(gdf[id_col].astype(str), gdf["region"]))

    # --- Irrigation class (remote sensing only) --------------------------------
    # Read before any flux file is opened, so the class assignment provably cannot
    # have been informed by validation truth.
    irr_by_fid_all, class_by_fid_all = read_irrigation_class(args.container, args.irr_threshold)
    container_class_counts = {
        cls: sum(1 for c in class_by_fid_all.values() if c == cls) for cls in IRR_CLASSES
    }
    print(
        f"Container irrigation classes (irr > {args.irr_threshold}): "
        + ", ".join(f"{cls}={container_class_counts[cls]}" for cls in IRR_CLASSES)
        + f"  (n={len(class_by_fid_all)})"
    )

    container = SwimContainer.open(args.container, mode="r")
    try:
        container_fids = set(container.field_uids)
        # PEST++ lowercases site tokens in the .par.csv; the container/shapefile
        # carry the canonical mixed-case fids. Match on lowercase, then relabel
        # the posterior table to the canonical fids the forward model expects.
        raw_table = per_site_median_table(args.par_csv)
        par_lower = set(raw_table.index)

        cohort = [
            s
            for s in gdf[id_col].astype(str).tolist()
            if s in container_fids and s.lower() in par_lower
        ]
        cohort = ev.apply_exclusions(cohort)  # drops MB_Pch etc.
        region_by_fid = {f: region_by_fid[f] for f in cohort}
        regions = sorted(set(region_by_fid.values()))
        table = raw_table.reindex([f.lower() for f in cohort])
        table.index = cohort
        table.index.name = "site"

        missing_class = [f for f in cohort if f not in class_by_fid_all]
        if missing_class:
            raise ValueError(
                f"Cohort sites absent from the container irrigation array: {missing_class}"
            )
        irr_by_fid = {f: irr_by_fid_all[f] for f in cohort}
        class_by_fid = {f: class_by_fid_all[f] for f in cohort}
        cohort_class_counts = {
            cls: sum(1 for c in class_by_fid.values() if c == cls) for cls in IRR_CLASSES
        }
        # class x region crosstab, recorded so the reader can see which stratified
        # LORO folds are supported by which part of the cohort.
        crosstab = {
            r: {
                cls: sum(1 for f in cohort if region_by_fid[f] == r and class_by_fid[f] == cls)
                for cls in IRR_CLASSES
            }
            for r in regions
        }

        print(f"Cohort: {len(cohort)} sites across {len(regions)} regions")
        for r in regions:
            members = [f for f in cohort if region_by_fid[f] == r]
            counts = " ".join(f"{cls[:4]}={crosstab[r][cls]}" for cls in IRR_CLASSES)
            print(f"  {r:<22} n={len(members):>2}  {counts:<16}{members}")
        print(
            "  Cohort classes: "
            + ", ".join(f"{cls}={cohort_class_counts[cls]}" for cls in IRR_CLASSES)
        )

        # --- Build parameter vectors (no flux involved) -----------------------
        # median-of-site-medians over a training subset; the four transfer schemes
        # differ ONLY in which rows of `table` enter the median.
        local_params = {f: table.loc[f].to_dict() for f in cohort}
        loso_params = {f: table.drop(index=f).median().to_dict() for f in cohort}
        loro_params, loro_train = {}, {}
        loso_strat_params, loso_strat_train = {}, {}
        loro_strat_params, loro_strat_train = {}, {}
        for f in cohort:
            train = [s for s in cohort if region_by_fid[s] != region_by_fid[f]]
            loro_train[f] = train
            loro_params[f] = table.loc[train].median().to_dict()

            # same irrigation class, held-out site itself excluded
            strat_loso = [s for s in cohort if s != f and class_by_fid[s] == class_by_fid[f]]
            loso_strat_train[f] = strat_loso
            loso_strat_params[f] = table.loc[strat_loso].median().to_dict()

            # same irrigation class AND outside the held-out site's region
            strat_loro = [
                s
                for s in cohort
                if class_by_fid[s] == class_by_fid[f] and region_by_fid[s] != region_by_fid[f]
            ]
            loro_strat_train[f] = strat_loro
            loro_strat_params[f] = table.loc[strat_loro].median().to_dict()

        # --- Fold support gate (hard stop; no pooled fallback) ------------------
        support_rows = [
            {
                "fid": f,
                "region": region_by_fid[f],
                "irr": irr_by_fid[f],
                "irr_class": class_by_fid[f],
                "n_train_loro": len(loro_train[f]),
                "n_train_loro_strat": len(loro_strat_train[f]),
                "n_train_loso_strat": len(loso_strat_train[f]),
            }
            for f in cohort
        ]
        check_class_support(support_rows, args.min_class_train, out_of=cohort_class_counts)
        support_df = pd.DataFrame(support_rows)
        support_df.to_csv(out_dir / "class_fold_support.csv", index=False)
        min_strat = int(support_df[["n_train_loro_strat", "n_train_loso_strat"]].min().min())
        print(
            f"  Stratified fold support OK: min same-class n_train={min_strat} "
            f"(>= --min-class-train={args.min_class_train})"
        )

        # --- Forward runs (6) --------------------------------------------------
        print("\nForward run: local site calibration (Run 22)...")
        local_res = run_fixed_params(cfg, container, local_params)
        print("Forward run: leave-one-site-out transfer (pooled)...")
        loso_res = run_fixed_params(cfg, container, loso_params)
        print("Forward run: leave-one-site-out transfer (irrigation-stratified)...")
        loso_strat_res = run_fixed_params(cfg, container, loso_strat_params)
        print("Forward run: leave-region-out transfer (pooled)...")
        loro_res = run_fixed_params(cfg, container, loro_params)
        print("Forward run: leave-region-out transfer (irrigation-stratified)...")
        loro_strat_res = run_fixed_params(cfg, container, loro_strat_params)
        print("Forward run: generic defaults (uncalibrated)...")
        default_res = run_default_params(cfg, container, cohort)

        # --- Score every config on identical paired days/months ----------------
        print("\nScoring against flux (daily + monthly, shared paired support)...")
        daily_rows, monthly_rows = [], []
        for fid in cohort:
            flux = ev.load_flux_et(fid, flux_dir)
            if flux.empty or not passes_site_minimum(flux):
                continue
            series = {
                "loro": loro_res[fid]["et_act"],
                "loro_strat": loro_strat_res[fid]["et_act"],
                "loso": loso_res[fid]["et_act"],
                "loso_strat": loso_strat_res[fid]["et_act"],
                "local": local_res[fid]["et_act"],
                "default": default_res[fid],
            }
            keys = {
                "fid": fid,
                "region": region_by_fid[fid],
                "irr_class": class_by_fid[fid],
                "irr": irr_by_fid[fid],
            }

            idx, finite = daily_paired_mask(flux, series)
            if idx is not None:
                obs = flux.reindex(idx).values[finite]
                drow = dict(keys)
                for c in CONFIGS:
                    mv = series[c].reindex(idx).values[finite]
                    m = metrics(obs, mv)
                    for k in ["n"] + METRIC_KEYS:
                        drow[f"{c}_{k}"] = m[k]
                daily_rows.append(drow)

            mm = monthly_paired(flux, series)
            if mm is not None:
                mrow = dict(keys)
                for c in CONFIGS:
                    for k in ["n"] + METRIC_KEYS:
                        mrow[f"{c}_{k}"] = mm[c][k]
                monthly_rows.append(mrow)

        daily_df = pd.DataFrame(daily_rows).set_index("fid")
        monthly_df = pd.DataFrame(monthly_rows).set_index("fid")
        daily_df.to_csv(out_dir / "persite_daily.csv")
        monthly_df.to_csv(out_dir / "persite_monthly.csv")

        # --- Summaries: median + bootstrap CI + win rates ----------------------
        # Every stratum is bootstrapped on its own common support (site as the
        # resample unit, paired across all six configs).
        summary_rows = []
        for basis, df in [("daily", daily_df), ("monthly", monthly_df)]:
            if df.empty:
                continue
            for stratum, sdf in stratum_frames(df, regions):
                for metric in SUMMARY_METRICS:
                    boot, common_sites = bootstrap_medians(
                        sdf, CONFIGS, metric, args.n_boot, args.seed
                    )
                    for c in CONFIGS:
                        med, lo, hi = boot[c]
                        self_med = float(sdf[f"{c}_{metric}"].median())
                        n_self = int(sdf[f"{c}_{metric}"].notna().sum())
                        summary_rows.append(
                            {
                                "basis": basis,
                                "stratum": stratum,
                                "metric": metric,
                                "config": c,
                                "label": CONFIG_LABELS[c],
                                "median_common": med,
                                "ci_lo": lo,
                                "ci_hi": hi,
                                "median_self": self_med,
                                "n_self": n_self,
                                "n_common": len(common_sites),
                            }
                        )
                    for a, b in DELTA_PAIRS:
                        key = f"{a}_minus_{b}"
                        if key in boot:
                            med, lo, hi = boot[key]
                            summary_rows.append(
                                {
                                    "basis": basis,
                                    "stratum": stratum,
                                    "metric": metric,
                                    "config": key,
                                    "label": f"{CONFIG_LABELS[a]} minus {b}",
                                    "median_common": med,
                                    "ci_lo": lo,
                                    "ci_hi": hi,
                                    "median_self": np.nan,
                                    "n_self": np.nan,
                                    "n_common": len(common_sites),
                                }
                            )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

        win_rows = []
        for basis, df in [("daily", daily_df), ("monthly", monthly_df)]:
            if df.empty:
                continue
            for stratum, sdf in stratum_frames(df, regions):
                for metric, hb in [
                    ("r2", True),
                    ("kge", True),
                    ("rmse", False),
                    ("abs_bias", False),
                    ("bias", None),
                ]:
                    if hb is None:
                        continue
                    for a, b in WIN_PAIRS:
                        wr, n = win_rate(sdf, a, b, metric, higher_better=hb)
                        win_rows.append(
                            {
                                "basis": basis,
                                "stratum": stratum,
                                "metric": metric,
                                "config": a,
                                "vs": b,
                                "win_rate": wr,
                                "n_common": n,
                            }
                        )
        win_df = pd.DataFrame(win_rows)
        win_df.to_csv(out_dir / "win_rates.csv", index=False)

        # --- Region breakdown (LORO held-out fold performance) ------------------
        region_rows = []
        if not daily_df.empty:
            for r in regions:
                sub = daily_df[daily_df["region"] == r]
                if sub.empty:
                    continue
                region_rows.append(
                    {
                        "region": r,
                        "n_sites": len(sub),
                        "n_irrigated": int((sub["irr_class"] == "irrigated").sum()),
                        "n_rainfed": int((sub["irr_class"] == "rainfed").sum()),
                        "loro_r2_med": float(sub["loro_r2"].median()),
                        "loro_strat_r2_med": float(sub["loro_strat_r2"].median()),
                        "local_r2_med": float(sub["local_r2"].median()),
                        "default_r2_med": float(sub["default_r2"].median()),
                        "loro_kge_med": float(sub["loro_kge"].median()),
                        "loro_strat_kge_med": float(sub["loro_strat_kge"].median()),
                        "local_kge_med": float(sub["local_kge"].median()),
                        "default_kge_med": float(sub["default_kge"].median()),
                        "loro_bias_med": float(sub["loro_bias"].median()),
                        "loro_strat_bias_med": float(sub["loro_strat_bias"].median()),
                        "local_bias_med": float(sub["local_bias"].median()),
                    }
                )
        region_df = pd.DataFrame(region_rows)
        region_df.to_csv(out_dir / "region_breakdown.csv", index=False)

        # --- Fold + vector provenance ------------------------------------------
        fold_rows = [
            {"fid": f, "region": region_by_fid[f], "n_train_loro": len(loro_train[f])}
            for f in cohort
        ]
        pd.DataFrame(fold_rows).to_csv(out_dir / "fold_definitions.csv", index=False)
        vectors = {
            "loro": loro_params,
            "loro_strat": loro_strat_params,
            "loso": loso_params,
            "loso_strat": loso_strat_params,
            "local": local_params,
        }
        with open(out_dir / "transfer_vectors.json", "w") as f:
            json.dump(vectors, f, indent=2)

        metadata = {
            "experiment": "within-E1 held-out spatial transfer (leave-region-out + LOSO), pooled and irrigation-stratified",
            "purpose": "support the main.md within-E1 held-out transfer claim and test irrigation-class regionalization of the E1 transfer vector",
            "source_par_csv": args.par_csv,
            "source_par_csv_sha256": _sha256(args.par_csv),
            "container": args.container,
            "config_toml": str(args.config or "5_Flux_Ensemble.toml (default)"),
            "aggregation": "median across TRAINING sites of each site's posterior median (median-of-site-medians); base realization excluded",
            "folds_defined_by": "fixed agroclimatic regions from shapefile state field; NO flux performance used",
            "flux_role": "validation only; never used in fold definition, aggregation, selection, or irrigation classification",
            "n_cohort": len(cohort),
            "regions": {r: [f for f in cohort if region_by_fid[f] == r] for r in regions},
            "state_to_region": STATE_TO_REGION,
            "param_families": PARAM_FAMILIES,
            "configs": CONFIGS,
            "config_labels": CONFIG_LABELS,
            "delta_pairs": [f"{a}_minus_{b}" for a, b in DELTA_PAIRS],
            "primary_delta": "loro_strat_minus_loro",
            "irrigation_class_rule": IRR_CLASS_RULE.format(threshold=args.irr_threshold),
            "irr_threshold": args.irr_threshold,
            "min_class_train": args.min_class_train,
            "irrigation_source": "container properties/irrigation/irr (remote-sensing irrigated fraction), read read-only in geometry/uid order",
            "container_class_counts": container_class_counts,
            "container_class_assignments": {
                fid: {"irr": irr_by_fid_all[fid], "irr_class": class_by_fid_all[fid]}
                for fid in sorted(class_by_fid_all)
            },
            "cohort_class_counts": cohort_class_counts,
            "cohort_class_assignments": {
                fid: {"irr": irr_by_fid[fid], "irr_class": class_by_fid[fid]} for fid in cohort
            },
            "class_by_region_crosstab": crosstab,
            "stratified_train_counts": {
                "loro_strat": {f: len(loro_strat_train[f]) for f in cohort},
                "loso_strat": {f: len(loso_strat_train[f]) for f in cohort},
            },
            "stratified_vector_policy": "per-site FIXED vector; class never varies by year; annual irrigation status still drives scheduler activation. No pooled fallback: a fold below min_class_train raises.",
            "strata_reported": ["all"] + IRR_CLASSES + [f"region:{r}" for r in regions],
            "n_boot": args.n_boot,
            "seed": args.seed,
            "date_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": _git_sha(),
        }
        with open(out_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # --- Console summary ---------------------------------------------------
        # The all-site block reproduces the pooled-only console output (widened to
        # fit the stratified labels); the class blocks below carry the stratified
        # experiment's headline numbers.
        def cell(frame, config, metric):
            row = frame[(frame["config"] == config) & (frame["metric"] == metric)]
            if row.empty:
                return "     n/a        "
            med, lo, hi = row.iloc[0][["median_common", "ci_lo", "ci_hi"]]
            return f"{med:6.3f}[{lo:5.2f},{hi:5.2f}]"

        print("\n" + "=" * 110)
        print("WITHIN-E1 HELD-OUT TRANSFER — median metrics (common-support sites)")
        print("=" * 110)
        for basis in ("daily", "monthly"):
            for stratum in ["all"] + IRR_CLASSES:
                sub = summary_df[
                    (summary_df["basis"] == basis) & (summary_df["stratum"] == stratum)
                ]
                if sub.empty:
                    continue
                n_common = int(sub["n_common"].iloc[0])
                print(f"\n{basis.upper()} [{stratum}] (n_common={n_common})")
                print(f"  {'config':<50}{'R2':>16}{'KGE':>16}{'bias':>16}")
                for c in CONFIGS:
                    print(
                        f"  {CONFIG_LABELS[c]:<50}"
                        f"{cell(sub, c, 'r2'):>16}{cell(sub, c, 'kge'):>16}"
                        f"{cell(sub, c, 'bias'):>16}"
                    )
                for a, b in DELTA_PAIRS:
                    key = f"{a}_minus_{b}"
                    row = sub[(sub["config"] == key) & (sub["metric"] == "kge")]
                    if row.empty:
                        continue
                    med, lo, hi = row.iloc[0][["median_common", "ci_lo", "ci_hi"]]
                    tag = "PRIMARY" if (a, b) == ("loro_strat", "loro") else ""
                    print(
                        f"  {a} - {b} (KGE)".ljust(50)
                        + f"delta={med:+.3f} [{lo:+.3f}, {hi:+.3f}]  {tag}"
                    )

        print(f"\nArtifacts written to: {out_dir}")
        for name in [
            "persite_daily.csv",
            "persite_monthly.csv",
            "summary_metrics.csv",
            "win_rates.csv",
            "region_breakdown.csv",
            "fold_definitions.csv",
            "class_fold_support.csv",
            "transfer_vectors.json",
            "run_metadata.json",
        ]:
            print(f"  {out_dir / name}")
    finally:
        container.close()


if __name__ == "__main__":
    main()

"""Ex5 -> Ex6 cropland parameter transferability stress test.

Applies the fixed Example 5 (Experiment 2) cropland median parameter vector to
every site in the canonical Example 6 (Experiment 3) 66-site international
cropland cohort, runs a forward simulation (no local recalibration), and scores
the transferred ET against flux-tower ET using the same validation rules as the
published E3 analysis (``evaluate.py``).

The transferred run is compared on common paired days/months against three
configurations drawn from the canonical E3 results so the comparison reconciles
with the published numbers:

    E3 calibrated   - site-specific E3 PEST++ IES calibration (per-site et_act)
    E3 uncalibrated - default/initial parameters (fresh forward run here)
    LS ensemble     - the E3 remote-sensing anchor (per-site et_rs); context only

``--params`` takes a single flat ``{param: value}`` vector applied to every site
(the pooled transfer). ``--params-by-site`` additionally takes a nested
``{site_id: {param: value}}`` mapping and scores it as the
``ex5_transfer_strat`` configuration -- the irrigation-stratified Run 22
transfer, in which each site receives the irrigated or the rainfed class vector
according to the canonical two-stage satellite irrigation classifier. Supplying
both yields all five handoff comparators from one invocation on identical common
support: E3 defaults, pooled Run 22 transfer, irrigation-stratified Run 22
transfer, local E3 satellite calibration, and the interpolated Landsat ensemble
context. Unlike the optional LULC comparator, the stratified mapping must cover
every cohort site: partial coverage raises rather than silently shrinking the
common support.

Transfer vectors are frozen upstream (``transfer/ex5_cropland_params.json``,
``paper/data/final/e2_run22_transfer_vector.json``,
``paper/data/final/e3_irrigation_stratified_param_mapping.json``) and are never
derived or tuned from Example 6 flux ET (see
``transfer/build_ex5_cropland_params.py``,
``transfer/build_ex5_irrigation_stratified_params.py``, and
``transfer/build_e3_irrigation_mapping.py``).

Outputs (under ``--out``, default ``{project_ws}/results/ex5_transfer_to_e3``):
    evaluation_metrics.csv            - Ex5-transferred daily per-site (E3 format)
    evaluation_monthly_metrics.csv    - Ex5-transferred monthly per-site
    pooled_metrics_daily.csv          - Ex5-transferred pooled (Volk methodology)
    pooled_metrics_monthly.csv
    transfer_comparison_summary.csv   - one row per config x {daily,monthly}
    transfer_comparison_persite.csv   - common-site daily metrics, all configs
    transfer_winrates.csv             - per-reference win rates vs each comparator
    run_metadata.json                 - inputs, params, rules, reconciliation
    {site}.csv                        - per-site ``--params`` daily series

With ``--params-by-site`` the stratified run gets its own parallel artifacts so
the two runs are never confused on disk: per-site series under
``ex5_transfer_strat/{site}.csv``, plus ``evaluation_metrics_strat.csv``,
``evaluation_monthly_metrics_strat.csv``, ``pooled_metrics_daily_strat.csv`` and
``pooled_metrics_monthly_strat.csv``.

``--require-empty-out`` (off by default, for backward compatibility) refuses to
run if the output directory already holds any ``*.csv``. Use it for any archived
run: ``_write_pooled_metrics`` re-reads ``{site}.csv`` off the output directory,
so a stale per-site CSV left by an earlier run would otherwise be pooled in
silently.

Usage:
    uv run python examples/6_Flux_International/transfer_ex5_params.py
    uv run python examples/6_Flux_International/transfer_ex5_params.py --sites US-KM1,DE-Kli
    uv run python examples/6_Flux_International/transfer_ex5_params.py \\
        --params /home/dgketchum/code/swim-rs/paper/data/final/e2_run22_transfer_vector.json \\
        --params-by-site \\
        /home/dgketchum/code/swim-rs/paper/data/final/e3_irrigation_stratified_param_mapping.json \\
        --out /data/ssd1/swim/6_Flux_International/results/e2_run22_transfer_by_irrigation_to_e3 \\
        --require-empty-out

This is a forward run with fixed parameters against existing container inputs.
It does NOT calibrate and does NOT call Earth Engine.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate as ev  # noqa: E402  (sibling module; needs HERE on sys.path)
import pooled_metrics as pm  # noqa: E402
from derived_metrics import run_uncalibrated_model  # noqa: E402

from swimrs.calibrate.flux_utils import (  # noqa: E402
    paired_monthly_sums,
    passes_site_minimum,
)
from swimrs.container import SwimContainer  # noqa: E402

DEFAULT_CONFIG = HERE / "6_Flux_International_LSEnsemble_POR_annual2yr.toml"
DEFAULT_PARAMS = HERE / "transfer" / "ex5_cropland_params.json"

# Validation gates, matched to evaluate.py / pooled_metrics.py (E3 protocol).
MIN_DAILY_OBS = 10  # paired days for a daily site metric
MIN_DAILY_FOR_MONTHLY = 30  # daily overlap needed before building months
MIN_DAYS_PER_MONTH = 20  # valid daily flux obs required to keep a month
MIN_MONTHLY_OBS = 6  # paired months for a monthly site metric

# Configuration display order and labels (core comparators). ``ex5_transfer_strat``
# is core -- when supplied it joins the common-site index and the paired
# win-rate/summary machinery -- but it is only instantiated when
# ``--params-by-site`` is given, so downstream code keys off frame presence.
STRAT_CONFIG = "ex5_transfer_strat"
CORE_CONFIGS = ["e3_uncal", "ex5_transfer", STRAT_CONFIG, "e3_cal", "ls_ensemble"]
CONFIG_LABELS = {
    "e3_uncal": "E3 uncalibrated/default",
    "ex5_transfer": "Ex5 transferred",
    STRAT_CONFIG: "Ex5 stratified transfer",
    "e3_cal": "E3 calibrated",
    "ls_ensemble": "LS ensemble",
    "lulc_defaults": "LULC defaults",
}
CONFIG_DESCRIPTIONS = {
    STRAT_CONFIG: (
        "irrigation-stratified Run 22 transfer: per-site vector selected by the "
        "canonical two-stage satellite irrigation class (equipped -> irrigated "
        "vector, not equipped -> rainfed vector)"
    ),
}


def full_metrics(obs, mod):
    """E3 calc_metrics (n, r2, r, rmse, bias, kge) plus mae, alpha, beta."""
    out = dict(ev.calc_metrics(obs, mod))
    mask = np.isfinite(obs) & np.isfinite(mod)
    o, m = obs[mask], mod[mask]
    if len(o) >= MIN_DAILY_OBS:
        so, mo = np.std(o), np.mean(o)
        out["mae"] = float(np.mean(np.abs(m - o)))
        out["alpha"] = float(np.std(m) / so) if so > 0 else np.nan
        out["beta"] = float(np.mean(m) / mo) if mo != 0 else np.nan
    else:
        out["mae"] = out["alpha"] = out["beta"] = np.nan
    return out


def daily_site_metrics(flux, model, rs):
    """Daily model-vs-flux metrics on rs-gated paired days (mirrors evaluate())."""
    common = model.index.intersection(flux.index)
    if len(common) < MIN_DAILY_OBS:
        return None
    obs = flux.loc[common].values
    mv = model.reindex(common).values
    rv = rs.reindex(common).values
    paired = np.isfinite(obs) & np.isfinite(mv) & np.isfinite(rv)
    if int(paired.sum()) < MIN_DAILY_OBS:
        return None
    m = full_metrics(obs[paired], mv[paired])
    m["n"] = int(paired.sum())
    return m


def monthly_site_metrics(flux, model, rs):
    """Monthly model-vs-flux metrics (mirrors evaluate_monthly()).

    Daily overlap >= 30, keep months with >= 20 valid daily flux obs, require
    >= 6 paired months. Months are flux-driven, so all configurations sharing
    this flux series are scored on identical paired months.
    """
    dc = model.index.intersection(flux.index)
    if len(dc) < MIN_DAILY_FOR_MONTHLY:
        return None
    model_d = model.loc[dc]
    flux_d = flux.loc[dc]
    rs_d = rs.reindex(dc)

    # sums over flux-valid days only, matching evaluate_monthly()
    model_m, flux_m, rs_m = paired_monthly_sums(
        model_d, flux_d, rs_d, month_min_days=MIN_DAYS_PER_MONTH
    )

    idx = flux_m.index
    paired_mask = flux_m.notna() & model_m.reindex(idx).notna() & rs_m.reindex(idx).notna()
    paired = idx[paired_mask]
    if len(paired) < MIN_MONTHLY_OBS:
        return None
    m = full_metrics(flux_m.loc[paired].values, model_m.reindex(paired).values)
    m["n"] = len(paired)
    return m


def _cohort_fids(cfg, container, sites_arg):
    """Resolve the cohort: explicit --sites, else shapefile cohort in container."""
    if sites_arg:
        return [s.strip() for s in sites_arg.split(",") if s.strip()]
    import geopandas as gpd

    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    id_col = cfg.feature_id_col if cfg.feature_id_col in gdf.columns else "sid"
    cohort = [str(s) for s in gdf[id_col].tolist()]
    container_fids = set(container.field_uids)
    fids = [f for f in cohort if f in container_fids]
    missing = [f for f in cohort if f not in container_fids]
    if missing:
        print(f"WARNING: {len(missing)} cohort site(s) absent from container: {missing}")
    return fids


def _run_fixed_params(cfg, container, fids, vector):
    """Forward run applying the same fixed parameter vector to every site."""
    params_by_fid = {fid: dict(vector) for fid in fids}
    return ev.run_calibrated_model(cfg, container, fids, params_by_fid)


def _load_params_by_site(path, fids):
    """Load a nested ``{site_id: {param: value}}`` mapping onto the cohort.

    Every cohort site must be present. The irrigation-stratified transfer is only
    interpretable against the pooled transfer on identical support, so missing
    sites raise with the full missing list instead of quietly reducing the cohort
    the way the optional LULC comparator does. Site ids are matched
    case-insensitively, but only after the completeness check, so a case
    mismatch can never be mistaken for absent coverage.

    Returns ``(params_by_fid, source_key_by_fid)``, both keyed by canonical
    container fid.
    """
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"--params-by-site: {path} is not a non-empty JSON object")

    flat = sorted(k for k, v in raw.items() if not isinstance(v, dict))
    if flat:
        raise ValueError(
            f"--params-by-site expects a nested {{site_id: {{param: value}}}} mapping, but "
            f"{path} has non-dict value(s) for {flat[:5]}. A flat {{param: value}} vector "
            "belongs on --params (broadcasting it here would corrupt into "
            "{fid: {site: value}})."
        )

    lowered = {}
    for site, vector in raw.items():
        key = str(site).lower()
        if key in lowered:
            raise ValueError(
                f"--params-by-site: site ids collide when lowercased in {path} "
                f"({lowered[key][0]!r} vs {site!r})"
            )
        lowered[key] = (str(site), {str(p): float(v) for p, v in vector.items()})

    missing = sorted(f for f in fids if f.lower() not in lowered)
    if missing:
        raise ValueError(
            f"--params-by-site: {path} covers {len(raw)} site(s) but is missing "
            f"{len(missing)} of the {len(fids)} cohort site(s): {missing}. The "
            "irrigation-stratified transfer must cover the whole cohort; filtering "
            "down would silently change the common support the five comparators are "
            "scored on."
        )

    param_sets = {frozenset(v[1]) for k, v in lowered.items() if k in {f.lower() for f in fids}}
    if len(param_sets) != 1:
        raise ValueError(
            f"--params-by-site: parameter names differ across sites in {path}; "
            f"found {len(param_sets)} distinct parameter sets"
        )

    params_by_fid = {f: dict(lowered[f.lower()][1]) for f in fids}
    source_keys = {f: lowered[f.lower()][0] for f in fids}
    return params_by_fid, source_keys


def _vector_sha256(vector):
    """Stable hash of one parameter vector (sorted keys, JSON-encoded floats)."""
    payload = json.dumps({k: float(vector[k]) for k in sorted(vector)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_class_assignment(params_by_fid, mapping_path):
    """Recover the class assignment the forward run actually applied.

    Sites are grouped by the hash of the vector handed to them, so the recorded
    assignment reflects the vectors used rather than a claim made elsewhere. Group
    names are taken from the mapping's companion ``*_metadata.json``
    (``class_vectors``) when its vectors hash-match; otherwise groups fall back to
    ``vector_1``, ``vector_2``, ... in hash order.
    """
    by_hash = {}
    for fid, vec in params_by_fid.items():
        by_hash.setdefault(_vector_sha256(vec), []).append(fid)

    mapping_path = Path(mapping_path)
    meta_path = mapping_path.with_name(mapping_path.stem + "_metadata.json")
    names = {}
    companion = None
    if meta_path.exists():
        companion = str(meta_path)
        try:
            with open(meta_path) as f:
                doc = json.load(f)
            for cls, vec in (doc.get("class_vectors") or {}).items():
                digest = _vector_sha256({str(k): float(v) for k, v in vec.items()})
                if digest in by_hash:
                    names[digest] = str(cls)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            names = {}
    for i, digest in enumerate(sorted(by_hash), start=1):
        names.setdefault(digest, f"vector_{i}")

    assignment = {fid: names[d] for d, group in by_hash.items() for fid in sorted(group)}
    counts = {names[d]: len(group) for d, group in by_hash.items()}
    vectors = {names[d]: params_by_fid[sorted(group)[0]] for d, group in by_hash.items()}
    hashes = {names[d]: d for d in by_hash}
    return assignment, counts, vectors, hashes, companion


def _load_e3_calibrated(e3_results_dir, fid):
    """Read calibrated SWIM et_act and LS-ensemble et_rs from canonical {fid}.csv."""
    path = Path(e3_results_dir) / f"{fid}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.normalize()
    return df


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE), text=True
        ).strip()
    except Exception:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _median_row(df, basis, n_common):
    """Build a summary row of medians for one config/basis over the given frame."""
    return {
        "n_sites_self": int(len(df)),
        "n_sites_common": int(n_common),
        "r2_med": df["r2"].median(),
        "kge_med": df["kge"].median(),
        "rmse_med": df["rmse"].median(),
        "bias_med": df["bias"].median(),
        "mae_med": df["mae"].median(),
        "alpha_med": df["alpha"].median(),
        "beta_med": df["beta"].median(),
    }


def _win_rate(ex5_df, other_df, col):
    """Fraction of common sites where Ex5 beats `other` on `col` (higher better)."""
    common = ex5_df.index.intersection(other_df.index)
    if len(common) == 0:
        return np.nan, 0
    a = ex5_df.loc[common, col]
    b = other_df.loc[common, col]
    valid = a.notna() & b.notna()
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    return float((a[valid] > b[valid]).sum()) / n, n


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--params", default=str(DEFAULT_PARAMS), help="Ex5 transfer vector JSON")
    parser.add_argument(
        "--params-by-site",
        default=None,
        help="Optional nested {site_id: {param: value}} JSON scored as the "
        "irrigation-stratified Run 22 transfer (config 'ex5_transfer_strat'). Must cover "
        "every cohort site; missing sites raise.",
    )
    parser.add_argument(
        "--sites", default=None, help="Comma-separated site IDs (default: full cohort)"
    )
    parser.add_argument("--container", default=None, help="Override container path")
    parser.add_argument(
        "--e3-results-dir",
        default=None,
        help="Canonical E3 results dir (for E3 calibrated + LS ensemble)",
    )
    parser.add_argument(
        "--out", default=None, help="Output dir (default: {project_ws}/results/ex5_transfer_to_e3)"
    )
    parser.add_argument(
        "--lulc-params",
        default=None,
        help="Optional LULC-default params JSON to add as a second no-calibration comparator. "
        "Off by default; the legacy lulc_global_params.json predates the 66-site cohort and "
        "should be regenerated before formal manuscript use.",
    )
    parser.add_argument(
        "--require-empty-out",
        action="store_true",
        help="Refuse to run if --out already contains any *.csv. Off by default for "
        "backward compatibility; recommended for archived runs, because pooled metrics "
        "are re-read from the per-site CSVs in that directory.",
    )
    args = parser.parse_args()

    conf_path = Path(args.config)
    cfg = ev._load_config(conf_path)
    container_path = args.container or ev._default_container_path(cfg)
    e3_results_dir = args.e3_results_dir or ev._results_dir(cfg, conf_path)
    out_dir = (
        Path(args.out)
        if args.out
        else Path(ev._results_dir(cfg, conf_path)).parent / "ex5_transfer_to_e3"
    )
    if args.require_empty_out and out_dir.exists():
        existing = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.csv"))
        if existing:
            raise FileExistsError(
                f"--require-empty-out: {out_dir} already contains {len(existing)} CSV file(s), "
                f"e.g. {existing[:5]}. Pooled metrics are re-read from the per-site {{site}}.csv "
                "in this directory, so a stale file from a prior run would be pooled in "
                "silently. Move or delete the directory, or point --out somewhere new."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    strat_dir = out_dir / STRAT_CONFIG

    with open(args.params) as f:
        vector = json.load(f)
    if not all(isinstance(v, int | float) for v in vector.values()):
        raise ValueError(
            f"--params expects a flat {{param: value}} vector; {args.params} holds nested "
            "values. A per-site mapping belongs on --params-by-site."
        )

    flux_sources = {}
    if Path(cfg.fields_shapefile).exists():
        flux_sources = ev.load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)

    container = SwimContainer.open(container_path, mode="r")
    try:
        fids = _cohort_fids(cfg, container, args.sites)
        print(f"Cohort: {len(fids)} sites")
        print(f"Container: {container_path}")
        print(f"E3 results dir (calibrated + LS ensemble): {e3_results_dir}")
        print(f"Transfer vector: {args.params}")
        print(f"  {vector}\n")

        strat_params, strat_meta = {}, None
        if args.params_by_site:
            strat_params, strat_source_keys = _load_params_by_site(args.params_by_site, fids)
            assignment, counts, class_vectors, class_hashes, companion = _resolve_class_assignment(
                strat_params, args.params_by_site
            )
            strat_meta = {
                "path": str(args.params_by_site),
                "sha256": _sha256(args.params_by_site),
                "companion_metadata": companion,
                "class_assignment": assignment,
                "class_counts": counts,
                "class_vectors": class_vectors,
                "class_vector_sha256": class_hashes,
                "source_site_keys": strat_source_keys,
            }
            print(f"Stratified mapping: {args.params_by_site}")
            print(f"  sha256           : {strat_meta['sha256']}")
            print(f"  class counts     : {counts}")
            for cls in sorted(class_vectors):
                print(f"  {cls:<16} : {class_vectors[cls]}")
            print()

        print("Forward run: Ex5 transferred (fixed cropland vector)...")
        ex5_results = _run_fixed_params(cfg, container, fids, vector)

        strat_results = {}
        if strat_params:
            print(f"Forward run: Ex5 stratified transfer ({len(strat_params)} per-site vectors)...")
            strat_results = _run_fixed_params_dict(cfg, container, fids, strat_params)

        print("Forward run: E3 uncalibrated (default parameters)...")
        uncal_results = run_uncalibrated_model(cfg, container, fids)

        lulc_results = {}
        if args.lulc_params:
            with open(args.lulc_params) as f:
                lulc_params = json.load(f)
            lulc_params_lower = {k.lower(): v for k, v in lulc_params.items()}
            lulc_fids = [f for f in fids if f.lower() in lulc_params_lower]
            lulc_subset = {f: lulc_params_lower[f.lower()] for f in lulc_fids}
            print(f"Forward run: LULC defaults ({len(lulc_fids)} sites with params)...")
            lulc_results = _run_fixed_params_dict(cfg, container, lulc_fids, lulc_subset)

        # Build RS (LS ensemble) ETa per site from the canonical per-site CSV
        # (et_rs); fall back to recomputation if the CSV is absent.
        print("\nScoring all configurations against flux (daily + monthly)...")
        active = [c for c in CORE_CONFIGS if c != STRAT_CONFIG or strat_results]
        if lulc_results:
            active.append("lulc_defaults")
        per_daily = {c: {} for c in active}
        per_monthly = {c: {} for c in per_daily}
        if strat_results:
            strat_dir.mkdir(parents=True, exist_ok=True)

        for fid in fids:
            flux = ev.load_flux_et(fid, flux_sources.get(fid))
            if flux.empty or not passes_site_minimum(flux):
                continue

            ex5_df = ex5_results.get(fid)
            if ex5_df is None:
                continue

            e3_csv = _load_e3_calibrated(e3_results_dir, fid)
            if e3_csv is not None and "et_rs" in e3_csv:
                rs = e3_csv["et_rs"]
            else:
                rs = ev._build_rs_eta_series(container, cfg, fid, ex5_df["etref"])
            if rs is None:
                continue

            series_by_config = {
                "ex5_transfer": ex5_df["et_act"],
                "ls_ensemble": rs,
            }
            unc = uncal_results.get(fid)
            if unc is not None:
                series_by_config["e3_uncal"] = unc
            if e3_csv is not None and "et_act" in e3_csv:
                series_by_config["e3_cal"] = e3_csv["et_act"]
            if fid in lulc_results:
                series_by_config["lulc_defaults"] = lulc_results[fid]["et_act"]
            strat_df = strat_results.get(fid)
            if strat_df is not None:
                series_by_config[STRAT_CONFIG] = strat_df["et_act"]

            for config, series in series_by_config.items():
                dm = daily_site_metrics(flux, series, rs)
                if dm is not None:
                    per_daily[config][fid] = dm
                mm = monthly_site_metrics(flux, series, rs)
                if mm is not None:
                    per_monthly[config][fid] = mm

            # Write per-site series (E3 {site}.csv layout). The top-level CSVs are
            # the --params run; the stratified run gets its own subdirectory so the
            # two are never confused, and so pooled metrics can be accumulated for
            # each without cross-contamination.
            site_out = ex5_df.copy()
            site_out["et_rs"] = rs.reindex(site_out.index)
            site_out.to_csv(out_dir / f"{fid}.csv")
            if strat_df is not None:
                strat_out = strat_df.copy()
                strat_out["et_rs"] = rs.reindex(strat_out.index)
                strat_out.to_csv(strat_dir / f"{fid}.csv")
    finally:
        container.close()

    daily_df = {c: pd.DataFrame(d).T for c, d in per_daily.items()}
    monthly_df = {c: pd.DataFrame(d).T for c, d in per_monthly.items()}

    _write_evaluation_metrics(out_dir, daily_df, monthly_df)
    _write_pooled_metrics(out_dir, fids, flux_sources)
    if strat_results:
        _write_evaluation_metrics(
            out_dir, daily_df, monthly_df, config=STRAT_CONFIG, suffix="_strat"
        )
        _write_pooled_metrics(
            out_dir,
            fids,
            flux_sources,
            label="Ex5 stratified transfer",
            suffix="_strat",
            series_dir=strat_dir,
        )
    common_daily, common_monthly = _write_comparison(out_dir, daily_df, monthly_df)
    winrates = _write_winrates(out_dir, daily_df, monthly_df)
    _write_metadata(
        out_dir,
        args,
        conf_path,
        container_path,
        e3_results_dir,
        vector,
        fids,
        daily_df,
        monthly_df,
        common_daily,
        common_monthly,
        bool(lulc_results),
        strat_meta,
        strat_dir if strat_results else None,
    )

    _print_report(daily_df, monthly_df, common_daily, common_monthly, winrates)
    print(f"\nOutputs written to {out_dir}")


def _run_fixed_params_dict(cfg, container, fids, params_by_fid):
    """Forward run with a per-site params dict (LULC defaults, stratified transfer)."""
    return ev.run_calibrated_model(cfg, container, fids, params_by_fid)


def _write_evaluation_metrics(out_dir, daily_df, monthly_df, config="ex5_transfer", suffix=""):
    """One config's standalone metrics in the E3 evaluate.py column layout."""
    for basis, frames, fname in [
        ("daily", daily_df, f"evaluation_metrics{suffix}.csv"),
        ("monthly", monthly_df, f"evaluation_monthly_metrics{suffix}.csv"),
    ]:
        ex5 = frames.get(config)
        ls = frames.get("ls_ensemble")
        if ex5 is None or len(ex5) == 0 or ls is None or len(ls) == 0:
            continue
        common = ex5.index.intersection(ls.index)
        rows = []
        for fid in common:
            e = ex5.loc[fid]
            r = ls.loc[fid]
            row = {"fid": fid, "n": int(e["n"])}
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[k] = e[k]
                row[f"{k}_swim"] = e[k]
                row[f"{k}_rs"] = r[k]
            rows.append(row)
        pd.DataFrame(rows).set_index("fid").to_csv(out_dir / fname)


def _write_pooled_metrics(
    out_dir, fids, flux_sources=None, label="Ex5 transferred", suffix="", series_dir=None
):
    """Pooled (Volk) metrics for one fixed-params run, reusing pooled_metrics.

    ``series_dir`` is where the per-site ``{fid}.csv`` series are read from
    (default: ``out_dir`` itself, the ``--params`` run); the CSVs always land in
    ``out_dir``.
    """
    out_dir = Path(out_dir)
    series_dir = str(series_dir or out_dir)
    d_station, d_obs, d_mod = pm._accumulate(
        series_dir, fids, monthly=False, flux_sources=flux_sources
    )
    daily = pm._report(f"DAILY ({label})", "mm/day", d_station, d_obs, d_mod)
    daily.to_csv(out_dir / f"pooled_metrics_daily{suffix}.csv", index=False)
    m_station, m_obs, m_mod = pm._accumulate(
        series_dir, fids, monthly=True, flux_sources=flux_sources
    )
    monthly = pm._report(f"MONTHLY ({label})", "mm/month", m_station, m_obs, m_mod)
    monthly.to_csv(out_dir / f"pooled_metrics_monthly{suffix}.csv", index=False)


def _common_index(frames, configs):
    """Sites present in every listed config's frame."""
    idx = None
    for c in configs:
        f = frames.get(c)
        if f is None or len(f) == 0:
            return pd.Index([])
        idx = f.index if idx is None else idx.intersection(f.index)
    return idx if idx is not None else pd.Index([])


def _core_configs(frames):
    """Core configs actually instantiated for this run.

    ``ex5_transfer_strat`` is core but only exists when ``--params-by-site`` was
    supplied, so the common-site index is taken over the core configs that were
    run. This keys off which forward runs happened, not off which sites paired --
    site-level support is never filtered here.
    """
    return [c for c in CORE_CONFIGS if c in frames]


def _write_comparison(out_dir, daily_df, monthly_df):
    """Common-site per-config comparison summary + per-site daily table."""
    common_daily = _common_index(daily_df, _core_configs(daily_df))
    common_monthly = _common_index(monthly_df, _core_configs(monthly_df))

    all_configs = list(daily_df.keys())  # core (+ strat) + optional lulc
    summary_rows = []
    for basis, frames, common in [
        ("daily", daily_df, common_daily),
        ("monthly", monthly_df, common_monthly),
    ]:
        for config in all_configs:
            f = frames.get(config)
            if f is None or len(f) == 0:
                continue
            on_common = f.loc[f.index.intersection(common)]
            row = {"config": CONFIG_LABELS[config], "basis": basis}
            row.update(_median_row(on_common if len(on_common) else f, basis, len(on_common)))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "transfer_comparison_summary.csv", index=False)

    # Per-site daily common-site table (all configs as columns).
    persite = {}
    for config in all_configs:
        f = daily_df.get(config)
        if f is None:
            continue
        sub = f.loc[f.index.intersection(common_daily)]
        for metric in ["kge", "r2", "rmse", "bias", "mae"]:
            persite[f"{config}_{metric}"] = sub[metric]
    pd.DataFrame(persite).sort_index().to_csv(out_dir / "transfer_comparison_persite.csv")
    return common_daily, common_monthly


def _write_winrates(out_dir, daily_df, monthly_df):
    """Per-reference R2/KGE win rates vs each comparator (daily + monthly).

    The pooled transfer is always a reference. When the irrigation-stratified
    transfer was run it is a second reference, so the stratified-minus-pooled
    comparison -- the primary question -- appears as its own row.
    """
    references = ["ex5_transfer"] + ([STRAT_CONFIG] if STRAT_CONFIG in daily_df else [])
    rows = []
    for ref in references:
        ref_d = daily_df.get(ref)
        ref_m = monthly_df.get(ref)
        comparators = [c for c in CORE_CONFIGS + ["lulc_defaults"] if c != ref and c in daily_df]
        for config in comparators:
            d = daily_df.get(config)
            m = monthly_df.get(config)
            dr2, n_dr2 = _win_rate(ref_d, d, "r2") if d is not None else (np.nan, 0)
            dkge, _ = _win_rate(ref_d, d, "kge") if d is not None else (np.nan, 0)
            mr2, n_mr2 = _win_rate(ref_m, m, "r2") if m is not None else (np.nan, 0)
            mkge, _ = _win_rate(ref_m, m, "kge") if m is not None else (np.nan, 0)
            rows.append(
                {
                    "reference": CONFIG_LABELS[ref],
                    "comparison": f"{CONFIG_LABELS[ref]} vs {CONFIG_LABELS[config]}",
                    "n_daily": n_dr2,
                    "daily_r2_win": dr2,
                    "daily_kge_win": dkge,
                    "n_monthly": n_mr2,
                    "monthly_r2_win": mr2,
                    "monthly_kge_win": mkge,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "transfer_winrates.csv", index=False)
    return df


def _reconcile(daily_df, monthly_df, e3_results_dir):
    """Cross-check recomputed E3-calibrated / LS metrics vs published E3 CSVs.

    Per-site max-abs difference is the strong check (valid on any cohort subset);
    medians are reported too but only equal the published medians on the full
    cohort. A near-zero per-site delta confirms the transfer script reproduces
    the published E3 pairing protocol exactly.
    """
    out = {}
    for basis, frames, fname in [
        ("daily", daily_df, "evaluation_metrics.csv"),
        ("monthly", monthly_df, "evaluation_monthly_metrics.csv"),
    ]:
        pub_path = Path(e3_results_dir) / fname
        if not pub_path.exists():
            continue
        pub = pd.read_csv(pub_path).set_index("fid")
        for config, swimcol, r2col in [
            ("e3_cal", "kge_swim", "r2_swim"),
            ("ls_ensemble", "kge_rs", "r2_rs"),
        ]:
            f = frames.get(config)
            if f is None or len(f) == 0 or swimcol not in pub:
                continue
            common = f.index.intersection(pub.index)
            entry = {
                "n_compared": int(len(common)),
                "kge_med_recomputed": float(f["kge"].median()),
                "kge_med_published_full_cohort": float(pub[swimcol].median()),
            }
            if len(common):
                entry["kge_persite_max_abs_diff"] = float(
                    (f.loc[common, "kge"] - pub.loc[common, swimcol]).abs().max()
                )
                entry["r2_persite_max_abs_diff"] = float(
                    (f.loc[common, "r2"] - pub.loc[common, r2col]).abs().max()
                )
            out[f"{basis}_{config}"] = entry
    return out


def _write_metadata(
    out_dir,
    args,
    conf_path,
    container_path,
    e3_results_dir,
    vector,
    fids,
    daily_df,
    monthly_df,
    common_daily,
    common_monthly,
    has_lulc,
    strat_meta=None,
    strat_dir=None,
):
    meta = {
        "config": str(conf_path),
        "container": str(container_path),
        "e3_results_dir": str(e3_results_dir),
        "out_dir": str(out_dir),
        "transfer_params_path": str(args.params),
        "transfer_vector": vector,
        "git_sha": _git_sha(),
        "cohort_size": len(fids),
        "n_daily_paired": {c: int(len(f)) for c, f in daily_df.items()},
        "n_monthly_paired": {c: int(len(f)) for c, f in monthly_df.items()},
        "n_common_daily_sites": int(len(common_daily)),
        "n_common_monthly_sites": int(len(common_monthly)),
        "lulc_comparator_included": has_lulc,
        "stratified_comparator_included": bool(strat_meta),
        "persite_series_config": "ex5_transfer",
        "require_empty_out": bool(getattr(args, "require_empty_out", False)),
        "metric_rules": {
            "daily_min_paired_obs": MIN_DAILY_OBS,
            "daily_basis": "rs-gated paired days (finite flux + model + LS ensemble)",
            "monthly_min_daily_overlap": MIN_DAILY_FOR_MONTHLY,
            "monthly_min_days_per_month": MIN_DAYS_PER_MONTH,
            "monthly_min_paired_months": MIN_MONTHLY_OBS,
            "flux_ET": "ET_corr preferred; raw ET fallback (E3 rule)",
        },
        "reconciliation_kge_med": _reconcile(daily_df, monthly_df, e3_results_dir),
    }
    if strat_meta:
        meta["stratified_transfer"] = {
            "config_key": STRAT_CONFIG,
            "label": CONFIG_LABELS[STRAT_CONFIG],
            "description": CONFIG_DESCRIPTIONS[STRAT_CONFIG],
            "params_by_site_path": strat_meta["path"],
            "params_by_site_sha256": strat_meta["sha256"],
            "mapping_companion_metadata": strat_meta["companion_metadata"],
            "class_assignment": strat_meta["class_assignment"],
            "class_counts": strat_meta["class_counts"],
            "class_vectors": strat_meta["class_vectors"],
            "class_vector_sha256": strat_meta["class_vector_sha256"],
            "class_assignment_source": (
                "recovered from the parameter vector actually applied to each site "
                "(sites grouped by vector hash); group names taken from the mapping's "
                "companion metadata class_vectors when they hash-match"
            ),
            "mapping_site_keys": strat_meta["source_site_keys"],
            "coverage_rule": (
                "every cohort site must appear in the mapping; missing sites raise "
                "rather than reducing the common support"
            ),
            "persite_series_dir": str(strat_dir) if strat_dir else None,
            "annual_stage2_note": (
                "annual (stage 2) irrigation classification continues to control "
                "scheduler activation year by year; it does not change a site's "
                "parameter vector"
            ),
        }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def _print_report(daily_df, monthly_df, common_daily, common_monthly, winrates):
    def table(title, frames, common):
        print(f"\n{title} (common sites: {len(common)})")
        print(
            f"  {'Configuration':<26} {'N':>4} {'R2med':>7} {'KGEmed':>7} {'RMSEmed':>8} {'Biasmed':>8} {'MAEmed':>7}"
        )
        print("  " + "-" * 70)
        for config in daily_df.keys():
            f = frames.get(config)
            if f is None or len(f) == 0:
                continue
            sub = f.loc[f.index.intersection(common)]
            if len(sub) == 0:
                sub = f
            print(
                f"  {CONFIG_LABELS[config]:<26} {len(sub):>4d} {sub['r2'].median():>7.3f} "
                f"{sub['kge'].median():>7.3f} {sub['rmse'].median():>8.3f} "
                f"{sub['bias'].median():>+8.3f} {sub['mae'].median():>7.3f}"
            )

    print("\n" + "=" * 74)
    print("EX5 -> EX6 CROPLAND PARAMETER TRANSFER")
    print("=" * 74)
    table("DAILY common-site medians", daily_df, common_daily)
    table("MONTHLY common-site medians", monthly_df, common_monthly)

    if STRAT_CONFIG in daily_df:
        print("\nPaired median delta, stratified minus pooled transfer (common sites):")
        for basis, frames, common in [
            ("daily", daily_df, common_daily),
            ("monthly", monthly_df, common_monthly),
        ]:
            a = frames.get(STRAT_CONFIG)
            b = frames.get("ex5_transfer")
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                continue
            idx = a.index.intersection(b.index).intersection(common)
            if len(idx) == 0:
                continue
            deltas = " ".join(
                f"{k}={(a.loc[idx, k] - b.loc[idx, k]).median():+.4f}"
                for k in ["r2", "kge", "rmse", "bias", "mae"]
            )
            print(f"  {basis:<8} n={len(idx):>3}  {deltas}")

    print("\nWin rates (reference wins fraction):")
    print(winrates.to_string(index=False))


if __name__ == "__main__":
    main()

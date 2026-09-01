"""Evaluate calibrated SWIM against flux-tower ET and the OpenET benchmark.

Runs the calibrated model in forecast mode and compares SWIM ET with
closure-corrected flux-tower ET alongside the six OpenET v2.1 member models
and the MAD-filtered ensemble supplied in the Volk extraction.

For the canonical benchmark, sparse 3 x 3 capture ET is divided by same-day
OpenET bias-corrected GridMET grass-reference ETo, ETf is reconstructed with
the shared openet-core temporal-support semantics, and daily ET is recovered
with the same ETo. Container ETf is available only through the explicitly
diagnostic DIY source.

The default scientific output is the grouped SWIM-OpenET benchmark defined in
``examples/VALIDATION_POLICY.md`` ("SWIM-OpenET Benchmark Aggregation"):
pooled KGE/RMSE/MBE/r/r^2/slope0 on the concatenated exactly-paired cohort and
sqrt(n)-weighted site KGE/RMSE/MBE, with whole-site bootstrap intervals and
SWIM-minus-OpenET contrasts. Per-site metric tables remain as secondary
diagnostics and compatibility artifacts.

Usage:
    python evaluate.py [--par-csv PATH] [--sites SITE1,SITE2] [--output-dir DIR]
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from swimrs.calibrate.benchmark import (
    BenchmarkConstructionError,
    reconstruct_daily_benchmark,
)
from swimrs.calibrate.flux_utils import (
    full_month_paired_sums,
    passes_site_minimum,
    write_excluded_sites,
)
from swimrs.container import SwimContainer

# The grouped-benchmark estimators, contract constants, and the paired-record
# contract live in the pure shared module (swimrs.evaluation.benchmark); they
# are re-exported here so path-importing callers and tests keep working.
from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    BENCHMARK_LABELS,  # noqa: F401  (re-export)
    BENCHMARK_SOURCE_MACHINE_TOKENS,  # noqa: F401  (re-export)
    BOOTSTRAP_REPS_DEFAULT,
    BOOTSTRAP_SEED_DEFAULT,
    CONSTRUCTION_TOKENS,  # noqa: F401  (re-export)
    ERROR_METRIC_UNITS,
    FAVORABLE_DIRECTION,  # noqa: F401  (re-export)
    GROUPED_CONTRAST_COLUMNS,  # noqa: F401  (re-export)
    GROUPED_FORMULAS,  # noqa: F401  (re-export)
    GROUPED_MASK_DEFINITION,  # noqa: F401  (re-export)
    GROUPED_METRIC_COLUMNS,  # noqa: F401  (re-export)
    GROUPED_MODEL_ORDER,
    MIN_OBS_FOR_METRICS,
    MODEL_DISPLAY,
    MOMENT_IDENTITY_ATOL,  # noqa: F401  (re-export)
    PAIRED_RECORD_COLUMNS,
    PAIRED_RECORD_FILENAME,
    PAIRED_RECORD_SCHEMA_VERSION,
    PAIRED_RECORD_SORT_ORDER,
    POOLED_METRICS,
    PRIMARY_METRICS,  # noqa: F401  (re-export)
    SUPPORT_CLASSES,
    TEMPORAL_CLASS_DEFINITION,
    TEMPORAL_CLASSES,
    WEIGHTED_METRICS,
    GroupedEstimationError,
    PairedSiteSeries,
    _bootstrap_multiplicities,  # noqa: F401  (re-export)
    bootstrap_grouped,  # noqa: F401  (re-export)
    bootstrap_grouped_from_counts,  # noqa: F401  (re-export)
    build_paired_site_series,  # noqa: F401  (re-export)
    grouped_metadata,
    grouped_metric_tables,
    grouped_point_estimates,
    paired_records_from_frame,  # noqa: F401  (re-export)
    paired_records_to_frame,
    pooled_metrics,  # noqa: F401  (re-export)
    pooled_metrics_direct,  # noqa: F401  (re-export)
    pooled_metrics_from_moments,  # noqa: F401  (re-export)
    read_paired_record_frame,
    site_effect_summary,
    site_metric_triads,  # noqa: F401  (re-export)
    site_secondary_metrics,
    site_sufficient_stats,  # noqa: F401  (re-export)
    sqrt_n_weighted_metrics,  # noqa: F401  (re-export)
    validate_paired_record_frame,
    write_paired_record_frame,
)
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.swim.config import ProjectConfig

OPEN_SOURCE_MODELS = ["geesebal", "ptjpl", "ssebop", "sims", "eemetric", "disalexi"]

EXCLUDED_SITES = {"MB_Pch"}


def apply_exclusions(fids):
    """Filter site list through the canonical exclusion policy."""
    before = len(fids)
    fids = [f for f in fids if f not in EXCLUDED_SITES]
    if before != len(fids):
        dropped = before - len(fids)
        print(f"Exclusion policy: dropped {dropped} site(s) {EXCLUDED_SITES}")
    return fids


VOLK_COLUMN_MAP = {
    "GEESEBAL_3x3": "geesebal",
    "PTJPL_3x3": "ptjpl",
    "SSEBOP_3x3": "ssebop",
    "SIMS_3x3": "sims",
    "EEMETRIC_3x3": "eemetric",
    "DISALEXI_3x3": "disalexi",
}

# May v2.1 is the only sanctioned OpenET capture source; the January
# "openet_flux" directory is rejected on source-version grounds.
OPENET_SOURCE_DIRNAME = "openet_flux_2pt1"
JANUARY_SOURCE_DIRNAME = "openet_flux"

_OPENET_ETO_CACHE = {}


def assert_may_source(path):
    """Hard-fail if a resolved OpenET data path is the January capture set."""
    parts = os.path.normpath(path).split(os.sep)
    if JANUARY_SOURCE_DIRNAME in parts:
        raise BenchmarkConstructionError(
            f"January OpenET source rejected: {path} — use {OPENET_SOURCE_DIRNAME}"
        )
    if OPENET_SOURCE_DIRNAME not in parts:
        raise BenchmarkConstructionError(f"OpenET data path is not the May v2.1 source: {path}")
    return path


def load_openet_eto(data_dir=None):
    """Load the extracted OpenET bias-corrected gridMET ETo (dates x sites).

    This is the sole ETo basis for benchmark reconstruction (identical to the
    run container's meteorology/gridmet/eto_corr; the archived
    site_daily_timeseries `eto` column is raw gridMET and must not be used).
    """
    candidates = []
    if data_dir:
        candidates.append(os.path.join(data_dir, "openet_refet", "openet_eto.csv"))
    candidates.append(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "openet_refet", "openet_eto.csv"
        )
    )
    for path in candidates:
        if os.path.exists(path):
            if path not in _OPENET_ETO_CACHE:
                wide = pd.read_csv(path, index_col="site_id")
                wide.columns = pd.to_datetime(wide.columns, format="%Y%m%d")
                _OPENET_ETO_CACHE[path] = (wide.T.sort_index(), path)
            return _OPENET_ETO_CACHE[path]
    raise FileNotFoundError(f"openet_eto.csv not found in: {candidates}")


def load_config(config_path=None):
    project_dir = Path(__file__).resolve().parent
    conf = Path(config_path) if config_path else project_dir / "5_Flux_Ensemble.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)
    return cfg


def parse_pest_params(par_csv, fids):
    """Parse PEST++ .par.csv into {fid: {param: value}} using median realization."""
    df = pd.read_csv(par_csv, index_col=0)

    numeric_rows = df.loc[df.index != "base"]
    row = numeric_rows.median()

    params_by_fid = {}
    for col in df.columns:
        parts = col.split("_ptype:")[0]
        parts = parts.replace("pname:p_", "")
        parts = parts.rsplit("_:0", 1)[0]

        matched_fid = None
        for fid in fids:
            if parts.lower().endswith(f"_{fid.lower()}"):
                matched_fid = fid
                param_name = parts[: -(len(fid) + 1)]
                break

        if matched_fid:
            if matched_fid not in params_by_fid:
                params_by_fid[matched_fid] = {}
            params_by_fid[matched_fid][param_name] = float(row[col])

    return params_by_fid


def run_calibrated_model(cfg, container, fids, calibrated_params):
    """Run model with calibrated parameters. Returns {fid: DataFrame}."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(calibrated_params, tmp)
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
            mask_mode=getattr(cfg, "mask_mode", "irrigation"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            transpiration_cover_mode=getattr(cfg, "transpiration_cover_mode", None),
            cover_linear_ndvi_bare=getattr(cfg, "cover_linear_ndvi_bare", None),
            cover_linear_ndvi_full=getattr(cfg, "cover_linear_ndvi_full", None),
            kcb_ndvi_mode=getattr(cfg, "kcb_ndvi_mode", None),
            stress_depletion_fraction=getattr(cfg, "stress_depletion_fraction", None),
        )

        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        etr = swim_input.get_time_series("etr")

        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {
                    "et_act": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "etref": etr[:, i],
                    "swe": output.swe[:, i],
                },
                index=dates,
            )

        swim_input.close()
    finally:
        for p in [temp_h5, params_json]:
            if os.path.exists(p):
                os.remove(p)

    return results


def load_flux_et(fid, flux_dir):
    """Load energy-balance-corrected ET from flux tower data."""
    path = os.path.join(flux_dir, f"{fid}_daily_data.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "ET_corr" in df.columns:
        return df["ET_corr"]
    return pd.Series(dtype=float)


def resolve_flux_dir(cfg):
    """Resolve the flux directory from config, with a shipped-data fallback."""
    if cfg.flux_dir and os.path.isdir(cfg.flux_dir):
        return cfg.flux_dir
    fallback = os.path.join(cfg.data_dir, "flux")
    if os.path.isdir(fallback):
        return fallback
    return cfg.flux_dir or os.path.join(cfg.data_dir, "daily_flux_files")


def load_openet_etf_nomask(container, fid):
    """Load per-model ETf from the container using no_mask (unmasked) data.

    Returns {model_name: pd.Series} of ETf values without irrigation masking.
    """
    etf_by_model = {}
    for model in OPEN_SOURCE_MODELS:
        etf_path = f"remote_sensing/etf/landsat/{model}/no_mask"
        try:
            etf_df = container.query.dataframe(etf_path, fields=[fid])
        except KeyError:
            continue
        if fid in etf_df.columns:
            series = etf_df[fid]
            if series.notna().any():
                etf_by_model[model] = series
    return etf_by_model


def load_volk_openet_et(fid, openet_daily_dir):
    """Load per-model daily ET from Volk OpenET 3x3 extractions.

    These CSVs contain actual ET (mm/day), not ETf fractions.
    Returns {model_name: pd.Series} of sparse ET on Landsat dates.
    """
    path = os.path.join(openet_daily_dir, f"{fid}.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, index_col="DATE", parse_dates=True)

    et_by_model = {}
    for raw_col, model_name in VOLK_COLUMN_MAP.items():
        if raw_col in df.columns:
            et_by_model[model_name] = df[raw_col].astype(float)

    if "ensemble_mean_3x3" in df.columns:
        et_by_model["ensemble"] = df["ensemble_mean_3x3"].astype(float)

    return et_by_model


def calc_metrics(obs, mod):
    """Legacy per-site metric row: R2/r/RMSE/bias/KGE via the shared helper.

    Delegates to ``swimrs.evaluation.benchmark.site_secondary_metrics`` and
    maps its unambiguous keys onto the legacy per-site column labels:
    ``r2`` <- ``nse`` (1 - SSE/SST) and ``bias`` <- signed ``mbe``.
    """
    m = site_secondary_metrics(obs, mod, min_n=MIN_OBS_FOR_METRICS)
    return {
        "n": m["n"],
        "r2": m["nse"],
        "r": m["r"],
        "rmse": m["rmse"],
        "bias": m["mbe"],
        "kge": m["kge"],
    }


@dataclass
class BenchmarkEvaluation:
    """Grouped-benchmark bundle: per-site diagnostics plus grouped estimands."""

    site_metrics: pd.DataFrame
    grouped_metrics: pd.DataFrame
    grouped_contrasts: pd.DataFrame
    paired_records: tuple
    site_effect_summary: pd.DataFrame | None
    metadata: dict


def _get_diy_openet(container, fid, irr_data, etref):
    """DIY: reconstruct daily ET from our sparse container ETf, ETf-first.

    Uses no_mask ETf exclusively — no fallback to masked ETf. Reconstruction
    goes through the shared helper (Volk ±32-day temporal-support rule, no
    unbounded tail padding).

    Returns {model_name: pd.Series} of daily ET (interpolated ETf × ETo).
    """
    etf_by_model = load_openet_etf_nomask(container, fid)
    et_daily = {}
    for model_name, etf_series in etf_by_model.items():
        recon = reconstruct_daily_benchmark(
            capture_series=etf_series,
            capture_space="etf",
            eto=etref,
            eto_name="model_etref",
            label=f"{fid}:{model_name}",
        )
        et_daily[model_name] = recon.daily_et
    return et_daily


def _get_volk_openet(fid, openet_daily_dir):
    """Load sparse capture-date ET from the May OpenET v2.1 3x3 CSVs.

    Daily benchmark construction occurs in ``evaluate``: capture ET is divided
    by same-day OpenET ETo, ETf is reconstructed with the shared OpenET-core
    temporal-support behavior, and daily ET is recovered with the same ETo.
    """
    return load_volk_openet_et(fid, openet_daily_dir)


def _retained_support_classes(fid, retained_dates, ensemble_recon):
    """Per-date OpenET support classes for the retained paired dates.

    Derives strictly from the retained ensemble BenchmarkReconstruction, whose
    classification comes from the raw May v2.1 ensemble_mean_3x3 capture
    availability BEFORE temporal reconstruction — calibration-target capture
    flags never participate. Any inconsistency hard-fails.
    """
    if ensemble_recon is None:
        raise GroupedEstimationError(
            f"{fid}: retained paired record without an ensemble BenchmarkReconstruction"
        )
    support_series = ensemble_recon.support_class.reindex(retained_dates)
    if support_series.isna().any():
        missing = retained_dates[support_series.isna().to_numpy()]
        raise GroupedEstimationError(
            f"{fid}: {len(missing)} retained dates missing from the ensemble "
            f"support classification (first: {missing[0].date()})"
        )
    support = tuple(str(s) for s in support_series)
    if "unsupported" in support:
        raise GroupedEstimationError(
            f"{fid}: {support.count('unsupported')} retained dates classified "
            "'unsupported' — an unsupported date cannot carry a finite "
            "reconstructed ensemble value"
        )
    capture_set = set(pd.DatetimeIndex(ensemble_recon.capture_dates))
    for date, cls in zip(retained_dates, support, strict=True):
        if (cls == "capture") != (date in capture_set):
            raise GroupedEstimationError(
                f"{fid}: support class {cls!r} on {date.date()} is inconsistent "
                "with the raw ensemble capture dates"
            )
    return support


def _hash_entry(path, reason_if_unresolved=None):
    """sha256 provenance entry for one consumed input file.

    A canonical input that cannot be found hard-fails; a legitimately
    non-file identifier records an explicit not-hashed reason — a requested
    hash is never silently omitted.
    """
    if path is None:
        if reason_if_unresolved is None:
            raise FileNotFoundError("input hash requested for an unresolved path")
        return {"path": None, "not_hashed_reason": reason_if_unresolved}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"cannot hash missing input file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _container_manifest_sha256(container_path):
    """Stable hash of the container's zarr metadata documents.

    Hashes relative path + content of every zarr metadata file (group/array
    structure and attributes) in sorted order; chunk data is deliberately not
    hashed (the container is multi-GB and append-only under the run policy).
    """
    root = Path(container_path)
    meta_names = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name in meta_names)
    if not files:
        raise FileNotFoundError(f"no zarr metadata documents under {container_path}")
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _volk_input_hashes(cfg, container, par_csv, flux_dir, openet_daily_dir, openet_eto_path, fids):
    """Hash manifest of the scientifically decisive inputs (canonical volk daily)."""
    container_path = str(getattr(container, "path", "")) or None
    if container_path is not None and os.path.isdir(container_path):
        container_entry = {
            "path": container_path,
            "sha256": _container_manifest_sha256(container_path),
            "scope": (
                "zarr metadata documents only (group/array structure and "
                "attributes); chunk data not hashed"
            ),
        }
    else:
        container_entry = {
            "path": container_path,
            "not_hashed_reason": "container path not exposed as a local zarr directory",
        }
    return {
        "par_csv": _hash_entry(par_csv),
        "config_toml": _hash_entry(
            getattr(cfg, "config_path", None),
            reason_if_unresolved="config path not exposed by ProjectConfig",
        ),
        "openet_eto_csv": _hash_entry(openet_eto_path),
        "container_manifest": container_entry,
        "flux_files": {
            fid: _hash_entry(os.path.join(flux_dir, f"{fid}_daily_data.csv")) for fid in fids
        },
        "openet_daily_files": {
            fid: _hash_entry(os.path.join(openet_daily_dir, f"{fid}.csv")) for fid in fids
        },
    }


def _collect_daily(
    cfg,
    container,
    par_csv,
    fids,
    flux_dir,
    openet_source="diy",
    quiet_sites=False,
    results_dir=None,
):
    """Run the forward model once and collect per-site rows plus paired arrays.

    Returns ``(site_metrics_df, records, collect_meta)`` where ``records`` is a
    tuple of PairedSiteSeries on the exact ensemble-headline mask (one per
    retained site) and ``collect_meta`` carries resolved input paths. No
    aggregate summaries are computed or printed here.

    openet_source: 'diy' uses container ETf extracts (diagnostic); 'volk'
    reconstructs daily ET from May OpenET v2.1 3x3 capture ET via same-day
    OpenET ETo and ETf-first interpolation.
    """
    fids = apply_exclusions(fids)
    print(f"Evaluating {len(fids)} fields from {par_csv} (openet_source={openet_source})")

    # Load irrigation data from container
    irr_data = {}
    try:
        props = container.query.properties()
        for fid in fids:
            if fid in props and "irr" in props[fid]:
                irr_data[fid] = props[fid]["irr"]
    except Exception:
        pass

    openet_daily_dir = None
    openet_eto_path = None
    if openet_source == "volk":
        openet_daily_dir = assert_may_source(
            os.path.join(cfg.data_dir, OPENET_SOURCE_DIRNAME, "daily_data")
        )
        openet_eto, openet_eto_path = load_openet_eto(cfg.data_dir)

    calibrated_params = parse_pest_params(par_csv, fids)
    missing = [f for f in fids if f not in calibrated_params]
    if missing:
        print(f"WARNING: No calibrated params for: {missing}")

    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    rows = []
    records = []
    excluded = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            print(f"  {fid}: no flux data, skipping")
            excluded.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            print(f"  {fid}: below VALIDATION_POLICY site minimum (90 valid days / 3 months)")
            excluded.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue

        model_df = model_results[fid]
        swim_et = model_df["et_act"]
        etref = model_df["etref"]

        # Common dates between model and flux
        common = swim_et.index.intersection(flux_et.index)
        if len(common) < MIN_OBS_FOR_METRICS:
            print(f"  {fid}: only {len(common)} overlapping days, skipping")
            excluded.append(
                {
                    "site": fid,
                    "reason": f"daily_overlap_{len(common)}_below_min_{MIN_OBS_FOR_METRICS}",
                }
            )
            continue

        obs = flux_et.loc[common].values
        swim_vals = swim_et.loc[common].values

        # Load all OpenET model daily ET on common dates
        ensemble_recon = None
        if openet_source == "diy":
            et_daily_by_model = _get_diy_openet(container, fid, irr_data, etref)
        else:
            # ETf-first reconstruction on the common OpenET bias-corrected
            # gridMET ETo basis: ETf_i = ET_i/ETo_i at captures, linear-in-time
            # ETf interpolation under the Volk ±32-day rule, × daily ETo.
            # Direct interpolation of sparse ET is invalid (it smooths the
            # daily demand signal) and must never be reintroduced here.
            if fid not in openet_eto.columns:
                raise BenchmarkConstructionError(f"{fid}: no extracted OpenET ETo")
            site_eto = openet_eto[fid].astype("float64")
            et_sparse_by_model = _get_volk_openet(fid, openet_daily_dir)
            et_daily_by_model = {}
            for mn, s in et_sparse_by_model.items():
                if not s.notna().any():
                    et_daily_by_model[mn] = s
                    continue
                recon = reconstruct_daily_benchmark(
                    capture_series=s,
                    capture_space="et",
                    eto=site_eto,
                    eto_name=openet_eto_path,
                    label=f"{fid}:{mn}",
                )
                et_daily_by_model[mn] = recon.daily_et
                if mn == "ensemble":
                    # retained explicitly: its support_class series (derived
                    # from the raw ensemble_mean_3x3 capture availability,
                    # never calibration flags) annotates the paired record
                    ensemble_recon = recon

        # Ensemble ET on common dates
        ens_vals = np.full(len(common), np.nan)
        if "ensemble" in et_daily_by_model:
            ens_vals = et_daily_by_model["ensemble"].reindex(common).values
        else:
            ensemble_source = getattr(cfg, "ensemble_source", "computed")
            if ensemble_source == "openet":
                ens_path = "remote_sensing/etf/landsat/ensemble/no_mask"
                try:
                    ens_df = container.query.dataframe(ens_path, fields=[fid])
                    if fid in ens_df.columns and ens_df[fid].notna().any():
                        recon = reconstruct_daily_benchmark(
                            capture_series=ens_df[fid],
                            capture_space="etf",
                            eto=etref,
                            eto_name="model_etref",
                            label=f"{fid}:ensemble",
                        )
                        ens_vals = recon.daily_et.reindex(common).values
                except KeyError:
                    pass
            else:
                # Computed ensemble: nanmean across available DIY models
                model_arrays = []
                for mn in OPEN_SOURCE_MODELS:
                    if mn in et_daily_by_model:
                        arr = et_daily_by_model[mn].reindex(common).values
                        model_arrays.append(arr)
                if model_arrays:
                    stack = np.column_stack(model_arrays)
                    ens_vals = np.nanmean(stack, axis=1)

        # Paired mask: flux, swim, and ensemble all finite on the same day.
        # This single ensemble-headline mask scores both models and populates
        # the retained PairedSiteSeries — never re-filtered per model.
        paired_mask = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(ens_vals)
        n_paired = int(paired_mask.sum())

        row = {"fid": fid, "n": n_paired}

        if n_paired >= MIN_OBS_FOR_METRICS:
            m = calc_metrics(obs[paired_mask], swim_vals[paired_mask])
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_swim"] = m[k]

            m = calc_metrics(obs[paired_mask], ens_vals[paired_mask])
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_ensemble"] = m[k]

            retained_dates = pd.DatetimeIndex(common)[paired_mask]
            support = None
            if openet_source == "volk":
                support = _retained_support_classes(fid, retained_dates, ensemble_recon)
            record = PairedSiteSeries(
                fid=fid,
                index=retained_dates,
                observed=obs[paired_mask],
                swim=swim_vals[paired_mask],
                openet=ens_vals[paired_mask],
                support_class=support,
            )
            if record.n != row["n"]:
                raise GroupedEstimationError(
                    f"{fid}: paired record n={record.n} != site row n={row['n']}"
                )
            records.append(record)
        else:
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_swim"] = np.nan
                row[f"{k}_ensemble"] = np.nan

        # Per-model OpenET metrics: each model paired with flux+swim+model on same days.
        # SWIM is re-scored per model so swim-vs-ptjpl uses the same days as ptjpl-vs-flux.
        for model_name in OPEN_SOURCE_MODELS:
            if model_name not in et_daily_by_model:
                for k in ["r2", "r", "rmse", "bias", "kge"]:
                    row[f"{k}_{model_name}"] = np.nan
                    row[f"r2_swim_vs_{model_name}"] = np.nan
                continue

            model_vals = et_daily_by_model[model_name].reindex(common).values
            model_paired = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(model_vals)
            if model_paired.sum() >= MIN_OBS_FOR_METRICS:
                m = calc_metrics(obs[model_paired], model_vals[model_paired])
                row[f"r2_swim_vs_{model_name}"] = r2_score(
                    obs[model_paired], swim_vals[model_paired]
                )
            else:
                m = {"r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan, "kge": np.nan}
                row[f"r2_swim_vs_{model_name}"] = np.nan

            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_{model_name}"] = m[k]

        rows.append(row)

        if not quiet_sites:
            status = "retained" if n_paired >= MIN_OBS_FOR_METRICS else "below-min"
            print(f"  {fid}: n_paired={n_paired:>5d}  {status}")

    write_excluded_sites(excluded, results_dir or os.path.join(cfg.project_ws, "results"))

    collect_meta = {
        "paths": {
            "config": getattr(cfg, "config_path", None),
            "container": str(getattr(container, "path", "")) or None,
            "par_csv": par_csv,
            "flux_dir": flux_dir,
            "openet_daily_dir": openet_daily_dir,
            "openet_eto_csv": openet_eto_path,
        },
        "openet_source": openet_source,
        "excluded_sites": excluded,
        "static_exclusions": sorted(EXCLUDED_SITES),
    }

    if openet_source == "volk":
        collect_meta["input_hashes"] = _volk_input_hashes(
            cfg,
            container,
            par_csv,
            flux_dir,
            openet_daily_dir,
            openet_eto_path,
            [r.fid for r in records],
        )

    if not rows:
        print("No fields with sufficient data for evaluation.")
        return pd.DataFrame(), (), collect_meta

    metrics_df = pd.DataFrame(rows).set_index("fid")
    return metrics_df, tuple(records), collect_meta


def evaluate(cfg, container, par_csv, fids, flux_dir, openet_source="diy"):
    """Compatibility wrapper: per-site daily metrics DataFrame only.

    Retained for existing callers (archive gates, ablations). The preferred
    grouped-benchmark reporting path is ``evaluate_benchmark_daily``, which
    additionally returns pooled and sqrt(n)-weighted estimates.
    """
    site_metrics, _records, _meta = _collect_daily(
        cfg, container, par_csv, fids, flux_dir, openet_source
    )
    return site_metrics


def evaluate_benchmark_daily(
    cfg,
    container,
    par_csv,
    fids,
    flux_dir,
    openet_source="volk",
    bootstrap_reps=BOOTSTRAP_REPS_DEFAULT,
    bootstrap_seed=BOOTSTRAP_SEED_DEFAULT,
    with_site_effect=False,
    quiet_sites=False,
    results_dir=None,
):
    """Canonical daily grouped benchmark: one forward run, grouped estimands.

    Returns a BenchmarkEvaluation bundle with per-site metrics (secondary
    diagnostic), pooled and sqrt(n)-weighted grouped metrics, SWIM-minus-OpenET
    contrasts with whole-site bootstrap intervals, retained paired records,
    the optional median site-effect summary, and provenance metadata.
    """
    site_metrics, records, collect_meta = _collect_daily(
        cfg,
        container,
        par_csv,
        fids,
        flux_dir,
        openet_source,
        quiet_sites=quiet_sites,
        results_dir=results_dir,
    )
    grouped, contrasts = grouped_metric_tables(
        records, "daily", bootstrap_reps, bootstrap_seed, openet_source
    )
    effect = (
        site_effect_summary(records, bootstrap_reps, bootstrap_seed, "daily")
        if with_site_effect
        else None
    )
    metadata = grouped_metadata(
        records, "daily", bootstrap_reps, bootstrap_seed, openet_source, collect_meta
    )
    return BenchmarkEvaluation(
        site_metrics=site_metrics,
        grouped_metrics=grouped,
        grouped_contrasts=contrasts,
        paired_records=records,
        site_effect_summary=effect,
        metadata=metadata,
    )


def evaluate_etf(cfg, container, par_csv, fids):
    """Compare SWIM ETf against OpenET ETf from the container at capture dates.

    Runs the calibrated model and compares predicted ETf directly against
    per-model ETf observations stored in the container (at Landsat overpass
    dates only).  This isolates model skill from ETo conversion issues.

    Returns DataFrame with per-field, per-model ETf metrics.
    """
    fids = apply_exclusions(fids)
    print(f"ETf evaluation: {len(fids)} fields from {par_csv}")

    calibrated_params = parse_pest_params(par_csv, fids)
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    rows = []
    for fid in fids:
        if fid not in model_results:
            continue
        swim_etf = model_results[fid]["etf_model"]

        nomask_etf = load_openet_etf_nomask(container, fid)

        for model in OPEN_SOURCE_MODELS:
            if model not in nomask_etf:
                continue
            combined = nomask_etf[model]

            obs_etf = combined.dropna()
            obs_etf = obs_etf[obs_etf > 0]
            if len(obs_etf) < 10:
                continue

            common = swim_etf.index.intersection(obs_etf.index)
            if len(common) < 10:
                continue

            s = swim_etf.loc[common].values
            o = obs_etf.loc[common].values
            valid = np.isfinite(s) & np.isfinite(o)
            s, o = s[valid], o[valid]
            if len(s) < 10:
                continue

            m = calc_metrics(o, s)
            rows.append({"fid": fid, "model": model, **m})

    if not rows:
        print("No fields with sufficient ETf data.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Per-field summary (median across models)
    by_fid = df.groupby("fid").agg(
        n=("n", "sum"),
        r2_median=("r2", "median"),
        rmse_median=("rmse", "median"),
        bias_median=("bias", "median"),
    )

    # Per-model summary
    print("\n" + "=" * 70)
    print("ETf: SWIM vs OpenET (at Landsat capture dates)")
    print("=" * 70)
    header = f"{'model':<12}  {'combos':>6}  {'r2_mean':>8}  {'r2_med':>8}  {'rmse_mean':>10}  {'bias_mean':>10}"
    print(header)
    print("-" * len(header))
    for model in OPEN_SOURCE_MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        print(
            f"{model:<12}  {len(sub):>6}  {sub['r2'].mean():>8.3f}  "
            f"{sub['r2'].median():>8.3f}  {sub['rmse'].mean():>10.3f}  "
            f"{sub['bias'].mean():>10.3f}"
        )
    print(
        f"{'ALL':<12}  {len(df):>6}  {df['r2'].mean():>8.3f}  "
        f"{df['r2'].median():>8.3f}  {df['rmse'].mean():>10.3f}  "
        f"{df['bias'].mean():>10.3f}"
    )

    # Worst / best fields
    ranked = by_fid.sort_values("r2_median")
    print("\nWorst 10 fields (median R2 across models):")
    for fid, row in ranked.head(10).iterrows():
        print(f"  {fid:<20} R2={row['r2_median']:.3f}  RMSE={row['rmse_median']:.3f}")
    print("\nBest 10 fields:")
    for fid, row in ranked.tail(10).iterrows():
        print(f"  {fid:<20} R2={row['r2_median']:.3f}  RMSE={row['rmse_median']:.3f}")

    # Set fid+model as index so saved CSVs are self-identifying
    df = df.set_index(["fid", "model"])
    return df


def load_volk_monthly_et(fid, monthly_dir):
    """Load Volk OpenET monthly ET totals (mm/month).

    Returns {model_name: pd.Series} indexed by month start date.
    """
    path = os.path.join(monthly_dir, f"{fid}.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, index_col="DATE", parse_dates=True)

    et_by_model = {}
    for raw_col, model_name in VOLK_COLUMN_MAP.items():
        if raw_col in df.columns:
            et_by_model[model_name] = df[raw_col].astype(float)

    if "ensemble_mean_3x3" in df.columns:
        et_by_model["ensemble"] = df["ensemble_mean_3x3"].astype(float)

    return et_by_model


def _collect_monthly(cfg, container, par_csv, fids, flux_dir, quiet_sites=False, results_dir=None):
    """Monthly ET comparison with strictly paired months.

    The Volk references are full calendar-month totals, so SWIM is summed over
    full months too, and only months with >= 28 valid daily flux observations
    are kept so the flux total misses at most a few days. SWIM and each OpenET
    model are scored on the exact same months per site. The ensemble defines
    the paired month index — all models share it. Admission requires
    MIN_OBS_FOR_METRICS paired months (the metric floor), so no all-NaN metric
    rows are emitted; shorter sites go to the exclusion ledger instead.

    Returns ``(site_metrics_df, records, collect_meta)``.
    """
    fids = apply_exclusions(fids)
    monthly_dir = assert_may_source(
        os.path.join(cfg.data_dir, OPENET_SOURCE_DIRNAME, "monthly_data")
    )
    print(f"Monthly evaluation: {len(fids)} fields from {par_csv}")

    calibrated_params = parse_pest_params(par_csv, fids)
    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    all_models = OPEN_SOURCE_MODELS + ["ensemble"]
    rows = []
    records = []
    excluded = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            excluded.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            print(f"  {fid}: below VALIDATION_POLICY site minimum (90 valid days / 3 months)")
            excluded.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue

        model_df = model_results[fid]
        swim_et = model_df["et_act"]

        # Intersect daily indices first, then aggregate to monthly
        daily_common = swim_et.index.intersection(flux_et.index)
        if len(daily_common) < 30:
            print(f"  {fid}: only {len(daily_common)} daily overlap, skipping")
            excluded.append({"site": fid, "reason": f"daily_overlap_{len(daily_common)}_below_30"})
            continue

        flux_daily = flux_et.loc[daily_common]

        # Full-calendar-month totals gated on nearly-complete flux months,
        # matching the full-month Volk reference totals
        swim_monthly, flux_monthly = full_month_paired_sums(swim_et, flux_daily)

        # Load Volk monthly ensemble to define the paired month index
        volk_monthly = load_volk_monthly_et(fid, monthly_dir)
        ens_monthly = volk_monthly.get("ensemble")

        if ens_monthly is not None:
            # Paired months: flux, swim, AND ensemble all finite
            all_idx = flux_monthly.index
            ens_on_idx = ens_monthly.reindex(all_idx)
            paired_mask = (
                flux_monthly.notna() & swim_monthly.reindex(all_idx).notna() & ens_on_idx.notna()
            )
            paired_months = all_idx[paired_mask]
        else:
            # No ensemble data — use flux ∩ swim months
            paired_months = swim_monthly.index.intersection(flux_monthly.index)

        n_paired = len(paired_months)
        if n_paired < MIN_OBS_FOR_METRICS:
            # metric-floor admission: below MIN_OBS_FOR_METRICS the metrics
            # would be all-NaN anyway; ledger the site instead of emitting
            # an empty metric row
            print(f"  {fid}: only {n_paired} paired months, skipping")
            excluded.append(
                {
                    "site": fid,
                    "reason": f"paired_months_{n_paired}_below_min_{MIN_OBS_FOR_METRICS}",
                }
            )
            continue

        obs = flux_monthly.loc[paired_months].values

        row = {"fid": fid, "n": n_paired}
        m = calc_metrics(obs, swim_monthly.reindex(paired_months).values)
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_swim"] = m[k]

        # Score each model — per-model paired months (flux + swim + model all finite)
        swim_on_paired = swim_monthly.reindex(paired_months).values
        for model_name in all_models:
            if model_name not in volk_monthly:
                for k in ["r2", "r", "rmse", "bias", "kge"]:
                    row[f"{k}_{model_name}"] = np.nan
                continue

            model_vals = volk_monthly[model_name].reindex(paired_months).values
            model_valid = np.isfinite(model_vals) & np.isfinite(obs) & np.isfinite(swim_on_paired)
            if model_valid.sum() >= MIN_OBS_FOR_METRICS:
                m = calc_metrics(obs[model_valid], model_vals[model_valid])
            else:
                m = {"r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan, "kge": np.nan}

            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_{model_name}"] = m[k]

        rows.append(row)

        if ens_monthly is not None:
            record = PairedSiteSeries(
                fid=fid,
                index=pd.DatetimeIndex(paired_months),
                observed=obs,
                swim=swim_monthly.reindex(paired_months).values,
                openet=ens_monthly.reindex(paired_months).values,
            )
            if record.n != row["n"]:
                raise GroupedEstimationError(
                    f"{fid}: paired record n={record.n} != site row n={row['n']}"
                )
            records.append(record)

        if not quiet_sites:
            print(f"  {fid}: n_paired={n_paired:>3d} mo  retained")

    write_excluded_sites(excluded, results_dir or os.path.join(cfg.project_ws, "results"))

    collect_meta = {
        "paths": {
            "config": getattr(cfg, "config_path", None),
            "container": str(getattr(container, "path", "")) or None,
            "par_csv": par_csv,
            "flux_dir": flux_dir,
            "openet_monthly_dir": monthly_dir,
        },
        "openet_source": "volk",
        "monthly_gates": {
            "full_month_flux_min_days": 28,
            "min_paired_months": MIN_OBS_FOR_METRICS,
        },
        "excluded_sites": excluded,
        "static_exclusions": sorted(EXCLUDED_SITES),
    }

    if not rows:
        print("No fields with sufficient data.")
        return pd.DataFrame(), (), collect_meta

    metrics_df = pd.DataFrame(rows).set_index("fid")
    return metrics_df, tuple(records), collect_meta


def evaluate_monthly(cfg, container, par_csv, fids, flux_dir):
    """Compatibility wrapper: per-site monthly metrics DataFrame only.

    The preferred grouped-benchmark reporting path is
    ``evaluate_benchmark_monthly``.
    """
    site_metrics, _records, _meta = _collect_monthly(cfg, container, par_csv, fids, flux_dir)
    return site_metrics


def evaluate_benchmark_monthly(
    cfg,
    container,
    par_csv,
    fids,
    flux_dir,
    bootstrap_reps=BOOTSTRAP_REPS_DEFAULT,
    bootstrap_seed=BOOTSTRAP_SEED_DEFAULT,
    with_site_effect=False,
    quiet_sites=False,
    results_dir=None,
):
    """Canonical monthly grouped benchmark (independent full-month totals)."""
    site_metrics, records, collect_meta = _collect_monthly(
        cfg,
        container,
        par_csv,
        fids,
        flux_dir,
        quiet_sites=quiet_sites,
        results_dir=results_dir,
    )
    grouped, contrasts = grouped_metric_tables(
        records, "monthly", bootstrap_reps, bootstrap_seed, "volk"
    )
    effect = (
        site_effect_summary(records, bootstrap_reps, bootstrap_seed, "monthly")
        if with_site_effect
        else None
    )
    metadata = grouped_metadata(
        records, "monthly", bootstrap_reps, bootstrap_seed, "volk", collect_meta
    )
    return BenchmarkEvaluation(
        site_metrics=site_metrics,
        grouped_metrics=grouped,
        grouped_contrasts=contrasts,
        paired_records=records,
        site_effect_summary=effect,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Grouped-output rendering and writing (no metric math here)
# ---------------------------------------------------------------------------


def print_grouped_summary(bundle, scale):
    """Console tables for the grouped benchmark (the default headline)."""
    meta = bundle.metadata
    grouped = bundle.grouped_metrics
    contrasts = bundle.grouped_contrasts
    reps = int(meta["bootstrap"]["reps"])
    seed = int(meta["bootstrap"]["seed"])
    pair_word = "paired site-days" if scale == "daily" else "paired site-months"
    unit_error = ERROR_METRIC_UNITS[scale]

    print("\n" + "=" * 84)
    print(f"GROUPED SWIM-OPENET BENCHMARK — {scale.upper()}")
    boot_note = (
        f"site bootstrap: {reps:,}, seed {seed}"
        if reps > 0
        else ("bootstrap disabled (development run — CI fields null)")
    )
    print(f"{meta['n_sites']} sites; {meta['n_pairs']:,} {pair_word}; {boot_note}")
    print(f"Benchmark: {meta['benchmark']}; RMSE/MBE in {unit_error}")
    print("=" * 84)

    g = grouped.set_index(["aggregation", "model", "metric"])["estimate"]

    print("\nPOOLED OBSERVATIONS")
    header = f"{'Model':<18}" + "".join(f"{h:>9}" for h in ("KGE", "RMSE", "MBE", "r", "r^2"))
    header += f"{'slope (0 intercept)':>21}"
    print(header)
    for model in GROUPED_MODEL_ORDER:
        line = f"{MODEL_DISPLAY[model]:<18}"
        for k in POOLED_METRICS[:5]:
            line += f"{g[(AGG_POOLED, model, k)]:>9.3f}"
        line += f"{g[(AGG_POOLED, model, 'slope0')]:>21.3f}"
        print(line)

    print("\nSQRT(N)-WEIGHTED SITE METRICS")
    print(f"{'Model':<18}" + "".join(f"{h:>9}" for h in ("KGE", "RMSE", "MBE")))
    for model in GROUPED_MODEL_ORDER:
        line = f"{MODEL_DISPLAY[model]:<18}"
        for k in WEIGHTED_METRICS:
            line += f"{g[(AGG_WEIGHTED, model, k)]:>9.3f}"
        print(line)

    print("\nSWIM - OPENET CONTRASTS (whole-site bootstrap 95% CI)")
    print(
        f"{'aggregation':<28}{'metric':<8}{'delta':>10}{'ci95_low':>11}{'ci95_high':>11}  favorable"
    )
    for _, row in contrasts.iterrows():
        lo = f"{row['ci95_low']:>11.4f}" if np.isfinite(row["ci95_low"]) else f"{'null':>11}"
        hi = f"{row['ci95_high']:>11.4f}" if np.isfinite(row["ci95_high"]) else f"{'null':>11}"
        print(
            f"{row['aggregation']:<28}{row['metric']:<8}{row['estimate']:>10.4f}{lo}{hi}"
            f"  {row['favorable_direction']}"
        )

    if bundle.site_effect_summary is not None:
        print("\nSECONDARY: MEDIAN PAIRED SITE EFFECT (site-level diagnostic, not the headline)")
        print(f"{'metric':<8}{'median delta':>14}{'ci95_low':>11}{'ci95_high':>11}")
        for _, row in bundle.site_effect_summary.iterrows():
            lo = f"{row['ci95_low']:>11.4f}" if np.isfinite(row["ci95_low"]) else f"{'null':>11}"
            hi = f"{row['ci95_high']:>11.4f}" if np.isfinite(row["ci95_high"]) else f"{'null':>11}"
            print(f"{row['metric']:<8}{row['estimate']:>14.4f}{lo}{hi}")


def grouped_output_paths(output_dir, scale, openet_source="volk"):
    """Grouped artifact paths; diagnostic DIY runs never claim canonical names."""
    suffix = "" if openet_source == "volk" else f"_{openet_source}"
    return {
        "metrics": os.path.join(output_dir, f"evaluation_grouped_{scale}_metrics{suffix}.csv"),
        "contrasts": os.path.join(output_dir, f"evaluation_grouped_{scale}_contrasts{suffix}.csv"),
        "metadata": os.path.join(output_dir, f"evaluation_grouped_{scale}_metadata{suffix}.json"),
        "site_effect": os.path.join(
            output_dir, f"evaluation_site_effect_summary_{scale}{suffix}.csv"
        ),
    }


def paired_record_output_path(output_dir):
    """Canonical daily paired-record artifact path (volk daily runs only)."""
    return os.path.join(output_dir, PAIRED_RECORD_FILENAME)


RECORD_ROUNDTRIP_ATOL = 1e-12


def _paired_record_roundtrip_gate(records, rebuilt, grouped_metrics):
    """Write/read identity gate for the paired-record artifact.

    The reloaded records must be exactly identical (site IDs, counts, dates,
    values, support classes) and must reproduce every grouped point estimate
    within RECORD_ROUNDTRIP_ATOL. Bootstrap replicates are deliberately not
    recomputed here (determinism has separate tests).
    """
    orig = {r.fid: r for r in records}
    if [r.fid for r in rebuilt] != sorted(orig):
        raise GroupedEstimationError(
            f"record round-trip: site IDs differ ({sorted(orig)} vs {[r.fid for r in rebuilt]})"
        )
    for r in rebuilt:
        o = orig[r.fid]
        if r.n != o.n:
            raise GroupedEstimationError(f"record round-trip: {r.fid} count {r.n} != {o.n}")
        if not r.index.equals(o.index):
            raise GroupedEstimationError(f"record round-trip: {r.fid} dates differ")
        if r.support_class != o.support_class:
            raise GroupedEstimationError(f"record round-trip: {r.fid} support classes differ")
        for name in ("observed", "swim", "openet"):
            if not np.array_equal(getattr(r, name), getattr(o, name)):
                raise GroupedEstimationError(
                    f"record round-trip: {r.fid} {name} values not bit-identical"
                )
    est = grouped_point_estimates(rebuilt)
    ref = grouped_metrics.set_index(["aggregation", "model", "metric"])["estimate"]
    for key, val in est.items():
        diff = abs(val - float(ref[key]))
        if diff > RECORD_ROUNDTRIP_ATOL:
            raise GroupedEstimationError(
                f"record round-trip: grouped point {key} differs by {diff!r} "
                f"(> {RECORD_ROUNDTRIP_ATOL})"
            )


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=here,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def _write_atomic(write_fn, path):
    tmp = f"{path}.tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def write_grouped_outputs(bundle, output_dir, scale, openet_source="volk", cli_args=None):
    """Write grouped metrics/contrasts CSVs and the provenance sidecar.

    CSVs keep full float precision (rounding is presentation-only). The
    site-effect summary is written only when present on the bundle. Returns
    the written paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = grouped_output_paths(output_dir, scale, openet_source)

    _write_atomic(lambda p: bundle.grouped_metrics.to_csv(p, index=False), paths["metrics"])
    _write_atomic(lambda p: bundle.grouped_contrasts.to_csv(p, index=False), paths["contrasts"])
    written = {"metrics": paths["metrics"], "contrasts": paths["contrasts"]}

    if bundle.site_effect_summary is not None:
        _write_atomic(
            lambda p: bundle.site_effect_summary.to_csv(p, index=False), paths["site_effect"]
        )
        written["site_effect"] = paths["site_effect"]

    metadata = dict(bundle.metadata)

    if scale == "daily" and openet_source == "volk":
        # Mandatory canonical parent artifact: the paired daily record
        # (e1_openet_paired_daily/v1). No opt-out — a citable evaluator bundle
        # must always carry its record. Serialized, reloaded, and gated on
        # grouped point-estimate identity before the metadata is finalized.
        record_path = paired_record_output_path(output_dir)
        frame = paired_records_to_frame(bundle.paired_records)
        counts = validate_paired_record_frame(frame)
        write_paired_record_frame(frame, record_path)
        rebuilt = paired_records_from_frame(read_paired_record_frame(record_path))
        _paired_record_roundtrip_gate(bundle.paired_records, rebuilt, bundle.grouped_metrics)
        written["paired_records"] = record_path
        metadata["paired_record_contract"] = {
            "schema_version": PAIRED_RECORD_SCHEMA_VERSION,
            "filename": os.path.basename(record_path),
            "sha256": _sha256(record_path),
            "byte_size": os.path.getsize(record_path),
            **counts,
            "ordered_columns": list(PAIRED_RECORD_COLUMNS),
            "sort_order": PAIRED_RECORD_SORT_ORDER,
            "units": {
                "flux_et_mm_d": "mm d-1",
                "swim_et_mm_d": "mm d-1",
                "openet_et_mm_d": "mm d-1",
            },
            "allowed_support_classes": list(SUPPORT_CLASSES),
            "allowed_temporal_classes": list(TEMPORAL_CLASSES),
            "temporal_class_definition": TEMPORAL_CLASS_DEFINITION,
            "mask_definition": GROUPED_MASK_DEFINITION,
            "minimum_all_days_count": MIN_OBS_FOR_METRICS,
        }

    metadata["git"] = _git_info()
    metadata["cli_args"] = cli_args
    metadata["output_hashes"] = {os.path.basename(p): _sha256(p) for p in written.values()}

    def _dump(p):
        with open(p, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    _write_atomic(_dump, paths["metadata"])
    written["metadata"] = paths["metadata"]
    return written


def find_par_csv(results_dir, project_name):
    """Find the latest .par.csv in results directory."""
    for i in range(10, -1, -1):
        candidate = os.path.join(results_dir, f"{project_name}.{i}.par.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def find_reference_par_csv(results_dir, project_name):
    """Resolve the canonical Example 5 parameter file when none is provided.

    Prefers the canonical run22 results (examples/VALIDATION_POLICY.md); any
    automatically discovered fallback is diagnostic-only.
    """
    candidate_dirs = []

    run22_dir = os.path.join(results_dir, "run22")
    if os.path.isdir(run22_dir):
        candidate_dirs.append(run22_dir)

    candidate_dirs.append(results_dir)

    if os.path.isdir(results_dir):
        for name in sorted(os.listdir(results_dir)):
            path = os.path.join(results_dir, name)
            if os.path.isdir(path) and path not in candidate_dirs:
                candidate_dirs.append(path)

    for directory in candidate_dirs:
        par_csv = find_par_csv(directory, project_name)
        if par_csv is not None:
            return par_csv

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated SWIM against flux tower ET and OpenET models"
    )
    parser.add_argument(
        "--par-csv", type=str, default=None, help="Override automatic par.csv discovery"
    )
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated site IDs (default: all)"
    )
    parser.add_argument(
        "--openet-source",
        type=str,
        choices=["diy", "volk"],
        default="volk",
        help=(
            "'diy' = our ETf extracts reconstructed to daily ET; "
            "'volk' = ETf-first reconstruction from May OpenET v2.1 3x3 capture ET"
        ),
    )
    parser.add_argument(
        "--etf",
        action="store_true",
        help="Compare SWIM ETf vs OpenET ETf at capture dates (instead of ET vs flux)",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Monthly ET totals: SWIM vs flux vs Volk 3x3 monthly CSVs",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help="Override container path (default: derived from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config TOML (default: 5_Flux_Ensemble.toml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output/staging directory (default: the configured results directory)",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=BOOTSTRAP_REPS_DEFAULT,
        help="Whole-site bootstrap resamples; 0 = development run with null CI fields",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED_DEFAULT,
        help="Bootstrap RNG seed",
    )
    parser.add_argument(
        "--site-effect-summary",
        action="store_true",
        help="Also emit the secondary median paired site-effect summary",
    )
    parser.add_argument(
        "--quiet-sites",
        action="store_true",
        help="Suppress per-site progress lines",
    )
    args = parser.parse_args()
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be a non-negative integer")

    cfg = load_config(args.config)
    flux_dir = resolve_flux_dir(cfg)
    results_dir = os.path.join(cfg.project_ws, "results")

    if args.par_csv:
        par_csv = args.par_csv
    else:
        par_csv = find_reference_par_csv(results_dir, cfg.project_name)
        print(
            "WARNING: --par-csv not given; automatic parameter discovery is "
            "diagnostic-only (see examples/VALIDATION_POLICY.md). Pass the "
            "explicit canonical run22 paths for citable results."
        )
    if par_csv is None:
        raise FileNotFoundError(f"No .par.csv found in {results_dir}")
    print(f"Using parameters: {par_csv}")

    if args.container:
        container_path = args.container
    else:
        container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}_run22.swim")
    container = SwimContainer.open(container_path, mode="r")

    if args.sites:
        fids = [s.strip() for s in args.sites.split(",")]
    else:
        fids = container.field_uids
    fids = apply_exclusions(fids)

    output_dir = args.output_dir or results_dir
    cli_args = vars(args)

    if args.bootstrap_reps == 0:
        print("DEVELOPMENT RUN: bootstrap disabled (--bootstrap-reps 0); CI fields will be null")

    try:
        os.makedirs(output_dir, exist_ok=True)
        if args.etf:
            metrics = evaluate_etf(cfg, container, par_csv, fids)
            out_csv = os.path.join(output_dir, "evaluation_etf_metrics.csv")
            metrics.to_csv(out_csv)
            print(f"\nMetrics saved to {out_csv}")
        else:
            if args.monthly:
                bundle = evaluate_benchmark_monthly(
                    cfg,
                    container,
                    par_csv,
                    fids,
                    flux_dir,
                    bootstrap_reps=args.bootstrap_reps,
                    bootstrap_seed=args.bootstrap_seed,
                    with_site_effect=args.site_effect_summary,
                    quiet_sites=args.quiet_sites,
                    results_dir=output_dir,
                )
                scale = "monthly"
                source = "volk"
                per_site_name = "evaluation_monthly_metrics.csv"
            else:
                bundle = evaluate_benchmark_daily(
                    cfg,
                    container,
                    par_csv,
                    fids,
                    flux_dir,
                    openet_source=args.openet_source,
                    bootstrap_reps=args.bootstrap_reps,
                    bootstrap_seed=args.bootstrap_seed,
                    with_site_effect=args.site_effect_summary,
                    quiet_sites=args.quiet_sites,
                    results_dir=output_dir,
                )
                scale = "daily"
                source = args.openet_source
                # diagnostic DIY runs never claim the canonical filename
                per_site_name = (
                    "evaluation_metrics.csv" if source == "volk" else "evaluation_metrics_diy.csv"
                )

            print_grouped_summary(bundle, scale)
            written = write_grouped_outputs(
                bundle, output_dir, scale, openet_source=source, cli_args=cli_args
            )
            per_site_csv = os.path.join(output_dir, per_site_name)
            bundle.site_metrics.to_csv(per_site_csv)
            print(f"\nGrouped estimates: {written['metrics']}")
            print(f"Grouped contrasts: {written['contrasts']}")
            if "paired_records" in written:
                n_rows = sum(r.n for r in bundle.paired_records)
                n_ret = sum(r.support_class.count("capture") for r in bundle.paired_records)
                print(
                    f"Paired daily records ({PAIRED_RECORD_SCHEMA_VERSION}): "
                    f"{written['paired_records']}"
                )
                print(
                    f"  {len(bundle.paired_records)} sites, {n_rows:,} rows; "
                    f"retrieval {n_ret:,}, between_retrieval {n_rows - n_ret:,}; "
                    "record sha256 in the provenance sidecar"
                )
            print(f"Provenance sidecar: {written['metadata']}")
            if "site_effect" in written:
                print(f"Site-effect summary (secondary): {written['site_effect']}")
            print(f"Per-site metrics (secondary diagnostic): {per_site_csv}")
    finally:
        container.close()

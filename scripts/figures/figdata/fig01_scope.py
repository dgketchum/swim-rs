"""Figure 1: scope map, evidence matrix, example record, and architecture."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from . import paths
from .cohorts import EXPECTED, EXPERIMENT_MAP, FIG01_CLASS_COUNTS
from .fig01_architecture import (
    EVALUATION_NODES,
    FIG01_COMMON_DATE_MAPPING_COLUMNS,
    FIG01_CONSTRAINT_INTO,
    FIG01_CYCLE_EDGES,
    FIG01_CYCLE_EXIT_EDGE,
    FIG01_CYCLE_EXIT_LABEL,
    FIG01_CYCLE_STAGES,
    FIG01_DISPLAY_DOMAINS,
    FIG01_DRIVER_INTO,
    FIG01_E1_APPLIED_WATER_NODES,
    FIG01_E3_METER_NODE,
    FIG01_EXAMPLE_CAPTURES,
    FIG01_EXAMPLE_DAYS,
    FIG01_EXAMPLE_END,
    FIG01_EXAMPLE_SITE,
    FIG01_EXAMPLE_START,
    FIG01_FORBIDDEN_EXAMPLE_COLUMNS,
    FIG01_IDENTIFICATION_COLOR,
    FIG01_LEGACY_OPENET_DAILY,
    FIG01_MAX_PLOTTING_REGIONS,
    FIG01_PARAM_TOKENS,
    FIG01_RECORD_ID,
    FIG01_TRANSFER_BRANCH_AT,
    FIG01_TRANSFER_DESTINATIONS,
    FIG01_TRANSFER_SOURCE,
    FITTING_NODES,
    VISIBLE_STRING_CLASSES,
    _assert_no_caption_facts_visible,
    architecture_spec,
)
from .fig02_formulation import E0_SITES_FILE
from .fig04_reliability import E1_MEMBERS
from .manifest import (
    MANIFEST,
    BuildError,
    require_columns,
    require_count,
    require_unique,
    sha256,
    write_table,
)
from .paths import (
    E1_ARCHIVE,
    E1_CONTAINER,
    E2_CONTAINER,
    E2_FROZEN_POOL,
    E3_CONTAINER,
    E3_LOCAL,
    FIG01_BUILDER_VERSION,
    FINAL,
    NE_COUNTRIES,
    NE_STATES,
    REPO,
    SCRIPT_VERSION,
    SLV_BASIN,
    SLV_BASIN_SOURCE,
)
from .sources import (
    container_coord_crs,
    e1_configured,
    e2_configured,
    e3_configured,
    flag_evaluation_pool,
    require_layer_crs_consistent,
)

# --------------------------------------------------------------------------
# Figure 1 -- scope, evidence matrix, architecture
# --------------------------------------------------------------------------


def _audit_key(src: str) -> str:
    """Non-identifying, deterministic audit key for a restricted source record.

    A salted SHA256 truncated to 16 hex characters.  It is stable across
    rebuilds (so an auditor holding the restricted source list can verify the
    linkage) but carries no source-agency identifier into the public layer.
    """
    salt = "swim-rs/fig01/e3-display/v1"
    return hashlib.sha256(f"{salt}|{src}".encode()).hexdigest()[:16]


def _fig01_load_e1_site_series(fid: str):
    """Archived canonical-run daily series joined to the raw + interpolated benchmark.

    Resurrected verbatim from the retired seasonal fig03 builder (commit
    782ca3c) so the example-source regeneration below reproduces the original
    derivation byte-compatibly.  The benchmark columns exist only so the
    frozen step-2 window rule can be evaluated; they are refused from the
    Figure 1 display table by FIG01_FORBIDDEN_EXAMPLE_COLUMNS.
    """
    p = E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{fid}.csv"
    b = FIG01_LEGACY_OPENET_DAILY / f"{fid}.csv"
    if not p.exists() or not b.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    raw = pd.read_csv(b)
    date_col = "DATE" if "DATE" in raw.columns else "date"
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.set_index(date_col)
    if "ensemble_mean_3x3" not in raw.columns:
        return None
    ens = pd.to_numeric(raw["ensemble_mean_3x3"], errors="coerce")
    finite = ens.dropna()
    if finite.empty:
        return None
    idx = pd.date_range(finite.index.min(), finite.index.max(), freq="D")
    ens_daily = ens.reindex(idx).interpolate(method="linear", limit_area="inside")
    df = df.rename(columns={"is_overpass": "is_calibration_capture_archived"})
    df["benchmark_raw"] = ens.reindex(df.index)
    df["benchmark_interpolated"] = ens_daily.reindex(df.index)
    df["is_direct_benchmark"] = df["benchmark_raw"].notna()
    return df


def _fig01_first_qualifying_window(series: pd.DataFrame, window_days: int):
    """The frozen step-2 window rule and step-5 earliest-start tiebreak (782ca3c)."""
    need = ["flux_ET", "swim_ET", "benchmark_interpolated"]
    for c in need:
        if c not in series.columns:
            return None
    for yr in sorted(set(series.index.year)):
        for month, day in [(4, 1), (5, 1), (6, 1), (7, 1)]:
            start = pd.Timestamp(year=yr, month=month, day=day)
            end = start + pd.Timedelta(days=window_days - 1)
            if end > series.index.max():
                continue
            w = series.loc[start:end]
            if len(w) < window_days:
                continue
            complete = w[need].notna().all(axis=1)
            frac = float(complete.mean())
            n_direct = int((w["is_direct_benchmark"] & complete).sum())
            n_paired = int(complete.sum())
            n_interp = n_paired - n_direct
            n_cal = int(w["is_calibration_capture_archived"].fillna(False).astype(bool).sum())
            if (
                frac >= 0.90
                and n_direct >= 3
                and n_paired > 0
                and (n_interp / n_paired) >= 0.50
                and n_cal >= 1
            ):
                return (
                    start,
                    end,
                    {
                        "complete_fraction": frac,
                        "n_paired_days": n_paired,
                        "n_direct_benchmark_days": n_direct,
                        "n_benchmark_interpolated_days": n_interp,
                        "n_calibration_captures": n_cal,
                    },
                )
    return None


def _fig01_raw_ndvi(fid: str) -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E1_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    j = uid.index(fid)
    t = pd.to_datetime(z["time/daily"][:])
    return pd.DataFrame(
        {
            "ndvi_landsat_raw": np.asarray(
                z["remote_sensing/ndvi/landsat/no_mask"][:, j], dtype=float
            ),
            "ndvi_sentinel_raw": np.asarray(
                z["remote_sensing/ndvi/sentinel/no_mask"][:, j], dtype=float
            ),
        },
        index=t,
    )


def _build_fig01_example_source() -> None:
    """Regenerate the example source series/selection for the directed site.

    The original six-step rule (frozen in fig03_example_selection.json) chose
    US-Bi1 as the site whose benchmark-interpolated d_kge sat exactly at the
    43-site cohort median.  The 2026-08-27 user ruling overrides steps 3-4 on
    agronomic-representativeness grounds -- US-Bi1 is a below-sea-level
    Sacramento-San Joaquin Delta island (elev -2.7 m) on organic soils with a
    shallow managed water table, atypical of the irrigated semi-arid interior
    West this paper centers on -- and directs the site to S2 (Harney Basin,
    OR; 1,255 m; irrigated alfalfa; eddy covariance), the intermountain
    irrigated site in the frozen eligible list (rank 10 of 43).  Steps 1-2
    (cohort and window qualification) and the step-5 earliest-qualifying-
    window tiebreak are unchanged and re-run live; the frozen eligible list,
    ranking and rule text are carried forward, never recomputed (their
    ranking inputs were superseded by the ETf-reconstructed Figure 3 design).

    The historical fig03_* filenames are retained: downstream consumers
    (_build_fig01_example, the architecture, the manifest) address the files
    by these names and the manifest records their fig01-source role.
    """
    prior_path = paths.OUT / "fig03_example_selection.json"
    if not prior_path.exists():
        raise BuildError("fig01 example source: fig03_example_selection.json is missing")
    prior = json.loads(prior_path.read_text())
    eligible = prior["eligible_sites"]
    if FIG01_EXAMPLE_SITE not in eligible:
        raise BuildError(
            f"fig01 example source: directed site {FIG01_EXAMPLE_SITE!r} is not in the "
            "frozen eligible list; the override may not widen eligibility"
        )

    # Quarantine the superseded US-Bi1 source files once, on the first
    # regeneration after the override.
    if prior.get("selected_site") != FIG01_EXAMPLE_SITE:
        qdir = FINAL / "figures" / "superseded_fig01_usbi1_example"
        qdir.mkdir(parents=True, exist_ok=True)
        for name in ("fig03_example_timeseries.csv", "fig03_example_selection.json"):
            src = paths.OUT / name
            if src.exists():
                (qdir / name).write_bytes(src.read_bytes())

    series = _fig01_load_e1_site_series(FIG01_EXAMPLE_SITE)
    if series is None:
        raise BuildError(f"fig01 example source: no archived series for {FIG01_EXAMPLE_SITE!r}")
    win = _fig01_first_qualifying_window(series, int(prior["selection_rule"]["window_length_days"]))
    if win is None:
        raise BuildError(f"fig01 example source: {FIG01_EXAMPLE_SITE!r} has no qualifying window")
    start, end, diag = win
    if str(start.date()) != FIG01_EXAMPLE_START or str(end.date()) != FIG01_EXAMPLE_END:
        raise BuildError(
            "fig01 example source: the earliest qualifying window "
            f"({start.date()}..{end.date()}) differs from the frozen contract "
            f"({FIG01_EXAMPLE_START}..{FIG01_EXAMPLE_END}); update the constants only "
            "after re-auditing the record"
        )
    sub = series.loc[start:end].copy()
    sub = sub.join(_fig01_raw_ndvi(FIG01_EXAMPLE_SITE), how="left")

    om = pd.read_csv(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv")
    om = om[(om["site"] == FIG01_EXAMPLE_SITE) & (om["model"] == "ensemble")].copy()
    om["date"] = pd.to_datetime(om["date"])
    om = om.set_index("date")
    sub["calibration_target_etf"] = om["target_etf"].reindex(sub.index)
    sub["calibration_target_member_count"] = om["member_count"].reindex(sub.index)
    sub["calibration_target_ensemble_std"] = om["ensemble_std"].reindex(sub.index)
    sub["calibration_target_final_weight"] = om["final_weight"].reindex(sub.index)
    sub["is_calibration_capture"] = sub["calibration_target_etf"].notna()

    out = sub.reset_index().rename(columns={"index": "date"})
    out.insert(0, "site_id", FIG01_EXAMPLE_SITE)
    out.insert(0, "legacy_prefix", "e2_")
    out.insert(0, "experiment", "E1")
    n = write_table(out, "fig03_example_timeseries.csv")
    require_count(n, FIG01_EXAMPLE_DAYS, "fig01 example source daily rows")

    payload = dict(prior)
    payload["figure"] = "fig01"
    payload["role"] = (
        "Figure 1 example source (historical fig03_ filename retained). Regenerated by "
        "the fig01-owned example-source generator resurrected 2026-08-27 from the "
        "retired seasonal fig03 builder (commit 782ca3c)."
    )
    payload["selection_override_2026_08_27"] = {
        "directed_site": FIG01_EXAMPLE_SITE,
        "directed_by": "user ruling, 2026-08-27",
        "overrides": "steps 3-4 (median-rank site selection) only",
        "reason": (
            "agronomic representativeness: US-Bi1 is a below-sea-level Sacramento-San "
            "Joaquin Delta island (elev -2.7 m) on organic soils with a shallow managed "
            "water table, atypical of the irrigated semi-arid interior West; S2 (Harney "
            "Basin, OR; 1,255 m; irrigated alfalfa; eddy covariance) is the intermountain "
            "irrigated site in the frozen eligible list, rank 10 of 43 by the original "
            "median-distance ranking"
        ),
        "independence": (
            "the override was directed on irrigation setting and physiographic region "
            "only; no performance metric, residual or ranking value informed it. The "
            "step-2 window rule and step-5 earliest-qualifying-window tiebreak are "
            "unchanged and were re-run live against the archived run22 series."
        ),
        "superseded_source_files": "superseded_fig01_usbi1_example/",
    }
    payload["selected_site"] = FIG01_EXAMPLE_SITE
    payload["selected_window_start"] = str(start.date())
    payload["selected_window_end"] = str(end.date())
    payload["selected_window_diagnostics"] = diag
    rank_row = next(r for r in prior["ranking"] if r["site_id"] == FIG01_EXAMPLE_SITE)
    payload["selected_site_rank_value"] = float(rank_row["rank_value"])
    payload["selected_site_d_kge_benchmark_interpolated"] = float(rank_row["d_kge"])
    payload["sources"] = {
        "site_daily_series": {
            "path": str(
                E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{FIG01_EXAMPLE_SITE}.csv"
            ),
            "sha256": sha256(
                E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{FIG01_EXAMPLE_SITE}.csv"
            ),
        },
        "raw_openet_benchmark": {
            "path": str(FIG01_LEGACY_OPENET_DAILY / f"{FIG01_EXAMPLE_SITE}.csv"),
            "sha256": sha256(FIG01_LEGACY_OPENET_DAILY / f"{FIG01_EXAMPLE_SITE}.csv"),
        },
        "observation_metadata": {
            "path": str(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv"),
            "sha256": sha256(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv"),
        },
        "container_raw_ndvi": {
            "path": str(E1_CONTAINER),
            "arrays": [
                "remote_sensing/ndvi/landsat/no_mask",
                "remote_sensing/ndvi/sentinel/no_mask",
            ],
            "note": "zarr store; per-array hash not computed (directory store)",
        },
    }
    payload["generator_script"] = "scripts/figures/build_figure_data.py"
    payload["generator_version"] = SCRIPT_VERSION
    (paths.OUT / "fig03_example_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )

    src_note = (
        "Figure 1 example source (historical fig03_ filename retained); regenerated by "
        "the fig01-owned generator under the 2026-08-27 directed-site override recorded "
        "in fig03_example_selection.json. Not a Figure 3 input."
    )
    MANIFEST.add(
        "fig03_example_timeseries.csv",
        rows=n,
        role=src_note,
        selected_site=FIG01_EXAMPLE_SITE,
        window=[FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
    )
    MANIFEST.add("fig03_example_selection.json", role=src_note)
    print(
        f"  fig01: example source regenerated for {FIG01_EXAMPLE_SITE} "
        f"{FIG01_EXAMPLE_START}..{FIG01_EXAMPLE_END} ({n} rows, "
        f"{diag['n_calibration_captures']} captures)"
    )


def _build_fig01_example() -> tuple[pd.DataFrame, dict, dict]:
    """Freeze the directed example record for Figure 1 panel (a).

    Reads only two audited frozen artifacts -- the example source series
    (regenerated by _build_fig01_example_source under the 2026-08-27
    directed-site override) and the Figure 4 per-capture member values -- so
    the display table derives from audited records, never ad hoc from a
    container.

    Returns ``(display_frame, selection_payload, column_provenance)``.
    """
    src_series = paths.OUT / "fig03_example_timeseries.csv"
    src_members = paths.OUT / "fig04_spread_capture_values.csv"
    for p in (src_series, src_members):
        if not p.exists():
            raise BuildError(
                f"fig01 example: required frozen source {p.name} is missing; build it first "
                "(fig01 is built after fig03/fig04 by design)"
            )

    ser = pd.read_csv(src_series, parse_dates=["date"])
    ser = ser[(ser["experiment"] == "E1") & (ser["site_id"] == FIG01_EXAMPLE_SITE)].copy()
    start = pd.Timestamp(FIG01_EXAMPLE_START)
    end = pd.Timestamp(FIG01_EXAMPLE_END)
    ser = ser[(ser["date"] >= start) & (ser["date"] <= end)].sort_values("date")

    # ---- section 11: the example must match the frozen selection exactly ----
    sites = sorted(set(ser["site_id"]))
    if sites != [FIG01_EXAMPLE_SITE]:
        raise BuildError(
            f"fig01 example: expected the frozen site {FIG01_EXAMPLE_SITE!r}, got {sites}"
        )
    require_count(len(ser), FIG01_EXAMPLE_DAYS, "fig01 example daily rows")
    if str(ser["date"].min().date()) != FIG01_EXAMPLE_START or (
        str(ser["date"].max().date()) != FIG01_EXAMPLE_END
    ):
        raise BuildError(
            "fig01 example: date window differs from the frozen selection "
            f"({ser['date'].min().date()}..{ser['date'].max().date()} vs "
            f"{FIG01_EXAMPLE_START}..{FIG01_EXAMPLE_END})"
        )
    expected_index = pd.date_range(start, end, freq="D")
    if not ser["date"].reset_index(drop=True).equals(pd.Series(expected_index)):
        raise BuildError("fig01 example: the 120-day window is not a contiguous daily index")
    require_columns(
        ser,
        [
            "eto",
            "precip",
            "rz_depletion",
            "irr_applied",
            "swim_ET",
            "flux_ET",
            "swe",
            "sensor",
            "ndvi_landsat_raw",
            "ndvi_sentinel_raw",
            "is_calibration_capture",
            "calibration_target_etf",
            "calibration_target_member_count",
            "calibration_target_ensemble_std",
        ],
        "fig01 example source series",
    )
    caps_mask = ser["is_calibration_capture"].astype(bool)
    require_count(
        int(caps_mask.sum()), FIG01_EXAMPLE_CAPTURES, "fig01 example calibration captures"
    )
    for col in ("eto", "precip", "rz_depletion", "irr_applied", "swim_ET", "flux_ET"):
        n_null = int(ser[col].isna().sum())
        if n_null:
            raise BuildError(
                f"fig01 example: daily column {col!r} has {n_null} missing value(s) in the frozen "
                "window; investigate upstream rather than filling or dropping them"
            )

    mem = pd.read_csv(src_members, parse_dates=["date"])
    mem = mem[(mem["experiment"] == "E1") & (mem["site_id"] == FIG01_EXAMPLE_SITE)].copy()
    mem = mem[(mem["date"] >= start) & (mem["date"] <= end)].sort_values("date")
    require_count(len(mem), FIG01_EXAMPLE_CAPTURES, "fig01 example member-value captures")
    member_cols = [f"etf_{m}" for m in E1_MEMBERS]
    require_columns(
        mem,
        ["member_count", "spread_retrieval", "ensemble_mean_etf", *member_cols],
        "fig01 example member source",
    )
    if set(ser.loc[caps_mask, "date"]) != set(mem["date"]):
        raise BuildError(
            "fig01 example: the Figure 3 calibration-capture dates and the Figure 4 member "
            "capture dates disagree; the two frozen artifacts are not describing one record"
        )

    joined = ser.loc[caps_mask, ["date"]].merge(mem, on="date", how="left", validate="one_to_one")

    # ---- section 11: plotted member marks must reconcile to the frozen target ----
    n_members = joined[member_cols].notna().sum(axis=1)
    if not bool((n_members == joined["member_count"]).all()):
        bad = joined.loc[n_members != joined["member_count"], "date"].dt.date.tolist()
        raise BuildError(
            f"fig01 example: non-null member marks disagree with the frozen member_count at {bad}"
        )
    plain_mean = joined[member_cols].mean(axis=1, skipna=True)
    d_mean = float((plain_mean - joined["ensemble_mean_etf"]).abs().max())
    if not np.isfinite(d_mean) or d_mean > 1e-6:
        raise BuildError(
            "fig01 example: the frozen target mean is not the plain mean of the available "
            f"members (max abs difference {d_mean:.3e}); investigate the target composition "
            "before plotting member marks against it"
        )
    plain_spread = joined[member_cols].std(axis=1, skipna=True, ddof=1)
    d_spread = float((plain_spread - joined["spread_retrieval"]).abs().max())
    if not np.isfinite(d_spread) or d_spread > 1e-6:
        raise BuildError(
            "fig01 example: the frozen ensemble spread is not the sample standard deviation "
            f"of the available members (max abs difference {d_spread:.3e})"
        )
    # The Figure 3 series and the Figure 4 capture table must carry the same
    # target, member count and spread, or the panel would mix two vintages.
    x = ser.loc[
        caps_mask,
        [
            "date",
            "calibration_target_etf",
            "calibration_target_member_count",
            "calibration_target_ensemble_std",
        ],
    ]
    x = x.merge(mem, on="date", how="left", validate="one_to_one")
    for a, b in (
        ("calibration_target_etf", "ensemble_mean_etf"),
        ("calibration_target_member_count", "member_count"),
        ("calibration_target_ensemble_std", "spread_retrieval"),
    ):
        d = float((x[a] - x[b]).abs().max())
        if d != 0.0:
            raise BuildError(
                f"fig01 example: fig03 {a!r} and fig04 {b!r} disagree by {d:.3e}; the two frozen "
                "artifacts must describe the identical calibration targets"
            )

    out = ser[["experiment", "legacy_prefix", "site_id", "date"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["ndvi_landsat_raw"] = ser["ndvi_landsat_raw"].to_numpy()
    out["ndvi_sentinel_raw"] = ser["ndvi_sentinel_raw"].to_numpy()
    out["is_calibration_capture"] = caps_mask.to_numpy()
    out["capture_sensor"] = ser["sensor"].to_numpy()
    cap_index = ser.index[caps_mask]
    for col in member_cols:
        s = pd.Series(np.nan, index=ser.index, dtype=float)
        s.loc[cap_index] = joined[col].to_numpy()
        out[col] = s.to_numpy()
    for name, src in (
        ("etf_target_mean", "ensemble_mean_etf"),
        ("etf_member_count", "member_count"),
        ("etf_ensemble_spread", "spread_retrieval"),
    ):
        s = pd.Series(np.nan, index=ser.index, dtype=float)
        s.loc[cap_index] = joined[src].to_numpy()
        out[name] = s.to_numpy()
    out["eto"] = ser["eto"].to_numpy()
    out["precip"] = ser["precip"].to_numpy()
    out["rz_depletion"] = ser["rz_depletion"].to_numpy()
    out["irr_applied"] = ser["irr_applied"].to_numpy()
    out["swim_ET"] = ser["swim_ET"].to_numpy()
    out["flux_ET"] = ser["flux_ET"].to_numpy()
    out["swe_audit"] = ser["swe"].to_numpy()
    out = out.reset_index(drop=True)

    fig03_src = "paper/data/final/figures/fig03_example_timeseries.csv"
    fig04_src = "paper/data/final/figures/fig04_spread_capture_values.csv"
    prov: dict[str, dict] = {
        "experiment": {
            "source": fig03_src,
            "source_column": "experiment",
            "units": None,
            "display_role": "key",
        },
        "legacy_prefix": {
            "source": fig03_src,
            "source_column": "legacy_prefix",
            "units": None,
            "display_role": "audit_only",
        },
        "site_id": {
            "source": fig03_src,
            "source_column": "site_id",
            "units": None,
            "display_role": "key",
            "note": "public AmeriFlux site identifier",
        },
        "date": {
            "source": fig03_src,
            "source_column": "date",
            "units": "date (YYYY-MM-DD)",
            "display_role": "shared_date_axis",
        },
        "ndvi_landsat_raw": {
            "source": fig03_src,
            "source_column": "ndvi_landsat_raw",
            "units": "dimensionless",
            "display_role": "evidence_row_ndvi",
            "note": "raw Landsat capture; never connected or filled",
        },
        "ndvi_sentinel_raw": {
            "source": fig03_src,
            "source_column": "ndvi_sentinel_raw",
            "units": "dimensionless",
            "display_role": "evidence_row_ndvi",
            "note": "raw Sentinel-2 capture; never connected or filled",
        },
        "is_calibration_capture": {
            "source": fig03_src,
            "source_column": "is_calibration_capture",
            "units": None,
            "display_role": "structure",
            "note": (
                f"marks the {FIG01_EXAMPLE_CAPTURES} acquisition dates that carry an ETf "
                "calibration target"
            ),
        },
        "capture_sensor": {
            "source": fig03_src,
            "source_column": "sensor",
            "units": None,
            "display_role": "audit_only",
            "note": (
                "instrument of the ETf calibration capture (all 15 are landsat here); it is NOT "
                "the NDVI capture sensor -- NDVI sensor identity is carried by the two separate "
                "raw NDVI columns"
            ),
        },
    }
    for m, col in zip(E1_MEMBERS, member_cols, strict=True):
        prov[col] = {
            "source": fig04_src,
            "source_column": col,
            "units": "dimensionless ETf",
            "display_role": "evidence_row_etf_member",
            "note": (
                f"one OpenET v2.1 retrieval member at the capture; plotted as a small neutral "
                f"mark without a member-name legend (handoff section 5.2). Member key {m!r} is "
                "audit provenance, not reader-facing copy."
            ),
        }
    prov["etf_target_mean"] = {
        "source": fig04_src,
        "source_column": "ensemble_mean_etf",
        "units": "dimensionless ETf",
        "display_role": "evidence_row_etf_target",
        "note": (
            "the calibration target; verified equal to the plain mean of the available members "
            f"to {d_mean:.3e} and equal to fig03 calibration_target_etf exactly"
        ),
    }
    prov["etf_member_count"] = {
        "source": fig04_src,
        "source_column": "member_count",
        "units": "count",
        "display_role": "audit_only",
        "note": "verified equal to the number of non-null member columns at every capture",
    }
    prov["etf_ensemble_spread"] = {
        "source": fig04_src,
        "source_column": "spread_retrieval",
        "units": "dimensionless ETf",
        "display_role": "evidence_row_etf_target",
        "note": (
            "retrieval-member dispersion; verified equal to the ddof=1 sample standard deviation "
            f"of the available members to {d_spread:.3e}"
        ),
    }
    prov["eto"] = {
        "source": fig03_src,
        "source_column": "eto",
        "units": "mm d-1",
        "display_role": "evidence_row_forcing",
    }
    prov["precip"] = {
        "source": fig03_src,
        "source_column": "precip",
        "units": "mm d-1",
        "display_role": "evidence_row_forcing",
    }
    prov["rz_depletion"] = {
        "source": fig03_src,
        "source_column": "rz_depletion",
        "units": "mm",
        "display_role": "evidence_row_state",
        "note": "root-zone depletion; the primary visual evidence of state propagation",
    }
    prov["irr_applied"] = {
        "source": fig03_src,
        "source_column": "irr_applied",
        "units": "mm d-1",
        "display_role": "e1_example_irrigation_stems_only",
        "note": (
            f"simulated gross applied water for the {FIG01_EXAMPLE_SITE} E1 example. It is drawn ONCE, as the "
            "magnitude-bearing applied-water stems in the daily-outputs row, and nowhere else: "
            "the triangle event lane that repeated these events in the daily-state row is "
            "removed (handoff section 5.2). This column supports only the E1 example display. It "
            "must never be treated as an E3 modeled series, aggregated to an annual total for "
            "display, or connected -- by arrow, bracket, tie or shared glyph -- to the separate "
            "'Metered Water · E3' key. E3 evaluates applied water simulated for E3 fields, not "
            "this record (handoff sections 5.2, 5.4 and 10.2)."
        ),
        "forbidden_uses": [
            "an E3 modeled series",
            "any seasonal or annual aggregation drawn in the artwork",
            "any drawn connection to the E3 meter key or its Annual Total operation",
            "a second irrigation-event encoding in the daily-state row",
        ],
    }
    prov["swim_ET"] = {
        "source": fig03_src,
        "source_column": "swim_ET",
        "units": "mm d-1",
        "display_role": "evidence_row_output",
        "note": (
            "drawn full width on the shared date axis; it and flux_ET must use one common date "
            "mapping at comparable horizontal extent (handoff sections 5.2, 5.4 and 11)"
        ),
    }
    prov["flux_ET"] = {
        "source": fig03_src,
        "source_column": "flux_ET",
        "units": "mm d-1",
        "display_role": "held_out_observation",
        "note": (
            "the actual example-site flux record, drawn as a thin near-black trace at FULL panel width "
            "on the same date mapping as swim_ET -- never compressed into a right-hand side "
            "strip. A quiet held-out rule or neutral comparison bracket separates the two; no "
            "arrowhead may point into it, and no metric, residual or agreement claim may "
            "accompany it (handoff sections 5.4 and 11)."
        ),
    }
    prov["swe_audit"] = {
        "source": fig03_src,
        "source_column": "swe",
        "units": "mm",
        "display_role": "audit_only_not_plotted",
        "note": (
            "gridded SWE is identically zero across this growing-season window. Handoff section "
            "5.3 requires SWE to be represented symbolically as an auxiliary calibration "
            "constraint; a zero-valued trace must not be forced into the panel. Frozen here only "
            "so the absence is auditable."
        ),
    }

    # ---- section 11: refused and unprovenanced columns ----
    for col in out.columns:
        for pat, why in FIG01_FORBIDDEN_EXAMPLE_COLUMNS:
            if re.search(pat, col, flags=re.IGNORECASE):
                raise BuildError(f"fig01 example: column {col!r} is refused -- {why}")
    missing_prov = [c for c in out.columns if c not in prov]
    if missing_prov:
        raise BuildError(
            f"fig01 example: emitted column(s) {missing_prov} have no recorded provenance"
        )
    stale_prov = [c for c in prov if c not in out.columns]
    if stale_prov:
        raise BuildError(f"fig01 example: provenance recorded for absent column(s) {stale_prov}")

    selection = {
        "figure": "fig01",
        "panel": "a",
        "experiment": "E1",
        "legacy_prefix": "e2_",
        "role": (
            "Illustrative cross-figure visual motif. Figure 1 uses this record to show data "
            "form, acquisition cadence and state propagation only. Figure 3 retains ownership "
            "of the SWIM-RS-versus-benchmark reconstruction comparison, the direct-versus-"
            "interpolated benchmark distinction, and every performance metric."
        ),
        "selected_site": FIG01_EXAMPLE_SITE,
        "full_window": {
            "start": FIG01_EXAMPLE_START,
            "end": FIG01_EXAMPLE_END,
            "n_days": int(len(out)),
            "n_calibration_captures": int(caps_mask.sum()),
        },
        "displayed_window": {
            "start": FIG01_EXAMPLE_START,
            "end": FIG01_EXAMPLE_END,
            "n_days": int(len(out)),
            "n_calibration_captures": int(caps_mask.sum()),
            "cropped": False,
            "crop_rationale": None,
            "note": (
                "The full 120-day record is displayed; no crop is applied. If a later "
                "composition crops it, the exact dates and an input/state-based rationale must "
                "be recorded here before the crop is rendered (handoff section 5.1)."
            ),
        },
        "rationale": (
            "The record comes from the frozen six-step rule in fig03_example_selection.json "
            "as amended by the recorded 2026-08-27 directed-site override: the site is S2 "
            "(Harney Basin, OR; irrigated intermountain alfalfa), directed on agronomic-"
            "representativeness grounds from within the frozen eligible list, with the window "
            "chosen by the unchanged step-2 rule and earliest-qualifying tiebreak. It is a "
            f"growing-season window containing sparse Landsat and Sentinel-2 NDVI captures, "
            f"{FIG01_EXAMPLE_CAPTURES} ETf calibration captures with visible retrieval-member "
            "disagreement, precipitation and simulated irrigation events, and an uninterrupted "
            "daily state trajectory -- the data forms panel (a) must show."
        ),
        "selection_independence": (
            "Neither the choice of this example nor its displayed window used evaluation "
            "residuals, flux or meter errors, benchmark errors, or any performance metric. "
            "The original rule ranked candidate sites toward the cohort MEDIAN benchmark-"
            "interpolated delta rather than toward favorable performance; the 2026-08-27 "
            "override directed the site on irrigation setting and physiographic region only, "
            "from within the frozen eligible list, and no metric value is carried into, or "
            "plotted from, this Figure 1 file."
        ),
        "excluded_from_this_file": {
            "benchmark_raw / benchmark_interpolated / is_direct_benchmark": (
                "Figure 3 owns the benchmark reconstruction comparison (handoff section 8)"
            ),
            "ndvi_kcb": (
                "the daily filled/model NDVI trajectory; refused outright because its fill "
                "provenance is unresolved (handoff sections 5.2 and 11). It is not carried even "
                "as an audit column."
            ),
            "observed_etf / obs_weight / calibration_target_final_weight": (
                "observation weights and the fitted objective are caption/Methods-owned"
            ),
            "etf_model / ks": "not required by any panel (a) evidence row",
            "all performance metrics": "owned by Figures 2-6 and the text",
        },
        "display_role_constraints": {
            "_contract": (
                "paper/notes/fig01_production_handoff.md sections 5.2, 5.4, 10.2 and 11, revised "
                "2026-08-25. Re-audited for architecture 3.1.0 and carried through 3.1.1 into "
                "3.2.0: the frozen column values are unchanged in every revision, the permitted "
                "display roles are not."
            ),
            "irr_applied": (
                "Supports ONLY the E1 example display, as one magnitude-bearing applied-water "
                "stem series in the daily-outputs row. It is not an E3 modeled series. It must "
                "not be summed for display, and it must not be connected by arrow, bracket, tie "
                "or shared glyph to the 'Daily Gross Applied Water -> Annual Total <-> Metered "
                "Water · E3' key, which is an experiment-level schematic describing E3 "
                "simulations. The "
                "builder additionally refuses any drawn path from the E1 applied-water node to "
                "the E3 meter node. If a future proof replaces the schematic key with data "
                "marks, a separately frozen E3 modeled-and-observed record with an independently "
                "justified selection record must be added; this file's display role must not be "
                "extended to cover it."
            ),
            "irrigation_encoding": (
                "One encoding only. The applied-water stems already carry timing and magnitude, "
                "so the same events must not be repeated as triangles or another symbol lane. "
                "For architecture 3.2.0 the stems are integrated into the coordinated "
                "state-and-irrigation region with rz_depletion, on their own explicit magnitude "
                "scale; they must NOT be normalized to the depletion scale (handoff section 5.2)."
            ),
            "swim_ET_and_flux_ET": (
                "Both ET traces are drawn in ONE merged plotting region (architecture 3.2.0, "
                "handoff section 5.2 item 5) at full panel width, on identical date support and "
                "an identical vertical scale. Neither may be compressed into a side strip or "
                "separated into an independent full-height lane, and the relationship between "
                "them is shared alignment alone -- no bracket, no tie glyph, and never an "
                "arrowhead pointing into the flux observations."
            ),
            "swe_audit": (
                "Never plotted. SWE is represented as an inline '+ SWE' label on the constraint "
                "route entering the 'Compare' stage of the inverse cycle; the separate SWE chip "
                "of architecture 3.0.0 is removed."
            ),
        },
        "reconciliation": {
            "member_count_matches_non_null_member_marks": True,
            "target_mean_is_plain_mean_of_available_members": True,
            "target_mean_max_abs_difference": d_mean,
            "spread_is_ddof1_sample_sd_of_available_members": True,
            "spread_max_abs_difference": d_spread,
            "fig03_targets_equal_fig04_targets": True,
            "tolerance_note": (
                "The residual ~1e-7 difference on the target mean is a storage precision "
                "artifact, not a method difference: fig04 member values are read from the "
                "float32 container arrays while the target and spread come from the archived "
                "float64 observation analysis. Member counts, and the fig03/fig04 target, "
                "member-count and spread columns, agree exactly (0.0)."
            ),
            "member_count_range": [
                int(joined["member_count"].min()),
                int(joined["member_count"].max()),
            ],
        },
        "sources": {
            "example_series": {
                "path": fig03_src,
                "sha256": sha256(src_series),
                "note": "frozen Figure 3 example record; read, not re-derived from a container",
            },
            "member_values": {
                "path": fig04_src,
                "sha256": sha256(src_members),
                "note": "frozen Figure 4 per-capture retrieval-member ETf values",
            },
            "upstream_selection_record": {
                "path": "paper/data/final/figures/fig03_example_selection.json",
                "sha256": sha256(paths.OUT / "fig03_example_selection.json"),
                "note": "the six-step rule that chose the site and window Figure 1 inherits",
            },
        },
        "column_provenance": prov,
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": FIG01_BUILDER_VERSION,
        "contract": (
            "paper/notes/fig01_production_handoff.md sections 5.1, 10.2 and 11, revised 2026-08-25"
        ),
        "record_identification": {
            "visible_string": FIG01_RECORD_ID,
            "color": FIG01_IDENTIFICATION_COLOR,
            "requirement": (
                "handoff sections 5.1, 11 and 13 require the record to be identified in the "
                "artwork; the reader must not have to infer the site and year from the caption. "
                "The parenthesized form and never-muted colour rule of architecture 3.2.0 "
                "(section 5.1, stated there for the superseded US-Bi1 record) carry over to "
                f"the directed record: identify it as {FIG01_RECORD_ID} in full-black text, "
                "no muted gray subtitle, no middle-dot construction. Section 11 stops the "
                "build if it is rendered as muted gray secondary text."
            ),
        },
        "re_audit": {
            "date": "2026-08-27",
            "against": (
                "fig01_production_handoff.md sections 10.2 and 15.2 (revised 2026-08-24) and "
                "the 2026-08-27 directed-site override in fig03_example_selection.json"
            ),
            "column_values_changed": True,
            "column_set_changed": False,
            "display_roles_changed": False,
            "detail": (
                "The example record moved from US-Bi1 (2017) to S2 (2018) under the recorded "
                "directed-site override; every column value is therefore new data from the "
                "same audited run22 archive derivation. The 2026-08-24 display-role re-audit "
                "(irr_applied narrowed to the E1 example display, single irrigation encoding, "
                "common swim_ET/flux_ET date mapping, SWE as inline label) carries forward "
                "unchanged."
            ),
        },
    }
    return out, selection, prov


def build_fig01() -> None:
    import geopandas as gpd
    from shapely.geometry import box

    # Regenerate the example source for the directed site before anything
    # consumes it (2026-08-27 override; see _build_fig01_example_source).
    _build_fig01_example_source()

    e1 = e1_configured()
    e2 = e2_configured()
    e3 = e3_configured()
    e2_sites = set(e2["site_id"])
    e1["in_e2"] = e1["site_id"].isin(e2_sites)
    e2["in_e1"] = e2["site_id"].isin(set(e1["site_id"]))
    require_count(int(e1["in_e2"].sum()), EXPECTED["E1_E2_overlap"], "fig01 E1->E2 overlap")
    require_count(int(e2["in_e1"].sum()), EXPECTED["E1_E2_overlap"], "fig01 E2->E1 overlap")
    # The paper's E2 evaluation pool (closure-corrected towers, frozen 2026-09-21).
    e2_pool_path = E2_FROZEN_POOL / "closure_pool_sites.csv"
    e2_pool = pd.read_csv(e2_pool_path)
    e2, n_pool_countries, n_pool_continents = flag_evaluation_pool(e2, e2_pool, "fig01")

    # ---- handoff section 11: configured counts, source classes, MB_Pch ----
    require_count(len(e1), EXPECTED["E1_configured"], "fig01 E1 configured sites")
    require_count(len(e2), EXPECTED["E2_configured"], "fig01 E2 configured sites")
    require_count(len(e3), EXPECTED["E3_fields"], "fig01 E3 configured fields")
    if "MB_Pch" not in set(e1["site_id"]):
        raise BuildError("fig01: MB_Pch is missing from the configured E1 scope")
    class_counts: dict[str, int] = {}
    for exp, frame in (("e1", e1), ("e2", e2)):
        if "irrigation_class" not in frame.columns:
            raise BuildError(f"fig01: {exp} carries no frozen irrigation_class assignment")
        counts = frame["irrigation_class"].value_counts().to_dict()
        seen = set(counts)
        if seen != {"irrigated", "rainfed"}:
            raise BuildError(
                f"fig01 {exp}: irrigation_class must be exactly irrigated/rainfed, got "
                f"{sorted(seen)}"
            )
        for cls_name in ("irrigated", "rainfed"):
            key = f"{exp}_{cls_name}"
            got = int(counts[cls_name])
            require_count(got, FIG01_CLASS_COUNTS[key], f"fig01 {exp} {cls_name} source class")
            class_counts[key] = got

    e1_daily = set(pd.read_csv(FINAL / "e2_primary_daily_site_metrics.csv")["fid"])
    e1["configured"] = True
    e1["in_daily_evaluation"] = e1["site_id"].isin(e1_daily)
    e1["source_key"] = e1["site_id"]
    e1["display_id"] = e1["site_id"]

    e2["configured"] = True
    e2["source_key"] = e2["site_id"]
    e2["display_id"] = e2["site_id"]

    # ---- E3 generalization: 1 km snapped centroids, no source identifiers ----
    grid_m = 1000.0
    e3_gdf = gpd.GeoDataFrame(
        e3.copy(),
        geometry=gpd.points_from_xy(e3["lon"], e3["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    snapped = gpd.points_from_xy(
        np.round(e3_gdf.geometry.x / grid_m) * grid_m,
        np.round(e3_gdf.geometry.y / grid_m) * grid_m,
    )
    e3_gdf = e3_gdf.set_geometry(
        gpd.GeoSeries(snapped, index=e3_gdf.index, crs="EPSG:5070")
    ).to_crs("EPSG:4326")
    e3_method = (
        f"field centroid snapped to a {int(grid_m)} m EPSG:5070 grid "
        "(approved 2026-08-19); no exact polygon, no source-agency identifier"
    )
    e3_disp = gpd.GeoDataFrame(
        {
            "display_id": [f"E3_{i:03d}" for i in range(len(e3_gdf))],
            "geometry_method": e3_method,
            "source_group": "SLV metered field",
            "source_key": [_audit_key(s) for s in e3_gdf["src_id"]],
            "crop_group": np.where(e3_gdf["crop"].str.upper() == "ALFALFA", "alfalfa", "other"),
        },
        geometry=e3_gdf.geometry,
        crs="EPSG:4326",
    )
    require_count(len(e3_disp), EXPECTED["E3_fields"], "fig01 E3 display rows")
    banned = {"src_id", "source", "site_id", "acres", "lat", "lon", "basin", "state"}
    if banned & set(e3_disp.columns):
        raise BuildError("fig01: restricted E3 attribute leaked into the public layer")
    # The public audit key must not be, contain, or be contained by any source
    # identifier, and the published geometry must be generalized points only.
    restricted = set(e3["src_id"].astype(str)) | set(e3["site_id"].astype(str))
    for col in e3_disp.columns:
        if col == "geometry":
            continue
        vals = {str(v) for v in e3_disp[col]}
        if vals & restricted:
            raise BuildError(f"fig01 e3_display: restricted identifier leaked into column {col!r}")
    # A substring scan is unsound here (source ids are 3-7 digit numbers that
    # occur inside hex digests by chance), so the public per-record fields are
    # instead constrained to formats that cannot carry a source identifier.
    if not all(re.fullmatch(r"E3_\d{3}", str(v)) for v in e3_disp["display_id"]):
        raise BuildError("fig01 e3_display: display_id must be a sequential E3_NNN display label")
    if not all(re.fullmatch(r"[0-9a-f]{16}", str(v)) for v in e3_disp["source_key"]):
        raise BuildError(
            "fig01 e3_display: source_key must be a 16-hex-character non-identifying audit key"
        )
    if set(e3_disp.geom_type) != {"Point"}:
        raise BuildError(
            f"fig01 e3_display: only generalized points may be published, got "
            f"{sorted(set(e3_disp.geom_type))}"
        )
    if len(set(e3_disp["source_key"])) != len(e3_disp):
        raise BuildError("fig01 e3_display: source_key is not one-to-one with the display records")

    e1_gdf = gpd.GeoDataFrame(
        e1[
            [
                "display_id",
                "configured",
                "irrigation_class",
                "in_e2",
                "in_daily_evaluation",
                "source_key",
            ]
        ],
        geometry=gpd.points_from_xy(e1["lon"], e1["lat"]),
        crs="EPSG:4326",
    )
    if not NE_COUNTRIES.exists():
        raise BuildError(f"fig01: boundary source missing {NE_COUNTRIES}")
    world = gpd.read_file(NE_COUNTRIES, engine="fiona")
    world_ctx = world[["ADMIN", "ISO_A3", "CONTINENT", "geometry"]].copy()
    world_ctx["boundary_source"] = "Natural Earth 110m admin 0 countries"
    world_ctx["boundary_version"] = _ne_version()

    e2_gdf = gpd.GeoDataFrame(
        e2[
            [
                "display_id",
                "configured",
                "network",
                "country",
                "in_e1",
                "in_evaluation_pool",
                "equipped_for_irrigation",
                "irrigation_class",
                "source_key",
            ]
        ].rename(columns={"country": "country_container_raw"}),
        geometry=gpd.points_from_xy(e2["lon"], e2["lat"]),
        crs="EPSG:4326",
    )
    # The container's country attribute is null for 13 sites and mixes ISO codes
    # with full names, so it cannot support the ten-country claim. Derive country
    # and continent from the archived boundary layer instead, falling back to the
    # nearest polygon for coastal sites the 110m coastline misses.
    poly = world_ctx[["ADMIN", "CONTINENT", "geometry"]].reset_index(drop=True)
    j = gpd.sjoin(e2_gdf[["geometry"]].reset_index(drop=True), poly, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    admin = list(j["ADMIN"])
    cont = list(j["CONTINENT"])
    assign = ["within_polygon"] * len(admin)
    proj_poly = poly.to_crs("EPSG:6933")
    for i, a in enumerate(admin):
        if isinstance(a, str):
            continue
        pt = e2_gdf.geometry.iloc[i : i + 1].to_crs("EPSG:6933").iloc[0]
        k = int(proj_poly.geometry.distance(pt).idxmin())
        admin[i] = str(proj_poly.loc[k, "ADMIN"])
        cont[i] = str(proj_poly.loc[k, "CONTINENT"])
        assign[i] = "nearest_polygon"
    if any(not isinstance(a, str) or a in ("nan", "None") for a in admin):
        raise BuildError("fig01: a configured E2 site could not be assigned a country")
    e2_gdf["country"] = admin
    e2_gdf["continent"] = cont
    e2_gdf["country_assignment"] = assign

    # Handoff section 10.1 required-attribute contract.
    required_attrs = {
        "e1_sites": ["display_id", "configured", "irrigation_class", "in_e2", "source_key"],
        "e2_sites": ["display_id", "network", "country", "in_e1", "source_key"],
        "e3_display": ["display_id", "geometry_method", "source_group"],
    }
    for lyr, cols in required_attrs.items():
        have = set({"e1_sites": e1_gdf, "e2_sites": e2_gdf, "e3_display": e3_disp}[lyr].columns)
        missing = [c for c in cols if c not in have]
        if missing:
            raise BuildError(f"fig01 {lyr}: missing required attributes {missing}")

    conus_bbox = box(-125.0, 24.0, -66.5, 49.5)
    conus_ctx = gpd.clip(world_ctx[world_ctx["ADMIN"] == "United States of America"], conus_bbox)
    conus_ctx = gpd.GeoDataFrame(conus_ctx, geometry="geometry", crs=world_ctx.crs)

    # ---- handoff sections 6.4 and 6.7: faint CONUS state boundaries ----
    # Archived Natural Earth 1:50m admin-1 units, the same NE release (5.1.1) as
    # the admin-0 source already used for world_context.  Public domain.  Alaska
    # and Hawaii are dropped so the layer matches the CONUS locator extent, and
    # the clip is the same box conus_context uses.
    if not NE_STATES.exists():
        raise BuildError(f"fig01: state-boundary source missing {NE_STATES}")
    states = gpd.read_file(NE_STATES, engine="fiona")
    states = states[(states["iso_a2"] == "US") & (~states["postal"].isin(["AK", "HI"]))]
    states_ctx = gpd.clip(states[["name", "postal", "geometry"]], conus_bbox)
    states_ctx = gpd.GeoDataFrame(states_ctx, geometry="geometry", crs=world_ctx.crs)
    states_ctx = states_ctx.sort_values("postal").reset_index(drop=True)
    states_ctx["boundary_source"] = "Natural Earth 50m admin 1 states provinces lakes"
    states_ctx["boundary_version"] = _ne_states_version()
    if len(states_ctx) != 49:
        raise BuildError(
            "fig01 conus_states_context: expected the 48 contiguous states plus the District of "
            f"Columbia, got {len(states_ctx)}"
        )

    slv_hull = e3_disp.union_all().convex_hull.buffer(0.15)
    slv_ctx = gpd.GeoDataFrame(
        {
            "name": ["San Luis Valley generalized study extent"],
            "geometry_method": [
                "convex hull of the 1 km snapped display centroids, buffered 0.15 degrees"
            ],
            "boundary_source": ["derived from generalized E3 display geometry"],
        },
        geometry=gpd.GeoSeries([slv_hull], crs="EPSG:4326"),
        crs="EPSG:4326",
    )

    # ---- handoff sections 6.6 and 6.7: privacy-safe E3 basin context ----
    # The five HUC8 hydrologic units of the Rio Grande headwaters accounting unit
    # (HUC6 130100), extracted from the USGS Watershed Boundary Dataset and
    # archived in-repo.  Each unit covers 1,987-6,576 km2, so the layer orients
    # the reader without localizing any field more precisely than the approved
    # 1 km centroid snap already does.
    if not SLV_BASIN.exists():
        raise BuildError(f"fig01: E3 basin-context source missing {SLV_BASIN}")
    basin_ctx = gpd.read_file(SLV_BASIN, engine="fiona")
    basin_ctx = basin_ctx.sort_values("huc8").reset_index(drop=True)
    basin_ctx["boundary_source"] = "USGS Watershed Boundary Dataset, WBDHU8"
    if len(basin_ctx) != 5 or set(basin_ctx["huc8"]) != {
        "13010001",
        "13010002",
        "13010003",
        "13010004",
        "13010005",
    }:
        raise BuildError(
            "fig01 slv_basin_context: expected the five HUC8 units of the Rio Grande headwaters "
            f"accounting unit, got {sorted(basin_ctx['huc8'])}"
        )
    # The basin layer must stay coarser than the approved display generalization:
    # it may not be a field boundary, and every generalized display point must
    # fall inside it, so it can add no location information the points lack.
    if float(basin_ctx["areasqkm"].min()) < 1000.0:
        raise BuildError(
            "fig01 slv_basin_context: a context unit smaller than 1000 km2 is not a coarse "
            "watershed boundary; refusing to publish it alongside the generalized field display"
        )
    inside = e3_disp.geometry.apply(lambda p: bool(basin_ctx.contains(p).any()))
    if not bool(inside.all()):
        raise BuildError(
            "fig01 slv_basin_context: the basin context does not contain every generalized E3 "
            "display point; a context layer that crops or excludes points would carry location "
            "information the approved display withholds"
        )

    gpkg = paths.OUT / "fig01_scope.gpkg"
    gpkg_tmp = paths.OUT / "fig01_scope.gpkg.tmp"
    if gpkg_tmp.exists():
        gpkg_tmp.unlink()
    gpkg_layers = {
        "e1_sites": e1_gdf,
        "e2_sites": e2_gdf,
        "e3_display": e3_disp,
        "conus_context": conus_ctx,
        "conus_states_context": states_ctx,
        "world_context": world_ctx,
        "slv_context": slv_ctx,
        "slv_basin_context": basin_ctx,
    }
    # Every published layer is EPSG:4326; verify the stored coordinates can
    # actually be degrees before writing, so a projected-metres layer can never
    # ship under a 4326 tag again (bug D3).
    for layer_name, layer_gdf in gpkg_layers.items():
        if layer_gdf.crs is None or layer_gdf.crs.to_epsg() != 4326:
            raise BuildError(
                f"fig01 {layer_name}: published layers must be EPSG:4326, got "
                f"{layer_gdf.crs.to_string() if layer_gdf.crs is not None else None}"
            )
        require_layer_crs_consistent(layer_gdf, f"fig01 {layer_name}")
    # Written to a sibling temp path and moved into place, so a concurrent
    # reader never observes a half-written GeoPackage.
    for layer_name, layer_gdf in gpkg_layers.items():
        layer_gdf.to_file(gpkg_tmp, layer=layer_name, driver="GPKG", engine="fiona")
    gpkg_tmp.replace(gpkg)

    countries = sorted(set(e2_gdf["country"]))
    n_countries = len(countries)
    continents = sorted(set(e2_gdf["continent"]))
    n_continents = len(continents)
    if n_countries != 10 or n_continents != 4:
        raise BuildError(
            "fig01: manuscript states ten E2 countries on four continents; "
            f"derived {n_countries} countries on {n_continents} continents"
        )

    evid = pd.DataFrame(
        [
            {
                "experiment": "E0",
                "evidence_role": "model_development",
                "configured_n": 37,
                "configured_unit": "E2 cropland flux sites not in E1",
                "domain": "E2 sites in eight countries; E2 inputs and processing",
                "primary_etf_target": "per-capture mean of coincident Landsat SSEBop and PT-JPL ETf (matched across formulations)",
                "primary_weighting": "spread-based (sigma_ensemble + 0.05)",
                "daily_evaluation_n": EXPECTED["E0_pooled_sites"],
                "monthly_supported_n": EXPECTED["E0_pooled_monthly_sites"],
                "monthly_finite_metric_n": EXPECTED["E0_effect_monthly_sites"],
                "field_year_n": None,
                "external_evaluation": "flux ET withheld from calibration, then compared across the three vegetation formulations to select the cover-scaled sigmoid",
                "parameter_source": "locally calibrated per formulation on E2 inputs",
                "scientific_roles": "vegetation-formulation comparison",
                "independence_statement": "model-development evidence on E2 sites not in E1, not independent validation",
                "source_artifact": f"paper/text/main.md Table 3; {E0_SITES_FILE.relative_to(REPO)}",
                "source_sha256": sha256(E0_SITES_FILE),
            },
            {
                "experiment": "E1",
                "evidence_role": "evaluation",
                "configured_n": 60,
                "configured_unit": "CONUS cropland flux sites",
                "domain": "CONUS cropland",
                "primary_etf_target": "per-capture mean of six OpenET v2.1 ETf members",
                "primary_weighting": "spread-based (sigma_ensemble + 0.1)",
                "daily_evaluation_n": 45,
                "monthly_supported_n": 31,
                "monthly_finite_metric_n": 29,
                "field_year_n": None,
                "external_evaluation": "Volk et al. (2024) v2.1 closure-corrected flux ET; separately extracted OpenET ensemble benchmark",
                "parameter_source": "local calibration; source cohort for the fixed irrigation-class parameter sets",
                "scientific_roles": "ET agreement; temporal reconstruction; ensemble reliability; held-out transfer",
                "independence_statement": "external to parameter estimation; no E1 site was used in the E0 formulation comparison",
                "source_artifact": "paper/data/final/e2_primary_daily_site_metrics.csv",
                "source_sha256": sha256(FINAL / "e2_primary_daily_site_metrics.csv"),
            },
            {
                "experiment": "E2",
                "evidence_role": "evaluation",
                "configured_n": 66,
                "configured_unit": "cropland flux sites",
                "domain": (
                    f"{n_countries} countries on {n_continents} continents configured; "
                    f"evaluated closure-corrected pool in {n_pool_countries} countries on "
                    f"{n_pool_continents} continents; CONUS and ex-CONUS"
                ),
                "primary_etf_target": "per-capture mean of coincident Landsat SSEBop and PT-JPL ETf",
                "primary_weighting": "spread-based (sigma_ensemble + 0.05)",
                "daily_evaluation_n": EXPECTED["E2_pool_daily"],
                "monthly_supported_n": EXPECTED["E2_pool_monthly_support"],
                "monthly_finite_metric_n": EXPECTED["E2_pool_monthly_finite"],
                "field_year_n": None,
                "external_evaluation": "AmeriFlux, FLUXNET, ICOS and OzFlux ET",
                "parameter_source": "local calibration arm; fixed E1-derived irrigated/rainfed sets for the transfer arm",
                "scientific_roles": "international evaluation; E1-to-E2 transfer under changed geography and inputs",
                "independence_statement": (
                    "13 of 66 configured sites (10 of the 47 evaluated) also occur in E1 and test "
                    "changed inputs; the rest are unseen fields; 37 of the E2 sites not in E1 carry "
                    "the E0 formulation comparison"
                ),
                "source_artifact": str(e2_pool_path),
                "source_sha256": sha256(e2_pool_path),
            },
            {
                "experiment": "E3",
                "evidence_role": "evaluation",
                "configured_n": 50,
                "configured_unit": "metered San Luis Valley fields",
                "domain": "San Luis Valley, Colorado",
                "primary_etf_target": "per-capture mean of six OpenET v2.1 ETf members, at least two valid members",
                "primary_weighting": "spread-based (sigma_ensemble + 0.1)",
                "daily_evaluation_n": None,
                "monthly_supported_n": None,
                "monthly_finite_metric_n": None,
                "field_year_n": EXPECTED["E3_field_years"],
                "external_evaluation": "state-agency groundwater pumping records",
                "parameter_source": "local calibration arm; fixed E1-derived irrigated set for the transfer arm",
                "scientific_roles": "applied-water consistency under local and transferred parameters",
                "independence_statement": "no overlap with the E1 source cohort; meters withheld from parameter estimation",
                "source_artifact": str(E3_LOCAL / "per_field_year.csv"),
                "source_sha256": sha256(E3_LOCAL / "per_field_year.csv"),
            },
        ]
    )
    # ---- handoff section 11 assertions on the evidence matrix ----
    require_columns(
        evid,
        [
            "experiment",
            "evidence_role",
            "configured_n",
            "configured_unit",
            "domain",
            "primary_etf_target",
            "primary_weighting",
            "daily_evaluation_n",
            "monthly_supported_n",
            "monthly_finite_metric_n",
            "field_year_n",
            "external_evaluation",
            "parameter_source",
            "scientific_roles",
            "source_artifact",
            "source_sha256",
        ],
        "fig01 evidence matrix",
    )
    require_unique(evid, ["experiment"], "fig01 evidence matrix")
    roles = dict(zip(evid["experiment"], evid["evidence_role"], strict=True))
    if roles.get("E0") != "model_development":
        raise BuildError("fig01: the E0 row must be typed evidence_role=model_development")
    for exp in ("E1", "E2", "E3"):
        if roles.get(exp) != "evaluation":
            raise BuildError(f"fig01: {exp} must be typed evidence_role=evaluation")
    conf = dict(zip(evid["experiment"], evid["configured_n"], strict=True))
    for exp, want in (("E1", 60), ("E2", 66), ("E3", 50)):
        require_count(int(conf[exp]), want, f"fig01 evidence matrix {exp} configured_n")
    tgt = dict(zip(evid["experiment"], evid["primary_etf_target"], strict=True))
    for exp in ("E1", "E3"):
        if "six" not in tgt[exp].lower():
            raise BuildError(
                f"fig01: {exp} primary ETf target must be the six-member OpenET mean, "
                f"not a two-member target; got {tgt[exp]!r}"
            )
    if "six" in tgt["E2"].lower() or "6-member" in tgt["E2"].lower():
        raise BuildError("fig01: E2 must not be labelled with a six-member primary ETf target")
    if not ("ssebop" in tgt["E2"].lower() and "pt-jpl" in tgt["E2"].lower()):
        raise BuildError("fig01: E2 primary ETf target must name the coincident Landsat pair")
    for exp, w in zip(evid["experiment"], evid["primary_weighting"], strict=True):
        head = w.split(";")[0].lower()
        if "fixed" in head or "spread" not in head:
            raise BuildError(f"fig01: {exp} primary weighting must be spread-weighted, got {w!r}")
    e0_ind = evid.loc[evid["experiment"] == "E0", "independence_statement"].iloc[0]
    if "independent validation" in e0_ind.lower().replace("not independent validation", ""):
        raise BuildError("fig01: E0 must not be labelled independent validation")

    n_ev = write_table(evid, "fig01_evidence_matrix.csv")

    # ---- panel (a) example record (handoff sections 5.1, 10.2, 11) ----
    example, example_selection, example_prov = _build_fig01_example()
    n_ex = write_table(example, "fig01_example_timeseries.csv")
    (paths.OUT / "fig01_example_selection.json").write_text(
        json.dumps(example_selection, indent=2, ensure_ascii=False)
    )
    n_ex_caps = int(example["is_calibration_capture"].sum())

    # ---- panel (a) plotting-region domains (handoff sections 5.2, 11 and 13) ----
    # Section 5.2: "The proof script must record the final limits and assert that
    # no plotted value is clipped."  The contract records the display limits and
    # the observed range of every plotted column behind each region; the section
    # 11 check below stops the build if a value falls outside its region's
    # recorded limits.  These are the same frozen columns in every revision, so
    # later tick or padding refinements inside these limits stay Level 1.
    def _region_data_range(columns: list[str]) -> list[float]:
        vals = pd.concat([example[c].astype(float) for c in columns], ignore_index=True)
        vals = vals[vals.notna()]
        if vals.empty:
            raise BuildError(
                f"fig01: plotting-region columns {columns} carry no values in the frozen example"
            )
        return [float(vals.min()), float(vals.max())]

    _etf_cols = [f"etf_{m}" for m in E1_MEMBERS] + ["etf_target_mean"]
    _region_ranges = {
        "etf_ensemble": _region_data_range(_etf_cols),
        "ndvi_captures": _region_data_range(["ndvi_landsat_raw", "ndvi_sentinel_raw"]),
        "daily_forcing": _region_data_range(["eto", "precip"]),
        "state_and_irrigation": _region_data_range(["rz_depletion"]),
        "state_and_irrigation_secondary": _region_data_range(["irr_applied"]),
        "et_comparison": _region_data_range(["swim_ET", "flux_ET"]),
    }

    def _y_axis(key: str, units: str, candidate, note: str | None = None) -> dict:
        lo, hi = FIG01_DISPLAY_DOMAINS[key]
        block = {
            "visible_spine": True,
            "display_domain": [lo, hi],
            "labeled_bounds": [lo, hi],
            "n_labeled_bounds": 2,
            "units": units,
            "units_placement": "adjacent to the axis or to the region label (handoff section 5.2)",
            "data_range": _region_ranges[key],
            "clipping_forbidden": True,
            "candidate_domain_handoff_5_2": candidate,
        }
        if note:
            block["domain_note"] = note
        return block

    arch = architecture_spec(_etf_cols, _region_ranges, _y_axis)

    # ---- frozen visible-string classification (handoff sections 9 and 10.3) ----
    cls: dict[str, str] = {}

    def _classify(s, k):
        if s is None:
            return
        if s in cls and cls[s] != k:
            raise BuildError(f"fig01: string {s!r} classified both {cls[s]} and {k}")
        cls[s] = k

    for p in arch["panels"]:
        _classify(p["letter"], "title")
        _classify(p["title"], "title")
    _classify(
        arch["example_record"]["record_label"],
        arch["example_record"]["record_label_string_class"],
    )
    for row in arch["plotting_regions"]:
        _classify(row["heading"], row["heading_string_class"])
        _classify(row["axis_label"], row["axis_label_string_class"])
        for dl in row.get("direct_labels", []):
            _classify(dl["label"], dl["string_class"])
        for lane in row.get("lanes", []):
            _classify(lane["label"], lane["string_class"])
        agg = row.get("aggregation_mark")
        if agg:
            _classify(agg["label"], agg["string_class"])
    _classify(arch["date_axis"]["label"], arch["date_axis"]["string_class"])
    _classify(arch["inverse_cycle"]["title"], arch["inverse_cycle"]["title_string_class"])
    for stage in arch["inverse_cycle"]["stages"]:
        _classify(stage["label"], stage["string_class"])
    _classify(
        arch["inverse_cycle"]["exit"]["label"],
        arch["inverse_cycle"]["exit"]["string_class"],
    )
    _classify(arch["swe_constraint"]["label"], arch["swe_constraint"]["string_class"])
    _classify(arch["driver_routing"]["label"], arch["driver_routing"]["string_class"])
    for o in arch["outputs"]:
        _classify(o["label"], o["string_class"])
    _classify(arch["held_out"]["heading"], arch["held_out"]["heading_string_class"])
    for o in arch["held_out"]["observations"]:
        _classify(o["label"], o["string_class"])
    for node in arch["e3_evaluation_key"]["nodes"]:
        _classify(node["label"], node["string_class"])
    for m in arch["map_nodes"]:
        _classify(m["heading"], m["heading_string_class"])
        _classify(m["count_line"], m["count_line_string_class"])
    for t in arch["parameter_tokens"]:
        _classify(t["label"], t["string_class"])
    _classify(arch["development_tag"]["label"], arch["development_tag"]["string_class"])
    arch["string_classification"] = dict(sorted(cls.items()))
    arch["visible_string_count"] = len(cls)
    arch["visible_strings_by_class"] = {
        k: sorted(s for s, c in cls.items() if c == k) for k in sorted(set(cls.values()))
    }

    if not set(cls.values()) <= VISIBLE_STRING_CLASSES:
        raise BuildError(f"fig01: unknown string class in {sorted(set(cls.values()))}")
    n_annot = sum(1 for v in cls.values() if v == "annotation")
    if n_annot > 3:
        raise BuildError(f"fig01: {n_annot} explanatory annotations exceed the budget of three")
    if arch["annotation_budget"]["used"] != n_annot:
        raise BuildError(
            f"fig01: annotation_budget.used={arch['annotation_budget']['used']} disagrees with "
            f"the {n_annot} string(s) classified as annotation"
        )
    for s, k in cls.items():
        if k == "annotation" and len(s.split()) > 7:
            raise BuildError(f"fig01: annotation {s!r} exceeds seven words")
    if len(cls) > 50:
        raise BuildError(f"fig01: {len(cls)} reader-facing strings exceed the limit of 50")
    _assert_no_caption_facts_visible(cls)
    # No string retired by ANY superseded architecture may reappear in visible
    # copy.  The union is taken over every supersession block so that adding a
    # revision cannot silently drop an older block's retired strings.
    retired: set[str] = set()
    for block in arch.values():
        if isinstance(block, dict) and "removed_strings" in block:
            retired |= set(block["removed_strings"])
    if (
        not {
            "Flux ET · E1–E2",
            "Σ year",
            "annual sum",
            "PEST++ IES",
            "US-Bi1 · 2017",
            "US-Bi1 (2017)",
            "Flux ET",
            "Held-Out Evaluation",
            "Inverse Estimation",
            "spread-weighted",
            "Daily State",
            "Daily Outputs",
            "Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3",
        }
        <= retired
    ):
        raise BuildError(
            "fig01: the retired-string set lost an entry; every superseded architecture's "
            "removed_strings must stay under the revival guard"
        )
    revived = sorted(retired & set(cls))
    if revived:
        raise BuildError(f"fig01: superseded string(s) {revived} reached visible copy")

    # ---- section 11 (2026-08-25): visible record identification ----
    if cls.get(FIG01_RECORD_ID) not in {"title", "direct_label"}:
        raise BuildError(
            f"fig01: the record identification {FIG01_RECORD_ID!r} must be visible artwork copy; "
            "the reader may not be left to infer the site and year from the caption "
            "(handoff sections 5.1, 11 and 13)"
        )
    if not arch["example_record"]["site_id_is_visible"]:
        raise BuildError(
            "fig01: example_record.site_id_is_visible must be true for 3.1.0 and later"
        )
    if FIG01_EXAMPLE_SITE not in FIG01_RECORD_ID or FIG01_EXAMPLE_START[:4] not in FIG01_RECORD_ID:
        raise BuildError(
            f"fig01: the record identification {FIG01_RECORD_ID!r} does not name the frozen "
            f"site {FIG01_EXAMPLE_SITE!r} and year {FIG01_EXAMPLE_START[:4]!r}"
        )

    # ---- section 11 (2026-08-25): the identification is near-black, not gray ----
    # "US-Bi1 (2017) is absent from the visible record identification or is
    # rendered as muted gray secondary text" stops the build.
    rec = arch["example_record"]
    if rec["record_label_rendered_as_muted_gray"]:
        raise BuildError(
            f"fig01: the record identification {FIG01_RECORD_ID!r} is marked as muted gray "
            "secondary text. Handoff sections 5.1, 9 and 11 require near-black identification "
            "and forbid washing it into a gray subtitle."
        )
    if rec["record_label_color"] != FIG01_IDENTIFICATION_COLOR:
        raise BuildError(
            f"fig01: the record identification must be near-black {FIG01_IDENTIFICATION_COLOR!r}, "
            f"got {rec['record_label_color']!r} (handoff sections 5.1 and 9)"
        )
    if arch["typography"]["identification_color"] != FIG01_IDENTIFICATION_COLOR:
        raise BuildError(
            "fig01: typography.identification_color must be the near-black "
            f"{FIG01_IDENTIFICATION_COLOR!r} required by handoff section 9"
        )
    gray_scope = {s.lower() for s in arch["typography"]["muted_gray_scope"]}
    reader_facing = {
        "the record identification",
        "plotting-region names",
        "units",
        "map count lines",
        "inverse-stage labels",
    }
    if gray_scope & reader_facing:
        raise BuildError(
            "fig01: muted gray is reserved for axes, rules and quiet geographic context; "
            f"{sorted(gray_scope & reader_facing)} is reader-facing copy (handoff section 9)"
        )
    if "·" in FIG01_RECORD_ID:
        raise BuildError(
            "fig01: handoff section 5.1 retires the middle-dot record identification used in "
            f"revisions 1-3; {FIG01_RECORD_ID!r} still carries it"
        )

    # ---- section 11 (2026-08-25): five quantitative plotting regions ----
    regions = arch["plotting_regions"]
    if len(regions) > FIG01_MAX_PLOTTING_REGIONS:
        raise BuildError(
            f"fig01: panel (a) carries {len(regions)} full-height plotting regions; handoff "
            f"sections 4, 5.2 and 11 allow at most {FIG01_MAX_PLOTTING_REGIONS}. Root-zone "
            "depletion and irrigation share one region; daily ET and held-out flux ET share "
            "another."
        )
    if [r["order"] for r in regions] != list(range(1, len(regions) + 1)):
        raise BuildError(
            f"fig01: the plotting regions must be ordered 1..{len(regions)}, got "
            f"{[r['order'] for r in regions]}"
        )
    if {r["id"] for r in regions} != set(FIG01_DISPLAY_DOMAINS) - {
        "state_and_irrigation_secondary"
    }:
        raise BuildError(
            f"fig01: the frozen plotting-region ids {sorted(r['id'] for r in regions)} do not "
            "match the recorded display domains"
        )

    # ---- section 11 (2026-08-25): y-spine, two bounds, units, no clipping ----
    def _check_axis(region_id: str, axis: dict, key: str) -> None:
        if not axis["visible_spine"]:
            raise BuildError(
                f"fig01: plotting region {region_id!r} has no visible y-spine; every region must "
                "read as a quantitative small multiple (handoff sections 5.2, 9, 11 and 13)"
            )
        bounds = axis["labeled_bounds"]
        if len(bounds) < 2:
            raise BuildError(
                f"fig01: plotting region {region_id!r} labels {len(bounds)} bound value(s); at "
                "least two are required to define the displayed lower and upper bounds"
            )
        if not axis["units"]:
            raise BuildError(
                f"fig01: plotting region {region_id!r} declares no units; handoff section 5.2 "
                "requires units adjacent to the axis or the region label"
            )
        lo, hi = axis["display_domain"]
        if lo >= hi:
            raise BuildError(f"fig01: plotting region {region_id!r} has an empty display domain")
        dmin, dmax = _region_ranges[key]
        if dmin < lo or dmax > hi:
            raise BuildError(
                f"fig01: plotting region {region_id!r} clips its data: the frozen columns span "
                f"{dmin:.4g} to {dmax:.4g} but the recorded display limits are {lo} to {hi}. "
                "Handoff section 11 stops the build rather than let a plotted value be clipped."
            )

    for region in regions:
        _check_axis(region["id"], region["y_axis"], region["id"])
        if "secondary_y_axis" in region:
            _check_axis(
                region["id"],
                region["secondary_y_axis"],
                f"{region['id']}_secondary",
            )

    # ---- section 11 (2026-08-25): the merged state/irrigation region ----
    state = next(r for r in regions if r["id"] == "state_and_irrigation")
    if not state["merged"] or "secondary_y_axis" not in state:
        raise BuildError(
            "fig01: root-zone depletion and the irrigation stems share one coordinated region "
            "and each carries an explicit scale (handoff section 5.2 item 4)"
        )
    if state["irrigation_stems"]["normalized_to_state_scale"]:
        raise BuildError(
            "fig01: handoff section 5.2 item 4 prefers a compact secondary scale or a clearly "
            "separated internal sub-band to normalizing the irrigation stems"
        )

    # ---- section 11 (2026-08-25): one merged ET region, one y mapping ----
    etr = next(r for r in regions if r["id"] == "et_comparison")
    if not (etr["merged"] and etr["shared_axis"] and etr["shared_vertical_scale"]):
        raise BuildError(
            "fig01: daily SWIM-RS ET and held-out flux ET are plotted together on ONE axis with "
            "identical date support and vertical scale; they may not be separated into "
            "independent full-height lanes (handoff sections 5.2 item 5, 5.4 and 11)"
        )
    if {lane["id"] for lane in etr["lanes"]} != set(cdm_lanes := {"daily_et", "flux_et"}):
        raise BuildError(
            f"fig01: the merged ET region must carry exactly {sorted(cdm_lanes)}, got "
            f"{sorted(lane['id'] for lane in etr['lanes'])}"
        )

    # ---- section 11 (2026-08-24): one common date mapping for the ET traces ----
    cdm = arch["common_date_mapping"]
    if tuple(cdm["columns"]) != FIG01_COMMON_DATE_MAPPING_COLUMNS:
        raise BuildError(
            "fig01: the common date mapping must cover exactly the two compared ET traces "
            f"{list(FIG01_COMMON_DATE_MAPPING_COLUMNS)}, got {cdm['columns']}"
        )
    missing_cdm = [c for c in cdm["columns"] if c not in set(example.columns)]
    if missing_cdm:
        raise BuildError(
            f"fig01: common-date-mapping column(s) {missing_cdm} are absent from the frozen "
            "example record"
        )
    if cdm["axis"] != arch["date_axis"]["id"]:
        raise BuildError(
            f"fig01: the two ET traces must share the frozen date axis {arch['date_axis']['id']!r}"
        )
    if not (cdm["full_width"] and cdm["comparable_horizontal_extent"]):
        raise BuildError(
            "fig01: the SWIM-RS and flux ET traces are presented as a direct visual comparison, "
            "so both must be drawn full width at comparable horizontal extent (handoff sections "
            "5.4 and 11)"
        )
    if not (cdm["single_region"] and cdm["shared_vertical_scale"]):
        raise BuildError(
            "fig01: the two ET traces must share one plotting region and one vertical scale; "
            "different y mappings, or separation into independent full-height lanes, would "
            "misrepresent the comparison (handoff sections 5.2 item 5, 5.4 and 11)"
        )
    if cdm["region"] != "et_comparison":
        raise BuildError(
            f"fig01: the common date mapping must name the merged ET region, got {cdm['region']!r}"
        )
    if arch["held_out"]["narrow_right_hand_column"]:
        raise BuildError(
            "fig01: the held-out treatments must not be squeezed into one narrow right-hand "
            "column; the flux trace may not become a side strip (handoff section 5.4)"
        )
    out_lanes = {
        lane["id"]: lane for row in arch["plotting_regions"] for lane in row.get("lanes", [])
    }
    for lane_id in cdm["lanes"]:
        if lane_id not in out_lanes:
            raise BuildError(f"fig01: common-date-mapping lane {lane_id!r} is not a frozen lane")
    extents = {out_lanes[lane_id]["horizontal_extent"] for lane_id in cdm["lanes"]}
    if not all("full panel-(a) plotting width" in e for e in extents):
        raise BuildError(
            f"fig01: both compared ET lanes must span the full panel width, got {sorted(extents)}"
        )

    # ---- frozen edge assertions (handoff sections 5.4, 6.3, 11) ----
    edges = [tuple(e) for e in arch["edges"]]
    if len(set(edges)) != len(edges):
        raise BuildError("fig01: the frozen edge list contains duplicates")
    for src, dst in edges:
        if src in EVALUATION_NODES and dst in FITTING_NODES:
            raise BuildError(
                f"fig01: evaluation observation {src!r} has a directed edge into {dst!r}"
            )
        if src in EVALUATION_NODES:
            raise BuildError(f"fig01: no edge may originate at held-out observation {src!r}")
    for e in arch["forbidden_edges"]:
        if tuple(e) in edges:
            raise BuildError(f"fig01: forbidden edge {e} is present in the frozen edge list")
    for dest, tokens in FIG01_TRANSFER_DESTINATIONS.items():
        for tok in tokens:
            if (tok, dest) not in edges:
                raise BuildError(f"fig01: missing transfer path {tok!r} -> {dest!r}")
        inbound = {s for s, d in edges if d == dest}
        if inbound != set(tokens):
            raise BuildError(
                f"fig01: {dest!r} must be reached only from {sorted(tokens)}, got {sorted(inbound)}"
            )
    for tok in FIG01_PARAM_TOKENS:
        inbound = {s for s, d in edges if d == tok}
        if inbound != {FIG01_TRANSFER_SOURCE}:
            raise BuildError(
                f"fig01: parameter token {tok!r} must originate only at "
                f"{FIG01_TRANSFER_SOURCE!r}, got {sorted(inbound)}"
            )
    if ("e2_map", "e3_map") in edges or ("e3_map", "e2_map") in edges:
        raise BuildError("fig01: transfer paths must both originate at E1; no E2-to-E3 edge")
    if ("rainfed_params", "e3_map") in edges:
        raise BuildError("fig01: E3 evaluates the irrigated transfer set only")

    # ---- section 11 (2026-08-25): the inverse relationships form a closed cycle ----
    cyc = arch["inverse_cycle"]
    if [s["id"] for s in cyc["stages"]] != list(FIG01_CYCLE_STAGES):
        raise BuildError(
            f"fig01: the inverse cycle must run through {list(FIG01_CYCLE_STAGES)} in order, got "
            f"{[s['id'] for s in cyc['stages']]}"
        )
    for a, b in FIG01_CYCLE_EDGES:
        if (a, b) not in edges:
            raise BuildError(
                f"fig01: the inverse-estimation relationships do not form the directed internal "
                f"cycle 'Run Balance -> Compare -> Update Parameters -> Run Balance'; the edge "
                f"{a!r} -> {b!r} is missing (handoff sections 5.3, 10.3 and 11)"
            )
    # Walk the three edges and confirm they actually close on themselves rather
    # than merely being present.
    node = FIG01_CYCLE_STAGES[0]
    for _ in FIG01_CYCLE_STAGES:
        nxt = [d for s, d in FIG01_CYCLE_EDGES if s == node]
        if len(nxt) != 1:
            raise BuildError(f"fig01: cycle stage {node!r} does not have exactly one successor")
        node = nxt[0]
    if node != FIG01_CYCLE_STAGES[0]:
        raise BuildError(
            f"fig01: the inverse cycle does not close; walking from "
            f"{FIG01_CYCLE_STAGES[0]!r} ends at {node!r}"
        )
    if tuple(cyc["exit"]["edge"]) != FIG01_CYCLE_EXIT_EDGE or FIG01_CYCLE_EXIT_EDGE not in edges:
        raise BuildError(
            f"fig01: the {FIG01_CYCLE_EXIT_LABEL!r} exit "
            f"{FIG01_CYCLE_EXIT_EDGE[0]!r} -> {FIG01_CYCLE_EXIT_EDGE[1]!r} is absent; handoff "
            "section 5.3 requires a separate exit that leaves the update stage after convergence "
            "and feeds the displayed daily state and ET"
        )
    if cyc["exit"]["label"] != FIG01_CYCLE_EXIT_LABEL:
        raise BuildError(
            f"fig01: the cycle exit must be labelled {FIG01_CYCLE_EXIT_LABEL!r}, got "
            f"{cyc['exit']['label']!r}"
        )
    if arch["edge_labels"].get("update_parameters -> daily_balance") != FIG01_CYCLE_EXIT_LABEL:
        raise BuildError(
            f"fig01: the exit edge carries no {FIG01_CYCLE_EXIT_LABEL!r} label in edge_labels"
        )
    for c in cyc["constraint_inputs"]:
        if c["into"] != FIG01_CONSTRAINT_INTO:
            raise BuildError(
                f"fig01: the calibration constraint {c['from']!r} must enter "
                f"{FIG01_CONSTRAINT_INTO!r}, got {c['into']!r} (handoff section 5.3)"
            )
    for d in cyc["driver_inputs"]:
        if d["into"] != FIG01_DRIVER_INTO:
            raise BuildError(
                f"fig01: the driver {d['from']!r} must enter {FIG01_DRIVER_INTO!r}; NDVI and "
                "daily forcing drive the forward balance and are not calibration targets "
                "(handoff section 5.3)"
            )
    if cyc["boxed"] or cyc["title"] is not None:
        raise BuildError(
            "fig01: the inverse cycle stays compact and unboxed and carries no title; the three "
            "stage labels name it (handoff sections 5.3, 9 and 13)"
        )
    # Handoff section 10.3: held-out flux and meter observations are forbidden
    # as cycle inputs.  The generic firewall above covers directed edges; this
    # states the cycle-specific requirement explicitly so a negative test can
    # reach it.
    for obs in sorted(EVALUATION_NODES):
        for stage in FIG01_CYCLE_STAGES:
            if (obs, stage) in edges:
                raise BuildError(
                    f"fig01: held-out observation {obs!r} has an edge into the calibration cycle "
                    f"stage {stage!r} (handoff sections 5.3, 10.3 and 11)"
                )
            if [obs, stage] not in arch["forbidden_edges"]:
                raise BuildError(
                    f"fig01: {obs!r} -> {stage!r} must be listed as a forbidden edge so the "
                    "held-out firewall is asserted, not merely absent"
                )

    # ---- section 11 (2026-08-24): comparisons are ties, never arrows ----
    comparisons = [(c["a"], c["b"]) for c in arch["comparisons"]]
    for c in arch["comparisons"]:
        if c["directed"]:
            raise BuildError(
                f"fig01: comparison {c['a']!r}<->{c['b']!r} is marked directed; comparison with a "
                "held-out observation must be a neutral tie, never an arrow"
            )
        if (c["a"], c["b"]) in edges or (c["b"], c["a"]) in edges:
            raise BuildError(
                f"fig01: {c['a']!r}<->{c['b']!r} is both a comparison tie and a directed edge"
            )

    # ---- section 11 (2026-08-24): the E1 example never reaches the E3 meters ----
    # Reachability over the union of directed edges and undirected comparison
    # ties, so an indirect route through any intermediate node is caught too.
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    for a, b in comparisons:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for start in FIG01_E1_APPLIED_WATER_NODES:
        seen_nodes = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt in seen_nodes:
                    continue
                if nxt == FIG01_E3_METER_NODE:
                    raise BuildError(
                        f"fig01: the E1 example's applied-water node {start!r} reaches the E3 "
                        f"meter node {FIG01_E3_METER_NODE!r} through {node!r}. E3 evaluates "
                        "applied water simulated for E3 fields, not this record; no drawn "
                        "relationship may connect them (handoff sections 5.2, 5.4 and 11)."
                    )
                seen_nodes.add(nxt)
                stack.append(nxt)
    aw = next(o for o in arch["outputs"] if o["id"] == "applied_water")
    if aw["compared_with"] is not None or aw["aggregation"] is not None:
        raise BuildError(
            "fig01: the E1 example's applied-water series carries no drawn comparison and no "
            "drawn aggregation"
        )
    for row in arch["plotting_regions"]:
        if row.get("aggregation_mark") is not None:
            raise BuildError(
                f"fig01: plotting region {row['id']!r} carries an aggregation mark; annual "
                "aggregation belongs only to the separate E3 evaluation key (handoff section 9)"
            )
        if row["id"] == "state_and_irrigation" and row.get("irrigation_events_repeated_here"):
            raise BuildError(
                "fig01: irrigation events are encoded once, as the applied-water stems; a second "
                "event lane must not be restored (handoff section 5.2)"
            )

    # ---- section 11 (2026-08-24): no data-like meter glyph without observations ----
    key = arch["e3_evaluation_key"]
    if key["frozen_observation_source"] is None:
        if key["data_like_glyph"]:
            raise BuildError(
                "fig01: no E3 observations are frozen in the Figure 1 package, so the E3 "
                "evaluation key must stay typographic and schematic; a data-like meter glyph "
                "would present a symbol as evidence (handoff sections 5.4 and 11)"
            )
        if key["treatment_class"] != "typographic_schematic":
            raise BuildError(
                "fig01: absent frozen E3 observations the evaluation key's treatment_class must "
                f"be 'typographic_schematic', got {key['treatment_class']!r}"
            )
        for node in key["nodes"]:
            if node["is_actual_data"]:
                raise BuildError(
                    f"fig01: E3 key node {node['id']!r} claims to be actual data, but no E3 "
                    "observations are frozen in this package"
                )
    else:
        # Actual E3 marks are permitted only with their own frozen values AND a
        # separately documented selection record (handoff sections 5.4 and 11).
        for field in ("frozen_observation_source", "frozen_observation_selection_record"):
            name = key.get(field)
            if not name or not (paths.OUT / str(name)).exists():
                raise BuildError(
                    "fig01: actual E3 meter marks require both frozen observation values and a "
                    f"separately documented selection record; {field} is {name!r}"
                )
    meters_node = next(o for o in arch["held_out"]["observations"] if o["id"] == "meters")
    if meters_node["data_like_glyph"] or meters_node["is_actual_data"]:
        raise BuildError(
            "fig01: the 'Metered Water · E3' term must not be drawn as data while no E3 "
            "observations are frozen"
        )
    if meters_node["column"] is not None:
        raise BuildError("fig01: the E3 meter term has no source column and must not claim one")

    # ---- section 11 (2026-08-24): the E3 path branches at the irrigated token ----
    branch = arch["transfer_branch"]
    if branch["at"] != FIG01_TRANSFER_BRANCH_AT:
        raise BuildError(
            f"fig01: the transfer branch must sit at {FIG01_TRANSFER_BRANCH_AT!r}, got "
            f"{branch['at']!r}; a branch drawn at the E2 map reads as E2 -> E3"
        )
    if not branch["visible"]:
        raise BuildError(
            "fig01: the irrigated branch junction must be visible so both outgoing paths are "
            "seen to originate at the token (handoff sections 6.1, 6.3 and 13)"
        )
    if set(branch["outgoing"]) != set(FIG01_TRANSFER_DESTINATIONS):
        raise BuildError(
            f"fig01: the irrigated branch must serve {sorted(FIG01_TRANSFER_DESTINATIONS)}, got "
            f"{sorted(branch['outgoing'])}"
        )
    for dest in branch["outgoing"]:
        if (branch["at"], dest) not in edges:
            raise BuildError(
                f"fig01: the branch at {branch['at']!r} declares an outgoing path to {dest!r} "
                "that the frozen edge list does not carry"
            )
    if not any("E2" in r for r in branch["forbidden_routes"]):
        raise BuildError(
            "fig01: the branch must explicitly forbid the E3 route that runs along or beneath "
            "the E2 frame"
        )

    # ---- section 11 (2026-08-24): E0 placement, SWE inline, engine label ----
    if arch["development_tag"]["placement"] != "adjacent_to_e2_heading":
        raise BuildError(
            "fig01: the E0 tag belongs beside the E2 heading (E0 ran at 37 E2 sites not in "
            "E1), not below the map as a detached footnote (handoff sections 6.4 and 7)"
        )
    if arch["development_tag"]["attached_to"] != "e2_map":
        raise BuildError("fig01: the E0 tag attaches to the E2 map")
    if arch["swe_constraint"]["drawn_as_separate_node"] or arch["swe_constraint"]["drawn_as_edge"]:
        raise BuildError(
            "fig01: SWE is an inline label on the constraint route into 'Compare', not a "
            "separate chip with its own arrow (handoff section 5.3)"
        )
    if arch["swe_constraint"]["enters_stage"] != FIG01_CONSTRAINT_INTO:
        raise BuildError(
            f"fig01: the '+ SWE' constraint enters {FIG01_CONSTRAINT_INTO!r}, got "
            f"{arch['swe_constraint']['enters_stage']!r} (handoff section 5.3)"
        )
    if arch["inverse_cycle"]["engine_label_ownership"] != "caption":
        raise BuildError(
            "fig01: 'PEST++ IES' moves to the caption by default; restoring it to the artwork "
            "requires a final-size proof showing it adds specificity without enlarging the "
            "inverse element (handoff section 5.3)"
        )
    if arch["inverse_cycle"]["spread_label_condition"]["string"] in cls:
        raise BuildError(
            "fig01: 'spread-weighted' is caption-owned; handoff section 5.3 permits its return "
            "on the incoming target route only if a final-size proof shows it materially "
            "clarifies the member-spread encoding"
        )
    # Every classified string must belong to a node the edge list or the frozen
    # display package actually knows about; nothing may be drawn from nowhere.
    if (
        arch["example_record"]["n_days"] != n_ex
        or arch["example_record"]["n_calibration_captures"] != n_ex_caps
    ):
        raise BuildError(
            "fig01: the architecture's example-record shape disagrees with the frozen "
            "fig01_example_timeseries.csv"
        )

    (paths.OUT / "fig01_architecture.json").write_text(
        json.dumps(arch, indent=2, ensure_ascii=False)
    )

    meta = {
        "figure": "fig01",
        "architecture_schema_version": arch["schema_version"],
        "canvas_mm": [190, 120],
        "canvas_note": (
            "Target for the revised Gate A proof per fig01_production_handoff.md section 4 "
            "(rewritten 2026-08-20, unchanged 2026-08-24): 190 mm x 120 mm with 3 mm outer "
            "margins (usable 184 x 114 mm), two labelled horizontal panels separated by a 3-4 mm "
            "gutter, no overall title inside the artwork. The rejected 145 mm composition and "
            "the superseded 110 mm map-plus-framework composition must not be restored. "
            "six_figure_plan.md section 3.1 was reconciled to 120 mm on 2026-08-20. The proof "
            "rendered against architecture 3.0.0 (fig01_evidence_190.png) is review provenance "
            "and must not be revised into the next proof."
        ),
        "crs": {
            "published_layers": "EPSG:4326",
            "e1_container_native": container_coord_crs(E1_CONTAINER),
            "e2_container_native": container_coord_crs(E2_CONTAINER),
            "e3_container_native": container_coord_crs(E3_CONTAINER),
            "e3_snapping_crs": "EPSG:5070",
            "note": (
                "Container geometry/lon,lat are stored in the native CRS of each container's "
                "source fields shapefile and are reprojected to EPSG:4326 here; every published "
                "layer is checked for coordinate/CRS-tag consistency before writing."
            ),
        },
        "boundary_dataset": {
            "name": "Natural Earth 110m admin 0 countries",
            "version": _ne_version(),
            "path": str(NE_COUNTRIES),
            "sha256": sha256(NE_COUNTRIES),
            "license": "public domain (Natural Earth)",
            "used_for": ["world_context", "conus_context", "E2 country and continent assignment"],
            "disclaimer": (
                "Map lines delineate study areas and do not necessarily depict accepted "
                "national boundaries."
            ),
        },
        "context_datasets": {
            "_contract": (
                "handoff sections 6.4, 6.6 and 6.7 (revised 2026-08-24) direct the next Gate A "
                "proof to TEST faint CONUS state boundaries and a subtle, privacy-safe E3 basin "
                "context. Both sources below are archived in-repo, public domain, and hashed. "
                "Retention is a proof-level decision, so both layers are published as optional "
                "context and neither is required by the composition."
            ),
            "conus_states": {
                "layer": "conus_states_context",
                "name": "Natural Earth 50m admin 1 states provinces lakes",
                "version": _ne_states_version(),
                "path": str(NE_STATES),
                "sha256": sha256(NE_STATES),
                "archive_zip": str(NE_STATES.with_suffix(".zip")),
                "archive_zip_sha256": sha256(NE_STATES.with_suffix(".zip")),
                "url": (
                    "https://naciscdn.org/naturalearth/50m/cultural/"
                    "ne_50m_admin_1_states_provinces_lakes.zip"
                ),
                "license": "public domain (Natural Earth)",
                "archived": "2026-08-24",
                "selection": (
                    "iso_a2 == 'US' excluding AK and HI, clipped to the same "
                    "-125..-66.5 E, 24..49.5 N box conus_context uses; 49 features "
                    "(48 contiguous states plus the District of Columbia)"
                ),
                "note": (
                    "the same Natural Earth release (5.1.1) as the admin-0 source already used "
                    "for world_context, so the two boundary hierarchies are consistent"
                ),
            },
            "e3_basin": {
                "layer": "slv_basin_context",
                "name": "USGS Watershed Boundary Dataset, WBDHU8",
                "path": str(SLV_BASIN),
                "sha256": sha256(SLV_BASIN),
                "provenance_record": str(SLV_BASIN_SOURCE),
                "provenance_record_sha256": sha256(SLV_BASIN_SOURCE),
                "source_publication_date": "2025-01-07",
                "url": (
                    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/"
                    "GDB/WBD_National_GDB.zip"
                ),
                "license": (
                    "public domain (U.S. Geological Survey; 17 U.S.C. 105 - works of the U.S. "
                    "Government are not subject to copyright)"
                ),
                "archived": "2026-08-24",
                "selection": (
                    "the five HUC8 units of the Rio Grande headwaters accounting unit "
                    "(HUC6 130100): 13010001 Rio Grande Headwaters, 13010002 "
                    "Alamosa-Trinchera, 13010003 San Luis, 13010004 Saguache, 13010005 Conejos"
                ),
                "privacy_review": (
                    "HUC8 units span 1,987-6,576 km2. The builder asserts that no published unit "
                    "is smaller than 1000 km2 and that every generalized E3 display point falls "
                    "inside the layer, so the context cannot localize a field more precisely "
                    "than the approved 1 km EPSG:5070 centroid snap already does. No field "
                    "polygon, parcel, acreage, owner, or source-agency attribute is carried."
                ),
            },
        },
        "e3_generalization": {
            "method": "field centroid, reprojected to EPSG:5070, snapped to a 1000 m grid, reprojected to EPSG:4326",
            "grid_m": 1000,
            "deterministic": True,
            "seed": None,
            "approved": {
                "date": "2026-08-19",
                "outcome": "approved",
                "authority": "paper/notes/fig01_production_handoff.md section 6.4",
                "statement": (
                    "The approved generalized display is field centroids snapped to a 1 km grid, "
                    "frozen in the e3_display layer of fig01_scope.gpkg with the method recorded "
                    "here. This satisfies the section 15 item 2 approval requirement."
                ),
            },
            "removed_attributes": [
                "src_id (source-agency field identifier)",
                "source",
                "exact field polygon",
                "acres",
                "basin",
                "state",
                "original site_id",
            ],
            "retained_attributes": [
                "display_id",
                "geometry_method",
                "source_group",
                "source_key",
                "crop_group",
            ],
            "public_audit_key": {
                "attribute": "source_key",
                "construction": "sha256('swim-rs/fig01/e3-display/v1|' + src_id), first 16 hex characters",
                "property": (
                    "deterministic and one-to-one with the display records, but carries no "
                    "source-agency meter identifier; the source-to-display linkage is retained "
                    "only in restricted metadata outside this public package"
                ),
            },
        },
        "layer_attributes": {
            "e1_sites": list(e1_gdf.columns),
            "e2_sites": list(e2_gdf.columns),
            "e3_display": list(e3_disp.columns),
            "conus_context": list(conus_ctx.columns),
            "conus_states_context": list(states_ctx.columns),
            "world_context": list(world_ctx.columns),
            "slv_context": list(slv_ctx.columns),
            "slv_basin_context": list(basin_ctx.columns),
        },
        "cohort_assertions": {
            "e1_configured": int(len(e1)),
            "e2_configured": int(len(e2)),
            "e3_display": int(len(e3_disp)),
            "e1_e2_overlap": int(e1["in_e2"].sum()),
            "mb_pch_present_in_e1_scope": bool("MB_Pch" in set(e1["site_id"])),
            "e1_daily_evaluation": int(e1["in_daily_evaluation"].sum()),
            "e2_countries": n_countries,
            "e2_continents": n_continents,
            "e2_evaluation_pool": int(e2["in_evaluation_pool"].sum()),
            "e2_pool_countries": n_pool_countries,
            "e2_pool_continents": n_pool_continents,
            "e1_e2_overlap_pool": int((e2["in_evaluation_pool"] & e2["in_e1"]).sum()),
            "e2_continent_names": continents,
            "e2_countries_names": countries,
            "e3_display_geometry_types": sorted(set(e3_disp.geom_type)),
            "evidence_matrix_rows": int(n_ev),
            "e0_evidence_role": roles.get("E0"),
            "visible_string_count": arch["visible_string_count"],
            "explanatory_annotation_count": n_annot,
            "example_rows": int(n_ex),
            "example_calibration_captures": int(n_ex_caps),
            "conus_states_context_rows": int(len(states_ctx)),
            "slv_basin_context_rows": int(len(basin_ctx)),
            "record_identification_visible": arch["example_record"]["record_label"],
        },
        "class_assertions": {
            "attribute": "irrigation_class",
            "definition": (
                "the frozen source-class assignment used to construct the E1-derived transfer "
                "parameter sets, not annual irrigation activation (handoff section 6.3)"
            ),
            "e1_irrigated": class_counts["e1_irrigated"],
            "e1_rainfed": class_counts["e1_rainfed"],
            "e2_irrigated": class_counts["e2_irrigated"],
            "e2_rainfed": class_counts["e2_rainfed"],
            "e3_all_fields_class": "irrigated",
            "e3_class_reason": (
                "E3 evaluates only the irrigated transfer set, so every E3 field is drawn with "
                "the irrigated symbol"
            ),
            "symbol_encoding": {"irrigated": "triangle", "rainfed": "circle"},
            "layers": {"e1": "fig01_scope.gpkg::e1_sites", "e2": "fig01_scope.gpkg::e2_sites"},
        },
        "example_selection": {
            "record": "fig01_example_selection.json",
            "display_file": "fig01_example_timeseries.csv",
            "site_id": FIG01_EXAMPLE_SITE,
            "full_window": [FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
            "displayed_window": [FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
            "display_crop": None,
            "n_days": int(n_ex),
            "n_calibration_captures": int(n_ex_caps),
            "inherited_from": "fig03_example_selection.json",
            "selection_independence": example_selection["selection_independence"],
            "reconciliation": example_selection["reconciliation"],
            "refused_columns": sorted(example_selection["excluded_from_this_file"]),
            "column_provenance_recorded_for": sorted(example_prov),
        },
        "legacy_to_current_map": EXPERIMENT_MAP,
        "legacy_to_current_label_map": {
            "experiments": {
                "legacy e2_* / examples/5_Flux_Ensemble": "E1",
                "legacy e3_* / examples/6_Flux_International": "E2",
                "legacy e4_* / examples/7_Applied_Water": "E3",
                "legacy e1_* (removed broad-land-cover experiment)": None,
            },
            "reader_facing_strings": {
                "Landsat + Sentinel-2 NDVI": "NDVI · Vegetation State",
                "satellite ETf at retained captures": "ETf + SWE · Calibration Targets",
                "gridded SWE": "ETf + SWE · Calibration Targets",
                "meteorology · soils · land cover · irrigation status": "Gridded Forcing · Model Drivers",
                "surface evaporation layer / active root zone / below-root reservoir": "Daily Water Balance",
                "daily ET / ETf": "Daily ET",
                "gross applied water": "Gross Applied Water",
                "flux ET — E1 and E2": "Flux ET · E1–E2",
                "metered applied water — E3": "Metered Water · E3",
                "PEST++ IES calibration": "Inverse Estimation (subtitle PEST++ IES)",
                "withheld from parameter estimation and transfer-vector construction": "Held-Out Evaluation (full qualification moved to the caption)",
                "E0 model development": "E0 · Formulation Selection",
                "panel (c) evidence cards": "removed; replaced by the unlettered transfer ribbon",
            },
            "reader_facing_strings_note": (
                "The two maps above describe the SUPERSEDED architecture 2.1.0 copy. They are "
                "retained as provenance so an earlier proof can be traced; none of those strings "
                "is visible in 3.0.0."
            ),
            "reader_facing_strings_2026_08_20": {
                "Configured Geography": "Study Domains",
                "E1 · 60 CONUS Cropland Sites": "E1 · CONUS / 60 Cropland Sites",
                "E2 · 66 International Cropland Sites": "E2 · International / 66 Cropland Sites",
                "E3 · 50 San Luis Valley Fields": "E3 · San Luis Valley / 50 Metered Fields",
                "E2 · Geographic/Input Transfer": "E2 · Geographic and Input Transfer",
                "daily state propagation": None,
            },
            "reader_facing_strings_architecture_3_0_0": {
                "Study Domains": "E1 Source Cohort and Parallel Transfer",
                "SWIM-RS Framework": "Sparse Satellite Constraints to Daily State",
                "Satellite and Gridded Inputs": None,
                "NDVI · Vegetation State": "NDVI Captures",
                "ETf + SWE · Calibration Targets": "ETf Ensemble (+ the separate SWE constraint)",
                "Gridded Forcing · Model Drivers": "Daily Forcing",
                "Daily Water Balance": "Daily State (an actual root-zone depletion trace)",
                "E2 · International": "E2 · 10 Countries",
                "E1 · Source Parameters": "the E1 · CONUS map itself",
                "E2 · Geographic and Input Transfer": "the E2 · 10 Countries map itself",
                "E3 · Applied-Water Transfer": "the E3 · San Luis Valley map itself",
                "E0 · Formulation Selection": "E0 · Model-Form Selection",
                "the unlettered transfer ribbon": (
                    "removed; replaced by 'Irrigated Parameters' and 'Rainfed Parameters' tokens "
                    "drawn between the maps"
                ),
            },
            "reader_facing_strings_architecture_3_1_0": {
                "PEST++ IES": "moved to the caption; the inverse node carries no engine subtitle",
                "SWE": "+ SWE (an inline label on the ETf-target relay, not a separate chip)",
                "annual sum": "Σ year",
                "Gross Applied Water": (
                    "Irrigation (the E1 example lane); 'Daily Gross Applied Water' is now "
                    "reserved for the separate E3 evaluation key"
                ),
                "irrigation": (
                    "removed; the duplicate triangle event lane in the daily-state row is gone "
                    "and the magnitude-bearing applied-water stems are the only irrigation "
                    "encoding"
                ),
                "2017": "US-Bi1 · 2017 (required visible record identification)",
            },
            "reader_facing_strings_architecture_3_1_0_note": (
                "Strings added by the 2026-08-24 revision, with no 3.0.0 predecessor: "
                "'US-Bi1 · 2017', '+ SWE', 'daily drivers', 'Irrigation', 'Daily Gross Applied "
                "Water', 'Σ year' and 'Daily Gross Applied Water → Σ year ⇄ Metered Water · E3'. "
                "Two of them were retired again by 3.1.1; see the next map."
            ),
            "reader_facing_strings_architecture_3_1_1": {
                "Flux ET · E1–E2": (
                    "Flux ET; the drawn trace is one US-Bi1 / E1 record and the E1-E2 "
                    "flux-evaluation scope is caption-owned"
                ),
                "Σ year": (
                    "Annual Total; the E3 key reads 'Daily Gross Applied Water → Annual Total "
                    "⊢—⊣ Metered Water · E3'"
                ),
            },
            "reader_facing_strings_architecture_3_1_1_note": (
                "The 2026-08-25 user review of r2 changed two strings and one string class. No "
                "string was added without a predecessor. 'Held-Out Evaluation' was unchanged as "
                "copy but moved from title to direct_label; 3.2.0 then retired it outright."
            ),
            "reader_facing_strings_architecture_3_2_0": {
                "US-Bi1 · 2017": (
                    "US-Bi1 (2017); near-black identification, no middle dot, no muted-gray "
                    "subtitle (handoff section 5.1)"
                ),
                "Flux ET": (
                    "Flux ET (Held Out); the held-out qualification travels with the trace in "
                    "the merged ET region (sections 5.2 item 5 and 5.4)"
                ),
                "Held-Out Evaluation": (
                    "removed; section 5.4 asks for the direct label 'rather than a separate "
                    "Held-Out Evaluation heading or rule competing for space'"
                ),
                "Inverse Estimation": (
                    "removed; the closed cycle is named by 'Run Balance', 'Compare' and 'Update "
                    "Parameters' (sections 4, 5.3, 9 and 13)"
                ),
                "spread-weighted": (
                    "removed; caption-owned by section 5.3, with a conditional return on the "
                    "incoming target route recorded in the architecture"
                ),
                "Daily State": "State + Irrigation (the merged region, section 4 wireframe)",
                "Daily Outputs": "Daily ET (the merged ET region, section 4 wireframe)",
                "Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3": (
                    "Daily Gross Applied Water → Annual Total — Metered Water · E3; sections 4 "
                    "and 5.4 write the relation with a plain em dash instead of the improvised "
                    "'⊢—⊣' glyph"
                ),
            },
            "reader_facing_strings_architecture_3_2_0_note": (
                "Strings added by the 2026-08-25 handoff rewrite with no 3.1.1 predecessor: "
                "'Run Balance', 'Compare', 'Update Parameters' and 'Conditioned Parameters', the "
                "four labels of the closed inverse cycle. 'Daily ET' is not new -- it was "
                "already the model-trace direct label and now doubles as the merged region's "
                "facet name."
            ),
            "metrics": {"legacy r2_*": "nse", "legacy bias_*": "mbe"},
        },
        "sources": {
            "e1_container": str(E1_CONTAINER),
            "e2_container": str(E2_CONTAINER),
            "e3_container": str(E3_CONTAINER),
            "e2_cohort_mapping": str(
                FINAL / "e3_irrigation_stratified_param_mapping_metadata.json"
            ),
            "e1_daily_cohort": str(FINAL / "e2_primary_daily_site_metrics.csv"),
            "boundary_dataset": str(NE_COUNTRIES),
            "example_series": {
                "path": "paper/data/final/figures/fig03_example_timeseries.csv",
                "sha256": sha256(paths.OUT / "fig03_example_timeseries.csv"),
            },
            "example_member_values": {
                "path": "paper/data/final/figures/fig04_spread_capture_values.csv",
                "sha256": sha256(paths.OUT / "fig04_spread_capture_values.csv"),
            },
            "example_upstream_selection": {
                "path": "paper/data/final/figures/fig03_example_selection.json",
                "sha256": sha256(paths.OUT / "fig03_example_selection.json"),
            },
            "example_source_note": (
                "The Figure 1 example is built by joining two already-frozen, already-audited "
                "sibling artifacts in this package, not by re-deriving anything from a "
                "container. fig01 is therefore built AFTER fig03 and fig04 in a --all run."
            ),
        },
        "package_files": [
            "fig01_scope.gpkg",
            "fig01_evidence_matrix.csv",
            "fig01_example_timeseries.csv",
            "fig01_example_selection.json",
            "fig01_architecture.json",
            "fig01_metadata.json",
        ],
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
        "fig01_builder_version": FIG01_BUILDER_VERSION,
        "data_contract": (
            "paper/notes/fig01_production_handoff.md section 10, rewritten 2026-08-25. The "
            "panel (a) redesign is recorded in fig01_architecture.json under "
            "panel_a_redesign_2026_08_25; the earlier r2 user-review directives remain under "
            "review_directives_2026_08_25"
        ),
        "revision_level": (
            "Level 2 contract change (handoff sections 15.1 and 16). The frozen example, "
            "geography and evidence values are unchanged; the architecture, contract-bearing "
            "metadata and manifest entries are the expected rewrites."
        ),
        "review": {
            "handoff_rewrite_2026_08_25": {
                "date": "2026-08-25",
                "subject": "revision 3 of the proof",
                "status_line": (
                    "Revision 3 reviewed; panel (a) redesign and a new Gate A proof are required"
                ),
                "outcome": (
                    "panel (a) is redesigned into five quantitative plotting regions with a "
                    "closed inverse cycle; panel (b) is accepted as built in revision 3 "
                    "(handoff section 6.1). Architecture 3.2.0 freezes the new copy, grouping, "
                    "y-axis requirements and cycle relationships."
                ),
                "frozen_contract_changes": [
                    "five plotting regions replace the seven-lane structure",
                    "root-zone depletion and the irrigation stems merge into one coordinated "
                    "region with an explicit scale for each; the stems are not normalized",
                    "daily SWIM-RS ET and held-out flux ET merge onto one axis with identical "
                    "date support and vertical scale",
                    "inverse estimation becomes the closed cycle 'Run Balance -> Compare -> "
                    "Update Parameters -> Run Balance' with ETf and '+ SWE' entering Compare and "
                    "a 'Conditioned Parameters' exit from the update stage",
                    "'US-Bi1 · 2017' -> 'US-Bi1 (2017)' in near-black; 'Flux ET' -> 'Flux ET "
                    "(Held Out)'; 'Held-Out Evaluation', 'Inverse Estimation' and "
                    "'spread-weighted' retired",
                    "every region carries a visible y-spine, two labelled bounds, units, and "
                    "recorded display limits that clip no plotted value",
                    "near-black #202124 for all reader-facing identification; muted gray "
                    "confined to axes, rules and quiet geographic context",
                    "conus_states_context accepted and drawn; slv_basin_context archived but not "
                    "drawn (section 6.7)",
                ],
                "open": (
                    "a faint privacy-safe E3 hillshade remains optional and does not block "
                    "Gate B; it needs an archived, licensed, hashed DEM derivation first"
                ),
            },
            "user_review_2026_08_25": {
                "date": "2026-08-25",
                "subject": "the r2 proof",
                "outcome": (
                    "strong improvement; one revision before Gate B. Architecture 3.1.1 freezes "
                    "the subset of the directives that changes reader-facing copy, string class, "
                    "comparison treatment, or frozen geometry guidance."
                ),
                "frozen_contract_changes": [
                    "'Flux ET · E1–E2' -> 'Flux ET' (the drawn trace is one US-Bi1 / E1 record; "
                    "the E1-E2 flux-evaluation scope is caption-owned)",
                    "'Σ year' -> 'Annual Total' (a subscripted sigma would put 'year' below the "
                    "7.5 pt floor); the sigma exemption is removed from the caption guard",
                    "'Held-Out Evaluation' reclassified title -> direct_label, drawn as a "
                    "compact label on the held-out rule",
                    "the daily ET / flux ET comparison is shared alignment only, with no drawn "
                    "bracket or tie glyph",
                    "the irrigated triangle sits at the fork with its label to the left, and the "
                    "E3 leg leaves as a distinct diagonal or shallow curve",
                    "'+ SWE' sits on the incoming constraint route, not beside the inverse title",
                ],
                "open": (
                    "a faint privacy-safe E3 hillshade is user-optional and not required for "
                    "Gate B; it needs an archived, licensed DEM derivation first"
                ),
                "partly_superseded_by": (
                    "the 2026-08-25 handoff rewrite. Architecture 3.2.0 keeps the 'Annual Total' "
                    "operation, the shared-alignment ET comparison, the branch topology and the "
                    "'+ SWE' route, but replaces 'Flux ET' with 'Flux ET (Held Out)' and retires "
                    "'Held-Out Evaluation' entirely rather than reclassifying it."
                ),
            },
            "scientific_review": {
                "date": "2026-08-25",
                "outcome": (
                    "data contract revised for architecture 3.1.0, refined by 3.1.1, regrouped "
                    "without value change by 3.2.0"
                ),
                "detail": (
                    "All 2026-08-20 assertions are retained: E0 typed as model_development; "
                    "configured 60/66/50, the 13-site E1/E2 overlap, MB_Pch presence, and the "
                    "frozen source classes (E1 39 irrigated / 21 rainfed, E2 13 irrigated / 53 "
                    "rainfed); spread-weighted primaries and per-experiment ETf target "
                    "composition; the US-Bi1 example frozen from the audited fig03 and fig04 "
                    "artifacts with member marks reconciled to the frozen member count and "
                    "target mean; both transfer paths originating at E1 with no E2-to-E3 edge "
                    "and no evaluation-to-fitting edge. Added 2026-08-24: the false E1-to-E3 "
                    "linkage is removed and refused by a reachability check over edges and "
                    "comparison ties; comparison with a held-out observation must be a neutral "
                    "tie rather than a directed edge; the two ET traces must share one date "
                    "mapping at full width; the E3 evaluation key must stay typographic while no "
                    "E3 observations are frozen; the E3 transfer path must branch visibly at the "
                    "irrigated parameter token; and 'US-Bi1 · 2017' must appear as visible copy."
                ),
                "revised_2026_08_25": (
                    "Architecture 3.2.0 changes how these assertions are GROUPED and LABELLED, "
                    "not what they assert. The scientific content is unchanged: the same frozen "
                    "columns, the same 15 calibration captures, the same member count and target "
                    "mean, the same source classes, the same transfer topology, and the same "
                    "held-out firewall -- now extended so that no held-out observation may enter "
                    "any of the three inverse cycle stages. Two label changes carry scientific "
                    "weight: the drawn flux trace is qualified in place as 'Flux ET (Held Out)', "
                    "and 'spread-weighted' moves from artwork to caption, where the weighting "
                    "scheme can be stated precisely instead of compressed into two words."
                ),
                "revised_2026_08_27": (
                    "Architecture 3.3.0 (user-directed example swap): the example record moves "
                    f"from US-Bi1 (2017) to {FIG01_RECORD_ID} under the recorded steps-3-4 "
                    "override in fig03_example_selection.json. Assertions stated above for the "
                    "then-selected record re-bind to the current record: the frozen columns are "
                    f"unchanged, the calibration-capture count is now {FIG01_EXAMPLE_CAPTURES} "
                    "(was 15), member marks are reconciled to the frozen member count and target "
                    "mean of the new record, and the visible-copy identification requirement now "
                    f"reads '{FIG01_RECORD_ID}'. Source classes, transfer topology, and the "
                    "held-out firewall are untouched."
                ),
                "supersedes": "2026-08-20 review of architecture 3.0.0",
            },
            "privacy_review": {
                "date": "2026-08-25",
                "outcome": (
                    "approved; the basin context layer is archived but no longer drawn, and the "
                    "optional hillshade stays unbuilt"
                ),
                "detail": (
                    "E3 display remains 1 km-snapped centroids; no exact polygons, no "
                    "source-agency identifiers, public source_key is a salted truncated hash. "
                    "The example record adds no restricted content: it is one public AmeriFlux "
                    "site. New for 3.1.0: slv_basin_context publishes five HUC8 watershed units "
                    "of 1,987-6,576 km2 from the public-domain USGS WBD. The builder refuses any "
                    "unit smaller than 1000 km2 and requires every generalized display point to "
                    "fall inside the layer, so the context adds orientation without adding "
                    "location precision. conus_states_context carries only state names and "
                    "postal codes. Open item carried forward: the E3 audit-key salt is published "
                    "in this public metadata file; the mechanism is left as-is."
                ),
                "revised_2026_08_25": (
                    "Handoff section 6.7 retains the CONUS state boundaries as drawn context, "
                    "keeps HUC8 watershed units out of the artwork, and leaves a faint "
                    "privacy-safe hillshade optional. slv_basin_context therefore stays in "
                    "fig01_scope.gpkg as archived, hashed provenance -- removing it would break "
                    "the Level 2 value-equality requirement -- but e3_map.optional_context_layer "
                    "is set to null so no proof draws it. Nothing is published that was not "
                    "already published under 3.1.0, and one layer moves from candidate artwork to "
                    "provenance only, which is a strict reduction in what a reader can see."
                ),
            },
            "visual_review": {
                "date": "2026-08-25",
                "outcome": (
                    "architecture 3.0.0 and its proof reviewed and superseded; the r2 proof "
                    "reviewed as a strong improvement and superseded by 3.1.1; revision 3 "
                    "reviewed and panel (a) sent back for redesign, panel (b) accepted as built; "
                    "3.2.0 frozen, no proof yet exists for it"
                ),
                "detail": (
                    "The evidence-first thesis survived review; the first proof "
                    "(fig01_evidence_190.png) did not. It compressed the actual flux record into "
                    "a side strip beside the full-width model trace, paired that actual trace "
                    "with a symbolic three-circle meter glyph inside one apparently equivalent "
                    "region, drew an 'annual sum' bracket across the E1 example's seasonal "
                    "applied-water record that asserted an E1-to-E3 data linkage, repeated the "
                    "irrigation events in a second triangle lane, boxed the inverse node and hung "
                    "a 'PEST++ IES' subtitle on it, ran a faint driver lane that read as an axis "
                    "spine, routed the E3 transfer path along the bottom of the E2 frame, and "
                    "left the record unidentified. Architecture 3.1.0 corrects each of those, "
                    "and the r2 proof rendered against it was accepted as a strong improvement. "
                    "The 2026-08-25 user review of r2 returned four must-address items -- the "
                    "'Flux ET · E1–E2' mislabel, the improvised 'Σ year' operator, 'Held-Out "
                    "Evaluation' as a left-column heading opening a large empty region, and a "
                    "right-margin comparison bracket that read as an empty box at thumbnail "
                    "scale -- plus refinements to the E3 branch geometry and the '+ SWE' "
                    "placement. Architecture 3.1.1 freezes those. Revision 3, rendered against "
                    "3.1.1, was reviewed on 2026-08-25: panel (b) was ACCEPTED as built, and "
                    "panel (a) was sent back for redesign. The r3 panel (a) still read as seven "
                    "thin sparklines without y-axes or units, so no reader could recover a "
                    "magnitude; the state and irrigation lanes were separated so event-to-state "
                    "response was invisible; the two ET traces sat in different lanes rather than "
                    "on one shared axis; the inverse element read as two disconnected L-shaped "
                    "legs under a heading instead of a closed iteration; and the record "
                    "identification was set as a muted gray middle-dot subtitle that read as "
                    "decoration. Architecture 3.2.0 freezes the five-region regrouping, the "
                    "y-axis contract, the closed cycle with its Conditioned Parameters exit, and "
                    "the near-black identification policy. A fresh Gate A revision must be "
                    "written to a NEW proof directory; no earlier proof image may be revised "
                    "into it."
                ),
                "gate_a_status": "not yet produced against architecture 3.2.0",
                "gate_b_status": "not started",
                "panel_b_status": (
                    "accepted as built in revision 3 (handoff section 6.1); subsequent panel (b) "
                    "changes are render-only unless a path source or destination changes"
                ),
                "context_layer_decision": {
                    "conus_states_context": "accepted, drawn as quiet muted-gray context",
                    "slv_basin_context": "archived and hashed, not drawn (handoff section 6.7)",
                    "e3_hillshade": (
                        "optional, not built; needs an archived, licensed, hashed DEM derivation "
                        "first and does not block Gate B"
                    ),
                },
            },
        },
        "output_dimensions": {
            "canvas_mm": [190, 120],
            "outer_margin_mm": 3,
            "usable_mm": [184, 114],
            "panel_gutter_mm": [3, 4],
            "rendered": None,
        },
        "font_inventory": {
            "planned_family": "Source Sans 3, embedded, with editable vector text",
            "minimum_reader_facing_pt": 7.5,
            "normal_body_pt": 8,
            "panel_label_pt": [10, 11],
            "panel_heading_pt": [8.5, 9],
            "structural_label_pt": [8, 8.5],
            "direct_label_and_axis_pt": [7.5, 8],
            "math_face": "STIX Two Math or another embedded math face, only if notation is drawn",
            "measured": None,
            "note": "render-level font auditing belongs to the figure builder, not this data builder",
        },
    }
    (paths.OUT / "fig01_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    MANIFEST.add(
        "fig01_scope.gpkg",
        rows={
            "e1_sites": len(e1_gdf),
            "e2_sites": len(e2_gdf),
            "e3_display": len(e3_disp),
            "conus_context": len(conus_ctx),
            "conus_states_context": len(states_ctx),
            "world_context": len(world_ctx),
            "slv_context": len(slv_ctx),
            "slv_basin_context": len(basin_ctx),
        },
        sources=meta["sources"],
        experiment_mapping=EXPERIMENT_MAP,
        cohort_key="display_id",
        inclusion_rule="Configured scope: all 60 E1 sites (MB_Pch retained), all 66 E2 sites, all 50 E3 SLV fields.",
        layer_attributes=meta["layer_attributes"],
        display_transformations=[
            "E1/E2 sites shown as container centroids, reprojected to EPSG:4326 from the "
            "native CRS of each container's source fields shapefile (E1 EPSG:5071, E2 EPSG:4326)",
            "E3 shown as 1 km snapped generalized centroids (approved 2026-08-19); no exact "
            "polygons or source-agency identifiers; source_key is a salted truncated SHA256",
            "E2 country/continent derived from the archived Natural Earth layer, not from the "
            "container's partly null country attribute, and published as 'country'",
            "conus_context is the Natural Earth USA polygon clipped to a -125..-66.5 E, 24..49.5 N box",
            "conus_states_context is the archived Natural Earth 50m admin-1 layer, US units "
            "excluding Alaska and Hawaii, clipped to the same box; ACCEPTED 2026-08-25 (handoff "
            "section 6.7) and drawn as quiet muted-gray context in the E1 and E2 maps",
            "slv_context is a convex hull of the generalized E3 centroids buffered 0.15 degrees",
            "slv_basin_context is the archived five-unit HUC8 Rio Grande headwaters extract from "
            "the USGS WBD; REJECTED as artwork 2026-08-25 (handoff section 6.7 keeps HUC8 units "
            "out of the figure) and retained here as hashed provenance only, not drawn",
        ],
        units={"coordinates": "decimal degrees, EPSG:4326"},
        deterministic_seed=None,
        configured_counts={"E1": 60, "E2": 66, "E3": 50},
        evaluated_counts={
            "E1_daily": 45,
            "E2_daily": 63,
            "E3_field_years": EXPECTED["E3_field_years"],
        },
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_evidence_matrix.csv",
        rows=n_ev,
        note=(
            "E0 is typed evidence_role=model_development so generic plotting code cannot count "
            "it as a fourth evaluation geography. Configured, target-composition, weighting and "
            "independence labels are asserted against handoff section 11 before writing."
        ),
        cohort_key="experiment",
        inclusion_rule="One row per current evaluation experiment plus one E0 model-development record.",
        experiment_mapping=EXPERIMENT_MAP,
        deterministic_seed=None,
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_example_timeseries.csv",
        rows=n_ex,
        note=(
            f"The panel (a) example record: the {FIG01_EXAMPLE_SITE} window regenerated by "
            "_build_fig01_example_source under the frozen window rule with the recorded "
            "2026-08-27 directed-site override, and joined to the frozen Figure 4 per-capture "
            "retrieval member values. Benchmark series and flags, every performance metric, and the daily "
            "filled NDVI trajectory are refused by assertion; every emitted column carries "
            "recorded provenance in fig01_example_selection.json. Re-audited 2026-08-24 against "
            "handoff sections 10.2 and 15.2: column values and the column set are unchanged, but "
            "irr_applied is now typed display-role e1_example_irrigation_stems_only -- it is E1 "
            "example evidence, never an E3 modeled series, and may not be aggregated in the "
            "artwork or connected to the separate 'Metered Water · E3' key."
        ),
        sources=example_selection["sources"],
        experiment_mapping={"E1": "legacy e2_*"},
        cohort_key="site_id + date",
        inclusion_rule=(
            f"single frozen site {FIG01_EXAMPLE_SITE}, {FIG01_EXAMPLE_START} to "
            f"{FIG01_EXAMPLE_END} inclusive, no display crop"
        ),
        temporal_support_rule=(
            "daily rows throughout; the six member ETf columns, the target mean, the member "
            f"count and the ensemble spread are populated only on the {FIG01_EXAMPLE_CAPTURES} "
            "calibration captures and are null elsewhere by design -- sparse observations must "
            "remain visibly discontinuous"
        ),
        units={c: v["units"] for c, v in example_prov.items() if v.get("units")},
        display_transformations=[
            "read from the frozen fig03/fig04 package artifacts; nothing is re-derived from a "
            "container",
            "member marks reconciled to the frozen member count (exact) and to the target mean "
            f"and ensemble spread (max abs difference "
            f"{example_selection['reconciliation']['target_mean_max_abs_difference']:.3e} and "
            f"{example_selection['reconciliation']['spread_max_abs_difference']:.3e}, a float32 "
            "container-storage artifact)",
            "swe_audit is frozen as a non-plotted audit column: SWE is identically zero across "
            "this growing-season window and is represented symbolically, never as a zero trace",
        ],
        deterministic_seed=None,
        configured_counts={"E1": 60},
        evaluated_counts={
            "days_in_window": int(n_ex),
            "calibration_captures": int(n_ex_caps),
        },
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_example_selection.json",
        rows=None,
        note=(
            "Site, full and displayed windows, rationale, source hashes, the member-mark "
            "reconciliation record, the per-column provenance map, and the explicit statement "
            "that neither the selection nor any crop used evaluation residuals or performance "
            "metrics. The 2026-08-24 re-audit block records that no column value changed and "
            "adds per-column display roles and forbidden uses, so a downstream plotting script "
            "cannot silently promote the E1 example's irr_applied series into E3 evidence."
        ),
        selected_site=FIG01_EXAMPLE_SITE,
        selected_window=[FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
        cohort_key="site_id",
        inclusion_rule=(
            "frozen window rule re-run by _build_fig01_example_source under the recorded "
            "2026-08-27 directed-site override in fig03_example_selection.json; no Figure 1 "
            "performance ranking"
        ),
        deterministic_seed=None,
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_architecture.json",
        rows=None,
        note=(
            "Architecture 3.3.0 (supersedes 3.2.0, 3.1.1, 3.1.0, 3.0.0 and 2.1.0): the "
            "two-panel evidence-first composition with panel (a) redesigned per the "
            "2026-08-25 handoff rewrite and panel (b) accepted as built in revision 3. Frozen "
            f"reader-facing strings including the required full-black '{FIG01_RECORD_ID}' record "
            "identification, the FIVE panel (a) quantitative plotting regions with their source "
            "columns, per-region y-axis contract and recorded display domains -- root-zone "
            "depletion merged with the magnitude-bearing irrigation stems on two explicit "
            "scales, and daily ET merged with 'Flux ET (Held Out)' on one shared axis and "
            "vertical scale compared by shared alignment alone -- the closed directed inverse "
            "cycle 'Run Balance -> Compare -> Update Parameters -> Run Balance' with the ETf "
            "targets and '+ SWE' entering Compare, the NDVI and daily-forcing drivers entering "
            "Run Balance, and the labelled 'Conditioned Parameters' exit to the displayed daily "
            "trajectory; the separate typographic E3 evaluation key with its 'Annual Total' "
            "operation, the held-out boundary carried by the direct trace label rather than a "
            "heading, the three map nodes, the two class-specific parameter tokens with the "
            "visible irrigated branch junction, the allowed edges, the undirected comparisons, "
            "the forbidden-edge list extended so no held-out observation reaches any cycle "
            "stage, and a title/direct_label/annotation/proof_only classification of every "
            "visible string. "
            "Objective notation, retrieval members, weighting, the PEST++ IES engine name, "
            "realization and parameter counts, paired support, the ECOSTRESS sensitivity, the "
            "eight caption-contract items and the working caption live in a separate "
            "caption_facts block that the builder asserts never reaches visible copy."
        ),
        cohort_key=None,
        inclusion_rule=None,
        deterministic_seed=None,
        schema_version=arch["schema_version"],
        supersedes_schema_version=arch["supersedes"]["schema_version"],
        visible_string_count=arch["visible_string_count"],
        explanatory_annotations=n_annot,
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_metadata.json",
        rows=None,
        note=(
            "Boundary dataset version and hash, the two archived optional context datasets with "
            "their URLs, hashes and public-domain licences, CRS provenance, legacy-to-current "
            "label map, E3 generalization method with its 2026-08-19 approval, cohort and "
            "irrigation-class assertions, the example-selection reference, font inventory, "
            "190 x 120 mm output dimensions, and the scientific / privacy / visual review record."
        ),
        cohort_key=None,
        inclusion_rule=None,
        deterministic_seed=None,
        generator_version=FIG01_BUILDER_VERSION,
    )
    print(
        f"  fig01: gpkg layers e1={len(e1_gdf)} e2={len(e2_gdf)} e3={len(e3_disp)} "
        f"(overlap {int(e1['in_e2'].sum())}); classes E1 "
        f"{class_counts['e1_irrigated']}i/{class_counts['e1_rainfed']}r, E2 "
        f"{class_counts['e2_irrigated']}i/{class_counts['e2_rainfed']}r; "
        f"evidence matrix {n_ev} rows; example {n_ex} rows / {n_ex_caps} captures; "
        f"architecture {arch['schema_version']} with {arch['visible_string_count']} visible "
        f"strings, {n_annot} annotation(s), {len(arch['edges'])} edges"
    )


def _ne_version() -> str:
    p = NE_COUNTRIES.with_suffix("").parent / "ne_110m_admin_0_countries.VERSION.txt"
    if p.exists():
        return p.read_text().strip()
    return "unknown"


def _ne_states_version() -> str:
    p = NE_STATES.with_suffix(".VERSION.txt")
    if p.exists():
        return p.read_text().strip()
    return "unknown"

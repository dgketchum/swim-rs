"""Figure 3: pooled daily ET agreement and temporal-support effects."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .cohorts import EXPECTED, METRIC_DEFS
from .manifest import (
    MANIFEST,
    BuildError,
    require_columns,
    require_count,
    require_unique,
    sha256,
    write_table,
)
from .paths import FINAL, SCRIPT_VERSION

# --------------------------------------------------------------------------
# Figure 3 -- pooled daily ET agreement and temporal-support effects
# --------------------------------------------------------------------------
#
# Figure 3 is a display-only transformation of the promoted E1 benchmark
# package. It never reconstructs OpenET independently and never reads a model
# archive, container, or e2_* metric table. The promoted evaluator-owned
# paired record supplies every plotted daily value and temporal class; grouped
# contrasts and interactions are copied only after exact recomputation checks.
#
# The fig03_example_* files are Figure 1 example-source artifacts only (the
# fig03_* filenames are fixed). This builder re-registers their manifest
# records but never touches their content.

FIG03_SOURCE = FINAL / "e1_openet_benchmark"
FIG03_SOURCE_MANIFEST = FIG03_SOURCE / "MANIFEST.json"
FIG03_SOURCE_FILES = {
    "daily_record": "daily/evaluation_paired_daily_records.csv",
    "temporal_metrics": "temporal/evaluation_temporal_grouped_metrics.csv",
    "temporal_contrasts": "temporal/evaluation_temporal_grouped_contrasts.csv",
    "temporal_interactions": "temporal/evaluation_temporal_interactions.csv",
    "temporal_eligibility": "temporal/evaluation_temporal_site_eligibility.csv",
}
FIG03_TEMPORAL_CLASSES = ["retrieval", "between_retrieval"]
FIG03_AGGREGATIONS = ["sqrt_n_weighted_site_metric", "pooled_observations"]
FIG03_AGGREGATION_LABELS = {
    "sqrt_n_weighted_site_metric": "Station-weighted",
    "pooled_observations": "Pooled",
}
FIG03_METHODS = [
    ("openet_ensemble", "OpenET", "openet_et"),
    ("swim", "SWIM-RS", "swim_et"),
]
FIG03_METRICS = ["kge", "rmse", "mbe"]
FIG03_MIN_PAIRED = 10
FIG03_BOOTSTRAP_SEED = 42
FIG03_BOOTSTRAP_REPS = 10_000
FIG03_AXIS_LO = -2.0
FIG03_AXIS_HI = 16.0
FIG03_AXIS_TICKS = [0, 4, 8, 12, 16]
FIG03_DRAW_ORDER_SEED = 27082026
FIG03_COMPOSITION_ID = "fig03_temporal_grouped_v2 (2026-09-01)"


def _fig03_signed_display(value: float) -> str:
    """Two-decimal display with an explicit sign and a true minus (U+2212)."""
    rounded = round(value, 2)
    if rounded == 0:
        return "0.00"
    if rounded > 0:
        return f"+{rounded:.2f}"
    return f"−{abs(rounded):.2f}"


def _fig03_metrics(obs: np.ndarray, mod: np.ndarray, label: str) -> dict[str, float]:
    """Pearson r, KGE (Gupta 2009), RMSE, and signed MBE on paired vectors."""
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    if len(obs) != len(mod) or len(obs) < FIG03_MIN_PAIRED:
        raise BuildError(f"fig03 metrics {label}: n={len(obs)} below minimum {FIG03_MIN_PAIRED}")
    if not (np.isfinite(obs).all() and np.isfinite(mod).all()):
        raise BuildError(f"fig03 metrics {label}: nonfinite paired value")
    if np.std(obs) <= 0 or np.mean(obs) <= 0:
        raise BuildError(f"fig03 metrics {label}: degenerate flux vector")
    r = float(np.corrcoef(obs, mod)[0, 1])
    rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
    mbe = float(np.mean(mod - obs))
    alpha = float(np.std(mod) / np.std(obs))
    beta = float(np.mean(mod) / np.mean(obs))
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {"r": r, "kge": kge, "rmse": rmse, "mbe": mbe}


def _fig03_source_artifacts() -> tuple[dict, dict[str, Path]]:
    """Load the promoted manifest and hash-verify every Figure 3 source."""
    if not FIG03_SOURCE_MANIFEST.exists():
        raise BuildError(f"fig03 promoted manifest missing: {FIG03_SOURCE_MANIFEST}")
    manifest = json.loads(FIG03_SOURCE_MANIFEST.read_text())
    if manifest.get("status") != "frozen_for_results_reporting":
        raise BuildError(
            f"fig03 source package is not frozen_for_results_reporting: {manifest.get('status')!r}"
        )
    source_paths = {key: FIG03_SOURCE / rel for key, rel in FIG03_SOURCE_FILES.items()}
    frozen_hashes = manifest.get("artifact_sha256", {})
    for key, path in source_paths.items():
        if not path.exists():
            raise BuildError(f"fig03 promoted source missing: {path}")
        rel = FIG03_SOURCE_FILES[key]
        want = frozen_hashes.get(rel)
        if not want:
            raise BuildError(f"fig03 promoted manifest has no hash for {rel}")
        got = sha256(path)
        if got != want:
            raise BuildError(f"fig03 promoted source drift for {rel}: {got[:12]} != {want[:12]}")
    return manifest, source_paths


def _fig03_grouped_estimate(
    pooled: pd.DataFrame,
    site_metrics: pd.DataFrame,
    temporal_class: str,
    aggregation: str,
    model: str,
    metric: str,
) -> float:
    """Recompute one canonical grouped estimate from the plotted record."""
    if aggregation == "pooled_observations":
        sub = pooled[pooled["temporal_support"] == temporal_class]
        col = "swim_et" if model == "swim" else "openet_et"
        return _fig03_metrics(
            sub["flux_et"].to_numpy(),
            sub[col].to_numpy(),
            f"{temporal_class}/{aggregation}/{model}",
        )[metric]
    sub = site_metrics[
        (site_metrics["temporal_support"] == temporal_class) & (site_metrics["model"] == model)
    ]
    require_count(
        len(sub),
        EXPECTED["E1_split_common"],
        f"fig03 station metrics {temporal_class}/{model}/{metric}",
    )
    weights = np.sqrt(sub["n_paired"].to_numpy(dtype=float))
    return float(np.average(sub[metric].to_numpy(dtype=float), weights=weights))


def _fig03_quarantine_legacy_package() -> list[str]:
    """Move tables from a previous build of this figure to ``superseded_fig03_pre_openet_core``.

    Runs only when the existing ``fig03_metadata.json`` names a source package other
    than ``paper/data/final/e1_openet_benchmark``.
    """
    meta_path = paths.OUT / "fig03_metadata.json"
    if not meta_path.exists():
        return []
    try:
        current = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise BuildError(f"fig03 existing metadata is unreadable: {exc}") from exc
    if current.get("source_package") == "paper/data/final/e1_openet_benchmark":
        return []

    qdir = paths.OUT / "superseded_fig03_pre_openet_core"
    qdir.mkdir(exist_ok=True)
    names = [
        "fig03_pooled_daily_agreement.csv",
        "fig03_scatter_metrics.csv",
        "fig03_temporal_site_metrics.csv",
        "fig03_temporal_site_effects.csv",
        "fig03_temporal_cohort_effects.csv",
        "fig03_metadata.json",
    ]
    moved = []
    for name in names:
        src = paths.OUT / name
        if not src.exists():
            continue
        dst = qdir / name
        if dst.exists():
            raise BuildError(
                f"fig03 quarantine target already exists: {dst}; inspect before replacing"
            )
        src.rename(dst)
        moved.append(name)
    return moved


def build_fig03() -> None:
    """Freeze Figure 3 directly from the promoted E1 reporting package."""
    source_manifest, source_paths = _fig03_source_artifacts()

    record = pd.read_csv(source_paths["daily_record"], float_precision="round_trip")
    eligibility = pd.read_csv(source_paths["temporal_eligibility"])
    grouped_metrics = pd.read_csv(source_paths["temporal_metrics"], float_precision="round_trip")
    grouped_contrasts = pd.read_csv(
        source_paths["temporal_contrasts"], float_precision="round_trip"
    )
    interactions = pd.read_csv(source_paths["temporal_interactions"], float_precision="round_trip")

    require_columns(
        record,
        [
            "fid",
            "date",
            "flux_et_mm_d",
            "swim_et_mm_d",
            "openet_et_mm_d",
            "openet_support_class",
            "temporal_class",
        ],
        "fig03 promoted daily record",
    )
    require_unique(record, ["fid", "date"], "fig03 promoted daily record")
    require_columns(
        eligibility,
        [
            "fid",
            "n_retrieval",
            "n_between_retrieval",
            "in_common_cohort",
        ],
        "fig03 temporal eligibility",
    )
    require_unique(eligibility, ["fid"], "fig03 temporal eligibility")
    common_sites = sorted(
        eligibility.loc[eligibility["in_common_cohort"].astype(bool), "fid"].astype(str).tolist()
    )
    require_count(len(common_sites), EXPECTED["E1_split_common"], "fig03 common temporal cohort")

    pooled = record[record["fid"].astype(str).isin(common_sites)].copy()
    if set(pooled["temporal_class"]) != set(FIG03_TEMPORAL_CLASSES):
        raise BuildError("fig03 temporal classes do not match retrieval/between_retrieval")
    retrieval_support = pooled.loc[pooled["temporal_class"] == "retrieval", "openet_support_class"]
    between_support = pooled.loc[
        pooled["temporal_class"] == "between_retrieval", "openet_support_class"
    ]
    if set(retrieval_support) != {"capture"}:
        raise BuildError("fig03 retrieval rows are not exactly OpenET capture rows")
    if (between_support == "capture").any():
        raise BuildError("fig03 between-retrieval rows include an OpenET capture")

    pooled = pooled.rename(
        columns={
            "fid": "site_id",
            "temporal_class": "temporal_support",
            "flux_et_mm_d": "flux_et",
            "swim_et_mm_d": "swim_et",
            "openet_et_mm_d": "openet_et",
        }
    )
    pooled.insert(0, "experiment", "E1")
    pooled["date"] = pd.to_datetime(pooled["date"]).dt.strftime("%Y-%m-%d")
    pooled = pooled[
        [
            "experiment",
            "site_id",
            "date",
            "temporal_support",
            "openet_support_class",
            "flux_et",
            "swim_et",
            "openet_et",
        ]
    ].sort_values(["site_id", "date"])
    require_unique(pooled, ["site_id", "date"], "fig03 pooled daily agreement")
    if not np.isfinite(pooled[["flux_et", "swim_et", "openet_et"]].to_numpy(dtype=float)).all():
        raise BuildError("fig03 plotted record contains nonfinite ET")

    n_retrieval = int((pooled["temporal_support"] == "retrieval").sum())
    n_between = int((pooled["temporal_support"] == "between_retrieval").sum())
    require_count(n_retrieval, EXPECTED["E1_pool_retrieval"], "fig03 retrieval site-days")
    require_count(
        n_between,
        EXPECTED["E1_pool_between_retrieval"],
        "fig03 between-retrieval site-days",
    )
    require_count(
        len(pooled),
        EXPECTED["E1_pool_temporal_common"],
        "fig03 common temporal site-days",
    )
    per_site = pooled.groupby(["site_id", "temporal_support"]).size().unstack()
    if (per_site[FIG03_TEMPORAL_CLASSES] < FIG03_MIN_PAIRED).any().any():
        raise BuildError("fig03 common cohort contains a support class below 10 days")

    plotted = pooled[["flux_et", "swim_et", "openet_et"]].to_numpy(dtype=float)
    v_lo, v_hi = float(np.min(plotted)), float(np.max(plotted))
    if v_lo < FIG03_AXIS_LO or v_hi > FIG03_AXIS_HI:
        raise BuildError(
            f"fig03 plotted extrema [{v_lo:.3f}, {v_hi:.3f}] exceed "
            f"[{FIG03_AXIS_LO}, {FIG03_AXIS_HI}]"
        )

    # Panel (a): descriptive pooled statistics from the exact plotted rows,
    # cross-checked against the promoted grouped products.
    scatter_rows = []
    for model, label, col in FIG03_METHODS:
        for temporal_class in FIG03_TEMPORAL_CLASSES:
            sub = pooled[pooled["temporal_support"] == temporal_class]
            metrics = _fig03_metrics(
                sub["flux_et"].to_numpy(),
                sub[col].to_numpy(),
                f"scatter/{label}/{temporal_class}",
            )
            for metric in ("r", "kge", "rmse", "mbe"):
                source = grouped_metrics[
                    (grouped_metrics["temporal_class"] == temporal_class)
                    & (grouped_metrics["aggregation"] == "pooled_observations")
                    & (grouped_metrics["model"] == model)
                    & (grouped_metrics["metric"] == metric)
                ]
                require_count(
                    len(source),
                    1,
                    f"fig03 source scatter {label}/{temporal_class}/{metric}",
                )
                if abs(float(source.iloc[0]["estimate"]) - metrics[metric]) > 1e-12:
                    raise BuildError(
                        f"fig03 scatter {label}/{temporal_class}/{metric} "
                        "does not reproduce promoted grouped metric"
                    )
            scatter_rows.append(
                {
                    "temporal_support": temporal_class,
                    "method": label,
                    "n_sites": int(sub["site_id"].nunique()),
                    "n_site_days": int(len(sub)),
                    "pearson_r": metrics["r"],
                    "kge": metrics["kge"],
                    "mbe": metrics["mbe"],
                    "rmse": metrics["rmse"],
                    "display_r": f"{metrics['r']:.2f}",
                    "display_kge": f"{metrics['kge']:.2f}",
                    "display_mbe": _fig03_signed_display(metrics["mbe"]),
                    "display_rmse": f"{metrics['rmse']:.2f}",
                }
            )
    scatter = pd.DataFrame(scatter_rows)

    # Site metrics are retained in memory to reproduce station weighting and to
    # expose the direct per-site temporal interaction in panel (c).
    site_metric_rows = []
    pooled_dt = pooled.assign(date=pd.to_datetime(pooled["date"]))
    for site_id in common_sites:
        sdf = pooled_dt[pooled_dt["site_id"] == site_id]
        for temporal_class in FIG03_TEMPORAL_CLASSES:
            sub = sdf[sdf["temporal_support"] == temporal_class]
            for model, _label, col in FIG03_METHODS:
                metrics = _fig03_metrics(
                    sub["flux_et"].to_numpy(),
                    sub[col].to_numpy(),
                    f"site/{site_id}/{temporal_class}/{model}",
                )
                site_metric_rows.append(
                    {
                        "site_id": site_id,
                        "temporal_support": temporal_class,
                        "model": model,
                        "n_paired": int(len(sub)),
                        **metrics,
                    }
                )
    site_metrics = pd.DataFrame(site_metric_rows)
    require_unique(
        site_metrics,
        ["site_id", "temporal_support", "model"],
        "fig03 site metrics",
    )
    require_count(
        len(site_metrics),
        EXPECTED["E1_split_common"] * 2 * 2,
        "fig03 site metric rows",
    )

    # Reproduce every promoted grouped estimate used by panels (b-c).
    recomputed = {}
    for temporal_class in FIG03_TEMPORAL_CLASSES:
        for aggregation in FIG03_AGGREGATIONS:
            for model, _label, _col in FIG03_METHODS:
                for metric in FIG03_METRICS:
                    value = _fig03_grouped_estimate(
                        pooled,
                        site_metrics,
                        temporal_class,
                        aggregation,
                        model,
                        metric,
                    )
                    recomputed[(temporal_class, aggregation, model, metric)] = value
                    source = grouped_metrics[
                        (grouped_metrics["temporal_class"] == temporal_class)
                        & (grouped_metrics["aggregation"] == aggregation)
                        & (grouped_metrics["model"] == model)
                        & (grouped_metrics["metric"] == metric)
                    ]
                    require_count(
                        len(source),
                        1,
                        f"fig03 grouped metric {temporal_class}/{aggregation}/{model}/{metric}",
                    )
                    if abs(float(source.iloc[0]["estimate"]) - value) > 1e-12:
                        raise BuildError(
                            "fig03 grouped estimate does not reproduce promoted value: "
                            f"{temporal_class}/{aggregation}/{model}/{metric}"
                        )

    contrast = grouped_contrasts[
        grouped_contrasts["temporal_class"].isin(FIG03_TEMPORAL_CLASSES)
        & grouped_contrasts["aggregation"].isin(FIG03_AGGREGATIONS)
        & grouped_contrasts["metric"].isin(FIG03_METRICS)
    ].copy()
    require_count(len(contrast), 12, "fig03 grouped contrast rows")
    require_unique(
        contrast,
        ["temporal_class", "aggregation", "metric"],
        "fig03 grouped contrasts",
    )
    contrast["aggregation_label"] = contrast["aggregation"].map(FIG03_AGGREGATION_LABELS)
    for row in contrast.itertuples(index=False):
        expected = (
            recomputed[(row.temporal_class, row.aggregation, "swim", row.metric)]
            - recomputed[(row.temporal_class, row.aggregation, "openet_ensemble", row.metric)]
        )
        if abs(float(row.estimate) - expected) > 1e-12:
            raise BuildError(
                "fig03 grouped contrast does not reproduce promoted value: "
                f"{row.temporal_class}/{row.aggregation}/{row.metric}"
            )
        if not (
            np.isfinite([row.estimate, row.ci95_low, row.ci95_high]).all()
            and row.ci95_low <= row.estimate <= row.ci95_high
        ):
            raise BuildError("fig03 grouped contrast has an invalid interval")
        if (
            int(row.n_sites) != EXPECTED["E1_split_common"]
            or int(row.bootstrap_reps) != FIG03_BOOTSTRAP_REPS
            or int(row.bootstrap_seed) != FIG03_BOOTSTRAP_SEED
        ):
            raise BuildError("fig03 grouped contrast metadata drifted")
    contrast.insert(0, "experiment", "E1")
    contrast = contrast.sort_values(["metric", "temporal_class", "aggregation"])

    interaction = interactions[
        interactions["aggregation"].isin(FIG03_AGGREGATIONS)
        & interactions["metric"].isin(FIG03_METRICS)
    ].copy()
    require_count(len(interaction), 6, "fig03 interaction rows")
    require_unique(interaction, ["aggregation", "metric"], "fig03 interactions")
    interaction["aggregation_label"] = interaction["aggregation"].map(FIG03_AGGREGATION_LABELS)
    for row in interaction.itertuples(index=False):
        between = (
            recomputed[("between_retrieval", row.aggregation, "swim", row.metric)]
            - recomputed[
                (
                    "between_retrieval",
                    row.aggregation,
                    "openet_ensemble",
                    row.metric,
                )
            ]
        )
        retrieval = (
            recomputed[("retrieval", row.aggregation, "swim", row.metric)]
            - recomputed[("retrieval", row.aggregation, "openet_ensemble", row.metric)]
        )
        expected = between - retrieval
        if abs(float(row.estimate) - expected) > 1e-12:
            raise BuildError(
                f"fig03 interaction does not reproduce promoted value: "
                f"{row.aggregation}/{row.metric}"
            )
        if not (
            np.isfinite([row.estimate, row.ci95_low, row.ci95_high]).all()
            and row.ci95_low <= row.estimate <= row.ci95_high
        ):
            raise BuildError("fig03 interaction has an invalid interval")
        if (
            int(row.n_sites) != EXPECTED["E1_split_common"]
            or int(row.bootstrap_reps) != FIG03_BOOTSTRAP_REPS
            or int(row.bootstrap_seed) != FIG03_BOOTSTRAP_SEED
        ):
            raise BuildError("fig03 interaction metadata drifted")
    interaction.insert(0, "experiment", "E1")
    interaction = interaction.sort_values(["metric", "aggregation"])

    # Direct site-level temporal interaction:
    # (SWIM - OpenET)_between - (SWIM - OpenET)_retrieval.
    site_interaction_rows = []
    indexed = site_metrics.set_index(["site_id", "temporal_support", "model"])
    for site_id in common_sites:
        row = {
            "experiment": "E1",
            "site_id": site_id,
            "n_retrieval": int(indexed.loc[(site_id, "retrieval", "swim"), "n_paired"]),
            "n_between_retrieval": int(
                indexed.loc[(site_id, "between_retrieval", "swim"), "n_paired"]
            ),
        }
        for metric in FIG03_METRICS:
            retrieval_delta = float(
                indexed.loc[(site_id, "retrieval", "swim"), metric]
                - indexed.loc[(site_id, "retrieval", "openet_ensemble"), metric]
            )
            between_delta = float(
                indexed.loc[(site_id, "between_retrieval", "swim"), metric]
                - indexed.loc[(site_id, "between_retrieval", "openet_ensemble"), metric]
            )
            row[f"retrieval_d_{metric}"] = retrieval_delta
            row[f"between_retrieval_d_{metric}"] = between_delta
            row[f"interaction_{metric}"] = between_delta - retrieval_delta
        site_interaction_rows.append(row)
    site_interaction = pd.DataFrame(site_interaction_rows)
    require_count(
        len(site_interaction),
        EXPECTED["E1_split_common"],
        "fig03 site interaction rows",
    )
    require_unique(site_interaction, ["site_id"], "fig03 site interactions")
    order = site_interaction.sort_values(["interaction_kge", "site_id"]).reset_index(drop=True)
    rank = {site_id: idx + 1 for idx, site_id in enumerate(order["site_id"])}
    site_interaction["site_order_interaction_kge"] = site_interaction["site_id"].map(rank)
    if sorted(site_interaction["site_order_interaction_kge"]) != list(
        range(1, EXPECTED["E1_split_common"] + 1)
    ):
        raise BuildError("fig03 site interaction order is not a 1..43 permutation")
    site_interaction = site_interaction.sort_values("site_order_interaction_kge").reset_index(
        drop=True
    )

    quarantined = _fig03_quarantine_legacy_package()

    n_pool = write_table(pooled, "fig03_pooled_daily_agreement.csv")
    n_scatter = write_table(scatter, "fig03_scatter_metrics.csv")
    n_contrast = write_table(contrast, "fig03_grouped_contrasts.csv")
    n_interaction = write_table(interaction, "fig03_interactions.csv")
    n_site = write_table(site_interaction, "fig03_site_interactions.csv")

    source_hashes = {FIG03_SOURCE_FILES[key]: sha256(path) for key, path in source_paths.items()}
    metadata = {
        "figure": "Figure 3 -- daily ET agreement by temporal support",
        "contract": "paper/notes/fig03_production_handoff.md (2026-09-01)",
        "composition_id": FIG03_COMPOSITION_ID,
        "source_package": "paper/data/final/e1_openet_benchmark",
        "source_manifest_sha256": sha256(FIG03_SOURCE_MANIFEST),
        "source_status": source_manifest["status"],
        "source_artifact_sha256": source_hashes,
        "counts": {
            "sites": EXPECTED["E1_split_common"],
            "retrieval_site_days": n_retrieval,
            "between_retrieval_site_days": n_between,
            "total_site_days": len(pooled),
        },
        "temporal_support": {
            "retrieval": "OpenET support class capture",
            "between_retrieval": (
                "OpenET support class interpolated or flat_fill; classification "
                "comes from the evaluator-owned paired record"
            ),
        },
        "metrics": {
            "pearson_r": "Pearson correlation on the exact plotted facet rows",
            "kge": METRIC_DEFS["kge"]["definition"],
            "rmse": "root mean square error against flux ET, mm d-1",
            "mbe": "mean(model - flux), mm d-1, signed",
            "model_contrast": "SWIM-RS minus OpenET",
            "temporal_interaction": (
                "(SWIM-RS minus OpenET) between retrievals minus "
                "(SWIM-RS minus OpenET) on retrieval dates"
            ),
        },
        "aggregation": {
            "primary": {
                "machine_name": "sqrt_n_weighted_site_metric",
                "reader_label": "Station-weighted",
            },
            "complementary": {
                "machine_name": "pooled_observations",
                "reader_label": "Pooled",
            },
        },
        "bootstrap": {
            "kind": "whole-site resampling with replacement",
            "n_resamples": FIG03_BOOTSTRAP_REPS,
            "seed": FIG03_BOOTSTRAP_SEED,
            "interval": "95% percentile",
        },
        "panel_a": {
            "axes_mm_day": [FIG03_AXIS_LO, FIG03_AXIS_HI],
            "ticks": FIG03_AXIS_TICKS,
            "plotted_extrema_mm_day": [v_lo, v_hi],
            "draw_order_seed": FIG03_DRAW_ORDER_SEED,
        },
        "panel_b": (
            "station-weighted and pooled SWIM-RS-minus-OpenET contrasts within "
            "retrieval and between-retrieval classes"
        ),
        "panel_c": (
            "43 direct site-level temporal interactions plus promoted "
            "station-weighted and pooled interactions"
        ),
        "superseded_display_package": {
            "directory": str(paths.OUT / "superseded_fig03_pre_openet_core"),
            "files_moved": quarantined,
            "reason": (
                "the prior package used a pre-openet-core unbounded ETf interpolation "
                "and secondary median site-effect summaries"
            ),
        },
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
        "frozen_utc": datetime.now(UTC).isoformat(),
    }
    (paths.OUT / "fig03_metadata.json").write_text(json.dumps(metadata, indent=2))

    common_meta = {
        "experiment_mapping": {"E1": "examples/5_Flux_Ensemble"},
        "source_package": "paper/data/final/e1_openet_benchmark",
        "source_manifest_sha256": sha256(FIG03_SOURCE_MANIFEST),
        "contract": "paper/notes/fig03_production_handoff.md (2026-09-01)",
    }
    MANIFEST.add(
        "fig03_pooled_daily_agreement.csv",
        role="Panel (a) exact pooled daily plotting rows",
        rows=n_pool,
        grain="site_id x date",
        **common_meta,
    )
    MANIFEST.add(
        "fig03_scatter_metrics.csv",
        role="Panel (a) Pearson r, signed MBE, and RMSE display values",
        rows=n_scatter,
        grain="method x temporal support",
        **common_meta,
    )
    MANIFEST.add(
        "fig03_grouped_contrasts.csv",
        role="Panel (b) promoted station-weighted and pooled model contrasts",
        rows=n_contrast,
        grain="temporal support x aggregation x metric",
        **common_meta,
    )
    MANIFEST.add(
        "fig03_interactions.csv",
        role="Panel (c) promoted grouped temporal interactions",
        rows=n_interaction,
        grain="aggregation x metric",
        **common_meta,
    )
    MANIFEST.add(
        "fig03_site_interactions.csv",
        role="Panel (c) direct site-level temporal interactions",
        rows=n_site,
        grain="site_id",
        **common_meta,
    )
    MANIFEST.add(
        "fig03_metadata.json",
        role="Figure 3 source, estimand, count, and display contract",
        **common_meta,
    )

    # Keep the fig03_* names that are owned by Figure 1.
    prior_manifest_path = paths.OUT / "fig_manifest.json"
    if not prior_manifest_path.exists():
        raise BuildError("fig03: fig_manifest.json missing; cannot re-register Figure 1 provenance")
    prior_tables = json.loads(prior_manifest_path.read_text()).get("tables", {})
    for name in ("fig03_example_timeseries.csv", "fig03_example_selection.json"):
        if not (paths.OUT / name).exists():
            raise BuildError(f"fig03: retained Figure 1 provenance file missing: {name}")
        rec = dict(prior_tables.get(name, {}))
        rec.pop("output_sha256", None)
        rec.pop("output_bytes", None)
        rec.setdefault(
            "role",
            "Figure 1 example source (historical fig03_ filename retained); not a Figure 3 input",
        )
        MANIFEST.add(name, **rec)

    print(
        f"  fig03: pooled {n_pool} rows ({n_retrieval} retrieval + "
        f"{n_between} between), scatter {n_scatter}, grouped contrasts "
        f"{n_contrast}, interactions {n_interaction}, site interactions {n_site}; "
        f"quarantined {quarantined or 'nothing'}"
    )

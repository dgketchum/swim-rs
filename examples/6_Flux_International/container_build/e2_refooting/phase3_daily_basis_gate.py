"""Phase 3 / Gate G3 (redefined 2026-09-04): daily ET consistency and low-ETo stability.

The daily ERA5-Land ``ETr/ETo`` conversion is retained (user decision). This gate no longer asks
whether the corrected ETf matches the USGS NHM product's overpass-instantaneous grass basis
(``phase3_nhm_basis_gate.py`` reports that as a diagnostic). It asks two things that matter for
SWIM, which applies ETf to a daily ETo:

1. **Daily ET consistency.** Every written grass-basis value, read back from the CSVs the
   container will ingest (not from the ledger), satisfies ``corrected * ETo == native * ETr`` on
   its own site-day; every ``ok`` ledger row is written exactly once; nothing else is written.
2. **Low-ETo stability on the weighted observations.** Very small daily ETo inflates the ratio
   (winter ETr/ETo reaches 30+). The question is whether such dates enter the calibration
   objective materially. The two-member ensemble weights are reproduced exactly as
   ``swimrs.calibrate.pest_builder.PestBuilder._write_etf_obs`` computes them for the E2
   configuration (target = member mean, SD = two-member sample SD, weight = target / (SD + 0.05),
   zero weight below two members, zero weight on dates with daily ETo below the declared
   ``etf_weighting_eto_floor`` of 1.0 mm — adopted 2026-09-07 after the complete-record run
   without the floor failed the increase threshold) after the ingest rules (0.05 <= ETf <= 2.0),
   for the native and the corrected SSEBop member against the unchanged PT-JPL member, on the
   66-site cohort. The objective contribution of an observation scales with weight**2, so shares
   of the pooled sum(weight**2) are reported by daily-ETo bin, month, ratio-screen flag and site.
   The weights the floor removes are kept in ``*_nofloor`` columns and summarised under
   ``eto_floor`` / ``low_eto_without_floor`` so the rule's effect is visible, not hidden.

Thresholds are the §3 rows predeclared on 2026-09-04 before this script was first run. No flux
data is read; the primary container is opened read-only.

Outputs (QA root):
    daily_basis_gate_weights.csv    one row per 66-cohort site-date with any member: members,
                                    both configurations' target/SD/count/weight, ETo bin, flags
    daily_basis_gate_by_site.csv    per-site weight shares, ceiling losses, low-ETo shares
    daily_basis_gate_summary.json   thresholds, consistency result, shares, gate verdict

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase3_daily_basis_gate.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import fiona
import numpy as np
import pandas as pd
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
CONTAINER = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")
COHORT_66 = os.path.join(DATA, "gis", "flux_crop_pub_66_150m.shp")
LEDGER = os.path.join(QA_ROOT, "ssebop_conversion_ledger.csv")
GRASS_DIR = os.path.join(
    DATA, "remote_sensing", "landsat", "extracts", "ssebop_etf_grass", "no_mask"
)
YEARS = (2013, 2025)

GRASS_RE = re.compile(r"^ssebop_etf_grass_(?P<site>.+)_no_mask_(?P<year>\d{4})\.csv$")

# exactly the E2 configuration (6_Flux_International_LSEnsemble_POR_annual2yr.toml + ingestor)
INGEST = {"min_etf": 0.05, "max_etf": 2.0}
WEIGHTING = {
    "mode": "spread",
    "spread_floor": 0.05,
    "min_members": 2,
    "std_ddof": 1,
    "eto_floor": 1.0,  # etf_weighting_eto_floor in the GrassBasis TOML (mm/day)
}
MEMBERS = ["ssebop", "ptjpl"]

THRESHOLDS = {
    "consistency_rtol": 1e-8,
    "low_eto_mm": 1.0,
    "low_eto_pooled_w2_share_max": 0.05,
    "low_eto_pooled_w2_share_increase_max": 0.02,
    "low_eto_site_w2_share_max": 0.25,
    "ceiling_loss_site_frac_max": 0.01,
    "ratio_screen": [0.9, 1.7],
    "ratio_screen_w2_share_review": 0.02,
}
ETO_BINS = [-np.inf, 1.0, 2.0, 3.0, np.inf]
ETO_LABELS = ["<1", "1-2", "2-3", ">=3"]


# --------------------------------------------------------------------------- inputs
def cohort_sites(path: str) -> set[str]:
    with fiona.open(path) as src:
        return {f["properties"]["sid"] for f in src}


def container_table(container: str, sites: set[str], years: tuple[int, int]) -> pd.DataFrame:
    """(site, date, ssebop_container, ptjpl, eto) for every cohort site-date with any member."""
    root = zarr.open(container, mode="r")
    dates = pd.to_datetime(root["time/daily"][:]).strftime("%Y%m%d")
    uids = [str(u) for u in root["geometry/uid"][:]]
    ss = np.asarray(root["remote_sensing/etf/landsat/ssebop/no_mask"][:], float)
    pj = np.asarray(root["remote_sensing/etf/landsat/ptjpl/no_mask"][:], float)
    eto = np.asarray(root["meteorology/era5/eto"][:], float)
    frames = []
    for j, sid in enumerate(uids):
        if sid not in sites:
            continue
        keep = np.isfinite(ss[:, j]) | np.isfinite(pj[:, j])
        frames.append(
            pd.DataFrame(
                {
                    "site": sid,
                    "date": dates[keep],
                    "ssebop_container": ss[keep, j],
                    "ptjpl": pj[keep, j],
                    "eto": eto[keep, j],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["date"].str[:4].astype(int)
    return out[(out["year"] >= years[0]) & (out["year"] <= years[1])].reset_index(drop=True)


def read_grass_csvs(grass_dir: str, sites: set[str], years: tuple[int, int]) -> pd.DataFrame:
    """Melt the written grass-basis CSVs to (site, date, written, grass_csv)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(grass_dir, "ssebop_etf_grass_*_no_mask_*.csv"))):
        m = GRASS_RE.match(os.path.basename(path))
        if not m or m.group("site") not in sites:
            continue
        if not years[0] <= int(m.group("year")) <= years[1]:
            continue
        wide = pd.read_csv(path)
        if len(wide) != 1:
            raise ValueError(f"{path}: expected one row, got {len(wide)}")
        for col in wide.columns[1:]:
            if not (col.startswith("ETF_") and len(col) == 12 and col[4:].isdigit()):
                raise ValueError(f"{path}: unexpected column {col!r}")
            rows.append(
                {
                    "site": m.group("site"),
                    "date": col[4:],
                    "written": float(wide.iloc[0][col]),
                    "grass_csv": path,
                }
            )
    out = pd.DataFrame(rows, columns=["site", "date", "written", "grass_csv"])
    dup = out.duplicated(["site", "date"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate written (site, date):\n{out.loc[dup].head()}")
    return out


# --------------------------------------------------------------------------- pure pieces
def consistency_check(ledger: pd.DataFrame, written: pd.DataFrame, rtol: float) -> dict:
    """``written * eto == native * etr`` for every ok ledger row; one-to-one with the CSVs."""
    ok = ledger[ledger["status"] == "ok"][["site", "date", "native", "eto", "etr", "status"]]
    m = ok.merge(written, on=["site", "date"], how="outer", indicator=True)
    missing = m[m["_merge"] == "left_only"]
    extra = m[m["_merge"] == "right_only"]
    both = m[m["_merge"] == "both"].copy()
    lhs = both["written"] * both["eto"]
    rhs = both["native"] * both["etr"]
    both["rel_err"] = (lhs - rhs).abs() / rhs.abs()
    non_ok_written = ledger[ledger["status"] != "ok"].merge(
        written, on=["site", "date"], how="inner"
    )
    worst = both.sort_values("rel_err", ascending=False).head(5)
    max_rel = float(both["rel_err"].max()) if len(both) else 0.0
    return {
        "n_ok_ledger_rows": int(len(ok)),
        "n_written": int(len(written)),
        "n_compared": int(len(both)),
        "n_ok_rows_missing_from_csvs": int(len(missing)),
        "n_csv_values_without_ok_row": int(len(extra)),
        "n_non_ok_rows_written": int(len(non_ok_written)),
        "max_rel_err": max_rel,
        "worst": worst[["site", "date", "native", "eto", "etr", "written", "rel_err"]].to_dict(
            "records"
        ),
        "pass": bool(
            len(missing) == 0
            and len(extra) == 0
            and len(non_ok_written) == 0
            and len(both) == len(ok)
            and max_rel <= rtol
        ),
    }


def apply_ingest_rules(values: pd.Series, min_etf: float, max_etf: float) -> pd.Series:
    """The ingestor's validity rule: below ``min_etf`` or above ``max_etf`` becomes NaN."""
    v = values.astype(float)
    return v.where((v >= min_etf) & (v <= max_etf))


def ensemble_weights(
    members: pd.DataFrame,
    spread_floor: float,
    min_members: int,
    std_ddof: int = 1,
    eto: pd.Series | None = None,
    eto_floor: float | None = None,
) -> pd.DataFrame:
    """Reproduce ``PestBuilder._write_etf_obs`` for ``etf_weighting_mode = "spread"``.

    ``members`` holds one column per ensemble member (NaN = no retrieval). Target is the member
    mean, SD the sample SD across members (pandas default ddof=1), weight = target / (SD + floor)
    on dates with at least ``min_members`` members, else 0 (the obsval still exists but carries no
    weight). With ``eto_floor`` set, dates whose daily ``eto`` is below the floor are also zero
    weight (``etf_weighting_eto_floor``); the floor is inclusive, as in the builder.
    """
    present = members.notna()
    ct = present.sum(axis=1)
    mean = members.mean(axis=1)
    std = members.std(axis=1, ddof=std_ddof)
    eligible = ct >= min_members
    if eto_floor is not None:
        if eto is None:
            raise ValueError("eto_floor requires the daily eto series")
        eto_vals = pd.Series(eto, index=members.index).astype(float)
        if not np.isfinite(eto_vals).all():
            raise ValueError("daily ETo missing on a member date; cannot apply the ETo floor")
        eto_excluded = eto_vals < eto_floor
        eligible = eligible & ~eto_excluded
    else:
        eto_excluded = pd.Series(False, index=members.index)
    weight = np.where(eligible, mean / (std + spread_floor), 0.0)
    out = pd.DataFrame(
        {
            "target": mean,
            "sd": std,
            "ct": ct,
            "eligible": eligible,
            "eto_floor_excluded": eto_excluded,
            "weight": weight,
        },
        index=members.index,
    )
    out["w2"] = out["weight"] ** 2
    return out


def w2_shares(df: pd.DataFrame, w2_col: str, by: str) -> dict:
    total = float(df[w2_col].sum())
    if total <= 0:
        return {}
    g = df.groupby(by, observed=False)[w2_col].sum()
    return {str(k): float(v / total) for k, v in g.items()}


def eto_floor_effect(table: pd.DataFrame) -> dict:
    """What the ETo floor removed: weighted observations and sum(w**2) share, per configuration."""
    if "weight_native_nofloor" not in table.columns:
        return {"eto_floor_mm": None}
    out = {"eto_floor_mm": WEIGHTING["eto_floor"]}
    for name in ("native", "grass"):
        w0, w = table[f"weight_{name}_nofloor"], table[f"weight_{name}"]
        tot0 = float(table[f"w2_{name}_nofloor"].sum())
        out[name] = {
            "weighted_obs_without_floor": int((w0 > 0).sum()),
            "weighted_obs_with_floor": int((w > 0).sum()),
            "obs_zeroed_by_floor": int(((w0 > 0) & (w == 0)).sum()),
            "w2_share_removed": float(
                (table[f"w2_{name}_nofloor"].sum() - table[f"w2_{name}"].sum()) / tot0
            )
            if tot0
            else 0.0,
        }
    return out


def low_eto_without_floor(table: pd.DataFrame, low: pd.Series, sites: pd.Series) -> dict:
    """The low-ETo shares the gate would see without the floor (the pre-rule diagnostic)."""
    if "w2_native_nofloor" not in table.columns:
        return {}
    tot_nat = float(table["w2_native_nofloor"].sum())
    tot_grass = float(table["w2_grass_nofloor"].sum())
    share_nat = float(table.loc[low, "w2_native_nofloor"].sum() / tot_nat) if tot_nat else 0.0
    share_grass = float(table.loc[low, "w2_grass_nofloor"].sum() / tot_grass) if tot_grass else 0.0
    site_share = (
        table.assign(low_w2=table["w2_grass_nofloor"].where(low, 0.0))
        .groupby("site")[["low_w2", "w2_grass_nofloor"]]
        .sum()
    )
    site_share = (
        site_share["low_w2"] / site_share["w2_grass_nofloor"].replace(0.0, np.nan)
    ).fillna(0.0)
    return {
        "n_weighted_native": int((low & (table["weight_native_nofloor"] > 0)).sum()),
        "n_weighted_grass": int((low & (table["weight_grass_nofloor"] > 0)).sum()),
        "pooled_w2_share_native": share_nat,
        "pooled_w2_share_grass": share_grass,
        "pooled_w2_share_increase": share_grass - share_nat,
        "max_site_w2_share_grass": float(site_share.max()) if len(site_share) else 0.0,
        "max_site": str(site_share.idxmax()) if len(site_share) else None,
    }


def evaluate(table: pd.DataFrame, consistency: dict, thresholds: dict = THRESHOLDS) -> dict:
    """Shares, per-site summaries and the gate verdict from the assembled site-date table."""
    low = table["eto"] < thresholds["low_eto_mm"]
    lo, hi = thresholds["ratio_screen"]
    outside = table["ratio"].notna() & ((table["ratio"] < lo) | (table["ratio"] > hi))

    tot_nat = float(table["w2_native"].sum())
    tot_grass = float(table["w2_grass"].sum())
    low_share_nat = float(table.loc[low, "w2_native"].sum() / tot_nat) if tot_nat else 0.0
    low_share_grass = float(table.loc[low, "w2_grass"].sum() / tot_grass) if tot_grass else 0.0
    outside_share_grass = (
        float(table.loc[outside, "w2_grass"].sum() / tot_grass) if tot_grass else 0.0
    )

    # per site: low-ETo share of the site's own weight; paired dates lost to the ingest ceiling
    site_rows = []
    for site, g in table.groupby("site"):
        s_nat, s_grass = float(g["w2_native"].sum()), float(g["w2_grass"].sum())
        g_low = g["eto"] < thresholds["low_eto_mm"]
        paired_native = int((g["ssebop_container"].notna() & g["ptjpl"].notna()).sum())
        paired_grass = int((g["ssebop_grass"].notna() & g["ptjpl"].notna()).sum())
        lost = int((g["corrected_above_ceiling"] & g["ptjpl"].notna()).sum())
        site_rows.append(
            {
                "site": site,
                "n_dates": int(len(g)),
                "paired_native": paired_native,
                "paired_grass": paired_grass,
                "weighted_native": int((g["weight_native"] > 0).sum()),
                "weighted_grass": int((g["weight_grass"] > 0).sum()),
                "sum_w2_native": s_nat,
                "sum_w2_grass": s_grass,
                "low_eto_dates": int(g_low.sum()),
                "low_eto_weighted_grass": int((g_low & (g["weight_grass"] > 0)).sum()),
                "low_eto_w2_share_native": float(g.loc[g_low, "w2_native"].sum() / s_nat)
                if s_nat
                else 0.0,
                "low_eto_w2_share_grass": float(g.loc[g_low, "w2_grass"].sum() / s_grass)
                if s_grass
                else 0.0,
                "corrected_above_ceiling": int(g["corrected_above_ceiling"].sum()),
                "ceiling_loss_frac_of_paired": float(lost / paired_native)
                if paired_native
                else 0.0,
                "corrected_gt_1": int((g["ssebop_grass"] > 1.0).sum()),
                "mean_weight_native": float(g.loc[g["weight_native"] > 0, "weight_native"].mean())
                if (g["weight_native"] > 0).any()
                else np.nan,
                "mean_weight_grass": float(g.loc[g["weight_grass"] > 0, "weight_grass"].mean())
                if (g["weight_grass"] > 0).any()
                else np.nan,
            }
        )
    by_site = pd.DataFrame(site_rows)

    def weight_stats(mask: pd.Series, col: str) -> dict:
        w = table.loc[mask & (table[col] > 0), col]
        return {
            "n_weighted": int(len(w)),
            "median_weight": float(w.median()) if len(w) else None,
            "p95_weight": float(w.quantile(0.95)) if len(w) else None,
        }

    summary = {
        "n_site_dates": int(len(table)),
        "n_sites": int(table["site"].nunique()),
        "weighted_obs_native": int((table["weight_native"] > 0).sum()),
        "weighted_obs_grass": int((table["weight_grass"] > 0).sum()),
        "sum_w2_native": tot_nat,
        "sum_w2_grass": tot_grass,
        "w2_share_by_eto_bin": {
            "native": w2_shares(table, "w2_native", "eto_bin"),
            "grass": w2_shares(table, "w2_grass", "eto_bin"),
        },
        "weighted_obs_by_eto_bin": {
            "native": table[table["weight_native"] > 0]
            .groupby("eto_bin", observed=False)
            .size()
            .to_dict(),
            "grass": table[table["weight_grass"] > 0]
            .groupby("eto_bin", observed=False)
            .size()
            .to_dict(),
        },
        "w2_share_by_month": {
            "native": w2_shares(table, "w2_native", "month"),
            "grass": w2_shares(table, "w2_grass", "month"),
        },
        "low_eto": {
            "n_site_dates": int(low.sum()),
            "n_weighted_native": int((low & (table["weight_native"] > 0)).sum()),
            "n_weighted_grass": int((low & (table["weight_grass"] > 0)).sum()),
            "pooled_w2_share_native": low_share_nat,
            "pooled_w2_share_grass": low_share_grass,
            "pooled_w2_share_increase": low_share_grass - low_share_nat,
            "weights_native": weight_stats(low, "weight_native"),
            "weights_grass": weight_stats(low, "weight_grass"),
            "weights_grass_eto_ge_3": weight_stats(table["eto"] >= 3.0, "weight_grass"),
            "max_site_w2_share_grass": float(by_site["low_eto_w2_share_grass"].max()),
            "max_site": by_site.sort_values("low_eto_w2_share_grass", ascending=False)
            .head(5)[["site", "low_eto_w2_share_grass", "low_eto_weighted_grass"]]
            .to_dict("records"),
        },
        "eto_floor": eto_floor_effect(table),
        "low_eto_without_floor": low_eto_without_floor(table, low, by_site["site"]),
        "ratio_screen": {
            "n_outside": int(outside.sum()),
            "n_outside_weighted_grass": int((outside & (table["weight_grass"] > 0)).sum()),
            "pooled_w2_share_grass": outside_share_grass,
            "max_ratio_weighted": float(table.loc[table["weight_grass"] > 0, "ratio"].max())
            if (table["weight_grass"] > 0).any()
            else None,
        },
        "ingest_range": {
            "corrected_above_ceiling": int(table["corrected_above_ceiling"].sum()),
            "corrected_above_ceiling_paired": int(
                (table["corrected_above_ceiling"] & table["ptjpl"].notna()).sum()
            ),
            "corrected_above_ceiling_rows": table.loc[
                table["corrected_above_ceiling"],
                ["site", "date", "ssebop_container", "ratio", "eto", "corrected"],
            ].to_dict("records"),
            "rose_above_min_etf": int(table["rose_above_min"].sum()),
            "fell_below_min_etf": int(table["fell_below_min"].sum()),
            "max_site_ceiling_loss_frac": float(by_site["ceiling_loss_frac_of_paired"].max()),
            "corrected_gt_1_share_of_grass_member": float(
                (table["ssebop_grass"] > 1.0).sum() / table["ssebop_grass"].notna().sum()
            ),
        },
    }
    gate = {
        "daily_et_consistency": bool(consistency["pass"]),
        "low_eto_pooled_share_ok": low_share_grass <= thresholds["low_eto_pooled_w2_share_max"],
        "low_eto_pooled_increase_ok": (low_share_grass - low_share_nat)
        <= thresholds["low_eto_pooled_w2_share_increase_max"],
        "low_eto_site_share_ok": bool(
            (by_site["low_eto_w2_share_grass"] <= thresholds["low_eto_site_w2_share_max"]).all()
        ),
        "ceiling_loss_ok": bool(
            (
                by_site["ceiling_loss_frac_of_paired"] <= thresholds["ceiling_loss_site_frac_max"]
            ).all()
        ),
    }
    gate["pass"] = all(gate.values())
    gate["ratio_screen_review_needed"] = (
        outside_share_grass > thresholds["ratio_screen_w2_share_review"]
    )
    return {"summary": summary, "gate": gate, "by_site": by_site}


def assemble(members: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Join the container members with the conversion ledger and compute both weightings."""
    led = ledger[["site", "date", "native", "corrected", "ratio", "etr", "status"]].rename(
        columns={"eto": "eto_ledger"}
    )
    t = members.merge(led, on=["site", "date"], how="left")
    # ledger native must equal the container member wherever both exist (ingest rules aside)
    both = t["native"].notna() & t["ssebop_container"].notna()
    t["native_dev"] = np.where(both, (t["native"] - t["ssebop_container"]).abs(), np.nan)

    corrected = t["corrected"].where(t["status"] == "ok")
    t["ssebop_grass"] = apply_ingest_rules(corrected, INGEST["min_etf"], INGEST["max_etf"])
    t["corrected_above_ceiling"] = corrected.notna() & (corrected > INGEST["max_etf"])
    t["rose_above_min"] = (
        t["native"].notna()
        & (t["native"] < INGEST["min_etf"])
        & (corrected >= INGEST["min_etf"])
        & (corrected <= INGEST["max_etf"])
    )
    t["fell_below_min"] = (
        t["ssebop_container"].notna() & corrected.notna() & (corrected < INGEST["min_etf"])
    )

    for name, cols in (
        ("native", ["ssebop_container", "ptjpl"]),
        ("grass", ["ssebop_grass", "ptjpl"]),
    ):
        w = ensemble_weights(
            t[cols],
            WEIGHTING["spread_floor"],
            WEIGHTING["min_members"],
            WEIGHTING["std_ddof"],
            eto=t["eto"],
            eto_floor=WEIGHTING["eto_floor"],
        )
        for col in ("target", "sd", "ct", "weight", "w2"):
            t[f"{col}_{name}"] = w[col].values
        # the same weights without the ETo floor, so the rule's effect stays visible
        w0 = ensemble_weights(
            t[cols], WEIGHTING["spread_floor"], WEIGHTING["min_members"], WEIGHTING["std_ddof"]
        )
        t[f"weight_{name}_nofloor"] = w0["weight"].values
        t[f"w2_{name}_nofloor"] = w0["w2"].values
    t["eto_floor_excluded"] = w["eto_floor_excluded"].values

    t["month"] = t["date"].str[4:6].astype(int)
    t["eto_bin"] = pd.cut(t["eto"], ETO_BINS, labels=ETO_LABELS, right=False)
    t["low_eto"] = t["eto"] < THRESHOLDS["low_eto_mm"]
    lo, hi = THRESHOLDS["ratio_screen"]
    t["ratio_outside_screen"] = t["ratio"].notna() & ((t["ratio"] < lo) | (t["ratio"] > hi))
    return t


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--grass-dir", default=GRASS_DIR)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--out-dir", default=QA_ROOT)
    args = ap.parse_args(argv)

    sites = cohort_sites(COHORT_66)
    ledger = pd.read_csv(args.ledger, dtype={"date": str})
    ledger = ledger[ledger["site"].isin(sites)]
    written = read_grass_csvs(args.grass_dir, sites, YEARS)
    consistency = consistency_check(ledger, written, THRESHOLDS["consistency_rtol"])

    members = container_table(args.container, sites, YEARS)
    table = assemble(members, ledger)
    result = evaluate(table, consistency)

    result["summary"]["native_vs_container_max_dev"] = float(np.nanmax(table["native_dev"]))
    result["summary"]["container_ssebop_without_ledger_row"] = int(
        (table["ssebop_container"].notna() & table["native"].isna()).sum()
    )
    report = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "definition": "G3 redefined 2026-09-04: daily ET consistency + low-ETo stability of the "
        "weighted observations; NHM comparison is a separate diagnostic",
        "thresholds": THRESHOLDS,
        "ingest_rules": INGEST,
        "weighting": WEIGHTING | {"members": MEMBERS, "cohort": COHORT_66},
        "inputs": {
            "ledger": args.ledger,
            "grass_dir": args.grass_dir,
            "container": args.container,
            "years": list(YEARS),
        },
        "consistency": consistency,
        **{k: v for k, v in result.items() if k != "by_site"},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    cols = [
        "site",
        "date",
        "year",
        "month",
        "eto",
        "eto_bin",
        "low_eto",
        "etr",
        "ratio",
        "ratio_outside_screen",
        "ssebop_container",
        "native",
        "corrected",
        "status",
        "ssebop_grass",
        "corrected_above_ceiling",
        "rose_above_min",
        "fell_below_min",
        "ptjpl",
        "target_native",
        "sd_native",
        "ct_native",
        "weight_native",
        "w2_native",
        "target_grass",
        "sd_grass",
        "ct_grass",
        "weight_grass",
        "w2_grass",
        "eto_floor_excluded",
        "weight_native_nofloor",
        "w2_native_nofloor",
        "weight_grass_nofloor",
        "w2_grass_nofloor",
    ]
    table.sort_values(["site", "date"])[cols].to_csv(
        os.path.join(args.out_dir, "daily_basis_gate_weights.csv"), index=False, float_format="%.8g"
    )
    result["by_site"].to_csv(
        os.path.join(args.out_dir, "daily_basis_gate_by_site.csv"), index=False, float_format="%.6g"
    )
    with open(os.path.join(args.out_dir, "daily_basis_gate_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    show = {
        "consistency": {k: v for k, v in consistency.items() if k != "worst"},
        "low_eto": {
            k: v for k, v in report["summary"]["low_eto"].items() if not isinstance(v, list)
        },
        "w2_share_by_eto_bin": report["summary"]["w2_share_by_eto_bin"],
        "ratio_screen": report["summary"]["ratio_screen"],
        "ingest_range": {
            k: v for k, v in report["summary"]["ingest_range"].items() if not isinstance(v, list)
        },
        "gate": report["gate"],
    }
    print(json.dumps(show, indent=2, default=str))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

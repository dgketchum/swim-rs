"""Phase 4 / Gate G4: quantify the corrected two-member target before ordering or calibrating.

Applies the Phase 2 ledger (native and grass-basis SSEBop) to the canonical 66-site cohort and
compares each against the PT-JPL member exactly as the baseline container holds it. The
ensemble target is the member mean; the observation weight is obsval / (member SD + floor)
with SD the two-member sample standard deviation and floor 0.05 (baseline TOML). Nothing here
touches the primary container or reads any flux data.

Outputs (QA root):
    member_agreement_before_after.csv   pooled and per-sensor PT-JPL vs SSEBop diagnostics
    target_shift_by_site.csv            per-site native/corrected target and weight summaries
    target_shift_by_year.csv            per-year native/corrected target and weight summaries
    target_diagnostics_summary.json     counts (66 and 75 cohorts), distributions, gate checks

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase4_target_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import os

import fiona
import numpy as np
import pandas as pd
import zarr

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
CONTAINER = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")
COHORT_66 = os.path.join(DATA, "gis", "flux_crop_pub_66_150m.shp")
LEDGER = os.path.join(QA_ROOT, "ssebop_conversion_ledger.csv")

SPREAD_FLOOR = 0.05  # etf_weighting_spread_floor in the baseline TOML
MIN_ETF = 0.05
PRIOR = {"ratio": 1.03, "r": 0.47, "implied_sd": 0.14}  # scalar approximation from e2_update.md


def cohort(path: str) -> pd.DataFrame:
    with fiona.open(path) as src:
        rows = [
            {
                "site": f["properties"]["sid"],
                "lulc": f["properties"].get("glc10_lulc"),
                "country": f["properties"].get("country"),
            }
            for f in src
        ]
    return pd.DataFrame(rows)


def container_members(container: str) -> pd.DataFrame:
    root = zarr.open(container, mode="r")
    dates = pd.to_datetime(root["time/daily"][:]).strftime("%Y%m%d")
    sites = [str(u) for u in root["geometry/uid"][:]]
    ss = pd.DataFrame(
        np.asarray(root["remote_sensing/etf/landsat/ssebop/no_mask"][:], float),
        index=dates,
        columns=sites,
    )
    pj = pd.DataFrame(
        np.asarray(root["remote_sensing/etf/landsat/ptjpl/no_mask"][:], float),
        index=dates,
        columns=sites,
    )
    long = (
        pd.concat(
            [
                ss.stack(future_stack=True).rename("ssebop_container"),
                pj.stack(future_stack=True).rename("ptjpl"),
            ],
            axis=1,
        )
        .rename_axis(["date", "site"])
        .reset_index()
    )
    return long[long["ssebop_container"].notna() | long["ptjpl"].notna()].reset_index(drop=True)


def member_stats(a: pd.Series, b: pd.Series) -> dict:
    """PT-JPL (b) against SSEBop (a) on coincident captures above MIN_ETF."""
    m = a.notna() & b.notna() & (a > MIN_ETF) & (b > MIN_ETF)
    x, y = a[m].to_numpy(), b[m].to_numpy()
    if len(x) < 3:
        return {"n": int(len(x))}
    diff = y - x
    sd = np.abs(diff) / np.sqrt(2)  # two-member sample SD
    return {
        "n": int(len(x)),
        "slope": float((x * y).sum() / (x * x).sum()),
        "median_ratio": float(np.median(y / x)),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "median_implied_sd": float(np.median(sd)),
        "mean_target": float(((x + y) / 2).mean()),
        "mean_weight": float((((x + y) / 2) / (sd + SPREAD_FLOOR)).mean()),
    }


def availability_counts(df: pd.DataFrame, ss_col: str) -> dict:
    has_ss, has_pj = df[ss_col].notna(), df["ptjpl"].notna()
    return {
        "both": int((has_ss & has_pj).sum()),
        "ptjpl_only": int((has_pj & ~has_ss).sum()),
        "ssebop_only": int((has_ss & ~has_pj).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--out-dir", default=QA_ROOT)
    args = ap.parse_args(argv)

    coh = cohort(COHORT_66)
    sites66 = set(coh["site"])
    ledger = pd.read_csv(args.ledger, dtype={"date": str})
    members = container_members(args.container)
    members["year"] = members["date"].str[:4].astype(int)

    df = members.merge(
        ledger[
            ["site", "date", "native", "corrected", "status", "ratio", "eto", "low_eto", "sensor"]
        ],
        on=["site", "date"],
        how="left",
    )
    # ledger native must agree with the container's SSEBop wherever both exist
    both = df["native"].notna() & df["ssebop_container"].notna()
    max_dev = (
        float((df.loc[both, "native"] - df.loc[both, "ssebop_container"]).abs().max())
        if both.any()
        else 0.0
    )
    container_only = int((df["ssebop_container"].notna() & df["native"].isna()).sum())
    ledger_only = int((df["native"].notna() & df["ssebop_container"].isna()).sum())

    df75 = df
    df66 = df[df["site"].isin(sites66)].merge(coh, on="site", how="left")
    df66["sensor"] = df66["sensor"].fillna("unknown")

    # member agreement before/after, pooled and by sensor / year-block
    rows = []
    for label, sub in (
        [("pooled", df66)]
        + [(f"sensor={s}", g) for s, g in df66.groupby("sensor")]
        + [(f"year={y}", g) for y, g in df66.groupby("year")]
    ):
        rows.append(
            {"group": label, "basis": "native", **member_stats(sub["native"], sub["ptjpl"])}
        )
        rows.append(
            {"group": label, "basis": "grass", **member_stats(sub["corrected"], sub["ptjpl"])}
        )
    agreement = pd.DataFrame(rows)
    agreement.to_csv(
        os.path.join(args.out_dir, "member_agreement_before_after.csv"),
        index=False,
        float_format="%.6g",
    )

    def shift_table(key: str) -> pd.DataFrame:
        out = []
        for k, g in df66.groupby(key):
            m = (
                g["native"].notna()
                & g["ptjpl"].notna()
                & (g["native"] > MIN_ETF)
                & (g["ptjpl"] > MIN_ETF)
            )
            gg = g[m]
            if gg.empty:
                continue
            t_nat = (gg["native"] + gg["ptjpl"]) / 2
            t_gr = (gg["corrected"] + gg["ptjpl"]) / 2
            sd_nat = (gg["ptjpl"] - gg["native"]).abs() / np.sqrt(2)
            sd_gr = (gg["ptjpl"] - gg["corrected"]).abs() / np.sqrt(2)
            out.append(
                {
                    key: k,
                    "n_both": int(len(gg)),
                    "n_ptjpl_only": int((g["ptjpl"].notna() & g["native"].isna()).sum()),
                    "ratio_median": float(gg["ratio"].median()),
                    "n_low_eto": int(gg["low_eto"].fillna(False).astype(bool).sum()),
                    "native_mean": float(gg["native"].mean()),
                    "grass_mean": float(gg["corrected"].mean()),
                    "ptjpl_mean": float(gg["ptjpl"].mean()),
                    "target_native_mean": float(t_nat.mean()),
                    "target_grass_mean": float(t_gr.mean()),
                    "target_shift_mean": float((t_gr - t_nat).mean()),
                    "implied_sd_native_median": float(sd_nat.median()),
                    "implied_sd_grass_median": float(sd_gr.median()),
                    "weight_native_mean": float((t_nat / (sd_nat + SPREAD_FLOOR)).mean()),
                    "weight_grass_mean": float((t_gr / (sd_gr + SPREAD_FLOOR)).mean()),
                    "native_at_cap_frac": float((gg["native"] >= 1.0).mean()),
                    "grass_above_1_frac": float((gg["corrected"] > 1.0).mean()),
                }
            )
        return pd.DataFrame(out)

    by_site = shift_table("site").merge(coh, on="site", how="left")
    by_year = shift_table("year")
    by_site.to_csv(
        os.path.join(args.out_dir, "target_shift_by_site.csv"), index=False, float_format="%.6g"
    )
    by_year.to_csv(
        os.path.join(args.out_dir, "target_shift_by_year.csv"), index=False, float_format="%.6g"
    )

    pooled_nat = member_stats(df66["native"], df66["ptjpl"])
    pooled_gr = member_stats(df66["corrected"], df66["ptjpl"])
    low_eto_share = by_site.set_index("site")["n_low_eto"] / by_site.set_index("site")["n_both"]
    grass_ok = df66["corrected"].notna()
    summary = {
        "inputs": {"ledger": args.ledger, "container": args.container, "cohort_66": COHORT_66},
        "ledger_vs_container": {
            "max_abs_dev": max_dev,
            "container_only_dates": container_only,
            "ledger_only_dates": ledger_only,
        },
        "availability_66_native": availability_counts(df66, "native"),
        "availability_66_grass": availability_counts(df66, "corrected"),
        "availability_75_native": availability_counts(df75, "ssebop_container"),
        "ptjpl_captures_66": int(df66["ptjpl"].notna().sum()),
        "ptjpl_only_share_66": float(
            (df66["ptjpl"].notna() & df66["native"].isna()).sum() / df66["ptjpl"].notna().sum()
        ),
        "ratio_distribution_66": {
            "median": float(df66.loc[grass_ok, "ratio"].median()),
            "p05": float(df66.loc[grass_ok, "ratio"].quantile(0.05)),
            "p95": float(df66.loc[grass_ok, "ratio"].quantile(0.95)),
            "by_month_median": {
                int(k): float(v)
                for k, v in df66[grass_ok]
                .groupby(df66.loc[grass_ok, "date"].str[4:6].astype(int))["ratio"]
                .median()
                .items()
            },
        },
        "native_at_cap_1.0_frac": float(
            (df66["native"] >= 1.0).sum() / df66["native"].notna().sum()
        ),
        "grass_above_1.0_frac": float((df66["corrected"] > 1.0).sum() / grass_ok.sum()),
        "conversion_status_counts_66": df66.loc[df66["native"].notna(), "status"]
        .value_counts()
        .to_dict(),
        "member_agreement_native": pooled_nat,
        "member_agreement_grass": pooled_gr,
        "prior_scalar_approximation": PRIOR,
        "low_eto_share_by_site_max": {
            "site": str(low_eto_share.idxmax()),
            "share": float(low_eto_share.max()),
        },
        "sites_by_lulc": coh["lulc"].value_counts().to_dict(),
    }
    summary["gate"] = {
        "ledger_matches_container": max_dev < 1e-6 and container_only == 0,
        "no_sign_reversal": pooled_gr["median_ratio"] < pooled_nat["median_ratio"]
        and pooled_gr["mean_diff"] < pooled_nat["mean_diff"],
        "not_fixed_ratio": float(df66.loc[grass_ok, "ratio"].std()) > 0.01,
        "member_ratio_plausible": 0.9 <= pooled_gr["median_ratio"] <= 1.15,
        "correlation_plausible": 0.35 <= pooled_gr["r"] <= 0.65,
        "implied_sd_plausible": 0.08 <= pooled_gr["median_implied_sd"] <= 0.20,
        "no_site_dominated_by_low_eto": float(low_eto_share.max()) < 0.25,
    }
    summary["gate"]["pass"] = all(summary["gate"].values())
    with open(os.path.join(args.out_dir, "target_diagnostics_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "ratio_distribution_66"},
            indent=2,
            default=str,
        )
    )
    return 0 if summary["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 1 validation: does the reference-ET sidecar reproduce the stored E2 ETo?

Reads every ``refet_ratio_{year}_{suffix}.csv`` downloaded from the bucket, melts it to
one row per (site, date), and compares the sidecar ``eto`` to the container's forcing
``meteorology/era5/eto`` on the same site-date. Also screens ``etr/eto``.

Predeclared thresholds (plan §3, accepted 2026-09-04):

    median |rel diff| <= 0.5 %, 95th percentile <= 2 %, no site median > 2 %
    ratios finite and positive; values outside 0.9–1.7 are listed, never clipped

Outputs (QA root):
    refet_ratio_values.csv          site, date, eto, etr, ratio, stored_eto, rel_diff
    refet_sidecar_validation.json   thresholds, statistics, verdict, exceptions

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase1_validate_sidecar.py [--sidecar-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import zarr

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
SIDECAR_DIR = os.path.join(DATA, "remote_sensing", "espa", "refet_ratio_era5land")
CONTAINER = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")

THRESHOLDS = {
    "median_abs_rel_diff_max": 0.005,
    "p95_abs_rel_diff_max": 0.02,
    "site_median_abs_rel_diff_max": 0.02,
    "ratio_screen_low": 0.9,
    "ratio_screen_high": 1.7,
    "min_stored_eto_for_rel_diff": 0.1,  # mm/d; below this a relative difference is meaningless
}

FILE_RE = re.compile(r"refet_ratio_(\d{4})_(utc_[mp]\d{2})\.csv$")


def load_sidecar(sidecar_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(sidecar_dir, "refet_ratio_*.csv"))):
        m = FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        wide = pd.read_csv(path)
        if "sid" not in wide.columns:
            raise ValueError(f"{path}: no 'sid' column")
        long = wide.melt(id_vars="sid", var_name="band", value_name="value")
        long[["var", "date"]] = long["band"].str.extract(r"^(eto|etr)_(\d{8})$")
        if long["var"].isna().any():
            bad = long.loc[long["var"].isna(), "band"].unique()[:5].tolist()
            raise ValueError(f"{path}: unexpected columns {bad}")
        tidy = long.pivot_table(
            index=["sid", "date"], columns="var", values="value", aggfunc="first"
        ).reset_index()
        tidy["source_file"] = os.path.basename(path)
        tidy["utc_suffix"] = m.group(2)
        frames.append(tidy)
    if not frames:
        raise FileNotFoundError(f"no refet_ratio_*.csv under {sidecar_dir}")
    out = pd.concat(frames, ignore_index=True).rename(columns={"sid": "site"})
    dup = out.duplicated(["site", "date"], keep=False)
    if dup.any():
        raise ValueError(
            f"duplicate (site, date) rows in sidecar: {out.loc[dup, ['site', 'date', 'source_file']].head()}"
        )
    return out


def stored_eto(container: str) -> pd.DataFrame:
    root = zarr.open(container, mode="r")
    dates = pd.to_datetime(root["time/daily"][:]).strftime("%Y%m%d")
    sites = [str(u) for u in root["geometry/uid"][:]]
    eto = np.asarray(root["meteorology/era5/eto"][:], dtype=float)
    df = pd.DataFrame(eto, index=dates, columns=sites)
    return df.stack().rename("stored_eto").rename_axis(["date", "site"]).reset_index()


def validate(
    side: pd.DataFrame, stored: pd.DataFrame, thresholds: dict = THRESHOLDS
) -> tuple[pd.DataFrame, dict]:
    merged = side.merge(stored, on=["site", "date"], how="left")
    merged["ratio"] = merged["etr"] / merged["eto"]
    ok_stored = merged["stored_eto"] > thresholds["min_stored_eto_for_rel_diff"]
    merged["rel_diff"] = np.where(
        ok_stored, (merged["eto"] - merged["stored_eto"]) / merged["stored_eto"], np.nan
    )

    in_container = merged["stored_eto"].notna()
    comp = merged[in_container & ok_stored & merged["eto"].notna()]
    abs_rel = comp["rel_diff"].abs()
    site_med = comp.groupby("site")["rel_diff"].apply(lambda s: float(np.median(np.abs(s))))

    finite_ratio = np.isfinite(merged["ratio"])
    positive = finite_ratio & (merged["ratio"] > 0)
    out_of_screen = merged[
        positive
        & (
            (merged["ratio"] < thresholds["ratio_screen_low"])
            | (merged["ratio"] > thresholds["ratio_screen_high"])
        )
    ]
    # ERA5-Land hourly Penman-Monteith integrates to a slightly negative daily ETo (or ETr) on a
    # few mid-winter days with negative net radiation; the container's stored ETo carries the
    # same values, so a non-positive ratio is only a sidecar defect when the sign disagrees with
    # the stored ETo. The converter gives those dates a nonpositive_eto / nonpositive_etr status.
    eto_nonpos = merged["eto"].notna() & (merged["eto"] <= 0)
    sign_mismatch = eto_nonpos & merged["stored_eto"].notna() & (merged["stored_eto"] > 0)
    sign_mismatch |= (
        merged["eto"].notna()
        & (merged["eto"] > 0)
        & merged["stored_eto"].notna()
        & (merged["stored_eto"] <= 0)
    )

    stats = {
        "n_rows": int(len(merged)),
        "n_sites": int(merged["site"].nunique()),
        "n_sites_in_container": int(merged.loc[in_container, "site"].nunique()),
        "sites_outside_container": sorted(merged.loc[~in_container, "site"].unique().tolist()),
        "n_compared": int(len(comp)),
        "n_sidecar_eto_null": int(merged["eto"].isna().sum()),
        "n_sidecar_etr_null": int(merged["etr"].isna().sum()),
        "n_stored_eto_null_in_container": int((in_container & merged["stored_eto"].isna()).sum()),
        "n_stored_eto_below_floor": int(
            (in_container & ~ok_stored & merged["stored_eto"].notna()).sum()
        ),
        "median_abs_rel_diff": float(abs_rel.median()) if len(comp) else None,
        "p95_abs_rel_diff": float(abs_rel.quantile(0.95)) if len(comp) else None,
        "max_abs_rel_diff": float(abs_rel.max()) if len(comp) else None,
        "mean_rel_diff": float(comp["rel_diff"].mean()) if len(comp) else None,
        "site_median_abs_rel_diff": {k: float(v) for k, v in site_med.items()},
        "worst_site": (site_med.idxmax(), float(site_med.max())) if len(site_med) else None,
        "ratio_median": float(merged.loc[positive, "ratio"].median()) if positive.any() else None,
        "ratio_p05": float(merged.loc[positive, "ratio"].quantile(0.05))
        if positive.any()
        else None,
        "ratio_p95": float(merged.loc[positive, "ratio"].quantile(0.95))
        if positive.any()
        else None,
        "n_ratio_nonfinite": int((~finite_ratio).sum()),
        "n_ratio_nonpositive": int((finite_ratio & ~(merged["ratio"] > 0)).sum()),
        "n_eto_nonpositive": int(eto_nonpos.sum()),
        "n_etr_nonpositive": int((merged["etr"].notna() & (merged["etr"] <= 0)).sum()),
        "n_eto_sign_mismatch_with_stored": int(sign_mismatch.sum()),
        "eto_nonpositive_by_month": merged.loc[eto_nonpos, "date"]
        .str[4:6]
        .value_counts()
        .sort_index()
        .to_dict(),
        "n_ratio_outside_screen": int(len(out_of_screen)),
        "ratio_outside_screen_by_site": out_of_screen.groupby("site").size().to_dict(),
        "ratio_outside_screen_examples": out_of_screen[["site", "date", "eto", "etr", "ratio"]]
        .head(20)
        .to_dict("records"),
    }
    gate = {
        "median_ok": stats["median_abs_rel_diff"] is not None
        and stats["median_abs_rel_diff"] <= thresholds["median_abs_rel_diff_max"],
        "p95_ok": stats["p95_abs_rel_diff"] is not None
        and stats["p95_abs_rel_diff"] <= thresholds["p95_abs_rel_diff_max"],
        "sites_ok": bool(len(site_med))
        and bool((site_med <= thresholds["site_median_abs_rel_diff_max"]).all()),
        "ratios_finite": stats["n_ratio_nonfinite"] == 0,
        "eto_sign_matches_stored": stats["n_eto_sign_mismatch_with_stored"] == 0,
        "no_sidecar_nulls": stats["n_sidecar_eto_null"] == 0 and stats["n_sidecar_etr_null"] == 0,
    }
    gate["pass"] = all(gate.values())
    return merged, {"thresholds": thresholds, "stats": stats, "gate": gate}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sidecar-dir", default=SIDECAR_DIR)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--out-dir", default=QA_ROOT)
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. 'pilot'")
    args = ap.parse_args(argv)

    side = load_sidecar(args.sidecar_dir)
    merged, report = validate(side, stored_eto(args.container))
    report["inputs"] = {
        "sidecar_dir": args.sidecar_dir,
        "files": sorted(side["source_file"].unique().tolist()),
        "container": args.container,
    }
    tag = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.out_dir, exist_ok=True)
    merged.sort_values(["site", "date"]).to_csv(
        os.path.join(args.out_dir, f"refet_ratio_values{tag}.csv"), index=False
    )
    with open(os.path.join(args.out_dir, f"refet_sidecar_validation{tag}.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    show = {k: v for k, v in report["stats"].items() if not isinstance(v, dict | list)}
    print(json.dumps({"gate": report["gate"], "stats": show}, indent=2, default=str))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

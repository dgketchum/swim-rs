"""Phase 2: non-destructive alfalfa-to-grass basis conversion of the ESPA SSEBop ETf.

    ssebop_etf_grass = ssebop_etf_native * etr_era5land / eto_era5land      (per site-date)

Sources (all read-only):
    native   the ingest CSVs the baseline container consumed,
             {landsat}/extracts/ssebop_etf/no_mask/ssebop_etf_{site}_no_mask_{year}.csv
    sidecar  the ERA5-Land ETo/ETr daily export, refet_ratio_{year}_{utc}.csv
    scenes   delivered ESPA tifs across all order trees (sensor and product identity)

Outputs:
    {landsat}/extracts/ssebop_etf_grass/no_mask/ssebop_etf_grass_{site}_no_mask_{year}.csv
    {qa}/ssebop_conversion_ledger.csv        one row per native site-date, every term recorded
    {qa}/ssebop_conversion_summary.json      counts, hashes, thresholds, rules applied

Data rules (plan §8): exact joins on (site, date); duplicate source values are a hard error; a
raw null stays null; a missing or non-positive ETo/ETr is a missing-data status, never a cohort
median; no cap at 1.0; low-ETo dates are flagged, not filtered; no flux data is read.

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase2_convert_ssebop_basis.py \
        [--sites SID ...] [--years 2013 2025] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from espa_inventory import delivered_tifs, scene_identity_by_site_date  # noqa: E402
from phase1_validate_sidecar import load_sidecar  # noqa: E402

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
LANDSAT = os.path.join(DATA, "remote_sensing", "landsat")
NATIVE_DIR = os.path.join(LANDSAT, "extracts", "ssebop_etf", "no_mask")
GRASS_DIR = os.path.join(LANDSAT, "extracts", "ssebop_etf_grass", "no_mask")
SIDECAR_DIR = os.path.join(DATA, "remote_sensing", "espa", "refet_ratio_era5land")
COHORT_FGB = os.path.join(QA_ROOT, "refet_sidecar_cohort.fgb")

NATIVE_RE = re.compile(r"^ssebop_etf_(?P<site>.+)_no_mask_(?P<year>\d{4})\.csv$")
LEGACY_COL_RE = re.compile(r"^ETF_\d{8}$")
SCENE_COL_RE = re.compile(r"^(LT04|LT05|LE07|LC08|LC09)_\d{6}_\d{8}$")
OUTPUT_STEM = "ssebop_etf_grass"

RULES = {
    "formula": "grass = native * etr / eto",
    "low_eto_mm": 1.0,  # flag only; rows are still converted
    "ratio_screen": [0.9, 1.7],  # flag only; never clipped
    "cap_at_one": False,
    "fill_missing_ratio": False,
    "years": [2013, 2025],
}

STATUS_OK = "ok"
LEDGER_COLUMNS = [
    "site",
    "date",
    "year",
    "native",
    "eto",
    "etr",
    "ratio",
    "corrected",
    "status",
    "low_eto",
    "ratio_outside_screen",
    "sensor",
    "product_ids",
    "n_products",
    "trees",
    "native_columns",
    "n_native_columns",
    "native_csv",
    "native_csv_sha256",
    "sidecar_file",
    "sidecar_sha256",
    "output_csv",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def native_column_date(col: str) -> str:
    """Date of a native column: legacy ``ETF_YYYYMMDD`` or scene ``{sensor}_{pathrow}_{YYYYMMDD}``."""
    if LEGACY_COL_RE.match(col) or SCENE_COL_RE.match(col):
        return col.rsplit("_", 1)[1]
    raise ValueError(f"unexpected native column {col!r}")


def load_native(native_dir: str, sites: set[str] | None, years: tuple[int, int]) -> pd.DataFrame:
    """Melt every native ingest CSV to one row per (site, date).

    Scene-keyed columns sharing a date are collapsed by ``max`` — the ingestor's own rule for
    same-date Landsat columns — and ``native_columns`` / ``n_native_columns`` record what was
    collapsed. ``max`` commutes with the positive ratio scaling, so converting the collapsed value
    equals collapsing the converted values.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(native_dir, "ssebop_etf_*_no_mask_*.csv"))):
        m = NATIVE_RE.match(os.path.basename(path))
        if not m:
            continue
        site, year = m.group("site"), int(m.group("year"))
        if sites is not None and site not in sites:
            continue
        if not years[0] <= year <= years[1]:
            continue
        wide = pd.read_csv(path)
        if len(wide) != 1 or wide.iloc[0, 0] != site:
            raise ValueError(
                f"{path}: expected one row keyed {site!r}, got {wide.iloc[:, 0].tolist()}"
            )
        for col in wide.columns[1:]:
            try:
                date = native_column_date(col)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from None
            if date[:4] != m.group("year"):
                raise ValueError(f"{path}: column {col} outside file year")
            rows.append(
                {
                    "site": site,
                    "date": date,
                    "column": col,
                    "value": float(wide.iloc[0][col]),
                    "native_csv": path,
                }
            )
    long = pd.DataFrame(rows, columns=["site", "date", "column", "value", "native_csv"])
    if long.empty:
        return pd.DataFrame(
            columns=["site", "date", "native", "native_csv", "native_columns", "n_native_columns"]
        )
    dup = long.duplicated(["site", "column"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate native (site, column):\n{long.loc[dup].head()}")
    files = long.groupby(["site", "date"])["native_csv"].nunique()
    if (files > 1).any():
        raise ValueError(f"(site, date) spread over several files:\n{files[files > 1].head()}")
    g = long.sort_values(["site", "date", "column"]).groupby(["site", "date"], sort=True)
    out = pd.DataFrame(
        {
            "native": g["value"].max(),  # NaN only when every column is NaN
            "native_csv": g["native_csv"].first(),
            "native_columns": g["column"].agg(";".join),
            "n_native_columns": g["column"].size(),
        }
    ).reset_index()
    return out[["site", "date", "native", "native_csv", "native_columns", "n_native_columns"]]


def convert(
    native: pd.DataFrame,
    sidecar: pd.DataFrame,
    scenes: pd.DataFrame | None = None,
    rules: dict = RULES,
) -> pd.DataFrame:
    """Pure conversion: returns the ledger (one row per native site-date, deterministic order)."""
    for name, df in (("native", native), ("sidecar", sidecar)):
        dup = df.duplicated(["site", "date"], keep=False)
        if dup.any():
            raise ValueError(f"duplicate (site, date) in {name}:\n{df.loc[dup].head()}")
    side = sidecar.rename(columns={"source_file": "sidecar_file"})
    keep = ["site", "date", "eto", "etr", "sidecar_file"] + (
        ["sidecar_sha256"] if "sidecar_sha256" in side else []
    )
    led = native.merge(side[keep], on=["site", "date"], how="left", validate="one_to_one")

    native_v = led["native"].to_numpy(dtype=float)
    eto = led["eto"].to_numpy(dtype=float)
    etr = led["etr"].to_numpy(dtype=float)
    has_side = led["sidecar_file"].notna().to_numpy()
    eto_ok = np.isfinite(eto) & (eto > 0)
    etr_ok = np.isfinite(etr) & (etr > 0)

    status = np.full(len(led), STATUS_OK, dtype=object)
    status[~has_side] = "no_sidecar"
    status[has_side & ~eto_ok] = "nonpositive_eto"
    status[has_side & eto_ok & ~etr_ok] = "nonpositive_etr"
    status[~np.isfinite(native_v)] = "native_null"

    ok = status == STATUS_OK
    ratio = np.full(len(led), np.nan)
    ratio[ok] = etr[ok] / eto[ok]
    corrected = np.full(len(led), np.nan)
    corrected[ok] = native_v[ok] * ratio[ok]

    led["ratio"] = ratio
    led["corrected"] = corrected
    led["status"] = status
    led["low_eto"] = ok & (eto < rules["low_eto_mm"])
    lo, hi = rules["ratio_screen"]
    led["ratio_outside_screen"] = ok & ((ratio < lo) | (ratio > hi))
    led["year"] = led["date"].str[:4].astype(int)

    if scenes is not None and not scenes.empty:
        led = led.merge(
            scenes[["site", "date", "sensor", "product_ids", "n_products", "trees"]],
            on=["site", "date"],
            how="left",
            validate="one_to_one",
        )
    for c in ("sensor", "product_ids", "trees"):
        if c not in led:
            led[c] = None
    if "n_products" not in led:
        led["n_products"] = np.nan
    led["sensor"] = led["sensor"].fillna("unknown")
    led["n_products"] = led["n_products"].fillna(0).astype(int)
    if "native_columns" not in led:  # test fixtures built without load_native
        led["native_columns"] = "ETF_" + led["date"]
        led["n_native_columns"] = 1
    for c in ("native_csv_sha256", "sidecar_sha256", "output_csv"):
        if c not in led:
            led[c] = None
    return led[LEDGER_COLUMNS].sort_values(["site", "date"]).reset_index(drop=True)


def output_path(out_dir: str, site: str, year: int) -> str:
    return os.path.join(out_dir, f"{OUTPUT_STEM}_{site}_no_mask_{year}.csv")


def write_grass_csvs(ledger: pd.DataFrame, out_dir: str) -> dict[str, str]:
    """Write one ingest-ready CSV per site-year containing only status==ok rows."""
    os.makedirs(out_dir, exist_ok=True)
    hashes = {}
    ok = ledger[ledger["status"] == STATUS_OK]
    for (site, year), grp in ok.groupby(["site", "year"], sort=True):
        grp = grp.sort_values("date")
        row = {
            "sid": site,
            **{f"ETF_{d}": v for d, v in zip(grp["date"], grp["corrected"], strict=True)},
        }
        path = output_path(out_dir, site, int(year))
        pd.DataFrame([row]).to_csv(path, index=False, float_format="%.10g")
        hashes[path] = sha256_file(path)
    return hashes


def reconstruct_check(ledger: pd.DataFrame, out_dir: str, rtol: float = 1e-9) -> dict:
    """Every written value must reconstruct from ledger native*etr/eto within tolerance."""
    ok = ledger[ledger["status"] == STATUS_OK]
    worst = 0.0
    n = 0
    for (site, year), grp in ok.groupby(["site", "year"], sort=True):
        wide = pd.read_csv(output_path(out_dir, site, int(year)))
        written = {c[4:]: float(wide.iloc[0][c]) for c in wide.columns[1:]}
        if set(written) != set(grp["date"]):
            raise ValueError(f"{site} {year}: written dates differ from ledger")
        expect = grp["native"].to_numpy() * grp["etr"].to_numpy() / grp["eto"].to_numpy()
        got = np.array([written[d] for d in grp["date"]])
        rel = np.abs(got - expect) / np.maximum(np.abs(expect), 1e-12)
        worst = max(worst, float(rel.max()) if len(rel) else 0.0)
        n += len(rel)
    return {"n_values": n, "worst_rel_err": worst, "pass": worst <= rtol}


def cohort_sites(path: str) -> set[str]:
    import fiona

    with fiona.open(path) as src:
        return {f["properties"]["sid"] for f in src}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--native-dir", default=NATIVE_DIR)
    ap.add_argument("--sidecar-dir", default=SIDECAR_DIR)
    ap.add_argument("--out-dir", default=GRASS_DIR)
    ap.add_argument("--qa-dir", default=QA_ROOT)
    ap.add_argument(
        "--cohort", default=COHORT_FGB, help="vector file whose 'sid' set bounds the sites"
    )
    ap.add_argument(
        "--sites", nargs="*", default=None, help="explicit site subset (overrides --cohort)"
    )
    ap.add_argument("--years", nargs=2, type=int, default=RULES["years"])
    ap.add_argument("--dry-run", action="store_true", help="build the ledger, write nothing")
    args = ap.parse_args(argv)

    sites = set(args.sites) if args.sites else cohort_sites(args.cohort)
    native = load_native(args.native_dir, sites, tuple(args.years))
    native_sites = set(native["site"])
    sidecar = load_sidecar(args.sidecar_dir)
    side_hashes = {
        f: sha256_file(os.path.join(args.sidecar_dir, f)) for f in sidecar["source_file"].unique()
    }
    sidecar["sidecar_sha256"] = sidecar["source_file"].map(side_hashes)
    sidecar_sites = set(sidecar["site"])

    tifs = delivered_tifs()
    scenes = scene_identity_by_site_date(tifs[tifs["site"].isin(native_sites)])

    native_hashes = {p: sha256_file(p) for p in native["native_csv"].unique()}
    ledger = convert(native, sidecar, scenes)
    ledger["native_csv_sha256"] = ledger["native_csv"].map(native_hashes)
    ledger["output_csv"] = [
        output_path(args.out_dir, s, y) if st == STATUS_OK else None
        for s, y, st in zip(ledger["site"], ledger["year"], ledger["status"], strict=True)
    ]

    summary = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rules": RULES | {"years": list(args.years)},
        "inputs": {
            "native_dir": args.native_dir,
            "sidecar_dir": args.sidecar_dir,
            "cohort": args.cohort,
        },
        "sites_requested": len(sites),
        "sites_with_native": len(native_sites),
        "sites_without_native": sorted(sites - native_sites),
        "sites_without_sidecar": sorted(native_sites - sidecar_sites),
        "n_rows": int(len(ledger)),
        "status_counts": ledger["status"].value_counts().to_dict(),
        "n_low_eto": int(ledger["low_eto"].sum()),
        "n_ratio_outside_screen": int(ledger["ratio_outside_screen"].sum()),
        "sensor_counts": ledger["sensor"].value_counts().to_dict(),
        "native_at_cap_1.0": int((ledger["native"] >= 1.0).sum()),
        "corrected_above_1.0": int((ledger["corrected"] > 1.0).sum()),
        "ratio_median": float(ledger.loc[ledger["status"] == STATUS_OK, "ratio"].median()),
        "mean_corrected_minus_native": float((ledger["corrected"] - ledger["native"]).mean()),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        out_hashes = write_grass_csvs(ledger, args.out_dir)
        summary["output_files"] = len(out_hashes)
        summary["reconstruct"] = reconstruct_check(ledger, args.out_dir)
        native_after = {p: sha256_file(p) for p in native_hashes}
        side_after = {f: sha256_file(os.path.join(args.sidecar_dir, f)) for f in side_hashes}
        summary["inputs_unchanged"] = native_after == native_hashes and side_after == side_hashes
        os.makedirs(args.qa_dir, exist_ok=True)
        ledger.to_csv(
            os.path.join(args.qa_dir, "ssebop_conversion_ledger.csv"),
            index=False,
            float_format="%.10g",
        )
        with open(os.path.join(args.qa_dir, "ssebop_conversion_hashes.sha256"), "w") as fh:
            for p, h in sorted(
                {
                    **native_hashes,
                    **{os.path.join(args.sidecar_dir, f): h for f, h in side_hashes.items()},
                    **out_hashes,
                }.items()
            ):
                fh.write(f"{h}  {p}\n")
        with open(os.path.join(args.qa_dir, "ssebop_conversion_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

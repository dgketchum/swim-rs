"""Phase 7, step 2: consolidate every extracted ESPA SSEBop value into one complete native record.

Why this exists (2026-09-07): the three legacy order trees (``espa``, ``espa_crop99``,
``espa_ext_2008_2017``) all wrote their ingest CSVs to the same directory with the same file
names, so the last writer (``espa``, 2026-05-08) silently replaced the ``espa_crop99`` /
``espa_ext_2008_2017`` CSVs for the same site-years. 1,522 cohort site-dates (2013-2025) that
were delivered, extracted, and sitting in JSON never reached the baseline container. The new
Landsat 7 repair tree adds 5,989 scene-keyed values on top.

This script reads every ``extracts/etf_json`` tree (read-only), preserves the currently ingested
value for every existing site-date byte-for-byte, adds the JSON-only dates, appends the repair
scenes as scene-keyed columns (``{sensor}_{pathrow}_{YYYYMMDD}``, the PT-JPL export convention
the ingestor collapses by ``max``), and writes a fresh native tree:

    {landsat}/extracts/ssebop_etf_complete/no_mask/ssebop_etf_{site}_no_mask_{year}.csv

Rules: a legacy date present in several trees with identical means is one column; a conflicting
legacy date keeps the value the baseline container ingested (identity), or the ``max`` when it
was never ingested (the ingestor's own same-date rule), and is flagged; a repair scene never
replaces a legacy value (same-date alternates coexist as two columns); two product generations of
one scene key are a hard error; a native CSV value that cannot be reproduced is a hard error.
Nothing under the source trees or the existing ``ssebop_etf`` directory is modified.

Outputs: the CSV tree, ``{qa}/ssebop_native_consolidation_ledger.csv`` (one row per column),
``{qa}/ssebop_native_consolidation_summary.json``, ``{qa}/ssebop_native_complete_hashes.sha256``.
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

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from espa_inventory import RS_ROOT, TREES, json_values, parse_product_id  # noqa: E402

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
LANDSAT = os.path.join(DATA, "remote_sensing", "landsat")
NATIVE_DIR = os.path.join(LANDSAT, "extracts", "ssebop_etf", "no_mask")
COMPLETE_DIR = os.path.join(LANDSAT, "extracts", "ssebop_etf_complete", "no_mask")
COHORT_66 = os.path.join(DATA, "gis", "flux_crop_pub_66_150m.shp")
YEARS = (2013, 2025)
ROUND = 6  # espa_write_etf_csvs.py convention; the baseline CSVs hold 6-decimal means
ROUND_TOL = 10.0**-ROUND + 1e-12  # one unit in the last written decimal
TOL = 1e-9  # exact round-trip of what this script writes

NATIVE_RE = re.compile(r"^ssebop_etf_(?P<site>.+)_no_mask_(?P<year>\d{4})\.csv$")
LEDGER_COLUMNS = [
    "site",
    "year",
    "date",
    "column",
    "value",
    "kind",
    "sources",
    "n_sources",
    "product_id",
    "conflict",
    "resolution",
    "in_native",
    "native_value",
    "same_date_legacy",
    "output_csv",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cohort_sites(path: str) -> set[str]:
    import fiona

    with fiona.open(path) as src:
        return {f["properties"]["sid"] for f in src}


def load_native(native_dir: str, sites: set[str], years: tuple[int, int]) -> pd.DataFrame:
    """(site, date, native_value) for the currently ingested legacy CSVs (date-keyed only)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(native_dir, "ssebop_etf_*_no_mask_*.csv"))):
        m = NATIVE_RE.match(os.path.basename(path))
        if not m or m.group("site") not in sites:
            continue
        if not years[0] <= int(m.group("year")) <= years[1]:
            continue
        wide = pd.read_csv(path)
        if len(wide) != 1 or wide.iloc[0, 0] != m.group("site"):
            raise ValueError(f"{path}: expected one row keyed {m.group('site')!r}")
        for col in wide.columns[1:]:
            if not (col.startswith("ETF_") and len(col) == 12 and col[4:].isdigit()):
                raise ValueError(f"{path}: unexpected column {col!r}")
            rows.append(
                {"site": m.group("site"), "date": col[4:], "native_value": wide.iloc[0][col]}
            )
    return pd.DataFrame(rows, columns=["site", "date", "native_value"])


def consolidate(
    values: pd.DataFrame, native: pd.DataFrame, sites: set[str], years: tuple[int, int]
) -> pd.DataFrame:
    """Pure step: one ledger row per output column. Raises on identity or duplicate violations."""
    v = values[values["site"].isin(sites)].copy()
    v = v[v["site"] == v["file_site"]]
    v["year"] = v["date"].str[:4].astype(int)
    v = v[(v["year"] >= years[0]) & (v["year"] <= years[1])]
    if v["mean"].isna().any():
        bad = v[v["mean"].isna()].head()
        raise ValueError(f"JSON entries without a mean (extractor never writes these):\n{bad}")
    v["value"] = v["mean"].astype(float).round(ROUND)

    nat = native.set_index(["site", "date"])["native_value"].astype(float)
    if nat.index.duplicated().any():
        raise ValueError("duplicate (site, date) in the existing native CSVs")

    rows = []
    legacy = v[v["product_id"].isna()]
    for (site, date), grp in legacy.groupby(["site", "date"], sort=True):
        per_tree = grp.set_index("tree")["value"]
        distinct = sorted(set(per_tree))
        in_native = (site, date) in nat.index
        native_value = float(nat.loc[(site, date)]) if in_native else None
        # Values within one unit of the 6th decimal are the same observation (the legacy writer
        # and this script may round a half-way mean in opposite directions); larger gaps are
        # different scenes (path/row alternates) or product generations.
        conflict = (distinct[-1] - distinct[0]) > ROUND_TOL
        if in_native:
            if not any(abs(x - native_value) <= ROUND_TOL for x in distinct):
                raise ValueError(
                    f"{site} {date}: no tree value matches ingested native {native_value} "
                    f"(trees: {per_tree.to_dict()})"
                )
            value = native_value  # byte-for-byte identity with the ingested record
            resolution = "kept_ingested_value" if conflict else "single_value"
        elif conflict:
            value, resolution = max(distinct), "max_of_trees"
        else:
            value, resolution = distinct[0], "single_value"
        rows.append(
            {
                "site": site,
                "year": int(date[:4]),
                "date": date,
                "column": f"ETF_{date}",
                "value": value,
                "kind": "legacy_date",
                "sources": ";".join(sorted(per_tree.index)),
                "n_sources": len(per_tree),
                "product_id": None,
                "conflict": conflict,
                "resolution": resolution,
                "in_native": in_native,
                "native_value": native_value,
                "same_date_legacy": False,
            }
        )
    legacy_dates = {(r["site"], r["date"]) for r in rows}

    missing = [
        k for k in nat.index if k not in legacy_dates and years[0] <= int(k[1][:4]) <= years[1]
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} ingested native site-dates have no JSON source: {missing[:5]}"
        )

    scenes = v[v["product_id"].notna()]
    seen: dict[tuple[str, str], str] = {}
    for _, r in scenes.sort_values(["site", "date", "product_id"]).iterrows():
        ident = parse_product_id(r["product_id"])
        if ident is None:
            raise ValueError(f"{r['tree']} {r['site']}: unparseable product id {r['product_id']}")
        if ident["acquired"] != r["date"]:
            raise ValueError(f"{r['product_id']}: JSON date {r['date']} != product date")
        key = f"{ident['sensor']}_{ident['pathrow']}_{ident['acquired']}"
        if (r["site"], key) in seen:
            raise ValueError(
                f"{r['site']}: scene key {key} delivered twice "
                f"({seen[(r['site'], key)]}, {r['product_id']}); keep one product generation"
            )
        seen[(r["site"], key)] = r["product_id"]
        rows.append(
            {
                "site": r["site"],
                "year": int(r["date"][:4]),
                "date": r["date"],
                "column": key,
                "value": float(r["value"]),
                "kind": "scene",
                "sources": r["tree"],
                "n_sources": 1,
                "product_id": r["product_id"],
                "conflict": False,
                "resolution": "scene_column",
                "in_native": False,
                "native_value": None,
                "same_date_legacy": (r["site"], r["date"]) in legacy_dates,
            }
        )
    out = pd.DataFrame(rows, columns=LEDGER_COLUMNS[:-1])
    dup = out.duplicated(["site", "column"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate output columns:\n{out.loc[dup].head()}")
    return out.sort_values(["site", "date", "column"]).reset_index(drop=True)


def output_path(out_dir: str, site: str, year: int) -> str:
    return os.path.join(out_dir, f"ssebop_etf_{site}_no_mask_{year}.csv")


def write_complete_csvs(ledger: pd.DataFrame, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    hashes = {}
    for (site, year), grp in ledger.groupby(["site", "year"], sort=True):
        grp = grp.sort_values(["date", "column"])
        row = {"sid": site, **dict(zip(grp["column"], grp["value"], strict=True))}
        path = output_path(out_dir, site, int(year))
        pd.DataFrame([row]).to_csv(path, index=False, float_format=f"%.{ROUND}f")
        hashes[path] = sha256_file(path)
    return hashes


def verify_written(ledger: pd.DataFrame, native: pd.DataFrame, out_dir: str) -> dict:
    """Re-read the tree: every ledger value round-trips, every native value is reproduced."""
    written = {}
    for path in glob.glob(os.path.join(out_dir, "ssebop_etf_*_no_mask_*.csv")):
        wide = pd.read_csv(path)
        site = wide.iloc[0, 0]
        for col in wide.columns[1:]:
            written[(site, col)] = float(wide.iloc[0][col])
    if len(written) != len(ledger):
        raise ValueError(f"written columns {len(written)} != ledger rows {len(ledger)}")
    worst = 0.0
    for s, c, val in zip(ledger["site"], ledger["column"], ledger["value"], strict=True):
        worst = max(worst, abs(written[(s, c)] - float(val)))
    nat_worst = 0.0
    for s, d, val in zip(native["site"], native["date"], native["native_value"], strict=True):
        nat_worst = max(nat_worst, abs(written[(s, f"ETF_{d}")] - float(val)))
    return {
        "n_columns": len(written),
        "ledger_roundtrip_max_abs": worst,
        "native_identity_max_abs": nat_worst,
        "native_values_checked": int(len(native)),
        "pass": worst <= TOL and nat_worst <= TOL,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--trees",
        nargs="*",
        default=sorted(TREES),
        help=f"ESPA order trees under {RS_ROOT} (default: every registered tree)",
    )
    ap.add_argument("--native-dir", default=NATIVE_DIR)
    ap.add_argument("--out-dir", default=COMPLETE_DIR)
    ap.add_argument("--qa-dir", default=QA_ROOT)
    ap.add_argument("--cohort", default=COHORT_66)
    ap.add_argument("--years", nargs=2, type=int, default=list(YEARS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    trees = {t: TREES[t] for t in args.trees}
    years = (args.years[0], args.years[1])
    sites = cohort_sites(args.cohort)
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.dry_run:
        raise SystemExit(f"{args.out_dir} is not empty; remove it deliberately before rebuilding")

    values = json_values(trees)
    native = load_native(args.native_dir, sites, years)
    ledger = consolidate(values, native, sites, years)

    by_kind = ledger.groupby("kind").size().to_dict()
    summary = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "inputs": {"trees": trees, "native_dir": args.native_dir, "cohort": args.cohort},
        "years": list(years),
        "sites_requested": len(sites),
        "sites_with_values": int(ledger["site"].nunique()),
        "sites_without_values": sorted(sites - set(ledger["site"])),
        "n_columns": int(len(ledger)),
        "n_by_kind": by_kind,
        "n_site_dates": int(ledger.groupby(["site", "date"]).ngroups),
        "legacy_in_native": int(ledger["in_native"].sum()),
        "legacy_json_only": int(((ledger["kind"] == "legacy_date") & ~ledger["in_native"]).sum()),
        "legacy_json_only_by_source": ledger[
            (ledger["kind"] == "legacy_date") & ~ledger["in_native"]
        ]
        .groupby("sources")
        .size()
        .to_dict(),
        "legacy_conflicts": int(ledger["conflict"].sum()),
        "legacy_conflict_resolutions": ledger[ledger["conflict"]]
        .groupby("resolution")
        .size()
        .to_dict(),
        "scene_columns_by_tree": ledger[ledger["kind"] == "scene"]
        .groupby("sources")
        .size()
        .to_dict(),
        "scene_same_date_as_legacy": int(ledger["same_date_legacy"].sum()),
        "scene_multi_per_date": int(
            (ledger[ledger["kind"] == "scene"].groupby(["site", "date"]).size() > 1).sum()
        ),
        "values_below_0.05": int((ledger["value"] < 0.05).sum()),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        hashes = write_complete_csvs(ledger, args.out_dir)
        ledger["output_csv"] = [
            output_path(args.out_dir, s, int(y)) for s, y in zip(ledger["site"], ledger["year"])
        ]
        summary["output_files"] = len(hashes)
        summary["verify"] = verify_written(ledger, native, args.out_dir)
        os.makedirs(args.qa_dir, exist_ok=True)
        ledger.to_csv(
            os.path.join(args.qa_dir, "ssebop_native_consolidation_ledger.csv"), index=False
        )
        with open(os.path.join(args.qa_dir, "ssebop_native_complete_hashes.sha256"), "w") as fh:
            for p, h in sorted(hashes.items()):
                fh.write(f"{h}  {p}\n")
        with open(os.path.join(args.qa_dir, "ssebop_native_consolidation_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if args.dry_run or summary["verify"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

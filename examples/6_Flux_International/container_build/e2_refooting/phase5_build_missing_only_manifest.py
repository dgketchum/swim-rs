"""Phase 5 / Gate G5: build the missing-only ESPA repair manifest for the 66-site cohort.

Selection (plan §11, scope decision 2026-09-04): every canonical site-date, 2013–2025, with a
valid PT-JPL Landsat observation in the baseline container and no valid native ESPA SSEBop
value on disk. Each such date is traced through the three historical ESPA trees and classified:

    never_ordered            no delivered tif and no submitted order lists the scene  -> request
    ordered_cancelled        the only orders listing the scene were cancelled          -> request (status label kept)
    ordered_not_delivered    a submitted, non-cancelled order listed it; no ETF tif    -> request (LE07 and OLI; user decision 2026-09-04)
    delivered_nodata         an ETF tif exists but the extractor produced no valid value -> residual
    delivered_not_ingested   a valid extracted value exists in some tree but not in the ingest CSV -> residual (repair locally)
    no_product_id            the scene key has no Collection 2 Level-2 product ID     -> residual

Requests map each PT-JPL scene key (sensor_pathrow_date) to exactly one product ID from the local
Landsat metadata parquet (L2SP before L2SR, T1 < T2 < RT, newest generation first). Extents are
the canonical 4 km UTM chips from ``espa/extents`` and are checked against the 66-site polygons.

Same-date alternates (a site imaged by two adjacent rows of one overpass) are separate units with
their own product IDs; they are kept because the extractor now preserves scene identity and the
CSV writer emits one scene-key column per product ID, which the ingestor collapses by per-date
``max`` exactly as it does for PT-JPL (``espa/espa_extract_etf.py``, ``espa/espa_write_etf_csvs.py``).

Nothing under the historical trees is modified. Outputs go to the QA root and to a new tree,
``remote_sensing/espa_le07_repair/`` (manifest + payloads; submission is Phase 6 and needs A2,
recorded here through ``--a2-granted``).

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase5_build_missing_only_manifest.py \
        [--availability-sample 18] [--a2-granted "text of the approval"]
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

import fiona
import numpy as np
import pandas as pd
import zarr
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from espa_inventory import (  # noqa: E402
    SENSOR_TO_COLLECTION,
    TREES,
    delivered_tifs,
    json_values,
    manifest_rows,
    parse_product_id,
    payload_requests,
)

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
CONTAINER = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")
COHORT_66 = os.path.join(DATA, "gis", "flux_crop_pub_66_150m.shp")
PTJPL_DIR = os.path.join(DATA, "remote_sensing", "landsat", "extracts", "ptjpl_etf", "no_mask")
NATIVE_DIR = os.path.join(DATA, "remote_sensing", "landsat", "extracts", "ssebop_etf", "no_mask")
EXTENT_DIR = os.path.join(DATA, "remote_sensing", "espa", "extents")
METADATA = "/data/ssd1/swim/landsat_metadata/LANDSAT_C2_L2_product_ids.parquet"
REPAIR_TREE = os.path.join(DATA, "remote_sensing", "espa_le07_repair")
ESPA_API = "https://espa.cr.usgs.gov/api/v1"
CRED_FILE = os.path.expanduser("~/usgs_pswd.txt")

YEARS = (2013, 2025)
EXCLUDED_YEARS = list(range(2008, 2013))
CEILINGS = {"total_site_dates": 8527, "LE07_site_dates": 6341, "OLI_site_dates": 2186}
OPEN_UNIT_CAP = 10000
OLI = {"LC08", "LC09"}

SCENE_KEY_RE = re.compile(r"^(LT04|LT05|LE07|LC08|LC09)_(\d{6})_(\d{8})$")
PTJPL_FILE_RE = re.compile(r"^ptjpl_etf_(?P<site>.+)_no_mask_(?P<year>\d{4})(_b\d{2})?\.csv$")

REQUEST_STATUSES = {"never_ordered", "ordered_cancelled", "ordered_not_delivered"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- inputs
def cohort_geometries(path: str) -> dict[str, dict]:
    out = {}
    with fiona.open(path) as src:
        for f in src:
            sid = f["properties"]["sid"]
            coords = np.array(f["geometry"]["coordinates"][0])
            out[sid] = {
                "centroid_lon": float(coords[:, 0].mean()),
                "centroid_lat": float(coords[:, 1].mean()),
                "wkt_sha256": hashlib.sha256(
                    json.dumps(f["geometry"]["coordinates"]).encode()
                ).hexdigest(),
            }
    return out


def container_ptjpl_dates(container: str, sites: set[str], years: tuple[int, int]) -> pd.DataFrame:
    root = zarr.open(container, mode="r")
    dates = pd.to_datetime(root["time/daily"][:]).strftime("%Y%m%d")
    uids = [str(u) for u in root["geometry/uid"][:]]
    pj = np.asarray(root["remote_sensing/etf/landsat/ptjpl/no_mask"][:], float)
    ss = np.asarray(root["remote_sensing/etf/landsat/ssebop/no_mask"][:], float)
    rows = []
    for j, sid in enumerate(uids):
        if sid not in sites:
            continue
        for i in np.flatnonzero(np.isfinite(pj[:, j])):
            d = dates[i]
            if years[0] <= int(d[:4]) <= years[1]:
                rows.append(
                    {
                        "site": sid,
                        "date": d,
                        "ptjpl_container": float(pj[i, j]),
                        "ssebop_container": float(ss[i, j]) if np.isfinite(ss[i, j]) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def ptjpl_scene_keys(ptjpl_dir: str, sites: set[str], years: tuple[int, int]) -> pd.DataFrame:
    """Every (site, scene_key) with a finite PT-JPL value in the extract CSVs."""
    rows = []
    for path in sorted(glob.glob(os.path.join(ptjpl_dir, "ptjpl_etf_*_no_mask_*.csv"))):
        m = PTJPL_FILE_RE.match(os.path.basename(path))
        if (
            not m
            or m.group("site") not in sites
            or not years[0] <= int(m.group("year")) <= years[1]
        ):
            continue
        wide = pd.read_csv(path)
        row = wide.iloc[0]
        for col in wide.columns[1:]:
            sk = SCENE_KEY_RE.match(col)
            if sk is None:
                raise ValueError(f"{path}: unexpected PT-JPL column {col!r}")
            v = row[col]
            if pd.notna(v):
                rows.append(
                    {
                        "site": m.group("site"),
                        "scene_key": col,
                        "sensor": sk.group(1),
                        "pathrow": sk.group(2),
                        "date": sk.group(3),
                        "ptjpl_value": float(v),
                        "ptjpl_csv": path,
                    }
                )
    df = pd.DataFrame(rows)
    return (
        df.drop_duplicates(["site", "scene_key"])
        .sort_values(["site", "date", "scene_key"])
        .reset_index(drop=True)
    )


def native_valid_dates(native_dir: str, sites: set[str], years: tuple[int, int]) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(native_dir, "ssebop_etf_*_no_mask_*.csv"))):
        stem = os.path.basename(path)[len("ssebop_etf_") : -len(".csv")]
        site, year = stem.rsplit("_no_mask_", 1)
        if site not in sites or not years[0] <= int(year) <= years[1]:
            continue
        row = pd.read_csv(path).iloc[0]
        for col in row.index[1:]:
            if pd.notna(row[col]):
                rows.append(
                    {
                        "site": site,
                        "date": col[4:],
                        "native_value": float(row[col]),
                        "native_csv": path,
                    }
                )
    return pd.DataFrame(rows, columns=["site", "date", "native_value", "native_csv"])


def load_metadata(path: str) -> pd.DataFrame:
    s = pd.read_parquet(path).iloc[:, 0].astype(str)
    parsed = pd.DataFrame(
        {
            "product_id": s.values,
            "sensor": s.str[:4].values,
            "level": s.str[5:9].values,
            "pathrow": s.str[10:16].values,
            "acquired": s.str[17:25].values,
            "generated": s.str[26:34].values,
            "tier": s.str[-2:].values,
        }
    )
    parsed["level_rank"] = parsed["level"].map({"L2SP": 0, "L2SR": 1}).fillna(9).astype(int)
    parsed["tier_rank"] = parsed["tier"].map({"T1": 0, "T2": 1, "RT": 2}).fillna(9).astype(int)
    return parsed


def best_product_ids(scenes: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One product ID per scene key: L2SP before L2SR, T1 < T2 < RT, newest generation first."""
    keys = (
        scenes[["sensor", "pathrow", "date"]].drop_duplicates().rename(columns={"date": "acquired"})
    )
    cand = keys.merge(meta, on=["sensor", "pathrow", "acquired"], how="left")
    cand = cand.sort_values(
        ["sensor", "pathrow", "acquired", "level_rank", "tier_rank", "generated"],
        ascending=[True, True, True, True, True, False],
    )
    best = cand.groupby(["sensor", "pathrow", "acquired"], sort=False).head(1)
    n_cand = (
        cand.dropna(subset=["product_id"])
        .groupby(["sensor", "pathrow", "acquired"])
        .size()
        .rename("n_candidates")
    )
    best = best.merge(n_cand, on=["sensor", "pathrow", "acquired"], how="left")
    best["n_candidates"] = best["n_candidates"].fillna(0).astype(int)
    return best.rename(columns={"acquired": "date"})[
        ["sensor", "pathrow", "date", "product_id", "level", "tier", "n_candidates"]
    ]


# --------------------------------------------------------------------------- classification
def classify(
    candidates: pd.DataFrame,
    tifs: pd.DataFrame,
    payloads: pd.DataFrame,
    manifests: pd.DataFrame,
    jvals: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Attach delivery evidence and a status to every PT-JPL-only scene candidate.

    ``candidates``: site, scene_key, sensor, pathrow, date (+ any extra columns).
    """
    c = candidates.merge(products, on=["sensor", "pathrow", "date"], how="left")

    # delivered evidence at the scene level (any product ID for this sensor/pathrow/date at this site)
    t = tifs[["site", "sensor", "pathrow", "acquired", "tree", "tif_path"]].rename(
        columns={"acquired": "date"}
    )
    t = (
        t.groupby(["site", "sensor", "pathrow", "date"])
        .agg(
            delivered_trees=("tree", lambda s: ";".join(sorted(set(s)))),
            tif_paths=("tif_path", lambda s: ";".join(sorted(s))),
        )
        .reset_index()
    )
    c = c.merge(t, on=["site", "sensor", "pathrow", "date"], how="left")

    # ordered evidence: payload listing in a site-year whose manifest row carries an order id
    man = manifests.copy()
    man["submitted"] = man["order_id"].notna() & (man["order_id"] != "")
    man["cancelled"] = man["order_status"].fillna("") == "cancelled"
    p = payloads.merge(
        man[["tree", "site", "year", "submitted", "cancelled", "order_id", "order_status"]],
        on=["tree", "site", "year"],
        how="left",
    )
    p = p[p["submitted"].fillna(False)]
    p = (
        p.rename(columns={"acquired": "date"})
        .groupby(["site", "sensor", "pathrow", "date"])
        .agg(
            ordered_trees=("tree", lambda s: ";".join(sorted(set(s)))),
            all_orders_cancelled=("cancelled", "all"),
            order_ids=("order_id", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    c = c.merge(p, on=["site", "sensor", "pathrow", "date"], how="left")

    # extracted-value evidence at the site-date level (json is keyed by date, not scene)
    jv = (
        jvals[jvals["mean"].notna()]
        .groupby(["site", "date"])
        .agg(json_trees=("tree", lambda s: ";".join(sorted(set(s)))), json_mean=("mean", "first"))
        .reset_index()
    )
    c = c.merge(jv, on=["site", "date"], how="left")

    delivered = c["delivered_trees"].notna()
    ordered = c["ordered_trees"].notna()
    cancelled_only = ordered & c["all_orders_cancelled"].fillna(False).astype(bool)
    has_json = c["json_trees"].notna()
    no_pid = c["product_id"].isna()

    native_present = (
        c["native_value"].notna() if "native_value" in c else pd.Series(False, index=c.index)
    )
    below_min = (
        native_present & (c["native_value"] < 0.05) if "native_value" in c else native_present
    )
    status = np.select(
        [
            no_pid,
            below_min,
            native_present,
            delivered & has_json,
            delivered & ~has_json,
            ordered & ~cancelled_only,
            cancelled_only,
        ],
        [
            "no_product_id",
            "native_below_min_etf",
            "native_valid_not_in_container",
            "delivered_not_ingested",
            "delivered_nodata",
            "ordered_not_delivered",
            "ordered_cancelled",
        ],
        default="never_ordered",
    )
    c["status"] = status
    # 2026-09-04: ordered-not-delivered OLI scenes are re-requested like LE07 (user decision);
    # the status label distinguishes them from never-ordered scenes in the ledger.
    c["request"] = c["status"].isin(REQUEST_STATUSES)
    c["selection_reason"] = np.where(
        c["request"],
        "ptjpl_only_" + c["sensor"].str.lower() + "_" + c["status"],
        "residual_" + c["sensor"].str.lower() + "_" + c["status"],
    )
    return c


# --------------------------------------------------------------------------- extents / payloads
def read_extent(site: str, extent_dir: str = EXTENT_DIR) -> dict:
    path = os.path.join(extent_dir, f"{site}_extent.csv")
    row = pd.read_csv(path).iloc[0]
    return {
        "extent_csv": path,
        "extent_sha256": sha256_file(path),
        "epsg": int(row["epsg"]),
        "utm_zone": int(row["utm_zone"]),
        "utm_hemisphere": str(row["utm_hemisphere"]),
        "minx": float(row["minx"]),
        "miny": float(row["miny"]),
        "maxx": float(row["maxx"]),
        "maxy": float(row["maxy"]),
        "chip_size_m": float(row["chip_size_m"]),
    }


def extent_matches_geometry(extent: dict, geom: dict, tol_m: float = 5.0) -> tuple[bool, float]:
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{extent['epsg']}", always_xy=True)
    x, y = tr.transform(geom["centroid_lon"], geom["centroid_lat"])
    cx, cy = (extent["minx"] + extent["maxx"]) / 2, (extent["miny"] + extent["maxy"]) / 2
    d = float(np.hypot(x - cx, y - cy))
    return d <= tol_m, d


def build_payload(site: str, year: int, product_ids: list[str], extent: dict) -> dict:
    payload = {
        "projection": {"utm": {"zone": extent["utm_zone"], "zone_ns": extent["utm_hemisphere"]}},
        "image_extents": {
            "north": extent["maxy"],
            "south": extent["miny"],
            "east": extent["maxx"],
            "west": extent["minx"],
            "units": "meters",
        },
        "format": "gtiff",
        "resampling_method": "nn",
        "note": f"{site}_{year}_e2_refooting_missing_only",
    }
    by_sensor: dict[str, list[str]] = {}
    for pid in sorted(set(product_ids)):
        parsed = parse_product_id(pid)
        by_sensor.setdefault(SENSOR_TO_COLLECTION[parsed["sensor"]], []).append(pid)
    for key in sorted(by_sensor):
        payload[key] = {"inputs": sorted(by_sensor[key]), "products": ["et"]}
    return payload


def availability_check(product_ids: list[str], cred_file: str = CRED_FILE) -> list[dict]:
    """Read-only ESPA ``available-products`` lookups; records request and response verbatim."""
    import requests

    text = open(cred_file).read().strip()
    auth = tuple(text.split(":", 1)) if ":" in text else tuple(text.splitlines()[:2])
    out = []
    for pid in product_ids:
        url = f"{ESPA_API}/available-products/{pid}"
        resp = requests.get(url, auth=auth, timeout=60)
        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else resp.text
        )
        et_ok = False
        if isinstance(body, dict):
            for key, block in body.items():
                if (
                    isinstance(block, dict)
                    and pid in block.get("inputs", [])
                    and "et" in block.get("products", [])
                ):
                    et_ok = True
        out.append(
            {
                "product_id": pid,
                "url": url,
                "http_status": resp.status_code,
                "et_advertised": et_ok,
                "response": body,
                "checked": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
        )
    return out


def sample_for_availability(req: pd.DataFrame, n: int) -> list[str]:
    if req.empty or n <= 0:
        return []
    out = []
    for sensor, g in req.groupby("sensor"):
        g = g.sort_values("date")
        k = max(2, round(n * len(g) / len(req)))
        idx = np.linspace(0, len(g) - 1, num=min(k, len(g)), dtype=int)
        out.extend(g.iloc[idx]["product_id"].tolist())
    return sorted(set(out))


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=QA_ROOT)
    ap.add_argument("--repair-tree", default=REPAIR_TREE)
    ap.add_argument(
        "--availability-sample", type=int, default=18, help="0 disables the read-only ESPA lookup"
    )
    ap.add_argument(
        "--a2-granted",
        default="",
        help="verbatim text of the user's A2 approval; empty means A2 not granted",
    )
    args = ap.parse_args(argv)

    existing_manifest = os.path.join(args.repair_tree, "espa_manifest.csv")
    if os.path.exists(existing_manifest):
        prev = pd.read_csv(existing_manifest, dtype=str)
        submitted = prev["order_id"].notna() & (prev["order_id"] != "")
        if submitted.any():
            raise SystemExit(
                f"{existing_manifest} has {int(submitted.sum())} submitted rows; "
                "refusing to regenerate a manifest that already carries order IDs"
            )

    geoms = cohort_geometries(COHORT_66)
    sites = set(geoms)
    ptjpl_c = container_ptjpl_dates(CONTAINER, sites, YEARS)
    keys = ptjpl_scene_keys(PTJPL_DIR, sites, YEARS)
    native = native_valid_dates(NATIVE_DIR, sites, YEARS)

    # container PT-JPL dates must all have a scene key; scene keys with no container date are not candidates
    kd = keys[["site", "date"]].drop_duplicates()
    cont_no_key = ptjpl_c.merge(kd.assign(k=True), on=["site", "date"], how="left")
    cont_no_key = cont_no_key[cont_no_key["k"].isna()]

    # PT-JPL-only is defined by the container (the objective's member availability). Where the
    # native ingest CSV nevertheless holds a value, the ingestor dropped it (min_etf = 0.05) or the
    # container predates the CSV; both are residual classes, never re-order candidates.
    cs = ptjpl_c.merge(native, on=["site", "date"], how="left")
    ssebop_valid = cs["ssebop_container"].notna()
    native_present = cs["native_value"].notna()
    container_stale = int((ssebop_valid & ~native_present).sum())
    ptjpl_only_dates = cs[~ssebop_valid][["site", "date", "ptjpl_container", "native_value"]]
    cand = keys.merge(ptjpl_only_dates, on=["site", "date"], how="inner")

    tifs = delivered_tifs()
    payloads = payload_requests()
    manifests = manifest_rows()
    jvals = json_values()
    meta = load_metadata(METADATA)
    products = best_product_ids(cand, meta)
    classified = classify(cand, tifs, payloads, manifests, jvals, products)

    # extents and geometry check
    ext_rows = []
    for site in sorted(sites):
        ext = read_extent(site)
        ok, d = extent_matches_geometry(ext, geoms[site])
        ext_rows.append(
            {
                "site": site,
                "extent_ok": ok,
                "extent_center_offset_m": d,
                "geometry_sha256": geoms[site]["wkt_sha256"],
                **ext,
            }
        )
    extents = pd.DataFrame(ext_rows)
    classified = classified.merge(
        extents[
            [
                "site",
                "extent_csv",
                "extent_sha256",
                "geometry_sha256",
                "extent_ok",
                "extent_center_offset_m",
            ]
        ],
        on="site",
        how="left",
    )
    classified["year"] = classified["date"].str[:4].astype(int)

    # exact-duplicate removal (site, product_id, extent)
    before = len(classified)
    classified = classified.drop_duplicates(["site", "product_id", "extent_sha256"]).reset_index(
        drop=True
    )
    n_exact_dupes = before - len(classified)

    req = classified[classified["request"]].copy()
    req["same_date_alternate"] = req.duplicated(["site", "date"], keep="first")
    # every same-date group must be one overpass: one sensor, one path, adjacent rows
    alt_groups = req[req.duplicated(["site", "date"], keep=False)].groupby(["site", "date"])
    alt_same_overpass = bool(
        all(
            g["sensor"].nunique() == 1
            and g["pathrow"].str[:3].nunique() == 1
            and g["pathrow"].str[3:].astype(int).diff().dropna().abs().eq(1).all()
            for _, g in alt_groups
        )
    )

    # payloads (dry run) into the new repair tree
    payload_dir = os.path.join(args.repair_tree, "payloads")
    os.makedirs(payload_dir, exist_ok=True)
    payload_rows = []
    classified["payload_id"] = None
    for (site, year), g in req.groupby(["site", "year"], sort=True):
        ext = extents.set_index("site").loc[site].to_dict()
        payload = build_payload(site, int(year), g["product_id"].tolist(), ext)
        path = os.path.join(payload_dir, f"{site}_{year}_payload.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        payload_id = f"{site}_{year}"
        classified.loc[g.index, "payload_id"] = payload_id
        payload_rows.append(
            {
                "site": site,
                "year": int(year),
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "chip_size_m": ext["chip_size_m"],
                "n_scenes": int(len(g)),
                "n_le07": int((g["sensor"] == "LE07").sum()),
                "n_oli": int(g["sensor"].isin(OLI).sum()),
                "scene_csv": None,
                "extent_csv": ext["extent_csv"],
                "payload_json": path,
                "payload_sha256": sha256_file(path),
                "order_id": None,
                "order_status": "not_submitted",
                "download_status": None,
                "extract_status": None,
                "output_dir": os.path.join(args.repair_tree, "orders", site, str(year)),
                "notes": "e2_refooting_missing_only",
            }
        )
    repair_manifest = pd.DataFrame(payload_rows)
    repair_manifest.to_csv(os.path.join(args.repair_tree, "espa_manifest.csv"), index=False)

    # deterministic cap-safe rounds (units = scenes)
    repair_manifest = repair_manifest.sort_values(["year", "site"]).reset_index(drop=True)
    rounds, cum, r = [], 0, 1
    for n in repair_manifest["n_scenes"]:
        if cum + n > OPEN_UNIT_CAP:
            r, cum = r + 1, 0
        rounds.append(r)
        cum += n
    repair_manifest["round"] = rounds
    repair_manifest.to_csv(os.path.join(args.repair_tree, "espa_manifest.csv"), index=False)

    # availability sample (read-only)
    avail = (
        availability_check(sample_for_availability(req, args.availability_sample))
        if args.availability_sample
        else []
    )
    avail_map = {a["product_id"]: a["et_advertised"] for a in avail}
    classified["espa_et_advertised"] = classified["product_id"].map(avail_map)

    # write manifest with the plan's columns
    manifest_cols = [
        "site",
        "date",
        "year",
        "sensor",
        "scene_key",
        "pathrow",
        "product_id",
        "level",
        "tier",
        "n_candidates",
        "extent_csv",
        "extent_sha256",
        "geometry_sha256",
        "extent_ok",
        "ptjpl_value",
        "ptjpl_csv",
        "ptjpl_container",
        "native_status",
        "delivered_trees",
        "ordered_trees",
        "order_ids",
        "json_trees",
        "json_mean",
        "status",
        "request",
        "selection_reason",
        "espa_et_advertised",
        "payload_id",
        "order_id",
        "order_status",
        "download_status",
        "checksum_status",
        "extraction_status",
        "corrected_value_status",
    ]
    classified["native_status"] = "no_valid_native_ssebop"
    for c in (
        "order_id",
        "order_status",
        "download_status",
        "checksum_status",
        "extraction_status",
        "corrected_value_status",
    ):
        classified[c] = None
    classified["order_status"] = np.where(classified["request"], "not_submitted", "not_requested")
    classified = classified.sort_values(["site", "date", "scene_key"]).reset_index(drop=True)
    classified[manifest_cols].to_csv(
        os.path.join(args.out_dir, "le07_missing_only_manifest.csv"), index=False
    )
    classified[~classified["request"]][manifest_cols].to_csv(
        os.path.join(args.out_dir, "le07_residual_ledger.csv"), index=False
    )

    site_dates = req.drop_duplicates(["site", "date"])
    n_le07_sd = int((site_dates["sensor"] == "LE07").sum())
    n_oli_sd = int(site_dates["sensor"].isin(OLI).sum())
    ptjpl_captures_66 = int(len(ptjpl_c))
    residual_dates = classified[~classified["request"]].drop_duplicates(["site", "date"])
    residual_after = residual_dates[
        ~residual_dates.set_index(["site", "date"]).index.isin(
            site_dates.set_index(["site", "date"]).index
        )
    ]

    audit = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "scope": {
            "cohort": COHORT_66,
            "n_sites": len(sites),
            "years": list(YEARS),
            "excluded_years": EXCLUDED_YEARS,
            "excluded_years_reason": "outside the 2013-2025 E2 configuration; raw 2008-2012 ingest CSVs left untouched",
            "order_scope": "every PT-JPL-only scene that is never ordered, cancelled-only, or "
            "ordered-not-delivered, for LE07 and OLI alike (user decision 2026-09-04, replacing the "
            "earlier OLI ordered-not-delivered exclusion); delivered-nodata and "
            "delivered-not-ingested scenes stay residual",
            "same_date_alternates": "kept as separate units; the extractor keeps each product ID "
            "and the CSV writer emits one scene-key column per product ID; the ingestor collapses "
            "same-date Landsat columns by max, as for PT-JPL",
        },
        "approvals": {"A2": args.a2_granted or None},
        "inputs": {
            "container": CONTAINER,
            "ptjpl_dir": PTJPL_DIR,
            "native_dir": NATIVE_DIR,
            "metadata": METADATA,
            "metadata_sha256": sha256_file(METADATA),
            "trees": TREES,
        },
        "counts": {
            "ptjpl_container_dates_66": ptjpl_captures_66,
            "ptjpl_container_dates_without_scene_key": int(len(cont_no_key)),
            "container_valid_but_native_csv_missing": container_stale,
            "native_below_min_etf_site_dates": int((ptjpl_only_dates["native_value"] < 0.05).sum()),
            "ptjpl_only_site_dates": int(len(ptjpl_only_dates)),
            "ptjpl_only_scene_candidates": int(len(cand)),
            "exact_duplicates_removed": n_exact_dupes,
            "status_counts_scenes": classified["status"].value_counts().to_dict(),
            "status_counts_by_sensor": {
                f"{s}:{st}": int(n)
                for (s, st), n in classified.groupby(["sensor", "status"]).size().items()
            },
            "requested_scenes": int(len(req)),
            "requested_by_status": req["status"].value_counts().to_dict(),
            "requested_same_date_alternates": int(req["same_date_alternate"].sum()),
            "requested_site_dates": int(len(site_dates)),
            "requested_site_dates_LE07": n_le07_sd,
            "requested_site_dates_OLI": n_oli_sd,
            "requested_by_year": req.drop_duplicates(["site", "date"])
            .groupby("year")
            .size()
            .to_dict(),
            "requested_by_site": req.drop_duplicates(["site", "date"])
            .groupby("site")
            .size()
            .to_dict(),
            "payloads": int(len(repair_manifest)),
            "rounds": int(repair_manifest["round"].max()) if len(repair_manifest) else 0,
            "units_per_round": repair_manifest.groupby("round")["n_scenes"].sum().to_dict(),
            "residual_site_dates_not_requested": int(len(residual_after)),
            "residual_share_of_ptjpl_captures_if_all_delivered": float(
                len(residual_after) / ptjpl_captures_66
            ),
        },
        "ceilings": CEILINGS,
        "availability_check": avail,
        "gate": {
            "all_sites_in_66": bool(set(classified["site"]) <= sites),
            "every_request_is_ptjpl_only": True,  # by construction: candidates are the PT-JPL-only set
            "no_valid_ssebop_requested": bool(req["native_value"].isna().all())
            and container_stale == 0,
            "one_product_id_per_request": bool(
                req["product_id"].notna().all() and not req.duplicated(["site", "product_id"]).any()
            ),
            "extents_reproduce_geometry": bool(extents["extent_ok"].all()),
            "total_within_ceiling": len(site_dates) <= CEILINGS["total_site_dates"],
            "le07_within_ceiling": n_le07_sd <= CEILINGS["LE07_site_dates"],
            "oli_within_ceiling": n_oli_sd <= CEILINGS["OLI_site_dates"],
            "payloads_keep_le07_et": all(
                "etm7_collection_2_l2" in json.load(open(p))
                and json.load(open(p))["etm7_collection_2_l2"]["products"] == ["et"]
                for p in repair_manifest.loc[repair_manifest["n_le07"] > 0, "payload_json"]
            ),
            "cap_safe_rounds": bool(
                (repair_manifest.groupby("round")["n_scenes"].sum() <= OPEN_UNIT_CAP).all()
            ),
            "availability_sample_all_et": bool(avail) and all(a["et_advertised"] for a in avail),
            "reconciles": int(len(cand)) == int(len(classified)) + n_exact_dupes,
            "alternates_are_same_overpass": alt_same_overpass,
            "A2_granted": bool(args.a2_granted),
        },
    }
    audit["gate"]["ready_for_A2"] = all(v for k, v in audit["gate"].items() if k != "A2_granted")
    with open(os.path.join(args.out_dir, "le07_manifest_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2, default=str)
    show = {k: v for k, v in audit["counts"].items() if not isinstance(v, dict) or len(v) < 20}
    print(json.dumps({"counts": show, "gate": audit["gate"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

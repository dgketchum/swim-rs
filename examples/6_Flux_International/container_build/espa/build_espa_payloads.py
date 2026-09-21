# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
# ]
# ///
"""Turn manifest rows into ESPA API order payloads.

Reads the manifest CSV, scene CSVs, and extent CSVs to produce one JSON
payload per site-year, ready for submission to the ESPA API.

Example:
    uv run examples/6_Flux_International/espa/build_espa_payloads.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PRODUCT_ID_RE = re.compile(
    r"^(LT04|LT05|LE07|LC08|LC09)_(L2SP|L2SR)_(\d{6})_(\d{8})_(\d{8})_(\d{2})_(RT|T1|T2)$"
)

SENSOR_TO_COLLECTION = {
    "LC08": "olitirs8_collection_2_l2",
    "LC09": "olitirs9_collection_2_l2",
    "LE07": "etm7_collection_2_l2",
    "LT05": "tm5_collection_2_l2",
    "LT04": "tm4_collection_2_l2",
}


def _sensor_collection_key(sensor: str) -> str:
    base = SENSOR_TO_COLLECTION.get(sensor)
    if not base:
        raise ValueError(f"Unknown sensor: {sensor}")
    return base


def _group_scenes_by_sensor(scene_csv: Path) -> dict[str, list[str]]:
    df = pd.read_csv(scene_csv)
    col = df.columns[0]
    groups: dict[str, list[str]] = {}
    for pid in df[col].dropna().astype(str):
        pid = pid.strip()
        m = PRODUCT_ID_RE.match(pid)
        if not m:
            continue
        sensor = m.group(1)
        key = _sensor_collection_key(sensor)
        groups.setdefault(key, []).append(pid)
    return groups


def _read_extent(extent_csv: Path) -> dict:
    df = pd.read_csv(extent_csv)
    row = df.iloc[0]
    return {
        "minx": float(row["minx"]),
        "miny": float(row["miny"]),
        "maxx": float(row["maxx"]),
        "maxy": float(row["maxy"]),
        "utm_zone": int(row["utm_zone"]),
        "utm_hemisphere": str(row["utm_hemisphere"]),
    }


def _build_projection(extent: dict) -> dict:
    zone = extent["utm_zone"]
    south = extent["utm_hemisphere"] == "south"
    epsg = 32600 + zone if not south else 32700 + zone
    return {"utm": {"zone": zone, "zone_ns": "south" if south else "north"}, "epsg": epsg}


def build_payload(site: str, year: str, scene_csv: Path, extent_csv: Path) -> dict:
    groups = _group_scenes_by_sensor(scene_csv)
    extent = _read_extent(extent_csv)
    proj = _build_projection(extent)

    total_scenes = sum(len(v) for v in groups.values())

    # Validate ECOSTRESS era: only LC08/LC09 expected for 2018+
    year_int = int(year)
    if year_int >= 2018:
        for pid_list in groups.values():
            for pid in pid_list:
                sensor = pid.split("_")[0]
                if sensor not in ("LC08", "LC09"):
                    print(f"  WARN: non-OLI scene {pid} in ECOSTRESS era for {site}/{year}")

    payload: dict = {
        "projection": {"utm": proj["utm"]},
        "image_extents": {
            "north": extent["maxy"],
            "south": extent["miny"],
            "east": extent["maxx"],
            "west": extent["minx"],
            "units": "meters",
        },
        "format": "gtiff",
        "resampling_method": "nn",
        "note": f"{site}_{year}",
    }

    for collection_key, product_ids in groups.items():
        payload[collection_key] = {
            "inputs": sorted(product_ids),
            "products": ["et"],
        }

    return payload, total_scenes


def build_all_payloads(manifest_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype=str)
    payloads_dir = manifest_path.parent / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)

    updated_rows = []
    for _, row in manifest.iterrows():
        site = row["site"]
        year = row["year"]
        scene_csv = Path(row["scene_csv"]) if row.get("scene_csv") else None
        extent_csv = Path(row["extent_csv"]) if row.get("extent_csv") else None

        if not scene_csv or not scene_csv.exists():
            print(f"  SKIP {site}/{year}: no scene CSV")
            updated_rows.append(row.to_dict())
            continue
        if not extent_csv or not extent_csv.exists():
            print(f"  SKIP {site}/{year}: no extent CSV")
            updated_rows.append(row.to_dict())
            continue

        payload_path = payloads_dir / f"{site}_{year}_payload.json"
        try:
            payload, n = build_payload(site, year, scene_csv, extent_csv)
            expected = int(row.get("n_scenes", 0))
            if n != expected:
                print(f"  WARN {site}/{year}: payload has {n} scenes, manifest says {expected}")

            with open(payload_path, "w") as f:
                json.dump(payload, f, indent=2)

            row_dict = row.to_dict()
            row_dict["payload_json"] = str(payload_path)
            updated_rows.append(row_dict)
            print(f"  {site}/{year}: {n} scenes -> {payload_path.name}")
        except Exception as e:
            print(f"  ERROR {site}/{year}: {e}")
            updated_rows.append(row.to_dict())

    updated = pd.DataFrame(updated_rows)
    updated.to_csv(manifest_path, index=False)
    print(f"\nUpdated manifest: {manifest_path}")
    print(f"Payloads written: {len(list(payloads_dir.glob('*.json')))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ESPA API payloads from manifest")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    args = parser.parse_args()
    build_all_payloads(Path(args.manifest))


if __name__ == "__main__":
    main()

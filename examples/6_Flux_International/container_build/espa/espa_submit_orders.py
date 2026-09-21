# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "requests",
# ]
# ///
"""Submit ESPA orders with bounded concurrency.

Reads the manifest, loads payload JSONs, and submits orders to the ESPA API
with a configurable cap on active orders. Updates the manifest with order IDs
and status.

Example:
    uv run examples/6_Flux_International/espa/espa_submit_orders.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv \
        --max-active 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ESPA_API = "https://espa.cr.usgs.gov/api/v1"
SUBMIT_SLEEP = 5  # seconds between submissions


def _load_credentials(cred_file: Path) -> tuple[str, str]:
    text = cred_file.read_text().strip()
    if ":" in text:
        user, passwd = text.split(":", 1)
    else:
        lines = text.splitlines()
        user = lines[0].strip()
        passwd = lines[1].strip() if len(lines) > 1 else ""
    return user, passwd


def _count_open_units(manifest: pd.DataFrame) -> int:
    """Count scenes in orders that haven't finished downloading.

    ESPA limits users to 10,000 open units (scenes) in processing. We count
    from the manifest rather than hitting the API per-order, which doesn't
    scale.
    """
    has_order = manifest["order_id"].notna() & (manifest["order_id"] != "")
    not_done = manifest["download_status"] != "complete"
    open_rows = manifest[has_order & not_done]
    return int(open_rows["n_scenes"].fillna(0).astype(int).sum())


def _submit_order(auth: tuple[str, str], payload: dict) -> dict:
    resp = requests.post(
        f"{ESPA_API}/order",
        auth=auth,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def submit_orders(
    manifest_path: Path,
    cred_file: Path,
    max_open_units: int = 10000,
    dry_run: bool = False,
) -> None:
    auth = _load_credentials(cred_file)
    manifest = pd.read_csv(manifest_path, dtype=str)
    log_path = manifest_path.parent / "submission_log.json"
    log_entries: list[dict] = []

    # Ensure state columns exist
    for col in ["retry_count", "last_error"]:
        if col not in manifest.columns:
            manifest[col] = ""

    # Find submittable rows: have a payload, no order_id yet (or retryable failures)
    # Exclude deferred rows regardless of other conditions
    not_deferred = ~manifest["order_status"].str.startswith("deferred", na=False)
    has_payload = manifest["payload_json"].notna() & (manifest["payload_json"] != "")
    no_order = manifest["order_id"].isna() | (manifest["order_id"] == "")
    retryable = manifest["order_status"] == "submit_failed"
    submittable = manifest[not_deferred & has_payload & (no_order | retryable)]

    if submittable.empty:
        print("No submittable rows found.")
        return

    print(f"Submittable site-years: {len(submittable)}")

    if not dry_run:
        open_units = _count_open_units(manifest)
        print(f"Open units (scenes in processing): {open_units}/{max_open_units}")
    else:
        open_units = 0

    submitted = 0
    for idx, row in submittable.iterrows():
        site = row["site"]
        year = row["year"]
        raw = row.get("n_scenes")
        n_scenes = 0 if pd.isna(raw) or raw == "" else int(raw)

        if not dry_run and open_units + n_scenes > max_open_units:
            print(f"Unit cap ({max_open_units}) would be exceeded. Stopping submissions.")
            break

        payload_path = Path(row["payload_json"])
        if not payload_path.exists():
            print(f"  SKIP {site}/{year}: payload file missing")
            continue

        with open(payload_path) as f:
            payload = json.load(f)

        if dry_run:
            print(f"  DRY-RUN {site}/{year}: would submit {payload_path.name}")
            continue

        try:
            result = _submit_order(auth, payload)
            order_id = result.get("orderid", result.get("order_id", ""))
            status = result.get("status", "submitted")
            manifest.at[idx, "order_id"] = str(order_id)
            manifest.at[idx, "order_status"] = str(status)
            log_entries.append(
                {
                    "site": site,
                    "year": year,
                    "order_id": order_id,
                    "response": result,
                }
            )
            open_units += n_scenes
            submitted += 1
            print(f"  SUBMITTED {site}/{year}: {order_id}")
            time.sleep(SUBMIT_SLEEP)
        except requests.HTTPError as e:
            msg = str(e)
            manifest.at[idx, "order_status"] = "submit_failed"
            manifest.at[idx, "last_error"] = msg[:200]
            rc = row.get("retry_count")
            prev = 0 if pd.isna(rc) or rc == "" else int(rc)
            manifest.at[idx, "retry_count"] = str(prev + 1)
            log_entries.append(
                {
                    "site": site,
                    "year": year,
                    "error": msg,
                }
            )
            print(f"  FAILED {site}/{year}: {msg}")

    manifest.to_csv(manifest_path, index=False)

    if log_entries:
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)

    print(f"\nSubmitted: {submitted}")
    print(f"Manifest updated: {manifest_path}")
    if log_entries:
        print(f"Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit ESPA orders from manifest")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    parser.add_argument(
        "--credentials",
        default=str(Path.home() / "usgs_pswd.txt"),
        help="USGS credentials file (user:password)",
    )
    parser.add_argument(
        "--max-open-units", type=int, default=10000, help="Max scenes in processing (ESPA limit)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted")
    args = parser.parse_args()

    submit_orders(
        manifest_path=Path(args.manifest),
        cred_file=Path(args.credentials),
        max_open_units=args.max_open_units,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

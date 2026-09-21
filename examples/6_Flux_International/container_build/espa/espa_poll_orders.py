# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "requests",
# ]
# ///
"""Poll ESPA order and item status, update manifest.

Checks order-status and item-status endpoints, collapses per-item states
into summary counts, and marks rows ready for download.

Example:
    uv run examples/6_Flux_International/espa/espa_poll_orders.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ESPA_API = "https://espa.cr.usgs.gov/api/v1"


def _load_credentials(cred_file: Path) -> tuple[str, str]:
    text = cred_file.read_text().strip()
    if ":" in text:
        user, passwd = text.split(":", 1)
    else:
        lines = text.splitlines()
        user = lines[0].strip()
        passwd = lines[1].strip() if len(lines) > 1 else ""
    return user, passwd


def _get_order_status(auth: tuple[str, str], order_id: str) -> dict:
    resp = requests.get(f"{ESPA_API}/order/{order_id}", auth=auth, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _get_item_status(auth: tuple[str, str], order_id: str) -> dict:
    resp = requests.get(f"{ESPA_API}/item-status/{order_id}", auth=auth, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _summarize_items(item_status: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    # item_status response: {order_id: [{name, status, ...}, ...]}
    for _order_id, items in item_status.items():
        for item in items:
            st = item.get("status", "unknown")
            counts[st] = counts.get(st, 0) + 1
    return counts


def poll_orders(manifest_path: Path, cred_file: Path) -> None:
    auth = _load_credentials(cred_file)
    manifest = pd.read_csv(manifest_path, dtype=str)
    status_dir = manifest_path.parent / "order_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    # Ensure status count columns exist
    for col in ["n_complete", "n_processing", "n_tasked", "n_failed", "last_checked"]:
        if col not in manifest.columns:
            manifest[col] = ""

    has_order = manifest["order_id"].notna() & (manifest["order_id"] != "")
    terminal = manifest["order_status"].isin(["complete", "ready_for_download"])
    pollable = manifest[has_order & ~terminal]

    if pollable.empty:
        print("No pollable orders found.")
        return

    print(f"Polling {len(pollable)} orders...")
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    for idx, row in pollable.iterrows():
        site = row["site"]
        year = row["year"]
        order_id = row["order_id"]

        try:
            order_data = _get_order_status(auth, order_id)
            order_st = order_data.get("status", "unknown")

            item_data = _get_item_status(auth, order_id)
            counts = _summarize_items(item_data)

            manifest.at[idx, "order_status"] = order_st
            manifest.at[idx, "n_complete"] = str(counts.get("complete", 0))
            manifest.at[idx, "n_processing"] = str(counts.get("processing", 0))
            manifest.at[idx, "n_tasked"] = str(counts.get("tasked", 0) + counts.get("queued", 0))
            manifest.at[idx, "n_failed"] = str(
                counts.get("error", 0) + counts.get("unavailable", 0)
            )
            manifest.at[idx, "last_checked"] = now

            total_items = sum(counts.values())
            n_done = counts.get("complete", 0) + counts.get("oncache", 0)
            if order_st == "complete" or (total_items > 0 and n_done == total_items):
                manifest.at[idx, "order_status"] = "ready_for_download"

            # Save per-order status snapshot
            snapshot = {
                "order_id": order_id,
                "site": site,
                "year": year,
                "order_status": order_st,
                "item_counts": counts,
                "checked_at": now,
            }
            snap_path = status_dir / f"{site}_{year}_status.json"
            with open(snap_path, "w") as f:
                json.dump(snapshot, f, indent=2)

            print(f"  {site}/{year}: {order_st}  complete={n_done}/{total_items}")

        except requests.HTTPError as e:
            print(f"  ERROR {site}/{year} ({order_id}): {e}")
            manifest.at[idx, "last_checked"] = now

    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest updated: {manifest_path}")

    ready = manifest[manifest["order_status"] == "ready_for_download"]
    print(f"Ready for download: {len(ready)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll ESPA order status")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    parser.add_argument(
        "--credentials",
        default=str(Path.home() / "usgs_pswd.txt"),
        help="USGS credentials file",
    )
    args = parser.parse_args()

    poll_orders(
        manifest_path=Path(args.manifest),
        cred_file=Path(args.credentials),
    )


if __name__ == "__main__":
    main()

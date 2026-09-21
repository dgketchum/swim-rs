# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "requests",
#   "geopandas",
#   "rasterio",
#   "rasterstats",
#   "shapely",
#   "fiona",
# ]
# ///
"""Autonomous ESPA pipeline runner.

Cycles through poll -> download -> extract -> write_csvs -> submit_more
until the entire cohort is done.

Example:
    uv run examples/6_Flux_International/espa/espa_run_queue.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv \
        --max-active 10 --sleep 300

    # Single pass (no loop):
    uv run examples/6_Flux_International/espa/espa_run_queue.py \
        --manifest ... --once
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_ESPA_DIR = Path(__file__).resolve().parent


def _import(name: str):
    spec = importlib.util.spec_from_file_location(name, _ESPA_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_poll = _import("espa_poll_orders")
_download = _import("espa_download_orders")
_extract = _import("espa_extract_etf")
_csv = _import("espa_write_etf_csvs")
_submit = _import("espa_submit_orders")


def _cohort_summary(manifest_path: Path) -> dict:
    mf = pd.read_csv(manifest_path, dtype=str)
    total = len(mf)
    has_scenes = mf["n_scenes"].astype(int) > 0
    no_scenes = (~has_scenes).sum()
    pending_submit = has_scenes & (mf["order_id"].isna() | (mf["order_id"] == ""))
    submitted = mf["order_id"].notna() & (mf["order_id"] != "")
    ready = mf["order_status"] == "ready_for_download"
    downloaded = mf["download_status"] == "complete"
    extracted = mf["extract_status"] == "etf_extracted"
    terminal_etf = mf["extract_status"].isin(["etf_extracted", "no_etf"])
    csv_col = mf["csv_status"] if "csv_status" in mf.columns else pd.Series("", index=mf.index)
    csv_done = (csv_col == "written") | (mf["extract_status"] == "no_etf")
    return {
        "total": int(total),
        "no_scenes": int(no_scenes.sum()) if hasattr(no_scenes, "sum") else int(no_scenes),
        "pending_submit": int(pending_submit.sum()),
        "submitted": int(submitted.sum()),
        "ready": int(ready.sum()),
        "downloaded": int(downloaded.sum()),
        "extracted": int(extracted.sum()),
        "terminal": int(terminal_etf.sum()),
        "csv_done": int(csv_done.sum()),
    }


def _is_done(summary: dict) -> bool:
    actionable = summary["total"] - summary["no_scenes"]
    return actionable > 0 and summary["csv_done"] >= actionable


def run_queue(
    manifest_path: Path,
    cred_file: Path,
    shapefile: Path,
    max_open_units: int = 10000,
    sleep_seconds: int = 300,
    once: bool = False,
) -> None:
    iteration = 0

    while True:
        iteration += 1
        now = datetime.now(UTC).strftime("%H:%M:%S")
        print(f"\n{'=' * 60}", flush=True)
        print(f"[{now}] Cycle {iteration}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # 1. Poll active orders
        print("\n--- Poll ---", flush=True)
        try:
            _poll.poll_orders(manifest_path, cred_file)
        except Exception as e:
            print(f"  Poll error: {e}", flush=True)

        # 2. Download completed orders
        print("\n--- Download ---", flush=True)
        try:
            _download.download_orders(manifest_path, cred_file)
        except Exception as e:
            print(f"  Download error: {e}", flush=True)

        # 3. Extract ETF from rasters
        print("\n--- Extract ---", flush=True)
        try:
            _extract.extract_all(manifest_path, shapefile)
        except Exception as e:
            print(f"  Extract error: {e}", flush=True)

        # 4. Write ingest-ready CSVs
        print("\n--- Write CSVs ---", flush=True)
        try:
            _csv.write_csvs(manifest_path)
        except Exception as e:
            print(f"  CSV error: {e}", flush=True)

        # 5. Submit more orders if under cap
        print("\n--- Submit ---", flush=True)
        try:
            _submit.submit_orders(manifest_path, cred_file, max_open_units=max_open_units)
        except Exception as e:
            print(f"  Submit error: {e}", flush=True)

        # 6. Summary
        summary = _cohort_summary(manifest_path)
        print("\n--- Status ---", flush=True)
        print(
            f"  total={summary['total']}  no_scenes={summary['no_scenes']}  "
            f"pending={summary['pending_submit']}  submitted={summary['submitted']}  "
            f"ready={summary['ready']}  downloaded={summary['downloaded']}  "
            f"extracted={summary['extracted']}",
            flush=True,
        )

        if _is_done(summary):
            print("\nAll site-years extracted. Done.", flush=True)
            break

        if once:
            print("\nSingle pass complete (--once).", flush=True)
            break

        print(f"\nSleeping {sleep_seconds}s...", flush=True)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESPA autonomous pipeline runner")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    parser.add_argument(
        "--credentials",
        default=str(Path.home() / "usgs_pswd.txt"),
        help="USGS credentials file",
    )
    parser.add_argument(
        "--shapefile",
        default="/data/ssd1/swim/6_Flux_International/data/gis/flux_intl_150m_23MAR2026.shp",
        help="Example 6 shapefile",
    )
    parser.add_argument(
        "--max-open-units", type=int, default=10000, help="Max scenes in processing (ESPA limit)"
    )
    parser.add_argument("--sleep", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    args = parser.parse_args()

    run_queue(
        manifest_path=Path(args.manifest),
        cred_file=Path(args.credentials),
        shapefile=Path(args.shapefile),
        max_open_units=args.max_open_units,
        sleep_seconds=args.sleep,
        once=args.once,
    )


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "requests",
# ]
# ///
"""Download completed ESPA orders, verify checksums, and extract.

For each manifest row marked ready_for_download, downloads the bundle
tarballs, verifies md5 checksums, and extracts into a per-site/year
directory.

Example:
    uv run examples/6_Flux_International/espa/espa_download_orders.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

ESPA_API = "https://espa.cr.usgs.gov/api/v1"
CHUNK_SIZE = 8192


def _load_credentials(cred_file: Path) -> tuple[str, str]:
    text = cred_file.read_text().strip()
    if ":" in text:
        user, passwd = text.split(":", 1)
    else:
        lines = text.splitlines()
        user = lines[0].strip()
        passwd = lines[1].strip() if len(lines) > 1 else ""
    return user, passwd


def _get_item_status(auth: tuple[str, str], order_id: str) -> dict:
    resp = requests.get(f"{ESPA_API}/item-status/{order_id}", auth=auth, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _download_file(url: str, dest: Path, auth: tuple[str, str], max_retries: int = 4) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        resp = requests.get(url, auth=auth, stream=True, timeout=300)
        if resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"      429 rate-limited, retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return True
    resp.raise_for_status()  # raise on final failure
    return False


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(tarball: Path, cksum_path: Path) -> bool:
    if not cksum_path.exists():
        return True  # no checksum available, proceed
    expected = cksum_path.read_text().strip().split()[0]
    actual = _md5_file(tarball)
    return actual == expected


def _extract_tarball(tarball: Path, extract_dir: Path) -> bool:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    return True


def download_orders(
    manifest_path: Path,
    cred_file: Path,
    keep_tarballs: bool = True,
) -> None:
    auth = _load_credentials(cred_file)
    manifest = pd.read_csv(manifest_path, dtype=str)

    downloadable = manifest[
        (manifest["order_status"] == "ready_for_download")
        & (manifest["download_status"] != "complete")
    ]

    if downloadable.empty:
        print("No orders ready for download.")
        return

    print(f"Downloading {len(downloadable)} orders...")

    for idx, row in downloadable.iterrows():
        site = row["site"]
        year = row["year"]
        order_id = row["order_id"]
        output_dir = Path(row["output_dir"])
        raw_dir = output_dir / "raw"
        extract_dir = output_dir / "extract"
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            item_data = _get_item_status(auth, order_id)
            items = []
            for _oid, item_list in item_data.items():
                items.extend(item_list)

            complete_items = [it for it in items if it.get("status") == "complete"]
            if not complete_items:
                print(f"  SKIP {site}/{year}: no complete items")
                continue

            n_downloaded = 0
            n_failed = 0
            for item in complete_items:
                product_url = item.get("product_dload_url", "")
                cksum_url = item.get("cksum_download_url", "")
                name = item.get("name", "unknown")

                tarball_name = f"{name}.tar.gz"
                tarball_path = raw_dir / tarball_name
                cksum_path = raw_dir / f"{name}.md5"

                # Skip if already downloaded and verified
                if tarball_path.exists():
                    if cksum_path.exists() and _verify_checksum(tarball_path, cksum_path):
                        n_downloaded += 1
                        continue
                    elif not cksum_path.exists():
                        n_downloaded += 1
                        continue

                # Clear stale sentinel — we're redownloading, so prior extraction is invalid
                stale_sentinel = output_dir / ".extract_done" / f"{tarball_path.stem}.done"
                if stale_sentinel.exists():
                    stale_sentinel.unlink()

                # Download checksum first
                if cksum_url:
                    try:
                        _download_file(cksum_url, cksum_path, auth)
                    except requests.HTTPError:
                        pass  # proceed without checksum

                # Download tarball
                if not product_url:
                    print(f"    SKIP {name}: no download URL")
                    n_failed += 1
                    continue

                try:
                    _download_file(product_url, tarball_path, auth)
                except requests.HTTPError as e:
                    print(f"    FAILED {name}: {e}")
                    n_failed += 1
                    continue

                # Verify — move bad tarballs out of raw/ so extraction won't touch them
                if cksum_path.exists() and not _verify_checksum(tarball_path, cksum_path):
                    bad_dir = output_dir / "bad"
                    bad_dir.mkdir(parents=True, exist_ok=True)
                    tarball_path.rename(bad_dir / tarball_path.name)
                    # Clear any stale extraction sentinel for this tarball
                    stale_sentinel = output_dir / ".extract_done" / f"{tarball_path.stem}.done"
                    if stale_sentinel.exists():
                        stale_sentinel.unlink()
                    print(f"    CHECKSUM FAILED {name} (quarantined)")
                    n_failed += 1
                    continue

                n_downloaded += 1

            # Extract tarballs not yet extracted (sentinel-based)
            sentinel_dir = output_dir / ".extract_done"
            sentinel_dir.mkdir(parents=True, exist_ok=True)
            n_extracted = 0
            for tb in raw_dir.glob("*.tar.gz"):
                sentinel = sentinel_dir / f"{tb.stem}.done"
                if sentinel.exists():
                    n_extracted += 1
                    continue
                try:
                    _extract_tarball(tb, extract_dir)
                    sentinel.touch()
                    n_extracted += 1
                    if not keep_tarballs:
                        tb.unlink()
                except (tarfile.TarError, OSError) as e:
                    print(f"    EXTRACT FAILED {tb.name}: {e}")

            all_ok = n_failed == 0 and n_downloaded == len(complete_items)
            manifest.at[idx, "download_status"] = "complete" if all_ok else "partial"
            manifest.at[idx, "extract_status"] = ""
            print(
                f"  {site}/{year}: downloaded={n_downloaded}/{len(complete_items)} "
                f"extracted={n_extracted} failed={n_failed}"
            )

        except requests.HTTPError as e:
            print(f"  ERROR {site}/{year}: {e}")
            manifest.at[idx, "download_status"] = "error"
            manifest.at[idx, "last_error"] = str(e)[:200]

    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest updated: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download completed ESPA orders")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    parser.add_argument(
        "--credentials",
        default=str(Path.home() / "usgs_pswd.txt"),
        help="USGS credentials file",
    )
    parser.add_argument("--cleanup", action="store_true", help="Remove tarballs after extraction")
    args = parser.parse_args()

    download_orders(
        manifest_path=Path(args.manifest),
        cred_file=Path(args.credentials),
        keep_tarballs=not args.cleanup,
    )


if __name__ == "__main__":
    main()

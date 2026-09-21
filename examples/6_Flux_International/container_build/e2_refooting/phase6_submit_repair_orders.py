"""Phase 6, step 1 (approval A2 granted 2026-09-04): snapshot, verify and submit the repair orders.

Operates only on the missing-only repair tree written by ``phase5_build_missing_only_manifest.py``
(``remote_sensing/espa_le07_repair/``). Plan §12 rules applied here:

* the manifest is reloaded from disk and rewritten after every submission, so a crash leaves
  the true state on disk;
* the G5 audit must say ``ready_for_A2`` and ``A2_granted`` (``--a2-granted`` on the builder);
* every payload is hashed and compared to the manifest's ``payload_sha256`` before anything is
  sent; the approved hashes are snapshotted (``le07_submission_snapshot_{ts}.json``);
* the 10,000 open-unit cap is checked against **server** state (``list-orders`` filtered to
  ``ordered`` plus ``item-status`` per open order), not only local counts;
* order IDs and the verbatim server response are recorded row-for-row (manifest and
  ``le07_submission_log_{ts}.json``), and propagated to the scene-level
  ``le07_missing_only_manifest.csv`` through ``payload_id``;
* an HTTP error marks the row ``submit_failed`` with the response text; nothing about the payload
  (sensors, products, extents) is ever changed here.

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase6_submit_repair_orders.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time

import pandas as pd
import requests

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
REPAIR_TREE = os.path.join(DATA, "remote_sensing", "espa_le07_repair")
SCENE_MANIFEST = os.path.join(QA_ROOT, "le07_missing_only_manifest.csv")
AUDIT = os.path.join(QA_ROOT, "le07_manifest_audit.json")
CRED_FILE = os.path.expanduser("~/usgs_pswd.txt")
ESPA_API = "https://espa.cr.usgs.gov/api/v1"
OPEN_UNIT_CAP = 10000
SUBMIT_SLEEP = 5  # seconds between POSTs, as in espa/espa_submit_orders.py
TERMINAL_ITEM_STATES = {"complete", "unavailable", "error", "purged", "cancelled"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_credentials(path: str = CRED_FILE) -> tuple[str, str]:
    text = open(path).read().strip()
    if ":" in text:
        user, pw = text.split(":", 1)
        return user, pw
    lines = text.splitlines()
    return lines[0].strip(), lines[1].strip()


def verify_payload_hashes(manifest: pd.DataFrame) -> list[dict]:
    """Hash every payload on disk; raise if any differs from the manifest."""
    rows, bad = [], []
    for _, r in manifest.iterrows():
        h = sha256_file(r["payload_json"])
        rows.append(
            {"site": r["site"], "year": r["year"], "payload_json": r["payload_json"], "sha256": h}
        )
        if h != r["payload_sha256"]:
            bad.append((r["payload_json"], r["payload_sha256"], h))
    if bad:
        raise SystemExit(f"{len(bad)} payload(s) differ from the manifest hashes: {bad[:3]}")
    return rows


def server_open_units(auth: tuple[str, str]) -> dict:
    """Units still in processing on the ESPA side (orders in status ``ordered``)."""
    resp = requests.get(
        f"{ESPA_API}/list-orders", auth=auth, json={"status": "ordered"}, timeout=120
    )
    resp.raise_for_status()
    open_orders = resp.json()
    if not isinstance(open_orders, list):
        raise SystemExit(f"unexpected list-orders response: {str(open_orders)[:200]}")
    units, per_order = 0, {}
    for oid in open_orders:
        r = requests.get(f"{ESPA_API}/item-status/{oid}", auth=auth, timeout=120)
        r.raise_for_status()
        items = r.json().get(oid, [])
        n_open = sum(1 for it in items if it.get("status") not in TERMINAL_ITEM_STATES)
        per_order[oid] = {"items": len(items), "open": n_open}
        units += n_open
    return {"open_orders": len(open_orders), "open_units": units, "per_order": per_order}


def submit_payload(auth: tuple[str, str], payload: dict) -> dict:
    resp = requests.post(f"{ESPA_API}/order", auth=auth, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"{resp.status_code}: {resp.text[:500]}", response=resp)
    return resp.json()


def propagate_to_scene_manifest(scene_manifest_path: str, manifest: pd.DataFrame) -> None:
    """Copy order_id / order_status / submitted_at from site-year rows to scene rows."""
    scenes = pd.read_csv(scene_manifest_path, dtype=str, keep_default_na=False)
    by_pid = manifest.assign(payload_id=manifest["site"] + "_" + manifest["year"].astype(str))
    by_pid = by_pid.set_index("payload_id")
    for col in ("order_id", "order_status", "submitted_at"):
        if col not in scenes.columns:
            scenes[col] = ""
        mapped = scenes["payload_id"].map(by_pid[col]) if col in by_pid.columns else None
        if mapped is not None:
            has = scenes["payload_id"].isin(by_pid.index)
            scenes.loc[has, col] = mapped[has].fillna("").astype(str)
    scenes.to_csv(scene_manifest_path, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repair-tree", default=REPAIR_TREE)
    ap.add_argument("--scene-manifest", default=SCENE_MANIFEST)
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--out-dir", default=QA_ROOT)
    ap.add_argument("--credentials", default=CRED_FILE)
    ap.add_argument("--open-unit-cap", type=int, default=OPEN_UNIT_CAP)
    ap.add_argument("--dry-run", action="store_true", help="verify, snapshot and count; no POST")
    args = ap.parse_args(argv)

    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = os.path.join(args.repair_tree, "espa_manifest.csv")

    with open(args.audit) as fh:
        audit = json.load(fh)
    gate = audit["gate"]
    if not (gate.get("ready_for_A2") and gate.get("A2_granted")):
        raise SystemExit(
            f"G5 audit not cleared for submission: ready_for_A2={gate.get('ready_for_A2')} "
            f"A2_granted={gate.get('A2_granted')} ({args.audit})"
        )

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    for col in ("order_id", "order_status", "submitted_at", "last_error", "retry_count"):
        if col not in manifest.columns:
            manifest[col] = ""
    manifest["n_scenes"] = manifest["n_scenes"].astype(int)

    payload_hashes = verify_payload_hashes(manifest)
    snapshot = {
        "timestamp": ts,
        "manifest": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "audit": args.audit,
        "audit_sha256": sha256_file(args.audit),
        "audit_generated": audit.get("generated"),
        "A2": audit.get("approvals", {}).get("A2"),
        "n_payloads": int(len(manifest)),
        "n_units": int(manifest["n_scenes"].sum()),
        "payloads": payload_hashes,
    }
    snapshot_path = os.path.join(args.out_dir, f"le07_submission_snapshot_{ts}.json")
    with open(snapshot_path, "w") as fh:
        json.dump(snapshot, fh, indent=2)

    auth = load_credentials(args.credentials)
    server = server_open_units(auth)
    print(
        f"server open orders {server['open_orders']}, open units {server['open_units']}; "
        f"local payloads {len(manifest)}, units {int(manifest['n_scenes'].sum())}"
    )

    todo = manifest[(manifest["order_id"] == "") | (manifest["order_status"] == "submit_failed")]
    print(f"submittable site-years: {len(todo)}{' (dry run)' if args.dry_run else ''}")
    log_path = os.path.join(args.out_dir, f"le07_submission_log_{ts}.json")
    log: list[dict] = [{"snapshot": os.path.basename(snapshot_path), "server_state": server}]
    open_units = server["open_units"]
    submitted = failed = 0

    for idx, row in todo.iterrows():
        n = int(row["n_scenes"])
        if open_units + n > args.open_unit_cap:
            print(f"  STOP: {open_units} + {n} would exceed the {args.open_unit_cap} unit cap")
            log.append({"stop": "open_unit_cap", "open_units": open_units, "next_units": n})
            break
        with open(row["payload_json"]) as fh:
            payload = json.load(fh)
        if args.dry_run:
            print(f"  DRY-RUN {row['site']}/{row['year']}: {n} units, {row['payload_sha256'][:12]}")
            continue
        now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        try:
            result = submit_payload(auth, payload)
            order_id = str(result.get("orderid", result.get("order_id", "")))
            manifest.at[idx, "order_id"] = order_id
            manifest.at[idx, "order_status"] = str(result.get("status", "submitted"))
            manifest.at[idx, "submitted_at"] = now
            manifest.at[idx, "last_error"] = ""
            open_units += n
            submitted += 1
            log.append(
                {
                    "site": row["site"],
                    "year": row["year"],
                    "payload_sha256": row["payload_sha256"],
                    "order_id": order_id,
                    "submitted_at": now,
                    "response": result,
                }
            )
            print(f"  SUBMITTED {row['site']}/{row['year']}: {order_id} ({n} units)")
        except requests.HTTPError as e:
            msg = str(e)
            manifest.at[idx, "order_status"] = "submit_failed"
            manifest.at[idx, "last_error"] = msg[:300]
            prev = row["retry_count"]
            manifest.at[idx, "retry_count"] = str((int(prev) if prev else 0) + 1)
            failed += 1
            log.append(
                {
                    "site": row["site"],
                    "year": row["year"],
                    "payload_sha256": row["payload_sha256"],
                    "error": msg,
                    "at": now,
                }
            )
            print(f"  FAILED {row['site']}/{row['year']}: {msg[:160]}")
        manifest.to_csv(manifest_path, index=False)
        with open(log_path, "w") as fh:
            json.dump(log, fh, indent=2, default=str)
        time.sleep(SUBMIT_SLEEP)

    if not args.dry_run:
        manifest.to_csv(manifest_path, index=False)
        propagate_to_scene_manifest(args.scene_manifest, manifest)
        with open(log_path, "w") as fh:
            json.dump(log, fh, indent=2, default=str)
    print(f"\nsubmitted {submitted}, failed {failed}, open units now {open_units}")
    print(f"manifest: {manifest_path}")
    print(f"log: {log_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

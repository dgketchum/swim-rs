"""Correct the next-day reference-ET divisor in the Example 5 ETf extract tables.

``data_extract.py`` (before the calendar-day join fix) divided PT-JPL, geeSEBAL
and DisALEXI scene ET by the OpenET reference ETo image of the *following*
calendar day: the scene ``system:time_start`` carries the UTC acquisition hour,
and a one-day window opened there returns the next 00:00 image. The stored
fraction is therefore ET(d) / ETo(d+1). The three EToF-native members
(SSEBop, SIMS, eeMETRIC) are unaffected.

This script rewrites the affected wide tables as

    etf_fixed(d) = etf_stored(d) * ETo(d+1) / ETo(d)

using the calendar-stamped OpenET ETo table (``openet_refet/openet_eto.csv``),
which is the same collection the extractor divided by. Nothing is filtered
here: the builder (``build_container.ingest_new_etf``) applies the
[0.05, 2.0] validity window at ingest, so correcting before the filter lets
values that were wrongly dropped come back and drops values that were wrongly
kept. The output directory mirrors the input directory (unchanged members and
summaries are copied), so it can be passed straight to
``build_container.py --etf-dir``.

Usage:
    uv run python correct_eto_offset.py --etf-dir data/etf_v21_openet_eto \\
        --eto-csv data/openet_refet/openet_eto.csv \\
        --out-dir data/etf_v21_openet_eto_fixed
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_container import (  # noqa: E402
    ETF_END,
    ETF_START,
    MAX_VALID_ETF,
    MODELS,
    _read_etf_csv_max,
)

MIN_VALID_ETF = 0.05
AFFECTED_MODELS = ("ptjpl", "geesebal", "disalexi")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def load_eto(csv_path):
    """Read the calendar-stamped ETo table into a dates x sites frame."""
    raw = pd.read_csv(csv_path, index_col=0)
    raw.columns = pd.to_datetime(raw.columns, format="%Y%m%d")
    eto = raw.T.sort_index()
    if eto.index.duplicated().any():
        raise ValueError(f"Duplicate dates in ETo table {csv_path}")
    return eto


def next_day_factor(eto):
    """ETo(d+1) / ETo(d) on a contiguous daily index (NaN where either is missing)."""
    daily = pd.date_range(eto.index.min(), eto.index.max(), freq="D")
    full = eto.reindex(index=daily)
    return full.shift(-1) / full


def correct_frame(df, eto):
    """Undo the next-day divisor on a dates x sites ETf frame.

    Returns ``(corrected, factor)`` with the NaN pattern of ``df`` preserved.
    Raises ``ValueError`` if any valid value lacks a positive ETo on d or d+1.
    """
    daily = pd.date_range(eto.index.min(), eto.index.max(), freq="D")
    full = eto.reindex(index=daily)
    eto_d = full.reindex(index=df.index, columns=df.columns)
    eto_next = full.shift(-1).reindex(index=df.index, columns=df.columns)

    valid = df.notna()
    usable = (eto_d > 0) & (eto_next > 0)
    bad = valid & ~usable
    if bad.to_numpy().any():
        offenders = [
            (site, d.strftime("%Y%m%d"))
            for d, row in bad.iterrows()
            for site, flag in row.items()
            if flag
        ]
        raise ValueError(
            f"{len(offenders)} valid captures lack a positive ETo on d or d+1; "
            f"first: {offenders[:5]}"
        )

    factor = (eto_next / eto_d).where(valid)
    corrected = df * factor
    return corrected, factor


def bound_crossings(original, corrected, start=ETF_START, end=ETF_END):
    """Count validity changes the builder's [0.05, 2.0] filter will see."""
    win = (original.index >= start) & (original.index <= end)
    o = original.loc[win]
    c = corrected.loc[win]
    o_in = (o >= MIN_VALID_ETF) & (o <= MAX_VALID_ETF)
    c_in = (c >= MIN_VALID_ETF) & (c <= MAX_VALID_ETF)
    both = o.notna() & c.notna()
    return {
        "window": [start, end],
        "n_valid_window": int(both.sum().sum()),
        "regained_from_below": int((both & ~o_in & (o < MIN_VALID_ETF) & c_in).sum().sum()),
        "regained_from_above": int((both & ~o_in & (o > MAX_VALID_ETF) & c_in).sum().sum()),
        "lost_below": int((both & o_in & (c < MIN_VALID_ETF)).sum().sum()),
        "lost_above": int((both & o_in & (c > MAX_VALID_ETF)).sum().sum()),
    }


def factor_stats(factor):
    vals = factor.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    return {
        "n": int(vals.size),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "p01": float(np.percentile(vals, 1)),
        "p99": float(np.percentile(vals, 99)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _write_wide(df, path):
    wide = df.T
    wide.columns = [d.strftime("%Y%m%d") for d in wide.columns]
    wide.index.name = "site_id"
    wide.to_csv(path)


def correct_directory(etf_dir, eto_csv, out_dir, models=AFFECTED_MODELS):
    etf_dir = Path(etf_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists (non-clobbering): {out_dir}")
    out_dir.mkdir(parents=True)

    eto = load_eto(eto_csv)
    eto_sha = _sha256(eto_csv)
    git_sha = _git_sha()
    report = {}

    for model in MODELS:
        src_csv = etf_dir / f"{model}_etf_no_mask.csv"
        src_json = etf_dir / f"{model}_summary.json"
        if not src_csv.exists():
            print(f"  {model}: CSV not found, skipping")
            continue
        dst_csv = out_dir / src_csv.name
        if model not in models:
            shutil.copy2(src_csv, dst_csv)
            if src_json.exists():
                shutil.copy2(src_json, out_dir / src_json.name)
            print(f"  {model}: copied unchanged")
            continue

        print(f"  {model}: correcting")
        df = _read_etf_csv_max(src_csv)
        corrected, factor = correct_frame(df, eto)
        _write_wide(corrected, dst_csv)

        entry = {
            "model": model,
            "correction": "etf_fixed(d) = etf_stored(d) * eto(d+1) / eto(d)",
            "eto_csv": str(eto_csv),
            "eto_csv_sha256": eto_sha,
            "source_csv": str(src_csv),
            "source_csv_sha256": _sha256(src_csv),
            "corrected_csv": str(dst_csv),
            "corrected_csv_sha256": _sha256(dst_csv),
            "git_sha": git_sha,
            "n_values": int(corrected.notna().sum().sum()),
            "factor": factor_stats(factor),
            "bound_crossings": bound_crossings(df, corrected),
        }
        report[model] = entry
        with open(out_dir / f"{model}_eto_offset_correction.json", "w") as f:
            json.dump(entry, f, indent=2)

        if src_json.exists():
            with open(src_json) as f:
                summary = json.load(f)
            summary["eto_join"] = "calendar day (offline correction of next-day divisor)"
            summary["eto_offset_correction"] = f"{model}_eto_offset_correction.json"
            with open(out_dir / src_json.name, "w") as f:
                json.dump(summary, f, indent=2)

        fs = entry["factor"]
        bc = entry["bound_crossings"]
        print(
            f"    factor n={fs['n']:,} median={fs['median']:.4f} mean={fs['mean']:.4f} "
            f"sd={fs['sd']:.4f} p01={fs['p01']:.3f} p99={fs['p99']:.3f}"
        )
        print(
            f"    {ETF_START}..{ETF_END}: regained {bc['regained_from_below']}+"
            f"{bc['regained_from_above']}, lost {bc['lost_below']}+{bc['lost_above']}"
        )

    for extra in etf_dir.iterdir():
        if extra.is_file() and not (out_dir / extra.name).exists():
            shutil.copy2(extra, out_dir / extra.name)

    with open(out_dir / "eto_offset_correction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--etf-dir", required=True, help="Directory of {model}_etf_no_mask.csv")
    parser.add_argument("--eto-csv", required=True, help="Calendar-stamped OpenET ETo table")
    parser.add_argument("--out-dir", required=True, help="New directory for corrected tables")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(AFFECTED_MODELS),
        help="ET-denominated members to correct",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.eto_csv):
        raise FileNotFoundError(args.eto_csv)
    correct_directory(args.etf_dir, args.eto_csv, args.out_dir, tuple(args.models))


if __name__ == "__main__":
    main()

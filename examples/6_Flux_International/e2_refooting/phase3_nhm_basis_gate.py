"""Phase 3: NHM reference-basis comparison — a **diagnostic** since 2026-09-04.

Re-runs the established Test C (``notes/ssebop_reference_basis.py``) on the shared sites with
identical eligibility (both members > 0.05; NHM same-date scenes averaged), first for the native
ESPA ETF (must reproduce slope 1.2162 / median ratio 1.213 / r 0.954) and then for the
site/date-corrected grass ETf ``native * etr / eto`` from the daily ERA5-Land sidecar.

History. This script was Gate G3 with hard thresholds (corrected slope and median ratio
0.95–1.05, r >= 0.94, retention >= 95 %). Run on 2026-09-04 it produced slope 0.933 / median
0.939, failing Oct–Apr only. The cause is a convention difference, not a defect: the NHM asset's
grass basis is OpenET's ``et_fraction_type="grass"`` adjustment, which multiplies by the
**overpass-instantaneous** NLDAS-2 ETr/ETo, while SWIM applies ETf to a **daily** ERA5-Land ETo and
therefore needs the daily ratio. The user retained the daily conversion, redefined G3
(``phase3_daily_basis_gate.py``: daily ET consistency + low-ETo stability of the weighted
observations), and demoted this comparison to a reported diagnostic. The original run's artifacts
``ssebop_basis_gate_*`` are frozen evidence and are not rewritten; this script now writes to the
``ssebop_nhm_diagnostic_*`` stems and always exits 0.

Outputs (QA root):
    ssebop_nhm_diagnostic_pairs.csv     every pair: site, date, month, espa_native, ratio, espa_grass, nhm
    ssebop_nhm_diagnostic_summary.json  original thresholds, baseline reproduction, corrected metrics
    ssebop_nhm_diagnostic.png           native vs corrected scatter and per-site slopes

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase3_nhm_basis_gate.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase1_validate_sidecar import load_sidecar  # noqa: E402

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
ESPA_JSON = os.path.join(DATA, "remote_sensing", "espa", "extracts", "etf_json")
SIDECAR_DIR = os.path.join(DATA, "remote_sensing", "espa", "refet_ratio_era5land")
NHM_DIR = (
    "/data/ssd1/swim/4_Flux_Network/data/remote_sensing/landsat/extracts/ssebop_nhm_etf/no_mask"
)

MIN_ETF = 0.05  # identical to the baseline diagnostic
OUTPUT_STEM = "ssebop_nhm_diagnostic"  # the frozen 2026-09-04 gate run used ssebop_basis_gate

BASELINE = {"slope": 1.2162, "median_ratio": 1.213, "r": 0.954, "n_pairs": 1677, "n_sites": 16}
# Original G3 thresholds, kept so the diagnostic reports where the daily basis stands against
# the overpass-basis product; they no longer gate anything (plan §9, 2026-09-04).
THRESHOLDS = {
    "baseline_reproduction_tol": {"slope": 0.001, "median_ratio": 0.001, "r": 0.001},
    "corrected_slope": [0.95, 1.05],
    "corrected_median_ratio": [0.95, 1.05],
    "corrected_r_min": 0.94,
    "corrected_r_max_loss": 0.02,
    "pair_retention_min": 0.95,
}


def fit(x, y) -> dict:
    """Zero-intercept slope (y = s x), median y/x, Pearson r — as in the baseline diagnostic."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return {"n": int(len(x)), "slope": None, "median_ratio": None, "r": None}
    return {
        "n": int(len(x)),
        "slope": float((x * y).sum() / (x * x).sum()),
        "median_ratio": float(np.median(y / x)),
        "r": float(np.corrcoef(x, y)[0, 1]),
    }


def load_espa_native(json_dir: str = ESPA_JSON) -> dict[str, dict[str, float]]:
    out = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(json_dir, "*_etf.json")):
        with open(path) as fh:
            for site, dates in json.load(fh).items():
                for date_key, stats in dates.items():
                    if stats.get("mean") is not None:
                        out[site][date_key.replace("-", "")] = stats["mean"]
    return out


def load_nhm(sites: set[str], nhm_dir: str = NHM_DIR) -> dict[str, dict[str, float]]:
    nhm = collections.defaultdict(lambda: collections.defaultdict(list))
    pattern = re.compile(r"ssebop_etf_(.+)_no_mask_(\d{4})\.csv")
    for path in glob.glob(os.path.join(nhm_dir, "*.csv")):
        m = pattern.match(os.path.basename(path))
        if not m or m.group(1) not in sites:
            continue
        with open(path) as fh:
            reader = csv.reader(fh)
            header = next(reader)
            for row in reader:
                for i, v in enumerate(row):
                    if i == 0 or v in ("", "nan"):
                        continue
                    if not header[i].lower().startswith(("lc0", "le0", "lt0")):
                        continue
                    date = header[i].split("_")[-1]
                    if len(date) == 8 and date.isdigit():
                        nhm[m.group(1)][date].append(float(v))
    return {s: {d: float(np.mean(v)) for d, v in dd.items()} for s, dd in nhm.items()}


def build_pairs(espa: dict, nhm: dict, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Baseline-eligible pairs, with the sidecar ratio attached where available."""
    ratio = {
        (s, d): (e, r)
        for s, d, e, r in zip(
            sidecar["site"], sidecar["date"], sidecar["eto"], sidecar["etr"], strict=True
        )
    }
    rows = []
    for site in sorted(espa):
        for date, v in sorted(espa[site].items()):
            w = nhm.get(site, {}).get(date)
            if w is None or v <= MIN_ETF or w <= MIN_ETF:
                continue
            eto, etr = ratio.get((site, date), (np.nan, np.nan))
            rows.append(
                {
                    "site": site,
                    "date": date,
                    "month": int(date[4:6]),
                    "espa_native": v,
                    "nhm": w,
                    "eto": eto,
                    "etr": etr,
                }
            )
    df = pd.DataFrame(rows)
    ok = np.isfinite(df["eto"]) & (df["eto"] > 0) & np.isfinite(df["etr"]) & (df["etr"] > 0)
    df["ratio"] = np.where(ok, df["etr"] / df["eto"], np.nan)
    df["espa_grass"] = df["espa_native"] * df["ratio"]
    df["resid_native"] = df["nhm"] - df["espa_native"]
    df["resid_grass"] = df["nhm"] - df["espa_grass"]
    return df


def per_group(df: pd.DataFrame, key: str, x: str) -> dict:
    return {
        str(k): fit(g[x], g["nhm"]) | {"median_resid": float((g["nhm"] - g[x]).median())}
        for k, g in df.groupby(key)
    }


def evaluate(pairs: pd.DataFrame, thresholds: dict = THRESHOLDS, baseline: dict = BASELINE) -> dict:
    base = fit(pairs["espa_native"], pairs["nhm"])
    tol = thresholds["baseline_reproduction_tol"]
    base_ok = {k: abs(base[k] - baseline[k]) <= tol[k] for k in tol}

    kept = pairs[np.isfinite(pairs["espa_grass"])]
    corr = fit(kept["espa_grass"], kept["nhm"])
    retention = len(kept) / len(pairs)

    # fixed-scalar comparison: the best single multiplier for the same pairs
    scalar = base["slope"]
    kept = kept.assign(
        espa_scalar=kept["espa_native"] * scalar,
        resid_scalar=kept["nhm"] - kept["espa_native"] * scalar,
    )
    structure = {}
    for label, x, resid in (
        ("grass", "espa_grass", "resid_grass"),
        ("scalar", "espa_scalar", "resid_scalar"),
    ):
        site_med = kept.groupby("site")[resid].median()
        month_med = kept.groupby("month")[resid].median()
        site_slope = pd.Series({s: fit(g[x], g["nhm"])["slope"] for s, g in kept.groupby("site")})
        structure[label] = {
            "resid_sd": float(kept[resid].std()),
            "resid_mad": float((kept[resid] - kept[resid].median()).abs().median()),
            "site_median_resid_sd": float(site_med.std()),
            "month_median_resid_sd": float(month_med.std()),
            "site_slope_sd": float(site_slope.std()),
            "site_slope_range": [float(site_slope.min()), float(site_slope.max())],
        }
    lo, hi = thresholds["corrected_slope"]
    mlo, mhi = thresholds["corrected_median_ratio"]
    diagnostic = {
        "baseline_reproduces": all(base_ok.values()),
        "corrected_slope_within_original_gate": lo <= corr["slope"] <= hi,
        "corrected_median_within_original_gate": mlo <= corr["median_ratio"] <= mhi,
        "corrected_r_within_original_gate": corr["r"] >= thresholds["corrected_r_min"]
        and (base["r"] - corr["r"]) <= thresholds["corrected_r_max_loss"],
        "retention_within_original_gate": retention >= thresholds["pair_retention_min"],
        "better_than_scalar": (
            structure["grass"]["site_median_resid_sd"] < structure["scalar"]["site_median_resid_sd"]
            and structure["grass"]["month_median_resid_sd"]
            < structure["scalar"]["month_median_resid_sd"]
        ),
    }
    diagnostic["all_within_original_gate"] = all(diagnostic.values())
    return {
        "role": "diagnostic (not pass/fail) since 2026-09-04; NHM grass basis is "
        "overpass-instantaneous NLDAS-2 ETr/ETo, SWIM's is daily ERA5-Land",
        "thresholds": thresholds,
        "baseline_expected": baseline,
        "baseline_observed": base
        | {"n_sites": int(pairs["site"].nunique()), "per_metric_ok": base_ok},
        "corrected": corr
        | {
            "n_sites": int(kept["site"].nunique()),
            "pair_retention": retention,
            "fixed_scalar_used": scalar,
        },
        "residual_structure": structure,
        "per_site_native": per_group(pairs, "site", "espa_native"),
        "per_site_grass": per_group(kept, "site", "espa_grass"),
        "per_month_native": per_group(pairs, "month", "espa_native"),
        "per_month_grass": per_group(kept, "month", "espa_grass"),
        "ratio_summary": {
            "median": float(kept["ratio"].median()),
            "p05": float(kept["ratio"].quantile(0.05)),
            "p95": float(kept["ratio"].quantile(0.95)),
            "n_low_eto_lt_1mm": int((kept["eto"] < 1.0).sum()),
        },
        "diagnostic": diagnostic,
    }


def plot(pairs: pd.DataFrame, report: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = pairs[np.isfinite(pairs["espa_grass"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, x, label, res in (
        (axes[0], "espa_native", "native ESPA ETF (alfalfa basis)", report["baseline_observed"]),
        (axes[1], "espa_grass", "ESPA x ETr/ETo (grass basis)", report["corrected"]),
    ):
        ax.scatter(kept[x], kept["nhm"], s=6, alpha=0.35)
        ax.plot([0, 1.4], [0, 1.4], "k--", lw=0.8)
        ax.plot([0, 1.4], [0, 1.4 * res["slope"]], "r-", lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel("NHM SSEBop ETf (grass)")
        ax.set_title(
            f"slope {res['slope']:.3f}  med {res['median_ratio']:.3f}  r {res['r']:.3f}  n {res['n']}"
        )
        ax.set_xlim(0, 1.4)
        ax.set_ylim(0, 1.4)
    sites = sorted(report["per_site_grass"])
    axes[2].scatter(
        [report["per_site_native"][s]["slope"] for s in sites],
        range(len(sites)),
        label="native",
        s=14,
    )
    axes[2].scatter(
        [report["per_site_grass"][s]["slope"] for s in sites],
        range(len(sites)),
        label="grass",
        s=14,
    )
    axes[2].axvline(1.0, color="k", lw=0.8, ls="--")
    axes[2].set_yticks(range(len(sites)))
    axes[2].set_yticklabels(sites, fontsize=7)
    axes[2].set_xlabel("per-site zero-intercept slope (NHM / ESPA)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sidecar-dir", default=SIDECAR_DIR)
    ap.add_argument("--out-dir", default=QA_ROOT)
    args = ap.parse_args(argv)

    espa = load_espa_native()
    nhm = load_nhm(set(espa))
    shared = sorted(set(espa) & set(nhm))
    espa = {s: espa[s] for s in shared}
    sidecar = load_sidecar(args.sidecar_dir)
    pairs = build_pairs(espa, nhm, sidecar)
    report = evaluate(pairs)
    report["inputs"] = {
        "espa_json": ESPA_JSON,
        "nhm_dir": NHM_DIR,
        "sidecar_dir": args.sidecar_dir,
        "shared_sites": shared,
        "sites_missing_sidecar": sorted(set(shared) - set(sidecar["site"])),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    pairs.to_csv(
        os.path.join(args.out_dir, f"{OUTPUT_STEM}_pairs.csv"), index=False, float_format="%.8g"
    )
    with open(os.path.join(args.out_dir, f"{OUTPUT_STEM}_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    plot(pairs, report, os.path.join(args.out_dir, f"{OUTPUT_STEM}.png"))
    show = {
        k: report[k]
        for k in (
            "role",
            "baseline_observed",
            "corrected",
            "residual_structure",
            "ratio_summary",
            "diagnostic",
        )
    }
    print(json.dumps(show, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

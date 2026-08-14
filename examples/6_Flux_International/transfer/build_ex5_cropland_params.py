"""Freeze the Example 5 (Experiment 2) cropland median parameter vector.

Reads the Example 5 publication calibration parameter ensemble (PEST++ IES
``.par.csv``) and writes two frozen artifacts used by the Ex5 -> Ex6 parameter
transferability test (``../transfer_ex5_params.py``):

    ex5_cropland_params.json           - the 8-parameter transfer vector
    ex5_cropland_params_metadata.json  - provenance for the manuscript claim

Aggregation: for each of the eight calibrated parameters, take each Ex5 site's
posterior median (the median realization value, ``base`` excluded), then take
the median across the Ex5 cropland sites. This matches the "median Experiment 2
cropland posterior" framing in the transferability plan. All 60 Example 5 sites
are CONUS cropland, so the cropland median is the median over the whole cohort.

The script does NOT look at any Example 6 flux performance; the transfer vector
is frozen before evaluation by construction.

Usage:
    uv run python examples/6_Flux_International/transfer/build_ex5_cropland_params.py
    uv run python .../build_ex5_cropland_params.py --par-csv /path/to/run.3.par.csv
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# Example 5 publication run (Run 21): clean re-run reproducing Run 20 bit-for-bit
# with the complete RUN_POLICY archive. The final-iteration parameter ensemble.
DEFAULT_PAR_CSV = "/data/ssd1/swim/5_Flux_Ensemble/results/run21/5_Flux_Ensemble.3.par.csv"

# The eight calibrated parameters, with the PEST parameter-family token used in
# the .par.csv column names (``pname:p_<family>_<site>_:0_...``).
PARAM_FAMILIES = [
    "aw",
    "ndvi_k",
    "ndvi_0",
    "mad",
    "ks_alpha",
    "kr_alpha",
    "swe_alpha",
    "swe_beta",
]

# Legacy hard-coded vector from the old 9-site Ex5 cropland median used by
# legacy/forward_ex5_cropland.py. Recorded for provenance / comparison only.
LEGACY_VECTOR = {
    "aw": 202,
    "ndvi_k": 8.88,
    "ndvi_0": 0.545,
    "mad": 0.115,
    "ks_alpha": 0.280,
    "kr_alpha": 0.334,
    "swe_alpha": 0.337,
    "swe_beta": 1.496,
}


def _family_columns(columns):
    """Map each parameter family to its per-site .par.csv columns.

    Columns are ``pname:p_<family>_<site>_:0_...``. None of the eight family
    tokens is a prefix of another (sharing only the ``ndvi_``/``swe_``/... stem
    but differing at the final token), so a ``pname:p_<family>_`` prefix match
    assigns each column to exactly one family.
    """
    by_family = {fam: [] for fam in PARAM_FAMILIES}
    for col in columns:
        for fam in PARAM_FAMILIES:
            if col.startswith(f"pname:p_{fam}_"):
                by_family[fam].append(col)
                break
    return by_family


def _site_from_column(col, fam):
    """Extract the site token from ``pname:p_<family>_<site>_:0_...``."""
    remainder = col[len(f"pname:p_{fam}_") :]
    return remainder.split("_:")[0]


def compute_cropland_medians(par_csv):
    """Return (median_vector, site_count, per_site_medians, n_realizations).

    For each family: per-site posterior median (median over realizations,
    ``base`` excluded), then the median across sites. ``per_site_medians`` is a
    DataFrame (index=site, columns=families) of the per-site posterior medians.
    """
    df = pd.read_csv(par_csv, index_col=0)
    n_base = sum(1 for i in df.index if str(i) == "base")
    if n_base != 1:
        raise ValueError(f"Expected exactly one 'base' realization, found {n_base} in {par_csv}")
    realizations = df.loc[[i for i in df.index if str(i) != "base"]]
    by_family = _family_columns(df.columns)

    vector = {}
    per_site = {}
    site_sets = {}
    for fam in PARAM_FAMILIES:
        cols = by_family[fam]
        if not cols:
            raise ValueError(f"No columns found for parameter family {fam!r} in {par_csv}")
        sites = [_site_from_column(c, fam) for c in cols]
        if len(set(sites)) != len(sites):
            raise ValueError(f"Duplicate site columns for family {fam!r} in {par_csv}")
        per_site_median = realizations[cols].median()  # median realization per site
        per_site_median.index = sites
        if not per_site_median.map(lambda v: pd.notna(v) and abs(v) != float("inf")).all():
            raise ValueError(f"Non-finite per-site median for family {fam!r} in {par_csv}")
        vector[fam] = float(per_site_median.median())  # median across sites
        per_site[fam] = per_site_median
        site_sets[fam] = frozenset(sites)

    if len(set(site_sets.values())) != 1:
        raise ValueError(f"Site sets differ across parameter families in {par_csv}")
    medians_df = pd.DataFrame(per_site).sort_index()
    medians_df.index.name = "site"
    return vector, len(medians_df), medians_df, len(realizations)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE), text=True
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--par-csv", default=DEFAULT_PAR_CSV, help="Ex5 publication .par.csv")
    parser.add_argument("--out-dir", default=str(HERE), help="Where to write the JSON artifacts")
    parser.add_argument("--out-json", default=None, help="Explicit vector JSON path")
    parser.add_argument("--out-meta", default=None, help="Explicit metadata JSON path")
    parser.add_argument("--out-medians", default=None, help="Explicit per-site medians CSV path")
    parser.add_argument(
        "--source-run",
        default=None,
        help="Source-run label for the metadata (default: derived from the --par-csv path)",
    )
    args = parser.parse_args()

    par_csv = Path(args.par_csv)
    if not par_csv.exists():
        raise FileNotFoundError(f"Ex5 parameter ensemble not found: {par_csv}")

    vector, n_sites, medians_df, n_realizations = compute_cropland_medians(par_csv)
    source_run = args.source_run or f"{par_csv.parent.name} (derived from --par-csv path)"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = Path(args.out_json) if args.out_json else out_dir / "ex5_cropland_params.json"
    meta_path = (
        Path(args.out_meta) if args.out_meta else out_dir / "ex5_cropland_params_metadata.json"
    )
    medians_path = (
        Path(args.out_medians)
        if args.out_medians
        else params_path.with_name(params_path.stem + "_site_medians.csv")
    )
    medians_df.to_csv(medians_path)

    # Frozen transfer vector.
    with open(params_path, "w") as f:
        json.dump(vector, f, indent=2)
        f.write("\n")

    # Per-parameter comparison with the legacy hard-coded vector.
    comparison = {}
    matches_legacy = True
    for fam in PARAM_FAMILIES:
        cur = vector[fam]
        old = float(LEGACY_VECTOR[fam])
        same = abs(cur - old) <= 1e-6 * max(1.0, abs(old))
        matches_legacy &= same
        comparison[fam] = {
            "current": cur,
            "legacy": old,
            "abs_delta": cur - old,
            "matches": same,
        }

    metadata = {
        "source_experiment": "Example 5 / Experiment 2 (CONUS cropland)",
        "source_run": source_run,
        "calibration_target": "simple six-model OpenET ensemble mean (per-overpass nanmean)",
        "observation_weighting": "spread-based observation weights (per-overpass member std)",
        "aggregation": (
            "median across Ex5 cropland sites of each site's posterior median "
            "(median realization value; 'base' realization excluded)"
        ),
        "source_par_csv": str(par_csv),
        "source_par_csv_sha256": _sha256(par_csv),
        "n_ex5_sites": n_sites,
        "n_posterior_realizations": n_realizations,
        "excluded_realization": "base",
        "site_list": list(medians_df.index),
        "per_site_medians_csv": str(medians_path),
        "model_structure": "legacy single-mad coupling",
        "stress_depletion_fraction": None,
        "date_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "param_names": PARAM_FAMILIES,
        "param_values": vector,
        "legacy_vector": LEGACY_VECTOR,
        "matches_legacy_vector": matches_legacy,
        "legacy_comparison": comparison,
        "notes": (
            "Transfer vector frozen before any Example 6 flux or Example 7 meter "
            "evaluation. Not derived or tuned from Ex6 flux ET or Ex7 metered "
            "applied water. The legacy vector was the old 9-site Ex5 cropland "
            f"median; the current vector is the {n_sites}-site cropland median "
            f"from {source_run} and supersedes it."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Wrote {params_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {medians_path}")
    print(f"\nEx5 cropland median vector ({n_sites} sites):")
    print(f"  {'param':<11} {'current':>10} {'legacy':>10} {'delta':>10}")
    for fam in PARAM_FAMILIES:
        c = comparison[fam]
        print(f"  {fam:<11} {c['current']:>10.4f} {c['legacy']:>10.4f} {c['abs_delta']:>+10.4f}")
    print(f"\nMatches legacy hard-coded vector: {matches_legacy}")


if __name__ == "__main__":
    main()

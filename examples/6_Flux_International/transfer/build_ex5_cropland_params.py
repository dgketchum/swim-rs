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


def compute_cropland_medians(par_csv):
    """Return (median_vector, site_count) from the Ex5 .par.csv.

    For each family: per-site posterior median (median over realizations,
    ``base`` excluded), then the median across sites.
    """
    df = pd.read_csv(par_csv, index_col=0)
    realizations = df.loc[[i for i in df.index if str(i) != "base"]]
    by_family = _family_columns(df.columns)

    vector = {}
    site_counts = {}
    for fam in PARAM_FAMILIES:
        cols = by_family[fam]
        if not cols:
            raise ValueError(f"No columns found for parameter family {fam!r} in {par_csv}")
        per_site_median = realizations[cols].median()  # median realization per site
        vector[fam] = float(per_site_median.median())  # median across sites
        site_counts[fam] = len(cols)

    counts = set(site_counts.values())
    if len(counts) != 1:
        raise ValueError(f"Inconsistent per-family site counts: {site_counts}")
    return vector, counts.pop()


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
    args = parser.parse_args()

    par_csv = Path(args.par_csv)
    if not par_csv.exists():
        raise FileNotFoundError(f"Ex5 parameter ensemble not found: {par_csv}")

    vector, n_sites = compute_cropland_medians(par_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "ex5_cropland_params.json"
    meta_path = out_dir / "ex5_cropland_params_metadata.json"

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
        "source_run": "Run 21 (publication; reproduces Run 20 bit-for-bit)",
        "calibration_target": "simple six-model OpenET ensemble mean (per-overpass nanmean)",
        "observation_weighting": "spread-based observation weights (per-overpass member std)",
        "aggregation": (
            "median across Ex5 cropland sites of each site's posterior median "
            "(median realization value; 'base' realization excluded)"
        ),
        "source_par_csv": str(par_csv),
        "n_ex5_sites": n_sites,
        "date_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "param_names": PARAM_FAMILIES,
        "param_values": vector,
        "legacy_vector": LEGACY_VECTOR,
        "matches_legacy_vector": matches_legacy,
        "legacy_comparison": comparison,
        "notes": (
            "Transfer vector frozen before any Example 6 flux evaluation. Not "
            "derived or tuned from Ex6 flux ET. The legacy vector was the old "
            "9-site Ex5 cropland median; the current vector is the 60-site "
            "publication (Run 21) cropland median and supersedes it."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Wrote {params_path}")
    print(f"Wrote {meta_path}")
    print(f"\nEx5 cropland median vector ({n_sites} sites):")
    print(f"  {'param':<11} {'current':>10} {'legacy':>10} {'delta':>10}")
    for fam in PARAM_FAMILIES:
        c = comparison[fam]
        print(f"  {fam:<11} {c['current']:>10.4f} {c['legacy']:>10.4f} {c['abs_delta']:>+10.4f}")
    print(f"\nMatches legacy hard-coded vector: {matches_legacy}")


if __name__ == "__main__":
    main()

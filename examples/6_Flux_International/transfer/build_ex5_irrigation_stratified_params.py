"""Freeze irrigation-stratified Example 5 (Experiment 2) transfer vectors.

The pooled E2 transfer vector has ``mad = 0.136917``, which is inside the
irrigated prior (0.10-0.30) but outside the configured rainfed prior
(0.30-0.80). Applying it to rainfed targets therefore transfers a scheduler
trigger from outside the rainfed parameter domain. This builder replaces the
single pooled vector with two independently frozen vectors, selected downstream
by an independently inferred irrigation class.

Aggregation is identical to the pooled builder (median-of-site-medians: per-site
posterior median with the ``base`` realization excluded, then the median across
source sites), the only change being that the across-site median is taken within
an irrigation class rather than over the whole cohort.

Source classification uses ``properties/irrigation/irr > 0.5`` from the Run 22
container -- the same rule ``examples/5_Flux_Ensemble/archive_run.py`` already
uses to summarize the Run 22 posterior by irrigation class. It is derived from
remote sensing (IrrMapper/LANID) only. No flux ET and no meter record enters
classification or vector construction.

This is the canonical single-``mad`` Run 22 physics. It is NOT the later
``mad``/stress-threshold split held under the Example 5 ``e5split`` materials,
and ``stress_depletion_fraction`` stays unset.

Writes:
    e2_run22_transfer_vectors_by_irrigation.json           - the two vectors
    e2_run22_transfer_vectors_by_irrigation_metadata.json  - full provenance
    e2_run22_transfer_site_medians_by_irrigation.csv       - per-site medians + class

Usage:
    uv run python examples/6_Flux_International/transfer/build_ex5_irrigation_stratified_params.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_ex5_cropland_params import (  # noqa: E402  (sibling module)
    PARAM_FAMILIES,
    _git_sha,
    _sha256,
    compute_cropland_medians,
)

REPO_ROOT = HERE.parents[2]

DEFAULT_PAR_CSV = "/data/ssd1/swim/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv"
DEFAULT_CONTAINER = "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "data" / "final"

CLASSES = ("irrigated", "rainfed")
IRR_THRESHOLD = 0.5
CLASS_RULE = "properties/irrigation/irr > 0.5 (Run 22 source container)"

# Frozen audit values from the existing Run 22 posterior summary. A rebuild must
# reproduce these before any downstream flux or meter truth is opened.
EXPECTED_VECTORS = {
    "irrigated": {
        "aw": 316.705,
        "ndvi_k": 3.477500,
        "ndvi_0": 0.519265,
        "mad": 0.121930,
        "ks_alpha": 0.354274,
        "kr_alpha": 0.501016,
        "swe_alpha": 0.325388,
        "swe_beta": 1.468350,
    },
    "rainfed": {
        "aw": 312.539,
        "ndvi_k": 4.009340,
        "ndvi_0": 0.433295,
        "mad": 0.389557,
        "ks_alpha": 0.388515,
        "kr_alpha": 0.234779,
        "swe_alpha": 0.416753,
        "swe_beta": 1.247040,
    },
}
EXPECTED_COUNTS = {"irrigated": 39, "rainfed": 21}

# Configured parameter priors, used only to report whether each frozen class
# vector sits inside its own class domain. This is the scientific defect the
# stratification exists to fix; it is a reported diagnostic, not a filter.
CLASS_PRIORS = {
    "irrigated": {"mad": (0.10, 0.30)},
    "rainfed": {"mad": (0.30, 0.80)},
}


def read_source_irrigation(container_path, threshold=IRR_THRESHOLD):
    """Return a DataFrame (index=container fid) with ``irr`` and ``irr_class``.

    Reads ``properties/irrigation/irr`` from the source container. The array is
    stored in container field order, so it is labelled with ``geometry/uid`` --
    the same array ``SwimContainer.field_uids`` is loaded from.
    """
    import zarr

    root = zarr.open(str(container_path), mode="r")
    if "properties/irrigation/irr" not in root:
        raise KeyError(
            f"{container_path} has no properties/irrigation/irr; the source "
            "irrigation class rule cannot be applied to this container"
        )
    fids = [str(f) for f in root["geometry/uid"][:]]
    arr = np.asarray(root["properties/irrigation/irr"][:], dtype=float).ravel()

    if len(arr) != len(fids):
        raise ValueError(
            f"irr array length {len(arr)} != {len(fids)} field_uids in {container_path}"
        )
    if not np.isfinite(arr).all():
        bad = [fids[i] for i in np.flatnonzero(~np.isfinite(arr))]
        raise ValueError(f"Non-finite irrigation fraction for source sites: {sorted(bad)}")

    return pd.DataFrame(
        {"irr": arr, "irr_class": np.where(arr > threshold, "irrigated", "rainfed")},
        index=pd.Index(fids, name="fid"),
    )


def align_classes_to_par_sites(medians_df, irr_df):
    """Join container irrigation classes onto the ``.par.csv`` site index.

    PEST++ lowercases the site token in ``.par.csv`` column names while the
    container carries canonical mixed-case fids, so the join is on lowercase.
    Returns a DataFrame indexed by the par-csv site token with ``fid``, ``irr``,
    and ``irr_class``.
    """
    lowered = {}
    for fid, row in irr_df.iterrows():
        key = str(fid).lower()
        if key in lowered:
            raise ValueError(
                f"Container fids collide when lowercased ({lowered[key][0]!r} vs {fid!r}); "
                "cannot join irrigation classes onto the PEST-lowercased par.csv sites"
            )
        lowered[key] = (str(fid), float(row["irr"]), str(row["irr_class"]))

    missing = [s for s in medians_df.index if s not in lowered]
    if missing:
        raise ValueError(f"No source irrigation label for par.csv sites: {sorted(missing)}")

    return pd.DataFrame(
        [
            {
                "site": s,
                "fid": lowered[s][0],
                "irr": lowered[s][1],
                "irr_class": lowered[s][2],
            }
            for s in medians_df.index
        ]
    ).set_index("site")


def stratified_vectors(medians_df, class_by_site, expected_counts=None):
    """Median-of-site-medians within each irrigation class.

    ``medians_df`` is the per-site posterior-median table; ``class_by_site`` maps
    each of its sites to ``'irrigated'`` or ``'rainfed'``. Returns
    ``{class: {'n_sites': int, 'sites': [...], 'vector': {family: value}}}``.
    """
    unknown = sorted({c for c in class_by_site.values() if c not in CLASSES})
    if unknown:
        raise ValueError(f"Unrecognized irrigation class label(s): {unknown}")
    missing = [s for s in medians_df.index if s not in class_by_site]
    if missing:
        raise ValueError(f"No source irrigation label for par.csv sites: {sorted(missing)}")

    out = {}
    for cls in CLASSES:
        sites = sorted(s for s in medians_df.index if class_by_site[s] == cls)
        if not sites:
            raise ValueError(f"No source sites in irrigation class {cls!r}")
        sub = medians_df.loc[sites, list(PARAM_FAMILIES)]
        if not np.isfinite(sub.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite per-site median in class {cls!r}")
        out[cls] = {
            "n_sites": len(sites),
            "sites": sites,
            "vector": {fam: float(sub[fam].median()) for fam in PARAM_FAMILIES},
        }

    if expected_counts:
        got = {cls: out[cls]["n_sites"] for cls in CLASSES}
        if got != dict(expected_counts):
            raise ValueError(
                f"Source class counts {got} do not match the frozen expectation "
                f"{dict(expected_counts)}; stop and investigate the source cohort "
                "rather than substituting inputs"
            )
    return out


def check_expected(vectors, expected=EXPECTED_VECTORS, tol=1e-3):
    """Compare frozen class vectors against the audit table. Returns (ok, rows)."""
    rows = []
    ok = True
    for cls in CLASSES:
        for fam in PARAM_FAMILIES:
            got = vectors[cls]["vector"][fam]
            want = float(expected[cls][fam])
            match = abs(got - want) <= tol
            ok &= match
            rows.append(
                {
                    "irr_class": cls,
                    "param": fam,
                    "computed": got,
                    "expected": want,
                    "abs_delta": got - want,
                    "matches": bool(match),
                }
            )
    return ok, rows


def prior_domain_report(vectors, priors=CLASS_PRIORS):
    """Report whether each class vector sits inside its own configured prior."""
    report = {}
    for cls in CLASSES:
        entries = {}
        for fam, (lo, hi) in priors.get(cls, {}).items():
            v = vectors[cls]["vector"][fam]
            entries[fam] = {
                "value": v,
                "prior_lo": lo,
                "prior_hi": hi,
                "in_domain": bool(lo <= v <= hi),
            }
        report[cls] = entries
    return report


def _vector_sha256(vector):
    """Stable hash of a class vector (sorted keys, repr-stable float encoding)."""
    payload = json.dumps({k: vector[k] for k in sorted(vector)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _worktree_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(HERE), text=True
        ).strip()
        return bool(out)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--par-csv", default=DEFAULT_PAR_CSV, help="Run 22 posterior .par.csv")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Run 22 source container")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Artifact output dir")
    parser.add_argument(
        "--expect-par-sha256",
        default=None,
        help="Require the posterior to hash to this value before writing anything",
    )
    parser.add_argument(
        "--allow-unexpected",
        action="store_true",
        help="Write artifacts even if class counts or audit values do not reproduce",
    )
    args = parser.parse_args()

    par_csv = Path(args.par_csv)
    container_path = Path(args.container)
    if not par_csv.exists():
        raise FileNotFoundError(f"Run 22 posterior not found: {par_csv}")
    if not container_path.exists():
        raise FileNotFoundError(f"Run 22 source container not found: {container_path}")

    par_sha = _sha256(par_csv)
    if args.expect_par_sha256 and par_sha != args.expect_par_sha256:
        raise ValueError(
            f"Posterior hash mismatch: got {par_sha}, expected {args.expect_par_sha256}"
        )

    # Median-of-site-medians table (validates one 'base', all eight families,
    # no duplicate/non-finite sites, consistent site sets across families).
    pooled_vector, n_sites, medians_df, n_realizations = compute_cropland_medians(par_csv)

    irr_df = read_source_irrigation(container_path)
    joined = align_classes_to_par_sites(medians_df, irr_df)
    class_by_site = joined["irr_class"].to_dict()

    expected_counts = None if args.allow_unexpected else EXPECTED_COUNTS
    vectors = stratified_vectors(medians_df, class_by_site, expected_counts=expected_counts)
    audit_ok, audit_rows = check_expected(vectors)
    if not audit_ok and not args.allow_unexpected:
        frame = pd.DataFrame(audit_rows)
        raise ValueError(
            "Frozen class vectors do not reproduce the audit table; stop and "
            "investigate rather than substituting inputs.\n"
            + frame[~frame["matches"]].to_string(index=False)
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "e2_run22_transfer_vectors_by_irrigation.json"
    meta_path = out_dir / "e2_run22_transfer_vectors_by_irrigation_metadata.json"
    medians_path = out_dir / "e2_run22_transfer_site_medians_by_irrigation.csv"

    # --- per-site medians + class (the aggregation audit trail) ---------------
    medians_out = medians_df.copy()
    medians_out.insert(0, "fid", joined["fid"])
    medians_out.insert(1, "irr", joined["irr"])
    medians_out.insert(2, "irr_class", joined["irr_class"])
    medians_out.to_csv(medians_path)

    # --- the two frozen vectors ----------------------------------------------
    payload = {cls: vectors[cls]["vector"] for cls in CLASSES}
    with open(vectors_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    domain = prior_domain_report(vectors)
    pooled_domain = {
        cls: {
            fam: {
                "pooled_value": pooled_vector[fam],
                "prior_lo": lo,
                "prior_hi": hi,
                "in_domain": bool(lo <= pooled_vector[fam] <= hi),
            }
            for fam, (lo, hi) in CLASS_PRIORS.get(cls, {}).items()
        }
        for cls in CLASSES
    }

    metadata = {
        "experiment": "irrigation-stratified Example 5 / Experiment 2 parameter transfer",
        "objective": (
            "Replace the single pooled E2 cropland transfer vector with independently "
            "frozen irrigated and rainfed vectors, selected downstream by an "
            "independently inferred irrigation class."
        ),
        "source_experiment": "Example 5 / Experiment 2 (CONUS cropland)",
        "source_run": "Run 22 (2026-07-02 publication recal; source-exclusive physics + gw gate)",
        "calibration_target": "simple six-model OpenET ensemble mean (per-overpass nanmean)",
        "observation_weighting": "spread-based observation weights (per-overpass member std)",
        "aggregation": (
            "median within irrigation class of each source site's posterior median "
            "(median realization value; 'base' realization excluded)"
        ),
        "source_par_csv": str(par_csv),
        "source_par_csv_sha256": par_sha,
        "source_container": str(container_path),
        "source_class_rule": CLASS_RULE,
        "irr_threshold": IRR_THRESHOLD,
        "n_source_sites": n_sites,
        "n_posterior_realizations": n_realizations,
        "excluded_realization": "base",
        "param_names": list(PARAM_FAMILIES),
        "classes": {
            cls: {
                "n_sites": vectors[cls]["n_sites"],
                "sites": vectors[cls]["sites"],
                "fids": [joined.loc[s, "fid"] for s in vectors[cls]["sites"]],
                "irr_fraction": {s: float(joined.loc[s, "irr"]) for s in vectors[cls]["sites"]},
                "vector": vectors[cls]["vector"],
                "vector_sha256": _vector_sha256(vectors[cls]["vector"]),
            }
            for cls in CLASSES
        },
        "expected_class_counts": EXPECTED_COUNTS,
        "expected_vectors": EXPECTED_VECTORS,
        "reproduces_expected_vectors": bool(audit_ok),
        "audit_comparison": audit_rows,
        "prior_domain_check": domain,
        "pooled_vector_domain_check": pooled_domain,
        "pooled_comparator_vector": pooled_vector,
        "pooled_comparator_artifact": "paper/data/final/e2_run22_transfer_vector.json",
        "per_site_medians_csv": str(medians_path),
        "model_structure": "canonical Run 22 single-mad coupling",
        "stress_depletion_fraction": None,
        "e5split_materials_used": False,
        "flux_role": "validation only; never used in classification or vector construction",
        "meter_role": "validation only; never used in classification or vector construction",
        "date_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "worktree_dirty": _worktree_dirty(),
        "notes": (
            "Both class vectors frozen before any Example 6 flux or Example 7 meter "
            "truth was opened. The pooled vector and its metadata are preserved "
            "unchanged as the comparator. Source classification comes from "
            "remote-sensing-derived irrigation fractions in the Run 22 container."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    # --- console audit --------------------------------------------------------
    print(f"Source posterior : {par_csv}")
    print(f"  sha256         : {par_sha}")
    print(f"  sites          : {n_sites}   realizations: {n_realizations} ('base' excluded)")
    print(f"Source container : {container_path}")
    print(f"  class rule     : {CLASS_RULE}")
    for cls in CLASSES:
        print(f"  {cls:<10}     : n={vectors[cls]['n_sites']}")
    print("\nFrozen irrigation-stratified vectors (median-of-site-medians within class):")
    print(f"  {'param':<11}{'irrigated':>12}{'rainfed':>12}{'pooled':>12}{'irr-rf':>12}")
    for fam in PARAM_FAMILIES:
        i = vectors["irrigated"]["vector"][fam]
        r = vectors["rainfed"]["vector"][fam]
        print(f"  {fam:<11}{i:>12.6f}{r:>12.6f}{pooled_vector[fam]:>12.6f}{i - r:>+12.6f}")
    print(f"\nReproduces frozen audit table: {audit_ok}")
    print("\nParameter-domain check (the defect stratification fixes):")
    for cls in CLASSES:
        for fam, e in domain[cls].items():
            p = pooled_domain[cls][fam]
            print(
                f"  {cls:<10} {fam:<5} class vector {e['value']:.6f} "
                f"in [{e['prior_lo']}, {e['prior_hi']}] -> {e['in_domain']}   "
                f"| pooled {p['pooled_value']:.6f} -> {p['in_domain']}"
            )
    for cls in CLASSES:
        print(f"\n{cls} vector sha256: {_vector_sha256(vectors[cls]['vector'])}")
    print(f"\nWrote {vectors_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {medians_path}")


if __name__ == "__main__":
    main()

"""Build the Example 6 (Experiment 3) per-site irrigation-stratified param mapping.

Expands the two frozen Run 22 class vectors into a per-site
``{sid: {param: value}}`` mapping covering the full 66-site E3 publication
cohort, so ``../transfer_ex5_params.py --params-by-site`` can score the
irrigation-stratified transfer against the pooled transfer on identical support.

Target-class policy (handoff, "Target-class policy > E3"): the first-stage
site-level "equipped for irrigation" result of the canonical two-stage local
satellite ET/precipitation classifier selects the vector -- equipped site gets
the irrigated vector, not equipped gets the rainfed vector. The second-stage
annual classification continues to activate or deactivate irrigation by year; it
does NOT change a site's parameter vector. Parameters are fixed by site and never
switch annually.

Stage 1 is not persisted. In ``src/swimrs/container/components/calculator.py`` the
``equipped`` flag is a plain local variable (~L1203-1229) consumed by the stage-2
annual test (~L1268-1316) and then discarded: it is written to neither the
container nor any CSV/JSON. The only persisted record is
``derived/dynamics/irr_data``, whose per-year dicts carry
``('f_irr', 'irr_doys', 'irrigated')``. Stage 1 is therefore recovered here as
"the site has at least one irrigated year in ``irr_data``". That recovery has been
independently verified to reproduce the stage-1 rule exactly on this container:
equipped is True for exactly 14 sites, with zero equipped-but-never-activated
sites and zero activated-but-not-equipped sites. Note ``properties/irrigation/irr``
does not exist in any E3 container -- that is the CONUS ``use_mask`` path.

No flux ET and no meter record enters classification: irrigation status comes from
remote-sensing ET, precipitation, and land cover only.

Reconciliation, asserted here (handoff stop condition: "the final E3 cohort does
not reconcile independently of the 75-site container"):

    container scope (75 sites)  : 14 ever-irrigated sites / 175 irrigated site-years
    cohort scope    (66 sites)  : 13 ever-irrigated sites / 163 irrigated site-years
    container minus cohort      : exactly {ES-LJu}

Writes (under ``--out-dir``, default ``paper/data/final``):
    e3_irrigation_stratified_param_mapping.json           - {sid: {8 params}}, all 66 sites
    e3_irrigation_stratified_param_mapping_metadata.json  - assignments + provenance

Usage:
    uv run python \\
        /home/dgketchum/code/swim-rs/examples/6_Flux_International/transfer/build_e3_irrigation_mapping.py

    uv run python \\
        /home/dgketchum/code/swim-rs/examples/6_Flux_International/transfer/build_e3_irrigation_mapping.py \\
        --vectors /home/dgketchum/code/swim-rs/paper/data/final/e2_run22_transfer_vectors_by_irrigation.json \\
        --container /data/ssd1/swim/6_Flux_International/data/6_Flux_International_ls_ensemble_por_annual2yr.swim \\
        --shapefile /data/ssd1/swim/6_Flux_International/data/gis/flux_crop_pub_66_150m.shp

This script only reads existing artifacts. It does NOT run the model, rebuild a
container, calibrate, or call Earth Engine.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_ex5_cropland_params import (  # noqa: E402  (sibling module)
    PARAM_FAMILIES,
    _git_sha,
    _sha256,
)
from build_ex5_irrigation_stratified_params import (  # noqa: E402  (sibling module)
    CLASSES,
    _vector_sha256,
    _worktree_dirty,
)

REPO_ROOT = HERE.parents[2]

DEFAULT_VECTORS = (
    REPO_ROOT / "paper" / "data" / "final" / ("e2_run22_transfer_vectors_by_irrigation.json")
)
DEFAULT_CONTAINER = (
    "/data/ssd1/swim/6_Flux_International/data/6_Flux_International_ls_ensemble_por_annual2yr.swim"
)
DEFAULT_SHAPEFILE = "/data/ssd1/swim/6_Flux_International/data/gis/flux_crop_pub_66_150m.shp"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "data" / "final"
UID_COL = "sid"

# Frozen expectations. These are the handoff's cohort-reconciliation numbers; a
# mismatch is a stop condition, not something to work around.
EXPECTED_CONTAINER_SITES = 75
EXPECTED_COHORT_SITES = 66
EXPECTED_CONTAINER_EQUIPPED = 14
EXPECTED_CONTAINER_IRR_YEARS = 175
EXPECTED_COHORT_EQUIPPED = 13
EXPECTED_COHORT_IRR_YEARS = 163
EXPECTED_CONTAINER_ONLY = {"ES-LJu"}

STAGE1_RULE = (
    "site-level 'equipped for irrigation' from the canonical two-stage local "
    "satellite ET/precipitation classifier: with lulc_irr_method='annual_2yr', a "
    "site is equipped when more than one third of its rolling two-year "
    "ET/(PPT+1) balance windows exceed annual_subsidy_ratio "
    "(calculator.py stage 1, ~L1203-1229)"
)
STAGE1_RECOVERY = (
    "Stage 1 is NOT persisted: 'equipped' is a local variable in "
    "IrrigationCalculator (calculator.py ~L1203-1229), stored in neither the "
    "container nor any CSV/JSON. It is recovered here as 'the site has >= 1 "
    "irrigated year in derived/dynamics/irr_data'. That recovery was verified "
    "independently to reproduce the stage-1 rule exactly on this container: "
    "equipped is True for exactly 14 sites, with zero equipped-but-never-activated "
    "sites and zero activated-but-not-equipped sites. "
    "properties/irrigation/irr does not exist in any E3 container (CONUS use_mask "
    "path only) and is not consulted."
)
STAGE2_NOTE = (
    "The second-stage annual classification continues to activate or deactivate "
    "the irrigation scheduler year by year exactly as in the canonical model; it "
    "does NOT change a site's parameter vector. Parameters are fixed by site."
)
CLASS_TO_VECTOR = {True: "irrigated", False: "rainfed"}


def load_class_vectors(vectors_path):
    """Load and validate the frozen two-vector artifact."""
    vectors_path = Path(vectors_path)
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Frozen irrigation-stratified vectors not found: {vectors_path}. Build them "
            "first with examples/6_Flux_International/transfer/"
            "build_ex5_irrigation_stratified_params.py; this script will not invent a "
            "placeholder."
        )
    with open(vectors_path) as f:
        doc = json.load(f)

    missing_classes = [c for c in CLASSES if c not in doc]
    if missing_classes:
        raise ValueError(
            f"{vectors_path} is missing irrigation class(es) {missing_classes}; expected a "
            f"nested {{class: {{param: value}}}} mapping with {list(CLASSES)}"
        )
    extra = sorted(set(doc) - set(CLASSES))
    if extra:
        raise ValueError(f"{vectors_path} has unexpected top-level key(s) {extra}")

    vectors = {}
    for cls in CLASSES:
        vec = doc[cls]
        if not isinstance(vec, dict):
            raise ValueError(f"{vectors_path}: class {cls!r} is not a {{param: value}} object")
        missing = [p for p in PARAM_FAMILIES if p not in vec]
        if missing:
            raise ValueError(f"{vectors_path}: class {cls!r} is missing parameter(s) {missing}")
        unknown = sorted(set(vec) - set(PARAM_FAMILIES))
        if unknown:
            raise ValueError(f"{vectors_path}: class {cls!r} has unknown parameter(s) {unknown}")
        vectors[cls] = {p: float(vec[p]) for p in PARAM_FAMILIES}

    if vectors["irrigated"] == vectors["rainfed"]:
        raise ValueError(
            f"{vectors_path}: the irrigated and rainfed vectors are identical; the "
            "stratification would be a no-op"
        )
    return vectors


def read_irrigation_years(container_path):
    """Return ``{fid: n_irrigated_years}`` from ``derived/dynamics/irr_data``.

    Read-only. ``irr_data`` is one JSON blob per field in container field order,
    keyed by year string plus a ``fallow_years`` list; a year counts as irrigated
    when its ``irrigated`` flag is set.
    """
    import zarr

    container_path = Path(container_path)
    if not container_path.exists():
        raise FileNotFoundError(f"E3 container not found: {container_path}")

    root = zarr.open(str(container_path), mode="r")
    if "derived/dynamics/irr_data" not in root:
        raise KeyError(
            f"{container_path} has no derived/dynamics/irr_data; the stage-1 irrigation "
            "class cannot be recovered from this container"
        )
    fids = [str(u) for u in root["geometry/uid"][:]]
    raw = root["derived/dynamics/irr_data"][:]
    if len(raw) != len(fids):
        raise ValueError(
            f"irr_data length {len(raw)} != {len(fids)} field uids in {container_path}"
        )

    years = {}
    for fid, blob in zip(fids, raw):
        try:
            per_year = json.loads(blob) if isinstance(blob, str) else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unparseable irr_data for {fid} in {container_path}: {exc}") from exc
        n = 0
        for key, value in per_year.items():
            if key == "fallow_years" or not isinstance(value, dict):
                continue
            if int(value.get("irrigated", 0)):
                n += 1
        years[fid] = n
    return years


def read_cohort(shapefile, uid_col=UID_COL):
    """Return the ordered cohort site ids from the publication shapefile."""
    import geopandas as gpd

    shapefile = Path(shapefile)
    if not shapefile.exists():
        raise FileNotFoundError(f"E3 cohort shapefile not found: {shapefile}")
    gdf = gpd.read_file(shapefile, engine="fiona")
    if uid_col not in gdf.columns:
        raise KeyError(f"{shapefile} has no {uid_col!r} column; found {list(gdf.columns)}")
    cohort = [str(s) for s in gdf[uid_col].tolist()]
    dupes = sorted({s for s in cohort if cohort.count(s) > 1})
    if dupes:
        raise ValueError(f"Duplicate {uid_col} value(s) in {shapefile}: {dupes}")
    return cohort


def reconcile(irr_years, cohort, allow_unexpected=False):
    """Assert the container-vs-cohort irrigation reconciliation. Returns a report."""
    container_equipped = sorted(f for f, n in irr_years.items() if n > 0)
    cohort_set = set(cohort)
    cohort_equipped = sorted(f for f in container_equipped if f in cohort_set)
    container_only = sorted(set(container_equipped) - cohort_set)

    report = {
        "container_sites": len(irr_years),
        "container_equipped_sites": len(container_equipped),
        "container_irrigated_site_years": int(sum(irr_years.values())),
        "cohort_sites": len(cohort),
        "cohort_equipped_sites": len(cohort_equipped),
        "cohort_irrigated_site_years": int(sum(irr_years[f] for f in cohort_equipped)),
        "container_equipped_not_in_cohort": container_only,
        "container_equipped_not_in_cohort_years": {f: irr_years[f] for f in container_only},
        "expected": {
            "container_sites": EXPECTED_CONTAINER_SITES,
            "container_equipped_sites": EXPECTED_CONTAINER_EQUIPPED,
            "container_irrigated_site_years": EXPECTED_CONTAINER_IRR_YEARS,
            "cohort_sites": EXPECTED_COHORT_SITES,
            "cohort_equipped_sites": EXPECTED_COHORT_EQUIPPED,
            "cohort_irrigated_site_years": EXPECTED_COHORT_IRR_YEARS,
            "container_equipped_not_in_cohort": sorted(EXPECTED_CONTAINER_ONLY),
        },
    }

    stop = (
        "Stop condition (handoff, 'Stop conditions'): the final E3 cohort does not "
        "reconcile independently of the 75-site container. Investigate the cohort, "
        "the container, or the classifier rather than substituting inputs. Pass "
        "--allow-unexpected only to inspect a knowingly different cohort."
    )
    checks = [
        ("container site count", report["container_sites"], EXPECTED_CONTAINER_SITES),
        ("cohort site count", report["cohort_sites"], EXPECTED_COHORT_SITES),
        (
            "container ever-irrigated sites",
            report["container_equipped_sites"],
            EXPECTED_CONTAINER_EQUIPPED,
        ),
        (
            "container irrigated site-years",
            report["container_irrigated_site_years"],
            EXPECTED_CONTAINER_IRR_YEARS,
        ),
        ("cohort ever-irrigated sites", report["cohort_equipped_sites"], EXPECTED_COHORT_EQUIPPED),
        (
            "cohort irrigated site-years",
            report["cohort_irrigated_site_years"],
            EXPECTED_COHORT_IRR_YEARS,
        ),
    ]
    failures = [f"{name}: got {got}, expected {want}" for name, got, want in checks if got != want]
    if set(container_only) != EXPECTED_CONTAINER_ONLY:
        failures.append(
            f"container-minus-cohort equipped sites: got {container_only}, expected "
            f"{sorted(EXPECTED_CONTAINER_ONLY)}"
        )

    missing_from_container = sorted(f for f in cohort if f not in irr_years)
    if missing_from_container:
        raise ValueError(
            f"{len(missing_from_container)} cohort site(s) absent from the container's "
            f"irr_data: {missing_from_container}. Every cohort site must have a persisted "
            "irrigation record before a class can be assigned. " + stop
        )

    report["reconciles"] = not failures
    report["failures"] = failures
    if failures and not allow_unexpected:
        raise ValueError(
            "E3 cohort irrigation reconciliation failed:\n  - "
            + "\n  - ".join(failures)
            + "\n"
            + stop
        )
    return report


def build_mapping(vectors, irr_years, cohort):
    """Return ``({sid: vector}, {sid: assignment_record}, {class: n})``."""
    mapping, assignments, counts = {}, {}, dict.fromkeys(CLASSES, 0)
    for sid in cohort:
        n_irr = int(irr_years[sid])
        equipped = n_irr > 0
        cls = CLASS_TO_VECTOR[equipped]
        mapping[sid] = dict(vectors[cls])
        counts[cls] += 1
        assignments[sid] = {
            "irr_class": cls,
            "equipped": equipped,
            "n_irrigated_years": n_irr,
            "vector_sha256": _vector_sha256(vectors[cls]),
        }
    if counts["irrigated"] == 0 or counts["rainfed"] == 0:
        raise ValueError(
            f"Degenerate class assignment {counts}: every cohort site landed in one class, "
            "so the stratified mapping is identical to a pooled vector"
        )
    return mapping, assignments, counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vectors", default=str(DEFAULT_VECTORS), help="Frozen two-vector artifact JSON"
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Canonical E3 container")
    parser.add_argument("--shapefile", default=DEFAULT_SHAPEFILE, help="66-site cohort shapefile")
    parser.add_argument("--uid-col", default=UID_COL, help="Cohort shapefile uid column")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Artifact output dir")
    parser.add_argument(
        "--allow-unexpected",
        action="store_true",
        help="Write the mapping even if the container/cohort irrigation reconciliation "
        "does not match the frozen 14/175 and 13/163 expectations",
    )
    args = parser.parse_args()

    vectors_path = Path(args.vectors)
    vectors = load_class_vectors(vectors_path)
    irr_years = read_irrigation_years(args.container)
    cohort = read_cohort(args.shapefile, args.uid_col)
    report = reconcile(irr_years, cohort, allow_unexpected=args.allow_unexpected)
    mapping, assignments, counts = build_mapping(vectors, irr_years, cohort)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "e3_irrigation_stratified_param_mapping.json"
    meta_path = out_dir / "e3_irrigation_stratified_param_mapping_metadata.json"

    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")

    metadata = {
        "experiment": "irrigation-stratified Run 22 parameter transfer, Example 6 / Experiment 3",
        "purpose": (
            "Per-site parameter mapping consumed by "
            "examples/6_Flux_International/transfer_ex5_params.py --params-by-site as the "
            "'ex5_transfer_strat' comparator."
        ),
        "target_class_policy": (
            "equipped for irrigation -> irrigated vector; not equipped -> rainfed vector"
        ),
        "stage1_rule": STAGE1_RULE,
        "stage1_not_persisted_recovery": STAGE1_RECOVERY,
        "stage2_note": STAGE2_NOTE,
        "source_vectors_path": str(vectors_path),
        "source_vectors_sha256": _sha256(vectors_path),
        "class_vectors": vectors,
        "class_vector_sha256": {cls: _vector_sha256(vectors[cls]) for cls in CLASSES},
        "container": str(args.container),
        "container_irr_data_path": "derived/dynamics/irr_data",
        "container_irr_data_keys": ["f_irr", "irr_doys", "irrigated"],
        "cohort_shapefile": str(args.shapefile),
        "cohort_uid_col": args.uid_col,
        "cohort_size": len(cohort),
        "param_names": list(PARAM_FAMILIES),
        "class_counts": counts,
        "assignments": assignments,
        "sites_by_class": {
            cls: sorted(s for s, a in assignments.items() if a["irr_class"] == cls)
            for cls in CLASSES
        },
        "reconciliation": report,
        "allow_unexpected": bool(args.allow_unexpected),
        "flux_role": "validation only; never used in classification or vector construction",
        "meter_role": "validation only; never used in classification or vector construction",
        "model_structure": "canonical Run 22 single-mad coupling",
        "stress_depletion_fraction": None,
        "e5split_materials_used": False,
        "date_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "worktree_dirty": _worktree_dirty(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    # --- console audit --------------------------------------------------------
    print(f"Frozen vectors   : {vectors_path}")
    print(f"  sha256         : {metadata['source_vectors_sha256']}")
    print(f"Container        : {args.container}")
    print(f"Cohort shapefile : {args.shapefile}  ({len(cohort)} sites, uid={args.uid_col})")
    print("\nStage-1 class recovered as 'any irrigated year in derived/dynamics/irr_data'")
    print("  (stage 1 'equipped' is a local variable in calculator.py and is not persisted)")

    print("\nContainer-vs-cohort reconciliation:")
    print(f"  {'scope':<26}{'sites':>7}{'equipped':>10}{'irr site-years':>16}")
    print(
        f"  {'container (all)':<26}{report['container_sites']:>7}"
        f"{report['container_equipped_sites']:>10}"
        f"{report['container_irrigated_site_years']:>16}"
    )
    print(
        f"  {'publication cohort':<26}{report['cohort_sites']:>7}"
        f"{report['cohort_equipped_sites']:>10}"
        f"{report['cohort_irrigated_site_years']:>16}"
    )
    print(f"  container-only equipped   : {report['container_equipped_not_in_cohort_years']}")
    print(f"  reconciles to frozen expectation: {report['reconciles']}")
    if report["failures"]:
        for line in report["failures"]:
            print(f"    MISMATCH: {line}")

    print(f"\nClass counts: {counts}")
    print("\nPer-site assignment:")
    print(f"  {'site':<12}{'class':<11}{'equipped':>9}{'irr_years':>11}")
    for sid in sorted(assignments, key=lambda s: (assignments[s]["irr_class"], s)):
        a = assignments[sid]
        print(f"  {sid:<12}{a['irr_class']:<11}{str(a['equipped']):>9}{a['n_irrigated_years']:>11}")

    print("\nClass vectors applied:")
    print(f"  {'param':<11}{'irrigated':>13}{'rainfed':>13}{'irr-rf':>13}")
    for fam in PARAM_FAMILIES:
        i = vectors["irrigated"][fam]
        r = vectors["rainfed"][fam]
        print(f"  {fam:<11}{i:>13.6f}{r:>13.6f}{i - r:>+13.6f}")

    print(f"\nWrote {mapping_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()

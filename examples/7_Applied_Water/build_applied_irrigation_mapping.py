"""Build the Example 7 per-field parameter mapping for the irrigation-stratified transfer.

Expands the frozen two-vector Example 2 (Run 22) artifact
``paper/data/final/e2_run22_transfer_vectors_by_irrigation.json`` -- shape
``{"irrigated": {8 params}, "rainfed": {8 params}}`` -- into the nested
``{site_id: {8 params}}`` mapping that ``evaluate_applied_water.py --params-json``
consumes, covering every field in the 110-field Example 7 cohort.

Class rule (remote sensing only; **no metered value is ever read**):

  * ``ESPActl_*`` rainfed negative controls -> **rainfed** vector. Provenance is the
    IrrMapper 2000-2024 cache ``data/idwr_wmis/espa_control_irrmapper.csv``, joined on
    the shapefile ``src_id`` (= ``fid2015``); every control must carry
    ``max_irr == 0.0`` (strictly never classified irrigated), which is exactly the
    gate ``select_fields.py::select_espa_controls`` used to pick them.
  * all other cohort fields (``SLV_*``, ``ESPA_*`` metered) -> **irrigated** vector.

The metered truth table is opened for a *provenance cross-check only*, reading just
``site_id`` and ``source`` (the ``ESPA_rainfed_control`` label). ``metered_depth_mm``
and ``metered_volume_af`` are never read, so class assignment cannot leak meter
information into the transfer configuration.

Why full coverage is asserted: ``evaluate_applied_water.py::_resolve_params`` resolves a
nested mapping as ``{fid: vec[fid] for fid in fids if fid in vec}`` -- any field absent
from the mapping is **silently dropped**, never simulated, and silently shrinks ``n`` in
the summary metrics and the negative-control counts. This script therefore fails hard
unless the mapping covers all 110 cohort fields and every container field UID present in
the truth roster.

Outputs (under --out-dir):
    e4_irrigation_stratified_param_mapping.json           - {site_id: {8 params}}
    e4_irrigation_stratified_param_mapping_metadata.json  - provenance + audit

Usage:
    uv run python /home/dgketchum/code/swim-rs/examples/7_Applied_Water/build_applied_irrigation_mapping.py

    uv run python /home/dgketchum/code/swim-rs/examples/7_Applied_Water/build_applied_irrigation_mapping.py \
        --verify-keys /data/ssd1/swim/7_Applied_Water/results/applied_transfer_run22/per_field_year.csv

The downstream Example 7 forward run must target the calibrated container
``/data/ssd1/swim/7_Applied_Water/data/7_Applied_Water_e7cal.swim`` (NOT the base
``7_Applied_Water.swim``), and must use ``--label transfer_run22_by_irrigation`` so
``field_accuracy.py`` finds it at ``results/applied_transfer_run22_by_irrigation``.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
E6_TRANSFER = REPO / "examples" / "6_Flux_International" / "transfer"
if str(E6_TRANSFER) not in sys.path:
    sys.path.insert(0, str(E6_TRANSFER))

# Reuse the validated provenance helpers and the canonical eight-parameter family list
# from the historical pooled-vector builder (import only; that module has no import-time
# side effects).
from build_ex5_cropland_params import (  # noqa: E402
    PARAM_FAMILIES,
    _git_sha,
    _sha256,
)

from swimrs.container import open_container  # noqa: E402

DEFAULT_VECTORS = REPO / "paper" / "data" / "final" / "e2_run22_transfer_vectors_by_irrigation.json"
DEFAULT_FIELDS_SHP = "/data/ssd1/swim/7_Applied_Water/data/gis/applied_water_fields.shp"
DEFAULT_IRRMAPPER_CSV = REPO / "data" / "idwr_wmis" / "espa_control_irrmapper.csv"
DEFAULT_TRUTH_CSV = HERE / "data" / "metered_truth.csv"
DEFAULT_CONTAINER = "/data/ssd1/swim/7_Applied_Water/data/7_Applied_Water_e7cal.swim"
DEFAULT_OUT_DIR = REPO / "paper" / "data" / "final"

# Expected Example 7 cohort composition, keyed by the site_id prefix token.
EXPECTED_COMPOSITION = {"SLV": 50, "ESPA": 50, "ESPActl": 10}
EXPECTED_TOTAL = 110

CONTROL_PREFIX = "ESPActl"
IRRIGATED_CLASS = "irrigated"
RAINFED_CLASS = "rainfed"
TRUTH_CONTROL_SOURCE = "ESPA_rainfed_control"

# The forward-run contract the downstream E4 evaluation must honor.
E4_FORWARD_LABEL = "transfer_run22_by_irrigation"
E4_FORWARD_CONTAINER = DEFAULT_CONTAINER


def _prefix(site_id):
    """Cohort group token: the site_id up to the first underscore.

    Must not be a ``startswith`` test -- ``ESPActl_000`` also starts with ``ESPA``.
    """
    return str(site_id).split("_", 1)[0]


def _git_dirty():
    """True if the worktree has uncommitted changes, None if git is unavailable."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(HERE), text=True)
    except Exception:
        return None
    return bool(out.strip())


def load_vectors(path):
    """Load and validate the frozen two-vector artifact."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Frozen irrigation-stratified vectors not found: {path}\n"
            "Build them first with "
            "examples/6_Flux_International/transfer/build_ex5_irrigation_stratified_params.py "
            "(see paper/notes/irrigation_stratified_transfer_handoff.md). This script will "
            "not fabricate a placeholder."
        )
    with open(path) as f:
        vectors = json.load(f)

    expected_classes = {IRRIGATED_CLASS, RAINFED_CLASS}
    if set(vectors) != expected_classes:
        raise SystemExit(
            f"{path}: expected exactly the classes {sorted(expected_classes)}, "
            f"found {sorted(vectors)}"
        )
    for cls, vec in vectors.items():
        if not isinstance(vec, dict):
            raise SystemExit(f"{path}: class {cls!r} is not a {{param: value}} mapping")
        if set(vec) != set(PARAM_FAMILIES):
            missing = sorted(set(PARAM_FAMILIES) - set(vec))
            extra = sorted(set(vec) - set(PARAM_FAMILIES))
            raise SystemExit(
                f"{path}: class {cls!r} parameter set mismatch "
                f"(missing={missing}, unexpected={extra})"
            )
        for fam, val in vec.items():
            fval = float(val)
            if fval != fval or fval in (float("inf"), float("-inf")):
                raise SystemExit(f"{path}: class {cls!r} parameter {fam!r} is non-finite: {val!r}")
    # Canonical parameter order for every emitted vector.
    return {cls: {fam: float(vectors[cls][fam]) for fam in PARAM_FAMILIES} for cls in vectors}


def load_cohort(fields_shp, allow_unexpected):
    """Load the Example 7 field roster and assert its composition."""
    fields_shp = Path(fields_shp)
    if not fields_shp.exists():
        raise SystemExit(f"Example 7 fields shapefile not found: {fields_shp}")
    gdf = gpd.read_file(fields_shp, engine="fiona")  # fiona is mandatory in this repo

    for col in ("site_id", "src_id"):
        if col not in gdf.columns:
            raise SystemExit(
                f"{fields_shp}: required column {col!r} missing (have {list(gdf.columns)})"
            )
    fields = gdf.drop(columns="geometry", errors="ignore").copy()
    fields["site_id"] = fields["site_id"].astype(str)

    dupes = sorted(fields.loc[fields.site_id.duplicated(), "site_id"].unique())
    if dupes:
        raise SystemExit(f"{fields_shp}: duplicate site_id values: {dupes}")

    fields["group"] = fields.site_id.map(_prefix)
    composition = {g: int(n) for g, n in fields.group.value_counts().items()}

    problems = []
    unknown = sorted(set(composition) - set(EXPECTED_COMPOSITION))
    if unknown:
        problems.append(f"unexpected site_id prefixes {unknown}")
    for group, want in EXPECTED_COMPOSITION.items():
        got = composition.get(group, 0)
        if got != want:
            problems.append(f"{group}: expected {want} fields, found {got}")
    if len(fields) != EXPECTED_TOTAL:
        problems.append(f"total: expected {EXPECTED_TOTAL} fields, found {len(fields)}")

    if problems:
        message = f"{fields_shp}: cohort composition mismatch -- " + "; ".join(problems)
        if not allow_unexpected:
            raise SystemExit(
                message + "\nPass --allow-unexpected to proceed with a non-canonical cohort."
            )
        print(f"WARNING (--allow-unexpected): {message}")

    return fields.sort_values("site_id").reset_index(drop=True), composition


def assign_classes(fields, irrmapper_csv):
    """Assign each cohort field an irrigation class from remote sensing only.

    Returns (assignments, control_audit) where ``assignments`` maps site_id -> class and
    ``control_audit`` is a DataFrame of the IrrMapper evidence for the rainfed controls.
    """
    irrmapper_csv = Path(irrmapper_csv)
    if not irrmapper_csv.exists():
        raise SystemExit(
            f"IrrMapper rainfed cache not found: {irrmapper_csv}\n"
            "Rebuild it with examples/7_Applied_Water/espa_control_irrmapper.py."
        )
    irr = pd.read_csv(irrmapper_csv, usecols=["fid2015", "mean_irr", "max_irr"])
    if irr.fid2015.duplicated().any():
        dupes = sorted(irr.loc[irr.fid2015.duplicated(), "fid2015"].unique())
        raise SystemExit(f"{irrmapper_csv}: duplicate fid2015 keys: {dupes}")
    irr = irr.set_index(irr.fid2015.astype(int))

    controls = fields[fields.group == CONTROL_PREFIX]
    rows = []
    for _, rec in controls.iterrows():
        raw = rec.src_id
        if pd.isna(raw) or str(raw).strip() == "":
            raise SystemExit(
                f"control {rec.site_id} has no src_id (fid2015) in the fields shapefile, so its "
                "rainfed status cannot be traced to IrrMapper. Rebuild the cohort with "
                "select_fields.py rather than guessing the class."
            )
        try:
            fid2015 = int(str(raw).strip())
        except ValueError as exc:
            raise SystemExit(
                f"control {rec.site_id}: src_id {raw!r} is not an integer fid2015 key"
            ) from exc
        if fid2015 not in irr.index:
            raise SystemExit(
                f"control {rec.site_id}: fid2015 {fid2015} absent from {irrmapper_csv}. "
                "The rainfed class has no remote-sensing provenance; refusing to assign it."
            )
        max_irr = float(irr.at[fid2015, "max_irr"])
        mean_irr = float(irr.at[fid2015, "mean_irr"])
        if max_irr != 0.0:
            raise SystemExit(
                f"control {rec.site_id} (fid2015 {fid2015}): IrrMapper max_irr={max_irr} != 0.0, "
                "so it was classified irrigated in at least one year 2000-2024 and is not a "
                "valid rainfed control."
            )
        rows.append(
            {
                "site_id": rec.site_id,
                "fid2015": fid2015,
                "irrmapper_mean_irr": mean_irr,
                "irrmapper_max_irr": max_irr,
            }
        )
    # Columns are declared explicitly: with --allow-unexpected a control-less cohort
    # leaves `rows` empty, and an empty frame with no columns cannot be sorted.
    control_audit = (
        pd.DataFrame(
            rows, columns=["site_id", "fid2015", "irrmapper_mean_irr", "irrmapper_max_irr"]
        )
        .sort_values("site_id")
        .reset_index(drop=True)
    )

    assignments = {}
    for _, rec in fields.iterrows():
        cls = RAINFED_CLASS if rec.group == CONTROL_PREFIX else IRRIGATED_CLASS
        if rec.site_id in assignments:
            raise SystemExit(f"site_id {rec.site_id} assigned a class twice")
        assignments[rec.site_id] = cls

    if len(assignments) != len(fields):
        raise SystemExit(
            f"class assignment covers {len(assignments)} of {len(fields)} cohort fields"
        )
    return assignments, control_audit


def cross_check_truth(truth_csv, assignments):
    """Cross-check the class split against the truth table's ``source`` provenance label.

    Reads only ``site_id`` and ``source``. No metered value is read.
    """
    truth_csv = Path(truth_csv)
    if not truth_csv.exists():
        raise SystemExit(f"Truth roster not found: {truth_csv}")
    truth = pd.read_csv(truth_csv, usecols=["site_id", "source"])
    truth["site_id"] = truth.site_id.astype(str)

    label = truth.groupby("site_id").source.agg(lambda s: sorted(set(s)))
    mixed = {sid: srcs for sid, srcs in label.items() if len(srcs) > 1}
    expected = {}
    for sid, srcs in label.items():
        if TRUTH_CONTROL_SOURCE in srcs:
            expected[sid] = RAINFED_CLASS
        else:
            expected[sid] = IRRIGATED_CLASS

    agree, disagree, only_truth = [], [], []
    for sid, exp in expected.items():
        got = assignments.get(sid)
        if got is None:
            only_truth.append(sid)
        elif got == exp:
            agree.append(sid)
        else:
            disagree.append({"site_id": sid, "mapping_class": got, "truth_source_class": exp})
    only_mapping = sorted(set(assignments) - set(expected))

    return {
        "truth_csv": str(truth_csv),
        "truth_csv_sha256": _sha256(truth_csv),
        "columns_read": ["site_id", "source"],
        "metered_columns_read": [],
        "n_truth_sites": int(len(expected)),
        "n_agree": len(agree),
        "n_disagree": len(disagree),
        "disagreements": disagree,
        "sites_in_truth_not_in_mapping": only_truth,
        "sites_in_mapping_not_in_truth": only_mapping,
        "sites_with_mixed_source_labels": mixed,
    }


def check_container_coverage(container_path, truth_csv, assignments):
    """Assert the mapping covers every container field UID in the truth roster.

    This is the exact ``fids`` list ``evaluate_applied_water.py`` builds, so a gap here
    is a field that would be silently dropped from the forward run.
    """
    container_path = Path(container_path)
    if not container_path.exists():
        raise SystemExit(
            f"Example 7 container not found: {container_path}\n"
            "The E4 forward run targets the calibrated container "
            f"{E4_FORWARD_CONTAINER}."
        )
    truth_sites = set(pd.read_csv(truth_csv, usecols=["site_id"]).site_id.astype(str))
    container = open_container(str(container_path), mode="r")
    try:
        uids = [str(u) for u in container.field_uids]
    finally:
        close = getattr(container, "close", None)
        if callable(close):
            close()

    eval_fids = [u for u in uids if u in truth_sites]
    missing = sorted(set(eval_fids) - set(assignments))
    if missing:
        raise SystemExit(
            f"{len(missing)} container field(s) in the truth roster are absent from the "
            f"mapping and would be SILENTLY DROPPED by evaluate_applied_water.py: {missing}"
        )
    return {
        "container": str(container_path),
        "n_container_field_uids": len(uids),
        "n_eval_fids": len(eval_fids),
        "eval_fids_all_covered": True,
        "container_uids_not_in_mapping": sorted(set(uids) - set(assignments)),
    }


def verify_keys(per_field_year_csv, assignments):
    """Pre-flight the handoff's identical-``(site_id, year)``-key requirement."""
    path = Path(per_field_year_csv)
    if not path.exists():
        raise SystemExit(f"--verify-keys file not found: {path}")
    ref = pd.read_csv(path, usecols=["site_id", "year"])
    ref["site_id"] = ref.site_id.astype(str)
    sites = sorted(set(ref.site_id))
    missing = sorted(set(sites) - set(assignments))
    return {
        "per_field_year_csv": str(path),
        "n_rows": int(len(ref)),
        "n_sites": len(sites),
        "n_site_year_keys": int(len(ref.drop_duplicates(["site_id", "year"]))),
        "sites_missing_from_mapping": missing,
        "all_sites_covered": not missing,
        "mapping_sites_absent_from_file": sorted(set(assignments) - set(sites)),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vectors", default=str(DEFAULT_VECTORS), help="frozen two-vector JSON")
    ap.add_argument("--fields-shp", default=DEFAULT_FIELDS_SHP, help="Example 7 fields shapefile")
    ap.add_argument(
        "--irrmapper-csv", default=str(DEFAULT_IRRMAPPER_CSV), help="IrrMapper 2000-2024 cache"
    )
    ap.add_argument(
        "--truth-csv",
        default=str(DEFAULT_TRUTH_CSV),
        help="truth roster; only site_id/source are read (never a metered value)",
    )
    ap.add_argument("--container", default=DEFAULT_CONTAINER, help="Example 7 container to audit")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where to write the artifacts")
    ap.add_argument("--out-json", default=None, help="explicit mapping JSON path")
    ap.add_argument("--out-meta", default=None, help="explicit metadata JSON path")
    ap.add_argument(
        "--allow-unexpected",
        action="store_true",
        help="downgrade the cohort-composition assertions (50/50/10, 110 total) to warnings",
    )
    ap.add_argument(
        "--verify-keys",
        default=None,
        help="pooled run per_field_year.csv to pre-flight for identical (site_id, year) keys",
    )
    args = ap.parse_args()

    vectors = load_vectors(args.vectors)
    fields, composition = load_cohort(args.fields_shp, args.allow_unexpected)
    assignments, control_audit = assign_classes(fields, args.irrmapper_csv)
    cross_check = cross_check_truth(args.truth_csv, assignments)
    container_audit = check_container_coverage(args.container, args.truth_csv, assignments)

    mapping = {sid: dict(vectors[cls]) for sid, cls in sorted(assignments.items())}
    if len(mapping) != len(assignments):
        raise SystemExit("mapping lost fields during expansion")

    class_counts = {
        cls: int(sum(1 for c in assignments.values() if c == cls)) for cls in sorted(vectors)
    }
    group_class = (
        fields.assign(irrigation_class=fields.site_id.map(assignments))
        .groupby(["group", "irrigation_class"])
        .size()
        .rename("n")
        .reset_index()
    )

    keys_check = verify_keys(args.verify_keys, assignments) if args.verify_keys else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = (
        Path(args.out_json)
        if args.out_json
        else out_dir / "e4_irrigation_stratified_param_mapping.json"
    )
    meta_path = (
        Path(args.out_meta)
        if args.out_meta
        else out_dir / "e4_irrigation_stratified_param_mapping_metadata.json"
    )

    # site_id keys are already sorted; do NOT sort_keys, which would also reorder the
    # nested parameter names away from the canonical PARAM_FAMILIES order.
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")

    metadata = {
        "artifact": "Example 7 (E4) irrigation-stratified per-field parameter mapping",
        "purpose": (
            "Expand the frozen Example 2 Run 22 two-vector irrigation-stratified transfer "
            "into the nested {site_id: {param: value}} mapping consumed by "
            "evaluate_applied_water.py --params-json."
        ),
        "param_names": list(PARAM_FAMILIES),
        "class_rule": (
            "site_id prefix ESPActl_ AND IrrMapper max_irr == 0.0 over 2000-2024 -> rainfed "
            "vector; all other cohort fields (SLV_*, ESPA_*) -> irrigated vector. Parameters "
            "are fixed by site and never switch annually; the model's internal annual "
            "irrigation status continues to control scheduler activation."
        ),
        "class_basis": "remote sensing (IrrMapper 2000-2024) plus the cohort site_id prefix",
        "no_metered_value_read": True,
        "no_metered_value_read_statement": (
            "No metered applied-water value was read to build this mapping. The truth table was "
            "opened only for a provenance cross-check, reading exclusively the site_id and "
            "source columns; metered_depth_mm and metered_volume_af were never read. Class "
            "assignment is remote-sensing based (IrrMapper), so meter truth remains withheld "
            "until scoring."
        ),
        "source_vectors_json": str(Path(args.vectors).resolve()),
        "source_vectors_json_sha256": _sha256(args.vectors),
        "source_vectors": vectors,
        "fields_shapefile": str(Path(args.fields_shp).resolve()),
        "irrmapper_cache_csv": str(Path(args.irrmapper_csv).resolve()),
        "irrmapper_cache_csv_sha256": _sha256(args.irrmapper_csv),
        "cohort_composition": composition,
        "cohort_composition_expected": EXPECTED_COMPOSITION,
        "cohort_total": int(len(fields)),
        "cohort_total_expected": EXPECTED_TOTAL,
        "composition_assertions_enforced": not args.allow_unexpected,
        "n_fields_mapped": len(mapping),
        "class_counts": class_counts,
        "class_counts_by_group": group_class.to_dict(orient="records"),
        "field_class_assignment": {sid: assignments[sid] for sid in sorted(assignments)},
        "rainfed_control_irrmapper_audit": control_audit.to_dict(orient="records"),
        "truth_table_cross_check": cross_check,
        "container_coverage_audit": container_audit,
        "verify_keys": keys_check,
        "downstream_forward_run": {
            "container": E4_FORWARD_CONTAINER,
            "label": E4_FORWARD_LABEL,
            "reminder": (
                "The E4 forward run MUST pass --container "
                f"{E4_FORWARD_CONTAINER} (the calibrated _e7cal container, NOT the base "
                f"7_Applied_Water.swim) and --label {E4_FORWARD_LABEL} so field_accuracy.py "
                f"resolves results/applied_{E4_FORWARD_LABEL}."
            ),
            "silent_drop_hazard": (
                "evaluate_applied_water.py::_resolve_params keeps only fields present in this "
                "mapping; a missing field is never simulated and silently shrinks n and the "
                "negative-control counts. Coverage of all cohort fields is asserted here."
            ),
        },
        "mapping_json": str(map_path.resolve()),
        "git_sha": _git_sha(),
        "git_worktree_dirty": _git_dirty(),
        "date_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    # ---- audit table ----
    print(f"Wrote {map_path}")
    print(f"Wrote {meta_path}")

    print(f"\nFrozen source vectors ({Path(args.vectors).name}):")
    print(f"  {'param':<11} {IRRIGATED_CLASS:>12} {RAINFED_CLASS:>12}")
    for fam in PARAM_FAMILIES:
        print(
            f"  {fam:<11} {vectors[IRRIGATED_CLASS][fam]:>12.6g} "
            f"{vectors[RAINFED_CLASS][fam]:>12.6g}"
        )

    print(f"\nCohort composition and class assignment ({len(mapping)} fields mapped):")
    print(f"  {'group':<9} {'n':>4}  {'class':<10}")
    for rec in group_class.to_dict(orient="records"):
        print(f"  {rec['group']:<9} {rec['n']:>4}  {rec['irrigation_class']:<10}")
    print(f"  {'-' * 9} {'-' * 4}  {'-' * 10}")
    for cls in sorted(class_counts):
        print(f"  {'TOTAL':<9} {class_counts[cls]:>4}  {cls:<10}")

    print("\nRainfed controls -- IrrMapper 2000-2024 provenance:")
    print(f"  {'site_id':<13} {'fid2015':>8} {'mean_irr':>9} {'max_irr':>8}")
    for rec in control_audit.to_dict(orient="records"):
        print(
            f"  {rec['site_id']:<13} {rec['fid2015']:>8} "
            f"{rec['irrmapper_mean_irr']:>9.4f} {rec['irrmapper_max_irr']:>8.4f}"
        )

    cc = cross_check
    print("\nTruth-table cross-check (site_id + source only; no metered value read):")
    print(f"  truth sites          : {cc['n_truth_sites']}")
    print(f"  class agreement      : {cc['n_agree']}")
    print(f"  class disagreement   : {cc['n_disagree']}")
    print(f"  in truth, unmapped   : {len(cc['sites_in_truth_not_in_mapping'])}")
    print(f"  mapped, not in truth : {len(cc['sites_in_mapping_not_in_truth'])}")
    if (
        cc["n_disagree"]
        or cc["sites_in_truth_not_in_mapping"]
        or cc["sites_with_mixed_source_labels"]
    ):
        print("  WARNING: cross-check anomalies:")
        for row in cc["disagreements"]:
            print(
                f"    {row['site_id']}: mapping={row['mapping_class']} "
                f"truth_source={row['truth_source_class']}"
            )
        for sid in cc["sites_in_truth_not_in_mapping"]:
            print(f"    {sid}: in truth roster but not in the mapping")
        for sid, srcs in cc["sites_with_mixed_source_labels"].items():
            print(f"    {sid}: multiple source labels {srcs}")

    ca = container_audit
    print("\nContainer coverage audit:")
    print(f"  container            : {ca['container']}")
    print(f"  field UIDs           : {ca['n_container_field_uids']}")
    print(f"  evaluated fids       : {ca['n_eval_fids']} (container UIDs in the truth roster)")
    print("  all evaluated fids covered by the mapping: True")

    if keys_check:
        kc = keys_check
        print("\nKey pre-flight (--verify-keys):")
        print(f"  file                 : {kc['per_field_year_csv']}")
        print(f"  rows                 : {kc['n_rows']}")
        print(f"  (site_id, year) keys : {kc['n_site_year_keys']}")
        print(f"  distinct site_id     : {kc['n_sites']}")
        print(f"  all covered          : {kc['all_sites_covered']}")
        if kc["sites_missing_from_mapping"]:
            print(f"  MISSING from mapping : {kc['sites_missing_from_mapping']}")
        extra = kc["mapping_sites_absent_from_file"]
        print(f"  mapped but absent    : {len(extra)}" + (f" {extra}" if extra else ""))

    print(
        f"\nReminder: the E4 forward run must pass --container {E4_FORWARD_CONTAINER} "
        f"and --label {E4_FORWARD_LABEL}."
    )


if __name__ == "__main__":
    main()

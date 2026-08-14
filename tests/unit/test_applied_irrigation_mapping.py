"""Unit tests for the Example 7 applied-water per-field parameter mapping builder.

The builder (``examples/7_Applied_Water/build_applied_irrigation_mapping.py``) expands
the frozen two-class Run 22 transfer vector into ``{site_id: {param: value}}`` for the
E4 forward run. Two properties make it worth testing hard:

1. Class assignment is a *string* operation on the cohort ``site_id``, and the rainfed
   negative controls (``ESPActl_*``) share a string prefix with the metered ESPA fields
   (``ESPA_*``). A ``startswith`` rule silently hands all ten negative controls the
   irrigated vector, destroying the control arm without any error.
2. ``evaluate_applied_water.py::_resolve_params`` keeps only fields present in the
   mapping (``{fid: vec[fid] for fid in fids if fid in vec}``), so an incomplete mapping
   silently shrinks ``n`` instead of failing. The coverage assertions are the only thing
   standing between a partial mapping and an under-powered published metric.

Class assignment must also stay meter-blind: the truth table is opened for provenance
only, reading ``site_id`` and ``source`` and never a metered value.
"""

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
E7_DIR = REPO_ROOT / "examples" / "7_Applied_Water"
E6_TRANSFER = REPO_ROOT / "examples" / "6_Flux_International" / "transfer"


def _load_builder():
    """Import the builder by path (``examples/`` is not an importable package).

    Only the shared transfer dir goes on sys.path -- the builder reuses
    ``build_ex5_cropland_params`` from there for PARAM_FAMILIES and the provenance
    helpers. The Example 7 dir is deliberately left off sys.path so its generically
    named siblings (``calibrate.py``, ``data_extract.py``) cannot shadow real imports
    for the rest of the pytest session.
    """
    if str(E6_TRANSFER) not in sys.path:
        sys.path.insert(0, str(E6_TRANSFER))
    spec = importlib.util.spec_from_file_location(
        "build_applied_irrigation_mapping",
        E7_DIR / "build_applied_irrigation_mapping.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bld = _load_builder()
FAMILIES = list(bld.PARAM_FAMILIES)

# Two deliberately disjoint class vectors so a mis-assignment cannot go unnoticed.
IRRIGATED_VEC = {fam: 1.0 + i for i, fam in enumerate(FAMILIES)}
RAINFED_VEC = {fam: 100.0 + i for i, fam in enumerate(FAMILIES)}

CTL_FID_START = 100


# --------------------------------------------------------------------------- #
# Fixtures: cohort shapefile, IrrMapper cache, truth roster, stub container
# --------------------------------------------------------------------------- #
def _canonical_records(n_slv=50, n_espa=50, n_ctl=10):
    """The 50 SLV / 50 ESPA / 10 ESPActl Example 7 cohort as (site_id, src_id) pairs."""
    recs = [(f"SLV_{i:03d}", 900 + i) for i in range(n_slv)]
    recs += [(f"ESPA_{i:03d}", 500 + i) for i in range(n_espa)]
    recs += [(f"ESPActl_{i:03d}", CTL_FID_START + i) for i in range(n_ctl)]
    return recs


def _fields_shp(tmp_path, records, name="fields.shp", drop_src_id=False):
    """Write a synthetic Example 7 fields shapefile from (site_id, src_id) pairs.

    ``src_id`` of None becomes an empty attribute, which round-trips through the .dbf as
    NaN -- exactly how a cohort field with no ``fid2015`` provenance shows up.
    """
    data = {
        "site_id": [str(sid) for sid, _ in records],
        "geometry": [Point(i, i) for i in range(len(records))],
    }
    if not drop_src_id:
        data = {
            "site_id": data["site_id"],
            "src_id": ["" if src is None else str(src) for _, src in records],
            "geometry": data["geometry"],
        }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    path = tmp_path / name
    gdf.to_file(path, driver="ESRI Shapefile", engine="fiona")
    return path


def _irrmapper_csv(tmp_path, entries=None, name="espa_control_irrmapper.csv", duplicate=None):
    """Write the IrrMapper 2000-2024 control cache: {fid2015: (mean_irr, max_irr)}.

    Defaults to the ten canonical controls, all strictly never irrigated.
    """
    if entries is None:
        entries = {CTL_FID_START + i: (0.0, 0.0) for i in range(10)}
    fids = list(entries)
    rows = {
        "fid2015": fids,
        "f_acres": [40.0] * len(fids),
        "mean_irr": [entries[f][0] for f in fids],
        "max_irr": [entries[f][1] for f in fids],
    }
    df = pd.DataFrame(rows)
    if duplicate is not None:
        df = pd.concat([df, df[df.fid2015 == duplicate]], ignore_index=True)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _truth_csv(tmp_path, rows, name="metered_truth.csv"):
    """Write a truth roster with ONLY site_id and source columns.

    No ``metered_depth_mm`` / ``metered_volume_af`` column exists in this fixture at all,
    so any attempt by the builder to read a metered value would raise, not pass quietly.
    """
    df = pd.DataFrame(rows, columns=["site_id", "source"])
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _truth_rows(records, years=(2020, 2021)):
    """One truth row per (field, year); controls carry the rainfed provenance label."""
    rows = []
    for sid, _ in records:
        source = (
            bld.TRUTH_CONTROL_SOURCE
            if bld._prefix(sid) == bld.CONTROL_PREFIX
            else ("SLV_meter" if sid.startswith("SLV") else "ESPA_meter")
        )
        rows.extend([(sid, source) for _ in years])
    return rows


def _vectors_json(tmp_path, payload=None, name="vectors.json"):
    """Write the frozen two-class vector artifact."""
    if payload is None:
        payload = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _stub_container(tmp_path, name, fids):
    """Minimal .swim-shaped zarr store carrying only geometry/uid.

    Mirrors ``SwimContainer.create``: uid is a VariableLengthUTF8 array, which is what
    ``field_uids`` is loaded from (a plain object-dtype dataset fails on zarr 3.1.5).
    """
    zarr = pytest.importorskip("zarr")
    from zarr.core.dtype import VariableLengthUTF8

    path = tmp_path / name
    root = zarr.open(str(path), mode="w")
    uid = root.create_array("geometry/uid", shape=(len(fids),), dtype=VariableLengthUTF8())
    uid[:] = list(fids)
    return path


def _cohort(tmp_path, records=None, allow_unexpected=False, irr_entries=None, shp_name="c.shp"):
    """Load a cohort and assign classes end to end; returns (fields, assignments)."""
    records = _canonical_records() if records is None else records
    shp = _fields_shp(tmp_path, records, name=shp_name)
    fields, _ = bld.load_cohort(shp, allow_unexpected)
    assignments, audit = bld.assign_classes(fields, _irrmapper_csv(tmp_path, irr_entries))
    return fields, assignments, audit


# --------------------------------------------------------------------------- #
# The prefix trap: ESPActl_* vs ESPA_*
# --------------------------------------------------------------------------- #
class TestPrefixTrap:
    def test_prefix_splits_on_first_underscore_not_startswith(self):
        """``ESPActl_000`` also starts with ``ESPA``, so ``startswith`` is unusable here.

        The negative-control arm exists only because ``ESPActl`` resolves to its own
        group token. A ``startswith('ESPA')`` class test matches the controls too and
        would hand all ten of them the irrigated vector -- no error, no missing key, just
        a silently destroyed control arm and a fabricated 10-field agreement.
        """
        assert "ESPActl_000".startswith("ESPA")  # the trap this function must avoid
        assert bld._prefix("ESPActl_000") == "ESPActl"
        assert bld._prefix("ESPA_000") == "ESPA"
        assert bld._prefix("SLV_049") == "SLV"
        assert bld._prefix("ESPActl_000") != bld._prefix("ESPA_000")
        # Extra underscores in the tail must not split off further tokens.
        assert bld._prefix("ESPActl_000_b") == "ESPActl"
        assert bld._prefix("NOUNDERSCORE") == "NOUNDERSCORE"

    def test_control_and_metered_espa_land_in_different_classes(self, tmp_path):
        """One ESPA meter and one ESPA control must not share a parameter vector."""
        records = [("SLV_000", 900), ("ESPA_000", 500), ("ESPActl_000", CTL_FID_START)]
        _, assignments, _ = _cohort(tmp_path, records, allow_unexpected=True)
        assert assignments["ESPA_000"] == bld.IRRIGATED_CLASS
        assert assignments["ESPActl_000"] == bld.RAINFED_CLASS
        assert assignments["ESPA_000"] != assignments["ESPActl_000"]

    def test_startswith_rule_would_misclassify_all_ten_controls(self, tmp_path):
        """Quantify the trap on the real cohort: 100 irrigated / 10 rainfed, not 110/0."""
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        counts = pd.Series(list(assignments.values())).value_counts().to_dict()
        assert counts == {bld.IRRIGATED_CLASS: 100, bld.RAINFED_CLASS: 10}

        # What a startswith('ESPA') irrigated-test would have produced.
        n_startswith_espa = sum(1 for sid, _ in records if sid.startswith("ESPA"))
        assert n_startswith_espa == 60  # 50 metered + the 10 controls it swallows
        n_prefix_espa = sum(1 for sid, _ in records if bld._prefix(sid) == "ESPA")
        assert n_prefix_espa == 50


# --------------------------------------------------------------------------- #
# Class assignment and vector expansion
# --------------------------------------------------------------------------- #
class TestClassAssignment:
    def test_slv_and_espa_irrigated_controls_rainfed(self, tmp_path):
        _, assignments, _ = _cohort(tmp_path)
        for sid, cls in assignments.items():
            want = (
                bld.RAINFED_CLASS if bld._prefix(sid) == bld.CONTROL_PREFIX else bld.IRRIGATED_CLASS
            )
            assert cls == want, sid

    def test_expanded_mapping_equals_the_class_vector(self, tmp_path):
        """Every field receives its class vector verbatim, in canonical family order."""
        vectors = bld.load_vectors(_vectors_json(tmp_path))
        _, assignments, _ = _cohort(tmp_path)
        mapping = {sid: dict(vectors[cls]) for sid, cls in sorted(assignments.items())}

        assert len(mapping) == bld.EXPECTED_TOTAL
        assert mapping["SLV_000"] == IRRIGATED_VEC
        assert mapping["ESPA_049"] == IRRIGATED_VEC
        assert mapping["ESPActl_009"] == RAINFED_VEC
        assert list(mapping["ESPActl_009"]) == FAMILIES
        assert set(mapping) == {sid for sid in assignments}

    def test_control_audit_records_irrmapper_evidence(self, tmp_path):
        """The rainfed class is provenanced per control, not asserted in bulk."""
        _, _, audit = _cohort(tmp_path)
        assert len(audit) == 10
        assert list(audit.columns) == [
            "site_id",
            "fid2015",
            "irrmapper_mean_irr",
            "irrmapper_max_irr",
        ]
        assert audit.irrmapper_max_irr.eq(0.0).all()
        assert audit.site_id.tolist() == sorted(audit.site_id.tolist())


# --------------------------------------------------------------------------- #
# Control gating on the IrrMapper cache
# --------------------------------------------------------------------------- #
class TestControlGating:
    def test_control_absent_from_irrmapper_raises(self, tmp_path):
        """A control with no remote-sensing record cannot be declared rainfed."""
        entries = {CTL_FID_START + i: (0.0, 0.0) for i in range(9)}  # drops fid 109
        with pytest.raises(SystemExit, match="absent from"):
            _cohort(tmp_path, irr_entries=entries)

    def test_control_with_nonzero_max_irr_raises(self, tmp_path):
        """max_irr > 0 means IrrMapper called it irrigated at least once: not a control."""
        entries = {CTL_FID_START + i: (0.0, 0.0) for i in range(10)}
        entries[CTL_FID_START + 3] = (0.004, 0.09)
        with pytest.raises(SystemExit, match="classified irrigated in at least one year"):
            _cohort(tmp_path, irr_entries=entries)

    def test_control_gate_is_strict_zero(self, tmp_path):
        """Even a trace max_irr fails; the gate matches select_fields.py exactly."""
        entries = {CTL_FID_START + i: (0.0, 0.0) for i in range(10)}
        entries[CTL_FID_START] = (1e-9, 1e-9)
        with pytest.raises(SystemExit, match=r"max_irr=1e-09 != 0\.0"):
            _cohort(tmp_path, irr_entries=entries)

    def test_control_without_src_id_raises(self, tmp_path):
        """No fid2015 key means the rainfed status is untraceable; refuse to guess."""
        records = _canonical_records()
        records[-1] = ("ESPActl_009", None)
        with pytest.raises(SystemExit, match="has no src_id"):
            _cohort(tmp_path, records)

    def test_non_integer_src_id_raises(self, tmp_path):
        records = _canonical_records()
        records[-1] = ("ESPActl_009", "not-an-int")
        with pytest.raises(SystemExit, match="not an integer fid2015 key"):
            _cohort(tmp_path, records)

    def test_duplicate_fid2015_in_cache_raises(self, tmp_path):
        """An ambiguous cache key would silently pick one of two irrigation histories."""
        records = _canonical_records()
        shp = _fields_shp(tmp_path, records, name="dupcache.shp")
        fields, _ = bld.load_cohort(shp, False)
        csv = _irrmapper_csv(tmp_path, duplicate=CTL_FID_START + 2)
        with pytest.raises(SystemExit, match="duplicate fid2015 keys"):
            bld.assign_classes(fields, csv)

    def test_missing_irrmapper_cache_raises(self, tmp_path):
        records = _canonical_records()
        shp = _fields_shp(tmp_path, records, name="nocache.shp")
        fields, _ = bld.load_cohort(shp, False)
        with pytest.raises(SystemExit, match="IrrMapper rainfed cache not found"):
            bld.assign_classes(fields, tmp_path / "nope.csv")

    def test_irrigated_fields_need_no_irrmapper_entry(self, tmp_path):
        """Only the controls are gated; SLV/ESPA meters are irrigated by cohort design."""
        records = _canonical_records()
        _, assignments, audit = _cohort(tmp_path, records)
        assert len(audit) == 10  # 100 metered fields never touched the cache
        assert assignments["SLV_000"] == bld.IRRIGATED_CLASS


# --------------------------------------------------------------------------- #
# Cohort composition assertions
# --------------------------------------------------------------------------- #
class TestCohortComposition:
    def test_canonical_composition_accepted(self, tmp_path):
        shp = _fields_shp(tmp_path, _canonical_records(), name="canon.shp")
        fields, composition = bld.load_cohort(shp, False)
        assert bld.EXPECTED_COMPOSITION == {"SLV": 50, "ESPA": 50, "ESPActl": 10}
        assert bld.EXPECTED_TOTAL == 110
        assert composition == bld.EXPECTED_COMPOSITION
        assert len(fields) == 110
        assert fields.site_id.tolist() == sorted(fields.site_id.tolist())

    def test_wrong_composition_raises(self, tmp_path):
        """A dropped field changes n in the published metrics, so it must fail loudly."""
        shp = _fields_shp(tmp_path, _canonical_records(n_slv=49), name="short.shp")
        with pytest.raises(SystemExit, match="SLV: expected 50 fields, found 49"):
            bld.load_cohort(shp, False)

    def test_missing_control_arm_raises(self, tmp_path):
        shp = _fields_shp(tmp_path, _canonical_records(n_ctl=0), name="noctl.shp")
        with pytest.raises(SystemExit, match="ESPActl: expected 10 fields, found 0"):
            bld.load_cohort(shp, False)

    def test_allow_unexpected_downgrades_to_warning(self, tmp_path, capsys):
        shp = _fields_shp(tmp_path, _canonical_records(n_slv=49), name="short2.shp")
        fields, composition = bld.load_cohort(shp, True)
        assert len(fields) == 109
        assert composition["SLV"] == 49
        out = capsys.readouterr().out
        assert "WARNING (--allow-unexpected)" in out
        assert "cohort composition mismatch" in out

    def test_duplicate_site_id_raises(self, tmp_path):
        """Duplicate keys would collapse two fields onto one mapping entry."""
        records = _canonical_records()
        records[1] = ("SLV_000", 901)
        shp = _fields_shp(tmp_path, records, name="dupe.shp")
        with pytest.raises(SystemExit, match=r"duplicate site_id values: \['SLV_000'\]"):
            bld.load_cohort(shp, False)

    def test_unknown_prefix_raises(self, tmp_path):
        records = _canonical_records()
        records[0] = ("WIMAS_000", 900)
        shp = _fields_shp(tmp_path, records, name="unknown.shp")
        with pytest.raises(SystemExit, match=r"unexpected site_id prefixes \['WIMAS'\]"):
            bld.load_cohort(shp, False)

    def test_missing_required_column_raises(self, tmp_path):
        shp = _fields_shp(tmp_path, _canonical_records(), name="nosrc.shp", drop_src_id=True)
        with pytest.raises(SystemExit, match="required column 'src_id' missing"):
            bld.load_cohort(shp, False)

    def test_missing_shapefile_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="fields shapefile not found"):
            bld.load_cohort(tmp_path / "ghost.shp", False)


# --------------------------------------------------------------------------- #
# Frozen two-vector artifact validation
# --------------------------------------------------------------------------- #
class TestLoadVectors:
    def test_loads_both_classes_in_canonical_order(self, tmp_path):
        vectors = bld.load_vectors(_vectors_json(tmp_path))
        assert set(vectors) == {bld.IRRIGATED_CLASS, bld.RAINFED_CLASS}
        for vec in vectors.values():
            assert list(vec) == FAMILIES
            assert all(isinstance(v, float) for v in vec.values())
        assert vectors[bld.IRRIGATED_CLASS] != vectors[bld.RAINFED_CLASS]

    def test_family_order_is_normalized_not_inherited(self, tmp_path):
        """A shuffled source JSON still emits PARAM_FAMILIES order."""
        shuffled = {
            "irrigated": dict(reversed(list(IRRIGATED_VEC.items()))),
            "rainfed": dict(reversed(list(RAINFED_VEC.items()))),
        }
        vectors = bld.load_vectors(_vectors_json(tmp_path, shuffled, name="shuffled.json"))
        assert list(vectors["irrigated"]) == FAMILIES

    def test_missing_file_raises_without_fabricating(self, tmp_path):
        with pytest.raises(SystemExit, match="will not fabricate a placeholder"):
            bld.load_vectors(tmp_path / "absent.json")

    def test_single_class_artifact_raises(self, tmp_path):
        """A pooled (one-vector) artifact must not be accepted as stratified."""
        path = _vectors_json(tmp_path, {"irrigated": dict(IRRIGATED_VEC)}, name="pooled.json")
        with pytest.raises(SystemExit, match="expected exactly the classes"):
            bld.load_vectors(path)

    def test_flat_vector_artifact_raises(self, tmp_path):
        """The flat {param: value} shape has neither class key."""
        path = _vectors_json(tmp_path, dict(IRRIGATED_VEC), name="flat.json")
        with pytest.raises(SystemExit, match="expected exactly the classes"):
            bld.load_vectors(path)

    def test_missing_parameter_family_raises(self, tmp_path):
        incomplete = {fam: v for fam, v in RAINFED_VEC.items() if fam != "swe_beta"}
        path = _vectors_json(
            tmp_path,
            {"irrigated": dict(IRRIGATED_VEC), "rainfed": incomplete},
            name="missingfam.json",
        )
        with pytest.raises(SystemExit, match=r"missing=\['swe_beta'\]"):
            bld.load_vectors(path)

    def test_unexpected_parameter_raises(self, tmp_path):
        extra = dict(IRRIGATED_VEC)
        extra["p_stress"] = 0.5
        path = _vectors_json(
            tmp_path, {"irrigated": extra, "rainfed": dict(RAINFED_VEC)}, name="extra.json"
        )
        with pytest.raises(SystemExit, match=r"unexpected=\['p_stress'\]"):
            bld.load_vectors(path)

    def test_non_finite_value_raises(self, tmp_path):
        """A NaN would propagate into every field of that class and silently poison the run."""
        bad = dict(RAINFED_VEC)
        bad["mad"] = float("nan")
        path = tmp_path / "nan.json"
        path.write_text(json.dumps({"irrigated": dict(IRRIGATED_VEC), "rainfed": bad}) + "\n")
        with pytest.raises(SystemExit, match="parameter 'mad' is non-finite"):
            bld.load_vectors(path)

    def test_infinite_value_raises(self, tmp_path):
        bad = dict(IRRIGATED_VEC)
        bad["aw"] = float("inf")
        path = tmp_path / "inf.json"
        path.write_text(json.dumps({"irrigated": bad, "rainfed": dict(RAINFED_VEC)}) + "\n")
        with pytest.raises(SystemExit, match="parameter 'aw' is non-finite"):
            bld.load_vectors(path)

    def test_non_mapping_class_raises(self, tmp_path):
        path = tmp_path / "notdict.json"
        path.write_text(json.dumps({"irrigated": [1, 2], "rainfed": dict(RAINFED_VEC)}) + "\n")
        with pytest.raises(SystemExit, match="is not a .*mapping"):
            bld.load_vectors(path)


# --------------------------------------------------------------------------- #
# Container coverage: the silent-drop guard
# --------------------------------------------------------------------------- #
class TestContainerCoverage:
    def test_full_coverage_audit(self, tmp_path):
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        truth = _truth_csv(tmp_path, _truth_rows(records))
        container = _stub_container(
            tmp_path, "e7cal.swim", [sid for sid, _ in records] + ["OTHER_001"]
        )
        audit = bld.check_container_coverage(container, truth, assignments)
        assert audit["n_container_field_uids"] == 111
        assert audit["n_eval_fids"] == 110
        assert audit["eval_fids_all_covered"] is True
        # A container field outside the truth roster is never evaluated, but is reported.
        assert audit["container_uids_not_in_mapping"] == ["OTHER_001"]

    def test_uncovered_truth_field_raises(self, tmp_path):
        """The real hazard: ``_resolve_params`` drops uncovered fields and shrinks n silently.

        ``evaluate_applied_water.py`` builds params as
        ``{fid: vec[fid] for fid in fids if fid in vec}``, so a container field present in
        the truth roster but missing from the mapping is never simulated and never
        reported -- the summary metrics just quietly rest on fewer fields. This check is
        the only place that failure becomes visible.
        """
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        orphan = "SLV_999"
        truth = _truth_csv(tmp_path, _truth_rows(records) + [(orphan, "SLV_meter")])
        container = _stub_container(tmp_path, "gap.swim", [sid for sid, _ in records] + [orphan])
        assert orphan not in assignments
        with pytest.raises(SystemExit, match="SILENTLY DROPPED"):
            bld.check_container_coverage(container, truth, assignments)

    def test_partial_mapping_is_caught(self, tmp_path):
        """Dropping the whole control arm from the mapping must not pass the audit."""
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        crippled = {sid: cls for sid, cls in assignments.items() if cls != bld.RAINFED_CLASS}
        truth = _truth_csv(tmp_path, _truth_rows(records))
        container = _stub_container(tmp_path, "crippled.swim", [sid for sid, _ in records])
        with pytest.raises(SystemExit, match="10 container field"):
            bld.check_container_coverage(container, truth, crippled)

    def test_missing_container_raises(self, tmp_path):
        records = _canonical_records(n_slv=1, n_espa=1, n_ctl=1)
        _, assignments, _ = _cohort(tmp_path, records, allow_unexpected=True)
        truth = _truth_csv(tmp_path, _truth_rows(records))
        with pytest.raises(SystemExit, match="container not found"):
            bld.check_container_coverage(tmp_path / "ghost.swim", truth, assignments)


# --------------------------------------------------------------------------- #
# --verify-keys pre-flight against a pooled run's per_field_year.csv
# --------------------------------------------------------------------------- #
class TestVerifyKeys:
    def _per_field_year(self, tmp_path, sites, years=(2020, 2021, 2022), name="per_field_year.csv"):
        rows = [(sid, y) for sid in sites for y in years]
        df = pd.DataFrame(rows, columns=["site_id", "year"])
        # A metered column is present here on purpose: verify_keys must not need it.
        df["metered_depth_mm"] = np.arange(len(df), dtype=float)
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path

    def test_reports_full_coverage(self, tmp_path):
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        path = self._per_field_year(tmp_path, [sid for sid, _ in records])
        out = bld.verify_keys(path, assignments)
        assert out["all_sites_covered"] is True
        assert out["sites_missing_from_mapping"] == []
        assert out["n_sites"] == 110
        assert out["n_rows"] == 330
        assert out["n_site_year_keys"] == 330
        assert out["mapping_sites_absent_from_file"] == []

    def test_reports_the_gap(self, tmp_path):
        """An unmapped site in the pooled run is exactly the site the new run would lose."""
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        path = self._per_field_year(
            tmp_path, [sid for sid, _ in records] + ["ESPA_999"], name="gap.csv"
        )
        out = bld.verify_keys(path, assignments)
        assert out["all_sites_covered"] is False
        assert out["sites_missing_from_mapping"] == ["ESPA_999"]
        assert out["n_sites"] == 111

    def test_reports_mapped_sites_absent_from_the_pooled_run(self, tmp_path):
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        subset = [sid for sid, _ in records if bld._prefix(sid) != bld.CONTROL_PREFIX]
        path = self._per_field_year(tmp_path, subset, name="subset.csv")
        out = bld.verify_keys(path, assignments)
        assert out["all_sites_covered"] is True
        assert len(out["mapping_sites_absent_from_file"]) == 10

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="--verify-keys file not found"):
            bld.verify_keys(tmp_path / "ghost.csv", {})


# --------------------------------------------------------------------------- #
# Truth cross-check: provenance only, never a metered value
# --------------------------------------------------------------------------- #
class TestCrossCheckTruth:
    def test_never_reads_a_metered_value(self, tmp_path):
        """The cross-check succeeds on a truth table that has no metered column at all.

        The fixture is built with exactly ``site_id`` and ``source``; if the builder ever
        touched ``metered_depth_mm`` the pandas ``usecols`` read would raise. That is the
        structural guarantee that meter truth stays withheld until scoring, so class
        assignment cannot be tuned to the answer.
        """
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        truth = _truth_csv(tmp_path, _truth_rows(records))
        assert pd.read_csv(truth, nrows=0).columns.tolist() == ["site_id", "source"]

        out = bld.cross_check_truth(truth, assignments)
        assert out["columns_read"] == ["site_id", "source"]
        assert out["metered_columns_read"] == []
        assert out["n_truth_sites"] == 110
        assert out["n_agree"] == 110
        assert out["n_disagree"] == 0
        assert out["disagreements"] == []
        assert out["sites_in_truth_not_in_mapping"] == []
        assert out["sites_in_mapping_not_in_truth"] == []
        assert out["sites_with_mixed_source_labels"] == {}
        assert len(out["truth_csv_sha256"]) == 64

    def test_disagreement_is_reported_not_silently_resolved(self, tmp_path):
        """A field the meter roster calls a control but the prefix rule calls irrigated."""
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        rows = [
            (sid, bld.TRUTH_CONTROL_SOURCE if sid == "ESPA_000" else src)
            for sid, src in _truth_rows(records)
        ]
        out = bld.cross_check_truth(_truth_csv(tmp_path, rows, name="disagree.csv"), assignments)
        assert out["n_disagree"] == 1
        assert out["disagreements"] == [
            {
                "site_id": "ESPA_000",
                "mapping_class": bld.IRRIGATED_CLASS,
                "truth_source_class": bld.RAINFED_CLASS,
            }
        ]
        assert out["n_agree"] == 109

    def test_site_in_truth_but_not_in_mapping_is_reported(self, tmp_path):
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        rows = _truth_rows(records) + [("SLV_999", "SLV_meter")]
        out = bld.cross_check_truth(_truth_csv(tmp_path, rows, name="extra_truth.csv"), assignments)
        assert out["sites_in_truth_not_in_mapping"] == ["SLV_999"]
        assert out["n_agree"] == 110

    def test_mixed_source_labels_are_reported(self, tmp_path):
        """One site labeled both metered and control is a roster defect, not a class."""
        records = _canonical_records()
        _, assignments, _ = _cohort(tmp_path, records)
        rows = _truth_rows(records) + [("ESPActl_000", "ESPA_meter")]
        out = bld.cross_check_truth(_truth_csv(tmp_path, rows, name="mixed.csv"), assignments)
        assert set(out["sites_with_mixed_source_labels"]) == {"ESPActl_000"}
        assert out["sites_with_mixed_source_labels"]["ESPActl_000"] == [
            "ESPA_meter",
            bld.TRUTH_CONTROL_SOURCE,
        ]

    def test_missing_truth_roster_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="Truth roster not found"):
            bld.cross_check_truth(tmp_path / "ghost.csv", {})


# --------------------------------------------------------------------------- #
# Downstream forward-run contract
# --------------------------------------------------------------------------- #
class TestForwardRunContract:
    def test_container_and_label_constants(self):
        """The E4 run must target the calibrated container under the agreed label."""
        assert bld.E4_FORWARD_LABEL == "transfer_run22_by_irrigation"
        assert bld.E4_FORWARD_CONTAINER.endswith("_e7cal.swim")
        assert bld.E4_FORWARD_CONTAINER == bld.DEFAULT_CONTAINER

    def test_default_paths_are_the_frozen_artifacts(self):
        assert Path(bld.DEFAULT_VECTORS).name == "e2_run22_transfer_vectors_by_irrigation.json"
        assert Path(bld.DEFAULT_IRRMAPPER_CSV).name == "espa_control_irrmapper.csv"
        assert Path(bld.DEFAULT_TRUTH_CSV).name == "metered_truth.csv"
        assert Path(bld.DEFAULT_FIELDS_SHP).name == "applied_water_fields.shp"

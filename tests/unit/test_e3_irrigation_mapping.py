"""Unit tests for the Example 6 (Experiment 3) per-site irrigation-stratified mapping.

``examples/6_Flux_International/transfer/build_e3_irrigation_mapping.py`` expands the two
frozen Run 22 class vectors into ``{sid: {param: value}}`` over the 66-site E3 publication
cohort. Three properties make it worth testing hard:

1. **Stage 1 is not persisted.** The classifier's site-level ``equipped`` flag is a local
   variable in ``IrrigationCalculator`` (``calculator.py`` ~L1203-1229,
   ``equipped = n_win > 0 and (n_sub * 3 > n_win)``) and is written to neither the
   container nor any CSV/JSON. The builder recovers it as "at least one irrigated year in
   ``derived/dynamics/irr_data``". That recovery is sound in one direction only -- stage 2
   can never activate irrigation on a site that is not equipped -- so the tests pin both
   the implication that holds and the gap that does not.
2. **Reconciliation is a stop condition, not a warning.** The handoff freezes 75/14/175 at
   container scope, 66/13/163 at cohort scope, and ``container - cohort == {ES-LJu}``. A
   violated expectation must raise; downgrading it to a warning would let a silently
   different cohort reach the published comparator.
3. **Vector expansion must be byte-exact.** Every site carries all eight parameter
   families identical to the frozen class vector it was assigned, or the "transfer" is no
   longer a transfer of the frozen Run 22 posterior.
"""

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "transfer"


def _load_builder():
    """Import the builder by path (``examples/`` is not an importable package).

    The transfer dir must be on ``sys.path`` because the builder imports two sibling
    modules (``build_ex5_cropland_params``, ``build_ex5_irrigation_stratified_params``).
    """
    if str(TRANSFER_DIR) not in sys.path:
        sys.path.insert(0, str(TRANSFER_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_e3_irrigation_mapping",
        TRANSFER_DIR / "build_e3_irrigation_mapping.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bld = _load_builder()
FAMILIES = list(bld.PARAM_FAMILIES)

# Deliberately disjoint class vectors so a mis-assignment cannot go unnoticed.
IRRIGATED_VEC = {fam: 1.0 + i for i, fam in enumerate(FAMILIES)}
RAINFED_VEC = {fam: 100.0 + i for i, fam in enumerate(FAMILIES)}

# The canonical synthetic reconciliation, built to hit the frozen expectations exactly:
#   container 75 sites / 14 equipped / 175 irrigated site-years
#   cohort     66 sites / 13 equipped / 163 irrigated site-years
#   container - cohort == {ES-LJu}, carrying the residual 12 site-years
COHORT_SIDS = [f"C{i:02d}" for i in range(66)]
COHORT_EQUIPPED = COHORT_SIDS[:13]  # 12 x 13 years + 1 x 7 years = 163
CONTAINER_ONLY_RAINFED = [f"X{i}" for i in range(8)]
CONTAINER_ONLY_EQUIPPED = "ES-LJu"
ES_LJU_YEARS = 12


# --------------------------------------------------------------------------- #
# Fixtures: irr_data zarr stub, cohort shapefile, frozen vector artifact
# --------------------------------------------------------------------------- #
def _canonical_irr_years():
    """``{fid: n_irrigated_years}`` reproducing the frozen 75/14/175 + 66/13/163 counts."""
    years = dict.fromkeys(COHORT_SIDS, 0)
    for sid in COHORT_EQUIPPED[:12]:
        years[sid] = 13
    years[COHORT_EQUIPPED[12]] = 7
    years.update(dict.fromkeys(CONTAINER_ONLY_RAINFED, 0))
    years[CONTAINER_ONLY_EQUIPPED] = ES_LJU_YEARS
    return years


def _irr_blob(n_irrigated, n_total=20, extra_keys=True):
    """One field's ``irr_data`` JSON blob: per-year dicts plus the fallow_years list."""
    doc = {}
    fallow = []
    for i in range(n_total):
        year = 2000 + i
        irrigated = 1 if i < n_irrigated else 0
        if not irrigated:
            fallow.append(year)
        doc[str(year)] = {
            "f_irr": 1.0 if irrigated else 0.0,
            "irr_doys": [150, 151] if irrigated else [],
            "irrigated": irrigated,
        }
    if extra_keys:
        doc["fallow_years"] = fallow
    return json.dumps(doc)


def _stub_container(tmp_path, name, irr_years, blobs=None, irr_props=None, uid_extra=0):
    """Minimal .swim-shaped zarr store with geometry/uid + derived/dynamics/irr_data.

    Mirrors ``SwimContainer.create``: both are VariableLengthUTF8 arrays (a plain
    object-dtype dataset fails on zarr 3.1.5). ``irr_props`` optionally writes a
    contradictory ``properties/irrigation/irr`` array, which the E3 path must ignore.
    """
    zarr = pytest.importorskip("zarr")
    import numpy as np
    from zarr.core.dtype import VariableLengthUTF8

    fids = list(irr_years)
    path = tmp_path / name
    root = zarr.open(str(path), mode="w")
    uid = root.create_array(
        "geometry/uid", shape=(len(fids) + uid_extra,), dtype=VariableLengthUTF8()
    )
    uid[:] = fids + [f"PAD{i}" for i in range(uid_extra)]
    if blobs is None:
        blobs = [_irr_blob(irr_years[f]) for f in fids]
    if blobs is not False:
        arr = root.create_array(
            "derived/dynamics/irr_data", shape=(len(blobs),), dtype=VariableLengthUTF8()
        )
        arr[:] = list(blobs)
    if irr_props is not None:
        root.create_array("properties/irrigation/irr", data=np.asarray(irr_props, dtype=float))
    return path


def _cohort_shp(tmp_path, sids=None, name="cohort.shp", uid_col="sid"):
    """Write a synthetic cohort shapefile carrying only the uid column and geometry."""
    sids = COHORT_SIDS if sids is None else sids
    gdf = gpd.GeoDataFrame(
        {
            uid_col: [str(s) for s in sids],
            "geometry": [Point(i, i) for i in range(len(sids))],
        },
        crs="EPSG:4326",
    )
    path = tmp_path / name
    gdf.to_file(path, driver="ESRI Shapefile", engine="fiona")
    return path


def _vectors_json(tmp_path, payload=None, name="vectors.json"):
    if payload is None:
        payload = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Stage-1 rule and its recovery from the persisted annual record
# --------------------------------------------------------------------------- #
def _stage1_equipped(n_sub, n_win):
    """The real stage-1 rule, transcribed from calculator.py ~L1229."""
    return n_win > 0 and (n_sub * 3 > n_win)


class TestStage1Rule:
    def test_one_third_floor_is_strictly_greater_than(self):
        """``n_sub * 3 > n_win``: exactly one third is NOT equipped."""
        assert _stage1_equipped(5, 15) is False  # 15 == 15, not >
        assert _stage1_equipped(6, 15) is True
        assert _stage1_equipped(4, 13) is False  # 12 < 13
        assert _stage1_equipped(5, 13) is True  # 15 > 13
        # The US-MH2 intermittent-irrigation case the floor is tuned to admit.
        assert _stage1_equipped(6, 13) is True

    def test_lone_anomalous_year_cannot_flip_a_dryland_site(self):
        """A single drought year touches ~2 of ~13 windows, far under the floor."""
        assert _stage1_equipped(2, 13) is False

    def test_no_windows_means_not_equipped(self):
        """n_win == 0 (no capacity/precip record) short-circuits before the ratio test."""
        assert _stage1_equipped(0, 0) is False
        assert _stage1_equipped(5, 0) is False

    def test_rule_text_is_recorded_in_the_metadata_constant(self):
        assert "equipped" in bld.STAGE1_RULE
        assert "annual_2yr" in bld.STAGE1_RULE
        assert "one third" in bld.STAGE1_RULE
        assert "annual_subsidy_ratio" in bld.STAGE1_RULE
        assert "1203-1229" in bld.STAGE1_RULE


class TestStage1Recovery:
    def test_recovery_counts_irrigated_years_only(self, tmp_path):
        irr_years = {"A": 0, "B": 1, "C": 7}
        path = _stub_container(tmp_path, "c.swim", irr_years)
        assert bld.read_irrigation_years(path) == irr_years

    def test_fallow_years_list_and_non_dict_values_are_skipped(self, tmp_path):
        """``irr_data`` mixes per-year dicts with a ``fallow_years`` list at the top level."""
        doc = json.loads(_irr_blob(3))
        assert isinstance(doc["fallow_years"], list)
        doc["some_scalar"] = 5
        path = _stub_container(tmp_path, "mixed.swim", {"A": 3}, blobs=[json.dumps(doc)])
        assert bld.read_irrigation_years(path) == {"A": 3}

    def test_recovery_reproduces_stage1_when_every_equipped_site_activates(self):
        """The implication that holds: stage 2 gates on ``equipped``, so any irrigated year
        proves the site was equipped.

        In ``calculator.py`` the annual/annual_2yr stage-2 branch is ``if equipped: ... else:
        irrigated = False``, so ``n_irrigated_years > 0`` implies ``equipped is True``. The
        recovery is therefore exact precisely when no equipped site has zero activated
        years -- the condition the builder's docstring records as independently verified
        (14 equipped, zero equipped-but-never-activated, zero activated-but-not-equipped).
        """
        # (n_sub, n_win, n_activated_years) triples for a verified-clean container.
        sites = {"eq_active": (6, 13, 5), "dry": (2, 13, 0)}
        for _sid, (n_sub, n_win, n_act) in sites.items():
            equipped = _stage1_equipped(n_sub, n_win)
            recovered = n_act > 0
            assert recovered == equipped
            if n_act > 0:
                assert equipped, "stage 2 cannot activate a site that is not equipped"

    def test_equipped_but_never_activated_is_recovered_as_rainfed(self, tmp_path):
        """The documented gap: the recovery has no way to see such a site.

        This is not a bug in the builder -- it is the reason its docstring carries an
        independent verification that the container has zero such sites. The test makes the
        assumption visible so a future container that breaks it is caught here.
        """
        equipped = _stage1_equipped(6, 13)
        assert equipped is True  # stage 1 says equipped
        cohort = ["A", "B"]
        irr_years = {"A": 0, "B": 4}  # A is the equipped-but-never-activated site
        mapping, assignments, counts = bld.build_mapping(
            {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}, irr_years, cohort
        )
        assert assignments["A"]["irr_class"] == "rainfed"
        assert assignments["A"]["equipped"] is False
        assert mapping["A"] == RAINFED_VEC

    def test_properties_irrigation_irr_is_never_consulted(self, tmp_path):
        """``properties/irrigation/irr`` is the CONUS use_mask path and must be ignored.

        The stub writes an ``irr`` array that contradicts ``irr_data`` on every site; if the
        builder read it the classes would invert.
        """
        irr_years = {"A": 0, "B": 5}
        path = _stub_container(tmp_path, "conus.swim", irr_years, irr_props=[1.0, 0.0])
        assert bld.read_irrigation_years(path) == irr_years
        assert "properties/irrigation/irr" in bld.STAGE1_RECOVERY

    def test_missing_irr_data_raises(self, tmp_path):
        path = _stub_container(tmp_path, "noirr.swim", {"A": 0}, blobs=False)
        with pytest.raises(KeyError, match="no derived/dynamics/irr_data"):
            bld.read_irrigation_years(path)

    def test_length_mismatch_raises(self, tmp_path):
        path = _stub_container(tmp_path, "short.swim", {"A": 1, "B": 1}, uid_extra=1)
        with pytest.raises(ValueError, match=r"irr_data length 2 != 3 field uids"):
            bld.read_irrigation_years(path)

    def test_unparseable_blob_raises(self, tmp_path):
        path = _stub_container(tmp_path, "bad.swim", {"A": 1}, blobs=["{not json"])
        with pytest.raises(ValueError, match="Unparseable irr_data for A"):
            bld.read_irrigation_years(path)

    def test_missing_container_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="E3 container not found"):
            bld.read_irrigation_years(tmp_path / "ghost.swim")

    def test_recovery_note_records_that_stage1_is_not_persisted(self):
        assert "NOT persisted" in bld.STAGE1_RECOVERY
        assert "local variable" in bld.STAGE1_RECOVERY
        assert "derived/dynamics/irr_data" in bld.STAGE1_RECOVERY
        assert "exactly 14 sites" in bld.STAGE1_RECOVERY
        assert "zero equipped-but-never-activated" in bld.STAGE1_RECOVERY

    def test_stage2_note_says_parameters_are_fixed_by_site(self):
        assert "does NOT change a site's parameter vector" in bld.STAGE2_NOTE
        assert "Parameters are fixed by site" in bld.STAGE2_NOTE


# --------------------------------------------------------------------------- #
# Reconciliation: a stop condition, not a warning
# --------------------------------------------------------------------------- #
class TestReconciliation:
    def test_frozen_expectations(self):
        assert bld.EXPECTED_CONTAINER_SITES == 75
        assert bld.EXPECTED_CONTAINER_EQUIPPED == 14
        assert bld.EXPECTED_CONTAINER_IRR_YEARS == 175
        assert bld.EXPECTED_COHORT_SITES == 66
        assert bld.EXPECTED_COHORT_EQUIPPED == 13
        assert bld.EXPECTED_COHORT_IRR_YEARS == 163
        assert bld.EXPECTED_CONTAINER_ONLY == {"ES-LJu"}
        # The single container-only equipped site accounts for the whole 175-163 residual.
        assert bld.EXPECTED_CONTAINER_IRR_YEARS - bld.EXPECTED_COHORT_IRR_YEARS == ES_LJU_YEARS
        assert bld.EXPECTED_CONTAINER_EQUIPPED - bld.EXPECTED_COHORT_EQUIPPED == 1

    def test_canonical_counts_reconcile(self):
        report = bld.reconcile(_canonical_irr_years(), COHORT_SIDS)
        assert report["reconciles"] is True
        assert report["failures"] == []
        assert report["container_sites"] == 75
        assert report["container_equipped_sites"] == 14
        assert report["container_irrigated_site_years"] == 175
        assert report["cohort_sites"] == 66
        assert report["cohort_equipped_sites"] == 13
        assert report["cohort_irrigated_site_years"] == 163
        assert report["container_equipped_not_in_cohort"] == ["ES-LJu"]
        assert report["container_equipped_not_in_cohort_years"] == {"ES-LJu": ES_LJU_YEARS}
        assert report["expected"]["container_equipped_not_in_cohort"] == ["ES-LJu"]

    def test_wrong_equipped_count_raises(self):
        years = _canonical_irr_years()
        years[COHORT_SIDS[20]] = 1  # a 14th cohort-equipped site
        with pytest.raises(ValueError, match="cohort ever-irrigated sites: got 14, expected 13"):
            bld.reconcile(years, COHORT_SIDS)

    def test_wrong_site_year_total_raises(self):
        years = _canonical_irr_years()
        years[COHORT_EQUIPPED[0]] += 1
        with pytest.raises(ValueError, match="cohort irrigated site-years: got 164, expected 163"):
            bld.reconcile(years, COHORT_SIDS)

    def test_wrong_container_site_count_raises(self):
        years = _canonical_irr_years()
        years["EXTRA"] = 0
        with pytest.raises(ValueError, match="container site count: got 76, expected 75"):
            bld.reconcile(years, COHORT_SIDS)

    def test_wrong_cohort_size_raises(self):
        with pytest.raises(ValueError, match="cohort site count: got 65, expected 66"):
            bld.reconcile(_canonical_irr_years(), COHORT_SIDS[:65])

    def test_container_minus_cohort_set_is_pinned(self):
        """A different container-only equipped site fails even at identical counts."""
        years = _canonical_irr_years()
        years["X0"] = years.pop(CONTAINER_ONLY_EQUIPPED)  # same counts, different site
        years[CONTAINER_ONLY_EQUIPPED] = 0
        with pytest.raises(
            ValueError, match=r"container-minus-cohort equipped sites: got \['X0'\]"
        ):
            bld.reconcile(years, COHORT_SIDS)

    def test_failure_raises_rather_than_warning(self):
        """Every reconciliation failure is fatal by default, and names the stop condition."""
        years = _canonical_irr_years()
        years[COHORT_SIDS[20]] = 1
        with pytest.raises(ValueError) as exc:
            bld.reconcile(years, COHORT_SIDS)
        msg = str(exc.value)
        assert "reconciliation failed" in msg
        assert "Stop condition" in msg
        assert "does not reconcile independently of the 75-site container" in msg
        assert "--allow-unexpected" in msg

    def test_allow_unexpected_downgrades_to_a_recorded_failure(self):
        years = _canonical_irr_years()
        years[COHORT_SIDS[20]] = 1
        report = bld.reconcile(years, COHORT_SIDS, allow_unexpected=True)
        assert report["reconciles"] is False
        # A cohort site is also a container site, so both scopes drift together.
        assert len(report["failures"]) == 4
        assert any("cohort ever-irrigated sites" in f for f in report["failures"])
        assert any("container ever-irrigated sites" in f for f in report["failures"])

    def test_cohort_site_absent_from_container_always_raises(self):
        """Not overridable: a site with no persisted record cannot be assigned a class."""
        years = _canonical_irr_years()
        del years[COHORT_SIDS[3]]
        for allow in (False, True):
            with pytest.raises(ValueError, match=r"1 cohort site\(s\) absent from the container"):
                bld.reconcile(years, COHORT_SIDS, allow_unexpected=allow)


# --------------------------------------------------------------------------- #
# Vector expansion
# --------------------------------------------------------------------------- #
class TestVectorExpansion:
    def test_eight_families_in_canonical_order(self):
        assert FAMILIES == [
            "aw",
            "ndvi_k",
            "ndvi_0",
            "mad",
            "ks_alpha",
            "kr_alpha",
            "swe_alpha",
            "swe_beta",
        ]

    def test_every_site_carries_its_class_vector_byte_equal(self):
        vectors = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
        years = _canonical_irr_years()
        mapping, assignments, counts = bld.build_mapping(vectors, years, COHORT_SIDS)

        assert len(mapping) == 66
        assert counts == {"irrigated": 13, "rainfed": 53}
        assert sum(counts.values()) == len(COHORT_SIDS)
        for sid, vec in mapping.items():
            cls = assignments[sid]["irr_class"]
            assert list(vec) == FAMILIES, sid
            # Byte-equal to the frozen class vector, not merely numerically close.
            assert json.dumps(vec, sort_keys=True) == json.dumps(vectors[cls], sort_keys=True)
            assert bld._vector_sha256(vec) == bld._vector_sha256(vectors[cls])
            assert assignments[sid]["vector_sha256"] == bld._vector_sha256(vectors[cls])

    def test_assignment_record_fields(self):
        vectors = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
        _, assignments, _ = bld.build_mapping(vectors, _canonical_irr_years(), COHORT_SIDS)
        rec = assignments[COHORT_EQUIPPED[0]]
        assert set(rec) == {"irr_class", "equipped", "n_irrigated_years", "vector_sha256"}
        assert rec == {
            "irr_class": "irrigated",
            "equipped": True,
            "n_irrigated_years": 13,
            "vector_sha256": bld._vector_sha256(IRRIGATED_VEC),
        }
        dry = assignments[COHORT_SIDS[40]]
        assert dry["equipped"] is False and dry["n_irrigated_years"] == 0

    def test_class_partition_is_exact(self):
        vectors = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
        _, assignments, counts = bld.build_mapping(vectors, _canonical_irr_years(), COHORT_SIDS)
        irr = {s for s, a in assignments.items() if a["irr_class"] == "irrigated"}
        rf = {s for s, a in assignments.items() if a["irr_class"] == "rainfed"}
        assert irr == set(COHORT_EQUIPPED)
        assert not irr & rf
        assert irr | rf == set(COHORT_SIDS)
        assert (len(irr), len(rf)) == (counts["irrigated"], counts["rainfed"])

    def test_mapping_entries_are_copies_not_aliases(self):
        """A shared dict would let one site's mutation rewrite the whole class."""
        vectors = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
        mapping, _, _ = bld.build_mapping(vectors, _canonical_irr_years(), COHORT_SIDS)
        a, b = COHORT_EQUIPPED[0], COHORT_EQUIPPED[1]
        mapping[a]["mad"] = -999.0
        assert mapping[b]["mad"] == IRRIGATED_VEC["mad"]
        assert vectors["irrigated"]["mad"] == IRRIGATED_VEC["mad"]

    def test_degenerate_single_class_raises(self):
        vectors = {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(RAINFED_VEC)}
        with pytest.raises(ValueError, match="Degenerate class assignment"):
            bld.build_mapping(vectors, {"A": 0, "B": 0}, ["A", "B"])
        with pytest.raises(ValueError, match="identical to a pooled vector"):
            bld.build_mapping(vectors, {"A": 2, "B": 3}, ["A", "B"])

    def test_class_to_vector_policy(self):
        assert bld.CLASS_TO_VECTOR == {True: "irrigated", False: "rainfed"}
        assert bld.CLASSES == ("irrigated", "rainfed")


# --------------------------------------------------------------------------- #
# Frozen two-vector artifact validation
# --------------------------------------------------------------------------- #
class TestLoadClassVectors:
    def test_loads_and_normalizes_family_order(self, tmp_path):
        shuffled = {
            "irrigated": dict(reversed(list(IRRIGATED_VEC.items()))),
            "rainfed": {k: int(v) for k, v in RAINFED_VEC.items()},
        }
        vectors = bld.load_class_vectors(_vectors_json(tmp_path, shuffled))
        assert set(vectors) == {"irrigated", "rainfed"}
        for vec in vectors.values():
            assert list(vec) == FAMILIES
            assert all(isinstance(v, float) for v in vec.values())

    def test_missing_file_refuses_to_invent_a_placeholder(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="will not invent a"):
            bld.load_class_vectors(tmp_path / "ghost.json")

    def test_missing_class_raises(self, tmp_path):
        path = _vectors_json(tmp_path, {"irrigated": dict(IRRIGATED_VEC)}, name="one.json")
        with pytest.raises(ValueError, match=r"missing irrigation class\(es\) \['rainfed'\]"):
            bld.load_class_vectors(path)

    def test_extra_top_level_key_raises(self, tmp_path):
        payload = {
            "irrigated": dict(IRRIGATED_VEC),
            "rainfed": dict(RAINFED_VEC),
            "pooled": dict(IRRIGATED_VEC),
        }
        path = _vectors_json(tmp_path, payload, name="extra.json")
        with pytest.raises(ValueError, match=r"unexpected top-level key\(s\) \['pooled'\]"):
            bld.load_class_vectors(path)

    def test_non_dict_class_raises(self, tmp_path):
        path = _vectors_json(
            tmp_path, {"irrigated": [1, 2], "rainfed": dict(RAINFED_VEC)}, name="list.json"
        )
        with pytest.raises(ValueError, match=r"class 'irrigated' is not a"):
            bld.load_class_vectors(path)

    def test_missing_parameter_raises(self, tmp_path):
        incomplete = {k: v for k, v in RAINFED_VEC.items() if k != "swe_beta"}
        path = _vectors_json(
            tmp_path, {"irrigated": dict(IRRIGATED_VEC), "rainfed": incomplete}, name="miss.json"
        )
        with pytest.raises(ValueError, match=r"missing parameter\(s\) \['swe_beta'\]"):
            bld.load_class_vectors(path)

    def test_unknown_parameter_raises(self, tmp_path):
        extra = dict(IRRIGATED_VEC)
        extra["p_stress"] = 0.5
        path = _vectors_json(
            tmp_path, {"irrigated": extra, "rainfed": dict(RAINFED_VEC)}, name="unk.json"
        )
        with pytest.raises(ValueError, match=r"unknown parameter\(s\) \['p_stress'\]"):
            bld.load_class_vectors(path)

    def test_identical_class_vectors_raise(self, tmp_path):
        path = _vectors_json(
            tmp_path,
            {"irrigated": dict(IRRIGATED_VEC), "rainfed": dict(IRRIGATED_VEC)},
            "same.json",
        )
        with pytest.raises(ValueError, match="stratification would be a no-op"):
            bld.load_class_vectors(path)


# --------------------------------------------------------------------------- #
# Cohort roster
# --------------------------------------------------------------------------- #
class TestReadCohort:
    def test_reads_uid_column_in_file_order(self, tmp_path):
        path = _cohort_shp(tmp_path)
        assert bld.read_cohort(path) == COHORT_SIDS
        assert bld.UID_COL == "sid"

    def test_duplicate_uid_raises(self, tmp_path):
        path = _cohort_shp(tmp_path, ["A", "B", "A"], name="dupe.shp")
        with pytest.raises(ValueError, match=r"Duplicate sid value\(s\).*\['A'\]"):
            bld.read_cohort(path)

    def test_missing_uid_column_raises(self, tmp_path):
        path = _cohort_shp(tmp_path, ["A", "B"], name="othercol.shp", uid_col="fid")
        with pytest.raises(KeyError, match="has no 'sid' column"):
            bld.read_cohort(path)

    def test_missing_shapefile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="E3 cohort shapefile not found"):
            bld.read_cohort(tmp_path / "ghost.shp")


# --------------------------------------------------------------------------- #
# End-to-end artifact + metadata contract
# --------------------------------------------------------------------------- #
class TestArtifacts:
    def _run(self, tmp_path, monkeypatch, extra_argv=()):
        container = _stub_container(tmp_path, "e3.swim", _canonical_irr_years())
        shp = _cohort_shp(tmp_path)
        vectors = _vectors_json(tmp_path)
        out_dir = tmp_path / "final"
        argv = [
            "build_e3_irrigation_mapping.py",
            "--vectors",
            str(vectors),
            "--container",
            str(container),
            "--shapefile",
            str(shp),
            "--out-dir",
            str(out_dir),
            *extra_argv,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        bld.main()
        mapping = json.loads((out_dir / "e3_irrigation_stratified_param_mapping.json").read_text())
        meta = json.loads(
            (out_dir / "e3_irrigation_stratified_param_mapping_metadata.json").read_text()
        )
        return mapping, meta, vectors

    def test_mapping_covers_the_whole_cohort_with_all_eight_families(
        self, tmp_path, monkeypatch, capsys
    ):
        mapping, meta, _ = self._run(tmp_path, monkeypatch)
        capsys.readouterr()
        assert set(mapping) == set(COHORT_SIDS)
        assert len(mapping) == bld.EXPECTED_COHORT_SITES
        for sid, vec in mapping.items():
            assert list(vec) == FAMILIES, sid
        assert mapping[COHORT_EQUIPPED[0]] == IRRIGATED_VEC
        assert mapping[COHORT_SIDS[40]] == RAINFED_VEC
        assert meta["class_counts"] == {"irrigated": 13, "rainfed": 53}

    def test_metadata_provenance_fields(self, tmp_path, monkeypatch, capsys):
        mapping, meta, vectors_path = self._run(tmp_path, monkeypatch)
        capsys.readouterr()

        assert meta["reconciliation"]["reconciles"] is True
        assert meta["reconciliation"]["failures"] == []
        assert meta["reconciliation"]["container_equipped_not_in_cohort"] == ["ES-LJu"]
        assert meta["allow_unexpected"] is False

        assert meta["stage1_not_persisted_recovery"] == bld.STAGE1_RECOVERY
        assert meta["stage1_rule"] == bld.STAGE1_RULE
        assert meta["stage2_note"] == bld.STAGE2_NOTE

        assert isinstance(meta["git_sha"], str) or meta["git_sha"] is None
        assert isinstance(meta["worktree_dirty"], bool) or meta["worktree_dirty"] is None
        assert meta["source_vectors_path"] == str(vectors_path)
        assert len(meta["source_vectors_sha256"]) == 64
        assert meta["class_vectors"] == {"irrigated": IRRIGATED_VEC, "rainfed": RAINFED_VEC}
        assert meta["class_vector_sha256"] == {
            "irrigated": bld._vector_sha256(IRRIGATED_VEC),
            "rainfed": bld._vector_sha256(RAINFED_VEC),
        }
        assert meta["param_names"] == FAMILIES
        assert meta["container_irr_data_path"] == "derived/dynamics/irr_data"
        assert meta["container_irr_data_keys"] == ["f_irr", "irr_doys", "irrigated"]
        assert meta["cohort_uid_col"] == "sid"
        assert meta["cohort_size"] == 66
        assert meta["model_structure"] == "canonical Run 22 single-mad coupling"
        assert meta["stress_depletion_fraction"] is None
        assert meta["e5split_materials_used"] is False
        assert "validation only" in meta["flux_role"]
        assert "validation only" in meta["meter_role"]
        assert meta["date_generated_utc"].endswith("Z")

        # sites_by_class must partition the cohort and agree with the mapping.
        by_class = meta["sites_by_class"]
        assert set(by_class) == {"irrigated", "rainfed"}
        assert sorted(by_class["irrigated"]) == sorted(COHORT_EQUIPPED)
        assert set(by_class["irrigated"]) | set(by_class["rainfed"]) == set(mapping)
        assert not set(by_class["irrigated"]) & set(by_class["rainfed"])
        for cls, sids in by_class.items():
            for sid in sids:
                assert meta["assignments"][sid]["irr_class"] == cls

    def test_metadata_points_at_the_consuming_flag(self, tmp_path, monkeypatch, capsys):
        _, meta, _ = self._run(tmp_path, monkeypatch)
        capsys.readouterr()
        assert "--params-by-site" in meta["purpose"]
        assert "ex5_transfer_strat" in meta["purpose"]
        assert meta["target_class_policy"] == (
            "equipped for irrigation -> irrigated vector; not equipped -> rainfed vector"
        )

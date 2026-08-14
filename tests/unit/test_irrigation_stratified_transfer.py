"""Unit tests for the irrigation-stratified Ex5 -> Ex6/Ex7 transfer vector builder.

Covers the validation surface the handoff (``paper/notes/
irrigation_stratified_transfer_handoff.md``) requires before any downstream flux
or meter truth is opened: source classification, median-of-site-medians within
class, malformed source posteriors, missing irrigation labels, and the frozen
nested JSON structure.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "transfer"


def _load_builder():
    """Import the builder by path (``examples/`` is not an importable package)."""
    if str(TRANSFER_DIR) not in sys.path:
        sys.path.insert(0, str(TRANSFER_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_ex5_irrigation_stratified_params",
        TRANSFER_DIR / "build_ex5_irrigation_stratified_params.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bld = _load_builder()
FAMILIES = list(bld.PARAM_FAMILIES)


# --------------------------------------------------------------------------- #
# Fixtures: synthetic .par.csv and per-site median tables
# --------------------------------------------------------------------------- #
def _par_csv(tmp_path, sites, n_real=5, values=None, base=True, dup_site=None):
    """Write a synthetic PEST++ .par.csv with one column per (family, site).

    ``values[fam][site]`` sets the constant posterior value for that cell so the
    per-site median is exactly that value. ``base`` adds the excluded 'base' row.
    ``dup_site`` adds a second, distinctly named column resolving to that same
    site token, which is what a genuine duplicate looks like in a real .par.csv.
    """
    cols, data = [], {}
    for fam in FAMILIES:
        for s in sites:
            col = f"pname:p_{fam}_{s}_:0_1"
            cols.append(col)
            v = 1.0 if values is None else values[fam][s]
            data[col] = [v] * n_real
        if dup_site:
            col = f"pname:p_{fam}_{dup_site}_:0_2"
            cols.append(col)
            data[col] = [1.0 if values is None else values[fam][dup_site]] * n_real
    df = pd.DataFrame(data, index=[f"r{i}" for i in range(n_real)])
    if base:
        # A wildly different 'base' row: if it were included the medians shift.
        df.loc["base"] = {c: 999.0 for c in cols}
    df.index.name = "real_name"
    path = tmp_path / "synthetic.3.par.csv"
    df.to_csv(path)
    return path


def _medians(sites, per_site):
    """Per-site median table: index=site, columns=families."""
    return pd.DataFrame(
        {fam: [per_site[s][fam] for s in sites] for fam in FAMILIES},
        index=pd.Index(list(sites), name="site"),
    )


def _stub_container(tmp_path, name, fids, irr=None):
    """Minimal .swim-shaped zarr store with geometry/uid and (optionally) irr.

    Mirrors ``SwimContainer.create``: uid is a VariableLengthUTF8 array, which is
    what ``field_uids`` is loaded from.
    """
    zarr = pytest.importorskip("zarr")
    from zarr.core.dtype import VariableLengthUTF8

    path = tmp_path / name
    root = zarr.open(str(path), mode="w")
    uid = root.create_array("geometry/uid", shape=(len(fids),), dtype=VariableLengthUTF8())
    uid[:] = list(fids)
    if irr is not None:
        root.create_array("properties/irrigation/irr", data=np.asarray(irr, dtype=float))
    return path


# --------------------------------------------------------------------------- #
# Source classification
# --------------------------------------------------------------------------- #
class TestSourceClassification:
    def test_threshold_is_strictly_greater_than(self):
        """irr > 0.5 is irrigated; exactly 0.5 is rainfed."""
        df = pd.DataFrame(
            {"irr": [0.0, 0.4999, 0.5, 0.5001, 1.0]},
            index=pd.Index([f"s{i}" for i in range(5)], name="fid"),
        )
        cls = np.where(df["irr"].to_numpy() > bld.IRR_THRESHOLD, "irrigated", "rainfed")
        assert list(cls) == ["rainfed", "rainfed", "rainfed", "irrigated", "irrigated"]

    def test_reads_container_irr_and_classifies(self, tmp_path):
        fids = ["US-Ne1", "Almond_High", "b_01"]
        path = _stub_container(tmp_path, "src.swim", fids, irr=[0.9, 0.2, 0.51])
        out = bld.read_source_irrigation(path)
        assert list(out.index) == fids
        assert out["irr_class"].tolist() == ["irrigated", "rainfed", "irrigated"]
        assert out.loc["Almond_High", "irr"] == pytest.approx(0.2)

    def test_missing_irr_array_raises(self, tmp_path):
        """An E3-style container has no properties/irrigation group at all."""
        path = _stub_container(tmp_path, "noirr.swim", ["a"], irr=None)
        with pytest.raises(KeyError, match="no properties/irrigation/irr"):
            bld.read_source_irrigation(path)

    def test_length_mismatch_raises(self, tmp_path):
        path = _stub_container(tmp_path, "bad.swim", ["a", "b"], irr=[0.9])
        with pytest.raises(ValueError, match="!= 2 field_uids"):
            bld.read_source_irrigation(path)

    def test_non_finite_irr_raises(self, tmp_path):
        path = _stub_container(tmp_path, "nan.swim", ["a", "b"], irr=[0.9, np.nan])
        with pytest.raises(ValueError, match="Non-finite irrigation fraction"):
            bld.read_source_irrigation(path)


# --------------------------------------------------------------------------- #
# Joining container classes onto the PEST-lowercased par.csv site index
# --------------------------------------------------------------------------- #
class TestAlignClasses:
    def test_joins_on_lowercase(self):
        med = _medians(
            ["us-ne1", "almond_high"],
            {s: dict.fromkeys(FAMILIES, 1.0) for s in ["us-ne1", "almond_high"]},
        )
        irr = pd.DataFrame(
            {"irr": [0.9, 0.1], "irr_class": ["irrigated", "rainfed"]},
            index=pd.Index(["US-Ne1", "Almond_High"], name="fid"),
        )
        joined = bld.align_classes_to_par_sites(med, irr)
        assert joined.loc["us-ne1", "fid"] == "US-Ne1"
        assert joined.loc["us-ne1", "irr_class"] == "irrigated"
        assert joined.loc["almond_high", "irr_class"] == "rainfed"

    def test_missing_label_raises(self):
        med = _medians(["a", "ghost"], {s: dict.fromkeys(FAMILIES, 1.0) for s in ["a", "ghost"]})
        irr = pd.DataFrame(
            {"irr": [0.9], "irr_class": ["irrigated"]}, index=pd.Index(["A"], name="fid")
        )
        with pytest.raises(ValueError, match="No source irrigation label.*ghost"):
            bld.align_classes_to_par_sites(med, irr)

    def test_case_collision_raises(self):
        """Two container fids differing only in case cannot be joined unambiguously."""
        med = _medians(["a"], {"a": dict.fromkeys(FAMILIES, 1.0)})
        irr = pd.DataFrame(
            {"irr": [0.9, 0.1], "irr_class": ["irrigated", "rainfed"]},
            index=pd.Index(["A", "a"], name="fid"),
        )
        with pytest.raises(ValueError, match="collide when lowercased"):
            bld.align_classes_to_par_sites(med, irr)


# --------------------------------------------------------------------------- #
# Median-of-site-medians within class
# --------------------------------------------------------------------------- #
class TestStratifiedVectors:
    def test_median_within_class_only(self):
        """Each class median is taken over that class's sites alone."""
        sites = ["i1", "i2", "i3", "r1", "r2", "r3"]
        vals = {
            "i1": 1.0,
            "i2": 2.0,
            "i3": 3.0,  # irrigated median 2.0
            "r1": 10.0,
            "r2": 20.0,
            "r3": 30.0,  # rainfed median 20.0
        }
        med = _medians(sites, {s: dict.fromkeys(FAMILIES, vals[s]) for s in sites})
        cls = {s: ("irrigated" if s.startswith("i") else "rainfed") for s in sites}
        out = bld.stratified_vectors(med, cls)
        for fam in FAMILIES:
            assert out["irrigated"]["vector"][fam] == pytest.approx(2.0)
            assert out["rainfed"]["vector"][fam] == pytest.approx(20.0)
        assert out["irrigated"]["n_sites"] == 3
        assert out["rainfed"]["sites"] == ["r1", "r2", "r3"]

    def test_even_count_uses_true_median(self):
        sites = ["i1", "i2", "r1", "r2"]
        vals = {"i1": 1.0, "i2": 4.0, "r1": 10.0, "r2": 11.0}
        med = _medians(sites, {s: dict.fromkeys(FAMILIES, vals[s]) for s in sites})
        cls = {s: ("irrigated" if s.startswith("i") else "rainfed") for s in sites}
        out = bld.stratified_vectors(med, cls)
        assert out["irrigated"]["vector"]["mad"] == pytest.approx(2.5)
        assert out["rainfed"]["vector"]["mad"] == pytest.approx(10.5)

    def test_expected_counts_enforced(self):
        sites = ["i1", "r1"]
        med = _medians(sites, {s: dict.fromkeys(FAMILIES, 1.0) for s in sites})
        cls = {"i1": "irrigated", "r1": "rainfed"}
        bld.stratified_vectors(med, cls, expected_counts={"irrigated": 1, "rainfed": 1})
        with pytest.raises(ValueError, match="do not match the frozen expectation"):
            bld.stratified_vectors(med, cls, expected_counts={"irrigated": 39, "rainfed": 21})

    def test_empty_class_raises(self):
        sites = ["i1", "i2"]
        med = _medians(sites, {s: dict.fromkeys(FAMILIES, 1.0) for s in sites})
        with pytest.raises(ValueError, match="No source sites in irrigation class 'rainfed'"):
            bld.stratified_vectors(med, {"i1": "irrigated", "i2": "irrigated"})

    def test_unknown_class_label_raises(self):
        sites = ["i1", "x1"]
        med = _medians(sites, {s: dict.fromkeys(FAMILIES, 1.0) for s in sites})
        with pytest.raises(ValueError, match="Unrecognized irrigation class"):
            bld.stratified_vectors(med, {"i1": "irrigated", "x1": "fallow"})

    def test_missing_class_for_site_raises(self):
        sites = ["i1", "orphan"]
        med = _medians(sites, {s: dict.fromkeys(FAMILIES, 1.0) for s in sites})
        with pytest.raises(ValueError, match="No source irrigation label.*orphan"):
            bld.stratified_vectors(med, {"i1": "irrigated"})

    def test_non_finite_median_raises(self):
        sites = ["i1", "i2", "r1"]
        per = {s: dict.fromkeys(FAMILIES, 1.0) for s in sites}
        per["i2"]["mad"] = np.nan
        med = _medians(sites, per)
        cls = {"i1": "irrigated", "i2": "irrigated", "r1": "rainfed"}
        with pytest.raises(ValueError, match="Non-finite per-site median in class 'irrigated'"):
            bld.stratified_vectors(med, cls)

    def test_stratification_is_a_partition_of_the_pooled_cohort(self):
        """Every source site lands in exactly one class; the classes cover the cohort."""
        sites = [f"s{i}" for i in range(11)]
        rng = np.random.default_rng(0)
        per = {s: {fam: float(rng.uniform(0.1, 9.9)) for fam in FAMILIES} for s in sites}
        med = _medians(sites, per)
        cls = {s: ("irrigated" if i % 3 else "rainfed") for i, s in enumerate(sites)}
        out = bld.stratified_vectors(med, cls)
        got = set(out["irrigated"]["sites"]) | set(out["rainfed"]["sites"])
        assert got == set(sites)
        assert not set(out["irrigated"]["sites"]) & set(out["rainfed"]["sites"])
        assert out["irrigated"]["n_sites"] + out["rainfed"]["n_sites"] == len(sites)


# --------------------------------------------------------------------------- #
# Source posterior parsing (reused from the pooled builder)
# --------------------------------------------------------------------------- #
class TestSourcePosteriorParsing:
    def test_base_realization_excluded(self, tmp_path):
        sites = ["a", "b", "c"]
        vals = {fam: {"a": 1.0, "b": 2.0, "c": 3.0} for fam in FAMILIES}
        path = _par_csv(tmp_path, sites, values=vals, base=True)
        vector, n, med, n_real = bld.compute_cropland_medians(path)
        assert n == 3 and n_real == 5  # 'base' dropped
        assert vector["mad"] == pytest.approx(2.0)  # 999.0 base would move this
        assert med.loc["b", "mad"] == pytest.approx(2.0)

    def test_missing_base_raises(self, tmp_path):
        path = _par_csv(tmp_path, ["a", "b"], base=False)
        with pytest.raises(ValueError, match="exactly one 'base' realization, found 0"):
            bld.compute_cropland_medians(path)

    def test_duplicate_site_column_raises(self, tmp_path):
        path = _par_csv(tmp_path, ["a", "b"], dup_site="a")
        with pytest.raises(ValueError, match="Duplicate site columns"):
            bld.compute_cropland_medians(path)

    def test_missing_family_raises(self, tmp_path):
        path = _par_csv(tmp_path, ["a", "b"])
        df = pd.read_csv(path, index_col=0)
        df = df[[c for c in df.columns if not c.startswith("pname:p_swe_beta_")]]
        df.to_csv(path)
        with pytest.raises(ValueError, match="No columns found for parameter family 'swe_beta'"):
            bld.compute_cropland_medians(path)

    def test_non_finite_source_value_raises(self, tmp_path):
        sites = ["a", "b"]
        path = _par_csv(tmp_path, sites)
        df = pd.read_csv(path, index_col=0)
        col = next(c for c in df.columns if c.startswith("pname:p_mad_a_"))
        df.loc[[i for i in df.index if i != "base"], col] = np.nan
        df.to_csv(path)
        with pytest.raises(ValueError, match="Non-finite per-site median"):
            bld.compute_cropland_medians(path)

    def test_site_sets_must_agree_across_families(self, tmp_path):
        path = _par_csv(tmp_path, ["a", "b"])
        df = pd.read_csv(path, index_col=0)
        col = next(c for c in df.columns if c.startswith("pname:p_mad_b_"))
        df = df.rename(columns={col: col.replace("_b_", "_zz_")})
        df.to_csv(path)
        with pytest.raises(ValueError, match="Site sets differ across parameter families"):
            bld.compute_cropland_medians(path)


# --------------------------------------------------------------------------- #
# Frozen artifact structure, audit table, prior-domain diagnostic
# --------------------------------------------------------------------------- #
class TestFrozenArtifact:
    def test_expected_table_is_complete_and_distinct(self):
        """The frozen audit table covers both classes x all eight families."""
        for cls in bld.CLASSES:
            assert set(bld.EXPECTED_VECTORS[cls]) == set(FAMILIES)
        # The two classes must actually differ, else stratification is a no-op.
        assert bld.EXPECTED_VECTORS["irrigated"] != bld.EXPECTED_VECTORS["rainfed"]
        assert bld.EXPECTED_COUNTS == {"irrigated": 39, "rainfed": 21}
        assert sum(bld.EXPECTED_COUNTS.values()) == 60

    def test_check_expected_detects_drift(self):
        vectors = {
            cls: {"vector": dict(bld.EXPECTED_VECTORS[cls]), "n_sites": 1, "sites": ["x"]}
            for cls in bld.CLASSES
        }
        ok, rows = bld.check_expected(vectors)
        assert ok and len(rows) == 2 * len(FAMILIES)
        assert all(r["matches"] for r in rows)

        vectors["rainfed"]["vector"]["mad"] += 0.01
        ok, rows = bld.check_expected(vectors)
        assert not ok
        bad = [r for r in rows if not r["matches"]]
        assert len(bad) == 1
        assert bad[0]["irr_class"] == "rainfed" and bad[0]["param"] == "mad"

    def test_prior_domain_report_flags_the_defect_being_fixed(self):
        """Each class vector sits in its own prior; the pooled mad does not."""
        vectors = {
            cls: {"vector": dict(bld.EXPECTED_VECTORS[cls]), "n_sites": 1, "sites": ["x"]}
            for cls in bld.CLASSES
        }
        report = bld.prior_domain_report(vectors)
        assert report["irrigated"]["mad"]["in_domain"] is True
        assert report["rainfed"]["mad"]["in_domain"] is True

        # The pooled value the experiment exists to replace.
        pooled_mad = 0.136917
        lo, hi = bld.CLASS_PRIORS["rainfed"]["mad"]
        assert not (lo <= pooled_mad <= hi)
        lo_i, hi_i = bld.CLASS_PRIORS["irrigated"]["mad"]
        assert lo_i <= pooled_mad <= hi_i

    def test_vector_hash_is_stable_and_order_independent(self):
        v = dict(bld.EXPECTED_VECTORS["irrigated"])
        shuffled = dict(reversed(list(v.items())))
        assert bld._vector_sha256(v) == bld._vector_sha256(shuffled)
        other = dict(v)
        other["mad"] += 1e-9
        assert bld._vector_sha256(v) != bld._vector_sha256(other)

    def test_nested_json_payload_structure(self, tmp_path):
        """The frozen artifact is {class: {family: float}} for exactly two classes."""
        vectors = {
            cls: {"vector": dict(bld.EXPECTED_VECTORS[cls]), "n_sites": 1, "sites": ["x"]}
            for cls in bld.CLASSES
        }
        payload = {cls: vectors[cls]["vector"] for cls in bld.CLASSES}
        path = tmp_path / "vectors.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")

        loaded = json.loads(path.read_text())
        assert set(loaded) == {"irrigated", "rainfed"}
        for cls, vec in loaded.items():
            assert list(vec) == FAMILIES, f"{cls} families out of order"
            assert all(isinstance(x, float) for x in vec.values())
        # Nested (per-class) rather than flat: the downstream E4 evaluator
        # distinguishes the two shapes by whether any top-level value is a dict.
        assert any(isinstance(v, dict) for v in loaded.values())

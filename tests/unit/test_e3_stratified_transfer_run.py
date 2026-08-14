"""Unit tests for the irrigation-stratified comparator in the Ex5 -> E3 transfer runner.

``examples/6_Flux_International/transfer_ex5_params.py`` gained a fifth comparator,
``ex5_transfer_strat``: a per-site ``{site_id: {param: value}}`` mapping in which each E3
site receives the irrigated or the rainfed Run 22 class vector. The new surface is small
but every piece of it guards a way the published comparison could go wrong silently:

1. ``_load_params_by_site`` refuses partial cohort coverage. The optional LULC comparator in
   the same script *does* filter silently (``[f for f in fids if f.lower() in
   lulc_params_lower]``), which quietly shrinks the support the configurations are compared
   on. That behaviour was deliberately not copied, so the hard failure is tested as a
   contract rather than an implementation detail.
2. ``_resolve_class_assignment`` recovers the class labels from the vectors actually handed
   to the model (grouping sites by vector hash) instead of trusting a claim recorded
   elsewhere -- so the run metadata cannot disagree with the run.
3. Output-path separation. ``_write_pooled_metrics`` re-reads ``{fid}.csv`` off a directory,
   so the stratified run needs its own series subdirectory and its own ``*_strat.csv``
   files, and ``--require-empty-out`` exists to stop a stale CSV from being pooled in. Both
   are tested against real files on disk.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E6_DIR = REPO_ROOT / "examples" / "6_Flux_International"


SIBLINGS = ("evaluate", "pooled_metrics", "derived_metrics", "shapefile")


def _load_runner():
    """Import the runner by path (``examples/`` is not an importable package).

    The module puts its own directory on ``sys.path`` at import time (it needs the sibling
    ``evaluate`` / ``pooled_metrics`` / ``derived_metrics`` modules). Two collisions have to
    be handled or a full-suite run breaks where a single-file run passes:

    * every example directory has its own ``evaluate.py``, so a sibling already cached in
      ``sys.modules`` by an earlier test would satisfy the import with the wrong module
      (Example 5's ``evaluate`` has no ``load_flux_sources``). The cache entries are evicted
      before the import and restored afterwards.
    * this directory also holds a ``shapefile.py``, which would shadow the ``shapefile``
      package for the rest of the session, so the path entry is removed once loaded.
    """
    saved = {name: sys.modules.pop(name, None) for name in SIBLINGS}
    sys.path.insert(0, str(E6_DIR))
    spec = importlib.util.spec_from_file_location(
        "transfer_ex5_params", E6_DIR / "transfer_ex5_params.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        loaded = {name: sys.modules.get(name) for name in SIBLINGS}
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E6_DIR)]
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return module, loaded


tx, _E6_SIBLINGS = _load_runner()
pm = _E6_SIBLINGS["pooled_metrics"]

FAMILIES = ["aw", "ndvi_k", "ndvi_0", "mad", "ks_alpha", "kr_alpha", "swe_alpha", "swe_beta"]
IRRIGATED_VEC = {fam: 1.0 + i for i, fam in enumerate(FAMILIES)}
RAINFED_VEC = {fam: 100.0 + i for i, fam in enumerate(FAMILIES)}

# The four suffixed artifacts the stratified run must own, and the four un-suffixed ones
# that must stay the --params pooled run's.
STRAT_FILES = [
    "evaluation_metrics_strat.csv",
    "evaluation_monthly_metrics_strat.csv",
    "pooled_metrics_daily_strat.csv",
    "pooled_metrics_monthly_strat.csv",
]
POOLED_FILES = [f.replace("_strat.csv", ".csv") for f in STRAT_FILES]

RUNNER_SRC = (E6_DIR / "transfer_ex5_params.py").read_text()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _mapping_json(tmp_path, mapping, name="mapping.json"):
    path = tmp_path / name
    path.write_text(json.dumps(mapping, indent=2) + "\n")
    return path


def _two_class_mapping(irr_sites, rf_sites):
    m = {s: dict(IRRIGATED_VEC) for s in irr_sites}
    m.update({s: dict(RAINFED_VEC) for s in rf_sites})
    return m


def _metrics_frame(fids, seed=0, offset=0.0):
    """A frame in the shape ``daily_site_metrics`` produces (one row per site)."""
    rng = np.random.default_rng(seed)
    rows = {
        "n": [100 + i for i in range(len(fids))],
        "r2": rng.uniform(0.3, 0.9, len(fids)) + offset,
        "r": rng.uniform(0.5, 0.95, len(fids)),
        "rmse": rng.uniform(0.5, 2.0, len(fids)),
        "bias": rng.uniform(-0.3, 0.3, len(fids)),
        "kge": rng.uniform(0.2, 0.8, len(fids)) + offset,
        "mae": rng.uniform(0.3, 1.5, len(fids)),
        "alpha": rng.uniform(0.7, 1.3, len(fids)),
        "beta": rng.uniform(0.7, 1.3, len(fids)),
    }
    return pd.DataFrame(rows, index=pd.Index(list(fids), name="fid"))


SERIES_DAYS = 90
SERIES_INDEX = pd.date_range("2015-01-01", periods=SERIES_DAYS, freq="D")
# A non-constant reference series: pooled r2/slope are undefined on a constant one.
SERIES_BASE = 3.0 + 1.5 * np.sin(np.arange(SERIES_DAYS) * 2.0 * np.pi / 30.0)


def _site_csv(directory, fid, offset=0.0, rs_offset=0.5):
    """Write a per-site ``{fid}.csv`` in the E3 layout that pooled metrics re-read.

    ``et_act`` is the flux series plus a constant ``offset``, so the pooled bias of that
    series is exactly ``offset`` and a swapped series dir is immediately visible.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "etref": 5.0,
            "et_act": SERIES_BASE + offset,
            "et_rs": SERIES_BASE + rs_offset,
        },
        index=SERIES_INDEX,
    )
    df.index.name = "date"
    df.to_csv(directory / f"{fid}.csv")
    return df


@pytest.fixture
def flux_stub(monkeypatch):
    """Serve a constant synthetic flux series so pooled metrics can run offline.

    The real loader reads the QAQC archive under ``/nas``; only the flux side is stubbed,
    so the per-site CSVs that ``_write_pooled_metrics`` reads are genuine files on disk.
    """

    def _load(fid, source=None):
        return pd.Series(SERIES_BASE.copy(), index=SERIES_INDEX)

    monkeypatch.setattr(pm, "load_flux_et", _load)
    monkeypatch.setattr(pm, "passes_site_minimum", lambda flux: True)
    return SERIES_INDEX


# --------------------------------------------------------------------------- #
# Configuration registry
# --------------------------------------------------------------------------- #
class TestConfigRegistry:
    def test_core_config_order_is_exact(self):
        """Display order is the comparator narrative: defaults, pooled, stratified, local, RS."""
        assert tx.CORE_CONFIGS == [
            "e3_uncal",
            "ex5_transfer",
            "ex5_transfer_strat",
            "e3_cal",
            "ls_ensemble",
        ]
        assert tx.STRAT_CONFIG == "ex5_transfer_strat"
        assert tx.CORE_CONFIGS[2] == tx.STRAT_CONFIG
        # The stratified run sits immediately after the pooled transfer it is paired against.
        assert tx.CORE_CONFIGS.index(tx.STRAT_CONFIG) == tx.CORE_CONFIGS.index("ex5_transfer") + 1
        assert len(tx.CORE_CONFIGS) == len(set(tx.CORE_CONFIGS)) == 5
        assert "lulc_defaults" not in tx.CORE_CONFIGS

    def test_every_config_has_a_label(self):
        for config in tx.CORE_CONFIGS + ["lulc_defaults"]:
            assert config in tx.CONFIG_LABELS
        assert tx.CONFIG_LABELS[tx.STRAT_CONFIG] == "Ex5 stratified transfer"

    def test_strat_description_records_the_target_class_policy(self):
        desc = tx.CONFIG_DESCRIPTIONS[tx.STRAT_CONFIG]
        assert "two-stage" in desc
        assert "equipped -> irrigated vector" in desc
        assert "not equipped -> rainfed vector" in desc

    def test_core_configs_helper_preserves_declared_order(self):
        """Frames arrive in whatever order the run built them; display order is fixed."""
        frames = dict.fromkeys(["ls_ensemble", tx.STRAT_CONFIG, "e3_cal", "ex5_transfer"], None)
        assert tx._core_configs(frames) == [
            "ex5_transfer",
            tx.STRAT_CONFIG,
            "e3_cal",
            "ls_ensemble",
        ]

    def test_strat_dropped_from_core_when_not_run(self):
        frames = dict.fromkeys(["e3_uncal", "ex5_transfer", "e3_cal", "ls_ensemble"], None)
        assert tx.STRAT_CONFIG not in tx._core_configs(frames)
        assert tx._core_configs(frames) == [c for c in tx.CORE_CONFIGS if c != tx.STRAT_CONFIG]


# --------------------------------------------------------------------------- #
# --params-by-site loading
# --------------------------------------------------------------------------- #
class TestLoadParamsBySite:
    def test_happy_path_keys_on_container_fids(self, tmp_path):
        fids = ["US-Ne1", "DE-Kli", "AU-Rgf"]
        path = _mapping_json(tmp_path, _two_class_mapping(["US-Ne1"], ["DE-Kli", "AU-Rgf"]))
        params, source_keys = tx._load_params_by_site(path, fids)
        assert list(params) == fids
        assert params["US-Ne1"] == IRRIGATED_VEC
        assert params["DE-Kli"] == RAINFED_VEC
        assert source_keys == {f: f for f in fids}

    def test_case_insensitive_match_reports_the_source_key(self, tmp_path):
        """PEST lowercases site tokens; the mapping may carry either casing."""
        fids = ["US-Ne1", "DE-Kli"]
        path = _mapping_json(tmp_path, _two_class_mapping(["us-ne1"], ["de-kli"]), "lower.json")
        params, source_keys = tx._load_params_by_site(path, fids)
        assert params["US-Ne1"] == IRRIGATED_VEC
        assert source_keys == {"US-Ne1": "us-ne1", "DE-Kli": "de-kli"}

    def test_values_are_coerced_to_float(self, tmp_path):
        path = _mapping_json(
            tmp_path, {"A": {f: int(v) for f, v in IRRIGATED_VEC.items()}}, "ints.json"
        )
        params, _ = tx._load_params_by_site(path, ["A"])
        assert all(isinstance(v, float) for v in params["A"].values())

    def test_returned_vectors_are_copies(self, tmp_path):
        path = _mapping_json(tmp_path, _two_class_mapping(["A", "B"], ["C"]), "copies.json")
        params, _ = tx._load_params_by_site(path, ["A", "B", "C"])
        params["A"]["mad"] = -999.0
        assert params["B"]["mad"] == IRRIGATED_VEC["mad"]

    def test_flat_vector_raises_and_points_at_params(self, tmp_path):
        """A ``{param: value}`` file on --params-by-site would corrupt into {fid: {site: v}}."""
        path = _mapping_json(tmp_path, dict(IRRIGATED_VEC), "flat.json")
        with pytest.raises(ValueError) as exc:
            tx._load_params_by_site(path, ["A"])
        msg = str(exc.value)
        assert "expects a nested" in msg
        assert "belongs on --params" in msg
        assert "{fid: {site: value}}" in msg

    def test_single_non_dict_value_raises(self, tmp_path):
        mapping = _two_class_mapping(["A"], ["B"])
        mapping["C"] = 1.5
        path = _mapping_json(tmp_path, mapping, "nondict.json")
        with pytest.raises(ValueError, match=r"non-dict value\(s\) for \['C'\]"):
            tx._load_params_by_site(path, ["A", "B", "C"])

    def test_empty_object_raises(self, tmp_path):
        path = _mapping_json(tmp_path, {}, "empty.json")
        with pytest.raises(ValueError, match="not a non-empty JSON object"):
            tx._load_params_by_site(path, ["A"])

    def test_non_object_json_raises(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]\n")
        with pytest.raises(ValueError, match="not a non-empty JSON object"):
            tx._load_params_by_site(path, ["A"])

    def test_lowercase_collision_raises(self, tmp_path):
        """Two mapping keys differing only in case cannot be resolved to one fid."""
        mapping = {"US-Ne1": dict(IRRIGATED_VEC), "us-ne1": dict(RAINFED_VEC)}
        path = _mapping_json(tmp_path, mapping, "collide.json")
        with pytest.raises(ValueError, match="site ids collide when lowercased"):
            tx._load_params_by_site(path, ["US-Ne1"])

    def test_inconsistent_parameter_sets_raise(self, tmp_path):
        mapping = _two_class_mapping(["A"], ["B"])
        del mapping["B"]["swe_beta"]
        path = _mapping_json(tmp_path, mapping, "ragged.json")
        with pytest.raises(ValueError, match="parameter names differ across sites"):
            tx._load_params_by_site(path, ["A", "B"])

    def test_renamed_parameter_raises_even_at_equal_length(self, tmp_path):
        mapping = _two_class_mapping(["A"], ["B"])
        mapping["B"]["p_stress"] = mapping["B"].pop("mad")
        path = _mapping_json(tmp_path, mapping, "renamed.json")
        with pytest.raises(ValueError, match="found 2 distinct parameter sets"):
            tx._load_params_by_site(path, ["A", "B"])

    def test_parameter_set_check_is_scoped_to_the_cohort(self, tmp_path):
        """A ragged vector for a site outside the cohort is never applied, so it is ignored."""
        mapping = _two_class_mapping(["A"], ["B"])
        mapping["OUTSIDE"] = {"aw": 1.0}
        path = _mapping_json(tmp_path, mapping, "outside.json")
        params, _ = tx._load_params_by_site(path, ["A", "B"])
        assert set(params) == {"A", "B"}

    def test_missing_cohort_site_raises_with_the_full_list(self, tmp_path):
        """No silent filtering: partial coverage changes the common support, so it is fatal."""
        fids = ["A", "B", "C", "D"]
        path = _mapping_json(tmp_path, _two_class_mapping(["A"], ["B"]), "partial.json")
        with pytest.raises(ValueError) as exc:
            tx._load_params_by_site(path, fids)
        msg = str(exc.value)
        assert "is missing 2 of the 4 cohort site(s): ['C', 'D']" in msg
        assert "must cover the whole cohort" in msg
        assert "silently change the common support" in msg

    def test_completeness_is_checked_before_case_folding(self, tmp_path):
        """A case mismatch must read as covered, not as absent.

        The docstring states the completeness check runs on the case-folded index so a
        casing difference can never be misreported as missing coverage.
        """
        path = _mapping_json(tmp_path, _two_class_mapping(["us-ne1"], []), "case.json")
        params, _ = tx._load_params_by_site(path, ["US-Ne1"])
        assert params["US-Ne1"] == IRRIGATED_VEC

    def test_superset_mapping_is_accepted(self, tmp_path):
        """The frozen 66-site mapping may be reused on a --sites subset."""
        mapping = _two_class_mapping(["A", "B"], ["C", "D"])
        path = _mapping_json(tmp_path, mapping, "superset.json")
        params, source_keys = tx._load_params_by_site(path, ["A", "C"])
        assert set(params) == {"A", "C"} == set(source_keys)

    def test_the_lulc_comparator_still_filters_silently(self):
        """The contrast this hard failure exists against, pinned so it cannot drift.

        The optional LULC path keeps only the sites it happens to cover; the stratified path
        deliberately does not copy that. If someone ever "harmonises" the two by making the
        stratified path filter, this assertion plus the raising test above will disagree.
        """
        assert "lulc_fids = [f for f in fids if f.lower() in lulc_params_lower]" in RUNNER_SRC
        doc = tx._load_params_by_site.__doc__
        assert "Every cohort site must be present" in doc
        assert "the way the optional LULC comparator does" in doc


# --------------------------------------------------------------------------- #
# Class-assignment recovery from the applied vectors
# --------------------------------------------------------------------------- #
class TestResolveClassAssignment:
    def test_two_distinct_vectors_give_two_groups(self, tmp_path):
        params = _two_class_mapping(["A", "B"], ["C", "D", "E"])
        assignment, counts, vectors, hashes, companion = tx._resolve_class_assignment(
            params, tmp_path / "m.json"
        )
        assert companion is None
        assert set(counts.values()) == {2, 3}
        assert len(counts) == 2
        assert set(assignment) == set(params)
        groups = {}
        for fid, name in assignment.items():
            groups.setdefault(name, set()).add(fid)
        assert {frozenset(g) for g in groups.values()} == {
            frozenset({"A", "B"}),
            frozenset({"C", "D", "E"}),
        }
        assert all(set(v) == set(FAMILIES) for v in vectors.values())
        assert len(set(hashes.values())) == 2
        for name, vec in vectors.items():
            assert tx._vector_sha256(vec) == hashes[name]

    def test_identical_vectors_collapse_to_one_group(self, tmp_path):
        params = _two_class_mapping(["A", "B", "C"], [])
        assignment, counts, vectors, hashes, _ = tx._resolve_class_assignment(
            params, tmp_path / "m.json"
        )
        assert counts == {"vector_1": 3}
        assert set(assignment.values()) == {"vector_1"}
        assert vectors["vector_1"] == IRRIGATED_VEC
        assert len(hashes) == 1

    def test_singleton_cohort(self, tmp_path):
        params = {"A": dict(RAINFED_VEC)}
        assignment, counts, vectors, hashes, _ = tx._resolve_class_assignment(
            params, tmp_path / "m.json"
        )
        assert assignment == {"A": "vector_1"}
        assert counts == {"vector_1": 1}
        assert vectors == {"vector_1": RAINFED_VEC}
        assert hashes == {"vector_1": tx._vector_sha256(RAINFED_VEC)}

    def test_three_vectors_are_named_in_hash_order(self, tmp_path):
        """The fallback names follow sorted hash order, so they are reproducible."""
        third = {fam: 50.0 + i for i, fam in enumerate(FAMILIES)}
        params = {"A": dict(IRRIGATED_VEC), "B": dict(RAINFED_VEC), "C": dict(third)}
        assignment, counts, vectors, hashes, _ = tx._resolve_class_assignment(
            params, tmp_path / "m.json"
        )
        assert sorted(counts) == ["vector_1", "vector_2", "vector_3"]
        assert set(counts.values()) == {1}
        ordered = sorted(tx._vector_sha256(v) for v in (IRRIGATED_VEC, RAINFED_VEC, third))
        assert [hashes[f"vector_{i + 1}"] for i in range(3)] == ordered
        assert len(set(assignment.values())) == 3

    def test_companion_metadata_supplies_the_class_names(self, tmp_path):
        mapping_path = _mapping_json(tmp_path, _two_class_mapping(["A", "B"], ["C"]))
        meta_path = tmp_path / "mapping_metadata.json"
        meta_path.write_text(
            json.dumps(
                {"class_vectors": {"irrigated": IRRIGATED_VEC, "rainfed": RAINFED_VEC}}, indent=2
            )
        )
        params, _ = tx._load_params_by_site(mapping_path, ["A", "B", "C"])
        assignment, counts, vectors, hashes, companion = tx._resolve_class_assignment(
            params, mapping_path
        )
        assert companion == str(meta_path)
        assert assignment == {"A": "irrigated", "B": "irrigated", "C": "rainfed"}
        assert counts == {"irrigated": 2, "rainfed": 1}
        assert vectors == {"irrigated": IRRIGATED_VEC, "rainfed": RAINFED_VEC}
        assert hashes["rainfed"] == tx._vector_sha256(RAINFED_VEC)

    def test_companion_names_only_hash_matching_vectors(self, tmp_path):
        """A stale companion cannot relabel a vector the run did not actually apply."""
        mapping_path = _mapping_json(tmp_path, _two_class_mapping(["A"], ["B"]), "m2.json")
        (tmp_path / "m2_metadata.json").write_text(
            json.dumps(
                {
                    "class_vectors": {
                        "irrigated": IRRIGATED_VEC,
                        "rainfed": {fam: 7.0 for fam in FAMILIES},  # not applied anywhere
                    }
                }
            )
        )
        params, _ = tx._load_params_by_site(mapping_path, ["A", "B"])
        assignment, counts, _, _, companion = tx._resolve_class_assignment(params, mapping_path)
        assert companion is not None
        assert assignment["A"] == "irrigated"
        assert assignment["B"].startswith("vector_")
        assert set(counts) == {"irrigated", assignment["B"]}

    def test_corrupt_companion_metadata_falls_back_without_raising(self, tmp_path):
        mapping_path = _mapping_json(tmp_path, _two_class_mapping(["A"], ["B"]), "m3.json")
        (tmp_path / "m3_metadata.json").write_text("{not json")
        params, _ = tx._load_params_by_site(mapping_path, ["A", "B"])
        assignment, counts, _, _, companion = tx._resolve_class_assignment(params, mapping_path)
        assert companion is not None
        assert sorted(counts) == ["vector_1", "vector_2"]
        assert set(assignment.values()) == {"vector_1", "vector_2"}

    def test_companion_without_class_vectors_key_falls_back(self, tmp_path):
        mapping_path = _mapping_json(tmp_path, _two_class_mapping(["A"], ["B"]), "m4.json")
        (tmp_path / "m4_metadata.json").write_text(json.dumps({"experiment": "something"}))
        params, _ = tx._load_params_by_site(mapping_path, ["A", "B"])
        _, counts, _, _, _ = tx._resolve_class_assignment(params, mapping_path)
        assert sorted(counts) == ["vector_1", "vector_2"]

    def test_recovered_vectors_are_the_applied_ones(self, tmp_path):
        """The recorded vector is read back out of ``params_by_fid``, not out of the JSON."""
        params = _two_class_mapping(["A", "B"], ["C"])
        params["A"]["mad"] = 0.42  # A now forms its own group
        params["B"]["mad"] = 0.42
        _, counts, vectors, _, _ = tx._resolve_class_assignment(params, tmp_path / "m.json")
        assert sorted(counts.values()) == [1, 2]
        assert any(v["mad"] == 0.42 for v in vectors.values())

    def test_group_names_are_unique(self, tmp_path):
        params = {f"S{i}": {fam: float(i) for fam in FAMILIES} for i in range(6)}
        assignment, counts, vectors, hashes, _ = tx._resolve_class_assignment(
            params, tmp_path / "m.json"
        )
        assert len(counts) == len(vectors) == len(hashes) == 6
        assert len(set(assignment.values())) == 6


class TestVectorHash:
    def test_key_order_independent(self):
        shuffled = dict(reversed(list(IRRIGATED_VEC.items())))
        assert tx._vector_sha256(IRRIGATED_VEC) == tx._vector_sha256(shuffled)

    def test_int_and_float_encodings_agree(self):
        """The hash floats every value, so a JSON int and a JSON float are the same vector."""
        as_int = {fam: int(v) for fam, v in RAINFED_VEC.items()}
        as_float = {fam: float(int(v)) for fam, v in RAINFED_VEC.items()}
        assert tx._vector_sha256(as_int) == tx._vector_sha256(as_float)

    def test_sensitive_to_a_tiny_change(self):
        other = dict(IRRIGATED_VEC)
        other["mad"] += 1e-12
        assert tx._vector_sha256(other) != tx._vector_sha256(IRRIGATED_VEC)


# --------------------------------------------------------------------------- #
# --require-empty-out
# --------------------------------------------------------------------------- #
class _Reached(RuntimeError):
    """Sentinel raised once main() gets past the output-directory guard."""


class TestRequireEmptyOut:
    def _invoke(self, tmp_path, monkeypatch, out_dir, argv_extra=()):
        """Drive ``main()`` as far as the guard, then stop at a sentinel.

        Only the two external boundaries are stubbed -- config loading and container opening.
        Everything between them, the guard included, is the real code path.
        """
        cfg = SimpleNamespace(
            fields_shapefile=str(tmp_path / "no_such.shp"),
            feature_id_col="sid",
            project_ws=str(tmp_path),
        )
        monkeypatch.setattr(tx.ev, "_load_config", lambda p: cfg)

        def _boom(*a, **k):
            raise _Reached("past the guard")

        monkeypatch.setattr(tx, "SwimContainer", SimpleNamespace(open=_boom))

        params = tmp_path / "vector.json"
        params.write_text(json.dumps(IRRIGATED_VEC))
        argv = [
            "transfer_ex5_params.py",
            "--config",
            str(tmp_path / "project.toml"),
            "--params",
            str(params),
            "--container",
            str(tmp_path / "c.swim"),
            "--e3-results-dir",
            str(tmp_path / "e3results"),
            "--out",
            str(out_dir),
            *argv_extra,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        tx.main()

    def test_empty_dir_passes(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(_Reached):
            self._invoke(tmp_path, monkeypatch, out, ["--require-empty-out"])

    def test_absent_dir_passes(self, tmp_path, monkeypatch):
        out = tmp_path / "brand_new"
        with pytest.raises(_Reached):
            self._invoke(tmp_path, monkeypatch, out, ["--require-empty-out"])
        assert out.exists()  # created by main()

    def test_stray_csv_raises(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        (out / "US-Ne1.csv").write_text("date,et_act,et_rs\n")
        with pytest.raises(FileExistsError) as exc:
            self._invoke(tmp_path, monkeypatch, out, ["--require-empty-out"])
        msg = str(exc.value)
        assert "already contains 1 CSV file(s)" in msg
        assert "US-Ne1.csv" in msg
        # The whole point of the flag: pooled metrics are re-read from these CSVs.
        assert "Pooled metrics are re-read from the per-site {site}.csv" in msg
        assert "pooled in" in msg

    def test_nested_csv_raises(self, tmp_path, monkeypatch):
        """The stratified series live in a subdirectory, so the scan must recurse."""
        out = tmp_path / "out"
        (out / tx.STRAT_CONFIG).mkdir(parents=True)
        (out / tx.STRAT_CONFIG / "US-Ne1.csv").write_text("date\n")
        with pytest.raises(FileExistsError, match=r"ex5_transfer_strat/US-Ne1\.csv"):
            self._invoke(tmp_path, monkeypatch, out, ["--require-empty-out"])

    def test_non_csv_files_do_not_trip_the_guard(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        for name in ("notes.txt", "fix.patch", "run_metadata.json", "figure.png", "README.md"):
            (out / name).write_text("x")
        with pytest.raises(_Reached):
            self._invoke(tmp_path, monkeypatch, out, ["--require-empty-out"])

    def test_flag_omitted_skips_the_check(self, tmp_path, monkeypatch):
        """Default-off for backward compatibility with the already-archived runs."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "US-Ne1.csv").write_text("date,et_act,et_rs\n")
        with pytest.raises(_Reached):
            self._invoke(tmp_path, monkeypatch, out)

    def test_flag_is_documented_as_off_by_default(self):
        assert "--require-empty-out" in tx.__doc__
        assert "off by default" in tx.__doc__.lower()
        assert "_write_pooled_metrics`` re-reads ``{site}.csv``" in tx.__doc__


# --------------------------------------------------------------------------- #
# Output-path separation: the pooled run and the stratified run never mix
# --------------------------------------------------------------------------- #
class TestOutputPathSeparation:
    def test_strat_series_subdirectory_is_the_config_name(self):
        assert "strat_dir = out_dir / STRAT_CONFIG" in RUNNER_SRC
        assert 'site_out.to_csv(out_dir / f"{fid}.csv")' in RUNNER_SRC
        assert 'strat_out.to_csv(strat_dir / f"{fid}.csv")' in RUNNER_SRC
        assert (Path("out") / tx.STRAT_CONFIG / "US-Ne1.csv").as_posix() == (
            "out/ex5_transfer_strat/US-Ne1.csv"
        )

    def test_documented_artifact_names(self):
        for name in STRAT_FILES + POOLED_FILES:
            assert name in tx.__doc__, name
        assert "ex5_transfer_strat/{site}.csv" in tx.__doc__

    def test_suffixed_evaluation_metrics_are_a_separate_file(self, tmp_path):
        fids = [f"S{i}" for i in range(12)]
        daily = {
            "ex5_transfer": _metrics_frame(fids, seed=1),
            tx.STRAT_CONFIG: _metrics_frame(fids, seed=2),
            "ls_ensemble": _metrics_frame(fids, seed=3),
        }
        monthly = {k: _metrics_frame(fids, seed=10 + i) for i, k in enumerate(daily)}

        tx._write_evaluation_metrics(tmp_path, daily, monthly)
        tx._write_evaluation_metrics(
            tmp_path, daily, monthly, config=tx.STRAT_CONFIG, suffix="_strat"
        )

        for name in [
            "evaluation_metrics.csv",
            "evaluation_monthly_metrics.csv",
            "evaluation_metrics_strat.csv",
            "evaluation_monthly_metrics_strat.csv",
        ]:
            assert (tmp_path / name).exists(), name

        pooled = pd.read_csv(tmp_path / "evaluation_metrics.csv").set_index("fid")
        strat = pd.read_csv(tmp_path / "evaluation_metrics_strat.csv").set_index("fid")
        # Same sites and the same RS reference, different SWIM column.
        assert pooled.index.tolist() == strat.index.tolist()
        pd.testing.assert_series_equal(pooled["kge_rs"], strat["kge_rs"])
        assert not np.allclose(pooled["kge_swim"], strat["kge_swim"])
        pd.testing.assert_series_equal(
            pooled["kge_swim"], daily["ex5_transfer"].loc[pooled.index, "kge"], check_names=False
        )
        pd.testing.assert_series_equal(
            strat["kge_swim"], daily[tx.STRAT_CONFIG].loc[strat.index, "kge"], check_names=False
        )

    def test_unsuffixed_evaluation_metrics_stay_the_pooled_run(self, tmp_path):
        """The default ``config`` argument is the --params run, not the stratified one."""
        assert tx._write_evaluation_metrics.__defaults__[0] == "ex5_transfer"
        assert tx._write_evaluation_metrics.__defaults__[1] == ""

    def test_pooled_metrics_read_the_series_dir_and_write_to_out_dir(self, tmp_path, flux_stub):
        fids = ["S0", "S1", "S2"]
        strat_dir = tmp_path / tx.STRAT_CONFIG
        for fid in fids:
            _site_csv(tmp_path, fid, offset=0.0)  # pooled run: matches flux exactly
            _site_csv(strat_dir, fid, offset=1.0)  # stratified run: +1 mm/day

        tx._write_pooled_metrics(tmp_path, fids)
        tx._write_pooled_metrics(
            tmp_path,
            fids,
            label="Ex5 stratified transfer",
            suffix="_strat",
            series_dir=strat_dir,
        )

        for name in [
            "pooled_metrics_daily.csv",
            "pooled_metrics_monthly.csv",
            "pooled_metrics_daily_strat.csv",
            "pooled_metrics_monthly_strat.csv",
        ]:
            assert (tmp_path / name).exists(), name
        # Everything lands in out_dir; nothing is written into the series subdirectory.
        assert sorted(p.name for p in strat_dir.glob("*.csv")) == ["S0.csv", "S1.csv", "S2.csv"]

        pooled = pd.read_csv(tmp_path / "pooled_metrics_daily.csv").set_index("model")
        strat = pd.read_csv(tmp_path / "pooled_metrics_daily_strat.csv").set_index("model")
        assert pooled.loc["swim", "n_stations"] == 3
        assert pooled.loc["swim", "bias_pooled"] == pytest.approx(0.0, abs=1e-9)
        assert strat.loc["swim", "bias_pooled"] == pytest.approx(1.0, abs=1e-9)
        # The RS side comes from the same et_rs column in both series dirs.
        assert pooled.loc["rs", "bias_pooled"] == pytest.approx(strat.loc["rs", "bias_pooled"])

    def test_a_stale_csv_in_out_dir_is_pooled_in_silently(self, tmp_path, flux_stub):
        """Exactly the failure ``--require-empty-out`` exists to prevent.

        ``_write_pooled_metrics`` enumerates ``fids``, not the directory, but a leftover CSV
        for a cohort site from a *different* run is indistinguishable from this run's own
        output and is pooled in with no warning.
        """
        fids = ["S0", "S1"]
        _site_csv(tmp_path, "S0", offset=0.0)
        tx._write_pooled_metrics(tmp_path, fids)
        clean = pd.read_csv(tmp_path / "pooled_metrics_daily.csv").set_index("model")
        assert clean.loc["swim", "n_stations"] == 1
        assert clean.loc["swim", "bias_pooled"] == pytest.approx(0.0, abs=1e-9)

        # A stale S1.csv from an earlier, differently-parameterised run.
        _site_csv(tmp_path, "S1", offset=6.0)
        tx._write_pooled_metrics(tmp_path, fids)
        polluted = pd.read_csv(tmp_path / "pooled_metrics_daily.csv").set_index("model")
        assert polluted.loc["swim", "n_stations"] == 2
        assert polluted.loc["swim", "bias_pooled"] == pytest.approx(3.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Win rates and run metadata
# --------------------------------------------------------------------------- #
class TestWinRatesAndMetadata:
    def _frames(self, fids):
        configs = ["e3_uncal", "ex5_transfer", tx.STRAT_CONFIG, "e3_cal", "ls_ensemble"]
        daily = {c: _metrics_frame(fids, seed=i) for i, c in enumerate(configs)}
        monthly = {c: _metrics_frame(fids, seed=100 + i) for i, c in enumerate(configs)}
        return daily, monthly

    def test_stratified_transfer_becomes_a_second_reference(self, tmp_path):
        fids = [f"S{i}" for i in range(9)]
        daily, monthly = self._frames(fids)
        df = tx._write_winrates(tmp_path, daily, monthly)
        refs = set(df["reference"])
        assert refs == {tx.CONFIG_LABELS["ex5_transfer"], tx.CONFIG_LABELS[tx.STRAT_CONFIG]}
        # The primary question: stratified vs pooled, as its own row.
        primary = f"{tx.CONFIG_LABELS[tx.STRAT_CONFIG]} vs {tx.CONFIG_LABELS['ex5_transfer']}"
        assert primary in set(df["comparison"])
        assert (df["n_daily"] == len(fids)).all()
        assert df["daily_r2_win"].between(0.0, 1.0).all()
        # No config is ever compared against itself.
        assert all(
            row["comparison"] != f"{row['reference']} vs {row['reference']}"
            for _, row in df.iterrows()
        )

    def test_pooled_only_run_has_one_reference(self, tmp_path):
        fids = [f"S{i}" for i in range(9)]
        daily, monthly = self._frames(fids)
        for d in (daily, monthly):
            d.pop(tx.STRAT_CONFIG)
        df = tx._write_winrates(tmp_path, daily, monthly)
        assert set(df["reference"]) == {tx.CONFIG_LABELS["ex5_transfer"]}
        assert tx.CONFIG_LABELS[tx.STRAT_CONFIG] not in " ".join(df["comparison"])

    def test_win_rate_is_a_paired_common_site_fraction(self):
        a = _metrics_frame(["S0", "S1", "S2"], seed=1)
        b = a.copy()
        b["kge"] = a["kge"] - 0.1
        frac, n = tx._win_rate(a, b, "kge")
        assert (frac, n) == (1.0, 3)
        frac, n = tx._win_rate(b, a, "kge")
        assert (frac, n) == (0.0, 3)
        # Only common sites count.
        frac, n = tx._win_rate(a, b.loc[["S0"]], "kge")
        assert (frac, n) == (1.0, 1)

    def test_metadata_records_the_stratified_provenance(self, tmp_path):
        fids = [f"S{i}" for i in range(5)]
        daily, monthly = self._frames(fids)
        strat_dir = tmp_path / tx.STRAT_CONFIG
        strat_meta = {
            "path": str(tmp_path / "mapping.json"),
            "sha256": "0" * 64,
            "companion_metadata": str(tmp_path / "mapping_metadata.json"),
            "class_assignment": {"S0": "irrigated", "S1": "rainfed"},
            "class_counts": {"irrigated": 1, "rainfed": 4},
            "class_vectors": {"irrigated": IRRIGATED_VEC, "rainfed": RAINFED_VEC},
            "class_vector_sha256": {"irrigated": "a" * 64, "rainfed": "b" * 64},
            "source_site_keys": {f: f.lower() for f in fids},
        }
        args = SimpleNamespace(params=str(tmp_path / "vector.json"), require_empty_out=True)
        tx._write_metadata(
            tmp_path,
            args,
            tmp_path / "project.toml",
            tmp_path / "c.swim",
            tmp_path / "e3results",
            dict(IRRIGATED_VEC),
            fids,
            daily,
            monthly,
            pd.Index(fids),
            pd.Index(fids),
            False,
            strat_meta,
            strat_dir,
        )
        meta = json.loads((tmp_path / "run_metadata.json").read_text())

        assert meta["stratified_comparator_included"] is True
        assert meta["lulc_comparator_included"] is False
        assert meta["require_empty_out"] is True
        # Top-level per-site series belong to the pooled --params run.
        assert meta["persite_series_config"] == "ex5_transfer"

        st = meta["stratified_transfer"]
        assert st["config_key"] == tx.STRAT_CONFIG
        assert st["label"] == tx.CONFIG_LABELS[tx.STRAT_CONFIG]
        assert st["description"] == tx.CONFIG_DESCRIPTIONS[tx.STRAT_CONFIG]
        assert st["persite_series_dir"] == str(strat_dir)
        assert Path(st["persite_series_dir"]).name == tx.STRAT_CONFIG
        assert st["class_counts"] == strat_meta["class_counts"]
        assert st["class_vectors"] == strat_meta["class_vectors"]
        assert st["params_by_site_sha256"] == "0" * 64
        assert "must appear in the mapping" in st["coverage_rule"]
        assert "rather than reducing the common support" in st["coverage_rule"]
        assert "grouped by vector hash" in st["class_assignment_source"]
        assert "does not change a site's parameter vector" in st["annual_stage2_note"]

    def test_metadata_omits_the_strat_block_for_a_pooled_only_run(self, tmp_path):
        fids = [f"S{i}" for i in range(5)]
        daily, monthly = self._frames(fids)
        for d in (daily, monthly):
            d.pop(tx.STRAT_CONFIG)
        args = SimpleNamespace(params=str(tmp_path / "vector.json"), require_empty_out=False)
        tx._write_metadata(
            tmp_path,
            args,
            tmp_path / "project.toml",
            tmp_path / "c.swim",
            tmp_path / "e3results",
            dict(IRRIGATED_VEC),
            fids,
            daily,
            monthly,
            pd.Index(fids),
            pd.Index(fids),
            False,
            None,
            None,
        )
        meta = json.loads((tmp_path / "run_metadata.json").read_text())
        assert meta["stratified_comparator_included"] is False
        assert "stratified_transfer" not in meta
        assert meta["require_empty_out"] is False

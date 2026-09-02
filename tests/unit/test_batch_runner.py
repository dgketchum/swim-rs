"""Tests for swimrs.calibrate.batch_support and batch_runner modules."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from swimrs.calibrate.batch_support import (
    batch_is_built,
    coerce_fid,
    create_run_manifest,
    find_par_csv,
    get_uncovered_fids,
    load_batches_from_manifest,
    parse_nan_fids,
    partition_fields,
    read_batch_log,
    update_batch_entry,
    write_batch_log,
    write_manifest,
)

# ---------------------------------------------------------------------------
# FID coercion
# ---------------------------------------------------------------------------


class TestCoerceFid:
    """Tests for coerce_fid: normalizes field IDs to clean strings."""

    def test_integer_passthrough(self):
        assert coerce_fid(42) == "42"

    def test_float_upcasting(self):
        """pandas often upcasts int -> float; coerce_fid should strip .0"""
        assert coerce_fid(1.0) == "1"
        assert coerce_fid("1.0") == "1"

    def test_string_id_preserved(self):
        """String IDs like flux site codes must not be mangled."""
        assert coerce_fid("US-FPe") == "US-FPe"
        assert coerce_fid("AU-Ade") == "AU-Ade"

    def test_delimited_id_preserved(self):
        """Underscore-delimited numeric IDs must not be interpreted as float."""
        assert coerce_fid("001_000001") == "001_000001"

    def test_large_integer(self):
        assert coerce_fid(123456) == "123456"
        assert coerce_fid("123456.0") == "123456"

    def test_negative_number(self):
        """Negative numbers are not purely digit strings, preserved as-is."""
        assert coerce_fid("-1") == "-1"

    def test_zero(self):
        assert coerce_fid(0) == "0"
        assert coerce_fid("0.0") == "0"


# ---------------------------------------------------------------------------
# Batch log I/O
# ---------------------------------------------------------------------------


class TestBatchLog:
    """Tests for atomic batch_log.json I/O."""

    def test_empty_log_when_missing(self, tmp_path):
        assert read_batch_log(tmp_path) == {}

    def test_write_and_read_roundtrip(self, tmp_path):
        data = {"0": {"status": "built"}, "1": {"status": "ingested"}}
        write_batch_log(tmp_path, data)
        assert read_batch_log(tmp_path) == data

    def test_update_entry(self, tmp_path):
        write_batch_log(tmp_path, {"0": {"status": "built"}})
        update_batch_entry(tmp_path, 1, {"status": "ingested", "n_fields": 10})
        log = read_batch_log(tmp_path)
        assert log["0"]["status"] == "built"
        assert log["1"]["status"] == "ingested"
        assert log["1"]["n_fields"] == 10

    def test_overwrite_entry(self, tmp_path):
        write_batch_log(tmp_path, {"0": {"status": "built"}})
        update_batch_entry(tmp_path, 0, {"status": "run_failed", "error": "boom"})
        log = read_batch_log(tmp_path)
        assert log["0"]["status"] == "run_failed"


# ---------------------------------------------------------------------------
# Coverage detection
# ---------------------------------------------------------------------------


class TestGetUncoveredFids:
    """Tests for config-driven uncovered field detection."""

    def _make_mock_container(self, field_uids, arrays):
        """Create a mock container with given field UIDs and arrays.

        arrays: dict mapping path -> 2D numpy array (time x fields)
        """
        container = MagicMock()
        container._field_uids = field_uids

        class FakeRoot:
            def __init__(self, arrays):
                self._arrays = arrays

            def __contains__(self, key):
                return key in self._arrays

            def __getitem__(self, key):
                arr = self._arrays[key]
                mock_arr = MagicMock()
                mock_arr.__getitem__ = lambda self_, sl: arr
                return mock_arr

        container._root = FakeRoot(arrays)
        return container

    def test_no_mask_ssebop_all_covered(self):
        """All fields have observations -> no uncovered."""
        field_uids = ["US-FPe", "US-Var", "AU-Ade"]
        data = np.random.rand(100, 3)
        container = self._make_mock_container(
            field_uids,
            {
                "remote_sensing/ndvi/landsat/no_mask": data,
                "remote_sensing/etf/landsat/ssebop/no_mask": data,
            },
        )
        result = get_uncovered_fids(container, "ssebop", "none")
        assert result["all"] == []
        assert result["ndvi"] == []
        assert result["etf"] == []

    def test_no_mask_one_field_uncovered(self):
        """One field has all NaN -> should be in uncovered."""
        field_uids = ["A", "B", "C"]
        ndvi = np.random.rand(50, 3)
        etf = np.random.rand(50, 3)
        etf[:, 1] = np.nan  # Field "B" has no ETf observations
        container = self._make_mock_container(
            field_uids,
            {
                "remote_sensing/ndvi/landsat/no_mask": ndvi,
                "remote_sensing/etf/landsat/ssebop/no_mask": etf,
            },
        )
        result = get_uncovered_fids(container, "ssebop", "none")
        assert "B" in result["etf"]
        assert "B" in result["all"]
        assert "A" not in result["etf"]

    def test_irrigation_mask_union(self):
        """With irrigation mask_mode, checks irr + inv_irr union."""
        field_uids = ["X", "Y"]
        irr = np.full((10, 2), np.nan)
        irr[:, 0] = 1.0  # X has obs in irr
        inv_irr = np.full((10, 2), np.nan)
        inv_irr[:, 1] = 1.0  # Y has obs in inv_irr
        container = self._make_mock_container(
            field_uids,
            {
                "remote_sensing/ndvi/landsat/irr": irr,
                "remote_sensing/ndvi/landsat/inv_irr": inv_irr,
                "remote_sensing/etf/landsat/ssebop/irr": irr,
                "remote_sensing/etf/landsat/ssebop/inv_irr": inv_irr,
            },
        )
        result = get_uncovered_fids(container, "ssebop", "irrigation")
        # Both fields have observations via the union of masks
        assert result["all"] == []

    def test_ensemble_paths(self):
        """Ensemble ETf model checks each member path."""
        field_uids = ["A"]
        data = np.random.rand(10, 1)
        container = self._make_mock_container(
            field_uids,
            {
                "remote_sensing/ndvi/landsat/no_mask": data,
                "remote_sensing/etf/landsat/ssebop/no_mask": data,
                "remote_sensing/etf/landsat/ptjpl/no_mask": data,
            },
        )
        result = get_uncovered_fids(
            container,
            "ensemble",
            "none",
            etf_ensemble_members=["ssebop", "ptjpl"],
        )
        assert result["all"] == []

    def test_checked_paths_recorded(self):
        """Coverage detection records which paths were actually checked."""
        field_uids = ["A"]
        data = np.random.rand(10, 1)
        container = self._make_mock_container(
            field_uids,
            {
                "remote_sensing/ndvi/landsat/no_mask": data,
                "remote_sensing/etf/landsat/ssebop/no_mask": data,
            },
        )
        result = get_uncovered_fids(container, "ssebop", "none")
        assert "remote_sensing/etf/landsat/ssebop/no_mask" in result["_checked_paths"]["etf"]


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


class TestPartitionFields:
    """Tests for config-driven field partitioning."""

    @pytest.fixture
    def simple_shapefile(self, tmp_path):
        """Create a minimal shapefile with site_id and optional GFID."""
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {
                "site_id": [f"S{i}" for i in range(10)],
                "GFID": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
                "geometry": [Point(i, i) for i in range(10)],
            }
        )
        shp_path = tmp_path / "fields.shp"
        gdf.to_file(shp_path, engine="fiona")
        return shp_path

    def test_sequential_packing(self, simple_shapefile):
        """Without grouping, fields are packed sequentially."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=4,
            grouping_column=None,
        )
        assert len(batches) == 3  # 10 fields / 4 = 3 batches (4, 4, 2)
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 2

    def test_grouped_packing(self, simple_shapefile):
        """With GFID grouping, fields from the same group stay together."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=5,
            grouping_column="GFID",
        )
        # GFID groups: 0=[S0,S1,S2], 1=[S3,S4,S5], 2=[S6,S7], 3=[S8,S9]
        # Batch 0: group 0 (3) -> add group 1 (3) -> 6 > 5 -> flush -> [S0,S1,S2]
        # Batch 1: group 1 (3) -> add group 2 (2) -> 5 = 5 -> [S3,S4,S5,S6,S7]
        # Batch 2: group 3 (2) -> [S8,S9]
        assert len(batches) == 3
        # All FIDs from the same GFID must be in the same batch
        flat = [fid for batch in batches for fid in batch]
        assert sorted(flat) == sorted([f"S{i}" for i in range(10)])

    def test_exclude_fids(self, simple_shapefile):
        """Excluded FIDs are removed before partitioning."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=10,
            exclude_fids={"S0", "S1", "S2"},
        )
        all_fids = [fid for batch in batches for fid in batch]
        assert "S0" not in all_fids
        assert len(all_fids) == 7

    def test_invalid_feature_id_col_raises(self, simple_shapefile):
        """Invalid feature_id_col raises ValueError."""
        with pytest.raises(ValueError, match="Feature ID column"):
            partition_fields(simple_shapefile, "nonexistent_col", batch_size=5)

    def test_missing_grouping_column_falls_back(self, simple_shapefile):
        """Specifying a grouping column not in shapefile falls back to sequential."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=5,
            grouping_column="NONEXISTENT",
        )
        all_fids = [fid for batch in batches for fid in batch]
        assert len(all_fids) == 10

    def test_include_fids_restricts_candidates(self, simple_shapefile):
        """Only FIDs in include_fids are partitioned (container-restricted manifest)."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=10,
            include_fids={"S1", "S3", "S5"},
        )
        all_fids = [fid for batch in batches for fid in batch]
        assert sorted(all_fids) == ["S1", "S3", "S5"]
        assert all(batch for batch in batches)

    def test_include_and_exclude_compose(self, simple_shapefile):
        """A FID must be in include_fids and not in exclude_fids to survive."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=10,
            exclude_fids={"S1"},
            include_fids={"S1", "S3", "S5"},
        )
        all_fids = [fid for batch in batches for fid in batch]
        assert sorted(all_fids) == ["S3", "S5"]

    def test_include_fids_grouped(self, simple_shapefile):
        """include_fids also applies under GFID-grouped packing."""
        batches = partition_fields(
            simple_shapefile,
            "site_id",
            batch_size=5,
            grouping_column="GFID",
            include_fids={"S0", "S1", "S8"},
        )
        all_fids = [fid for batch in batches for fid in batch]
        assert sorted(all_fids) == ["S0", "S1", "S8"]


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


class TestManifest:
    """Tests for batch manifest read/write."""

    def test_write_and_read_roundtrip(self, tmp_path):
        batches = [["A", "B", "C"], ["D", "E"]]
        write_manifest(tmp_path, batches, feature_id_col="site_id")
        loaded = load_batches_from_manifest(tmp_path, "site_id")
        assert len(loaded) == 2
        assert loaded[0] == (0, ["A", "B", "C"])
        assert loaded[1] == (1, ["D", "E"])

    def test_fallback_fid_column(self, tmp_path):
        """When feature_id_col not in manifest, falls back to 'FID'."""
        batches = [["A", "B"]]
        # Write with FID column
        write_manifest(tmp_path, batches, feature_id_col="FID")
        # Read with a different column name -> should fall back
        loaded = load_batches_from_manifest(tmp_path, "site_id")
        assert loaded[0] == (0, ["A", "B"])

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_batches_from_manifest(tmp_path / "nonexistent", "FID")


# ---------------------------------------------------------------------------
# NaN FID parsing
# ---------------------------------------------------------------------------


class TestParseNanFids:
    """Tests for parse_nan_fids: extracts bad FIDs from error messages."""

    def test_parses_fids(self):
        msg = "NaN state in 3 field(s): ['US-FPe', 'US-Var', 'AU-Ade']"
        fids, n_expected = parse_nan_fids(msg)
        assert fids == ["US-FPe", "US-Var", "AU-Ade"]
        assert n_expected == 3

    def test_truncated_list(self):
        msg = "NaN state in 5 field(s): ['A', 'B', ...]"
        fids, n_expected = parse_nan_fids(msg)
        assert len(fids) == 2
        assert n_expected == 5

    def test_numeric_fids(self):
        msg = "NaN state in 2 field(s): ['1416', '1417']"
        fids, n_expected = parse_nan_fids(msg)
        assert fids == ["1416", "1417"]
        assert n_expected == 2

    def test_no_match(self):
        msg = "Something completely different went wrong"
        fids, n_expected = parse_nan_fids(msg)
        assert fids == []
        assert n_expected == 0


# ---------------------------------------------------------------------------
# Find par CSV / batch_is_built
# ---------------------------------------------------------------------------


class TestFindParCsv:
    def test_finds_latest(self, tmp_path):
        master = tmp_path / "master"
        master.mkdir()
        (master / "project.0.par.csv").write_text("data")
        (master / "project.3.par.csv").write_text("data")
        (master / "project.1.par.csv").write_text("data")
        result = find_par_csv(tmp_path)
        assert result.name == "project.3.par.csv"

    def test_two_digit_iterations(self, tmp_path):
        """Iteration 10 must be preferred over iteration 9 (not lexicographic)."""
        master = tmp_path / "master"
        master.mkdir()
        (master / "project.9.par.csv").write_text("data")
        (master / "project.10.par.csv").write_text("data")
        (master / "project.2.par.csv").write_text("data")
        result = find_par_csv(tmp_path)
        assert result.name == "project.10.par.csv"

    def test_none_when_empty(self, tmp_path):
        master = tmp_path / "master"
        master.mkdir()
        assert find_par_csv(tmp_path) is None


class TestBatchIsBuilt:
    def test_true_when_pst_exists(self, tmp_path):
        pest = tmp_path / "pest"
        pest.mkdir()
        (pest / "project.pst").write_text("pst")
        assert batch_is_built(tmp_path) is True

    def test_false_when_no_pst(self, tmp_path):
        pest = tmp_path / "pest"
        pest.mkdir()
        assert batch_is_built(tmp_path) is False

    def test_false_when_no_pest_dir(self, tmp_path):
        assert batch_is_built(tmp_path) is False


# ---------------------------------------------------------------------------
# Best-phi iteration selection (C-3)
# ---------------------------------------------------------------------------


def _write_phi_meas(master, project, rows):
    """Write a minimal {project}.phi.meas.csv with (iteration, mean) rows."""
    lines = ["iteration,total_runs,mean,standard_deviation,min,max"]
    for it, mean in rows:
        lines.append(f"{it},50,{mean},1.0,{mean - 1},{mean + 1}")
    (master / f"{project}.phi.meas.csv").write_text("\n".join(lines))


class TestFindParCsvBestPhi:
    def test_prefers_min_phi_iteration(self, tmp_path):
        """The last iteration is not always the best: pick min mean phi."""
        master = tmp_path / "master"
        master.mkdir()
        for i in range(4):
            (master / f"project.{i}.par.csv").write_text("data")
        # phi dips at iteration 2 then degrades at 3
        _write_phi_meas(master, "project", [(0, 5000.0), (1, 1500.0), (2, 900.0), (3, 1100.0)])
        result = find_par_csv(tmp_path)
        assert result.name == "project.2.par.csv"

    def test_falls_back_to_latest_without_phi(self, tmp_path):
        master = tmp_path / "master"
        master.mkdir()
        (master / "project.0.par.csv").write_text("data")
        (master / "project.3.par.csv").write_text("data")
        result = find_par_csv(tmp_path)
        assert result.name == "project.3.par.csv"

    def test_falls_back_when_phi_malformed(self, tmp_path):
        master = tmp_path / "master"
        master.mkdir()
        (master / "project.0.par.csv").write_text("data")
        (master / "project.2.par.csv").write_text("data")
        (master / "project.phi.meas.csv").write_text("not,a,phi\nfile,at,all")
        result = find_par_csv(tmp_path)
        assert result.name == "project.2.par.csv"

    def test_falls_back_when_best_iter_par_missing(self, tmp_path):
        """Min-phi iteration without a matching .par.csv falls back to latest."""
        master = tmp_path / "master"
        master.mkdir()
        (master / "project.0.par.csv").write_text("data")
        (master / "project.3.par.csv").write_text("data")
        _write_phi_meas(master, "project", [(0, 5000.0), (1, 800.0), (3, 1100.0)])
        result = find_par_csv(tmp_path)
        assert result.name == "project.3.par.csv"


class TestBestPhiIteration:
    def test_min_mean_selected(self, tmp_path):
        from swimrs.calibrate.batch_support import _best_phi_iteration

        master = tmp_path
        _write_phi_meas(master, "p", [(0, 100.0), (1, 40.0), (2, 60.0)])
        assert _best_phi_iteration(master) == 1

    def test_none_without_file(self, tmp_path):
        from swimrs.calibrate.batch_support import _best_phi_iteration

        assert _best_phi_iteration(tmp_path) is None


# ---------------------------------------------------------------------------
# PEST++ output archiving (C-1)
# ---------------------------------------------------------------------------


class TestArchivePestOutputs:
    def _make_batch(self, tmp_path):
        batch = tmp_path / "batch_000"
        pest = batch / "pest"
        master = batch / "master"
        pest.mkdir(parents=True)
        master.mkdir(parents=True)
        (pest / "project.pst").write_text("pst")
        (pest / "loc.mat").write_text("loc")
        (pest / "weight_audit.csv").write_text("audit")
        (master / "project.rec").write_text("rec")
        (master / "project.phi.meas.csv").write_text("phi")
        for i in range(4):
            (master / f"project.{i}.par.csv").write_text(f"par{i}")
            (master / f"project.{i}.obs.csv").write_text(f"obs{i}")
        # version-2 pst external tables (needed to reload the .pst with pyemu)
        (pest / "project.obs_data.csv").write_text("obsdata")
        (pest / "project.par_data.csv").write_text("pardata")
        (master / "project.obs+noise.csv").write_text("noise")
        (master / "project.base.rei").write_text("rei")
        # noise that must NOT be archived
        (master / "project.jcb").write_text("big binary")
        return batch

    def test_archives_full_trajectory(self, tmp_path):
        from swimrs.calibrate.batch_support import archive_pest_outputs

        batch = self._make_batch(tmp_path)
        archive = tmp_path / "pest_archive" / "batch_000"
        copied = archive_pest_outputs(batch, archive)

        # All iterations of par and obs survive (RUN_POLICY Category 4)
        for i in range(4):
            assert (archive / f"project.{i}.par.csv").exists()
            assert (archive / f"project.{i}.obs.csv").exists()
        assert (archive / "project.pst").exists()
        assert (archive / "project.rec").exists()
        assert (archive / "project.phi.meas.csv").exists()
        assert (archive / "loc.mat").exists()
        # external tables + noise ensemble + prior-mean residuals survive
        assert (archive / "project.obs_data.csv").exists()
        assert (archive / "project.par_data.csv").exists()
        assert (archive / "project.obs+noise.csv").exists()
        assert (archive / "project.base.rei").exists()
        # ETf weight decomposition survives cleanup (auxiliary-source runs)
        assert (archive / "weight_audit.csv").exists()
        assert not (archive / "project.jcb").exists()
        assert len(copied) == 17

    def test_verify_ok_after_archive(self, tmp_path):
        from swimrs.calibrate.batch_support import archive_pest_outputs, verify_pest_archive

        batch = self._make_batch(tmp_path)
        archive = tmp_path / "archive"
        archive_pest_outputs(batch, archive)
        ok, missing = verify_pest_archive(archive)
        assert ok
        assert missing == []

    def test_verify_fails_on_missing_dir(self, tmp_path):
        from swimrs.calibrate.batch_support import verify_pest_archive

        ok, missing = verify_pest_archive(tmp_path / "nope")
        assert not ok
        assert missing == ["archive directory"]

    def test_verify_reports_missing_kinds(self, tmp_path):
        from swimrs.calibrate.batch_support import verify_pest_archive

        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "project.pst").write_text("pst")
        ok, missing = verify_pest_archive(archive)
        assert not ok
        assert "rec" in missing
        assert "par.csv" in missing


# ---------------------------------------------------------------------------
# Archive retention tiers (RUN_POLICY Category 4)
# ---------------------------------------------------------------------------


class TestApplyArchiveRetention:
    def _make_archive(self, tmp_path):
        archive = tmp_path / "pest_archive" / "batch_000"
        archive.mkdir(parents=True)
        (archive / "project.pst").write_text("pst")
        (archive / "project.rec").write_text("run record contents")
        (archive / "project.phi.meas.csv").write_text("phi")
        (archive / "project.phi.actual.csv").write_text("phi")
        (archive / "project.pdc.csv").write_text("pdc")
        (archive / "loc.mat").write_text("localizer")
        (archive / "localizer_summary.json").write_text("{}")
        (archive / "params.csv").write_text("params")
        (archive / "weight_audit.csv").write_text("weights")
        (archive / "project.obs_data.csv").write_text("obsdata")
        (archive / "project.adjusted.obs_data.csv").write_text("adjusted")
        (archive / "project.obs+noise.csv").write_text("noise")
        (archive / "project.base.rei").write_text("rei")
        for i in range(4):
            (archive / f"project.{i}.par.csv").write_text(f"par{i}")
            (archive / f"project.{i}.obs.csv").write_text(f"obs{i}")
            (archive / f"project.{i}.base.rei").write_text(f"rei{i}")
        return archive

    def test_full_is_noop(self, tmp_path):
        from swimrs.calibrate.batch_support import apply_archive_retention

        archive = self._make_archive(tmp_path)
        before = sorted(f.name for f in archive.iterdir())
        assert apply_archive_retention(archive, "full") == []
        assert sorted(f.name for f in archive.iterdir()) == before

    def test_reference_keeps_prior_and_final_obs(self, tmp_path):
        from swimrs.calibrate.batch_support import apply_archive_retention

        archive = self._make_archive(tmp_path)
        removed = apply_archive_retention(archive, "reference")

        assert (archive / "project.0.obs.csv").exists()
        assert (archive / "project.3.obs.csv").exists()
        assert not (archive / "project.1.obs.csv").exists()
        assert not (archive / "project.2.obs.csv").exists()
        # only the final-iteration residuals survive
        assert (archive / "project.3.base.rei").exists()
        assert (archive / "project.base.rei").exists()
        for i in range(3):
            assert not (archive / f"project.{i}.base.rei").exists()
        assert not (archive / "loc.mat").exists()
        # observation-side tables and the run record are untouched
        assert (archive / "project.obs+noise.csv").exists()
        assert (archive / "project.obs_data.csv").exists()
        assert (archive / "project.rec").exists()
        for i in range(4):
            assert (archive / f"project.{i}.par.csv").exists()
        assert sorted(removed) == removed
        assert len(removed) == 6

    def test_slim_drops_obs_side_and_gzips_rec(self, tmp_path):
        import gzip

        from swimrs.calibrate.batch_support import apply_archive_retention

        archive = self._make_archive(tmp_path)
        removed = apply_archive_retention(archive, "slim")

        for i in range(4):
            assert not (archive / f"project.{i}.obs.csv").exists()
            assert not (archive / f"project.{i}.base.rei").exists()
            assert (archive / f"project.{i}.par.csv").exists()
        assert not (archive / "project.base.rei").exists()
        assert not (archive / "project.obs+noise.csv").exists()
        assert not (archive / "project.obs_data.csv").exists()
        assert not (archive / "project.adjusted.obs_data.csv").exists()
        assert not (archive / "loc.mat").exists()
        assert not (archive / "project.rec").exists()
        with gzip.open(archive / "project.rec.gz", "rb") as fh:
            assert fh.read() == b"run record contents"
        # parameter trajectory, phi history, and metadata survive
        assert (archive / "project.pst").exists()
        assert (archive / "project.phi.meas.csv").exists()
        assert (archive / "project.pdc.csv").exists()
        assert (archive / "params.csv").exists()
        assert (archive / "weight_audit.csv").exists()
        assert (archive / "localizer_summary.json").exists()
        assert "project.rec" in removed

    def test_slim_archive_passes_verification(self, tmp_path):
        from swimrs.calibrate.batch_support import apply_archive_retention, verify_pest_archive

        archive = self._make_archive(tmp_path)
        apply_archive_retention(archive, "slim")
        ok, missing = verify_pest_archive(archive)
        assert ok
        assert missing == []

    def test_invalid_tier_raises(self, tmp_path):
        from swimrs.calibrate.batch_support import apply_archive_retention

        archive = self._make_archive(tmp_path)
        with pytest.raises(ValueError, match="archive_retention must be one of"):
            apply_archive_retention(archive, "lean")

    def test_missing_par_iterations_raises(self, tmp_path):
        from swimrs.calibrate.batch_support import apply_archive_retention

        archive = tmp_path / "empty_archive"
        archive.mkdir()
        (archive / "project.pst").write_text("pst")
        with pytest.raises(RuntimeError, match="refusing to prune"):
            apply_archive_retention(archive, "slim")


# ---------------------------------------------------------------------------
# Stale-calibration guard (C-2)
# ---------------------------------------------------------------------------


class TestResolveIngestedBatches:
    def test_raises_on_stale_signature(self):
        from swimrs.calibrate.batch_support import resolve_ingested_batches

        with pytest.raises(RuntimeError, match="created by a different run"):
            resolve_ingested_batches({"0", "1"}, {}, False, "/c.swim", "/out")

    def test_override_clears_stale_set(self, capsys):
        from swimrs.calibrate.batch_support import resolve_ingested_batches

        result = resolve_ingested_batches({"0", "1"}, {}, True, "/c.swim", "/out")
        assert result == set()
        assert "WARNING" in capsys.readouterr().out

    def test_trusted_when_batch_log_present(self):
        from swimrs.calibrate.batch_support import resolve_ingested_batches

        log = {"0": {"status": "ingested"}}
        result = resolve_ingested_batches({"0"}, log, False, "/c.swim", "/out")
        assert result == {"0"}

    def test_clean_container_passes(self):
        from swimrs.calibrate.batch_support import resolve_ingested_batches

        assert resolve_ingested_batches(set(), {}, False, "/c.swim", "/out") == set()


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


class TestRunManifest:
    def test_creates_json(self, tmp_path):
        batches = [(0, ["A", "B"]), (1, ["C"])]
        create_run_manifest(
            tmp_path,
            "/fake.swim",
            "/fake.toml",
            None,
            batches,
            noptmax=3,
            reals=200,
            workers=10,
            batch_size=50,
            override=False,
            feature_id_col="site_id",
            grouping_column="GFID",
            mask_mode="none",
            etf_target_model="ssebop",
            project_name="test",
        )
        manifest_path = tmp_path / "run_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["feature_id_column"] == "site_id"
        assert data["mask_mode"] == "none"
        assert data["etf_target_model"] == "ssebop"
        assert data["gate_outcome"] == "SKIPPED"
        assert data["parameters"]["n_batches"] == 2
        assert data["parameters"]["n_fields"] == 3


# ---------------------------------------------------------------------------
# BatchContext resolution
# ---------------------------------------------------------------------------


class TestResolveContext:
    """Tests for resolve_context: builds BatchContext from TOML."""

    @pytest.fixture
    def ex4_toml(self, tmp_path):
        """Create a minimal TOML that mimics Example 4 config."""
        gis_dir = tmp_path / "data" / "gis"
        gis_dir.mkdir(parents=True)
        shp = gis_dir / "flux_fields.shp"
        shp.touch()

        content = f"""
project = "test_project"
root = "{tmp_path}"

[paths]
project_workspace = "{tmp_path}"
data = "{tmp_path / "data"}"
fields_shapefile = "{shp}"
container = "{tmp_path / "data" / "test.swim"}"

[ids]
feature_id = "site_id"

[ids.conus]
gridmet_id = "GFID"

[data_sources]
mask_mode = "none"

[date_range]
start_date = "2000-01-01"
end_date = "2020-12-31"

[calibration]
pest_run_dir = "{tmp_path / "pestrun"}"
etf_target_model = "ssebop"
workers = 20
realizations = 200
calibration_dir = "{tmp_path / "pestrun" / "pest" / "mult"}"
obs_folder = "{tmp_path / "pestrun" / "obs"}"
initial_values_csv = "{tmp_path / "pestrun" / "params.csv"}"
spinup = "{tmp_path / "pestrun" / "spinup.json"}"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(content)
        return toml_path

    def test_resolves_feature_id(self, ex4_toml):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml), batch_size=25)
        assert ctx.feature_id_col == "site_id"

    def test_resolves_etf_model(self, ex4_toml):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml))
        assert ctx.etf_target_model == "ssebop"

    def test_resolves_mask_mode(self, ex4_toml):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml))
        assert ctx.mask_mode == "none"

    def test_resolves_grouping_column(self, ex4_toml):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml))
        assert ctx.grouping_column == "GFID"

    def test_batch_size_override(self, ex4_toml):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml), batch_size=25)
        assert ctx.batch_size == 25

    def test_workers_override(self, ex4_toml):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml), workers=8)
        assert ctx.workers == 8

    def test_container_override(self, ex4_toml, tmp_path):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml), container_override="/custom/path.swim")
        assert ctx.container_path == "/custom/path.swim"

    def test_output_override(self, ex4_toml, tmp_path):
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml), output_override="/tmp/output")
        assert ctx.output_root == "/tmp/output"

    def test_grouping_shapefile_none_when_no_mapping(self, ex4_toml):
        """Without gridmet_mapping shapefile, grouping_shapefile is None."""
        from swimrs.calibrate.batch_runner import resolve_context

        ctx = resolve_context(str(ex4_toml))
        # No gridmet_mapping in the test TOML, so grouping_shapefile is None
        assert ctx.grouping_shapefile is None

    def test_grouping_shapefile_set_when_mapping_exists(self, ex4_toml, tmp_path):
        """When gridmet_mapping shapefile exists, it becomes grouping_shapefile."""
        from swimrs.calibrate.batch_runner import resolve_context

        # Create a mapping shapefile and add it to the TOML
        gis_dir = tmp_path / "data" / "gis"
        mapping_shp = gis_dir / "flux_fields_gfid.shp"
        mapping_shp.touch()

        content = ex4_toml.read_text()
        content += f'\n[paths.conus]\ngridmet_mapping = "{mapping_shp}"\n'
        ex4_toml.write_text(content)

        ctx = resolve_context(str(ex4_toml))
        assert ctx.grouping_shapefile == str(mapping_shp)

    def test_shapefile_override_disables_grouping_shapefile(self, ex4_toml, tmp_path):
        """CLI shapefile override suppresses grouping_shapefile resolution."""
        from swimrs.calibrate.batch_runner import resolve_context

        gis_dir = tmp_path / "data" / "gis"
        mapping_shp = gis_dir / "flux_fields_gfid.shp"
        mapping_shp.touch()

        content = ex4_toml.read_text()
        content += f'\n[paths.conus]\ngridmet_mapping = "{mapping_shp}"\n'
        ex4_toml.write_text(content)

        ctx = resolve_context(str(ex4_toml), shapefile_override="/custom/fields.shp")
        assert ctx.grouping_shapefile is None


# ========================= EOF ====================================================================

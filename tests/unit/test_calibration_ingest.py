"""Tests for calibration parameter ingestion and container roundtrip.

Covers:
- parse_pest_par_csv(): column parsing, median/mean, name mapping
- Ingestor.calibration(): batch ingestion, incremental updates, metadata
- build_swim_input container fallback: NaN masking, precedence over JSON
"""

import json

import numpy as np
import pandas as pd
import zarr

from swimrs.container.components.ingestor import (
    CALIBRATION_PARAMS,
    parse_pest_par_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_par_csv(tmp_path, fids, params, n_reals=5, name="test.0.par.csv"):
    """Create a synthetic PEST++ .par.csv.

    Parameters
    ----------
    fids : list[str]
    params : dict[str, dict[str, float]]
        {pest_param_name: {fid: center_value}}
    n_reals : int
    """
    columns = []
    for pest_name, fid_vals in params.items():
        for fid in fids:
            if fid in fid_vals:
                col = f"pname:p_{pest_name}_{fid}_ptype:tied_0_:0"
                columns.append((col, fid_vals[fid]))

    # Build DataFrame with n_reals rows + 1 "base" row
    rows = {"base": {c: v for c, v in columns}}
    rng = np.random.default_rng(42)
    for i in range(n_reals):
        rows[str(i)] = {c: v + rng.normal(0, 0.01) for c, v in columns}

    df = pd.DataFrame(rows).T
    path = tmp_path / name
    df.to_csv(path)
    return path


def _make_container_state(tmp_path, fids):
    """Create a minimal ContainerState backed by a directory store."""
    from swimrs.container.state import ContainerState

    store_path = tmp_path / "test.zarr"
    root = zarr.open_group(str(store_path), mode="w")

    # Minimal metadata expected by ContainerState
    root.attrs["field_uids"] = fids
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    root.attrs["time_index"] = [d.isoformat() for d in dates]
    root.attrs["start_date"] = dates[0].isoformat()
    root.attrs["end_date"] = dates[-1].isoformat()

    class FakeStorage:
        def __init__(self, root):
            self.root = root

    class FakeProvenance:
        def record(self, *args, **kwargs):
            return {"action": "test"}

    class FakeInventory:
        def refresh(self):
            pass

    state = ContainerState.__new__(ContainerState)
    state._provider = FakeStorage(root)
    state._field_uids = list(fids)
    state._uid_to_index = {uid: i for i, uid in enumerate(fids)}
    state._time_index = dates
    state._mode = "r+"
    state._modified = False
    state._dataset_cache = None
    state._provenance = FakeProvenance()
    state._inventory = FakeInventory()

    return state


# ---------------------------------------------------------------------------
# parse_pest_par_csv tests
# ---------------------------------------------------------------------------


class TestParsePestParCsv:
    def test_basic_median(self, tmp_path):
        fids = ["100", "200"]
        params = {
            "ndvi_k": {"100": 8.0, "200": 12.0},
            "ks_alpha": {"100": 0.3, "200": 0.5},
        }
        csv = _make_par_csv(tmp_path, fids, params, n_reals=10)
        values, stds = parse_pest_par_csv(csv, fids, summary_stat="median")

        assert "ndvi_k" in values
        assert "ks_damp" in values  # ks_alpha mapped to ks_damp
        assert "ks_alpha" not in values

        # Values should be close to center (noise is small)
        assert abs(values["ndvi_k"]["100"] - 8.0) < 0.1
        assert abs(values["ndvi_k"]["200"] - 12.0) < 0.1
        assert abs(values["ks_damp"]["100"] - 0.3) < 0.1

        # Stds should be small but positive
        assert stds["ndvi_k"]["100"] > 0
        assert stds["ndvi_k"]["100"] < 0.1

    def test_mean_stat(self, tmp_path):
        fids = ["100"]
        params = {"ndvi_k": {"100": 10.0}}
        csv = _make_par_csv(tmp_path, fids, params, n_reals=20)
        values, _ = parse_pest_par_csv(csv, fids, summary_stat="mean")
        assert abs(values["ndvi_k"]["100"] - 10.0) < 0.1

    def test_unknown_param_ignored(self, tmp_path):
        fids = ["100"]
        params = {"unknown_param": {"100": 1.0}}
        csv = _make_par_csv(tmp_path, fids, params)
        values, stds = parse_pest_par_csv(csv, fids)
        assert len(values) == 0

    def test_all_params(self, tmp_path):
        fids = ["1"]
        # Both NDVI-Kcb curve parameterizations. A real run carries only one
        # (the modes are exclusive), but the parser must recognize either.
        params = {
            "ndvi_k": {"1": 10.0},
            "ndvi_0": {"1": 0.5},
            "ndvi_alpha": {"1": 0.2},
            "ndvi_beta": {"1": 1.25},
            "swe_alpha": {"1": 0.4},
            "swe_beta": {"1": 2.5},
            "ks_alpha": {"1": 0.2},
            "kr_alpha": {"1": 0.15},
            "aw": {"1": 150.0},
            "mad": {"1": 0.45},
        }
        csv = _make_par_csv(tmp_path, fids, params)
        values, _ = parse_pest_par_csv(csv, fids)
        assert set(values.keys()) == set(CALIBRATION_PARAMS)


# ---------------------------------------------------------------------------
# Ingestor.calibration tests
# ---------------------------------------------------------------------------


class TestIngestorCalibration:
    def test_single_batch(self, tmp_path):
        fids = ["100", "200", "300"]
        state = _make_container_state(tmp_path, fids)

        from swimrs.container.components.ingestor import Ingestor

        ingestor = Ingestor(state)

        params = {"ndvi_k": {"100": 8.0, "200": 12.0}, "aw": {"100": 150.0}}
        csv = _make_par_csv(tmp_path, fids, params)

        ingestor.calibration(csv, batch_id=0)

        root = state.root
        # Parameters written
        assert "calibration/parameters/ndvi_k" in root
        ndvi_k = np.asarray(root["calibration/parameters/ndvi_k"][:])
        assert abs(ndvi_k[0] - 8.0) < 0.1  # field 100
        assert abs(ndvi_k[1] - 12.0) < 0.1  # field 200
        assert np.isnan(ndvi_k[2])  # field 300 — not in batch

        # Uncertainty written
        assert "calibration/uncertainty/ndvi_k" in root
        std = np.asarray(root["calibration/uncertainty/ndvi_k"][:])
        assert std[0] > 0
        assert np.isnan(std[2])

        # Metadata
        cal = np.asarray(root["calibration/metadata/calibrated"][:])
        assert cal[0] == 1
        assert cal[1] == 1
        assert cal[2] == 0

        batch = np.asarray(root["calibration/metadata/batch_id"][:])
        assert batch[0] == 0
        assert batch[1] == 0
        assert batch[2] == -1

        # Group attrs
        attrs = dict(root["calibration"].attrs)
        assert attrs["n_fields_calibrated"] == 2
        batches = json.loads(attrs["batches"])
        assert "0" in batches

    def test_incremental_batches(self, tmp_path):
        fids = ["A", "B", "C", "D"]
        state = _make_container_state(tmp_path, fids)
        from swimrs.container.components.ingestor import Ingestor

        ingestor = Ingestor(state)

        # Batch 0: fields A, B
        csv1 = _make_par_csv(
            tmp_path,
            ["A", "B"],
            {"ndvi_k": {"A": 5.0, "B": 6.0}},
            name="batch0.par.csv",
        )
        ingestor.calibration(csv1, fields=["A", "B"], batch_id=0)

        # Batch 1: fields C, D
        csv2 = _make_par_csv(
            tmp_path,
            ["C", "D"],
            {"ndvi_k": {"C": 7.0, "D": 8.0}},
            name="batch1.par.csv",
        )
        ingestor.calibration(csv2, fields=["C", "D"], batch_id=1)

        root = state.root
        ndvi_k = np.asarray(root["calibration/parameters/ndvi_k"][:])
        assert abs(ndvi_k[0] - 5.0) < 0.1  # A
        assert abs(ndvi_k[1] - 6.0) < 0.1  # B
        assert abs(ndvi_k[2] - 7.0) < 0.1  # C
        assert abs(ndvi_k[3] - 8.0) < 0.1  # D

        cal = np.asarray(root["calibration/metadata/calibrated"][:])
        assert all(cal == 1)

        attrs = dict(root["calibration"].attrs)
        assert attrs["n_fields_calibrated"] == 4
        batches = json.loads(attrs["batches"])
        assert len(batches) == 2


# ---------------------------------------------------------------------------
# Container fallback in build_swim_input
# ---------------------------------------------------------------------------


class TestContainerCalibrationHelpers:
    def test_container_has_calibration_false(self, tmp_path):
        """No calibration group → returns False."""
        root = zarr.open_group(str(tmp_path / "empty.zarr"), mode="w")

        class FakeContainer:
            _root = root

        from swimrs.process.input import _container_has_calibration

        assert _container_has_calibration(FakeContainer()) is False

    def test_container_has_calibration_true(self, tmp_path):
        root = zarr.open_group(str(tmp_path / "cal.zarr"), mode="w")
        cal = root.create_group("calibration")
        cal.create_group("parameters")

        class FakeContainer:
            _root = root

        from swimrs.process.input import _container_has_calibration

        assert _container_has_calibration(FakeContainer()) is True

    def test_load_calibrated_from_container(self, tmp_path):
        fids = ["10", "20", "30"]
        root = zarr.open_group(str(tmp_path / "cal2.zarr"), mode="w")
        params_grp = root.require_group("calibration/parameters")
        arr = params_grp.create_array("ndvi_k", shape=(3,), dtype="float64", fill_value=np.nan)
        arr[0] = 8.0
        arr[1] = 12.0
        # arr[2] stays NaN

        class FakeContainer:
            _root = root
            _field_uids = fids
            field_uids = fids

        from swimrs.process.input import _load_calibrated_from_container

        result = _load_calibrated_from_container(FakeContainer(), fids)
        assert "ndvi_k" in result
        assert result["ndvi_k"][0] == 8.0
        assert result["ndvi_k"][1] == 12.0
        assert np.isnan(result["ndvi_k"][2])

    def test_load_calibrated_subset(self, tmp_path):
        """Request only a subset of container fields."""
        all_fids = ["10", "20", "30"]
        root = zarr.open_group(str(tmp_path / "cal3.zarr"), mode="w")
        params_grp = root.require_group("calibration/parameters")
        arr = params_grp.create_array("ks_damp", shape=(3,), dtype="float64", fill_value=np.nan)
        arr[0] = 0.3
        arr[1] = 0.5
        arr[2] = 0.7

        class FakeContainer:
            _root = root
            _field_uids = all_fids
            field_uids = all_fids

        from swimrs.process.input import _load_calibrated_from_container

        # Request only fields 20 and 30
        result = _load_calibrated_from_container(FakeContainer(), ["20", "30"])
        assert result["ks_damp"][0] == 0.5  # "20" → index 0 in output
        assert result["ks_damp"][1] == 0.7  # "30" → index 1 in output

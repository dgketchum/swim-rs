import json
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from swimrs.process.input import (
    SwimInput,
    _load_calibrated_params,
    _load_spinup_json,
    _write_prescribed_irrigation,
)


def test_load_spinup_json_defaults_and_optional_arrays(tmp_path):
    fids = ["A", "B"]
    spinup = {
        "A": {"depl_root": 10.0, "zr": 0.25, "irr_frac_root": 0.9},
        # B omitted -> defaults should apply
    }
    path = tmp_path / "spinup.json"
    path.write_text(json.dumps(spinup))

    state = _load_spinup_json(path, fids=fids)

    assert np.allclose(state["depl_root"], np.array([10.0, 0.0]))
    assert np.allclose(state["zr"], np.array([0.25, 0.1]))
    assert np.allclose(state["albedo"], np.array([0.45, 0.45]))
    assert np.allclose(state["s"], np.array([84.7, 84.7]))
    assert "irr_frac_root" in state
    assert np.allclose(state["irr_frac_root"], np.array([0.9, 0.0]))
    assert "irr_frac_l3" not in state


def test_load_calibrated_params_is_case_insensitive_on_field_ids(tmp_path):
    fids = ["S2", "Other"]
    params = {
        "s2": {"ks_alpha": 0.7, "ndvi_k": 5.5, "f_sub": 0.2},
        "OTHER": {"ndvi_0": 0.42, "mad": 0.33},
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    out = _load_calibrated_params(path, fids=fids)

    assert out["ks_damp"][0] == 0.7
    assert np.isnan(out["ks_damp"][1])
    assert out["ndvi_k"][0] == 5.5
    assert np.isnan(out["ndvi_k"][1])

    assert np.isnan(out["ndvi_0"][0])
    assert out["ndvi_0"][1] == 0.42
    assert np.isnan(out["mad"][0])
    assert out["mad"][1] == 0.33

    # f_sub is never a PEST parameter; it must fall through to the container
    assert "f_sub" not in out


def test_load_spinup_json_extra_field_ids_ignored(tmp_path):
    fids = ["A"]
    spinup = {
        "A": {"depl_root": 5.0},
        "EXTRA_FIELD": {"depl_root": 99.0},
    }
    path = tmp_path / "spinup.json"
    path.write_text(json.dumps(spinup))

    state = _load_spinup_json(path, fids=fids)
    assert state["depl_root"].shape == (1,)
    assert np.allclose(state["depl_root"], np.array([5.0]))


def test_load_spinup_json_empty_json_gives_defaults(tmp_path):
    fids = ["A", "B"]
    path = tmp_path / "spinup.json"
    path.write_text("{}")

    state = _load_spinup_json(path, fids=fids)
    assert np.allclose(state["depl_root"], np.array([0.0, 0.0]))
    assert np.allclose(state["zr"], np.array([0.1, 0.1]))
    assert np.allclose(state["albedo"], np.array([0.45, 0.45]))
    assert np.allclose(state["s"], np.array([84.7, 84.7]))
    assert np.allclose(state["kr"], np.array([1.0, 1.0]))
    assert np.allclose(state["ks"], np.array([1.0, 1.0]))


def test_load_calibrated_params_extra_param_names_ignored(tmp_path):
    fids = ["X"]
    params = {"X": {"ks_alpha": 0.5, "bogus_param": 99.0}}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    out = _load_calibrated_params(path, fids=fids)
    assert np.allclose(out["ks_damp"], np.array([0.5]))
    assert "bogus_param" not in out


def test_load_calibrated_params_zero_vs_absent(tmp_path):
    fids = ["A", "B"]
    params = {
        "A": {"ks_alpha": 0.0},
        # B absent entirely
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    out = _load_calibrated_params(path, fids=fids)
    # Explicit 0.0 is kept; absent means "not calibrated" and stays NaN so the
    # masked assignment in the callers preserves container/default values
    assert out["ks_damp"][0] == 0.0
    assert np.isnan(out["ks_damp"][1])


# --- WP-B0 prescribed-irrigation IO -----------------------------------------


def _read_group(path):
    with h5py.File(path, "r") as h5:
        return h5["prescribed_irrigation/irr_mm"][:]


def test_write_prescribed_irrigation_aligns_dates_and_fields(tmp_path):
    fids = ["A", "B", "C"]
    start = datetime(2020, 6, 1)
    n_days, n_fields = 5, len(fids)
    # Table covers only days 1 & 3 (of 5), fields A and C. B and the deep-store
    # column are absent -> NaN sentinel; day 0/2/4 absent -> NaN sentinel.
    table = pd.DataFrame(
        {"A": [10.0, 30.0], "C": [11.0, 31.0]},
        index=pd.to_datetime(["2020-06-02", "2020-06-04"]),
    )
    src = tmp_path / "presc.parquet"
    table.to_parquet(src)

    out = tmp_path / "run.h5"
    with h5py.File(out, "w") as h5:
        _write_prescribed_irrigation(h5, src, fids, n_fields, n_days, start)

    arr = _read_group(out)
    assert arr.shape == (n_days, n_fields)
    # Day 1 (2020-06-02) and day 3 (2020-06-04) carry values for A(col0), C(col2).
    assert arr[1, 0] == 10.0 and arr[3, 0] == 30.0
    assert arr[1, 2] == 11.0 and arr[3, 2] == 31.0
    # Field B (col1) is entirely absent from the table -> all NaN.
    assert np.isnan(arr[:, 1]).all()
    # Unlisted days are NaN (the "use the scheduler" sentinel).
    assert np.isnan(arr[[0, 2, 4], 0]).all()


def test_write_prescribed_irrigation_accepts_date_column_csv(tmp_path):
    fids = ["A"]
    start = datetime(2021, 1, 1)
    src = tmp_path / "presc.csv"
    pd.DataFrame({"date": ["2021-01-02", "2021-01-03"], "A": [7.0, 8.0]}).to_csv(src, index=False)

    out = tmp_path / "run.h5"
    with h5py.File(out, "w") as h5:
        _write_prescribed_irrigation(h5, src, fids, n_fields=1, n_days=4, start_date=start)

    arr = _read_group(out)
    assert arr.shape == (4, 1)
    assert np.isnan(arr[0, 0])
    assert arr[1, 0] == 7.0 and arr[2, 0] == 8.0
    assert np.isnan(arr[3, 0])


def test_write_prescribed_irrigation_missing_file_raises(tmp_path):
    with h5py.File(tmp_path / "run.h5", "w") as h5:
        with pytest.raises(FileNotFoundError):
            _write_prescribed_irrigation(
                h5, tmp_path / "nope.parquet", ["A"], 1, 3, datetime(2020, 1, 1)
            )


def test_get_prescribed_irr_round_trip_and_absent(tmp_path):
    """SwimInput.get_prescribed_irr reads what the writer wrote, else None."""
    fids = ["A"]
    src = tmp_path / "presc.csv"
    pd.DataFrame({"date": ["2020-06-02"], "A": [9.0]}).to_csv(src, index=False)

    with_grp = tmp_path / "with.h5"
    with h5py.File(with_grp, "w") as h5:
        _write_prescribed_irrigation(h5, src, fids, 1, 3, datetime(2020, 6, 1))

    class _Stub:
        def __init__(self, h5f):
            self._h5_file = h5f

    with h5py.File(with_grp, "r") as h5:
        arr = SwimInput.get_prescribed_irr(_Stub(h5))
    assert arr is not None and arr.shape == (3, 1)
    assert arr[1, 0] == 9.0 and np.isnan(arr[[0, 2], 0]).all()

    without = tmp_path / "without.h5"
    with h5py.File(without, "w") as h5:
        h5.create_dataset("irrigation/irr_flag", data=np.zeros((3, 1)))
    with h5py.File(without, "r") as h5:
        assert SwimInput.get_prescribed_irr(_Stub(h5)) is None

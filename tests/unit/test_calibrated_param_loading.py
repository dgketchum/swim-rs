"""Regression tests for calibrated-parameter loading (review finding A1).

The JSON loader (`_load_calibrated_params`) must mirror the container loader
(`_load_calibrated_from_container`): a parameter absent from the calibration
source stays NaN, so the masked assignment in the property writers preserves
container/default values. The old zero-initialized loader silently overwrote
container-derived f_sub with 0.0 at every field.
"""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from swimrs.process.input import (
    _load_calibrated_from_container,
    _load_calibrated_params,
    _write_properties_from_container,
    build_swim_input,
)

GOLDEN_CONTAINER = Path(__file__).parents[1] / "fixtures" / "golden_loop" / "fort_peck.swim"

PEST_PARAMS = {
    "ks_alpha": 0.6,
    "kr_alpha": 0.4,
    "ndvi_k": 6.0,
    "ndvi_0": 0.35,
    "swe_alpha": 0.1,
    "swe_beta": 1.2,
    "aw": 150.0,
    "mad": 0.45,
}


def test_json_loader_excludes_f_sub_and_nans_missing_fields(tmp_path):
    fids = ["A", "B"]
    params = {"A": dict(PEST_PARAMS)}  # B absent entirely; no f_sub anywhere
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    out = _load_calibrated_params(path, fids=fids)

    assert "f_sub" not in out
    for internal in [
        "ks_damp",
        "kr_damp",
        "ndvi_k",
        "ndvi_0",
        "swe_alpha",
        "swe_beta",
        "aw",
        "mad",
    ]:
        assert np.isfinite(out[internal][0]), internal
        assert np.isnan(out[internal][1]), internal


def test_json_and_container_loaders_agree_on_missing_params(tmp_path):
    """Both loaders return NaN for anything the calibration never touched."""
    fids = ["A", "B"]

    params = {"A": {"ndvi_k": 6.0}, "B": {"ndvi_k": 7.0}}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    json_out = _load_calibrated_params(path, fids=fids)

    root = zarr.open_group(str(tmp_path / "cal.zarr"), mode="w")
    grp = root.require_group("calibration/parameters")
    arr = grp.create_array("ndvi_k", shape=(2,), dtype="float64", fill_value=np.nan)
    arr[0] = 6.0
    arr[1] = 7.0

    class FakeContainer:
        _root = root
        field_uids = fids

    cont_out = _load_calibrated_from_container(FakeContainer(), fids)

    assert np.allclose(json_out["ndvi_k"], cont_out["ndvi_k"])
    # ks_damp was never calibrated: NaN (json) / absent (container) — both
    # leave the caller's masked assignment untouched
    assert np.all(np.isnan(json_out["ks_damp"]))
    assert "ks_damp" not in cont_out


def test_properties_writer_preserves_container_f_sub(tmp_path):
    """A gw-subsidy field keeps its container-derived f_sub through the JSON
    calibrated-params path (the A1 regression)."""
    fids = ["GW", "DRY"]
    params = {fid: dict(PEST_PARAMS) for fid in fids}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    calibrated = _load_calibrated_params(path, fids=fids)

    container_data = {
        "props": {fid: {"awc": 0.15, "clay": 20.0} for fid in fids},
        "dynamics": {
            "gwsub": {"GW": {"2018": {"f_sub": 0.35}, "2019": {"f_sub": 0.45}}},
        },
    }

    h5_path = tmp_path / "props.h5"
    with h5py.File(h5_path, "w") as h5:
        _write_properties_from_container(h5, container_data, fids, len(fids), calibrated)

    with h5py.File(h5_path, "r") as h5:
        f_sub = h5["properties/f_sub"][:]
        gw_status = h5["properties/gw_status"][:]
        awc = h5["properties/awc"][:]

    assert gw_status[0] == 1 and gw_status[1] == 0
    assert np.isclose(f_sub[0], 0.4)  # mean of 0.35, 0.45 — NOT zeroed
    assert f_sub[1] == 0.0
    # calibrated params present in the JSON still override the container
    assert np.allclose(awc, PEST_PARAMS["aw"])


@pytest.mark.skipif(not GOLDEN_CONTAINER.exists(), reason="golden-loop fixture not available")
def test_use_container_calibration_gate(tmp_path):
    """A stale calibration/ group must not contaminate a fresh PEST base run
    (review finding C-2): use_container_calibration=False ignores it."""
    from swimrs.container import SwimContainer

    work = tmp_path / "fort_peck.swim"
    shutil.copytree(GOLDEN_CONTAINER, work)

    container = SwimContainer.open(str(work), mode="r")
    n_fields = len(container.field_uids)
    container.close()

    # Inject a calibration group with a distinctive ndvi_k, as a copied
    # calibrated container would carry
    root = zarr.open_group(str(work), mode="r+")
    grp = root.require_group("calibration/parameters")
    arr = grp.create_array("ndvi_k", shape=(n_fields,), dtype="float64", fill_value=np.nan)
    arr[:] = 6.5

    kwargs = dict(start_date="2007-01-01", end_date="2008-12-31", etf_model="ptjpl")

    container = SwimContainer.open(str(work), mode="r")
    try:
        swim_input = build_swim_input(container, tmp_path / "with_cal.h5", **kwargs)
        try:
            assert np.allclose(swim_input.parameters.ndvi_k, 6.5)
        finally:
            swim_input.close()

        swim_input = build_swim_input(
            container, tmp_path / "no_cal.h5", use_container_calibration=False, **kwargs
        )
        try:
            # calibration group ignored: ndvi_k stays at its 10.0 default
            assert np.allclose(swim_input.parameters.ndvi_k, 10.0)
        finally:
            swim_input.close()
    finally:
        container.close()

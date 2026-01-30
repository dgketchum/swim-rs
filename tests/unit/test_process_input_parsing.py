import json

import numpy as np

from swimrs.process.input import _load_calibrated_params, _load_spinup_json


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

    assert np.allclose(out["ks_damp"], np.array([0.7, 0.0]))
    assert np.allclose(out["ndvi_k"], np.array([5.5, 0.0]))
    assert np.allclose(out["f_sub"], np.array([0.2, 0.0]))

    assert np.allclose(out["ndvi_0"], np.array([0.0, 0.42]))
    assert np.allclose(out["mad"], np.array([0.0, 0.33]))


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
    # Both result in 0.0
    assert np.allclose(out["ks_damp"], np.array([0.0, 0.0]))

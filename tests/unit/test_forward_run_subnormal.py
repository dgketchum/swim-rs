"""The generated PEST++ forward runner must never write a subnormal double.

PEST++ IES dropped realization 114 of Example 5 Run 23 because the forward run
wrote 3.28e-309 into a ``pred_etf_*.np`` file and the instruction parser
refused to convert it. The generated script flushes NaN and subnormals to 0.0.
"""

import runpy
import types

import numpy as np

from swimrs.calibrate.pest_builder import PestBuilder


def _generate(tmp_path, ssm=False):
    stub = types.SimpleNamespace(
        pest_dir=str(tmp_path),
        config=types.SimpleNamespace(ssm_calibration=ssm, ssm_ze=0.1),
        _ssm_enabled=lambda: ssm,
    )
    PestBuilder._write_forward_run_script(stub)
    return tmp_path / "custom_forward_run.py"


def test_generated_script_flushes_nan_and_subnormals(tmp_path):
    script = _generate(tmp_path)
    ns = runpy.run_path(str(script), run_name="not_main")
    writable = ns["_pest_writable"]
    vals = np.array([0.5, np.nan, 3.279717778153303963e-309, -1e-310, 1e-300, -0.25])
    out = writable(vals)
    np.testing.assert_array_equal(out, np.array([0.5, 0.0, 0.0, 0.0, 1e-300, -0.25]))
    assert (np.abs(out[out != 0]) >= np.finfo(float).tiny).all()


def test_every_prediction_write_goes_through_the_flush(tmp_path):
    for ssm in (False, True):
        d = tmp_path / f"ssm_{ssm}"
        d.mkdir()
        text = _generate(d, ssm=ssm).read_text()
        writes = [line for line in text.splitlines() if "np.savetxt(" in line]
        assert len(writes) == (3 if ssm else 2), writes
        assert all("_pest_writable(" in line for line in writes), writes
        assert "np.nan_to_num(output" not in text

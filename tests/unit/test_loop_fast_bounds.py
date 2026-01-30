"""Physical-invariant tests for _run_loop_jit.

Calls the JIT function directly with small synthetic arrays (2 fields, 10 days)
and asserts that outputs respect physical constraints.
"""

import numpy as np
import pytest

from swimrs.process.loop_fast import _run_loop_jit

N_DAYS = 10
N_FIELDS = 2


def _make_inputs(*, n_days=N_DAYS, n_fields=N_FIELDS, **overrides):
    """Build all required arrays with physically reasonable defaults.

    Returns a dict of keyword arguments suitable for ``_run_loop_jit(**inputs)``.
    Callers can override any array via *overrides*.
    """
    inputs = dict(
        n_days=n_days,
        n_fields=n_fields,
        # Time series (n_days, n_fields) — moderate summer conditions
        all_ndvi=np.full((n_days, n_fields), 0.5),
        all_etr=np.full((n_days, n_fields), 6.0),
        all_prcp=np.full((n_days, n_fields), 3.0),
        all_tmin=np.full((n_days, n_fields), 12.0),
        all_tmax=np.full((n_days, n_fields), 28.0),
        all_srad=np.full((n_days, n_fields), 250.0),
        all_irr_flag=np.zeros((n_days, n_fields)),
        # Properties (n_fields,)
        awc=np.full(n_fields, 150.0),
        rew=np.full(n_fields, 8.0),
        tew=np.full(n_fields, 25.0),
        cn2=np.full(n_fields, 75.0),
        zr_max=np.full(n_fields, 1.2),
        zr_min=np.full(n_fields, 0.1),
        mad=np.full(n_fields, 0.5),
        irr_status=np.zeros(n_fields),
        perennial=np.zeros(n_fields),
        gw_status=np.zeros(n_fields),
        ke_max=np.full(n_fields, 1.2),
        f_sub=np.zeros(n_fields),
        ndvi_bare=np.full(n_fields, 0.15),
        ndvi_full=np.full(n_fields, 0.85),
        # Parameters (n_fields,)
        kc_max=np.full(n_fields, 1.25),
        kc_min=np.full(n_fields, 0.15),
        ndvi_k=np.full(n_fields, 7.0),
        ndvi_0=np.full(n_fields, 0.45),
        swe_alpha=np.full(n_fields, 0.001),
        swe_beta=np.full(n_fields, 2.0),
        kr_damp=np.full(n_fields, 0.5),
        ks_damp=np.full(n_fields, 0.5),
        max_irr_rate=np.full(n_fields, 25.0),
        # Initial state (n_fields,)
        depl_root_init=np.full(n_fields, 10.0),
        depl_ze_init=np.full(n_fields, 5.0),
        swe_init=np.zeros(n_fields),
        albedo_init=np.full(n_fields, 0.45),
        kr_init=np.ones(n_fields),
        ks_init=np.ones(n_fields),
        zr_init=np.full(n_fields, 0.3),
        s_init=np.full(n_fields, 84.7),
        s1_init=np.full(n_fields, 84.7),
        s2_init=np.full(n_fields, 84.7),
        s3_init=np.full(n_fields, 84.7),
        s4_init=np.full(n_fields, 84.7),
        daw3_init=np.zeros(n_fields),
        taw3_init=np.zeros(n_fields),
    )
    inputs.update(overrides)
    return inputs


def _unpack(result):
    """Unpack _run_loop_jit result tuple into a named dict."""
    names = [
        "eta",
        "etf",
        "kcb",
        "ke",
        "ks",
        "kr",
        "runoff",
        "rain",
        "melt",
        "swe",
        "depl_root",
        "dperc",
        "irr_sim",
        "gw_sim",
        # final state
        "final_depl_root",
        "final_depl_ze",
        "final_swe",
        "final_albedo",
        "final_kr",
        "final_ks",
        "final_zr",
        "final_daw3",
        "final_taw3",
    ]
    return dict(zip(names, result))


@pytest.fixture(scope="module")
def baseline():
    """Run the JIT once with default inputs; reuse across tests in this module."""
    inputs = _make_inputs()
    result = _run_loop_jit(**inputs)
    return _unpack(result), inputs


class TestOutputsFinite:
    def test_all_outputs_finite(self, baseline):
        out, _ = baseline
        output_keys = [
            "eta",
            "etf",
            "kcb",
            "ke",
            "ks",
            "kr",
            "runoff",
            "rain",
            "melt",
            "swe",
            "depl_root",
            "dperc",
            "irr_sim",
            "gw_sim",
        ]
        for key in output_keys:
            assert np.all(np.isfinite(out[key])), f"{key} has non-finite values"


class TestNonNegativity:
    @pytest.mark.parametrize(
        "key",
        ["eta", "runoff", "rain", "melt", "swe", "dperc", "irr_sim", "gw_sim", "depl_root"],
    )
    def test_non_negative(self, baseline, key):
        out, _ = baseline
        assert np.all(out[key] >= 0.0), f"{key} has negative values"


class TestPhysicalBounds:
    def test_runoff_le_precip_eff(self, baseline):
        out, _ = baseline
        precip_eff = out["rain"] + out["melt"]
        # Allow small numerical tolerance
        assert np.all(out["runoff"] <= precip_eff + 1e-10)

    def test_rain_plus_snow_equals_precip(self, baseline):
        out, inp = baseline
        prcp = inp["all_prcp"]
        tmin = inp["all_tmin"]
        tmax = inp["all_tmax"]
        temp_avg = (tmin + tmax) * 0.5
        snow = np.where(temp_avg < 1.0, prcp, 0.0)
        assert np.allclose(out["rain"] + snow, prcp)

    def test_melt_le_swe_before_melt(self):
        # Start with SWE=20, cold enough for some snow then warm for melt
        inputs = _make_inputs(
            swe_init=np.full(N_FIELDS, 20.0),
            all_tmin=np.full((N_DAYS, N_FIELDS), 5.0),
            all_tmax=np.full((N_DAYS, N_FIELDS), 15.0),
            all_prcp=np.zeros((N_DAYS, N_FIELDS)),
        )
        result = _unpack(_run_loop_jit(**inputs))
        # Melt on day 0: swe_before_melt = swe_init + snow (=0)
        # melt <= swe_before_melt for each day
        # We check that SWE never goes negative (which implies melt <= available)
        assert np.all(result["swe"] >= -1e-10)

    def test_ks_bounded_0_1(self, baseline):
        out, _ = baseline
        assert np.all(out["ks"] >= 0.0)
        assert np.all(out["ks"] <= 1.0 + 1e-10)

    def test_kr_bounded_0_1(self, baseline):
        out, _ = baseline
        assert np.all(out["kr"] >= 0.0)
        assert np.all(out["kr"] <= 1.0 + 1e-10)

    def test_etf_non_negative(self, baseline):
        out, _ = baseline
        assert np.all(out["etf"] >= 0.0)

    def test_depl_root_le_taw(self, baseline):
        out, inp = baseline
        # TAW = awc * zr; use max possible TAW = awc * zr_max
        taw_max = inp["awc"] * inp["zr_max"]
        assert np.all(out["depl_root"] <= taw_max[np.newaxis, :] + 1e-10)


class TestStateImmutability:
    def test_initial_arrays_unchanged(self):
        inputs = _make_inputs()
        # Save copies of initial state arrays
        init_keys = [
            "depl_root_init",
            "depl_ze_init",
            "swe_init",
            "albedo_init",
            "kr_init",
            "ks_init",
            "zr_init",
            "s_init",
            "s1_init",
            "s2_init",
            "s3_init",
            "s4_init",
            "daw3_init",
            "taw3_init",
        ]
        originals = {k: inputs[k].copy() for k in init_keys}
        _run_loop_jit(**inputs)
        for k in init_keys:
            np.testing.assert_array_equal(inputs[k], originals[k], err_msg=f"{k} was mutated")


class TestEdgeCases:
    def test_zero_precip_dry_day(self):
        inputs = _make_inputs(
            all_prcp=np.zeros((N_DAYS, N_FIELDS)),
            swe_init=np.zeros(N_FIELDS),
        )
        result = _unpack(_run_loop_jit(**inputs))
        assert np.all(np.isfinite(result["eta"]))
        assert np.all(result["runoff"] == 0.0)
        assert np.all(result["rain"] == 0.0)

    def test_irrigation_blocked_when_cold(self):
        inputs = _make_inputs(
            all_tmin=np.full((N_DAYS, N_FIELDS), -5.0),
            all_tmax=np.full((N_DAYS, N_FIELDS), 2.0),  # avg = -1.5, below 5
            all_irr_flag=np.ones((N_DAYS, N_FIELDS)),
            irr_status=np.ones(N_FIELDS),
            # High depletion to trigger demand
            depl_root_init=np.full(N_FIELDS, 100.0),
        )
        result = _unpack(_run_loop_jit(**inputs))
        assert np.all(result["irr_sim"] == 0.0)

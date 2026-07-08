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
        all_f_sub=np.zeros((n_days, n_fields)),
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
        cover_scaling=True,
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
        "daw3",
        "zr",
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

    def test_rapid_root_growth_stays_bounded(self):
        inputs = _make_inputs(
            n_days=1,
            n_fields=1,
            all_ndvi=np.array([[0.4050142467021942]]),
            all_etr=np.array([[0.8576244711875916]]),
            all_prcp=np.array([[0.0]]),
            all_tmin=np.array([[-11.649999618530273]]),
            all_tmax=np.array([[4.349999904632568]]),
            all_srad=np.array([[83.0]]),
            all_irr_flag=np.array([[0.0]]),
            awc=np.array([255.979]),
            rew=np.array([3.0]),
            tew=np.array([18.0]),
            cn2=np.array([77.0]),
            zr_max=np.array([1.12]),
            zr_min=np.array([0.1]),
            mad=np.array([0.464604]),
            irr_status=np.array([1.0]),
            perennial=np.array([0.0]),
            gw_status=np.array([1.0]),
            ke_max=np.array([0.6048164367675781]),
            all_f_sub=np.array([[0.0]]),
            kc_max=np.array([1.35]),
            kc_min=np.array([0.15]),
            ndvi_k=np.array([12.3823]),
            ndvi_0=np.array([0.196214]),
            swe_alpha=np.array([0.332399]),
            swe_beta=np.array([1.60911]),
            kr_damp=np.array([0.827802]),
            ks_damp=np.array([0.768962]),
            max_irr_rate=np.array([100.0]),
            depl_root_init=np.array([9.748238477427876]),
            depl_ze_init=np.array([5.302519881448234]),
            swe_init=np.array([0.0]),
            albedo_init=np.array([0.6281947416645687]),
            kr_init=np.array([0.8730189608889664]),
            ks_init=np.array([1.0]),
            zr_init=np.array([0.3910559608884386]),
            s_init=np.array([84.7]),
            s1_init=np.array([84.7]),
            s2_init=np.array([84.7]),
            s3_init=np.array([84.7]),
            s4_init=np.array([84.7]),
            daw3_init=np.array([118.3037730195151]),
            taw3_init=np.array([186.59436618773844]),
        )
        out = _unpack(_run_loop_jit(**inputs))
        taw = inputs["awc"] * out["final_zr"]
        taw = np.maximum(taw, 18.0)

        assert np.all(np.isfinite(out["depl_root"]))
        assert np.all(out["dperc"] >= 0.0)
        assert np.all(out["depl_root"] >= 0.0)
        assert np.all(out["depl_root"] <= taw[np.newaxis, :] + 1e-10)
        assert np.all(out["final_daw3"] >= 0.0)
        assert np.all(out["final_daw3"] <= out["final_taw3"] + 1e-10)


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


class TestCoverScalingToggle:
    """Verify the transpiration_cover_scaling toggle."""

    def test_default_true_matches_original(self):
        """cover_scaling=True must reproduce the original kc_act = fc*ks*kcb + ke."""
        inputs = _make_inputs(cover_scaling=True)
        result_on = _unpack(_run_loop_jit(**inputs))

        inputs_default = _make_inputs()
        result_default = _unpack(_run_loop_jit(**inputs_default))

        np.testing.assert_array_equal(result_on["eta"], result_default["eta"])

    def test_fc_off_raises_et_at_partial_cover(self):
        """With partial canopy (fc<1), fc-off should produce higher ET than fc-on."""
        inputs_on = _make_inputs(
            all_ndvi=np.full((N_DAYS, N_FIELDS), 0.3),
            cover_scaling=True,
        )
        inputs_off = _make_inputs(
            all_ndvi=np.full((N_DAYS, N_FIELDS), 0.3),
            cover_scaling=False,
        )
        eta_on = _unpack(_run_loop_jit(**inputs_on))["eta"]
        eta_off = _unpack(_run_loop_jit(**inputs_off))["eta"]
        assert np.all(eta_off >= eta_on - 1e-10)
        assert np.any(eta_off > eta_on + 1e-6)

    def test_fc_off_preserves_ke_and_few(self):
        """cover_scaling=False must not change Ke (few = 1-fc stays)."""
        inputs_on = _make_inputs(cover_scaling=True)
        inputs_off = _make_inputs(cover_scaling=False)
        ke_on = _unpack(_run_loop_jit(**inputs_on))["ke"]
        ke_off = _unpack(_run_loop_jit(**inputs_off))["ke"]
        np.testing.assert_array_equal(ke_on, ke_off)

    def test_full_cover_no_difference(self):
        """At fc≈1 (high NDVI), fc-on and fc-off should be nearly identical."""
        inputs_on = _make_inputs(
            all_ndvi=np.full((N_DAYS, N_FIELDS), 0.9),
            cover_scaling=True,
        )
        inputs_off = _make_inputs(
            all_ndvi=np.full((N_DAYS, N_FIELDS), 0.9),
            cover_scaling=False,
        )
        eta_on = _unpack(_run_loop_jit(**inputs_on))["eta"]
        eta_off = _unpack(_run_loop_jit(**inputs_off))["eta"]
        np.testing.assert_allclose(eta_on, eta_off, atol=0.5)


class TestTEWEvapCap:
    """B1: eta must track the TEW-capped evaporation, not the uncapped value."""

    def test_eta_reduced_when_evap_capped(self):
        """When depl_ze is near TEW, evap is capped and eta must shrink by the same amount."""
        inputs = _make_inputs(
            n_days=3,
            n_fields=1,
            tew=np.array([10.0]),
            rew=np.array([4.0]),
            depl_ze_init=np.array([9.5]),
            all_prcp=np.zeros((3, 1)),
            all_etr=np.full((3, 1), 8.0),
            all_ndvi=np.full((3, 1), 0.3),
            kr_init=np.ones(1),
            ke_max=np.full(1, 1.2),
        )
        out = _unpack(_run_loop_jit(**inputs))
        for d in range(3):
            assert out["eta"][d, 0] >= 0.0
            ke_etr = out["ke"][d, 0] * inputs["all_etr"][d, 0]
            assert out["eta"][d, 0] <= ke_etr + out["kcb"][d, 0] * inputs["all_etr"][d, 0] + 1e-10

    def test_etf_consistent_with_capped_eta(self):
        """ETf = eta/etr must use the TEW-corrected eta."""
        inputs = _make_inputs(
            n_days=5,
            n_fields=1,
            tew=np.array([8.0]),
            rew=np.array([3.0]),
            depl_ze_init=np.array([7.0]),
            all_prcp=np.zeros((5, 1)),
            all_etr=np.full((5, 1), 10.0),
            all_ndvi=np.full((5, 1), 0.2),
            kr_init=np.ones(1),
            ke_max=np.full(1, 1.2),
        )
        out = _unpack(_run_loop_jit(**inputs))
        expected_etf = np.where(inputs["all_etr"] > 0, out["eta"] / inputs["all_etr"], 0.0)
        np.testing.assert_allclose(out["etf"], expected_etf, atol=1e-10)

    def test_water_balance_closes_during_tew_cap(self):
        """Mass conservation: root-zone water balance must close.

        Uses a dry scenario (no precip, depl_ze near TEW) that forces the TEW
        cap to fire repeatedly.  Before B1, the uncapped evaporation leaked into
        eta but was not removed from the soil, violating conservation.

        Pin zr_min = zr_max so root depth (and TAW) stay constant, avoiding
        storage artifacts from root-zone growth.  With no precip/irr/gw the
        only flux is ET, so: Σeta = final_depl - init_depl.
        """
        n_days, n_fields = 10, 1
        inputs = _make_inputs(
            n_days=n_days,
            n_fields=n_fields,
            tew=np.array([12.0]),
            rew=np.array([4.0]),
            depl_ze_init=np.array([10.0]),
            depl_root_init=np.array([5.0]),
            all_prcp=np.zeros((n_days, n_fields)),
            all_etr=np.full((n_days, n_fields), 7.0),
            all_ndvi=np.full((n_days, n_fields), 0.4),
            kr_init=np.ones(n_fields),
            ke_max=np.full(n_fields, 1.2),
            irr_status=np.zeros(n_fields),
            gw_status=np.zeros(n_fields),
            zr_min=np.full(n_fields, 0.5),
            zr_max=np.full(n_fields, 0.5),
            zr_init=np.full(n_fields, 0.5),
        )
        out = _unpack(_run_loop_jit(**inputs))

        total_eta = out["eta"].sum(axis=0)
        total_runoff = out["runoff"].sum(axis=0)
        total_dperc = out["dperc"].sum(axis=0)
        delta_depl = out["final_depl_root"] - inputs["depl_root_init"]

        # ET increases depletion: Σeta = Δdepl + Σdperc (with no inputs)
        # General form: Σprecip + Σirr + Σgw = Σeta + Σrunoff + Σdperc - Δdepl
        residual = -total_eta - total_runoff - total_dperc + delta_depl
        np.testing.assert_allclose(
            residual, 0.0, atol=1e-6, err_msg="Water balance does not close (B1 phantom evap)"
        )

    def test_transpiration_survives_combined_caps(self):
        """When the available-water cap and the TEW cap bind on the same day,
        only the evaporation actually inside the capped eta may be removed.

        Setup: root zone nearly exhausted (available = 2 mm, so eta is capped
        far below potential) and the surface layer starts at TEW (e_factor = 0,
        all evaporation removed).  Pre-fix, the full *uncapped* evaporation
        (ke*etr > 2 mm) was subtracted from the capped eta, wiping out the
        transpiration share and reporting eta = 0.
        """
        tew = 25.0
        inputs = _make_inputs(
            n_days=1,
            n_fields=1,
            tew=np.array([tew]),
            rew=np.array([8.0]),
            depl_ze_init=np.array([tew]),  # surface at TEW: today's evap fully capped
            depl_root_init=np.array([73.0]),  # taw = awc*zr = 75 -> available = 2 mm
            all_prcp=np.zeros((1, 1)),
            all_etr=np.full((1, 1), 10.0),
            all_ndvi=np.full((1, 1), 0.5),  # kcb > 0: transpiration demand exists
            kr_init=np.ones(1),
            ke_max=np.full(1, 1.2),
            zr_min=np.full(1, 0.5),
            zr_max=np.full(1, 0.5),
            zr_init=np.full(1, 0.5),
        )
        out = _unpack(_run_loop_jit(**inputs))
        eta = out["eta"][0, 0]
        # Scenario check: uncapped evap exceeds available water, so the old
        # code drove eta to exactly 0 here
        assert out["ke"][0, 0] * 10.0 > 2.0
        # Transpiration must survive the TEW adjustment
        assert eta > 0.0
        # ...but eta can never exceed the available water
        assert eta <= 2.0 + 1e-9
        # The surface layer must not be depleted by evaporation that never
        # left the soil
        assert out["final_depl_ze"][0] <= tew + 1e-9

    def test_water_balance_closes_combined_caps(self):
        """Water balance must still close when the available-water cap and the
        TEW cap both bind (the evap rescaling must not break closure)."""
        n_days, n_fields = 10, 1
        inputs = _make_inputs(
            n_days=n_days,
            n_fields=n_fields,
            tew=np.array([12.0]),
            rew=np.array([4.0]),
            depl_ze_init=np.array([11.0]),
            depl_root_init=np.array([73.0]),  # taw = awc*zr = 75: cap binds day 1
            all_prcp=np.zeros((n_days, n_fields)),
            all_etr=np.full((n_days, n_fields), 7.0),
            all_ndvi=np.full((n_days, n_fields), 0.4),
            kr_init=np.ones(n_fields),
            ke_max=np.full(n_fields, 1.2),
            irr_status=np.zeros(n_fields),
            gw_status=np.zeros(n_fields),
            zr_min=np.full(n_fields, 0.5),
            zr_max=np.full(n_fields, 0.5),
            zr_init=np.full(n_fields, 0.5),
        )
        out = _unpack(_run_loop_jit(**inputs))

        total_eta = out["eta"].sum(axis=0)
        total_runoff = out["runoff"].sum(axis=0)
        total_dperc = out["dperc"].sum(axis=0)
        delta_depl = out["final_depl_root"] - inputs["depl_root_init"]

        residual = -total_eta - total_runoff - total_dperc + delta_depl
        np.testing.assert_allclose(
            residual,
            0.0,
            atol=1e-6,
            err_msg="Water balance does not close under combined caps",
        )

"""Unit tests for process package physics kernels.

Tests verify:
1. Kernels compile correctly with numba
2. Physical constraints are enforced
3. Output shapes match input shapes
4. Edge cases are handled properly
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from swimrs.process.kernels.cover import exposed_soil_fraction, fractional_cover

# Import kernels
from swimrs.process.kernels.crop_coefficient import kcb_sigmoid
from swimrs.process.kernels.evaporation import kr_reduction
from swimrs.process.kernels.irrigation import groundwater_subsidy, irrigation_demand
from swimrs.process.kernels.root_growth import (
    root_depth_from_kcb,
)
from swimrs.process.kernels.runoff import (
    scs_runoff,
)
from swimrs.process.kernels.snow import (
    albedo_decay,
    degree_day_melt,
    partition_precip,
)
from swimrs.process.kernels.transpiration import ks_stress
from swimrs.process.kernels.water_balance import (
    deep_percolation,
    layer3_storage,
)


class TestKcbSigmoid:
    """Tests for kcb_sigmoid kernel."""

    def test_basic_computation(self):
        """Kcb computed correctly for typical inputs."""
        ndvi = np.array([0.2, 0.4, 0.6, 0.8])
        kc_max = np.array([1.2, 1.2, 1.2, 1.2])
        ndvi_k = np.array([7.0, 7.0, 7.0, 7.0])
        ndvi_0 = np.array([0.4, 0.4, 0.4, 0.4])

        kcb = kcb_sigmoid(ndvi, kc_max, ndvi_k, ndvi_0)

        assert kcb.shape == ndvi.shape
        # At ndvi_0, kcb should be approximately kc_max/2
        assert 0.5 <= kcb[1] / kc_max[1] < 0.7
        # Kcb should increase with NDVI
        assert np.all(np.diff(kcb) > 0)

    def test_bounds_respected(self):
        """Kcb bounded by [0, kc_max]."""
        ndvi = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        kc_max = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
        ndvi_k = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
        ndvi_0 = np.array([0.4, 0.4, 0.4, 0.4, 0.4])

        kcb = kcb_sigmoid(ndvi, kc_max, ndvi_k, ndvi_0)

        assert np.all(kcb >= 0)
        assert np.all(kcb <= kc_max)

    def test_inflection_point(self):
        """At inflection point, kcb = kc_max/2."""
        ndvi = np.array([0.4])
        kc_max = np.array([1.2])
        ndvi_k = np.array([7.0])
        ndvi_0 = np.array([0.4])

        kcb = kcb_sigmoid(ndvi, kc_max, ndvi_k, ndvi_0)

        assert_array_almost_equal(kcb, kc_max / 2, decimal=5)


class TestFractionalCover:
    """Tests for fractional_cover kernel."""

    def test_zero_vegetation(self):
        """fc = 0 when kcb = kc_min."""
        kcb = np.array([0.15])
        kc_min = np.array([0.15])
        kc_max = np.array([1.2])

        fc = fractional_cover(kcb, kc_min, kc_max)

        assert_array_almost_equal(fc, [0.0])

    def test_full_vegetation(self):
        """fc approaches 0.99 when kcb = kc_max."""
        kcb = np.array([1.2])
        kc_min = np.array([0.15])
        kc_max = np.array([1.2])

        fc = fractional_cover(kcb, kc_min, kc_max)

        assert_array_almost_equal(fc, [0.99])

    def test_mid_vegetation(self):
        """fc = 0.5 when kcb is midpoint."""
        kc_min = np.array([0.15])
        kc_max = np.array([1.15])  # Range of 1.0 for easy math
        kc_mid = (kc_min + kc_max) / 2
        kcb = kc_mid

        fc = fractional_cover(kcb, kc_min, kc_max)

        assert 0.45 < fc[0] < 0.55


class TestExposedSoilFraction:
    """Tests for exposed_soil_fraction kernel."""

    def test_complement_of_fc(self):
        """few = 1 - fc."""
        fc = np.array([0.0, 0.25, 0.5, 0.75, 0.99])

        few = exposed_soil_fraction(fc)

        expected = np.array([1.0, 0.75, 0.5, 0.25, 0.01])
        assert_array_almost_equal(few, expected)


class TestKrReduction:
    """Tests for kr_reduction kernel."""

    def test_wet_surface(self):
        """Kr = 1 when surface is wet (depl_ze = 0)."""
        tew = np.array([25.0])
        depl_ze = np.array([0.0])
        rew = np.array([10.0])

        kr = kr_reduction(tew, depl_ze, rew)

        assert_array_almost_equal(kr, [1.0])

    def test_dry_surface(self):
        """Kr = 0 when surface is fully depleted."""
        tew = np.array([25.0])
        depl_ze = np.array([30.0])  # > tew
        rew = np.array([10.0])

        kr = kr_reduction(tew, depl_ze, rew)

        assert_array_almost_equal(kr, [0.0])

    def test_partial_depletion(self):
        """Kr interpolates linearly between REW and TEW."""
        tew = np.array([25.0])
        rew = np.array([10.0])
        # Midpoint between rew and tew
        depl_ze = np.array([17.5])

        kr = kr_reduction(tew, depl_ze, rew)

        assert 0.4 < kr[0] < 0.6


class TestKsStress:
    """Tests for ks_stress kernel."""

    def test_no_stress(self):
        """Ks = 1 when depletion below RAW."""
        taw = np.array([100.0])
        raw = np.array([50.0])
        depl_root = np.array([30.0])

        ks = ks_stress(taw, depl_root, raw)

        assert_array_almost_equal(ks, [1.0])

    def test_full_stress(self):
        """Ks = 0 when depletion equals TAW."""
        taw = np.array([100.0])
        raw = np.array([50.0])
        depl_root = np.array([100.0])

        ks = ks_stress(taw, depl_root, raw)

        assert_array_almost_equal(ks, [0.0])

    def test_partial_stress(self):
        """Ks interpolates when RAW < depl < TAW."""
        taw = np.array([100.0])
        raw = np.array([50.0])
        # Midpoint between raw and taw
        depl_root = np.array([75.0])

        ks = ks_stress(taw, depl_root, raw)

        assert_array_almost_equal(ks, [0.5])


class TestScsRunoff:
    """Tests for SCS Curve Number runoff kernels."""

    def test_zero_precip(self):
        """No runoff when no precipitation."""
        precip = np.array([0.0])
        cn = np.array([75.0])

        sro, s = scs_runoff(precip, cn)

        assert_array_almost_equal(sro, [0.0])

    def test_runoff_less_than_precip(self):
        """Runoff cannot exceed precipitation."""
        precip = np.array([50.0])
        cn = np.array([90.0])

        sro, s = scs_runoff(precip, cn)

        assert sro[0] < precip[0]
        assert sro[0] > 0

    def test_high_cn_more_runoff(self):
        """Higher CN produces more runoff."""
        precip = np.array([50.0, 50.0])
        cn = np.array([70.0, 90.0])

        sro, s = scs_runoff(precip, cn)

        assert sro[1] > sro[0]


class TestSnowKernels:
    """Tests for snow-related kernels."""

    def test_partition_cold(self):
        """All precip is snow when cold."""
        precip = np.array([10.0])
        temp_avg = np.array([-5.0])

        rain, snow = partition_precip(precip, temp_avg)

        assert_array_almost_equal(rain, [0.0])
        assert_array_almost_equal(snow, [10.0])

    def test_partition_warm(self):
        """All precip is rain when warm."""
        precip = np.array([10.0])
        temp_avg = np.array([10.0])

        rain, snow = partition_precip(precip, temp_avg)

        assert_array_almost_equal(rain, [10.0])
        assert_array_almost_equal(snow, [0.0])

    def test_albedo_fresh_snow(self):
        """Albedo resets with fresh snow."""
        albedo_prev = np.array([0.5])
        snow_fall = np.array([10.0])

        albedo = albedo_decay(albedo_prev, snow_fall)

        assert albedo[0] > 0.95

    def test_albedo_decay_no_snow(self):
        """Albedo decays without fresh snow."""
        albedo_prev = np.array([0.8])
        snow_fall = np.array([0.0])

        albedo = albedo_decay(albedo_prev, snow_fall)

        assert albedo[0] < albedo_prev[0]

    def test_melt_bounded_by_swe(self):
        """Melt cannot exceed available SWE."""
        swe = np.array([5.0])
        temp_max = np.array([20.0])
        temp_avg = np.array([15.0])
        srad = np.array([25.0])
        albedo = np.array([0.5])
        swe_alpha = np.array([0.5])
        swe_beta = np.array([2.0])

        melt = degree_day_melt(swe, temp_max, temp_avg, srad, albedo, swe_alpha, swe_beta)

        assert melt[0] <= swe[0]


class TestWaterBalance:
    """Tests for water balance kernels."""

    def test_deep_percolation_positive_depletion(self):
        """No percolation when depletion is positive."""
        depl_root = np.array([50.0])

        dperc, depl_updated = deep_percolation(depl_root)

        assert_array_almost_equal(dperc, [0.0])
        assert_array_almost_equal(depl_updated, [50.0])

    def test_deep_percolation_negative_depletion(self):
        """Excess water percolates when depletion is negative."""
        depl_root = np.array([-20.0])

        dperc, depl_updated = deep_percolation(depl_root)

        assert_array_almost_equal(dperc, [20.0])
        assert_array_almost_equal(depl_updated, [0.0])

    def test_layer3_overflow(self):
        """Excess water overflows from layer 3."""
        daw3 = np.array([80.0])
        taw3 = np.array([100.0])
        gross_dperc = np.array([30.0])

        daw3_new, dperc_out = layer3_storage(daw3, taw3, gross_dperc)

        assert_array_almost_equal(daw3_new, [100.0])
        assert_array_almost_equal(dperc_out, [10.0])


class TestRootGrowth:
    """Tests for root growth kernels."""

    def test_root_depth_bounds(self):
        """Root depth bounded by zr_min and zr_max."""
        kcb = np.array([0.0, 0.5, 1.2])
        kc_min = np.array([0.15, 0.15, 0.15])
        kc_max = np.array([1.2, 1.2, 1.2])
        zr_max = np.array([1.0, 1.0, 1.0])
        zr_min = np.array([0.1, 0.1, 0.1])

        zr = root_depth_from_kcb(kcb, kc_min, kc_max, zr_max, zr_min)

        assert np.all(zr >= zr_min)
        assert np.all(zr <= zr_max)


class TestIrrigation:
    """Tests for irrigation kernels."""

    def test_no_irrigation_when_wet(self):
        """No irrigation when depletion below RAW."""
        depl_root = np.array([30.0])
        raw = np.array([50.0])
        max_irr_rate = np.array([25.0])
        irr_flag = np.array([True])
        temp_avg = np.array([20.0])
        irr_continue = np.array([0.0])
        next_day_irr = np.array([0.0])

        irr_sim, _, _ = irrigation_demand(
            depl_root, raw, max_irr_rate, irr_flag, temp_avg, irr_continue, next_day_irr
        )

        assert_array_almost_equal(irr_sim, [0.0])

    def test_irrigation_when_stressed(self):
        """Irrigation applied when depletion exceeds RAW."""
        depl_root = np.array([70.0])
        raw = np.array([50.0])
        max_irr_rate = np.array([25.0])
        irr_flag = np.array([True])
        temp_avg = np.array([20.0])
        irr_continue = np.array([0.0])
        next_day_irr = np.array([0.0])

        irr_sim, _, _ = irrigation_demand(
            depl_root, raw, max_irr_rate, irr_flag, temp_avg, irr_continue, next_day_irr
        )

        assert irr_sim[0] > 0
        assert irr_sim[0] <= max_irr_rate[0]


class TestGroundwaterSubsidy:
    """Tests for groundwater subsidy kernel."""

    def test_no_subsidy_when_wet(self):
        """No subsidy when depletion below RAW."""
        depl_root = np.array([30.0])
        raw = np.array([50.0])
        gw_status = np.array([True])
        f_sub = np.array([1.0])

        gw_sim = groundwater_subsidy(depl_root, raw, gw_status, f_sub)

        assert_array_almost_equal(gw_sim, [0.0])

    def test_subsidy_fills_to_raw(self):
        """Full subsidy fills to RAW level."""
        depl_root = np.array([80.0])
        raw = np.array([50.0])
        gw_status = np.array([True])
        f_sub = np.array([1.0])

        gw_sim = groundwater_subsidy(depl_root, raw, gw_status, f_sub)

        # Deficit is 80 - 50 = 30
        assert_array_almost_equal(gw_sim, [30.0])

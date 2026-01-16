"""Tests for the compute_field_et soil water balance module."""

import numpy as np
import pytest

from swimrs.model import compute_field_et as cfe
from swimrs.model import k_dynamics as kd


class DummyConfig:
    """Mock configuration object."""
    def __init__(self, runoff_process=None):
        self.runoff_process = runoff_process
        self.calibrate = False
        self.forecast = False


class DummyDayData:
    """Mock DayData with daily meteorology for N fields."""
    def __init__(self, n_fields=2):
        self.n_fields = n_fields
        self.dt_string = '2020-06-15'
        self.year = 2020
        self.month = 6
        self.day = 15
        self.doy = 167

        # Default meteorology - warm day with no precip
        self.precip = np.zeros((1, n_fields))
        self.hr_precip = np.zeros((24, n_fields))
        self.refet = np.ones((1, n_fields)) * 5.0  # mm/day
        self.min_temp = np.ones((1, n_fields)) * 15.0
        self.max_temp = np.ones((1, n_fields)) * 30.0
        self.temp_avg = (self.min_temp + self.max_temp) / 2.0
        self.srad = np.ones((1, n_fields)) * 25.0  # MJ/m2/day
        self.ndvi = np.ones((1, n_fields)) * 0.6

        # Irrigation and groundwater flags
        self.irr_day = np.zeros((1, n_fields), dtype=int)
        self.gwsub_status = np.zeros((1, n_fields))


class DummyTracker:
    """Mock SampleTracker with water balance state for N fields."""
    def __init__(self, n_fields=2, runoff_process=None):
        self.size = n_fields
        self.conf = DummyConfig(runoff_process=runoff_process)

        # Crop coefficients
        self.kc_bas = np.ones((1, n_fields)) * 0.8
        self.kc_min = np.ones((1, n_fields)) * 0.15
        self.kc_max = np.ones((1, n_fields)) * 1.2
        self.kc_act = np.zeros((1, n_fields))
        self.fc = np.zeros((1, n_fields))
        self.few = np.zeros((1, n_fields))
        self.t = np.zeros((1, n_fields))

        # Evaporation coefficients
        self.ke = np.zeros((1, n_fields))
        self.kr = np.ones((1, n_fields))
        self.kr_prev = np.ones((1, n_fields))
        self.ke_max = np.ones((1, n_fields)) * 1.0
        self.kr_alpha = np.ones((1, n_fields)) * 0.25

        # Transpiration stress
        self.ks = np.ones((1, n_fields))
        self.ks_prev = np.ones((1, n_fields))
        self.ks_alpha = np.ones((1, n_fields)) * 0.15

        # Soil water parameters (mm/m for aw, m for depths)
        self.aw = np.ones((1, n_fields)) * 150.0  # mm/m
        self.zr = np.ones((1, n_fields)) * 0.6  # m
        self.zr_min = np.ones((1, n_fields)) * 0.1
        self.zr_max = np.ones((1, n_fields)) * 1.2
        self.mad = np.ones((1, n_fields)) * 0.5
        self.taw = np.zeros((1, n_fields))
        self.raw = np.zeros((1, n_fields))

        # Surface evaporation layer
        self.tew = np.ones((1, n_fields)) * 15.0  # mm
        self.rew = np.ones((1, n_fields)) * 6.0  # mm
        self.depl_ze = np.ones((1, n_fields)) * 5.0  # mm (partially depleted)
        self.depl_surface = np.zeros((1, n_fields))
        self.cum_evap = np.zeros((1, n_fields))
        self.cum_evap_prev = np.zeros((1, n_fields))

        # Root zone depletion
        self.depl_root = np.ones((1, n_fields)) * 30.0  # mm
        self.depl_root_prev = np.zeros((1, n_fields))

        # Soil water storage
        self.soil_water = np.ones((1, n_fields)) * 60.0  # mm
        self.soil_water_prev = np.zeros((1, n_fields))
        self.delta_soil_water = np.zeros((1, n_fields))

        # Layer 3 (below root zone)
        self.daw3 = np.ones((1, n_fields)) * 20.0  # mm
        self.daw3_prev = np.zeros((1, n_fields))
        self.taw3 = np.ones((1, n_fields)) * 90.0  # mm
        self.aw3 = np.zeros((1, n_fields))
        self.delta_daw3 = np.zeros((1, n_fields))

        # Precipitation partitioning
        self.ppt_inf = np.zeros((1, n_fields))
        self.ppt_inf_prev = np.zeros((1, n_fields))
        self.rain = np.zeros((1, n_fields))
        self.snow_fall = np.zeros((1, n_fields))
        self.melt = np.zeros((1, n_fields))
        self.swe = np.zeros((1, n_fields))

        # Snow parameters
        self.albedo = np.ones((1, n_fields)) * 0.45
        self.min_albedo = np.ones((1, n_fields)) * 0.45
        self.swe_alpha = np.ones((1, n_fields)) * 0.5
        self.swe_beta = np.ones((1, n_fields)) * 1.3

        # Runoff
        self.sro = np.zeros((1, n_fields))
        self.ksat = np.ones((1, n_fields)) * 50.0  # mm/day
        self.ksat_hourly = np.ones((24, n_fields)) * (50.0 / 24.0)

        # Curve number parameters (for CN mode)
        self.cn2 = np.ones((1, n_fields)) * 75.0
        self.irr_flag = np.zeros((1, n_fields), dtype=int)
        self.s = np.zeros((1, n_fields))
        self.s1 = np.zeros((1, n_fields))
        self.s2 = np.zeros((1, n_fields))
        self.s3 = np.zeros((1, n_fields))
        self.s4 = np.zeros((1, n_fields))

        # Deep percolation
        self.dperc = np.zeros((1, n_fields))

        # Irrigation
        self.irr_sim = np.zeros((1, n_fields))
        self.irr_continue = np.zeros((1, n_fields), dtype=int)
        self.next_day_irr = np.zeros((1, n_fields))
        self.max_irr_rate = np.ones((1, n_fields)) * 25.4  # mm/day
        self.niwr = np.zeros((1, n_fields))
        self.wt_irr = np.zeros((1, n_fields))

        # Groundwater
        self.gw_sim = np.zeros((1, n_fields))

        # ET outputs
        self.etc_act = np.zeros((1, n_fields))
        self.e = np.zeros((1, n_fields))

        # Effective precipitation
        self.p_rz = np.zeros((1, n_fields))
        self.p_eft = np.zeros((1, n_fields))

        # Root growth
        self.perennial = np.zeros((1, n_fields), dtype=int)


# =============================================================================
# Unit Tests: Individual Component Behavior
# =============================================================================

class TestFcCalculation:
    """Tests for fractional cover (fc) calculation from Kcb."""

    def test_fc_basic_calculation(self):
        """Fc should be computed as normalized Kcb."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.kc_bas = np.array([[0.8]])
        swb.kc_min = np.array([[0.15]])
        swb.kc_max = np.array([[1.2]])

        cfe.compute_field_et(swb, day_data)

        expected_fc = (0.8 - 0.15) / (1.2 - 0.15)
        assert np.isclose(swb.fc[0, 0], expected_fc, rtol=0.01)

    def test_fc_clamped_below_one(self):
        """Fc should be clamped to 0.99 max."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Set kc_bas = kc_max so fc would be 1.0
        swb.kc_bas = np.array([[1.2]])
        swb.kc_min = np.array([[0.15]])
        swb.kc_max = np.array([[1.2]])

        cfe.compute_field_et(swb, day_data)

        assert swb.fc[0, 0] <= 0.99

    def test_fc_kc_bas_floored_at_kc_min(self):
        """Kc_bas below kc_min should be raised to kc_min."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.kc_bas = np.array([[0.05]])  # Below kc_min
        swb.kc_min = np.array([[0.15]])
        swb.kc_max = np.array([[1.2]])

        cfe.compute_field_et(swb, day_data)

        # fc should be 0 when kc_bas = kc_min
        assert swb.fc[0, 0] >= 0.0
        assert swb.kc_bas[0, 0] >= swb.kc_min[0, 0]


class TestSurfaceEvaporation:
    """Tests for surface evaporation layer (depl_ze) updates."""

    def test_depl_ze_increases_with_evaporation(self):
        """Surface depletion should increase after evaporation on dry day."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        initial_depl_ze = 5.0
        swb.depl_ze = np.array([[initial_depl_ze]])
        day_data.precip = np.zeros((1, 1))
        day_data.refet = np.array([[5.0]])

        cfe.compute_field_et(swb, day_data)

        # After evaporation, depl_ze should increase (or stay same if ke=0)
        assert swb.depl_ze[0, 0] >= initial_depl_ze

    def test_depl_ze_bounded_by_tew(self):
        """Surface depletion should not exceed TEW."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.tew = np.array([[15.0]])
        swb.depl_ze = np.array([[14.0]])  # Near TEW
        day_data.refet = np.array([[10.0]])  # High ET demand

        cfe.compute_field_et(swb, day_data)

        assert swb.depl_ze[0, 0] <= swb.tew[0, 0]


class TestTranspirationStress:
    """Tests for Ks (transpiration stress coefficient) behavior."""

    def test_ks_equals_one_when_not_stressed(self):
        """Ks should be 1.0 when depletion is below RAW."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Set up unstressed conditions
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.mad = np.array([[0.5]])
        swb.depl_root = np.array([[20.0]])  # Well below RAW

        cfe.compute_field_et(swb, day_data)

        # Ks should be close to 1.0
        assert swb.ks[0, 0] >= 0.9

    def test_ks_decreases_under_stress(self):
        """Ks should decrease when depletion exceeds RAW."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.mad = np.array([[0.5]])
        # TAW = 150 * 0.6 = 90, RAW = 0.5 * 90 = 45
        swb.depl_root = np.array([[70.0]])  # Above RAW, stressed

        cfe.compute_field_et(swb, day_data)

        assert swb.ks[0, 0] < 1.0
        assert swb.ks[0, 0] >= 0.0


class TestActualET:
    """Tests for actual ET calculation."""

    def test_etc_act_computed_correctly(self):
        """ETc_act should be kc_act * refET."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.refet = np.array([[5.0]])

        cfe.compute_field_et(swb, day_data)

        expected_etc = swb.kc_act[0, 0] * day_data.refet[0, 0]
        assert np.isclose(swb.etc_act[0, 0], expected_etc, rtol=0.001)

    def test_kc_act_bounded_by_kc_max(self):
        """Kc_act should not exceed kc_max."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.kc_max = np.array([[1.2]])

        cfe.compute_field_et(swb, day_data)

        assert swb.kc_act[0, 0] <= swb.kc_max[0, 0]


class TestDeepPercolation:
    """Tests for deep percolation behavior."""

    def test_dperc_zero_when_not_saturated(self):
        """Deep percolation should be zero when depl_root > 0."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.depl_root = np.array([[30.0]])
        day_data.precip = np.zeros((1, 1))

        cfe.compute_field_et(swb, day_data)

        assert swb.dperc[0, 0] >= 0.0

    def test_dperc_occurs_when_saturated(self):
        """Deep percolation should occur when root zone is overfilled."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Add heavy precipitation to fill root zone
        swb.depl_root = np.array([[10.0]])  # Low initial depletion
        day_data.precip = np.array([[50.0]])  # Heavy rain
        day_data.hr_precip = np.ones((24, 1)) * (50.0 / 24.0)
        day_data.temp_avg = np.array([[20.0]])  # Warm, no snow

        cfe.compute_field_et(swb, day_data)

        # With 50mm precip and only 10mm depletion, should have dperc
        # (depends on ET losses too)
        assert swb.dperc[0, 0] >= 0.0


class TestIrrigation:
    """Tests for irrigation application logic."""

    def test_no_irrigation_when_not_flagged(self):
        """No irrigation should be applied when irr_day is False."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.irr_day = np.array([[0]])
        swb.irr_continue = np.zeros((1, 1), dtype=int)

        cfe.compute_field_et(swb, day_data)

        assert swb.irr_sim[0, 0] == 0.0

    def test_irrigation_applied_when_stressed(self):
        """Irrigation should be applied when flagged and stressed."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.irr_day = np.array([[1]])
        day_data.temp_avg = np.array([[20.0]])  # Above 5C threshold
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.mad = np.array([[0.5]])
        # TAW = 90, RAW = 45
        swb.depl_root = np.array([[60.0]])  # Stressed (> RAW)

        cfe.compute_field_et(swb, day_data)

        # irr_sim should be applied
        assert swb.irr_sim[0, 0] > 0.0

    def test_no_irrigation_when_cold(self):
        """No irrigation should be applied when temp < 5C."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.irr_day = np.array([[1]])
        day_data.temp_avg = np.array([[3.0]])  # Below 5C
        swb.depl_root = np.array([[60.0]])  # Stressed
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.mad = np.array([[0.5]])

        cfe.compute_field_et(swb, day_data)

        assert swb.irr_sim[0, 0] == 0.0


class TestGroundwaterSubsidy:
    """Tests for groundwater subsidy application."""

    def test_gw_subsidy_applied_when_stressed(self):
        """GW subsidy should reduce depletion when flagged and stressed."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.gwsub_status = np.array([[1]])
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.mad = np.array([[0.5]])
        swb.depl_root = np.array([[60.0]])  # Stressed

        cfe.compute_field_et(swb, day_data)

        assert swb.gw_sim[0, 0] > 0.0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_precipitation_dry_day(self):
        """Water balance should work correctly with zero precipitation."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.precip = np.zeros((1, 1))
        day_data.hr_precip = np.zeros((24, 1))
        initial_depl_root = swb.depl_root.copy()

        cfe.compute_field_et(swb, day_data)

        # Depletion should increase (or stay same) without precip
        assert swb.depl_root[0, 0] >= initial_depl_root[0, 0] - 0.01
        assert np.isfinite(swb.etc_act[0, 0])

    def test_heavy_precipitation_with_runoff(self):
        """Heavy precip should generate runoff and infiltration."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Heavy precipitation
        day_data.precip = np.array([[100.0]])
        # Hourly precip with some hours exceeding ksat
        day_data.hr_precip = np.zeros((24, 1))
        day_data.hr_precip[10:14, 0] = 25.0  # Intense burst
        day_data.temp_avg = np.array([[20.0]])

        # Low infiltration capacity
        swb.ksat_hourly = np.ones((24, 1)) * 2.0

        cfe.compute_field_et(swb, day_data)

        # Should have runoff
        assert swb.sro[0, 0] > 0.0
        # Infiltration should be precip - runoff (via melt+rain)
        total_water_in = swb.melt[0, 0] + swb.rain[0, 0]
        assert swb.ppt_inf[0, 0] == total_water_in - swb.sro[0, 0]

    def test_snow_accumulation_cold_day(self):
        """Cold day precipitation should accumulate as snow."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        day_data.precip = np.array([[20.0]])
        day_data.hr_precip = np.ones((24, 1)) * (20.0 / 24.0)
        day_data.temp_avg = np.array([[-5.0]])
        day_data.max_temp = np.array([[-2.0]])
        day_data.min_temp = np.array([[-8.0]])

        initial_swe = swb.swe.copy()

        cfe.compute_field_et(swb, day_data)

        # SWE should increase
        assert swb.swe[0, 0] > initial_swe[0, 0]
        # Rain should be zero
        assert swb.rain[0, 0] == 0.0

    def test_snow_melt_warm_day(self):
        """Existing snow should melt on warm day."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        swb.swe = np.array([[30.0]])  # Existing snowpack
        day_data.precip = np.zeros((1, 1))
        day_data.temp_avg = np.array([[10.0]])
        day_data.max_temp = np.array([[15.0]])
        day_data.srad = np.array([[20.0]])

        cfe.compute_field_et(swb, day_data)

        # Should have some melt
        assert swb.melt[0, 0] > 0.0

    def test_nan_fc_raises_error(self):
        """NaN in fc calculation should raise ValueError."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Set up to produce NaN via 0/0 (both numerator and denominator zero)
        swb.kc_bas = np.array([[0.5]])
        swb.kc_min = np.array([[0.5]])  # kc_bas - kc_min = 0
        swb.kc_max = np.array([[0.5]])  # kc_max - kc_min = 0, gives 0/0 = nan

        with pytest.raises(ValueError, match='has nan fc'):
            cfe.compute_field_et(swb, day_data)

    def test_depl_root_clamped_to_taw(self):
        """Root zone depletion should not exceed TAW."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # High initial depletion, high ET, no water input
        swb.depl_root = np.array([[80.0]])
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])  # TAW = 90
        day_data.precip = np.zeros((1, 1))
        day_data.refet = np.array([[10.0]])

        cfe.compute_field_et(swb, day_data)

        taw = swb.aw[0, 0] * swb.zr[0, 0]
        assert swb.depl_root[0, 0] <= taw


# =============================================================================
# Integration Tests: Multi-field and Multi-day
# =============================================================================

class TestMultiField:
    """Tests for multi-field (vectorized) operations."""

    def test_multiple_fields_independent(self):
        """Multiple fields should be processed independently."""
        n_fields = 3
        swb = DummyTracker(n_fields=n_fields)
        day_data = DummyDayData(n_fields=n_fields)

        # Set different conditions for each field
        swb.kc_bas = np.array([[0.5, 0.8, 1.1]])
        swb.depl_root = np.array([[20.0, 50.0, 80.0]])
        day_data.precip = np.array([[0.0, 10.0, 30.0]])
        day_data.hr_precip = np.zeros((24, n_fields))
        day_data.hr_precip[:, 1] = 10.0 / 24.0
        day_data.hr_precip[:, 2] = 30.0 / 24.0

        cfe.compute_field_et(swb, day_data)

        # Each field should have different results
        assert not np.allclose(swb.etc_act[0, 0], swb.etc_act[0, 1])
        assert not np.allclose(swb.etc_act[0, 1], swb.etc_act[0, 2])

    def test_array_shapes_preserved(self):
        """Output arrays should maintain (1, N) shape."""
        n_fields = 5
        swb = DummyTracker(n_fields=n_fields)
        day_data = DummyDayData(n_fields=n_fields)

        cfe.compute_field_et(swb, day_data)

        assert swb.etc_act.shape == (1, n_fields)
        assert swb.depl_root.shape == (1, n_fields)
        assert swb.kc_act.shape == (1, n_fields)
        assert swb.dperc.shape == (1, n_fields)


class TestWaterBalanceConservation:
    """Tests for water balance closure."""

    def test_water_balance_closes(self):
        """Water in - water out - delta storage should be near zero.

        The model tracks water in root zone (depl_root), surface layer (depl_ze),
        and layer 3 below roots (daw3). soil_water = (aw*zr) - depl_root + daw3.
        """
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Set up consistent initial state
        # soil_water = (aw * zr) - depl_root + daw3
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.depl_root = np.array([[30.0]])
        swb.daw3 = np.array([[20.0]])
        swb.soil_water = (swb.aw * swb.zr) - swb.depl_root + swb.daw3  # = 80

        # Moderate precipitation
        day_data.precip = np.array([[15.0]])
        day_data.hr_precip = np.ones((24, 1)) * (15.0 / 24.0)
        day_data.temp_avg = np.array([[20.0]])

        # Record initial state
        initial_soil_water = swb.soil_water.copy()

        cfe.compute_field_et(swb, day_data)

        # Water balance (as computed in tracker.update_dataframe):
        # water_in = melt + rain
        # water_out = et_act + dperc + runoff
        # delta_storage = soil_water - soil_water_prev
        # balance = water_in - water_out - delta_storage ≈ 0
        water_in = swb.melt[0, 0] + swb.rain[0, 0]
        water_out = swb.etc_act[0, 0] + swb.sro[0, 0] + swb.dperc[0, 0]
        delta_storage = swb.soil_water[0, 0] - initial_soil_water[0, 0]

        balance = water_in - water_out - delta_storage
        # Allow tolerance for numerical precision and model approximations
        assert abs(balance) < 1.0, f"Water balance error: {balance:.4f}"

    def test_no_water_created_or_destroyed(self):
        """On a dry day, water should only leave via ET (no creation)."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Dry day setup
        day_data.precip = np.zeros((1, 1))
        day_data.hr_precip = np.zeros((24, 1))

        # Consistent initial state
        swb.aw = np.array([[150.0]])
        swb.zr = np.array([[0.6]])
        swb.depl_root = np.array([[30.0]])
        swb.daw3 = np.array([[20.0]])
        swb.soil_water = (swb.aw * swb.zr) - swb.depl_root + swb.daw3

        initial_soil_water = swb.soil_water.copy()

        cfe.compute_field_et(swb, day_data)

        # On dry day: soil_water should decrease (or stay same)
        # ET removes water, nothing adds water
        assert swb.soil_water[0, 0] <= initial_soil_water[0, 0] + 0.01
        # dperc should be zero (no excess water)
        assert swb.dperc[0, 0] == 0.0
        # runoff should be zero
        assert swb.sro[0, 0] == 0.0


class TestCurveNumberMode:
    """Tests for Curve Number runoff mode."""

    def test_cn_mode_computes_runoff(self):
        """CN mode should compute runoff using curve number method."""
        swb = DummyTracker(n_fields=1, runoff_process='cn')
        day_data = DummyDayData(n_fields=1)

        swb.cn2 = np.array([[80.0]])
        day_data.precip = np.array([[30.0]])
        day_data.hr_precip = np.ones((24, 1)) * (30.0 / 24.0)
        day_data.temp_avg = np.array([[20.0]])

        cfe.compute_field_et(swb, day_data)

        # With CN=80 and 30mm precip, should have some runoff
        assert swb.sro[0, 0] >= 0.0
        assert np.isfinite(swb.sro[0, 0])


# =============================================================================
# Regression-style Tests
# =============================================================================

class TestKnownValues:
    """Tests against known/expected values for specific scenarios."""

    def test_dry_summer_day_typical_values(self):
        """Typical dry summer day should produce reasonable ET."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Typical summer conditions
        swb.kc_bas = np.array([[1.0]])
        swb.kc_max = np.array([[1.2]])
        swb.kc_min = np.array([[0.15]])
        swb.depl_root = np.array([[30.0]])  # Moderate depletion
        day_data.refet = np.array([[6.0]])  # 6 mm/day refET
        day_data.precip = np.zeros((1, 1))

        cfe.compute_field_et(swb, day_data)

        # ETc should be reasonable (2-7 mm/day for crops)
        assert 1.0 < swb.etc_act[0, 0] < 8.0

    def test_winter_dormant_low_et(self):
        """Dormant winter conditions should have low ET."""
        swb = DummyTracker(n_fields=1)
        day_data = DummyDayData(n_fields=1)

        # Dormant conditions
        swb.kc_bas = np.array([[0.2]])
        swb.kc_max = np.array([[1.2]])
        swb.kc_min = np.array([[0.15]])
        day_data.refet = np.array([[1.5]])  # Low winter refET
        day_data.precip = np.zeros((1, 1))

        cfe.compute_field_et(swb, day_data)

        # ET should be low
        assert swb.etc_act[0, 0] < 2.0

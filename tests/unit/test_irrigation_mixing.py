"""Unit tests for irrigation fraction tracking kernels and integration.

Tests verify:
1. Kernels compile correctly with numba
2. Fraction bounds [0, 1] are enforced
3. Conservation laws hold (mass balance)
4. Mixing rules are correct
5. Edge cases (empty pools, pure sources) handled properly
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from swimrs.process.kernels.irrigation_tracking import (
    redistribute_irrigation_fractions,
    transfer_fraction_with_water,
    update_irrigation_fraction_l3,
    update_irrigation_fraction_root,
)


def root_water(awc, zr, depl_root):
    """Return root-zone water for kernel fixtures without a TEW floor."""
    return np.maximum(awc * zr - depl_root, 0.0)


class TestUpdateIrrigationFractionRoot:
    """Tests for root zone irrigation fraction update kernel."""

    def test_pure_precipitation(self):
        """With only precipitation, fraction stays 0."""
        n = 3
        awc = np.full(n, 150.0)  # mm/m
        zr = np.full(n, 0.5)  # m
        depl_root = np.full(n, 20.0)  # mm
        irr_frac_root = np.zeros(n)
        infiltration = np.full(n, 10.0)  # mm
        irr_sim = np.zeros(n)
        gw_sim = np.zeros(n)
        eta = np.full(n, 5.0)  # mm
        dperc = np.zeros(n)

        frac_new, et_irr, drainage_irr = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        assert_array_almost_equal(frac_new, 0.0)
        assert_array_almost_equal(et_irr, 0.0)
        assert_array_almost_equal(drainage_irr, 0.0)

    def test_pure_irrigation(self):
        """With only irrigation input, fraction approaches 1."""
        n = 3
        awc = np.full(n, 150.0)
        zr = np.full(n, 0.5)
        depl_root = np.full(n, 20.0)
        irr_frac_root = np.ones(n)  # Start with all irrigation water
        infiltration = np.zeros(n)
        irr_sim = np.full(n, 30.0)  # Add more irrigation
        gw_sim = np.zeros(n)
        eta = np.full(n, 5.0)
        dperc = np.zeros(n)

        frac_new, et_irr, drainage_irr = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        # Fraction should stay 1 (only irrigation in, irrigation out)
        assert_array_almost_equal(frac_new, 1.0)
        assert_array_almost_equal(et_irr, eta)  # All ET is from irrigation
        assert_array_almost_equal(drainage_irr, 0.0)

    def test_mixed_sources(self):
        """With 50/50 mix, fraction should be ~0.5."""
        n = 3
        awc = np.full(n, 150.0)
        zr = np.full(n, 0.5)
        depl_root = np.full(n, 50.0)  # Water = 75 - 50 = 25 mm
        irr_frac_root = np.full(n, 0.5)  # Start at 50%
        infiltration = np.full(n, 10.0)  # 10 mm natural
        irr_sim = np.full(n, 10.0)  # 10 mm irrigation
        gw_sim = np.zeros(n)
        eta = np.full(n, 2.0)
        dperc = np.zeros(n)

        frac_new, et_irr, _ = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        # Water before: 75 - 50 = 25 mm, 50% irrigation = 12.5 mm irr water
        # Add 10 mm natural infiltration, then ET removes 2 * (12.5 / 35)
        # irrigation water. Adding 10 mm irrigation gives ~21.79 / 43.
        assert np.all(frac_new >= 0.4)
        assert np.all(frac_new <= 0.6)

    def test_et_carries_fraction_after_natural_infiltration(self):
        """Same-day natural infiltration dilutes the source fraction of ET."""
        n = 1
        awc = np.array([100.0])
        zr = np.array([1.0])
        depl_root = np.array([90.0])  # 10 mm, all irrigation-source
        irr_frac_root = np.array([1.0])
        infiltration = np.array([10.0])
        irr_sim = np.array([0.0])
        gw_sim = np.array([0.0])
        eta = np.array([10.0])
        dperc = np.array([0.0])

        frac_new, et_irr, drainage_irr = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        assert_array_almost_equal(et_irr, 5.0)
        assert_array_almost_equal(frac_new, 0.5)
        assert_array_almost_equal(drainage_irr, 0.0)

    def test_irrigation_forced_drainage_is_proportional(self):
        """Irrigation-forced drainage inherits the mixed root fraction."""
        awc = np.array([100.0])
        zr = np.array([1.0])
        depl_root = np.array([50.0])  # 50 mm natural water
        irr_frac_root = np.array([0.0])
        infiltration = np.array([0.0])
        irr_sim = np.array([50.0])
        gw_sim = np.array([0.0])
        eta = np.array([0.0])
        root_drainage = np.array([5.0])  # 10% irrigation-forced drainage

        frac_new, et_irr, drainage_irr = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            root_drainage,
        )

        assert_array_almost_equal(et_irr, 0.0)
        assert_array_almost_equal(drainage_irr, 2.5)
        assert_array_almost_equal(frac_new, 0.5)

    def test_fraction_bounds(self):
        """Fraction always in [0, 1]."""
        n = 5
        awc = np.full(n, 150.0)
        zr = np.full(n, 0.5)
        depl_root = np.random.uniform(0, 50, n)
        irr_frac_root = np.random.uniform(0, 1, n)
        infiltration = np.random.uniform(0, 20, n)
        irr_sim = np.random.uniform(0, 30, n)
        gw_sim = np.random.uniform(0, 10, n)
        eta = np.random.uniform(0, 10, n)
        dperc = np.random.uniform(0, 5, n)

        frac_new, et_irr, drainage_irr = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        assert np.all(frac_new >= 0.0)
        assert np.all(frac_new <= 1.0)
        assert np.all(et_irr >= 0.0)
        assert np.all(drainage_irr >= 0.0)

    def test_empty_pool(self):
        """Empty pool results in fraction 0."""
        n = 1
        awc = np.array([100.0])
        zr = np.array([0.1])  # TAW = 10 mm
        depl_root = np.array([10.0])  # Water = 0
        irr_frac_root = np.array([0.5])
        infiltration = np.array([0.0])
        irr_sim = np.array([0.0])
        gw_sim = np.array([0.0])
        eta = np.array([0.0])
        dperc = np.array([0.0])

        frac_new, et_irr, drainage_irr = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        assert_array_almost_equal(frac_new, 0.0)
        assert_array_almost_equal(et_irr, 0.0)
        assert_array_almost_equal(drainage_irr, 0.0)

    def test_gw_subsidy_dilutes_fraction(self):
        """Groundwater subsidy (natural water) dilutes irrigation fraction."""
        n = 1
        awc = np.array([100.0])
        zr = np.array([1.0])
        depl_root = np.array([50.0])  # Water = 50 mm
        irr_frac_root = np.array([1.0])  # All irrigation
        infiltration = np.array([0.0])
        irr_sim = np.array([0.0])
        gw_sim = np.array([50.0])  # Add 50 mm natural water
        eta = np.array([0.0])
        dperc = np.array([0.0])

        frac_new, et_irr, _ = update_irrigation_fraction_root(
            root_water(awc, zr, depl_root),
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        # 50 mm irr + 50 mm natural = 50% irrigation
        assert_array_almost_equal(frac_new, 0.5)


class TestUpdateIrrigationFractionL3:
    """Tests for layer 3 irrigation fraction update kernel."""

    def test_no_overflow(self):
        """When no overflow, fraction mixes but no dperc_irr."""
        n = 1
        daw3 = np.array([10.0])  # 10 mm in L3
        irr_frac_l3 = np.array([0.2])
        gross_dperc = np.array([5.0])  # 5 mm entering
        irrigation_inflow = np.array([4.0])  # 4 of 5 mm is irrigation-source
        dperc_out = np.array([0.0])  # No overflow

        frac_new, dperc_irr = update_irrigation_fraction_l3(
            daw3, irr_frac_l3, gross_dperc, irrigation_inflow, dperc_out
        )

        # Mixed: (10*0.2 + 5*0.8) / 15 = (2 + 4) / 15 = 0.4
        assert_array_almost_equal(frac_new, 0.4)
        assert_array_almost_equal(dperc_irr, 0.0)

    def test_with_overflow(self):
        """Overflow carries mixed fraction."""
        n = 1
        daw3 = np.array([10.0])
        irr_frac_l3 = np.array([0.5])
        gross_dperc = np.array([10.0])
        irrigation_inflow = np.array([5.0])
        dperc_out = np.array([5.0])  # 5 mm overflow

        frac_new, dperc_irr = update_irrigation_fraction_l3(
            daw3, irr_frac_l3, gross_dperc, irrigation_inflow, dperc_out
        )

        # All at 50%, so dperc_irr = 5 * 0.5 = 2.5
        assert_array_almost_equal(dperc_irr, 2.5)
        assert_array_almost_equal(frac_new, 0.5)

    def test_fraction_bounds(self):
        """Fraction always in [0, 1]."""
        n = 10
        daw3 = np.random.uniform(0, 50, n)
        irr_frac_l3 = np.random.uniform(0, 1, n)
        gross_dperc = np.random.uniform(0, 30, n)
        irrigation_inflow = gross_dperc * np.random.uniform(0, 1, n)
        dperc_out = np.minimum(np.random.uniform(0, 20, n), daw3 + gross_dperc)

        frac_new, dperc_irr = update_irrigation_fraction_l3(
            daw3, irr_frac_l3, gross_dperc, irrigation_inflow, dperc_out
        )

        assert np.all(frac_new >= 0.0)
        assert np.all(frac_new <= 1.0)


class TestRedistributeIrrigationFractions:
    """Tests for source transfer when the root-zone boundary moves."""

    def test_tew_buffer_change_is_natural(self):
        """Changing only the effective TEW buffer preserves irrigation mass."""
        root_frac, l3_frac = redistribute_irrigation_fractions(
            np.array([10.0]),
            np.array([0.4]),
            np.array([5.0]),
            np.array([0.2]),
            np.array([6.0]),
            np.array([5.0]),
        )

        assert_array_almost_equal(root_frac, 2.0 / 3.0)
        assert_array_almost_equal(l3_frac, 0.2)
        irrigation_before = 10.0 * 0.4 + 5.0 * 0.2
        irrigation_after = 6.0 * root_frac[0] + 5.0 * l3_frac[0]
        assert_array_almost_equal(irrigation_after, irrigation_before)

    def test_root_growth_moves_layer3_source(self):
        """Actual layer-3 withdrawal carries the layer-3 source fraction."""
        root_frac, l3_frac = redistribute_irrigation_fractions(
            np.array([7.0]),
            np.array([0.2]),
            np.array([10.0]),
            np.array([0.8]),
            np.array([9.0]),
            np.array([8.0]),
        )

        assert_array_almost_equal(root_frac, 1.0 / 3.0)
        assert_array_almost_equal(l3_frac, 0.8)
        irrigation_before = 7.0 * 0.2 + 10.0 * 0.8
        irrigation_after = 9.0 * root_frac[0] + 8.0 * l3_frac[0]
        assert_array_almost_equal(irrigation_after, irrigation_before)

    def test_growth_into_empty_effective_root_pool_is_capacity_safe(self):
        """Source mass stays represented when root growth only reduces deficit."""
        root_frac, l3_frac = redistribute_irrigation_fractions(
            np.array([0.0]),
            np.array([0.0]),
            np.array([10.0]),
            np.array([0.65]),
            np.array([0.0]),
            np.array([9.0]),
        )

        assert_array_almost_equal(root_frac, 0.0)
        assert_array_almost_equal(l3_frac, 6.5 / 9.0)
        irrigation_before = 10.0 * 0.65
        irrigation_after = 9.0 * l3_frac[0]
        assert_array_almost_equal(irrigation_after, irrigation_before)

    def test_root_recession_moves_root_source(self):
        """Actual layer-3 addition carries the root-zone source fraction."""
        root_frac, l3_frac = redistribute_irrigation_fractions(
            np.array([9.0]),
            np.array([2.0 / 3.0]),
            np.array([8.0]),
            np.array([0.25]),
            np.array([10.0]),
            np.array([10.0]),
        )

        assert_array_almost_equal(root_frac, 7.0 / 15.0)
        assert_array_almost_equal(l3_frac, 1.0 / 3.0)
        irrigation_before = 9.0 * (2.0 / 3.0) + 8.0 * 0.25
        irrigation_after = 10.0 * root_frac[0] + 10.0 * l3_frac[0]
        assert_array_almost_equal(irrigation_after, irrigation_before)


class TestTransferFractionWithWater:
    """Tests for fraction transfer during root growth."""

    def test_no_transfer(self):
        """No transfer means fractions unchanged."""
        n = 1
        water_from = np.array([50.0])
        frac_from = np.array([0.3])
        water_to = np.array([20.0])
        frac_to = np.array([0.7])
        transfer = np.array([0.0])

        frac_from_new, frac_to_new = transfer_fraction_with_water(
            water_from, frac_from, water_to, frac_to, transfer
        )

        assert_array_almost_equal(frac_from_new, 0.3)
        assert_array_almost_equal(frac_to_new, 0.7)

    def test_transfer_mixes_fractions(self):
        """Transfer causes mixing in destination pool."""
        n = 1
        water_from = np.array([50.0])
        frac_from = np.array([0.0])  # Natural water
        water_to = np.array([50.0])
        frac_to = np.array([1.0])  # All irrigation
        transfer = np.array([50.0])  # Transfer all

        frac_from_new, frac_to_new = transfer_fraction_with_water(
            water_from, frac_from, water_to, frac_to, transfer
        )

        # Source depleted
        assert_array_almost_equal(frac_from_new, 0.0)
        # Destination: (50*1.0 + 50*0.0) / 100 = 0.5
        assert_array_almost_equal(frac_to_new, 0.5)

    def test_fraction_bounds(self):
        """Fractions always in [0, 1]."""
        n = 10
        water_from = np.random.uniform(10, 100, n)
        frac_from = np.random.uniform(0, 1, n)
        water_to = np.random.uniform(10, 100, n)
        frac_to = np.random.uniform(0, 1, n)
        transfer = np.random.uniform(0, 50, n)

        frac_from_new, frac_to_new = transfer_fraction_with_water(
            water_from, frac_from, water_to, frac_to, transfer
        )

        assert np.all(frac_from_new >= 0.0)
        assert np.all(frac_from_new <= 1.0)
        assert np.all(frac_to_new >= 0.0)
        assert np.all(frac_to_new <= 1.0)


class TestConservation:
    """Tests verifying irrigation water conservation."""

    def test_irrigation_conservation_single_day(self):
        """Irrigation water is conserved across a single day."""
        n = 1
        awc = np.array([150.0])
        zr = np.array([0.5])  # TAW = 75 mm

        # Initial state: 50 mm water, 40% irrigation
        depl_root = np.array([25.0])  # water = 75 - 25 = 50 mm
        irr_frac_root = np.array([0.4])

        infiltration = np.array([10.0])  # Natural
        irr_sim = np.array([20.0])  # Irrigation
        gw_sim = np.array([0.0])
        eta = np.array([15.0])
        dperc = np.array([5.0])

        # Initial irrigation water
        water_before = awc * zr - depl_root
        irr_water_before = water_before * irr_frac_root

        frac_new, et_irr, dperc_irr_root = update_irrigation_fraction_root(
            water_before,
            irr_frac_root,
            infiltration,
            irr_sim,
            gw_sim,
            eta,
            dperc,
        )

        # Natural infiltration mixes before ET; irrigation mixes before drainage.
        water_after = water_before + infiltration - eta + irr_sim - dperc
        irr_water_after = water_after * frac_new

        irr_balance = irr_water_before + irr_sim - et_irr - dperc_irr_root
        assert_array_almost_equal(irr_balance, irr_water_after, decimal=12)

    def test_limiting_case_no_irrigation(self):
        """With no irrigation, et_irr = 0 and fractions decay."""
        n = 1
        awc = np.array([100.0])
        zr = np.array([1.0])
        depl_root = np.array([50.0])
        irr_frac_root = np.array([0.5])

        # Only precipitation, no irrigation
        for _ in range(10):
            infiltration = np.array([5.0])
            irr_sim = np.array([0.0])
            gw_sim = np.array([0.0])
            eta = np.array([3.0])
            dperc = np.array([0.0])

            irr_frac_root, et_irr, _ = update_irrigation_fraction_root(
                root_water(awc, zr, depl_root),
                irr_frac_root,
                infiltration,
                irr_sim,
                gw_sim,
                eta,
                dperc,
            )

        # Fraction should decrease toward 0
        assert irr_frac_root[0] < 0.3

    def test_limiting_case_only_irrigation(self):
        """With only irrigation input, fraction approaches 1."""
        n = 1
        awc = np.array([100.0])
        zr = np.array([1.0])
        depl_root = np.array([50.0])
        irr_frac_root = np.array([0.5])

        # Only irrigation, no precipitation
        for _ in range(10):
            infiltration = np.array([0.0])
            irr_sim = np.array([10.0])
            gw_sim = np.array([0.0])
            eta = np.array([8.0])
            dperc = np.array([0.0])

            irr_frac_root, et_irr, _ = update_irrigation_fraction_root(
                root_water(awc, zr, depl_root),
                irr_frac_root,
                infiltration,
                irr_sim,
                gw_sim,
                eta,
                dperc,
            )

        # Fraction should increase toward 1
        assert irr_frac_root[0] > 0.7


class TestStateInitialization:
    """Tests for irrigation fraction state initialization."""

    def test_from_spinup_without_source_state_starts_non_irrigation(self):
        """Irrigation status does not imply an initial source composition."""
        from swimrs.process.state import WaterBalanceState

        n = 3
        irr_status = np.array([True, True, False])

        state = WaterBalanceState.from_spinup(
            n_fields=n,
            depl_root=np.zeros(n),
            swe=np.zeros(n),
            kr=np.ones(n),
            ks=np.ones(n),
            zr=np.full(n, 0.3),
            irr_status=irr_status,
        )

        assert_array_almost_equal(state.irr_frac_root, 0.0)
        assert_array_almost_equal(state.irr_frac_l3, 0.0)

    def test_from_spinup_with_provided_fractions(self):
        """Provided spinup fractions override irr_status default."""
        from swimrs.process.state import WaterBalanceState

        n = 2
        irr_status = np.array([True, True])
        provided_frac = np.array([0.3, 0.7])

        state = WaterBalanceState.from_spinup(
            n_fields=n,
            depl_root=np.zeros(n),
            swe=np.zeros(n),
            kr=np.ones(n),
            ks=np.ones(n),
            zr=np.full(n, 0.3),
            irr_frac_root=provided_frac.copy(),
            irr_frac_l3=provided_frac.copy(),
            irr_status=irr_status,
        )

        assert_array_almost_equal(state.irr_frac_root, [0.3, 0.7])
        assert_array_almost_equal(state.irr_frac_l3, [0.3, 0.7])

    def test_copy_preserves_fractions(self):
        """State copy preserves irrigation fractions."""
        from swimrs.process.state import WaterBalanceState

        n = 2
        state = WaterBalanceState(n_fields=n)
        state.irr_frac_root = np.array([0.25, 0.75])
        state.irr_frac_l3 = np.array([0.15, 0.85])

        state_copy = state.copy()

        assert_array_almost_equal(state_copy.irr_frac_root, [0.25, 0.75])
        assert_array_almost_equal(state_copy.irr_frac_l3, [0.15, 0.85])

        # Verify it's a deep copy
        state_copy.irr_frac_root[0] = 0.99
        assert state.irr_frac_root[0] == 0.25

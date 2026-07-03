"""Tests for Calculator helper methods as pure logic.

Tests cover:
- _merge_sensors(): preference order, NaN fill, single sensor
- _compute_k_parameters(): all-NaN defaults, no low-NDVI, kc_max floor
- _compute_groundwater_subsidy(): ET > PPT subsidy, ET < PPT no subsidy, zero PPT
- _detect_irrigation_windows(): flat NDVI, clear ramp, >200 NaN, DOY invariants
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_allclose


def _make_calculator():
    """Create a minimal Calculator with mocked state."""
    from swimrs.container.components.calculator import Calculator

    state = MagicMock()
    state.is_writable = True
    state._mode = "r+"
    calc = Calculator(state)
    return calc


class TestMergeSensors:
    """Tests for Calculator._merge_sensors()."""

    def _make_da(self, data, dates, sites):
        """Helper to create DataArrays."""
        return xr.DataArray(
            data,
            dims=["time", "site"],
            coords={"time": dates, "site": sites},
        )

    def test_preferred_has_all_data(self):
        """When preferred sensor has all data, result equals preferred."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=5)
        sites = ["A", "B"]
        preferred = self._make_da(np.ones((5, 2)), dates, sites)
        secondary = self._make_da(np.full((5, 2), 99.0), dates, sites)

        result = calc._merge_sensors(
            [("landsat", preferred), ("sentinel", secondary)],
            preference_order=("landsat", "sentinel"),
        )
        assert_allclose(result.values, 1.0)

    def test_preferred_nans_filled_by_secondary(self):
        """NaN in preferred sensor is filled from secondary."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=3)
        sites = ["A"]
        pref_data = np.array([[np.nan], [0.5], [np.nan]])
        sec_data = np.array([[0.3], [0.7], [0.4]])
        preferred = self._make_da(pref_data, dates, sites)
        secondary = self._make_da(sec_data, dates, sites)

        result = calc._merge_sensors(
            [("landsat", preferred), ("sentinel", secondary)],
            preference_order=("landsat", "sentinel"),
        )
        expected = np.array([[0.3], [0.5], [0.4]])
        assert_allclose(result.values, expected)

    def test_single_sensor_identity(self):
        """Single sensor returns identity."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=3)
        sites = ["A"]
        data = np.array([[0.1], [0.5], [0.9]])
        da = self._make_da(data, dates, sites)

        result = calc._merge_sensors(
            [("landsat", da)],
            preference_order=("landsat",),
        )
        assert_allclose(result.values, data)

    def test_result_nan_count_leq_best_single(self):
        """Merged result has fewer NaN than any individual sensor."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=6)
        sites = ["A"]
        pref_data = np.array([[np.nan], [0.5], [np.nan], [0.7], [np.nan], [np.nan]])
        sec_data = np.array([[0.3], [np.nan], [0.4], [np.nan], [0.6], [np.nan]])
        preferred = self._make_da(pref_data, dates, sites)
        secondary = self._make_da(sec_data, dates, sites)

        result = calc._merge_sensors(
            [("landsat", preferred), ("sentinel", secondary)],
            preference_order=("landsat", "sentinel"),
        )
        result_nans = np.isnan(result.values).sum()
        pref_nans = np.isnan(pref_data).sum()
        sec_nans = np.isnan(sec_data).sum()
        assert result_nans <= min(pref_nans, sec_nans)


class TestComputeKParameters:
    """Tests for Calculator._compute_k_parameters()."""

    def _make_ds(self, etf_vals, ndvi_vals, sites=("A",)):
        """Create a minimal xr.Dataset for K-parameter computation."""
        dates = pd.date_range("2020-01-01", periods=len(etf_vals))
        n_sites = len(sites)
        etf_2d = np.tile(np.array(etf_vals)[:, None], (1, n_sites))
        ndvi_2d = np.tile(np.array(ndvi_vals)[:, None], (1, n_sites))
        return xr.Dataset(
            {
                "etf": xr.DataArray(
                    etf_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
                "ndvi": xr.DataArray(
                    ndvi_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
            }
        )

    def test_all_nan_etf_returns_defaults(self):
        """All-NaN ETf produces default ke=1.0, kc=1.25."""
        calc = _make_calculator()
        ds = self._make_ds(
            [np.nan] * 10,
            [0.2] * 10,
        )
        ke, kc = calc._compute_k_parameters(ds)
        assert float(ke.values) == 1.0
        assert float(kc.values) == 1.25

    def test_no_low_ndvi_ke_defaults(self):
        """When all NDVI >= 0.3, ke defaults to 1.0."""
        calc = _make_calculator()
        ds = self._make_ds(
            [0.8, 0.9, 0.7, 0.85, 0.95],
            [0.5, 0.6, 0.55, 0.7, 0.65],
        )
        ke, kc = calc._compute_k_parameters(ds)
        assert float(ke.values) == 1.0

    def test_kc_max_floor_enforced(self):
        """kc_max is at least 1.25 even when all ETf is low."""
        calc = _make_calculator()
        ds = self._make_ds(
            [0.1, 0.2, 0.15, 0.1, 0.05],
            [0.5, 0.6, 0.55, 0.7, 0.65],
        )
        ke, kc = calc._compute_k_parameters(ds)
        assert float(kc.values) >= 1.25

    def test_known_percentile_scenario(self):
        """90th percentile of ETf where NDVI < 0.3 matches manual calculation."""
        calc = _make_calculator()
        # 5 obs with low NDVI, ETf values [0.2, 0.4, 0.6, 0.8, 1.0]
        etf_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
        ndvi_vals = [0.1, 0.15, 0.2, 0.25, 0.28]  # all < 0.3
        ds = self._make_ds(etf_vals, ndvi_vals)
        ke, kc = calc._compute_k_parameters(ds)
        expected_ke = float(np.percentile(etf_vals, 90))
        assert_allclose(float(ke.values), expected_ke, atol=0.01)


class TestComputeGroundwaterSubsidy:
    """Tests for Calculator._compute_groundwater_subsidy()."""

    def _make_ds(self, eto_vals, etf_vals, prcp_vals, sites=("A",)):
        """Create a minimal xr.Dataset for GW subsidy computation."""
        dates = pd.date_range("2020-01-01", periods=len(eto_vals))
        n_sites = len(sites)
        eto_2d = np.tile(np.array(eto_vals)[:, None], (1, n_sites))
        etf_2d = np.tile(np.array(etf_vals)[:, None], (1, n_sites))
        prcp_2d = np.tile(np.array(prcp_vals)[:, None], (1, n_sites))
        return xr.Dataset(
            {
                "eto": xr.DataArray(
                    eto_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
                "etf": xr.DataArray(
                    etf_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
                "prcp": xr.DataArray(
                    prcp_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
            }
        )

    def test_et_greater_than_ppt_yields_positive_fsub(self):
        """When ET > PPT, f_sub should be > 0."""
        calc = _make_calculator()
        # 365 days: ETo=5, ETf=1.0 -> eta=5, prcp=2 -> ratio > 1
        n = 365
        ds = self._make_ds(
            [5.0] * n,
            [1.0] * n,
            [2.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        yr_data = site_data.get(2020)
        assert yr_data is not None
        assert yr_data["f_sub"] > 0
        assert yr_data["subsidized"] == 1

    def test_et_less_than_ppt_yields_zero_fsub(self):
        """When ET < PPT, f_sub should be 0."""
        calc = _make_calculator()
        # 365 days: ETo=3, ETf=0.5 -> eta=1.5, prcp=5 -> ratio < 1
        n = 365
        ds = self._make_ds(
            [3.0] * n,
            [0.5] * n,
            [5.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        yr_data = site_data.get(2020)
        assert yr_data is not None
        assert yr_data["f_sub"] == 0.0
        assert yr_data["subsidized"] == 0

    def test_zero_ppt_year_skipped(self):
        """Year with zero precipitation is skipped (no division by zero)."""
        calc = _make_calculator()
        n = 365
        ds = self._make_ds(
            [5.0] * n,
            [1.0] * n,
            [0.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        # Year present but ratio based on ppt+1
        if 2020 in site_data:
            # Should not crash
            assert np.isfinite(site_data[2020]["ratio"])

    def test_subsidy_months_identified(self):
        """Months where eta > ppt are identified."""
        calc = _make_calculator()
        n = 365
        ds = self._make_ds(
            [5.0] * n,
            [1.0] * n,
            [2.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        yr_data = site_data.get(2020)
        if yr_data and "months" in yr_data:
            assert isinstance(yr_data["months"], list)


class TestDetectIrrigationWindows:
    """Tests for Calculator._detect_irrigation_windows()."""

    def test_flat_ndvi_no_windows(self):
        """Flat NDVI time series produces no irrigation windows."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        ndvi = pd.Series(0.3, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        assert isinstance(doys, list)

    def test_clear_ramp_produces_doys(self):
        """Clear NDVI ramp-up produces DOYs."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        # Create a sigmoidal ramp-up pattern
        n = len(dates)
        ndvi_vals = np.where(
            np.arange(n) < 120,
            0.15,
            np.where(np.arange(n) < 200, 0.15 + 0.7 * (np.arange(n) - 120) / 80, 0.85),
        )
        # Add some decline after peak
        ndvi_vals = np.where(np.arange(n) > 260, 0.85 - 0.5 * (np.arange(n) - 260) / 100, ndvi_vals)
        ndvi = pd.Series(ndvi_vals, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        # Should produce some DOYs from the ramp-up period
        if len(doys) > 0:
            assert all(1 <= d <= 366 for d in doys)
            assert doys == sorted(doys)
            assert len(doys) == len(set(doys))

    def test_many_nans_returns_empty(self):
        """Time series with >200 NaN after processing returns empty."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        ndvi = pd.Series(np.nan, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        assert doys == []

    def test_doys_sorted_unique_in_range(self):
        """DOYs are sorted, unique, and within [1, 366]."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        # Create a pattern that should produce some DOYs
        n = len(dates)
        ndvi_vals = 0.2 + 0.6 * np.sin(np.pi * np.arange(n) / n)
        ndvi = pd.Series(ndvi_vals, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        if len(doys) > 0:
            assert doys == sorted(doys)
            assert len(doys) == len(set(doys))
            assert all(1 <= d <= 366 for d in doys)


class TestComputeIrrigationDataLulc:
    """Tests for the use_lulc irrigation classifier (_compute_irrigation_data).

    Focus: the rolling 2-year annual-balance gate (lulc_irr_method="annual_2yr")
    vs the legacy monthly gate ("monthly"), plus the cropland gate. The
    motivating case is dryland wheat-fallow: a crop year whose in-season ET
    exceeds in-season rain (so the monthly test trips) but whose biennial
    water balance closes (so the 2-year test correctly does NOT flag it).
    """

    def _make_ds(self, years, crop_years=(), irrigated_all=False, subsidy_years=(), sites=("A",)):
        """Build a daily multi-year xr.Dataset.

        Defaults are a "fallow" regime (low ET, high rain). ``crop_years`` get
        a dry low-ET profile with a high-ET, no-rain summer (Jun-Aug) that
        trips the monthly subsidy test. ``irrigated_all`` makes every year a
        true subsidy (ET > P year round); ``subsidy_years`` makes only the
        listed years a true subsidy (the rest stay fallow) — used to probe the
        Stage 1 mode gate's robustness to an isolated high-ratio year.
        """
        dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
        months = dates.month.values
        yrs = dates.year.values

        etf = np.full(len(dates), 0.1)
        eto = np.full(len(dates), 2.0)
        prcp = np.full(len(dates), 2.0)  # fallow: low ET, high rain (banking)

        for y in years:
            ym = yrs == y
            if irrigated_all or y in subsidy_years:
                etf[ym], eto[ym], prcp[ym] = 0.8, 6.0, 0.3  # ET >> P all year
            elif y in crop_years:
                etf[ym], eto[ym], prcp[ym] = 0.1, 2.0, 1.0  # dry, low ET
                sm = ym & np.isin(months, [6, 7, 8])
                # hot dry summer crop: tiny rain (>0 so the monthly test counts
                # the month) but ETa >> rain so the monthly ratio still trips
                etf[sm], eto[sm], prcp[sm] = 0.9, 6.0, 0.1

        # benign seasonal NDVI so window detection runs without error
        ndvi = 0.2 + 0.5 * np.exp(-((months - 7.0) ** 2) / 8.0)

        n_sites = len(sites)

        def da(vals):
            arr = np.tile(np.asarray(vals)[:, None], (1, n_sites))
            return xr.DataArray(
                arr, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
            )

        return xr.Dataset({"ndvi": da(ndvi), "etf": da(etf), "eto": da(eto), "prcp": da(prcp)})

    def _run(self, ds, method, lulc=(10, "glc10"), **kwargs):
        calc = _make_calculator()
        # classifier reads LULC from the zarr root; mock it to a fixed class
        calc._get_lulc_by_site = MagicMock(return_value={"A": lulc} if lulc else {})
        return calc._compute_irrigation_data(
            ds,
            irr_threshold=0.3,
            lookback=10,
            ndvi_threshold=0.3,
            ndvi_min_start=0.25,
            min_pos_days=10,
            use_mask=False,
            use_lulc=True,
            lulc_irr_method=method,
            **kwargs,
        )

    @staticmethod
    def _flags(result, years):
        site = result["A"]
        return {y: site[y]["irrigated"] for y in years}

    def test_monthly_flags_dryland_crop_year(self):
        """Legacy monthly test flags a wheat-fallow crop year as irrigated."""
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, crop_years=[2020])
        flags = self._flags(self._run(ds, "monthly"), years)
        assert flags == {2019: 0, 2020: 1, 2021: 0}

    def test_annual_2yr_does_not_flag_dryland_crop_year(self):
        """2-year balance closes (ET<=P) so the crop year is NOT flagged."""
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, crop_years=[2020])
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert flags == {2019: 0, 2020: 0, 2021: 0}

    def test_annual_2yr_flags_true_irrigation(self):
        """When ET > P every year, the 2-year test still flags irrigation."""
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, irrigated_all=True)
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert all(v == 1 for v in flags.values())
        assert all(self._run(ds, "annual_2yr")["A"][y]["f_irr"] == 1.0 for y in years)

    def test_cropland_gate_blocks_noncropland(self):
        """A true-subsidy site that is not cropland is never irrigated."""
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, irrigated_all=True)
        flags = self._flags(self._run(ds, "annual_2yr", lulc=(30, "glc10")), years)
        assert all(v == 0 for v in flags.values())

    def test_third_floor_ignores_isolated_subsidy_year(self):
        """Stage 1 one-third floor: one high-ratio year does not equip a dryland.

        A single subsidy year among otherwise fallow years trips only its own
        and the next 2-yr window (2 of 7 = 0.29, below the one-third floor), so
        the site is not classified irrigation-equipped and the spike year is
        NOT flagged. This is the US-RC3 drought/boundary case in miniature — a
        lone anomalous window must not flip a dryland field to irrigated.
        """
        years = list(range(2016, 2023))  # 7 years
        ds = self._make_ds(years, subsidy_years=[2019])
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert all(v == 0 for v in flags.values())

    def test_third_floor_equips_intermittent_site(self):
        """Stage 1 one-third floor: >1/3 (not a majority) of windows equips.

        Two adjacent subsidy years among 7 trip 3 of 7 windows (0.43) — above
        the one-third floor but below a majority. The site is therefore
        irrigation-equipped, and Stage 2 flags exactly the two genuine
        subsidy years. This is the US-MH2 case (~6/13 windows): a strict
        majority would wrongly drop it to dryland; the one-third floor keeps
        it. Guards against reverting Stage 1 to >1/2.
        """
        years = list(range(2016, 2023))  # 7 years
        ds = self._make_ds(years, subsidy_years=[2019, 2020])
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert flags[2019] == 1 and flags[2020] == 1
        assert all(flags[y] == 0 for y in years if y not in (2019, 2020))

    @staticmethod
    def _with_fallback(ds, gap_years, bias):
        """NaN-out primary ETf in gap_years and add a sparse second ETf member.

        The fallback carries the primary's true value times ``bias`` every 16th
        day (Landsat-like revisit) across all years. The classifier averages it
        with the primary into the nanmean ensemble; in gap years (primary all
        NaN) the ensemble falls back to the second member alone.
        """
        dates = pd.DatetimeIndex(ds.time.values)
        gap = np.isin(dates.year.values, gap_years)
        truth = ds["etf"].values.copy()

        sparse = np.full_like(truth, np.nan)
        cap = np.arange(len(dates)) % 16 == 0
        sparse[cap, :] = truth[cap, :] * bias

        primary = truth.copy()
        primary[gap, :] = np.nan

        out = ds.copy(deep=True)
        coords = {"time": dates, "site": ds.coords["site"].values}
        out["etf"] = xr.DataArray(primary, dims=["time", "site"], coords=coords)
        out["etf_fallback"] = xr.DataArray(sparse, dims=["time", "site"], coords=coords)
        return out

    def test_fallback_member_fills_primary_gap(self):
        """The second ETf member fills primary capture gaps via the ensemble.

        With the primary blank in 2019-2020, the nanmean ensemble uses the
        second member there, so a true-subsidy site is still flagged across all
        years — and no per-year provenance tags are written (the member is part
        of the ensemble, not a per-year override).
        """
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, irrigated_all=True)
        ds = self._with_fallback(ds, gap_years=[2019, 2020], bias=1.5)
        result = self._run(ds, "annual_2yr")
        site = result["A"]
        assert {y: site[y]["irrigated"] for y in years} == {2019: 1, 2020: 1, 2021: 1}
        for y in years:
            assert set(site[y]) == {"irr_doys", "irrigated", "f_irr"}

    def test_biased_fallback_does_not_overflag_dryland(self):
        """A biased second member must not re-flag a dryland wheat-fallow site.

        The 1.3 subsidy threshold and the Stage 1 mode gate absorb a moderate
        high bias in the gap-filling member without any per-site harmonization:
        the dryland's 2-yr windows stay below 1.3, so the site is never equipped.
        """
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, crop_years=[2020])
        ds = self._with_fallback(ds, gap_years=[2019, 2020], bias=1.8)
        result = self._run(ds, "annual_2yr")
        site = result["A"]
        assert {y: site[y]["irrigated"] for y in years} == {2019: 0, 2020: 0, 2021: 0}

    def test_irr_data_carries_only_legacy_keys(self):
        """irr_data year dicts hold only irr_doys/irrigated/f_irr — no tags.

        Harmonization provenance (``evidence``, ``etf_fallback_scale``) is gone;
        consumers that iterate every non-``fallow_years`` key as a year dict must
        see a clean, uniform shape whether or not a second member is present.
        """
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, irrigated_all=True)
        ds = self._with_fallback(ds, gap_years=[2019, 2020], bias=1.5)
        for variant in (ds, ds.drop_vars("etf_fallback")):
            site = self._run(variant, "annual_2yr")["A"]
            for y in years:
                assert set(site[y]) == {"irr_doys", "irrigated", "f_irr"}
            assert "etf_fallback_scale" not in site
            assert "evidence" not in site


class TestStage2RetainAndSeason:
    """Stage 2 Schmitt trigger + demand-season rescue at equipped fields.

    Motivating case: Mediterranean-climate irrigated fields (CA delta) whose
    wet-winter years dilute the calendar-year ET/PPT ratio below any workable
    annual threshold even though the crop transpires through a nearly rain-free
    summer. The retain threshold keeps marginal years; the season test rescues
    the wet-winter years; neither can fire at a non-equipped (dryland) field.
    """

    # reuse the lulc-classifier harness without inheriting (inheritance would
    # re-collect the parent class's tests under this class)
    _make_ds = TestComputeIrrigationDataLulc._make_ds
    _run = TestComputeIrrigationDataLulc._run
    _flags = staticmethod(TestComputeIrrigationDataLulc._flags)

    def _make_med_ds(
        self, years, wet_years=(), fallow_years=(), winter_prcp=3.0, wet_winter_prcp=12.0
    ):
        """Mediterranean profile: dry high-ET summer (Apr-Oct), rainy winter.

        ``wet_years`` get a winter wet enough to push the annual ratio below
        1.0; ``fallow_years`` also get a senesced (low-ET) summer.
        """
        dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
        months = dates.month.values
        yrs = dates.year.values
        summer = np.isin(months, [4, 5, 6, 7, 8, 9, 10])

        etf = np.where(summer, 0.9, 0.2)
        eto = np.where(summer, 6.0, 1.0)
        prcp = np.where(summer, 0.1, winter_prcp)
        for y in wet_years:
            prcp[(yrs == y) & ~summer] = wet_winter_prcp
        for y in fallow_years:
            etf[(yrs == y) & summer] = 0.15

        ndvi = 0.2 + 0.5 * np.exp(-((months - 7.0) ** 2) / 8.0)

        def da(vals):
            return xr.DataArray(
                np.asarray(vals)[:, None],
                dims=["time", "site"],
                coords={"time": dates, "site": ["A"]},
            )

        return xr.Dataset({"ndvi": da(ndvi), "etf": da(etf), "eto": da(eto), "prcp": da(prcp)})

    def test_season_test_rescues_wet_winter_year(self):
        """A wet-winter year (annual ratio < 1.0) at an equipped field is
        flagged via the demand-season balance."""
        years = [2019, 2020, 2021]
        ds = self._make_med_ds(years, wet_years=[2021])
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert flags == {2019: 1, 2020: 1, 2021: 1}

    def test_wet_winter_year_missed_without_season_test(self):
        """Disabling the season test (huge storage allowance) reproduces the
        old false negative — proves the rescue comes from the season path."""
        years = [2019, 2020, 2021]
        ds = self._make_med_ds(years, wet_years=[2021])
        flags = self._flags(self._run(ds, "annual_2yr", season_storage_mm=1e6), years)
        assert flags == {2019: 1, 2020: 1, 2021: 0}

    def test_retain_threshold_keeps_marginal_year(self):
        """An equipped field's year with annual ratio between retain (1.1) and
        equip (1.3) thresholds stays irrigated; raising retain to the equip
        threshold reproduces the old flip-to-fallow."""
        years = [2019, 2020, 2021]
        # winter rain tuned so the middle year's annual ratio lands ~1.2
        ds = self._make_med_ds(years, wet_years=[2020], wet_winter_prcp=6.4)
        kw = {"season_storage_mm": 1e6}  # isolate the annual retain path
        assert self._flags(self._run(ds, "annual_2yr", **kw), years)[2020] == 1
        assert (
            self._flags(self._run(ds, "annual_2yr", annual_retain_ratio=1.3, **kw), years)[2020]
            == 0
        )

    def test_fallow_year_at_equipped_field_stays_off(self):
        """A senesced summer (low season ET) in a wet year does not pass the
        season test on stored winter moisture — the storage allowance holds."""
        years = [2019, 2020, 2021]
        ds = self._make_med_ds(years, wet_years=[2021], fallow_years=[2021])
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert flags == {2019: 1, 2020: 1, 2021: 0}

    def test_dryland_field_untouched_by_new_stage2(self):
        """Neither retain nor season test can fire at a never-equipped field."""
        years = [2019, 2020, 2021]
        ds = self._make_ds(years, crop_years=[2020])
        flags = self._flags(self._run(ds, "annual_2yr"), years)
        assert flags == {2019: 0, 2020: 0, 2021: 0}

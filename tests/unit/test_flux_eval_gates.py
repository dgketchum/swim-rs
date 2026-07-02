"""Tests for shared flux-evaluation gates and monthly aggregation helpers.

Covers review findings A2 (monthly day-count mismatch) and A5 (VALIDATION_POLICY
site-minimum enforcement) in ``swimrs.calibrate.flux_utils``.
"""

import numpy as np
import pandas as pd

from swimrs.calibrate.flux_utils import (
    full_month_paired_sums,
    paired_monthly_sums,
    passes_site_minimum,
    write_excluded_sites,
)


def _daily(year=2020, start_month=1, n_months=6, value=1.0):
    idx = pd.date_range(f"{year}-{start_month:02d}-01", periods=n_months * 31, freq="D")
    return pd.Series(value, index=idx)


class TestPassesSiteMinimum:
    def test_below_90_days_fails(self):
        idx = pd.date_range("2020-01-01", periods=89, freq="D")
        assert passes_site_minimum(pd.Series(1.0, index=idx)) is False

    def test_90_days_but_two_qualifying_months_fails(self):
        # >=90 valid days but only 2 months reach 20 valid days: two full
        # months (Jan, Feb) plus 15 valid days each in March and April
        jan = pd.date_range("2020-01-01", periods=31, freq="D")
        feb = pd.date_range("2020-02-01", periods=29, freq="D")
        mar = pd.date_range("2020-03-01", periods=15, freq="D")
        apr = pd.date_range("2020-04-01", periods=15, freq="D")
        s = pd.Series(1.0, index=jan.append(feb).append(mar).append(apr))
        assert s.dropna().shape[0] >= 90
        assert passes_site_minimum(s) is False

    def test_90_days_three_qualifying_months_passes(self):
        jan = pd.date_range("2020-01-01", periods=31, freq="D")
        feb = pd.date_range("2020-02-01", periods=29, freq="D")
        mar = pd.date_range("2020-03-01", periods=30, freq="D")
        s = pd.Series(1.0, index=jan.append(feb).append(mar))
        assert passes_site_minimum(s) is True

    def test_nan_days_do_not_count(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        s = pd.Series(1.0, index=idx)
        s.iloc[:150] = np.nan  # only 50 finite days
        assert passes_site_minimum(s) is False


class TestPairedMonthlySums:
    def test_partial_flux_month_sums_same_days_on_both_sides(self):
        # One 31-day month, flux finite on 22 days; swim finite every day
        idx = pd.date_range("2020-06-01", periods=31, freq="D")
        swim = pd.Series(2.0, index=idx)
        flux = pd.Series(3.0, index=idx)
        flux.iloc[22:] = np.nan  # 22 valid flux days

        swim_m, flux_m, _ = paired_monthly_sums(swim, flux)
        assert len(swim_m) == 1
        # swim summed over the SAME 22 valid days, not all 31
        assert np.isclose(swim_m.iloc[0], 2.0 * 22)
        assert np.isclose(flux_m.iloc[0], 3.0 * 22)

    def test_month_below_min_days_dropped(self):
        idx = pd.date_range("2020-06-01", periods=31, freq="D")
        swim = pd.Series(2.0, index=idx)
        flux = pd.Series(3.0, index=idx)
        flux.iloc[15:] = np.nan  # only 15 valid days < 20
        swim_m, flux_m, _ = paired_monthly_sums(swim, flux)
        assert len(swim_m) == 0

    def test_reference_requires_all_valid_days_finite(self):
        idx = pd.date_range("2020-06-01", periods=31, freq="D")
        swim = pd.Series(2.0, index=idx)
        flux = pd.Series(3.0, index=idx)  # all 31 days valid
        ref = pd.Series(1.5, index=idx)
        ref.iloc[10:] = np.nan  # reference missing on some valid flux days

        swim_m, flux_m, ref_m = paired_monthly_sums(swim, flux, ref)
        # reference not finite on every valid day -> NaN, not a partial sum
        assert np.isnan(ref_m.iloc[0])

    def test_reference_full_month_summed(self):
        idx = pd.date_range("2020-06-01", periods=30, freq="D")
        swim = pd.Series(2.0, index=idx)
        flux = pd.Series(3.0, index=idx)
        ref = pd.Series(1.5, index=idx)
        swim_m, flux_m, ref_m = paired_monthly_sums(swim, flux, ref)
        assert np.isclose(ref_m.iloc[0], 1.5 * 30)


class TestFullMonthPairedSums:
    def test_nearly_complete_month_kept_full_month_sum(self):
        idx = pd.date_range("2020-06-01", periods=30, freq="D")
        swim = pd.Series(2.0, index=idx)
        flux = pd.Series(3.0, index=idx)
        flux.iloc[28:] = np.nan  # 28 valid days >= 28
        swim_m, flux_m = full_month_paired_sums(swim, flux)
        assert len(swim_m) == 1
        # swim summed over the FULL calendar month (matches full-month reference)
        assert np.isclose(swim_m.iloc[0], 2.0 * 30)

    def test_gappy_month_dropped(self):
        idx = pd.date_range("2020-06-01", periods=30, freq="D")
        swim = pd.Series(2.0, index=idx)
        flux = pd.Series(3.0, index=idx)
        flux.iloc[20:] = np.nan  # 20 valid days < 28
        swim_m, flux_m = full_month_paired_sums(swim, flux)
        assert len(swim_m) == 0


def test_write_excluded_sites_creates_file(tmp_path):
    excluded = [{"site": "US-Aaa", "reason": "no_flux_data"}]
    path = write_excluded_sites(excluded, str(tmp_path))
    df = pd.read_csv(path)
    assert list(df.columns) == ["site", "reason"]
    assert df.iloc[0]["site"] == "US-Aaa"


def test_write_excluded_sites_empty_writes_header(tmp_path):
    path = write_excluded_sites([], str(tmp_path))
    df = pd.read_csv(path)
    assert list(df.columns) == ["site", "reason"]
    assert len(df) == 0

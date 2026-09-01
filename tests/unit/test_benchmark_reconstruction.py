"""Tests for the shared ETf-first daily-benchmark reconstruction.

Covers the binding construction spec for the E1 OpenET benchmark rebuild:
capture-date identity, ETf-first vs direct-ET divergence under varying ETo,
the Volk +/-32-day temporal-support rule with openet-core operational
semantics (linear when two-sided, one-sided flat fill, NaN when unsupported),
time-based interpolation on irregular intervals, input validation, support
classification from raw capture availability only, and post-reconstruction
pairing.
"""

import numpy as np
import pandas as pd
import pytest

from swimrs.calibrate.benchmark import (
    BenchmarkConstructionError,
    assert_inside_support,
    classify_temporal_support,
    pair_on_common_dates,
    reconstruct_daily_benchmark,
)


def _eto(start="2020-01-01", n=200, value=5.0):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(value, index=idx, name="eto")


def _caps(dates, values):
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype="float64")


class TestCaptureIdentity:
    def test_et_space_identity_exact(self):
        eto = _eto()
        dates = ["2020-01-10", "2020-01-18", "2020-02-05"]
        caps = _caps(dates, [2.0, 3.5, 4.0])
        recon = reconstruct_daily_benchmark(
            capture_series=caps, capture_space="et", eto=eto, eto_name="test"
        )
        for d, v in caps.items():
            assert recon.daily_et.loc[d] == pytest.approx(v, abs=1e-12)
        assert recon.identity_max_abs_err <= 1e-10

    def test_etf_space_identity_and_et_product(self):
        eto = _eto(value=4.0)
        caps = _caps(["2020-01-10", "2020-01-20"], [0.5, 0.75])
        recon = reconstruct_daily_benchmark(
            capture_series=caps, capture_space="etf", eto=eto, eto_name="test"
        )
        assert recon.daily_etf.loc["2020-01-10"] == pytest.approx(0.5, abs=1e-12)
        assert recon.daily_et.loc["2020-01-20"] == pytest.approx(0.75 * 4.0, abs=1e-12)

    def test_same_eto_for_division_and_multiplication(self):
        # et -> etf -> et roundtrip is exact only when one series does both
        eto = _eto(value=3.0)
        eto.iloc[9] = 7.3  # capture day gets a distinctive eto
        caps = _caps(["2020-01-10"], [2.19])
        recon = reconstruct_daily_benchmark(
            capture_series=caps, capture_space="et", eto=eto, eto_name="test"
        )
        assert recon.capture_etf.iloc[0] == pytest.approx(2.19 / 7.3, abs=1e-14)
        assert recon.daily_et.loc["2020-01-10"] == pytest.approx(2.19, abs=1e-12)


class TestEtfFirstVsDirectEt:
    def test_sawtooth_eto_separates_constructions(self):
        eto = _eto(n=40)
        eto.iloc[:] = np.where(np.arange(40) % 2 == 0, 2.0, 8.0)
        d0, d1 = "2020-01-05", "2020-01-15"
        caps_etf = _caps([d0, d1], [0.4, 0.4])
        caps_et = caps_etf * eto.loc[[d0, d1]].values
        recon = reconstruct_daily_benchmark(
            capture_series=caps_et, capture_space="et", eto=eto, eto_name="test"
        )
        direct = caps_et.reindex(eto.index).interpolate(method="time", limit_area="inside")
        mid = "2020-01-10"
        # constant ETf: valid reconstruction tracks daily eto exactly
        assert recon.daily_et.loc[mid] == pytest.approx(0.4 * eto.loc[mid], abs=1e-12)
        # direct-ET interpolation smooths the demand signal instead
        assert abs(direct.loc[mid] - recon.daily_et.loc[mid]) > 0.5


class TestVolkWindowRule:
    def test_gap_50_flat_linear_flat(self):
        eto = _eto(n=120, value=1.0)  # eto=1 so daily_et == daily_etf
        d0 = pd.Timestamp("2020-01-10")
        d1 = d0 + pd.Timedelta(days=50)
        recon = reconstruct_daily_benchmark(
            capture_series=_caps([d0, d1], [0.2, 0.7]),
            capture_space="etf",
            eto=eto,
            eto_name="test",
        )
        # near-left days (offset 1..17): next anchor > 32 d away -> flat fill
        assert recon.daily_et.loc[d0 + pd.Timedelta(days=17)] == pytest.approx(0.2)
        assert recon.support_class.loc[d0 + pd.Timedelta(days=17)] == "flat_fill"
        # two-sided band (offset 18..32): linear in time
        assert recon.daily_et.loc[d0 + pd.Timedelta(days=25)] == pytest.approx(0.2 + 0.5 * 25 / 50)
        assert recon.support_class.loc[d0 + pd.Timedelta(days=25)] == "interpolated"
        # near-right days (offset 33..49): flat fill from the right anchor
        assert recon.daily_et.loc[d0 + pd.Timedelta(days=40)] == pytest.approx(0.7)

    def test_gap_80_has_unsupported_interior(self):
        eto = _eto(n=200, value=1.0)
        d0 = pd.Timestamp("2020-01-10")
        d1 = d0 + pd.Timedelta(days=80)
        recon = reconstruct_daily_benchmark(
            capture_series=_caps([d0, d1], [0.2, 0.7]),
            capture_space="etf",
            eto=eto,
            eto_name="test",
        )
        assert recon.daily_et.loc[d0 + pd.Timedelta(days=32)] == pytest.approx(0.2)
        assert np.isnan(recon.daily_et.loc[d0 + pd.Timedelta(days=33)])
        assert np.isnan(recon.daily_et.loc[d0 + pd.Timedelta(days=47)])
        assert recon.daily_et.loc[d0 + pd.Timedelta(days=48)] == pytest.approx(0.7)
        assert (
            recon.support_class.loc[d0 + pd.Timedelta(days=33) : d0 + pd.Timedelta(days=47)]
            == "unsupported"
        ).all()

    def test_edge_extension_flat_32_days_then_nan(self):
        eto = _eto(n=200, value=1.0)
        d0 = pd.Timestamp("2020-03-01")
        recon = reconstruct_daily_benchmark(
            capture_series=_caps([d0], [0.5]),
            capture_space="etf",
            eto=eto,
            eto_name="test",
        )
        assert recon.daily_et.loc[d0 - pd.Timedelta(days=32)] == pytest.approx(0.5)
        assert np.isnan(recon.daily_et.loc[d0 - pd.Timedelta(days=33)])
        assert recon.daily_et.loc[d0 + pd.Timedelta(days=32)] == pytest.approx(0.5)
        assert np.isnan(recon.daily_et.loc[d0 + pd.Timedelta(days=33)])
        assert recon.support_start == d0 - pd.Timedelta(days=32)
        assert recon.support_end == d0 + pd.Timedelta(days=32)


class TestTimeBasedInterpolation:
    def test_irregular_intervals_use_time_weights(self):
        eto = _eto(value=1.0)
        d = pd.Timestamp("2020-01-10")
        caps = _caps([d, d + pd.Timedelta(days=3), d + pd.Timedelta(days=23)], [0.0, 0.3, 0.5])
        recon = reconstruct_daily_benchmark(
            capture_series=caps, capture_space="etf", eto=eto, eto_name="test"
        )
        assert recon.daily_etf.loc[d + pd.Timedelta(days=1)] == pytest.approx(0.1)
        assert recon.daily_etf.loc[d + pd.Timedelta(days=13)] == pytest.approx(0.3 + 0.2 * 10 / 20)


class TestInputValidation:
    def test_rejects_zero_eto_on_capture(self):
        eto = _eto()
        eto.loc["2020-01-10"] = 0.0
        with pytest.raises(BenchmarkConstructionError, match="do not fill"):
            reconstruct_daily_benchmark(
                capture_series=_caps(["2020-01-10"], [2.0]),
                capture_space="et",
                eto=eto,
                eto_name="test",
            )

    def test_rejects_negative_and_nan_eto_on_capture(self):
        for bad in (-1.0, np.nan):
            eto = _eto()
            eto.loc["2020-01-10"] = bad
            with pytest.raises(BenchmarkConstructionError):
                reconstruct_daily_benchmark(
                    capture_series=_caps(["2020-01-10"], [2.0]),
                    capture_space="et",
                    eto=eto,
                    eto_name="test",
                )

    def test_rejects_capture_outside_eto(self):
        with pytest.raises(BenchmarkConstructionError, match="no eto"):
            reconstruct_daily_benchmark(
                capture_series=_caps(["2030-01-01"], [2.0]),
                capture_space="et",
                eto=_eto(),
                eto_name="test",
            )

    def test_rejects_duplicate_capture_dates(self):
        caps = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-10", "2020-01-10"]))
        with pytest.raises(BenchmarkConstructionError, match="duplicate"):
            reconstruct_daily_benchmark(
                capture_series=caps, capture_space="et", eto=_eto(), eto_name="test"
            )

    def test_rejects_all_nan_captures(self):
        caps = _caps(["2020-01-10", "2020-01-20"], [np.nan, np.nan])
        with pytest.raises(BenchmarkConstructionError, match="no finite"):
            reconstruct_daily_benchmark(
                capture_series=caps, capture_space="et", eto=_eto(), eto_name="test"
            )

    def test_rejects_bad_capture_space(self):
        with pytest.raises(BenchmarkConstructionError, match="capture_space"):
            reconstruct_daily_benchmark(
                capture_series=_caps(["2020-01-10"], [1.0]),
                capture_space="eta",
                eto=_eto(),
                eto_name="test",
            )


class TestSupportClassification:
    def test_classes_from_captures_only_decoy_flag_ignored(self):
        # support classes derive from raw capture availability; an is_overpass
        # style flag carried alongside plays no role
        eto = _eto(value=1.0)
        d0, d1 = pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-20")
        frame = pd.DataFrame({"etf": [0.3, 0.6], "is_overpass": [False, False]}, index=[d0, d1])
        support = classify_temporal_support(eto.index, frame.index)
        assert support.loc[d0] == "capture"
        assert support.loc[d0 + pd.Timedelta(days=5)] == "interpolated"
        assert (support.loc[d1 + pd.Timedelta(days=33) :] == "unsupported").all()

    def test_matches_reconstruction_support_class(self):
        eto = _eto(value=1.0)
        caps = _caps(["2020-01-10", "2020-03-15"], [0.2, 0.6])
        recon = reconstruct_daily_benchmark(
            capture_series=caps, capture_space="etf", eto=eto, eto_name="test"
        )
        standalone = classify_temporal_support(eto.index, caps.index)
        assert (standalone == recon.support_class).all()

    def test_captures_without_same_day_flux_still_anchor(self):
        eto = _eto(value=1.0)
        caps = _caps(["2020-01-10", "2020-01-20"], [0.2, 0.6])
        flux = pd.Series(1.0, index=pd.DatetimeIndex(["2020-01-10"]))  # no flux on d1
        recon = reconstruct_daily_benchmark(
            capture_series=caps, capture_space="etf", eto=eto, eto_name="test"
        )
        assert recon.n_captures == 2
        assert recon.daily_etf.loc["2020-01-15"] == pytest.approx(0.4)
        paired = pair_on_common_dates(bench=recon.daily_et, flux=flux)
        assert list(paired.index) == [pd.Timestamp("2020-01-10")]

    def test_per_member_support_differs(self):
        eto = _eto(value=1.0)
        member_a = _caps(["2020-01-10", "2020-02-25"], [0.2, 0.6])
        member_b = _caps(["2020-01-10"], [0.3])
        ra = reconstruct_daily_benchmark(
            capture_series=member_a, capture_space="etf", eto=eto, eto_name="test"
        )
        rb = reconstruct_daily_benchmark(
            capture_series=member_b, capture_space="etf", eto=eto, eto_name="test"
        )
        assert ra.support_end != rb.support_end
        assert np.isfinite(ra.daily_et.loc["2020-02-20"])
        assert np.isnan(rb.daily_et.loc["2020-02-20"])


class TestSupportAssertionAndPairing:
    def test_assert_inside_support_raises_outside(self):
        eto = _eto(value=1.0)
        recon = reconstruct_daily_benchmark(
            capture_series=_caps(["2020-01-10"], [0.5]),
            capture_space="etf",
            eto=eto,
            eto_name="test",
        )
        assert_inside_support(pd.DatetimeIndex(["2020-01-20"]), recon)
        with pytest.raises(BenchmarkConstructionError, match="outside"):
            assert_inside_support(pd.DatetimeIndex(["2020-06-01"]), recon)

    def test_pairing_mask_identical_across_series(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        a = pd.Series(np.arange(10.0), index=idx)
        b = a.copy()
        b.iloc[3] = np.nan
        c = a.loc[idx[2:]]
        paired = pair_on_common_dates(a=a, b=b, c=c)
        expected = idx[2:].drop(idx[3])
        assert list(paired.index) == list(expected)
        assert list(paired.columns) == ["a", "b", "c"]

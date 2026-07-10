"""WP-C4 SMAP SSM anomaly observation-construction tests.

Pure tests of swimrs.calibrate.ssm_obs (no container, no PEST run). They assert the
observation and prediction anomalies share one definition, growing-season + capture
gating is correct, footprint-gated and short-record sites are excluded, and the
deseasonalization matches the evaluate._deseasonalize (DOY-mean) semantics.
"""

import numpy as np
import pandas as pd

from swimrs.calibrate.ssm_obs import (
    build_ssm_observation,
    deseasonalize_doy,
    model_ssm_prediction,
    recursive_soil_water_index,
    surface_theta_from_depl_ze,
)


def _grid(start="2016-01-01", end="2020-12-31"):
    idx = pd.date_range(start, end, freq="D")
    return idx, idx.month.values, idx.year.values, idx.dayofyear.values


def test_deseasonalize_matches_pandas_doy_mean():
    # Reference: pandas groupby(dayofyear) - transform("mean"), the evaluate.py path.
    idx, _m, _y, doy = _grid()
    rng = np.random.default_rng(0)
    vals = rng.normal(size=len(idx)) + np.sin(doy / 366.0 * 2 * np.pi)
    s = pd.Series(vals, index=idx)
    ref = (s - s.groupby(s.index.dayofyear).transform("mean")).values
    got = deseasonalize_doy(vals, doy)
    assert np.allclose(got, ref, atol=1e-12)


def test_deseasonalized_series_has_zero_doy_mean():
    _idx, _m, _y, doy = _grid()
    rng = np.random.default_rng(1)
    vals = rng.normal(size=len(doy))
    an = deseasonalize_doy(vals, doy)
    # Each DOY group's anomalies sum to ~0.
    for d in np.unique(doy):
        assert abs(an[doy == d].mean()) < 1e-9


def test_surface_theta_sign_and_scale():
    # Drier surface (larger depletion) -> more negative proxy; ze=0.1 -> /100.
    depl = np.array([0.0, 10.0, 20.0])
    theta = surface_theta_from_depl_ze(depl, ze=0.1)
    assert np.allclose(theta, np.array([0.0, -0.10, -0.20]))
    assert theta[2] < theta[1] < theta[0]


def test_build_only_weights_growing_season_captures():
    idx, months, years, doy = _grid()
    # SMAP present every 3rd day (revisit), value = seasonal + noise.
    smap = np.full(len(idx), np.nan)
    cap = np.arange(0, len(idx), 3)
    rng = np.random.default_rng(2)
    smap[cap] = 0.2 + 0.05 * np.sin(doy[cap] / 366 * 2 * np.pi) + 0.01 * rng.normal(size=cap.size)

    anom, capture_idx, used = build_ssm_observation(smap, months, years, doy, min_seasons=2)
    assert used
    # every weighted day is a growing-season (Apr-Oct) capture day
    assert np.all(np.isin(months[capture_idx], [4, 5, 6, 7, 8, 9, 10]))
    assert np.all(np.isfinite(smap[capture_idx]))
    # non-capture / off-season days are NaN (unweighted)
    off = np.setdiff1d(np.arange(len(idx)), capture_idx)
    assert np.all(np.isnan(anom[off]))
    # winter captures exist in `cap` but are excluded
    assert not np.all(np.isin(cap, capture_idx))


def test_gated_site_returns_no_obs():
    idx, months, years, doy = _grid()
    smap = np.full(len(idx), 0.25)  # abundant data
    anom, capture_idx, used = build_ssm_observation(
        smap, months, years, doy, min_seasons=2, gated=True
    )
    assert not used
    assert capture_idx.size == 0
    assert np.all(np.isnan(anom))


def test_min_seasons_gate_excludes_short_record():
    # Only one growing season of captures -> excluded when min_seasons=2.
    idx = pd.date_range("2016-01-01", "2016-12-31", freq="D")
    months, years, doy = idx.month.values, idx.year.values, idx.dayofyear.values
    smap = np.full(len(idx), np.nan)
    gs = np.where(np.isin(months, [6, 7, 8]))[0]
    smap[gs] = 0.25
    _anom, capture_idx, used = build_ssm_observation(smap, months, years, doy, min_seasons=2)
    assert not used
    assert capture_idx.size == 0


def test_obs_and_pred_share_anomaly_definition():
    # A perfectly-tracking model (theta proxy == smap up to affine) must give
    # identical obs and pred anomalies on the capture days, since both are
    # deseasonalized on the same day set with the same function.
    idx, months, years, doy = _grid()
    smap = np.full(len(idx), np.nan)
    cap = np.where(np.isin(months, [4, 5, 6, 7, 8, 9, 10]))[0][::2]
    rng = np.random.default_rng(3)
    signal = 0.25 + 0.04 * np.sin(doy / 366 * 2 * np.pi) + 0.02 * rng.normal(size=len(idx))
    smap[cap] = signal[cap]

    anom_obs, capture_idx, used = build_ssm_observation(smap, months, years, doy, min_seasons=2)
    assert used

    # Model depl_ze chosen so surface theta proxy == smap signal exactly:
    # theta = -depl_ze/(1000*ze) == signal  ->  depl_ze = -signal*1000*ze
    ze = 0.10
    depl_ze = -signal * 1000.0 * ze
    pred = model_ssm_prediction(depl_ze, doy, capture_idx, ze=ze)

    assert np.allclose(anom_obs[capture_idx], pred[capture_idx], atol=1e-12)
    # off capture days the prediction is exactly zero (weight is zero there)
    off = np.setdiff1d(np.arange(len(idx)), capture_idx)
    assert np.all(pred[off] == 0.0)


def test_empty_capture_prediction_is_zero():
    idx, _m, _y, doy = _grid()
    depl_ze = np.linspace(0, 20, len(idx))
    pred = model_ssm_prediction(depl_ze, doy, np.array([], dtype=int), ze=0.1)
    assert pred.shape == (len(idx),)
    assert np.all(pred == 0.0)


def test_recursive_swi_uses_irregular_capture_interval():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05"])
    values = np.array([0.1, 0.3, 0.5])
    got = recursive_soil_water_index(values, dates.values, characteristic_time_days=2.0)

    gain_1 = 1.0 / (1.0 + np.exp(-1.0 / 2.0))
    expected_1 = 0.1 + gain_1 * (0.3 - 0.1)
    gain_2 = gain_1 / (gain_1 + np.exp(-3.0 / 2.0))
    expected_2 = expected_1 + gain_2 * (0.5 - expected_1)
    assert np.allclose(got, [0.1, expected_1, expected_2])


def test_recursive_swi_preserves_capture_calendar_and_constant_signal():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    values = np.array([0.25, np.nan, 0.25, np.nan, 0.25])
    got = recursive_soil_water_index(values, dates.values)
    assert np.allclose(got[[0, 2, 4]], 0.25)
    assert np.isnan(got[[1, 3]]).all()


def test_recursive_swi_validates_time_and_characteristic_scale():
    dates = pd.to_datetime(["2020-01-02", "2020-01-01"]).values
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        recursive_soil_water_index(np.array([0.1, 0.2]), dates)
    with np.testing.assert_raises_regex(ValueError, "greater than zero"):
        recursive_soil_water_index(np.array([0.1]), dates[:1], characteristic_time_days=0.0)

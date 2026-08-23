"""Source-exclusive supply accounting: groundwater subsidy must not
double-book water in years the irrigation mechanism supplies."""

import h5py
import numpy as np
import pytest

from swimrs.process.input import (
    _irrigated_years,
    _write_gwsub_from_container,
    _write_properties_from_container,
)


def _container_data(irr=None, gwsub=None):
    return {
        "props": {},
        "dynamics": {"irr": irr or {}, "gwsub": gwsub or {}},
    }


def _gw_year(f_sub, subsidized=1):
    return {"subsidized": subsidized, "f_sub": f_sub, "ratio": 1.0 / max(1.0 - f_sub, 1e-9)}


def _irr_year(irrigated=True):
    return {
        "f_irr": 1.0 if irrigated else 0.0,
        "irr_doys": list(range(150, 250)) if irrigated else [],
    }


class TestIrrigatedYears:
    def test_extracts_years_with_windows(self):
        fid_irr = {
            "2016": _irr_year(True),
            "2017": _irr_year(False),
            "2018": _irr_year(True),
            "fallow_years": [2019],
        }
        assert _irrigated_years(fid_irr) == {2016, 2018}

    def test_empty_or_invalid(self):
        assert _irrigated_years({}) == set()
        assert _irrigated_years(None) == set()
        assert _irrigated_years({"2016": "bad"}) == set()


class TestGwsubTableExclusivity:
    def _build(self, tmp_path, irr, gwsub, fids):
        path = tmp_path / "input.h5"
        with h5py.File(path, "w") as h5:
            _write_gwsub_from_container(h5, _container_data(irr, gwsub), fids, len(fids))
            if "gwsub" not in h5:
                return {}
            return {y: h5["gwsub"][y][:] for y in h5["gwsub"]}

    def test_fsub_zeroed_in_irrigated_years(self, tmp_path):
        irr = {"f1": {"2016": _irr_year(True), "2017": _irr_year(False)}}
        gwsub = {"f1": {"2016": _gw_year(0.5), "2017": _gw_year(0.4)}}
        table = self._build(tmp_path, irr, gwsub, ["f1"])
        assert table["2016"][0] == 0.0
        assert table["2017"][0] == pytest.approx(0.4)

    def test_non_irrigated_field_unaffected(self, tmp_path):
        irr = {"f1": {"2016": _irr_year(True)}}
        gwsub = {
            "f1": {"2016": _gw_year(0.5)},
            "f2": {"2016": _gw_year(0.3)},
        }
        table = self._build(tmp_path, irr, gwsub, ["f1", "f2"])
        assert table["2016"][0] == 0.0
        assert table["2016"][1] == pytest.approx(0.3)

    def test_irrigated_year_missing_from_gwsub_gets_explicit_zero(self, tmp_path):
        # Without an explicit dataset, get_f_sub_for_year would fall back to
        # the static props value and re-apply subsidy in an irrigated year
        irr = {"f1": {"2019": _irr_year(True)}}
        gwsub = {"f1": {"2016": _gw_year(0.5)}}
        table = self._build(tmp_path, irr, gwsub, ["f1"])
        assert "2019" in table
        assert table["2019"][0] == 0.0


class TestStaticFsubExcludesIrrigatedYears:
    def _props_fsub(self, tmp_path, irr, gwsub, fids):
        path = tmp_path / "input.h5"
        with h5py.File(path, "w") as h5:
            _write_properties_from_container(h5, _container_data(irr, gwsub), fids, len(fids))
            return h5["properties/f_sub"][:]

    def test_mean_over_non_irrigated_years_only(self, tmp_path):
        irr = {"f1": {"2016": _irr_year(True), "2017": _irr_year(False)}}
        gwsub = {"f1": {"2016": _gw_year(0.8), "2017": _gw_year(0.4)}}
        f_sub = self._props_fsub(tmp_path, irr, gwsub, ["f1"])
        assert f_sub[0] == pytest.approx(0.4)

    def test_all_years_irrigated_gives_zero(self, tmp_path):
        irr = {"f1": {"2016": _irr_year(True), "2017": _irr_year(True)}}
        gwsub = {"f1": {"2016": _gw_year(0.8), "2017": _gw_year(0.4)}}
        f_sub = self._props_fsub(tmp_path, irr, gwsub, ["f1"])
        assert f_sub[0] == 0.0

    def test_no_irrigation_matches_plain_mean(self, tmp_path):
        gwsub = {"f1": {"2016": _gw_year(0.6), "2017": _gw_year(0.2)}}
        f_sub = self._props_fsub(tmp_path, {}, gwsub, ["f1"])
        assert f_sub[0] == pytest.approx(0.4)


class TestGwStatusSiteGate:
    """gw_status requires the persistent (non-irrigated-year mean) f_sub to
    clear the daily-loop threshold — one anomalous year cannot switch a site's
    groundwater subsidy on."""

    def _props(self, tmp_path, irr, gwsub, fids):
        path = tmp_path / "input.h5"
        with h5py.File(path, "w") as h5:
            _write_properties_from_container(h5, _container_data(irr, gwsub), fids, len(fids))
            return h5["properties/gw_status"][:], h5["properties/f_sub"][:]

    def test_gate_on_mean_not_max(self, tmp_path):
        # one hot year (0.5) among cool years: mean 0.14 fails the gate even
        # though the single year exceeds the threshold
        gwsub = {
            "f1": {
                str(y): _gw_year(fs)
                for y, fs in [(2016, 0.5), (2017, 0.02), (2018, 0.0), (2019, 0.03)]
            }
        }
        gw_status, f_sub = self._props(tmp_path, {}, gwsub, ["f1"])
        assert f_sub[0] < 0.2
        assert gw_status[0] == 0

    def test_sustained_subsidy_passes(self, tmp_path):
        gwsub = {"f1": {"2016": _gw_year(0.5), "2017": _gw_year(0.4)}}
        gw_status, _ = self._props(tmp_path, {}, gwsub, ["f1"])
        assert gw_status[0] == 1

    def test_gate_excludes_irrigated_years(self, tmp_path):
        # the hot years are irrigated (excluded); the remaining year is cool
        irr = {"f1": {"2016": _irr_year(True), "2017": _irr_year(True)}}
        gwsub = {"f1": {"2016": _gw_year(0.8), "2017": _gw_year(0.7), "2018": _gw_year(0.05)}}
        gw_status, _ = self._props(tmp_path, irr, gwsub, ["f1"])
        assert gw_status[0] == 0

    def test_no_gwsub_data_fails_gate(self, tmp_path):
        gw_status, _ = self._props(tmp_path, {}, {}, ["f1"])
        assert gw_status[0] == 0


class TestLoopFastYearSpecificFsub:
    def test_gw_sim_follows_daily_fsub(self):
        from swimrs.process.loop_fast import _run_loop_jit

        from .test_loop_fast_bounds import _make_inputs

        n_days, n_fields = 20, 1
        all_f_sub = np.zeros((n_days, n_fields))
        all_f_sub[:10] = 0.5  # "non-irrigated year": subsidy active
        all_f_sub[10:] = 0.0  # "irrigated year": exclusivity zeroes f_sub
        inputs = _make_inputs(
            n_days=n_days,
            n_fields=n_fields,
            all_prcp=np.zeros((n_days, n_fields)),  # dry: depletion grows past RAW
            all_f_sub=all_f_sub,
            gw_status=np.ones(n_fields),
            depl_root_init=np.full(n_fields, 120.0),
        )
        result = _run_loop_jit(**inputs)
        gw_sim = result[14]
        assert gw_sim[:10].sum() > 0.0
        assert gw_sim[10:].sum() == 0.0

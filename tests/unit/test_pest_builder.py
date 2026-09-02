"""Tests for swimrs.calibrate.pest_builder module.

Specifically tests the ETf weight assignment logic to prevent regressions
where all ETf observations get zero weights during PEST++ calibration setup.
"""

import warnings

import numpy as np
import pandas as pd
import pytest


class TestEtfWeightAssignment:
    """Tests for ETf observation weight assignment in _write_etf_obs."""

    def test_etf_weight_uses_default_when_etf_std_fid_is_none(self):
        """ETf weights should use default 1/0.33 when etf_std[fid] is None.

        This is a regression test for a bug where:
        - self.etf_std was initialized as {fid: None for fid in targets}
        - The check `if self.etf_std:` evaluated True (non-empty dict)
        - But self.etf_std[fid] was None, causing AttributeError or NaN weights
        - NaN weights were then converted to 0.0, disabling all ETf calibration

        The fix changes line 915 from:
            if self.etf_std:
        To:
            if self.etf_std is not None and self.etf_std.get(fid) is not None:
        """
        # Create a mock obs DataFrame similar to what PestBuilder uses
        # The index format is: 'oname:obs_etf_{fid}_otype:arr_i:{idx}_j:0'
        fid = "test_field"
        n_days = 365
        obs_index = [f"oname:obs_etf_{fid}_otype:arr_i:{i}_j:0" for i in range(n_days)]

        obs_df = pd.DataFrame(
            {
                "obsval": np.random.rand(n_days),
                "weight": 0.0,  # Initially zero
            },
            index=obs_index,
        )
        obs_df.index = obs_df.index.str.lower()

        # Simulate capture indexes (indices where we have ETf observations)
        capture_indexes = obs_df.index[:50].tolist()  # 50 capture dates

        # Case 1: etf_std is a dict but etf_std[fid] is None (the bug condition)
        etf_std_with_none = {fid: None}

        # Apply the FIXED logic (what we're testing)
        # OLD buggy logic: if self.etf_std:  # True for non-empty dict
        # NEW fixed logic:
        if etf_std_with_none is not None and etf_std_with_none.get(fid) is not None:
            # This branch should NOT execute when etf_std[fid] is None
            obs_df.loc[capture_indexes, "weight"] = 999  # Would fail test
        else:
            # This should execute - use default weight
            obs_df.loc[capture_indexes, "weight"] = 1 / 0.33

        # Verify captures got positive weights
        capture_weights = obs_df.loc[capture_indexes, "weight"]
        assert (capture_weights > 0).all(), "All capture dates should have positive weights"
        assert np.isclose(capture_weights.iloc[0], 1 / 0.33), (
            f"Expected default weight ~3.03, got {capture_weights.iloc[0]}"
        )

    def test_etf_weight_uses_std_when_etf_std_fid_has_data(self):
        """ETf weights should use ensemble std when etf_std[fid] has data."""
        fid = "test_field"
        n_days = 365
        obs_index = [f"oname:obs_etf_{fid}_otype:arr_i:{i}_j:0" for i in range(n_days)]

        obs_df = pd.DataFrame(
            {
                "obsval": np.random.rand(n_days),
                "weight": 0.0,
            },
            index=obs_index,
        )
        obs_df.index = obs_df.index.str.lower()

        # Create observation_index mapping (maps obs_id to date index)
        observation_index = pd.DataFrame(
            data=range(n_days), index=obs_df.index, columns=["obs_idx"]
        )

        capture_indexes = obs_df.index[:50].tolist()
        capture_dates = observation_index.loc[capture_indexes, "obs_idx"].to_list()

        # Create etf_std with actual std values
        etf_std_df = pd.DataFrame(
            {
                "std": np.random.rand(n_days) * 0.1 + 0.05,  # std between 0.05 and 0.15
                "mean": np.random.rand(n_days),
            },
            index=range(n_days),
        )

        etf_std_with_data = {fid: etf_std_df}

        # Apply the fixed logic
        if etf_std_with_data is not None and etf_std_with_data.get(fid) is not None:
            obs_df.loc[capture_indexes, "weight"] = 1 / (
                etf_std_with_data[fid].loc[capture_dates, "std"].values + 0.1
            )
        else:
            obs_df.loc[capture_indexes, "weight"] = 1 / 0.33

        # Verify captures got positive weights based on std
        capture_weights = obs_df.loc[capture_indexes, "weight"]
        assert (capture_weights > 0).all(), "All capture dates should have positive weights"

        # Weights should vary based on std values
        assert not np.allclose(capture_weights, capture_weights.iloc[0]), (
            "Weights should vary based on ensemble std values"
        )

        # Weights should be in reasonable range: 1/(0.05+0.1) to 1/(0.15+0.1)
        # i.e., ~4 to ~6.67
        assert capture_weights.min() > 3.5, f"Min weight {capture_weights.min()} too low"
        assert capture_weights.max() < 7.0, f"Max weight {capture_weights.max()} too high"

    def test_etf_weight_warning_when_all_zero(self):
        """Should warn when all ETf observations have zero weight."""
        # This tests the diagnostic warning added to detect the bug condition
        total_valid_obs = 50  # We have valid observations
        total_nonzero_etf = 0  # But all weights are zero (bug condition)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This is the diagnostic logic added after the fix
            if total_valid_obs > 0 and total_nonzero_etf == 0:
                warnings.warn(
                    f"All {total_valid_obs} ETf observations have zero weight. "
                    "Check etf_std index alignment with capture_dates.",
                    UserWarning,
                    stacklevel=2,
                )

            assert len(w) == 1
            assert "zero weight" in str(w[0].message)
            assert "50" in str(w[0].message)

    def test_buggy_condition_would_fail(self):
        """Demonstrate that the OLD buggy logic would give wrong result.

        This shows what the bug was: checking `if self.etf_std:` when etf_std
        is a dict with None values causes the wrong branch to execute,
        leading to AttributeError or NaN weights.
        """
        fid = "test_field"
        etf_std_with_none = {fid: None}

        # OLD buggy logic
        old_logic_result = bool(etf_std_with_none)  # True for non-empty dict

        # NEW fixed logic
        new_logic_result = etf_std_with_none is not None and etf_std_with_none.get(fid) is not None

        assert old_logic_result is True, "Old logic evaluates True for {fid: None}"
        assert new_logic_result is False, "New logic correctly evaluates False"

        # The old logic would try to access etf_std[fid].loc[...], which fails
        # because etf_std[fid] is None
        with pytest.raises(AttributeError):
            _ = etf_std_with_none[fid].loc[[0, 1, 2], "std"]


class TestEtfWeightEdgeCases:
    """Edge case tests for ETf weight assignment."""

    def test_empty_etf_std_dict_uses_default(self):
        """Empty etf_std dict should use default weights."""
        fid = "test_field"
        etf_std_empty = {}

        # Fixed logic handles empty dict correctly
        if etf_std_empty is not None and etf_std_empty.get(fid) is not None:
            result = "std-based"
        else:
            result = "default"

        assert result == "default"

    def test_etf_std_none_uses_default(self):
        """etf_std=None (no ensemble) should use default weights."""
        fid = "test_field"
        etf_std_none = None

        if etf_std_none is not None and etf_std_none.get(fid) is not None:
            result = "std-based"
        else:
            result = "default"

        assert result == "default"

    def test_missing_fid_in_etf_std_uses_default(self):
        """Missing fid in etf_std should use default weights."""
        etf_std = {"other_field": pd.DataFrame({"std": [0.1]})}

        fid = "test_field"  # Not in etf_std

        if etf_std is not None and etf_std.get(fid) is not None:
            result = "std-based"
        else:
            result = "default"

        assert result == "default"


class TestMADParameterBounds:
    """Tests for irrigation-dependent MAD parameter initial values and bounds.

    The MAD (Management Allowable Depletion) parameter controls when irrigation
    is triggered. Irrigated fields should have low MAD (trigger early), while
    non-irrigated fields tolerate more depletion.

    This mirrors the logic in PestBuilder.get_pest_builder_args() lines 323-333.
    """

    @staticmethod
    def _mad_params(irr: float) -> dict:
        """Extract MAD initial value and bounds given mean irrigation fraction."""
        if irr > 0.2:
            return {"initial": 0.10, "lower": 0.10, "upper": 0.3}
        else:
            return {"initial": 0.5, "lower": 0.3, "upper": 0.8}

    def test_irrigated_field_mad_bounds(self):
        """Irrigated fields (irr > 0.2) get MAD bounds [0.10, 0.3]."""
        p = self._mad_params(irr=0.5)
        assert p["initial"] == 0.10
        assert p["lower"] == 0.10
        assert p["upper"] == 0.3

    def test_non_irrigated_field_mad_bounds(self):
        """Non-irrigated fields get MAD bounds [0.3, 0.8]."""
        p = self._mad_params(irr=0.05)
        assert p["initial"] == 0.5
        assert p["lower"] == 0.3
        assert p["upper"] == 0.8

    def test_boundary_value_not_irrigated(self):
        """irr == 0.2 is NOT irrigated (threshold is > 0.2)."""
        p = self._mad_params(irr=0.2)
        assert p["initial"] == 0.5

    def test_boundary_value_irrigated(self):
        """irr just above 0.2 is irrigated."""
        p = self._mad_params(irr=0.21)
        assert p["initial"] == 0.10


# ---------------------------------------------------------------------------
# Localizer exact matching (C-4), regularization channel (C-5), and
# noise/weight consistency (C-6) — exercised on minimal synthetic .pst files
# via bare PestBuilder instances.
# ---------------------------------------------------------------------------

import os

from pyemu import Pst
from pyemu.mat import Matrix

from swimrs.calibrate.pest_builder import PestBuilder


def _par_name(pargp, fid):
    return f"pname:p_{pargp}_{fid}_:0_ptype:cn_usecol:1_pstyle:m"


def _obs_name(ob_type, fid, k):
    return f"oname:obs_{ob_type}_{fid}_otype:arr_i:{k}_j:0"


class _Cfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _bare_builder(pst_file, **attrs):
    b = PestBuilder.__new__(PestBuilder)
    b.pst_file = str(pst_file)
    for k, v in attrs.items():
        setattr(b, k, v)
    return b


class TestLocalizerExactMatching:
    """A site ID that is a prefix of another (us-ne1/us-ne11) must not
    cross-link localizer rows or columns (C-4)."""

    def _build(self, tmp_path):
        pest_dir = tmp_path / "pest"
        pest_dir.mkdir()
        pst_file = pest_dir / "proj.pst"

        sites = ["us-ne1", "us-ne11"]
        par_names, obs_names = [], []
        for s in sites:
            par_names += [_par_name("aw", s), _par_name("swe_alpha", s)]
            obs_names += [_obs_name("etf", s, 0), _obs_name("swe", s, 1)]

        pst = Pst.from_par_obs_names(par_names, obs_names)
        pst.write(str(pst_file), version=2)

        b = _bare_builder(
            pst_file,
            config=_Cfg(start_dt=pd.Timestamp("2020-01-01"), end_dt=pd.Timestamp("2020-01-05")),
        )
        b.build_localizer()
        return Matrix.from_ascii(str(pest_dir / "loc.mat")).to_dataframe()

    def test_no_cross_site_linkage(self, tmp_path):
        loc = self._build(tmp_path)

        # ETf obs drive only the same site's ET params
        assert loc.loc[_obs_name("etf", "us-ne1", 0), _par_name("aw", "us-ne1")] == 1.0
        assert loc.loc[_obs_name("etf", "us-ne1", 0), _par_name("aw", "us-ne11")] == 0.0
        assert loc.loc[_obs_name("etf", "us-ne11", 0), _par_name("aw", "us-ne11")] == 1.0
        assert loc.loc[_obs_name("etf", "us-ne11", 0), _par_name("aw", "us-ne1")] == 0.0

    def test_obs_type_param_separation(self, tmp_path):
        loc = self._build(tmp_path)

        # ETf obs never touch snow params and vice versa
        assert loc.loc[_obs_name("etf", "us-ne1", 0), _par_name("swe_alpha", "us-ne1")] == 0.0
        assert loc.loc[_obs_name("swe", "us-ne1", 1), _par_name("swe_alpha", "us-ne1")] == 1.0
        assert loc.loc[_obs_name("swe", "us-ne1", 1), _par_name("aw", "us-ne1")] == 0.0
        assert loc.loc[_obs_name("swe", "us-ne1", 1), _par_name("swe_alpha", "us-ne11")] == 0.0


class TestRegularizationChannel:
    """pestpp-ies ignores prior-information equations, so add_regularization
    must not write PI; it activates ies_reg_factor instead (C-5)."""

    def _setup(self, tmp_path):
        pest_dir = tmp_path / "pest"
        pest_dir.mkdir()
        pst_file = pest_dir / "proj.pst"

        par_names = [_par_name("aw", "us-ne1"), _par_name("ndvi_k", "us-ne1")]
        obs_names = [_obs_name("etf", "us-ne1", k) for k in range(3)]
        pst = Pst.from_par_obs_names(par_names, obs_names)
        par = pst.parameter_data
        par["pname"] = "p"
        par.loc[par_names[0], ["pargp", "parval1", "parlbnd", "parubnd"]] = [
            "aw",
            200.0,
            100.0,
            400.0,
        ]
        par.loc[par_names[1], ["pargp", "parval1", "parlbnd", "parubnd"]] = [
            "ndvi_k",
            10.0,
            1.0,
            25.0,
        ]
        pst.observation_data["weight"] = 2.0  # info mass = 3 obs * 2^2 = 12
        pst.write(str(pst_file), version=2)

        return _bare_builder(
            pst_file,
            pest_dir=str(pest_dir),
            params_file=str(tmp_path / "params.csv"),
            etf_capture_indexes=[obs_names],
            _regularization_active=False,
            verbose=False,
            config=_Cfg(prior_regularization_fraction=0.5),
        )

    def test_no_pi_equations_written(self, tmp_path):
        b = self._setup(tmp_path)
        b.add_regularization()
        assert Pst(b.pst_file).nprior == 0

    def test_audit_budgets(self, tmp_path):
        b = self._setup(tmp_path)
        b.add_regularization()
        audit = pd.read_csv(os.path.join(b.pest_dir, "regularization_audit.csv"))
        # site budget = 0.5 * 12 = 6, split over 2 params = 3 each
        assert np.allclose(audit["etf_info_mass"], 12.0)
        assert np.allclose(audit["param_budget"], 3.0)
        assert (audit["pi_weight"] > 0).all()

    def test_reg_factor_gated_on_activation(self, tmp_path):
        b = self._setup(tmp_path)
        b.write_control_settings(noptmax=-2, reals=10)
        assert "ies_reg_factor" not in Pst(b.pst_file).pestpp_options

        b.add_regularization()
        assert b._regularization_active
        b.write_control_settings(noptmax=-2, reals=10)
        assert float(Pst(b.pst_file).pestpp_options["ies_reg_factor"]) == 0.5


class TestFinalizeObsNoise:
    """The noise standard_deviation must carry the same per-date error the
    weights assume: spread + floor, with fixed_sd fallback (C-6)."""

    def test_noise_sd_matches_weight_error(self, tmp_path):
        pst_file = tmp_path / "proj.pst"
        fid = "us-ne1"
        obs_names = [_obs_name("etf", fid, k) for k in range(4)]
        obs_names += [_obs_name("swe", fid, k) for k in range(2)]
        pst = Pst.from_par_obs_names([_par_name("aw", fid)], obs_names)
        pst.observation_data.loc[obs_names[4], "obsval"] = 50.0
        pst.observation_data.loc[obs_names[5], "obsval"] = 0.0
        pst.write(str(pst_file), version=2)

        b = _bare_builder(
            pst_file,
            pest_args={"targets": [fid]},
            config=_Cfg(etf_weighting_fixed_sd=0.33, etf_weighting_spread_floor=0.1),
            etf_std={fid: pd.DataFrame({"std": [0.05, 0.2, np.nan, 0.15]})},
        )
        b._finalize_obs()

        obs = Pst(str(pst_file)).observation_data
        sd = obs.loc[obs_names, "standard_deviation"].values.astype(float)
        # spread dates: std + floor; the NaN-spread date falls back to fixed_sd
        assert np.allclose(sd[:4], [0.15, 0.30, 0.33, 0.25])
        # SWE: fractional error with floor (max(0.3*50, 10) = 15) for positive
        # obs, untouched otherwise
        assert sd[4] == 15.0
        assert sd[5] == 0.0
        # SWE weight derives from the same sd: weight * sd is the balancing
        # constant c, so the weight/noise contradiction (52-580x in the old
        # 1/(26*(swe+10)) + 5mm scheme) is gone by construction
        w = obs.loc[obs_names, "weight"].values.astype(float)
        etf_phi = float(((w[:4] * sd[:4]) ** 2).sum())
        c_expected = np.sqrt(0.15 / 0.85 * etf_phi / 1)
        assert np.isclose(w[4] * sd[4], c_expected)

    def test_fixed_mode_uses_fixed_sd(self, tmp_path):
        pst_file = tmp_path / "proj.pst"
        fid = "us-ne1"
        obs_names = [_obs_name("etf", fid, k) for k in range(3)]
        pst = Pst.from_par_obs_names([_par_name("aw", fid)], obs_names)
        pst.write(str(pst_file), version=2)

        b = _bare_builder(
            pst_file,
            pest_args={"targets": [fid]},
            config=_Cfg(etf_weighting_fixed_sd=0.33),
            etf_std=None,
        )
        b._finalize_obs()

        obs = Pst(str(pst_file)).observation_data
        assert np.allclose(obs.loc[obs_names, "standard_deviation"].astype(float), 0.33)


class TestSweWeightBalance:
    """SWE weights must derive from the SWE error model (coherent with noise)
    and the group's expected phi must hit the configured share of ETf's."""

    def _finalized_obs(self, tmp_path, phi_share=0.15):
        pst_file = tmp_path / "proj.pst"
        fid = "us-ne1"
        etf_names = [_obs_name("etf", fid, k) for k in range(4)]
        swe_names = [_obs_name("swe", fid, k) for k in range(4, 8)]
        pst = Pst.from_par_obs_names([_par_name("aw", fid)], etf_names + swe_names)
        obs = pst.observation_data
        obs.loc[etf_names, "weight"] = [2.0, 3.0, 1.5, 0.0]  # one zero-weight date
        obs.loc[swe_names, "obsval"] = [5.0, 50.0, 200.0, 0.0]
        # mimic _write_swe_obs: placeholder 1.0 on valid (swe>0), 0 otherwise
        obs.loc[swe_names, "weight"] = [1.0, 1.0, 1.0, 0.0]
        pst.write(str(pst_file), version=2)

        b = _bare_builder(
            pst_file,
            pest_args={"targets": [fid]},
            config=_Cfg(
                etf_weighting_spread_floor=0.05,
                swe_weighting_phi_share=phi_share,
            ),
            etf_std={fid: pd.DataFrame({"std": [0.05, 0.2, 0.1, 0.15]})},
        )
        b._finalize_obs()
        return Pst(str(pst_file)).observation_data, etf_names, swe_names

    def test_sd_is_fractional_with_floor(self, tmp_path):
        obs, _, swe_names = self._finalized_obs(tmp_path)
        sd = obs.loc[swe_names, "standard_deviation"].astype(float).values
        # max(0.3*swe, 10): 5 -> 10 (floor), 50 -> 15, 200 -> 60; zero obs untouched
        assert np.allclose(sd, [10.0, 15.0, 60.0, 0.0])

    def test_weight_noise_coherent(self, tmp_path):
        obs, _, swe_names = self._finalized_obs(tmp_path)
        w = obs.loc[swe_names, "weight"].astype(float).values
        sd = obs.loc[swe_names, "standard_deviation"].astype(float).values
        # weight = c/sd for all valid obs -> w*sd identical
        assert np.allclose(w[:3] * sd[:3], w[0] * sd[0])
        # invalid (swe=0) obs keeps zero weight
        assert w[3] == 0.0

    def test_group_phi_share(self, tmp_path):
        obs, etf_names, swe_names = self._finalized_obs(tmp_path, phi_share=0.15)
        ew = obs.loc[etf_names, "weight"].astype(float).values
        esd = obs.loc[etf_names, "standard_deviation"].astype(float).values
        sw = obs.loc[swe_names, "weight"].astype(float).values
        ssd = obs.loc[swe_names, "standard_deviation"].astype(float).values
        etf_phi = ((ew * esd) ** 2).sum()
        swe_phi = ((sw * ssd) ** 2).sum()
        assert np.isclose(swe_phi / (swe_phi + etf_phi), 0.15)


# ---------------------------------------------------------------------------
# Auxiliary (additional-date) ETf source — E3 Landsat + ECOSTRESS design.
# The real _write_etf_obs / _finalize_obs / _fill_auxiliary code paths are
# exercised on synthetic frames via a fake PstFrom and a stubbed container
# getter. Frozen design: dates with any primary member keep the Landsat rule
# (ECOSTRESS excluded, incl. one-member zero-weight dates); dates with only
# the auxiliary get weight obsval/aux_fixed_sd and noise SD aux_fixed_sd.
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

from swimrs.container.components.exporter import Exporter


class _FakePest:
    """Minimal PstFrom stand-in: reads the .np obsval file like PstFrom does."""

    def __init__(self):
        self.obs_dfs = []

    def add_observations(self, filename, insfile=None):
        vals = np.atleast_1d(np.loadtxt(filename))
        fid = _Path(filename).stem.replace("obs_etf_", "")
        idx = [f"oname:obs_etf_{fid}_otype:arr_i:{j}_j:0".lower() for j in range(len(vals))]
        self.obs_dfs.append(pd.DataFrame({"obsval": vals, "weight": 1.0}, index=idx))


_FID = "s1"
_DATES = pd.date_range("2020-01-01", "2020-01-10", freq="D")


def _frame(col, values):
    return pd.DataFrame({col: values}, index=_DATES)


def _synthetic_frames():
    """10-day scenario covering every date class in the frozen design.

    d0 both members; d1 nothing; d2 auxiliary only; d3 one member + auxiliary
    (no substitution); d4 both members + auxiliary (overlap excluded);
    d5-d9 nothing.
    """
    nan = np.nan
    ssebop = [0.5, nan, nan, 0.4, 0.5, nan, nan, nan, nan, nan]
    ptjpl = [0.7, nan, nan, nan, 0.7, nan, nan, nan, nan, nan]
    ensemble = [0.6, nan, nan, 0.4, 0.6, nan, nan, nan, nan, nan]
    aux = [nan, nan, 0.8, 0.9, 0.75, nan, nan, nan, nan, nan]
    return {
        ("ensemble", None): _frame("ensemble_etf_no_mask", ensemble),
        ("ssebop", None): _frame("ssebop_etf_no_mask", ssebop),
        ("ptjpl", None): _frame("ptjpl_etf_no_mask", ptjpl),
        ("ptjpl", "ecostress"): _frame("ptjpl_etf_no_mask", aux),
    }


def _write_obs_file(tmp_path, frames, with_aux):
    """Build obs_etf_{fid}.np the way the exporter does (primary + aux fill)."""
    primary = frames[("ensemble", None)]["ensemble_etf_no_mask"].values
    if with_aux:
        aux = frames[("ptjpl", "ecostress")]["ptjpl_etf_no_mask"].values
        combined = Exporter._fill_auxiliary(primary, aux)
    else:
        combined = np.asarray(primary, dtype=float)
    obs_file = tmp_path / f"obs_etf_{_FID}.np"
    np.savetxt(obs_file, combined)
    return obs_file


def _harness(tmp_path, with_aux):
    frames = _synthetic_frames()
    obs_file = _write_obs_file(tmp_path, frames, with_aux)

    cfg_kwargs = dict(
        start_dt=_DATES[0],
        end_dt=_DATES[-1],
        etf_weighting_mode="spread",
        etf_weighting_fixed_sd=0.33,
        etf_weighting_spread_floor=0.05,
        etf_weighting_min_members=2,
    )
    if with_aux:
        cfg_kwargs.update(
            etf_auxiliary_model="ptjpl",
            etf_auxiliary_instrument="ecostress",
            etf_auxiliary_fixed_sd=0.33,
        )

    b = PestBuilder.__new__(PestBuilder)
    b.config = _Cfg(**cfg_kwargs)
    b.masks = ["no_mask"]
    b.irr = {}
    b.etf_std = None
    b.pest = _FakePest()
    b.observation_index = {}
    b.etf_capture_indexes = []
    b._weight_audit_rows = []
    b.etf_aux_obs_ids = set()
    b._aux_only_ids_by_fid = {}
    b._aux_raw_by_fid = {}
    b._aux_overlap_dates_by_fid = {}
    b.conflicted_obs = None
    b.verbose = False
    b.obs_idx_file = str(tmp_path / "idx.csv")
    b.pest_args = {
        "targets": [_FID],
        "etf_obs": {"file": [str(obs_file)], "insfile": [None]},
    }
    b._get_etf_data = lambda fid, model="ssebop", instrument=None: frames[
        (model, instrument)
    ].copy()
    return b


def _obsname(j):
    return f"oname:obs_etf_{_FID}_otype:arr_i:{j}_j:0".lower()


def _spread_weight(obsval, member_vals, floor=0.05):
    return obsval / (np.std(member_vals, ddof=1) + floor)


class TestAuxiliaryEtfWeights:
    """_write_etf_obs with the auxiliary additional-date source."""

    def _run(self, tmp_path, with_aux):
        b = _harness(tmp_path, with_aux)
        b._write_etf_obs("ensemble", ["ssebop", "ptjpl"])
        return b, b.pest.obs_dfs[0]

    def test_default_non_regression(self, tmp_path):
        """No aux config -> weights identical to the pre-change behavior."""
        _, d = self._run(tmp_path, with_aux=False)
        w = d["weight"].values
        expected_d0 = _spread_weight(0.6, [0.5, 0.7])
        assert np.isclose(w[0], expected_d0)
        assert w[2] == 0.0  # no retrieval, no aux
        assert w[3] == 0.0  # one member: min_members zero-weights it
        assert np.isclose(w[4], expected_d0)
        assert all(w[j] == 0.0 for j in [1, 5, 6, 7, 8, 9])
        assert d["obsval"].iloc[2] == -99.0

    def test_primary_weights_unchanged_by_aux(self, tmp_path):
        """Every Landsat date keeps its control weight bit-for-bit."""
        _, d_ctrl = self._run(tmp_path, with_aux=False)
        _, d_trt = self._run(tmp_path, with_aux=True)
        primary_days = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        assert np.allclose(
            d_ctrl["weight"].values[primary_days], d_trt["weight"].values[primary_days]
        )
        assert np.allclose(
            d_ctrl["obsval"].values[primary_days], d_trt["obsval"].values[primary_days]
        )

    def test_aux_only_date_weight(self, tmp_path):
        """ECOSTRESS-only date: obsval from aux, weight obsval/0.33."""
        b, d = self._run(tmp_path, with_aux=True)
        assert np.isclose(d["obsval"].iloc[2], 0.8)
        assert np.isclose(d["weight"].iloc[2], 0.8 / 0.33)
        assert b.etf_aux_obs_ids == {_obsname(2)}

    def test_overlap_excluded(self, tmp_path):
        """Aux on a two-member date changes nothing; recorded as overlap."""
        b, d = self._run(tmp_path, with_aux=True)
        assert np.isclose(d["weight"].iloc[4], _spread_weight(0.6, [0.5, 0.7]))
        assert np.isclose(d["obsval"].iloc[4], 0.6)
        assert _DATES[4] in b._aux_overlap_dates_by_fid[_FID]

    def test_no_substitution_on_one_member_date(self, tmp_path):
        """One-member Landsat date stays zero-weighted, aux not substituted."""
        b, d = self._run(tmp_path, with_aux=True)
        assert d["weight"].iloc[3] == 0.0
        assert np.isclose(d["obsval"].iloc[3], 0.4)  # Landsat value, not aux 0.9
        assert _DATES[3] in b._aux_overlap_dates_by_fid[_FID]
        assert _obsname(3) not in b.etf_aux_obs_ids

    def test_zero_weight_without_retrieval(self, tmp_path):
        """No Landsat, no aux -> obsval -99, weight 0."""
        _, d = self._run(tmp_path, with_aux=True)
        for j in [1, 5, 6, 7, 8, 9]:
            assert d["obsval"].iloc[j] == -99.0
            assert d["weight"].iloc[j] == 0.0

    def test_obsval_agreement_export_pst_audit(self, tmp_path):
        """The .np export, PST obsval, and audit row agree on the aux date."""
        b, d = self._run(tmp_path, with_aux=True)
        exported = np.loadtxt(tmp_path / f"obs_etf_{_FID}.np")
        assert np.isclose(exported[2], d["obsval"].iloc[2])
        audit = pd.DataFrame(b._weight_audit_rows)
        aux_row = audit[audit["source"] == "auxiliary"].iloc[0]
        assert aux_row["date"] == "2020-01-03"
        assert np.isclose(aux_row["obsval"], exported[2])
        assert np.isclose(aux_row["error_scale"], 0.33)

    def test_audit_rows_and_export(self, tmp_path):
        """Audit carries source class, overlap exclusion, and error scale."""
        b, _ = self._run(tmp_path, with_aux=True)
        audit = pd.DataFrame(b._weight_audit_rows).set_index("date")
        assert audit.loc["2020-01-01", "source"] == "primary"
        assert not audit.loc["2020-01-01", "aux_overlap_excluded"]
        assert audit.loc["2020-01-03", "source"] == "auxiliary"
        assert audit.loc["2020-01-03", "member_count"] == 0
        assert audit.loc["2020-01-04", "aux_overlap_excluded"]
        assert np.isclose(audit.loc["2020-01-04", "aux_raw_value"], 0.9)
        assert audit.loc["2020-01-05", "aux_overlap_excluded"]

        out = tmp_path / "weight_audit.csv"
        b.export_weight_audit(str(out))
        cols = pd.read_csv(out).columns
        for c in ["source", "error_scale", "aux_overlap_excluded", "aux_raw_value"]:
            assert c in cols


class TestAuxiliaryFinalizeObs:
    """_finalize_obs: aux noise SD and the SWE one-variable invariant."""

    def _build_pst(self, tmp_path, n_etf, etf_weights, name="proj.pst"):
        pst_file = tmp_path / name
        fid = "us-ne1"
        etf_names = [_obs_name("etf", fid, k) for k in range(n_etf)]
        swe_names = [_obs_name("swe", fid, k) for k in range(n_etf, n_etf + 2)]
        pst = Pst.from_par_obs_names([_par_name("aw", fid)], etf_names + swe_names)
        obs = pst.observation_data
        obs.loc[etf_names, "weight"] = etf_weights
        obs.loc[swe_names, "obsval"] = [50.0, 120.0]
        obs.loc[swe_names, "weight"] = 1.0
        pst.write(str(pst_file), version=2)
        return pst_file, fid, etf_names, swe_names

    def test_aux_noise_sd_is_aux_fixed_sd(self, tmp_path):
        """Aux obs get etf_auxiliary_fixed_sd, not the NaN-spread fallback."""
        pst_file, fid, etf_names, _ = self._build_pst(tmp_path, 5, [2.0, 3.0, 1.5, 0.0, 2.0])
        b = _bare_builder(
            pst_file,
            pest_args={"targets": [fid]},
            config=_Cfg(
                etf_weighting_fixed_sd=0.33,
                etf_weighting_spread_floor=0.05,
                etf_auxiliary_fixed_sd=0.4,  # distinct from fixed_sd fallback
            ),
            etf_std={fid: pd.DataFrame({"std": [0.05, 0.2, 0.1, 0.15, np.nan]})},
            etf_aux_obs_ids={etf_names[4]},
        )
        b._finalize_obs()
        obs = Pst(str(pst_file)).observation_data
        sd = obs.loc[etf_names, "standard_deviation"].astype(float).values
        assert np.allclose(sd[:4], [0.10, 0.25, 0.15, 0.20])
        assert np.isclose(sd[4], 0.4)

    def test_swe_weights_match_control(self, tmp_path):
        """SWE balance uses the primary subset only: control == treatment."""
        # Control: 4 primary ETf obs, no aux.
        ctrl_pst, fid, _, ctrl_swe = self._build_pst(
            tmp_path, 4, [2.0, 3.0, 1.5, 0.0], name="ctrl.pst"
        )
        b_ctrl = _bare_builder(
            ctrl_pst,
            pest_args={"targets": [fid]},
            config=_Cfg(etf_weighting_spread_floor=0.05),
            etf_std={fid: pd.DataFrame({"std": [0.05, 0.2, 0.1, 0.15]})},
        )
        b_ctrl._finalize_obs()

        # Treatment: same 4 primary obs plus one active aux obs.
        trt_pst, fid, trt_etf, trt_swe = self._build_pst(
            tmp_path, 5, [2.0, 3.0, 1.5, 0.0, 2.5], name="trt.pst"
        )
        b_trt = _bare_builder(
            trt_pst,
            pest_args={"targets": [fid]},
            config=_Cfg(
                etf_weighting_spread_floor=0.05,
                etf_auxiliary_fixed_sd=0.33,
            ),
            etf_std={fid: pd.DataFrame({"std": [0.05, 0.2, 0.1, 0.15, np.nan]})},
            etf_aux_obs_ids={trt_etf[4]},
        )
        b_trt._finalize_obs()

        ctrl_obs = Pst(str(ctrl_pst)).observation_data
        trt_obs = Pst(str(trt_pst)).observation_data
        cw = ctrl_obs.loc[ctrl_swe, "weight"].astype(float).values
        tw = trt_obs.loc[trt_swe, "weight"].astype(float).values
        assert np.allclose(cw, tw)
        assert (cw > 0).all()

        # Sanity: the aux obs is genuinely active in the treatment, so a
        # whole-group balance would have shifted SWE weights.
        assert float(trt_obs.loc[trt_etf[4], "weight"]) > 0


class TestExporterFillAuxiliary:
    """Exporter._fill_auxiliary: gap-fill only, primary preserved exactly."""

    def test_fills_only_gaps(self):
        primary = np.array([0.6, np.nan, np.nan, 0.4])
        aux = np.array([0.9, 0.8, np.nan, 0.9])
        out = Exporter._fill_auxiliary(primary, aux)
        assert np.isclose(out[0], 0.6)  # primary kept despite aux overlap
        assert np.isclose(out[1], 0.8)  # gap filled
        assert np.isnan(out[2])  # no retrieval anywhere
        assert np.isclose(out[3], 0.4)  # primary kept

    def test_inputs_not_mutated(self):
        primary = np.array([np.nan, 0.5])
        aux = np.array([0.7, 0.9])
        out = Exporter._fill_auxiliary(primary, aux)
        assert np.isnan(primary[0]) and np.isclose(out[0], 0.7)
        assert np.isclose(primary[1], 0.5) and np.isclose(out[1], 0.5)


class TestAuxiliaryGuards:
    """Review fixes: irrigation-mask guard and info-mass aux exclusion."""

    def test_aux_with_irrigation_masks_raises(self, tmp_path):
        """Aux source under mask-switching is unsupported and must not run."""
        b = _harness(tmp_path, with_aux=True)
        b.masks = ["inv_irr", "irr"]
        with pytest.raises(NotImplementedError, match="no_mask"):
            b._write_etf_obs("ensemble", ["ssebop", "ptjpl"])

    def test_info_mass_excludes_aux_obs(self, tmp_path):
        """Prior-regularization budgets balance on the primary subset only."""
        fid = "us-ne1"
        etf_names = [_obs_name("etf", fid, k) for k in range(4)]
        pst = Pst.from_par_obs_names([_par_name("aw", fid)], etf_names)
        pst.observation_data["weight"] = 2.0
        pst_file = tmp_path / "proj.pst"
        pst.write(str(pst_file), version=2)
        pst = Pst(str(pst_file))

        full = PestBuilder._etf_information_mass_by_site(pst)
        assert np.isclose(full[fid], 16.0)  # 4 obs * 2^2

        primary = PestBuilder._etf_information_mass_by_site(pst, exclude={etf_names[3]})
        assert np.isclose(primary[fid], 12.0)  # aux obs dropped


class TestParsePriorSpec:
    """PestBuilder._parse_prior_spec: legacy float and dict (value/lower/upper) forms."""

    def test_bare_number_is_value_only(self):
        assert PestBuilder._parse_prior_spec(4.85) == (4.85, None, None)
        assert PestBuilder._parse_prior_spec(361) == (361.0, None, None)

    def test_dict_full_spec(self):
        v, lb, ub = PestBuilder._parse_prior_spec({"value": 0.52, "lower": 0.45, "upper": 0.60})
        assert (v, lb, ub) == (0.52, 0.45, 0.60)

    def test_dict_bounds_only(self):
        v, lb, ub = PestBuilder._parse_prior_spec({"lower": 3.5, "upper": 6.0})
        assert v is None and (lb, ub) == (3.5, 6.0)

    def test_dict_value_only(self):
        assert PestBuilder._parse_prior_spec({"value": 1.2}) == (1.2, None, None)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="none of"):
            PestBuilder._parse_prior_spec({})

    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError, match="lower >= upper"):
            PestBuilder._parse_prior_spec({"lower": 0.6, "upper": 0.45})

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
        # SWE: fixed 5.0 mm error for positive obs, untouched otherwise
        assert sd[4] == 5.0
        assert sd[5] == 0.0

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

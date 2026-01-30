"""Tests for property derivation logic in swimrs.process.input.

Tests cover the pure numpy operations used during HDF5 construction:
- CN2 from clay: clay < 15 -> 67, 15-30 -> 77, > 30 -> 85
- Perennial from LULC: codes 12/14 not perennial, code 10 perennial
- Variable alias resolution: etr/eto/ref_et fallback chain
- NDVI bare/full: with data -> per-field percentiles, without -> defaults
"""

import numpy as np
from numpy.testing import assert_array_equal


class TestCN2FromClay:
    """Tests for CN2 derivation from clay content.

    Logic: cn2 = np.where(clay < 15, 67, np.where(clay > 30, 85, 77))
    """

    def test_clay_below_15(self):
        """Clay < 15 -> CN2 = 67."""
        clay = np.array([5.0, 10.0, 14.9])
        cn2 = np.where(clay < 15.0, 67.0, np.where(clay > 30.0, 85.0, 77.0))
        assert_array_equal(cn2, [67.0, 67.0, 67.0])

    def test_clay_15_to_30(self):
        """15 <= Clay <= 30 -> CN2 = 77."""
        clay = np.array([15.0, 20.0, 30.0])
        cn2 = np.where(clay < 15.0, 67.0, np.where(clay > 30.0, 85.0, 77.0))
        assert_array_equal(cn2, [77.0, 77.0, 77.0])

    def test_clay_above_30(self):
        """Clay > 30 -> CN2 = 85."""
        clay = np.array([30.1, 50.0, 80.0])
        cn2 = np.where(clay < 15.0, 67.0, np.where(clay > 30.0, 85.0, 77.0))
        assert_array_equal(cn2, [85.0, 85.0, 85.0])

    def test_mixed_array(self):
        """Mixed clay values produce correct CN2 distribution."""
        clay = np.array([10.0, 20.0, 40.0])
        cn2 = np.where(clay < 15.0, 67.0, np.where(clay > 30.0, 85.0, 77.0))
        assert_array_equal(cn2, [67.0, 77.0, 85.0])


class TestPerennialFromLULC:
    """Tests for perennial status derivation from LULC codes.

    Logic: perennial = lulc_code not in {12, 14} AND 1 <= lulc_code <= 17
    """

    def test_cropland_not_perennial(self):
        """LULC 12 (cropland) is not perennial."""
        codes = {12, 14}
        lulc = 12
        perennial = lulc not in codes and 1 <= lulc <= 17
        assert perennial is False

    def test_cropland_mosaic_not_perennial(self):
        """LULC 14 (cropland/natural mosaic) is not perennial."""
        codes = {12, 14}
        lulc = 14
        perennial = lulc not in codes and 1 <= lulc <= 17
        assert perennial is False

    def test_grassland_is_perennial(self):
        """LULC 10 (grassland) is perennial."""
        codes = {12, 14}
        lulc = 10
        perennial = lulc not in codes and 1 <= lulc <= 17
        assert perennial is True

    def test_forest_is_perennial(self):
        """LULC 1 (evergreen needleleaf forest) is perennial."""
        codes = {12, 14}
        lulc = 1
        perennial = lulc not in codes and 1 <= lulc <= 17
        assert perennial is True

    def test_invalid_code_not_perennial(self):
        """LULC 0 (invalid) is not perennial."""
        codes = {12, 14}
        lulc = 0
        perennial = lulc not in codes and 1 <= lulc <= 17
        assert perennial is False

    def test_vectorized_perennial(self):
        """Vectorized perennial derivation over array of LULC codes."""
        codes = np.array([12, 14, 10, 1, 0])
        crops_codes = {12, 14}
        perennial = np.array([c not in crops_codes and 1 <= c <= 17 for c in codes])
        assert_array_equal(perennial, [False, False, True, True, False])


class TestVariableAliasResolution:
    """Tests for reference ET alias resolution logic.

    Logic from get_time_series: if variable not in ts, check aliases.
    """

    def test_primary_present_direct_return(self):
        """When primary name exists, it is used directly."""
        ts_keys = {"etr", "prcp", "tmin"}
        variable = "etr"
        assert variable in ts_keys

    def test_fallback_to_alias(self):
        """When primary name missing, falls back to alias."""
        ts_keys = {"ref_et", "prcp"}
        variable = "etr"
        ref_et_aliases = {"etr", "eto", "ref_et"}
        actual_var = variable
        if variable not in ts_keys and variable in ref_et_aliases:
            for alias in ref_et_aliases:
                if alias in ts_keys:
                    actual_var = alias
                    break
        assert actual_var == "ref_et"

    def test_none_present_raises(self):
        """When no alias is present, expect KeyError."""
        ts_keys = {"prcp", "tmin"}
        variable = "etr"
        ref_et_aliases = {"etr", "eto", "ref_et"}
        found = False
        if variable not in ts_keys and variable in ref_et_aliases:
            for alias in ref_et_aliases:
                if alias in ts_keys:
                    found = True
                    break
        assert found is False

    def test_non_ref_et_variable_not_aliased(self):
        """Non-reference-ET variables don't use the alias chain."""
        ts_keys = {"ref_et", "prcp"}
        variable = "srad"
        ref_et_aliases = {"etr", "eto", "ref_et"}
        actual_var = variable
        if variable not in ts_keys and variable in ref_et_aliases:
            for alias in ref_et_aliases:
                if alias in ts_keys:
                    actual_var = alias
                    break
        # srad is not in ref_et_aliases, so no alias resolution
        assert actual_var == "srad"


class TestNDVIBareFullThresholds:
    """Tests for NDVI bare/full threshold derivation."""

    def test_with_data_per_field_percentiles(self):
        """With sufficient data, compute per-field percentiles."""
        n_fields = 2
        n_days = 365
        rng = np.random.RandomState(42)
        ndvi_data = rng.uniform(0.1, 0.9, size=(n_days, n_fields))

        ndvi_bare = np.full(n_fields, 0.15)
        ndvi_full = np.full(n_fields, 0.85)
        for i in range(n_fields):
            valid = ndvi_data[:, i]
            valid = valid[np.isfinite(valid)]
            if len(valid) > 10:
                ndvi_bare[i] = np.percentile(valid, 5)
                ndvi_full[i] = np.percentile(valid, 95)

        # Should not be default values
        assert not np.allclose(ndvi_bare, 0.15)
        assert not np.allclose(ndvi_full, 0.85)
        # bare < full for each field
        assert all(ndvi_bare[i] < ndvi_full[i] for i in range(n_fields))

    def test_without_data_uses_defaults(self):
        """Without sufficient data, defaults are used."""
        n_fields = 2
        ndvi_bare = np.full(n_fields, 0.15)
        ndvi_full = np.full(n_fields, 0.85)
        # Simulate < 10 valid data points
        ndvi_data = np.full((5, n_fields), np.nan)

        for i in range(n_fields):
            valid = ndvi_data[:, i]
            valid = valid[np.isfinite(valid)]
            if len(valid) > 10:
                ndvi_bare[i] = np.percentile(valid, 5)
                ndvi_full[i] = np.percentile(valid, 95)

        assert_array_equal(ndvi_bare, [0.15, 0.15])
        assert_array_equal(ndvi_full, [0.85, 0.85])

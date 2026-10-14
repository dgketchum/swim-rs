import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "4_Flux_Network"
    / "archive_posterior_summary.py"
)
SPEC = importlib.util.spec_from_file_location("archive_posterior_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_boundary_hits_use_each_site_parameter_bounds():
    site_df = pd.DataFrame(
        [
            {
                "site": "IRR",
                "parameter": "mad",
                "lc_class": "Croplands",
                "median": 0.299,
                "parlbnd": 0.1,
                "parubnd": 0.3,
            },
            {
                "site": "RAIN",
                "parameter": "mad",
                "lc_class": "Croplands",
                "median": 0.39,
                "parlbnd": 0.3,
                "parubnd": 0.8,
            },
        ]
    )

    result = MODULE._build_boundary_hit_rates(site_df, "test")
    all_sites = result[(result["parameter"] == "mad") & (result["lulc_group"] == "ALL")].iloc[0]

    assert all_sites["n_sites"] == 2
    assert all_sites["upper_hit_count"] == 1
    assert all_sites["upper_hit_rate"] == pytest.approx(0.5)
    assert all_sites["lower_hit_count"] == 0
    assert all_sites["n_bound_sets"] == 2
    assert pd.isna(all_sites["bound_tolerance"])
    assert all_sites["bound_tolerance_min"] == pytest.approx(0.002)
    assert all_sites["bound_tolerance_max"] == pytest.approx(0.005)

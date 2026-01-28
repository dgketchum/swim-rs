import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_build_obs_diagnostics_table_groups_and_counts():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from swimrs.calibrate.pest_builder import PestBuilder

    obs = pd.DataFrame(
        {
            "obgnme": ["etf", "etf", "etf", "swe", "swe"],
            "obsval": [0.5, -99.0, 0.7, 10.0, 0.0],
            "weight": [1.0, 0.0, 2.0, 0.001, 0.0],
            "standard_deviation": [0.1, np.nan, 0.2, 5.0, 5.0],
        },
        index=[
            "oname:obs_etf_x_otype:arr_i:0_j:0",
            "oname:obs_etf_x_otype:arr_i:1_j:0",
            "oname:obs_etf_x_otype:arr_i:2_j:0",
            "oname:obs_swe_x_otype:arr_i:0_j:0",
            "oname:obs_swe_x_otype:arr_i:1_j:0",
        ],
    )

    table = PestBuilder._build_obs_diagnostics_table(obs)
    assert set(table["group"]) == {"etf", "swe"}

    etf = table.set_index("group").loc["etf"]
    assert etf["n"] == 3
    assert etf["valid"] == 2  # excludes -99
    assert etf["w>0"] == 2
    assert np.isclose(etf["w_sum"], 3.0)
    assert np.isclose(etf["w_max"], 2.0)

    swe = table.set_index("group").loc["swe"]
    assert swe["n"] == 2
    assert swe["valid"] == 2
    assert swe["w>0"] == 1
    assert np.isclose(swe["w_sum"], 0.001)

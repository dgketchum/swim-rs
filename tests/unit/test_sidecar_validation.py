"""Unit tests for the E2 ERA5-Land ETo/ETr sidecar validator (Phase 1, Gate G1).

Covers the gate logic in
``examples/6_Flux_International/e2_refooting/phase1_validate_sidecar.py``: the ETo agreement
thresholds and the sign-agreement rule for the few mid-winter days where ERA5-Land hourly
Penman-Monteith integrates to a non-positive daily ETo in both the sidecar and the container.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase1_validate_sidecar", E2_DIR / "phase1_validate_sidecar.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


def _frames(eto, stored, etr=None):
    n = len(eto)
    side = pd.DataFrame(
        {
            "site": ["S"] * n,
            "date": [f"2016{i + 1:02d}15" for i in range(n)],
            "eto": eto,
            "etr": etr if etr is not None else [e * 1.3 for e in eto],
            "source_file": ["refet_ratio_2016_utc_p01.csv"] * n,
            "utc_suffix": ["utc_p01"] * n,
        }
    )
    stored_df = pd.DataFrame({"site": ["S"] * n, "date": side["date"], "stored_eto": stored})
    return side, stored_df


def test_exact_agreement_passes(mod):
    eto = [2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
    side, stored = _frames(eto, eto)
    _, report = mod.validate(side, stored)
    assert report["gate"]["pass"]
    assert report["stats"]["median_abs_rel_diff"] == 0.0
    assert report["stats"]["ratio_median"] == pytest.approx(1.3)


def test_negative_winter_eto_shared_with_container_passes(mod):
    # a slightly negative daily ETo present in both the sidecar and the container is not a defect
    eto = [-0.02, 2.0, 3.0, 4.0, 5.0, 4.0]
    side, stored = _frames(eto, eto, etr=[0.03, 2.6, 3.9, 5.2, 6.5, 5.2])
    _, report = mod.validate(side, stored)
    assert report["stats"]["n_ratio_nonpositive"] == 1
    assert report["stats"]["n_eto_nonpositive"] == 1
    assert report["stats"]["n_eto_sign_mismatch_with_stored"] == 0
    assert report["stats"]["eto_nonpositive_by_month"] == {"01": 1}
    assert report["gate"]["eto_sign_matches_stored"]
    assert report["gate"]["pass"]


def test_negative_eto_disagreeing_with_container_fails(mod):
    eto = [-0.02, 2.0, 3.0, 4.0, 5.0, 4.0]
    stored = [0.5, 2.0, 3.0, 4.0, 5.0, 4.0]
    side, stored_df = _frames(eto, stored)
    _, report = mod.validate(side, stored_df)
    assert report["stats"]["n_eto_sign_mismatch_with_stored"] == 1
    assert not report["gate"]["eto_sign_matches_stored"]
    assert not report["gate"]["pass"]


def test_biased_eto_fails_median_gate(mod):
    eto = [2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
    side, stored = _frames([e * 1.02 for e in eto], eto)
    _, report = mod.validate(side, stored)
    assert not report["gate"]["median_ok"]
    assert not report["gate"]["pass"]


def test_zero_eto_ratio_is_nonfinite_and_fails(mod):
    eto = [0.0, 2.0, 3.0, 4.0, 5.0, 4.0]
    side, stored = _frames(eto, eto, etr=[0.1, 2.6, 3.9, 5.2, 6.5, 5.2])
    _, report = mod.validate(side, stored)
    assert report["stats"]["n_ratio_nonfinite"] == 1
    assert not report["gate"]["ratios_finite"]

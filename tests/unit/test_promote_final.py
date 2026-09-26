"""``promote_final.py`` derives the frozen E1 supporting products from run outputs."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

EX5 = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"


@pytest.fixture(scope="module")
def mod():
    if str(EX5) not in sys.path:
        sys.path.insert(0, str(EX5))
    spec = importlib.util.spec_from_file_location("promote_final", EX5 / "promote_final.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VECTORS = {
    "loro": {"A": {"mad": 0.14, "kr_alpha": 0.4}, "B": {"mad": 0.14, "kr_alpha": 0.4}},
    "local": {"A": {"mad": 0.12, "kr_alpha": 0.5}, "B": {"mad": 0.39, "kr_alpha": 0.2}},
}
SUPPORT = pd.DataFrame(
    {"fid": ["A", "B"], "region": ["West", "East"], "irr_class": ["irrigated", "rainfed"]}
)
BOUNDS = pd.DataFrame(
    {
        "param": ["mad", "mad", "aw"],
        "site": ["A", "B", "A"],
        "lower_bound": [0.1, 0.3, 100.0],
        "upper_bound": [0.3, 0.8, 400.0],
    }
)


def test_fold_mad_domain_flags_pooled_rainfed_illegal(mod):
    df = mod.fold_mad_domain(VECTORS, SUPPORT, BOUNDS)
    assert list(df.columns) == mod.FOLD_MAD_COLUMNS
    assert len(df) == 4
    row = df.set_index(["arm", "fid"])
    assert row.loc[("loro", "B"), "mad_in_class_prior"] == False  # noqa: E712
    assert row.loc[("local", "B"), "mad_in_class_prior"] == True  # noqa: E712
    assert row.loc[("loro", "A"), "mad_in_class_prior"] == True  # noqa: E712
    assert (row.loc[("loro", "B"), ["prior_lo", "prior_hi"]] == [0.3, 0.8]).all()
    assert row.loc[("local", "B"), "kr_alpha"] == 0.2


def test_fold_mad_domain_rejects_inconsistent_class_bounds(mod):
    bounds = BOUNDS.copy()
    bounds.loc[bounds.site == "B", "irr"] = None
    support = SUPPORT.assign(irr_class=["irrigated", "irrigated"])
    with pytest.raises(ValueError, match="vary within"):
        mod.fold_mad_domain(VECTORS, support, bounds)


def test_stratified_summary_prepends_experiment_label(mod):
    summary = pd.DataFrame({"basis": ["daily"], "metric": ["kge"], "median_common": [0.5]})
    out = mod.stratified_summary(summary)
    assert out.columns[0] == "experiment"
    assert out["experiment"].iloc[0] == mod.EXPERIMENT_LABEL
    assert out.drop(columns="experiment").equals(summary)


def test_build_and_compare_round_trip(mod, tmp_path):
    run_dir = tmp_path / "results" / mod.ex5_paths.CANONICAL_RUN
    (run_dir / "spread_error").mkdir(parents=True)
    (run_dir / "archive" / "3_problem_definition").mkdir(parents=True)
    strat = run_dir / mod.TRANSFER_DIR
    strat.mkdir(parents=True)
    for part in ("persite", "quintiles", "summary"):
        (run_dir / "spread_error" / f"spread_error_{part}.csv").write_text(f"x\n{part}\n")
    for scale in ("daily", "monthly"):
        six_arm = pd.DataFrame({c: [1.0] for c in mod.POOLED_COLUMNS[2:]})
        six_arm.insert(0, "fid", ["A"])
        six_arm.insert(1, "region", ["West_Coast"])
        six_arm["irr_class"] = "rainfed"
        six_arm["loro_strat_kge"] = 0.5
        six_arm["loro_abs_bias"] = 0.1
        six_arm.to_csv(strat / f"persite_{scale}.csv", index=False)
    pd.DataFrame({"basis": ["daily"], "median_common": [0.5]}).to_csv(
        strat / "summary_metrics.csv", index=False
    )
    (strat / "transfer_vectors.json").write_text(json.dumps(VECTORS))
    SUPPORT.to_csv(strat / "class_fold_support.csv", index=False)
    BOUNDS.to_csv(
        run_dir / "archive" / "3_problem_definition" / "parameter_bounds.csv", index=False
    )

    products = mod.build_products(run_dir)
    assert len(products) == 7
    pooled = pd.read_csv(io.BytesIO(products["e2_within_transfer_daily_site_metrics.csv"]))
    assert list(pooled.columns) == mod.POOLED_COLUMNS
    final = tmp_path / "final"
    assert {s for _, s, _ in mod.compare(products, final)} == {"missing"}
    final.mkdir()
    for name, data in products.items():
        (final / name).write_bytes(data)
    assert {s for _, s, _ in mod.compare(products, final)} == {"match"}
    (final / "e2_spread_error_summary.csv").write_text("changed\n")
    statuses = dict((n, s) for n, s, _ in mod.compare(products, final))
    assert statuses["e2_spread_error_summary.csv"] == "differs"


def test_pooled_site_metrics_requires_every_pooled_arm_column(mod):
    six_arm = pd.DataFrame({c: [1.0] for c in mod.POOLED_COLUMNS[2:-1]})
    six_arm.insert(0, "fid", ["A"])
    six_arm.insert(1, "region", ["West_Coast"])
    with pytest.raises(ValueError, match="default_beta"):
        mod.pooled_site_metrics(six_arm)

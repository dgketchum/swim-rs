"""Unit tests for the weighting-ablation summary builders (Example 5).

Covers the required summary behaviors:

1. the phi parser reads real pestpp-ies phi.meas.csv structure correctly
   (rows are iterations; per-iteration phi comes from the ``mean`` column;
   realization columns are never mislabeled as iterations);
2. malformed phi files (missing columns, duplicate or non-contiguous
   iterations) raise instead of producing silent nonsense;
3. bootstrap output is deterministic under the fixed recorded seed;
4. finite-value filtering applies per metric;
5. the paired bias contrast uses absolute MBE, not signed cohort medians.

The script lives outside the swimrs package and is imported by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "5_Flux_Ensemble"
    / "run_weighting_ablation.py"
)


@pytest.fixture(scope="module")
def wa():
    spec = importlib.util.spec_from_file_location("run_weighting_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_phi(path, iterations, means, n_realizations=5):
    """Write a synthetic phi.meas.csv with pestpp-ies schema."""
    rows = []
    for it, mean in zip(iterations, means):
        row = {
            "iteration": it,
            "total_runs": 200 * (it + 1),
            "mean": mean,
            "standard_deviation": mean * 0.1,
            "min": mean * 0.5,
            "max": mean * 2.0,
        }
        # Realization columns: values chosen so any mislabeling as
        # iterations would corrupt phi_initial/phi_final detectably.
        for r in range(n_realizations):
            row[str(r)] = 500.0 + r
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


class TestPhiParser:
    def test_reads_iterations_from_rows_and_mean_column(self, wa, tmp_path):
        phi_path = tmp_path / "phi.meas.csv"
        _write_phi(phi_path, [0, 1, 2, 3], [1795750.0, 24947.5, 19685.6, 18981.1])
        phi = wa._read_phi_iterations(str(phi_path))
        assert list(phi["iteration"]) == [0, 1, 2, 3]
        assert phi["mean"].iloc[0] == pytest.approx(1795750.0)
        assert phi["mean"].iloc[-1] == pytest.approx(18981.1)

    def test_phi_summary_fields(self, wa, tmp_path):
        exp_dir = tmp_path / "e1_spread"
        exp_dir.mkdir()
        _write_phi(
            exp_dir / "5_Flux_Ensemble.phi.meas.csv",
            [0, 1, 2, 3],
            [1795750.0, 24947.5, 19685.6, 18981.1],
        )
        summary_dir = tmp_path / "summary"
        summary_dir.mkdir()
        wa._build_phi_summary({"e1_spread": str(exp_dir)}, str(summary_dir))
        out = pd.read_csv(summary_dir / "phi_summary.csv")
        row = out.iloc[0]
        assert row["phi_initial"] == pytest.approx(1795750.0)
        assert row["phi_final"] == pytest.approx(18981.1)
        assert row["n_phi_records"] == 4
        assert row["nopt_iterations"] == 3
        assert row["phi_reduction_pct"] == pytest.approx(98.943, abs=1e-3)
        # No realization columns mislabeled as iterations
        phi_iter_cols = [c for c in out.columns if c.startswith("phi_iter_")]
        assert sorted(phi_iter_cols) == ["phi_iter_0", "phi_iter_1", "phi_iter_2", "phi_iter_3"]

    def test_unsorted_rows_are_sorted(self, wa, tmp_path):
        phi_path = tmp_path / "phi.meas.csv"
        _write_phi(phi_path, [2, 0, 3, 1], [30.0, 100.0, 20.0, 50.0])
        phi = wa._read_phi_iterations(str(phi_path))
        assert list(phi["iteration"]) == [0, 1, 2, 3]
        assert list(phi["mean"]) == [100.0, 50.0, 30.0, 20.0]

    def test_duplicate_iterations_raise(self, wa, tmp_path):
        phi_path = tmp_path / "phi.meas.csv"
        _write_phi(phi_path, [0, 1, 1, 2], [100.0, 50.0, 45.0, 30.0])
        with pytest.raises(ValueError, match="duplicate"):
            wa._read_phi_iterations(str(phi_path))

    def test_non_contiguous_iterations_raise(self, wa, tmp_path):
        phi_path = tmp_path / "phi.meas.csv"
        _write_phi(phi_path, [0, 1, 3], [100.0, 50.0, 30.0])
        with pytest.raises(ValueError, match="contiguous"):
            wa._read_phi_iterations(str(phi_path))

    def test_missing_required_column_raises(self, wa, tmp_path):
        phi_path = tmp_path / "phi.meas.csv"
        pd.DataFrame({"iteration": [0, 1], "total_runs": [200, 400]}).to_csv(phi_path, index=False)
        with pytest.raises(ValueError, match="missing required"):
            wa._read_phi_iterations(str(phi_path))


def _write_eval_metrics(exp_dir, sites, r2, kge, rmse, bias, monthly=False):
    fname = "evaluation_monthly_metrics.csv" if monthly else "evaluation_metrics.csv"
    df = pd.DataFrame(
        {"r2_swim": r2, "kge_swim": kge, "rmse_swim": rmse, "bias_swim": bias},
        index=pd.Index(sites, name="fid"),
    )
    df.to_csv(Path(exp_dir) / fname)


@pytest.fixture()
def paired_dirs(tmp_path):
    """Two synthetic arm dirs with daily metrics for 12 sites."""
    rng = np.random.default_rng(7)
    sites = [f"S{i:02d}" for i in range(12)]
    e1 = tmp_path / "e1_spread"
    e2 = tmp_path / "e2_fixed_sd"
    e1.mkdir()
    e2.mkdir()
    base_r2 = rng.uniform(0.4, 0.8, 12)
    base_kge = rng.uniform(0.5, 0.9, 12)
    base_rmse = rng.uniform(0.8, 1.4, 12)
    _write_eval_metrics(
        e1, sites, base_r2 + 0.01, base_kge, base_rmse - 0.02, rng.normal(0, 0.2, 12)
    )
    _write_eval_metrics(e2, sites, base_r2, base_kge, base_rmse, rng.normal(0, 0.2, 12))
    return {"e1_spread": str(e1), "e2_fixed_sd": str(e2)}


class TestPairedDeltaSummary:
    def test_deterministic_under_fixed_seed(self, wa, paired_dirs, tmp_path):
        s1 = tmp_path / "sum1"
        s2 = tmp_path / "sum2"
        s1.mkdir()
        s2.mkdir()
        df1 = wa._build_paired_delta_summary(paired_dirs, str(s1), reps=2000)
        df2 = wa._build_paired_delta_summary(paired_dirs, str(s2), reps=2000)
        pd.testing.assert_frame_equal(df1, df2)

    def test_required_columns_no_win_rate(self, wa, paired_dirs, tmp_path):
        s = tmp_path / "sum"
        s.mkdir()
        df = wa._build_paired_delta_summary(paired_dirs, str(s), reps=500)
        assert list(df.columns) == [
            "scale",
            "metric",
            "n_sites",
            "delta_definition",
            "favorable_direction",
            "median_delta",
            "mean_delta",
            "bootstrap_seed",
            "bootstrap_reps",
            "ci_lower",
            "ci_upper",
        ]
        assert not any("win" in c for c in df.columns)
        assert (s / "paired_delta_summary.csv").exists()

    def test_finite_filtering_is_per_metric(self, wa, paired_dirs, tmp_path):
        # Inject a NaN into kge_swim for one site in the spread arm only
        e1_path = Path(paired_dirs["e1_spread"]) / "evaluation_metrics.csv"
        d = pd.read_csv(e1_path, index_col=0)
        d.loc[d.index[0], "kge_swim"] = np.nan
        d.to_csv(e1_path)
        s = tmp_path / "sum"
        s.mkdir()
        df = wa._build_paired_delta_summary(paired_dirs, str(s), reps=500)
        n = {r["metric"]: r["n_sites"] for _, r in df.iterrows()}
        assert n["kge"] == 11
        assert n["nse"] == 12
        assert n["rmse"] == 12
        assert n["abs_mbe"] == 12

    def test_bias_contrast_uses_absolute_mbe(self, wa, tmp_path):
        # Signed deltas would say spread is 0.7 lower; absolute deltas say
        # spread is 0.3 closer to zero. The artifact must report the latter.
        sites = ["A", "B", "C"]
        e1 = tmp_path / "e1_spread"
        e2 = tmp_path / "e2_fixed_sd"
        e1.mkdir()
        e2.mkdir()
        ones = np.ones(3)
        _write_eval_metrics(e1, sites, ones * 0.7, ones * 0.8, ones * 1.0, ones * -0.2)
        _write_eval_metrics(e2, sites, ones * 0.7, ones * 0.8, ones * 1.0, ones * 0.5)
        s = tmp_path / "sum"
        s.mkdir()
        df = wa._build_paired_delta_summary(
            {"e1_spread": str(e1), "e2_fixed_sd": str(e2)}, str(s), reps=500
        )
        abs_row = df[df["metric"] == "abs_mbe"].iloc[0]
        assert abs_row["median_delta"] == pytest.approx(-0.3)
        assert abs_row["delta_definition"] == "abs(mbe_spread) - abs(mbe_fixed_sd)"

    def test_missing_monthly_files_skipped(self, wa, paired_dirs, tmp_path):
        s = tmp_path / "sum"
        s.mkdir()
        df = wa._build_paired_delta_summary(paired_dirs, str(s), reps=500)
        assert set(df["scale"]) == {"daily"}
        assert len(df) == 4


class TestCli:
    def test_container_required_for_calibration_runs(self, wa, monkeypatch, capsys):
        # The tag defaults to run22; without an explicit container a bare run
        # would silently calibrate the stale base container into
        # Run-22-labeled result dirs. The CLI must refuse instead.
        monkeypatch.setattr("sys.argv", ["run_weighting_ablation.py"])
        with pytest.raises(SystemExit) as exc:
            wa.main()
        assert exc.value.code == 2
        assert "--container is required" in capsys.readouterr().err

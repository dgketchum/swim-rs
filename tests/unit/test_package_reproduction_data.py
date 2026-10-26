"""Tests for ``scripts/package_reproduction_data.py`` (file selection and trimming rules)."""

import importlib.util
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "package_reproduction_data.py"


@pytest.fixture(scope="module")
def pkg():
    spec = importlib.util.spec_from_file_location("package_reproduction_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path, size=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "swim"
    run = root / "5_Flux_Ensemble" / "results" / "run23"
    _touch(run / "5_Flux_Ensemble.3.par.csv", 3)
    _touch(run / "pestrun_kept" / "worker_0" / "big.bin", 100)
    _touch(run / "archive" / "4_pest_outputs" / "5_Flux_Ensemble.3.obs.csv", 50)
    _touch(run / "archive" / "4_pest_outputs" / "5_Flux_Ensemble.0.obs.csv", 50)
    _touch(run / "archive" / "4_pest_outputs" / "5_Flux_Ensemble.0.par.csv", 2)
    _touch(run / "archive" / "3_problem_definition" / "loc.mat", 40)
    _touch(run / "archive" / "3_problem_definition" / "run23.pst", 4)
    _touch(run / "archive" / "6_evaluation" / "site_daily_timeseries" / "US-Ne1.csv", 5)
    _touch(run / "__pycache__" / "x.pyc", 1)
    _touch(root / "5_Flux_Ensemble" / "data" / "5_Flux_Ensemble_run23.swim.lock", 0)
    _touch(root / "5_Flux_Ensemble" / "data" / "5_Flux_Ensemble_run23.swim" / "zarr.json", 6)
    e7 = root / "7_Applied_Water" / "results" / "e7cal"
    _touch(e7 / "archive" / "4_pest_outputs" / "batch_000" / "7_Applied_Water.3.obs.csv", 30)
    _touch(e7 / "archive" / "4_pest_outputs" / "batch_000" / "7_Applied_Water.base.rei", 30)
    _touch(e7 / "archive" / "4_pest_outputs" / "merged" / "merged_posterior.json", 7)
    _touch(e7 / "archive_superseded_singlerun" / "old.json", 8)
    return root


def test_collect_applies_the_trimming_rules(pkg, fake_root):
    files = dict(
        pkg.collect(
            fake_root,
            ["5_Flux_Ensemble/results/run23", "5_Flux_Ensemble/data/5_Flux_Ensemble_run23.swim"],
        )
    )
    assert "5_Flux_Ensemble/results/run23/5_Flux_Ensemble.3.par.csv" in files
    assert "5_Flux_Ensemble/results/run23/archive/4_pest_outputs/5_Flux_Ensemble.3.obs.csv" in files
    assert "5_Flux_Ensemble/results/run23/archive/4_pest_outputs/5_Flux_Ensemble.0.par.csv" in files
    assert "5_Flux_Ensemble/results/run23/archive/3_problem_definition/run23.pst" in files
    assert (
        "5_Flux_Ensemble/results/run23/archive/6_evaluation/site_daily_timeseries/US-Ne1.csv"
        in files
    )
    assert "5_Flux_Ensemble/data/5_Flux_Ensemble_run23.swim/zarr.json" in files
    dropped = {
        "5_Flux_Ensemble/results/run23/pestrun_kept/worker_0/big.bin",
        "5_Flux_Ensemble/results/run23/archive/4_pest_outputs/5_Flux_Ensemble.0.obs.csv",
        "5_Flux_Ensemble/results/run23/archive/3_problem_definition/loc.mat",
        "5_Flux_Ensemble/results/run23/__pycache__/x.pyc",
    }
    assert not dropped & set(files)
    assert sum(files.values()) == 3 + 50 + 2 + 4 + 5 + 6


def test_collect_excludes_batch_ensembles_and_superseded_dirs(pkg, fake_root):
    files = dict(
        pkg.collect(
            fake_root, ["7_Applied_Water/results/e7cal"], exclude_dirs=["archive_superseded_*"]
        )
    )
    assert set(files) == {
        "7_Applied_Water/results/e7cal/archive/4_pest_outputs/merged/merged_posterior.json"
    }


def test_collect_refuses_a_missing_item(pkg, fake_root):
    with pytest.raises(FileNotFoundError, match="package item missing"):
        pkg.collect(fake_root, ["5_Flux_Ensemble/results/run99"])


def test_collect_glob_takes_matching_files_without_descending(pkg, fake_root):
    base = fake_root / "6_Flux_International" / "results" / "baseline"
    _touch(base / "US-Ne1.csv", 3)
    _touch(base / "evaluation_metrics.csv", 4)
    _touch(base / "archive" / "6_evaluation" / "evaluation_metrics.csv", 5)
    _touch(base / "notes.txt", 1)
    files = dict(pkg.collect(fake_root, ["6_Flux_International/results/baseline/*.csv"]))
    assert set(files) == {
        "6_Flux_International/results/baseline/US-Ne1.csv",
        "6_Flux_International/results/baseline/evaluation_metrics.csv",
    }
    with pytest.raises(FileNotFoundError, match="glob matches nothing"):
        pkg.collect(fake_root, ["6_Flux_International/results/baseline/*.parquet"])


def test_write_tar_keeps_relative_members_and_leaves_no_sidecars(pkg, fake_root, tmp_path):
    files = pkg.collect(fake_root, ["5_Flux_Ensemble/results/run23"])
    out = tmp_path / "stage" / "swim_data_test.tar.gz"
    out.parent.mkdir()
    pkg.write_tar(fake_root, files, out)
    with tarfile.open(out) as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile())
    assert names == sorted(rel for rel, _ in files)
    assert sorted(p.name for p in (tmp_path / "stage").iterdir()) == ["swim_data_test.tar.gz"]


def test_every_package_item_exists_on_this_machine_or_is_skipped(pkg):
    """Dry-run guard: on the authoring machine every declared item must exist."""
    root = pkg.swim_root()
    if not root.exists():
        pytest.skip(f"data root {root} not present")
    for key, spec in pkg.ROOT_PACKAGES.items():
        for item in spec["items"]:
            if any(ch in item for ch in "*?["):
                assert any(p.is_file() for p in root.glob(item)), f"{key}: {item}"
            else:
                assert (root / item).exists(), f"{key}: {item}"
    for key, spec in pkg.REPO_PACKAGES.items():
        for item in spec["items"]:
            assert (pkg.REPO / item).exists(), f"{key}: {item}"


def test_reference_package_drops_the_obs_support_tables(pkg, tmp_path):
    repo = tmp_path / "clone"
    figures = repo / "paper" / "data" / "final" / "figures"
    _touch(figures / "fig02_ndvi_support.csv")
    _touch(figures / "figs02_ndvi_support.csv")
    _touch(figures / "figs02_etf_support.csv")
    _touch(figures / "fig_manifest.json")
    spec = pkg.REPO_PACKAGES["reference"]
    files = dict(pkg.collect(repo, spec["items"], exclude_files=spec["exclude_files"]))
    assert set(files) == {
        "paper/data/final/figures/fig02_ndvi_support.csv",
        "paper/data/final/figures/fig_manifest.json",
    }

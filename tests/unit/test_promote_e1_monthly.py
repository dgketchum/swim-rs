"""``promote_e1_monthly.py`` gates and re-freezes the Volk-protocol E1 monthly package."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EX5 = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"
NOW = "2026-09-23T12:00:00-06:00"
HEAD = "f" * 40


@pytest.fixture(scope="module")
def mod():
    if str(EX5) not in sys.path:
        sys.path.insert(0, str(EX5))
    spec = importlib.util.spec_from_file_location(
        "promote_e1_monthly", EX5 / "promote_e1_monthly.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _daily(start, end, seed):
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    eto = pd.Series(4.0 + 1.5 * np.sin(np.arange(len(idx)) / 58.0), index=idx)
    swim = pd.Series(0.72 * eto.values + rng.normal(0, 0.15, len(idx)), index=idx)
    flux = pd.Series(0.70 * eto.values + rng.normal(0, 0.2, len(idx)), index=idx)
    return idx, eto, swim, flux


@pytest.fixture
def world(tmp_path, mod):
    """Synthetic archive + flux + OpenET monthly files for three sites, and a source bundle.

    Site A: two full years; site B: one year with a few interior gaps; site C: two
    valid months only (pooled-only). The source bundle's grouped estimates come from
    the same helpers the script replicates with, so the round trip is exact.
    """
    run = tmp_path / mod.ex5_paths.CANONICAL_RUN
    ts_dir = run / "archive" / "6_evaluation" / "site_daily_timeseries"
    flux_dir = tmp_path / "daily_flux_files_2pt1"
    monthly_dir = tmp_path / "openet_flux_2pt1" / "monthly_data"
    for d in (ts_dir, flux_dir, monthly_dir):
        d.mkdir(parents=True)
    spans = {
        "A": ("2018-01-01", "2019-12-31"),
        "B": ("2018-01-01", "2018-12-31"),
        "C": ("2018-03-01", "2018-04-30"),
    }
    records = []
    for seed, (fid, (s, e)) in enumerate(spans.items()):
        idx, eto, swim, flux = _daily(s, e, seed)
        if fid == "B":
            flux.iloc[40:43] = np.nan
        pd.DataFrame({"swim_ET": swim, "eto": eto}, index=idx).rename_axis("date").to_csv(
            ts_dir / f"{fid}.csv"
        )
        pd.DataFrame({"ET_corr": flux}, index=idx).rename_axis("date").to_csv(
            flux_dir / f"{fid}_daily_data.csv"
        )
        months = pd.date_range(s, e, freq="MS")
        ens = flux.resample("MS").sum().reindex(months) * 0.96
        pd.DataFrame({"ensemble_mean_3x3": ens}, index=months).rename_axis("DATE").to_csv(
            monthly_dir / f"{fid}.csv"
        )
    # the archive-side replication is the reference used to write the bundle
    (run / "archive" / "6_evaluation" / "monthly_paired_metrics.csv").write_text("fid,n\nA,24\n")
    estimates, cohorts = mod.replicate_from_archive(ts_dir, flux_dir, monthly_dir)
    assert cohorts[mod.AGG_POOLED] == (("A", 24), ("B", 12), ("C", 2))
    assert cohorts[mod.AGG_WEIGHTED] == (("A", 24), ("B", 12))
    records = [
        {
            "scale": "monthly",
            "aggregation": agg,
            "model": model,
            "metric": metric,
            "estimate": value,
            "n_sites": len(cohorts[agg]),
            "n_pairs": sum(n for _, n in cohorts[agg]),
        }
        for (agg, model, metric), value in estimates.items()
    ]
    source = run / "monthly_volk2024"
    source.mkdir()
    pd.DataFrame(records).to_csv(source / "evaluation_grouped_monthly_metrics.csv", index=False)
    (source / "evaluation_grouped_monthly_contrasts.csv").write_text(
        "scale,aggregation\nmonthly,x\n"
    )
    (source / "evaluation_monthly_metrics.csv").write_text(
        "fid,n,station_weighted\nA,24,True\nB,12,True\nC,2,False\n"
    )
    (source / "evaluation_sites_excluded.csv").write_text("site,reason\n")
    meta = {
        "benchmark_construction": mod.MONTHLY_TOKEN,
        "git": {"sha": "a" * 40, "dirty": False},
        "paths": {"flux_dir": str(flux_dir), "openet_monthly_dir": str(monthly_dir)},
        "static_exclusions": [],
        "pooled_only_sites": ["C"],
        "cohorts": {
            agg: {
                "n_sites": len(c),
                "n_pairs": sum(n for _, n in c),
                "sites": [{"fid": f, "n": n} for f, n in c],
            }
            for agg, c in cohorts.items()
        },
        "output_hashes": {
            name: mod.sha256_file(source / name)
            for name in (
                "evaluation_grouped_monthly_metrics.csv",
                "evaluation_grouped_monthly_contrasts.csv",
            )
        },
    }
    (source / "evaluation_grouped_monthly_metadata.json").write_text(json.dumps(meta, indent=2))

    final = tmp_path / "final"
    monthly = final / mod.PACKAGE / "monthly"
    monthly.mkdir(parents=True)
    for name in mod.MONTHLY_FILES:
        (monthly / name).write_text(f"old {name}\n")
    prior = {
        "schema_version": "e1_openet_benchmark_reporting/v1",
        "promoted_at": "2026-09-01T10:26:21-06:00",
        "promoted_at_git_sha": "e" * 40,
        "reporting_contract": {"primary_metrics": ["kge"]},
        "cohorts": {
            "daily": {"n_sites": 45},
            "monthly": {"n_sites": 30, "n_paired_site_months": 1301},
        },
        "artifact_sha256": {
            "daily/x.csv": "d" * 64,
            "monthly/evaluation_monthly_metrics.csv": "0" * 64,
        },
        "analysis_code": {"daily_generation_git_sha": "b" * 40, "current_file_sha256": {}},
        "validation": {"daily_record_grouped_identity": "PASS"},
        "legacy_and_exclusions": {"superseded": "old note"},
    }
    (final / mod.PACKAGE / "MANIFEST.json").write_text(json.dumps(prior, indent=2) + "\n")

    repo = tmp_path / "repo"
    for rel in mod.ANALYSIS_FILES:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"# {rel}\n")
    return {
        "run": run,
        "source": source,
        "final": final,
        "repo": repo,
        "archive": run / "archive" / "6_evaluation",
        "ts_dir": ts_dir,
        "flux_dir": flux_dir,
        "monthly_dir": monthly_dir,
        "estimates": estimates,
        "prior": prior,
    }


def test_replication_pooled_only_site_enters_pooled_not_weighted(mod, world):
    est = world["estimates"]
    assert len(est) == mod.N_GROUPED_ESTIMATES
    pooled = {k for k in est if k[0] == mod.AGG_POOLED}
    weighted = {k for k in est if k[0] == mod.AGG_WEIGHTED}
    assert len(pooled) == 12 and len(weighted) == 6
    assert all(np.isfinite(v) for v in est.values())


def test_sidecar_gate_rejects_hash_mismatch(mod, world):
    files = mod.read_source(world["source"])
    assert mod.check_sidecar(files)["cohorts"]
    files["evaluation_grouped_monthly_contrasts.csv"].write_text("scale,aggregation\nmonthly,y\n")
    with pytest.raises(mod.PromotionError, match="sha256"):
        mod.check_sidecar(files)


def test_generation_tree_gate(mod, world):
    meta = mod.check_sidecar(mod.read_source(world["source"]))
    sha = meta["git"]["sha"]
    # the whole-worktree dirty flag is informational: a sha-at-HEAD bundle with clean
    # analysis files passes even when the sidecar says dirty
    meta["git"]["dirty"] = True
    assert mod.check_generation_tree(meta, sha, []) is True
    with pytest.raises(mod.PromotionError, match="!= HEAD"):
        mod.check_generation_tree(meta, "b" * 40, [])
    with pytest.raises(mod.PromotionError, match="uncommitted analysis files"):
        mod.check_generation_tree(meta, sha, ["src/swimrs/calibrate/flux_utils.py"])
    assert mod.check_generation_tree(meta, "b" * 40, ["x.py"], allow_dirty=True) is False


def test_sidecar_gate_rejects_wrong_construction_token(mod, world):
    files = mod.read_source(world["source"])
    meta_path = files["evaluation_grouped_monthly_metadata.json"]
    meta = json.loads(meta_path.read_text())
    meta["benchmark_construction"] = "full_month_openet_totals_v2pt1"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(mod.PromotionError, match="benchmark_construction"):
        mod.check_sidecar(files)


def test_replication_gate_catches_a_changed_estimate(mod, world):
    files = mod.read_source(world["source"])
    meta = mod.check_sidecar(files)
    est, cohorts = mod.replicate_from_archive(
        world["ts_dir"], world["flux_dir"], world["monthly_dir"]
    )
    assert mod.compare_replication(est, cohorts, files, meta) < 1e-12

    df = pd.read_csv(files["evaluation_grouped_monthly_metrics.csv"])
    df.loc[0, "estimate"] += 1e-4
    df.to_csv(files["evaluation_grouped_monthly_metrics.csv"], index=False)
    with pytest.raises(mod.PromotionError, match="replication differs"):
        mod.compare_replication(est, cohorts, files, meta)


def test_replication_uses_raw_eto_when_archive_carries_corrected(mod, world, tmp_path):
    """Archives that export the corrected ETo as ``eto`` must be replicated on ``eto_raw``."""
    ts_dir = tmp_path / "ts_corrected"
    ts_dir.mkdir()
    for f in sorted(world["ts_dir"].glob("*.csv")):
        ts = pd.read_csv(f, index_col="date", parse_dates=True)
        ts["eto_raw"] = ts["eto"]
        # a day-varying correction, as the model consumed it (a uniform scale
        # would cancel in the fraction-of-ETo gap fill)
        ts["eto"] = ts["eto"] * (0.7 + 0.3 * np.sin(np.arange(len(ts)) / 9.0))
        ts.to_csv(ts_dir / f.name)
    ref, ref_cohorts = mod.replicate_from_archive(
        world["ts_dir"], world["flux_dir"], world["monthly_dir"]
    )
    est, cohorts = mod.replicate_from_archive(ts_dir, world["flux_dir"], world["monthly_dir"])
    assert cohorts == ref_cohorts
    assert max(abs(est[k] - ref[k]) for k in ref) < 1e-12
    # and the corrected column, used by mistake, would not reproduce the bundle
    for f in ts_dir.glob("*.csv"):
        ts = pd.read_csv(f, index_col="date", parse_dates=True)
        ts.drop(columns=["eto_raw"]).to_csv(f)
    wrong, _ = mod.replicate_from_archive(ts_dir, world["flux_dir"], world["monthly_dir"])
    assert max(abs(wrong[k] - ref[k]) for k in ref) > 1e-9


def test_replication_gate_catches_a_cohort_mismatch(mod, world):
    files = mod.read_source(world["source"])
    meta = mod.check_sidecar(files)
    est, cohorts = mod.replicate_from_archive(
        world["ts_dir"], world["flux_dir"], world["monthly_dir"]
    )
    meta["cohorts"][mod.AGG_WEIGHTED]["sites"].append({"fid": "C", "n": 2})
    with pytest.raises(mod.PromotionError, match="cohort differs"):
        mod.compare_replication(est, cohorts, files, meta)


def test_promote_moves_old_package_and_rewrites_manifest(mod, world):
    files = mod.read_source(world["source"])
    meta = mod.check_sidecar(files)
    final, archive, repo = world["final"], world["archive"], world["repo"]

    manifest = mod.promote(files, meta, final, archive, repo, 2.3e-9, now=NOW, head_sha=HEAD)

    monthly = final / mod.PACKAGE / "monthly"
    for name in mod.MONTHLY_FILES:
        assert (monthly / name).read_bytes() == files[name].read_bytes()
    superseded = final / mod.SUPERSEDED_DIRNAME
    for name in mod.MONTHLY_FILES:
        assert (superseded / name).read_text() == f"old {name}\n"
    readme = (superseded / "README.md").read_text()
    assert "30 sites, 1301 site-months" in readme
    assert "Do not use these files" in readme
    assert mod.sha256_bytes(b"old evaluation_monthly_metrics.csv\n") in readme

    written = json.loads((final / mod.PACKAGE / "MANIFEST.json").read_text())
    assert written == manifest
    # daily and temporal fields are untouched
    assert written["cohorts"]["daily"] == {"n_sites": 45}
    assert written["artifact_sha256"]["daily/x.csv"] == "d" * 64
    assert written["analysis_code"]["daily_generation_git_sha"] == "b" * 40
    assert written["validation"]["daily_record_grouped_identity"] == "PASS"
    assert written["legacy_and_exclusions"]["superseded"] == "old note"
    # monthly fields rewritten
    m = written["cohorts"]["monthly"]
    assert m[mod.AGG_POOLED] == {"n_sites": 3, "n_paired_site_months": 38, "min_paired_months": 1}
    assert m[mod.AGG_WEIGHTED] == {"n_sites": 2, "n_paired_site_months": 36, "min_paired_months": 3}
    assert m["pooled_only_sites"] == ["C"]
    for name in mod.MONTHLY_FILES:
        assert written["artifact_sha256"][f"monthly/{name}"] == mod.sha256_file(files[name])
    assert written["analysis_code"]["monthly_generation_git_sha"] == "a" * 40
    assert written["analysis_code"]["monthly_generation_worktree_dirty"] is False
    assert set(mod.ANALYSIS_FILES) <= set(written["analysis_code"]["current_file_sha256"])
    assert "2.300e-09" in written["validation"]["monthly_replication_from_archive"]
    assert "30 sites, 1301 months" in written["legacy_and_exclusions"]["superseded_monthly_28day"]
    assert written["promoted_at"] == NOW
    (hist,) = written["promotion_history"]
    assert hist["from"]["n_sites"] == 30 and hist["to"][mod.AGG_POOLED]["n_sites"] == 3

    # archive Cat 6 refresh
    assert (archive / "monthly_paired_metrics_28day_superseded.csv").read_text() == "fid,n\nA,24\n"
    assert (archive / "monthly_paired_metrics.csv").read_bytes() == files[
        "evaluation_monthly_metrics.csv"
    ].read_bytes()
    bundle = archive / mod.ARCHIVE_SUBDIR
    for name in mod.MONTHLY_FILES:
        assert (bundle / name).read_bytes() == files[name].read_bytes()
    side = json.loads((bundle / "PROMOTION.json").read_text())
    assert side["generation_git_sha"] == "a" * 40 and side["promoted_at"] == NOW

    # a second promotion refuses to overwrite the superseded package
    with pytest.raises(mod.PromotionError, match="refusing to overwrite"):
        mod.promote(files, meta, final, archive, repo, 2.3e-9, now=NOW, head_sha=HEAD)
    # and the promoted copy now matches the source
    assert all(status == "match" for _, status, _ in mod.compare_promoted(files, monthly))


def test_promote_checks_every_target_before_writing(mod, world):
    files = mod.read_source(world["source"])
    meta = mod.check_sidecar(files)
    final, archive, repo = world["final"], world["archive"], world["repo"]
    (archive / mod.ARCHIVE_SUBDIR).mkdir()
    with pytest.raises(mod.PromotionError, match="archive monthly bundle"):
        mod.promote(files, meta, final, archive, repo, 0.0, now=NOW, head_sha=HEAD)
    # nothing moved
    assert not (final / mod.SUPERSEDED_DIRNAME).exists()
    assert (final / mod.PACKAGE / "monthly" / mod.MONTHLY_FILES[0]).read_text().startswith("old ")
    assert json.loads((final / mod.PACKAGE / "MANIFEST.json").read_text()) == world["prior"]


def test_main_check_mode_and_write(mod, world, monkeypatch):
    monkeypatch.setattr(mod, "_head_sha", lambda repo: "a" * 40)  # the sidecar sha
    monkeypatch.setattr(mod, "analysis_files_dirty", lambda repo: [])
    monkeypatch.setattr(mod.ex5_paths, "REPO", world["repo"])
    common = [
        "--source-dir",
        str(world["source"]),
        "--run-dir",
        str(world["run"]),
        "--final-dir",
        str(world["final"]),
    ]
    assert mod.main(common) == 1  # gates pass, package differs
    with pytest.raises(mod.PromotionError, match="requires the replication gate"):
        mod.main([*common, "--skip-replication", "--write"])
    assert mod.main([*common, "--write"]) == 0
    assert mod.main(common) == 0
    written = json.loads((world["final"] / mod.PACKAGE / "MANIFEST.json").read_text())
    assert written["promoted_at_git_sha"] == "a" * 40
    assert written["analysis_code"]["monthly_analysis_files_clean_at_promotion"] is True


def test_main_refuses_a_bundle_from_another_commit(mod, world, monkeypatch):
    monkeypatch.setattr(mod, "_head_sha", lambda repo: HEAD)
    monkeypatch.setattr(mod, "analysis_files_dirty", lambda repo: [])
    monkeypatch.setattr(mod.ex5_paths, "REPO", world["repo"])
    common = [
        "--source-dir",
        str(world["source"]),
        "--run-dir",
        str(world["run"]),
        "--final-dir",
        str(world["final"]),
    ]
    with pytest.raises(mod.PromotionError, match="not from the committed analysis code"):
        mod.main(common)
    assert mod.main([*common, "--allow-dirty", "--write"]) == 0
    written = json.loads((world["final"] / mod.PACKAGE / "MANIFEST.json").read_text())
    assert written["analysis_code"]["monthly_analysis_files_clean_at_promotion"] is False

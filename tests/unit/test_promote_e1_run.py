"""``promote_e1_run.py`` gates and promotes a whole Ex5 run's E1 benchmark record."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EX5 = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"
NOW = "2026-09-25T12:00:00-06:00"
HEAD = "f" * 40


@pytest.fixture(scope="module")
def mod():
    if str(EX5) not in sys.path:
        sys.path.insert(0, str(EX5))
    spec = importlib.util.spec_from_file_location("promote_e1_run", EX5 / "promote_e1_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def make_run(mod, results, run, value):
    """Synthetic daily/monthly/temporal bundles + ablation dirs for ``run``.

    ``value`` perturbs every metric so two runs produce different bytes.
    """
    run_dir = results / run
    par = run_dir / "5_Flux_Ensemble.3.par.csv"
    _csv(par, f"real,aw\n0,{value}\n")
    container = f"/data/5_Flux_Ensemble_{run}.swim"
    boot = {"unit": "site", "reps": mod.BOOTSTRAP_REPS_DEFAULT, "seed": 42}
    sites = [("A", 100), ("B", 80)]
    metric_rows = "fid,n,r2_swim,rmse_swim,bias_swim,kge_swim\n" + "".join(
        f"{f},{n},{value},{1 + value},{-value},{0.5 + value}\n" for f, n in sites
    )
    excluded = [{"site": "X", "reason": f"no flux {value}"}]
    ledger = f"site,reason\nX,no flux {value}\n"
    common = {
        "git": {"sha": "a" * 40},
        "bootstrap": boot,
        "paths": {"par_csv": str(par), "container": container},
        "excluded_sites": excluded,
        "sites": [{"fid": f, "n": n} for f, n in sites],
    }

    daily = run_dir / mod.DAILY_SUBDIR
    _csv(daily / "evaluation_grouped_daily_metrics.csv", f"model,kge\nswim,{value}\n")
    _csv(daily / "evaluation_grouped_daily_contrasts.csv", f"contrast,d\nswim-ens,{value}\n")
    _csv(daily / "evaluation_paired_daily_records.csv", f"fid,date,swim\nA,2020-01-01,{value}\n")
    _csv(daily / "evaluation_metrics.csv", metric_rows)
    _csv(daily / "evaluation_sites_excluded.csv", ledger)
    _json(
        daily / mod.DAILY_METADATA,
        {
            **common,
            "scale": "daily",
            "openet_source": "volk",
            "benchmark_source": mod.BENCHMARK_SOURCE_MACHINE_TOKENS["volk"],
            "benchmark_construction": mod.CONSTRUCTION_TOKENS["daily"],
            "n_sites": 2,
            "n_pairs": 180,
            "input_hashes": {"par_csv": {"sha256": mod.sha256_file(par)}},
            "output_hashes": {n: mod.sha256_file(daily / n) for n in mod.DAILY_HASHED},
        },
    )

    monthly = run_dir / mod.MONTHLY_SUBDIR
    _csv(monthly / "evaluation_grouped_monthly_metrics.csv", f"model,kge\nswim,{value}\n")
    _csv(monthly / "evaluation_grouped_monthly_contrasts.csv", f"contrast,d\nx,{value}\n")
    _csv(monthly / "evaluation_monthly_metrics.csv", metric_rows)
    _csv(monthly / "evaluation_sites_excluded.csv", ledger)
    _json(
        monthly / mod.MONTHLY_METADATA,
        {
            **common,
            "benchmark_construction": mod.CONSTRUCTION_TOKENS["monthly"],
            "pooled_only_sites": [],
            "paths": {**common["paths"], "flux_dir": "f", "openet_monthly_dir": "m"},
            "cohorts": {
                mod.AGG_POOLED: {"n_sites": 2, "n_pairs": 180},
                mod.AGG_WEIGHTED: {"n_sites": 2, "n_pairs": 180},
            },
            "output_hashes": {n: mod.sha256_file(monthly / n) for n in mod.pm.HASHED_IN_SIDECAR},
        },
    )

    temporal = run_dir / mod.TEMPORAL_SUBDIR
    for name in mod.TEMPORAL_FILES[:-1]:
        _csv(temporal / name, f"k,v\n{name},{value}\n")
    _json(
        temporal / mod.TEMPORAL_METADATA,
        {
            "git": {"sha": "a" * 40},
            "bootstrap": boot,
            "cohort": {
                "n_common_sites": 2,
                "class_row_counts": {"retrieval": 20, "between_retrieval": 160},
            },
            "parent_evaluator_metadata": {
                "benchmark_construction": mod.CONSTRUCTION_TOKENS["daily"]
            },
            "parent_bundle": {
                "rehashed_artifacts": {n: mod.sha256_file(daily / n) for n in mod.DAILY_HASHED},
                "metadata_sha256": mod.sha256_file(daily / mod.DAILY_METADATA),
            },
            "output_hashes": {n: mod.sha256_file(temporal / n) for n in mod.TEMPORAL_FILES[:-1]},
        },
    )

    ablation = results / f"ablation_{run}_summary"
    deltas = "fid,e1_r2_swim,e1_rmse_swim,e1_bias_swim,e1_kge_swim,n_paired\n" + "".join(
        f"{f},{value},{1 + value},{-value},{0.5 + value},{n}\n" for f, n in sites
    )
    _csv(ablation / "paired_site_deltas_daily.csv", deltas)
    _csv(ablation / "paired_site_deltas_monthly.csv", deltas)
    _csv(ablation / "paired_delta_summary.csv", f"scale,delta\ndaily,{value}\n")
    _csv(ablation / "ablation_summary.csv", f"arm,phi\ne1_spread,{value}\n")
    _json(
        results / f"ablation_{run}_e1_spread" / "runtime.json",
        {
            "experiment_id": "e1_spread",
            "container_path": f"/data/5_Flux_Ensemble_{run}ablation.swim",
        },
    )
    return mod.default_sources(run, results)


def _promote(mod, final, src, run, superseded="superseded_x", now=NOW):
    gates, metas, details = mod.run_gates(src, run, heavy=False)
    assert all(g.ok for g in gates), gates
    rows = mod.compare_entries(mod.package_entries(src), final)
    return mod.promote(
        final, src, run, rows, metas, details, superseded, mod.ex5_paths.REPO, now, HEAD, {}, True
    )


@pytest.fixture
def world(tmp_path, mod):
    """``run0`` frozen in ``final`` (plus an untouched reconstruction package); ``run1`` staged."""
    results, final = tmp_path / "results", tmp_path / "final"
    src0 = make_run(mod, results, "run0", 0.5)
    src1 = make_run(mod, results, "run1", 0.7)
    _promote(mod, final, src0, "run0", now="2026-09-01T00:00:00-06:00")
    _csv(final / mod.PACKAGE / "reconstruction_fidelity" / "x.csv", "keep\n")
    return {"results": results, "final": final, "src0": src0, "src1": src1}


@pytest.fixture
def cli(mod, world, monkeypatch):
    """``main`` pointed at the synthetic results root, heavy gates and git stubbed."""
    monkeypatch.setattr(mod.ex5_paths, "load_config", lambda *a, **k: None)
    monkeypatch.setattr(mod.ex5_paths, "results_root", lambda cfg=None: world["results"])
    monkeypatch.setattr(
        mod,
        "check_git",
        lambda repo, metas, allow_dirty=False: (mod.Gate("G-GIT", True, "PASS"), {}),
    )
    monkeypatch.setattr(
        mod,
        "validate_parent_bundle",
        lambda d: (None, None, {"grouped_point_identity_max_abs_diff": 0.0}),
    )
    monkeypatch.setattr(mod.pm, "replicate_from_archive", lambda *a, **k: (None, None))
    monkeypatch.setattr(mod.pm, "compare_replication", lambda *a, **k: 0.0)
    monkeypatch.setattr(mod.pm, "_head_sha", lambda repo: HEAD)

    def run(*args):
        return mod.main(["--final-dir", str(world["final"]), *args])

    return run


def test_check_mode_match(cli, capsys):
    assert cli("--run", "run0") == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "PROMOTION_STATE: MATCH"
    assert "DIFFERS" not in out and "MISSING" not in out


def test_check_mode_differs(cli, world, capsys):
    assert cli("--run", "run1") == 1
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "PROMOTION_STATE: DIFFERS"
    assert "superseded_e1_run0_nextday_eto" in out
    table = [ln for ln in out.splitlines() if ln.startswith(("daily ", "monthly ", "temporal "))]
    assert len(table) == 16 and all(" DIFFERS " in ln for ln in table)
    # nothing is written in check mode
    assert not (world["final"] / "superseded_e1_run0_nextday_eto").exists()


def test_missing_source_is_reported(cli, world, capsys):
    (world["src1"]["temporal"] / "evaluation_temporal_interactions.csv").unlink()
    assert cli("--run", "run1") == 1
    out = capsys.readouterr().out
    assert "NO-SOURCE" in out and "G-TEMPORAL   FAIL" in out and "G-EXIST      FAIL" in out


def test_refuses_when_superseded_exists(mod, cli, world, capsys):
    (world["final"] / "superseded_e1_run0_nextday_eto").mkdir()
    assert cli("--run", "run1") == 1
    assert "G-SUPERSEDE  FAIL" in capsys.readouterr().out
    with pytest.raises(mod.PromotionError, match="G-SUPERSEDE"):
        cli("--run", "run1", "--write")
    with pytest.raises(mod.PromotionError, match="already exists"):
        _promote(mod, world["final"], world["src1"], "run1", "superseded_e1_run0_nextday_eto")


def test_gate_failures_block_write(mod, cli, world):
    meta_path = world["src1"]["daily"] / mod.DAILY_METADATA
    meta = json.loads(meta_path.read_text())
    meta["paths"]["container"] = "/data/5_Flux_Ensemble_run0.swim"
    meta_path.write_text(json.dumps(meta))
    gates, _, _ = mod.run_gates(world["src1"], "run1", heavy=False)
    failed = {g.name: g.message for g in gates if not g.ok}
    assert "run1 container" in failed["G-DAILY"]
    # the temporal decomposition no longer hashes to this daily sidecar
    assert "parent sidecar hash" in failed["G-TEMPORAL"]
    with pytest.raises(mod.PromotionError, match="G-DAILY"):
        cli("--run", "run1", "--write")
    assert not (world["final"] / "superseded_e1_run0_nextday_eto").exists()


def test_tampered_output_hash_fails(mod, world):
    (world["src1"]["monthly"] / "evaluation_grouped_monthly_metrics.csv").write_text("x\n")
    gates, _, _ = mod.run_gates(world["src1"], "run1", heavy=False)
    assert not {g.name: g for g in gates}["G-MONTHLY"].ok


def test_write_moves_old_package_and_writes_records(mod, cli, world, capsys):
    final = world["final"]
    old_manifest = json.loads((final / mod.PACKAGE / mod.MANIFEST).read_text())
    old_daily = mod.sha256_file(final / mod.PACKAGE / "daily" / "evaluation_metrics.csv")
    assert cli("--run", "run1", "--write") == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "PROMOTION_STATE: MATCH"

    sup = final / "superseded_e1_run0_nextday_eto"
    moved = sorted(p.relative_to(sup).as_posix() for p in sup.rglob("*") if p.is_file())
    assert "README.md" in moved and f"{mod.PACKAGE}/{mod.MANIFEST}" in moved
    assert mod.ABLATION_METADATA in moved
    assert len([m for m in moved if m.startswith(f"{mod.PACKAGE}/")]) == 17
    assert set(mod.ABLATION_RENAMES.values()) <= set(moved)
    assert mod.sha256_file(sup / mod.PACKAGE / "daily" / "evaluation_metrics.csv") == old_daily
    readme = (sup / "README.md").read_text()
    assert "run0" in readme and "run1" in readme and old_daily in readme
    # the run-independent package stays put
    assert (final / mod.PACKAGE / "reconstruction_fidelity" / "x.csv").read_text() == "keep\n"
    assert not (sup / mod.PACKAGE / "reconstruction_fidelity").exists()

    manifest = json.loads((final / mod.PACKAGE / mod.MANIFEST).read_text())
    assert manifest["internal_archive_id"] == "run1"
    assert manifest["promoted_at_git_sha"] == HEAD
    assert len(manifest["promotion_history"]) == len(old_manifest["promotion_history"]) + 1
    entry = manifest["promotion_history"][-1]
    assert entry["from"]["internal_archive_id"] == "run0"
    assert entry["from"]["moved_to"] == "paper/data/final/superseded_e1_run0_nextday_eto/"
    assert "superseded_e1_run0_nextday_eto" in manifest["legacy_and_exclusions"]
    assert manifest["cohorts"]["temporal_common"]["n_retrieval_days"] == 20
    for rel, digest in manifest["artifact_sha256"].items():
        assert mod.sha256_file(final / mod.PACKAGE / rel) == digest
    sidecar = json.loads((final / mod.ABLATION_METADATA).read_text())
    assert sidecar["internal_archive_id"] == "run1"
    assert mod.previous_run_tag(final) == "run1"

    # re-check is clean; a second write with the default name refuses (prev == run)
    assert cli("--run", "run1") == 0
    capsys.readouterr()
    assert cli("--run", "run1", "--write") == 0
    assert "nothing to promote" in capsys.readouterr().out


def test_ablation_rename_mapping(mod, world):
    from rebuild_e1_benchmark_evidence import RESCORED_ABLATION_FILES

    assert sorted(mod.ABLATION_RENAMES.values()) == sorted(RESCORED_ABLATION_FILES)
    assert mod.ABLATION_RENAMES["paired_site_deltas_daily.csv"] == (
        "e2_weighting_ablation_daily_site_deltas.csv"
    )
    assert mod.ABLATION_RENAMES["paired_delta_summary.csv"] == (
        "e2_weighting_ablation_paired_deltas.csv"
    )
    final, src = world["final"], world["src0"]
    for name, frozen in mod.ABLATION_RENAMES.items():
        assert (final / frozen).read_bytes() == (src["ablation"] / name).read_bytes()
    sidecar = json.loads((final / mod.ABLATION_METADATA).read_text())
    assert set(sidecar["files"]) == set(mod.ABLATION_RENAMES.values())
    assert sidecar["g_ablation"]["gate"]["daily"]["n_sites"] == 2


def test_ablation_gate_rejects_spread_mismatch(mod, world):
    path = world["src1"]["ablation"] / "paired_site_deltas_monthly.csv"
    path.write_text(path.read_text().replace("\nB,", "\nZ,"))
    gates, _, _ = mod.run_gates(world["src1"], "run1", heavy=False)
    assert "ablation sites != primary cohort" in {g.name: g.message for g in gates}["G-ABLATION"]

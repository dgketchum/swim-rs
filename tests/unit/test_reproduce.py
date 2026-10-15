"""Tests for ``swimrs.swim.reproduce`` (re-rooting recorded paths, tolerant comparison)."""

import json

from swimrs.swim import reproduce as rp

AUTHOR_ROOT = "/data/ssd1/swim"
AUTHOR_REPO = "/home/author/code/swim-rs"


def test_split_and_portable_key():
    assert rp.split(f"{AUTHOR_ROOT}/5_Flux_Ensemble/results/run23/x.csv") == (
        "root",
        rp.Path("5_Flux_Ensemble/results/run23/x.csv"),
    )
    assert rp.split(f"{AUTHOR_REPO}/examples/5_Flux_Ensemble/data/openet_refet/eto.csv") == (
        "repo",
        rp.Path("examples/5_Flux_Ensemble/data/openet_refet/eto.csv"),
    )
    assert rp.split(f"{AUTHOR_REPO}/paper/data/final/x.json") == (
        "repo",
        rp.Path("paper/data/final/x.json"),
    )
    assert rp.split("/elsewhere/file.txt") == (None, rp.Path("/elsewhere/file.txt"))
    assert rp.portable_key(f"{AUTHOR_ROOT}/6_Flux_International/data/x.swim") == (
        "6_Flux_International/data/x.swim"
    )


def test_same_path_ignores_machine_roots():
    a = f"{AUTHOR_ROOT}/5_Flux_Ensemble/data/5_Flux_Ensemble_run23.swim"
    b = "/mnt/other/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run23.swim"
    assert rp.same_path(a, b)
    assert not rp.same_path(a, b.replace("run23", "run22"))
    assert rp.same_path(
        f"{AUTHOR_REPO}/examples/6_Flux_International/x.toml",
        "/tmp/clone/examples/6_Flux_International/x.toml",
    )


def test_rebase_prefers_existing_then_root_then_repo(tmp_path, monkeypatch):
    existing = tmp_path / "present.csv"
    existing.write_text("x")
    assert rp.rebase(existing, root="/nowhere") == existing

    root = tmp_path / "swim"
    repo = tmp_path / "clone"
    recorded = f"{AUTHOR_ROOT}/7_Applied_Water/results/e7cal/archive/x.json"
    assert rp.rebase(recorded, root=root) == root / "7_Applied_Water/results/e7cal/archive/x.json"
    recorded_repo = f"{AUTHOR_REPO}/examples/7_Applied_Water/data/metered_truth.csv"
    assert (
        rp.rebase(recorded_repo, repo=repo)
        == repo / "examples/7_Applied_Water/data/metered_truth.csv"
    )
    monkeypatch.setenv("SWIM_ROOT", str(root))
    assert rp.rebase(recorded) == root / "7_Applied_Water/results/e7cal/archive/x.json"
    monkeypatch.delenv("SWIM_ROOT")
    assert rp.rebase(recorded) == rp.Path(recorded)
    assert rp.rebase("/elsewhere/file.txt", root=root, repo=repo) == rp.Path("/elsewhere/file.txt")


def test_equivalent_csv_allows_round_off_only():
    a = b"site,kge,label\nA,0.29994004755583226,x\nB,0.5,y\n"
    b = b"site,kge,label\nA,0.29994004755583220,x\nB,0.5,y\n"
    c = b"site,kge,label\nA,0.2999,x\nB,0.5,y\n"
    d = b"site,kge,label\nA,0.29994004755583226,x\nB,0.5,z\n"
    assert rp.equivalent(a, a, "x.csv")
    assert rp.equivalent(a, b, "x.csv")
    assert not rp.equivalent(a, c, "x.csv")
    assert not rp.equivalent(a, d, "x.csv")
    assert not rp.equivalent(a, b"site,kge\nA,0.3\n", "x.csv")


def test_equivalent_json_allows_round_off_and_machine_paths():
    a = {
        "n": 47,
        "kge": 0.6999999999999,
        "ok": True,
        "cfg": f"{AUTHOR_REPO}/examples/6_Flux_International/a.toml",
    }
    b = dict(a, kge=0.7, cfg="/tmp/clone/examples/6_Flux_International/a.toml")
    c = dict(a, n=46)
    d = dict(a, ok=False)
    enc = lambda o: json.dumps(o).encode()  # noqa: E731
    assert rp.equivalent(enc(a), enc(b), "g.json")
    assert not rp.equivalent(enc(a), enc(c), "g.json")
    assert not rp.equivalent(enc(a), enc(d), "g.json")
    assert not rp.equivalent(b"x", b"y", "notes.txt")


def test_equivalent_json_ignores_build_stamps_and_path_keys():
    a = {
        "created_utc": "2026-09-26T14:27:54+00:00",
        "git_sha": "0c5b38b",
        "input_hashes": {f"{AUTHOR_ROOT}/6_Flux_International/results/r/x.csv": "abc"},
    }
    b = {
        "created_utc": "2026-10-15T17:17:25+00:00",
        "git_sha": "f6853a6",
        "input_hashes": {"/tmp/elsewhere/6_Flux_International/results/r/x.csv": "abc"},
    }
    c = dict(b, input_hashes={"/tmp/elsewhere/6_Flux_International/results/r/x.csv": "def"})
    enc = lambda o: json.dumps(o).encode()  # noqa: E731
    assert rp.equivalent(enc(a), enc(b), "meta.json")
    assert not rp.equivalent(enc(a), enc(c), "meta.json")

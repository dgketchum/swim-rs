"""Tests for the Example 6 path helpers (``examples/6_Flux_International/ex6_paths.py``)."""

import importlib.util
from pathlib import Path

import pytest

EX6 = Path(__file__).resolve().parents[2] / "examples" / "6_Flux_International"


@pytest.fixture(scope="module")
def ex6_paths():
    spec = importlib.util.spec_from_file_location("ex6_paths", EX6 / "ex6_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_toml_keeps_the_flux_archive_under_root(ex6_paths):
    root = ex6_paths.swim_root()
    assert ex6_paths.flux_root_static() == str(root / "6_Flux_International" / "data" / "flux_qaqc")
    assert ex6_paths.container_static() == (
        root
        / "6_Flux_International"
        / "data"
        / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
    )
    for toml in sorted(EX6.glob("*.toml")):
        try:
            flux_dir = ex6_paths.flux_root_static(toml)
        except KeyError:  # the managed-cap arm has no [validation] section
            assert toml.name.endswith("_ca_managedcap.toml"), toml.name
            continue
        assert flux_dir.startswith(str(root)), toml.name


def test_root_relocates_the_flux_archive_and_container(ex6_paths, tmp_path):
    toml = tmp_path / "relocated.toml"
    toml.write_text(
        'project = "6_Flux_International"\n'
        'root = "/elsewhere/swim"\n'
        "[paths]\n"
        'project_workspace = "{root}/{project}"\n'
        'data = "{project_workspace}/data"\n'
        'container = "{data}/{project}_x.swim"\n'
        "[validation]\n"
        'flux_dir = "{data}/flux_qaqc"\n'
    )
    ws = Path("/elsewhere/swim/6_Flux_International")
    assert ex6_paths.flux_root_static(toml) == str(ws / "data" / "flux_qaqc")
    assert ex6_paths.container_static(toml) == ws / "data" / "6_Flux_International_x.swim"
    assert not ws.exists()


def test_swim_root_env_relocates_without_editing_the_toml(ex6_paths, monkeypatch):
    monkeypatch.setenv("SWIM_ROOT", "/elsewhere/swim")
    assert ex6_paths.swim_root() == Path("/elsewhere/swim")
    assert ex6_paths.flux_root_static() == "/elsewhere/swim/6_Flux_International/data/flux_qaqc"
    assert ex6_paths.container_static().parent == Path("/elsewhere/swim/6_Flux_International/data")

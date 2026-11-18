"""Example 6 forward runs take their vegetation formulation from the config.

A posterior fitted under one formulation (cover-scaled sigmoid, unscaled sigmoid,
unscaled linear) is only scored correctly when the forward run is built with that
formulation. ``evaluate.physics_kwargs`` is the single place the ``[misc]`` settings
become ``build_swim_input`` keywords, and every forward run in the example must use it.
"""

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from swimrs.process.cover_modes import COVER_MODE_NAMES, resolve_cover_mode
from swimrs.process.kcb_modes import KCB_MODE_NAMES, resolve_kcb_mode

EX6 = Path(__file__).resolve().parents[2] / "examples" / "6_Flux_International"
RUN = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
PHYSICS_KEYS = (
    "transpiration_cover_scaling",
    "transpiration_cover_mode",
    "cover_linear_ndvi_bare",
    "cover_linear_ndvi_full",
    "kcb_ndvi_mode",
    "stress_depletion_fraction",
)
# arm TOML suffix -> (NDVI-Kcb curve, cover mode) the forward run must resolve to
ARMS = {
    "": ("sigmoid", "kcb"),
    "_fao56_sig": ("sigmoid", "none"),
    "_fao56": ("linear", "none"),
}


@pytest.fixture(scope="module")
def ev():
    if str(EX6) not in sys.path:
        sys.path.insert(0, str(EX6))
    spec = importlib.util.spec_from_file_location("ex6_evaluate_physics", EX6 / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_cfg(suffix):
    """The physics attributes ProjectConfig would carry for one arm's ``[misc]`` table."""
    misc = tomllib.loads((EX6 / f"{RUN}{suffix}.toml").read_text()).get("misc", {})
    cfg = SimpleNamespace(**{k: misc.get(k) for k in PHYSICS_KEYS})
    cfg.start_dt, cfg.end_dt = "2016-01-01", "2016-12-31"
    return cfg


def _effective(kwargs):
    mode = kwargs["transpiration_cover_mode"]
    scaling = None if mode is not None else kwargs["transpiration_cover_scaling"]
    return (
        KCB_MODE_NAMES[resolve_kcb_mode(kwargs["kcb_ndvi_mode"])],
        COVER_MODE_NAMES[resolve_cover_mode(mode, scaling)],
    )


def test_physics_kwargs_carries_all_six_settings(ev):
    cfg = SimpleNamespace(
        transpiration_cover_scaling=False,
        transpiration_cover_mode="none",
        cover_linear_ndvi_bare=0.15,
        cover_linear_ndvi_full=0.85,
        kcb_ndvi_mode="linear",
        stress_depletion_fraction=0.5,
    )
    assert ev.physics_kwargs(cfg) == vars(cfg)


def test_config_without_physics_keys_runs_the_default_formulation(ev):
    kwargs = ev.physics_kwargs(SimpleNamespace())
    assert set(kwargs) == set(PHYSICS_KEYS)
    assert _effective(kwargs) == ("sigmoid", "kcb")


@pytest.mark.parametrize("suffix", ARMS)
def test_forward_run_is_built_with_the_arm_formulation(ev, monkeypatch, suffix):
    """``run_calibrated_model`` hands each E0 arm's own formulation to the builder."""
    seen = {}

    class Built(Exception):
        pass

    def fake_build(container, **kwargs):
        seen.update(kwargs)
        raise Built

    monkeypatch.setattr(ev, "build_swim_input", fake_build)
    with pytest.raises(Built):
        ev.run_calibrated_model(_arm_cfg(suffix), object(), ["US-Ne1"], {"US-Ne1": {}})
    assert _effective(seen) == ARMS[suffix]


def test_the_three_arms_resolve_to_three_formulations(ev):
    resolved = {_effective(ev.physics_kwargs(_arm_cfg(s))) for s in ARMS}
    assert len(resolved) == len(ARMS)


def _spreads_physics_kwargs(call):
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call):
            fn = kw.value.func
            if getattr(fn, "id", getattr(fn, "attr", None)) == "physics_kwargs":
                return True
    return False


def test_every_forward_run_call_site_uses_physics_kwargs():
    """No Example 6 module builds a forward run without the config's formulation."""
    missing = []
    for path in sorted(EX6.rglob("*.py")):
        if "legacy" in path.parts or "notes" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if getattr(fn, "id", getattr(fn, "attr", None)) != "build_swim_input":
                continue
            if not _spreads_physics_kwargs(node):
                missing.append(f"{path.relative_to(EX6)}:{node.lineno}")
    assert not missing, f"build_swim_input without **physics_kwargs(cfg): {missing}"

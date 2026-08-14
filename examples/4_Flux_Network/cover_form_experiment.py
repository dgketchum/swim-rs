"""E1 transpiration-form experiment: five formulations, one cohort.

Archived replacement for the unreported eight-site fc comparison flagged in
``paper/notes/review_followup.md`` (item 6). Each arm is an independent PEST++
IES calibration of the same E1 cropland sites against the same SSEBop ETf
targets, differing only in the transpiration term:

    kc_act = fc_t * Ks * Kcb + Ke

Two things vary, and they are orthogonal — the NDVI->Kcb curve and the cover
weight fc_t:

    ==============  ==========  ===========================================
    arm             Kcb curve   fc_t
    ==============  ==========  ===========================================
    fao56_sig       sigmoid     1
    linear          sigmoid     ramp in NDVI over the logistic's 10-90% span
    sigmoid         sigmoid     logistic(NDVI) = Kcb / Kc_max
    current         sigmoid     (Kcb - Kc_min)/(Kc_max - Kc_min)   [default]
    fao56_std       linear      1
    ==============  ==========  ===========================================

The first four arms (run 2026-07-31) hold the sigmoid Kcb fixed and vary only
fc_t. That turned out to under-sample the question: because ``fc`` is derived
from ``Kcb`` and then multiplies ``Kcb``, all three weighted arms are quadratic
in the same logistic, and even the ``fao56_sig`` arm keeps the sigmoid Kcb — so
none of them is the standard FAO-56 model found in the literature. See
``notes/cover_form_experiment_results.md``, section "What the arms actually
sampled". (``fao56_sig`` was called ``fao56`` until 2026-08-07 — the old name
read as standard FAO-56, which the arm is not.)

``fao56_std`` is that standard model: a linear NDVI-Kcb relation
(``Kcb = ndvi_beta*NDVI + ndvi_alpha``, the SWIM predecessor's own formulation,
calibrated under its historical priors) with no cover re-weighting.

A sixth arm, ``linear_fc`` (the Eq. 76 cover weight on the linear Kcb),
completed the 2x2 at 4 sites and tied ``current`` there; it was withdrawn from
the reported study on 2026-08-07 as factorial filler — not a FAO-56 baseline,
so irrelevant to the study's claim. Its n=4 archive is retained at
``results/coverform/linear_fc/`` (see ARM_RENAME.txt at the tree root); its
45-site run was stopped during spinup and its partial outputs deleted.

Every arm carries two free NDVI-curve parameters — ``(ndvi_k, ndvi_0)`` under
the sigmoid, ``(ndvi_alpha, ndvi_beta)`` under the linear relation — plus the
identical remaining parameters, priors, bounds, realizations and targets, so no
arm can win by having more freedom. Arms are judged on validation against
flux-tower ET (never a calibration target), not on phi.

Usage:
    python cover_form_experiment.py --container /path/to/4_Flux_Network_julyphysics.swim
    python cover_form_experiment.py --container ... --forward-check   # no calibration
    python cover_form_experiment.py --container ... --arms fao56_sig,fao56_std
    python cover_form_experiment.py --container ... --evaluate-only
"""

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# Experiment arms. The key is the arm/directory name; `mode` is the
# swimrs.process.cover_modes name and `kcb` the swimrs.process.kcb_modes name,
# both written into the arm's TOML.
#
# Arm names are load-bearing: they are the directory names under results/, so a
# key here must match the archived directory name. On 2026-08-07 `fao56` was
# renamed to `fao56_sig` — the old name read as standard FAO-56, which the arm
# is not (it drops the cover weight but keeps the sigmoid Kcb, a hybrid found in
# no literature; `fao56_std` is the actual standard model). Both results trees
# were renamed in lockstep, so key and archive stay consistent; the mapping is
# recorded in ARM_RENAME.txt at each tree root. Archives, logs and commits from
# before 2026-08-07 still carry the old name `fao56`. Renaming a key again means
# renaming its archived directories in the same commit.
#
# `linear_fc` (Eq. 76 cover weight on the linear Kcb) was withdrawn from the
# study on 2026-08-07 and removed from this table; its n=4 archive remains at
# results/coverform/linear_fc/. Re-evaluating that archive requires
# reinstating the key.
ARMS = {
    "fao56_sig": {
        "mode": "none",
        "kcb": "sigmoid",
        "equation": "kc_act = Ks*Kcb + Ke, Kcb = Kc_max*sigmoid(NDVI)",
        "label": "Sigmoid Kcb, no cover weight (not standard FAO-56 — see fao56_std)",
    },
    "linear": {
        "mode": "linear",
        "kcb": "sigmoid",
        "equation": "kc_act = fc_lin*Ks*Kcb + Ke",
        "label": "Sigmoid Kcb, linear NDVI cover ramp (logistic 10-90% span)",
    },
    "sigmoid": {
        "mode": "sigmoid",
        "kcb": "sigmoid",
        "equation": "kc_act = sigmoid(NDVI)*Ks*Kcb + Ke",
        "label": "Sigmoid Kcb, sigmoid cover weight (Kcb/Kc_max)",
    },
    "current": {
        "mode": "kcb",
        "kcb": "sigmoid",
        "equation": "kc_act = fc*Ks*Kcb + Ke, fc = (Kcb-Kc_min)/(Kc_max-Kc_min)",
        "label": "SWIM default (sigmoid Kcb, FAO-56 Eq. 76 cover from Kcb)",
    },
    "fao56_std": {
        "mode": "none",
        "kcb": "linear",
        "equation": "kc_act = Ks*Kcb + Ke, Kcb = ndvi_beta*NDVI + ndvi_alpha",
        "label": "Standard FAO-56 dual crop coefficient (linear Kcb-NDVI, no cover weight)",
    },
}

# Four E1 croplands spanning the cover gradient, all with long paired flux
# records in the julyphysics evaluation cohort:
#   US-Ne1      irrigated maize, closes canopy      (JJA NDVI median 0.82)
#   US-Bo1      rainfed maize/soy, closes canopy    (JJA NDVI median 0.80)
#   US-ARM      rainfed winter wheat + fallow       (JJA NDVI median 0.35)
#   Almond_Med  orchard, persistent partial cover   (JJA NDVI median 0.39)
DEFAULT_SITES = ["US-Ne1", "US-Bo1", "US-ARM", "Almond_Med"]

EXPERIMENT_TAG = "coverform"
NOPTMAX = 3  # matches the hardcoded IES iteration count in calibrate.run_pest_sequence


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def load_arm_config(toml_path):
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(toml_path), calibrate=True)
    else:
        cfg.read_config(
            str(toml_path), project_root_override=str(PROJECT_DIR.parent), calibrate=True
        )
    return cfg


def write_arm_toml(arm, spec, out_dir, pest_run_dir=None):
    """Write the arm's project TOML (base E1 config + the two physics-mode keys).

    This file is what `evaluate.py --config` needs to reproduce the arm, so it
    lives in the arm's results directory and is copied into the archive.

    ``pest_run_dir`` overrides the base config's PEST working directory so an
    experiment never shares (and never deletes) the canonical E1 working tree.
    """
    base = toml.load(PROJECT_DIR / "4_Flux_Network.toml")
    base.setdefault("misc", {})["transpiration_cover_mode"] = spec["mode"]
    base["misc"]["kcb_ndvi_mode"] = spec["kcb"]
    # transpiration_cover_scaling is derived from the mode by ProjectConfig;
    # writing it too would create a second source of truth.
    base["misc"].pop("transpiration_cover_scaling", None)
    if pest_run_dir:
        # calibrate.py rmtree's pest_run_dir at startup, so an experiment left on
        # the shared default would wipe the canonical E1 working tree. Every
        # dependent path interpolates from this one key.
        base.setdefault("calibration", {})["pest_run_dir"] = pest_run_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"4_Flux_Network_{arm}.toml")
    with open(path, "w") as f:
        toml.dump(base, f)
    return path


def write_provenance(arc_dir, arm, spec, container_path, sites, toml_path, cfg):
    """RUN_POLICY Category 1: reproduce the exact environment and command."""
    cat1 = os.path.join(arc_dir, "1_provenance")
    os.makedirs(cat1, exist_ok=True)

    with open(os.path.join(cat1, "command.txt"), "w") as f:
        f.write(" ".join([sys.executable, *sys.argv]) + "\n")
    for name, out in (
        ("git_sha.txt", _git("rev-parse", "HEAD")),
        ("git_status.txt", _git("status", "--short")),
        ("git_diff.patch", _git("diff")),
        ("git_diff_cached.patch", _git("diff", "--cached")),
    ):
        with open(os.path.join(cat1, name), "w") as f:
            f.write(out)

    shutil.copyfile(toml_path, os.path.join(cat1, "config.toml"))
    with open(os.path.join(cat1, "container_path.txt"), "w") as f:
        f.write(f"{container_path}\n")

    freeze = subprocess.run(
        ["uv", "pip", "freeze"], cwd=PROJECT_DIR, capture_output=True, text=True
    ).stdout
    pestpp = subprocess.run(["pestpp-ies", "--version"], capture_output=True, text=True)
    with open(os.path.join(cat1, "environment.json"), "w") as f:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "pestpp_ies_version": (pestpp.stdout or pestpp.stderr).strip(),
                "pip_freeze": freeze.splitlines(),
            },
            f,
            indent=2,
        )

    with open(os.path.join(cat1, "run_metadata.json"), "w") as f:
        json.dump(
            {
                "experiment": "E1 transpiration-form comparison",
                "arm": arm,
                "cover_mode": spec["mode"],
                "kcb_ndvi_mode": spec["kcb"],
                "equation": spec["equation"],
                "label": spec["label"],
                "sites": sites,
                "container": container_path,
                "realizations": cfg.realizations,
                "workers": cfg.workers,
                "noptmax": NOPTMAX,
                "etf_target_model": cfg.etf_target_model,
                "mask_mode": cfg.mask_mode,
                "refet_type": getattr(cfg, "refet_type", "eto"),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )


def write_experiment_spec(results_root, sites, arms, container_path, cfg):
    """RUN_POLICY ablation requirement: state what is held fixed and what varies."""
    os.makedirs(results_root, exist_ok=True)
    spec = {
        "experiment": "E1 transpiration-form comparison",
        "purpose": (
            "Archived, reportable replacement for the unreported eight-site fc "
            "comparison cited in main.md (review_followup item 6). Tests whether "
            "SWIM's Kcb-quadratic transpiration term improves ET skill against "
            "flux towers relative to the standard FAO-56 dual crop coefficient "
            "(linear Kcb-NDVI, no cover re-weighting) and to intermediate forms."
        ),
        "controlled_variables": ["kcb_ndvi_mode", "transpiration_cover_mode"],
        "design": (
            "2x2 factorial on (NDVI->Kcb curve) x (transpiration cover weight), "
            "plus two extra sigmoid-Kcb cover shapes retained from the "
            "2026-07-31 four-arm run."
        ),
        "arms": {a: ARMS[a] for a in arms},
        "held_fixed": [
            "container and all forcing (met, NDVI, ETf, soils)",
            "site cohort",
            "calibration targets (SSEBop ETf, no_mask) and observation weights",
            "number of free parameters (2 NDVI-curve parameters in every arm)",
            "non-curve parameters, prior means, bounds",
            f"realizations={cfg.realizations}, noptmax={NOPTMAX}, IES settings",
            "spinup procedure (re-run per arm under that arm's own physics)",
        ],
        "not_held_fixed": [
            "the identity of the two NDVI-curve parameters and their priors: "
            "(ndvi_k 10.0 [3,20], ndvi_0 0.55 [0.1,0.8]) under the sigmoid vs "
            "(ndvi_alpha 0.2 [-0.7,1.5], ndvi_beta 1.25 [0.5,1.7]) under the "
            "linear relation. Unavoidable — the curves are parameterized "
            "differently. Both prior sets are the historical SWIM values, not "
            "tuned for this experiment."
        ],
        "sites": sites,
        "container": container_path,
        "decision_metric": (
            "Median NSE / RMSE / |MBE| / KGE of SWIM ET vs flux ET_corr, daily and "
            "monthly, strictly paired. Flux ET is never a calibration target, so "
            "this is validation, not fit. Phi is reported but is not a cross-arm "
            "skill measure."
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git("rev-parse", "HEAD").strip(),
    }
    # Arms are added to the experiment in batches (the four sigmoid-Kcb arms on
    # 2026-07-31, the two linear-Kcb arms afterwards), so a later batch must not
    # erase the earlier ones from the spec — the archive would then under-report
    # what the results/ tree actually contains.
    spec_path = os.path.join(results_root, "experiment_spec.json")
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            prior = json.load(f)
        merged = dict(prior.get("arms", {}))
        merged.update(spec["arms"])
        spec["arms"] = merged
        if "created_utc" in prior:
            spec["created_utc"] = prior["created_utc"]
            spec["updated_utc"] = datetime.now(UTC).isoformat()
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    return spec


def forward_check(container_path, sites, arms=None):
    """Run each form forward on prior parameters and report ET.

    A cheap wiring check: if two arms produce identical ET the mode is not
    reaching the loop, and the whole experiment would be null by construction.
    """
    arms = list(arms) if arms else list(ARMS)
    import tempfile

    from swimrs.container import SwimContainer
    from swimrs.process.input import build_swim_input
    from swimrs.process.loop_fast import run_daily_loop_fast

    container = SwimContainer.open(container_path, mode="r")
    cfg = load_arm_config(PROJECT_DIR / "4_Flux_Network.toml")

    rows = []
    for arm in arms:
        spec = ARMS[arm]
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            h5 = tmp.name
        try:
            swim_input = build_swim_input(
                container,
                output_h5=h5,
                start_date=cfg.start_dt,
                end_date=cfg.end_dt,
                refet_type=(getattr(cfg, "refet_type", "eto") or "eto").lower(),
                fields=sites,
                empirical_kc_max=True,
                mask_mode=getattr(cfg, "mask_mode", "irrigation"),
                transpiration_cover_mode=spec["mode"],
                kcb_ndvi_mode=spec["kcb"],
                use_container_calibration=False,
            )
            output, _ = run_daily_loop_fast(swim_input)
            years = swim_input.n_days / 365.25
            for i, fid in enumerate(swim_input.fids):
                rows.append(
                    {
                        "arm": arm,
                        "mode": spec["mode"],
                        "kcb": spec["kcb"],
                        "site": fid,
                        "annual_et_mm": float(np.nansum(output.eta[:, i]) / years),
                    }
                )
            swim_input.close()
        finally:
            if os.path.exists(h5):
                os.remove(h5)

    df = pd.DataFrame(rows)
    print("\nForward check — annual ET (mm) at prior parameters:")
    print(df.pivot(index="site", columns="arm", values="annual_et_mm").round(1).to_string())
    # Note the sigmoid and linear Kcb arms run at *different* priors (the curves
    # are parameterized differently), so this is a wiring check only — the
    # magnitudes are not an ET comparison. Calibration is what makes the arms
    # comparable.
    n_unique = df.groupby("site")["annual_et_mm"].nunique()
    if (n_unique < len(arms)).any():
        raise SystemExit(
            "Forms are NOT all distinct in the forward model — fix the "
            "plumbing before calibrating:\n" + n_unique.to_string()
        )
    print(f"\nAll {len(arms)} forms are distinct in the forward model.")
    return df


def final_phi(results_dir, project):
    """Mean measurement phi of the final IES iteration, for reporting only."""
    path = os.path.join(results_dir, f"{project}.phi.meas.csv")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    return float(df.iloc[-1]["mean"]) if "mean" in df.columns else np.nan


def run_arm(arm, spec, container_path, sites, results_root, workers=None, realizations=None):
    from calibrate import run_pest_sequence

    results_dir = os.path.join(results_root, arm)
    os.makedirs(results_dir, exist_ok=True)

    # One isolated PEST tree per experiment tag. Arms run sequentially, so they
    # can share it; what they must NOT share is the canonical E1 "pestrun".
    # Must stay a {project_workspace} TEMPLATE, not an absolute path: config
    # _resolve_paths only registers a key as an interpolation variable when its
    # value contains braces, so an absolute pest_run_dir would silently leave
    # calibration_dir/obs_folder/spinup as literal "{pest_run_dir}/..." strings.
    tag = os.path.basename(os.path.normpath(results_root))
    toml_path = write_arm_toml(
        arm, spec, results_dir, pest_run_dir="{project_workspace}/pestrun_" + tag
    )
    cfg = load_arm_config(toml_path)
    if workers:
        cfg.workers = workers
    if realizations:
        cfg.realizations = realizations

    if cfg.transpiration_cover_mode != spec["mode"]:
        raise RuntimeError(
            f"arm {arm}: config resolved cover mode {cfg.transpiration_cover_mode!r}, "
            f"expected {spec['mode']!r}"
        )
    if cfg.kcb_ndvi_mode != spec["kcb"]:
        raise RuntimeError(
            f"arm {arm}: config resolved kcb mode {cfg.kcb_ndvi_mode!r}, expected {spec['kcb']!r}"
        )
    # An unresolved "{...}" here means PEST would build its tree at a literal
    # brace path and the arm would fail hours in. Fail now instead.
    for key in ("pest_run_dir", "calibration_dir", "obs_folder", "initial_values_csv", "spinup"):
        val = getattr(cfg, key, None)
        if isinstance(val, str) and ("{" in val or "}" in val):
            raise RuntimeError(f"arm {arm}: {key} did not interpolate: {val!r}")

    print(f"\n{'=' * 80}")
    print(f"ARM {arm}: {spec['label']}")
    print(f"  {spec['equation']}")
    print(f"  Kcb curve = {cfg.kcb_ndvi_mode}, cover mode = {cfg.transpiration_cover_mode}")
    print(f"  sites = {sites}")
    print(f"  reals = {cfg.realizations}, workers = {cfg.workers}")
    print(f"  results = {results_dir}")
    print(f"{'=' * 80}\n")

    write_provenance(
        os.path.join(results_dir, "archive"), arm, spec, container_path, sites, toml_path, cfg
    )

    t0 = time.time()
    # select_fields (not debug_fields) so the arm keeps the canonical
    # realization count instead of dropping to the 20-realization debug setting.
    run_pest_sequence(
        cfg,
        results_dir,
        pdc_remove=False,
        select_fields=sites,
        container_path=container_path,
        keep_pestrun=False,
    )
    elapsed = time.time() - t0
    with open(os.path.join(results_dir, "runtime.json"), "w") as f:
        json.dump({"arm": arm, "wall_minutes": round(elapsed / 60, 1)}, f, indent=2)
    print(f"ARM {arm} calibration done in {elapsed / 60:.1f} min")
    return results_dir, cfg


def archive_posterior(arm, cfg, results_dir):
    """RUN_POLICY Category 5: posterior parameter medians and boundary-hit rates."""
    from archive_posterior_summary import build_posterior_summary
    from evaluate import find_par_csv

    arc = os.path.join(results_dir, "archive")
    par_csv = find_par_csv(results_dir, cfg.project_name)
    # PstFrom lowercases its sidecar filenames, so glob rather than rebuild the
    # name from cfg.project_name (which is mixed case).
    matches = glob.glob(os.path.join(arc, "3_problem_definition", "*.par_data.csv"))
    par_data = matches[0] if matches else None
    if par_csv is None or par_data is None:
        print(f"  Cat 5 skipped for {arm}: par_csv={par_csv}, par_data={par_data}")
        return
    try:
        build_posterior_summary(par_csv, par_data, cfg.fields_shapefile, arc, run_name=arm)
    except Exception as exc:  # noqa: BLE001 - summary is reporting, not physics
        print(f"  Cat 5 posterior summary failed for {arm}: {exc}")


def evaluate_arm(arm, cfg, container_path, sites, results_dir):
    from evaluate import evaluate, evaluate_monthly, find_par_csv

    from swimrs.container import SwimContainer

    par_csv = find_par_csv(results_dir, cfg.project_name)
    if par_csv is None:
        raise FileNotFoundError(f"arm {arm}: no par.csv in {results_dir}")

    container = SwimContainer.open(container_path, mode="r")
    cat6 = os.path.join(results_dir, "archive", "6_evaluation")
    os.makedirs(cat6, exist_ok=True)

    print(f"\n--- arm {arm}: daily evaluation ---")
    daily = evaluate(cfg, container, par_csv, list(sites), cfg.flux_dir, out_dir=cat6)
    daily.to_csv(os.path.join(cat6, "daily_paired_metrics.csv"))

    print(f"\n--- arm {arm}: monthly evaluation ---")
    monthly = evaluate_monthly(cfg, container, par_csv, list(sites), cfg.flux_dir, out_dir=cat6)
    monthly.to_csv(os.path.join(cat6, "monthly_paired_metrics.csv"))

    with open(os.path.join(cat6, "evaluation_metadata.json"), "w") as f:
        json.dump(
            {
                "arm": arm,
                "cover_mode": cfg.transpiration_cover_mode,
                "kcb_ndvi_mode": cfg.kcb_ndvi_mode,
                "par_csv": par_csv,
                "container": container_path,
                "flux_dir": cfg.flux_dir,
                "sites": sites,
                "evaluated_utc": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )
    return daily, monthly


def _med(df, col):
    return df[col].median() if (not df.empty and col in df.columns) else np.nan


def compare(results_root, project, arm_results):
    """Cross-arm comparison tables: per-site rows and per-arm medians."""
    per_site, summary = [], []
    for arm, (daily, monthly, results_dir) in arm_results.items():
        spec = ARMS[arm]
        for scale, df, ncol in (("daily", daily, "n"), ("monthly", monthly, "n_months")):
            if df.empty:
                print(f"WARNING: arm {arm} produced no {scale} metrics")
            for fid, row in df.iterrows():
                per_site.append(
                    {
                        "arm": arm,
                        "cover_mode": spec["mode"],
                        "kcb_mode": spec["kcb"],
                        "site": fid,
                        "scale": scale,
                        "n": row.get(ncol),
                        "nse": row.get("r2_swim"),
                        "r": row.get("r_swim"),
                        "rmse": row.get("rmse_swim"),
                        "mbe": row.get("bias_swim"),
                        "kge": row.get("kge_swim"),
                        "nse_ssebop": row.get("r2_ssebop"),
                        "rmse_ssebop": row.get("rmse_ssebop"),
                    }
                )
            summary.append(
                {
                    "arm": arm,
                    "cover_mode": spec["mode"],
                    "kcb_mode": spec["kcb"],
                    "equation": spec["equation"],
                    "scale": scale,
                    "n_sites": int(df["r2_swim"].notna().sum()) if not df.empty else 0,
                    "nse_med": _med(df, "r2_swim"),
                    "r_med": _med(df, "r_swim"),
                    "rmse_med": _med(df, "rmse_swim"),
                    "mbe_med": _med(df, "bias_swim"),
                    "kge_med": _med(df, "kge_swim"),
                    "phi_final": final_phi(results_dir, project),
                }
            )

    per_site_df = pd.DataFrame(per_site)
    summary_df = pd.DataFrame(summary)
    per_site_df.to_csv(os.path.join(results_root, "cover_form_per_site.csv"), index=False)
    summary_df.to_csv(os.path.join(results_root, "cover_form_summary.csv"), index=False)

    cols = [
        "arm",
        "kcb_mode",
        "cover_mode",
        "n_sites",
        "nse_med",
        "rmse_med",
        "mbe_med",
        "kge_med",
        "phi_final",
    ]
    print("\n" + "=" * 96)
    print(
        "TRANSPIRATION-FORM COMPARISON — SWIM vs flux ET_corr (flux is never a calibration target)"
    )
    print("=" * 96)
    for scale in ("daily", "monthly"):
        sub = summary_df[summary_df["scale"] == scale]
        if sub.empty:
            continue
        print(f"\n{scale.upper()} medians across sites")
        print(sub[cols].round(3).to_string(index=False))

    if not per_site_df.empty:
        daily_ps = per_site_df[per_site_df["scale"] == "daily"]
        print("\nPer-site daily NSE by arm")
        print(daily_ps.pivot(index="site", columns="arm", values="nse").round(3).to_string())

    print(f"\nWrote {os.path.join(results_root, 'cover_form_summary.csv')}")
    return per_site_df, summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container", required=True, help="Container path (no default: never guess the base)"
    )
    parser.add_argument("--sites", default=None, help="Comma-separated site IDs")
    parser.add_argument("--arms", default=None, help=f"Subset of {list(ARMS)}")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument(
        "--results-tag",
        default=EXPERIMENT_TAG,
        help="Subdirectory under results/ holding all arms",
    )
    parser.add_argument(
        "--forward-check",
        action="store_true",
        help="Only verify the selected forms differ in the forward model, then exit",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip calibration; re-evaluate and re-compare existing arm results",
    )
    args = parser.parse_args()

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else list(DEFAULT_SITES)
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"Unknown arm(s) {unknown}; choose from {list(ARMS)}")
    if not os.path.exists(args.container):
        raise SystemExit(f"Container not found: {args.container}")

    if args.forward_check:
        forward_check(args.container, sites, arms)
        return

    base_cfg = load_arm_config(PROJECT_DIR / "4_Flux_Network.toml")
    if args.realizations:
        base_cfg.realizations = args.realizations
    results_root = os.path.join(base_cfg.project_ws, "results", args.results_tag)
    os.makedirs(results_root, exist_ok=True)
    write_experiment_spec(results_root, sites, arms, args.container, base_cfg)

    # Cat 2 input audit is a property of the container, identical across arms,
    # so it is captured once at the experiment root.
    if not args.evaluate_only:
        from archive_input_audit import capture_input_audit

        audit_dir = Path(results_root) / "archive" / "2_input_audit"
        try:
            gate = capture_input_audit(audit_dir, args.container)
            print(f"Cat 2 input audit gate = {gate}")
        except Exception as exc:  # noqa: BLE001 - audit is reporting, not physics
            print(f"Cat 2 input audit failed: {exc}")

    arm_results = {}
    t_start = time.time()
    for arm in arms:
        spec = ARMS[arm]
        if args.evaluate_only:
            results_dir = os.path.join(results_root, arm)
            cfg = load_arm_config(os.path.join(results_dir, f"4_Flux_Network_{arm}.toml"))
        else:
            results_dir, cfg = run_arm(
                arm,
                spec,
                args.container,
                sites,
                results_root,
                workers=args.workers,
                realizations=args.realizations,
            )
            archive_posterior(arm, cfg, results_dir)
        daily, monthly = evaluate_arm(arm, cfg, args.container, sites, results_dir)
        arm_results[arm] = (daily, monthly, results_dir)

    compare(results_root, base_cfg.project_name, arm_results)
    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")
    print(f"Results: {results_root}")


if __name__ == "__main__":
    main()

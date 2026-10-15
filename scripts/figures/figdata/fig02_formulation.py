"""Figure 2: cover scaling and the E0 vegetation-formulation trio."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.swim import reproduce

from . import paths
from .cohorts import EXPECTED
from .manifest import (
    MANIFEST,
    BuildError,
    require_columns,
    require_count,
    require_unique,
    sha256,
    write_table,
)
from .paths import _CFG6, REPO, SCRIPT_VERSION, ex6_paths

# --------------------------------------------------------------------------
# Figure 2 -- cover scaling makes the vegetation formulation coherent (E0)
# --------------------------------------------------------------------------

# E0 vegetation-formulation trio: three formulations calibrated with the
# identical E2 inputs (ERA5-Land meteorology, HWSD soils, Landsat ETf-ensemble
# target), IES budget, and prior families, and evaluated against withheld
# flux-tower ET at the 37 E2 cropland sites that are not in E1
# (examples/6_Flux_International/e0_disjoint; rationale and checked results in
# examples/6_Flux_International/notes/e0_disjoint_confirmation.md).  Only the
# vegetation formulation (and, for the linear arm, its formulation-specific
# slope/intercept priors) differs between arms.  Internal run labels are
# provenance only and never appear in a reader-facing field.
E0_RESULTS = ex6_paths.results_root(_CFG6)
E0_DATA = Path(_CFG6.data_dir)
E0_EXAMPLE = "examples/6_Flux_International"
E0_SITES_FILE = REPO / E0_EXAMPLE / "e0_disjoint" / "disjoint_sites.txt"
E0_MERGED_POSTERIOR = Path("archive/4_pest_outputs/merged/merged_posterior.csv")

E0_ARMS = {
    "cover_scaled_sigmoid": {
        "display_label": "Cover-scaled sigmoid",
        "run_label": "grassbasis",
        "par_csv": E0_RESULTS
        / "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
        / E0_MERGED_POSTERIOR,
        "container": E0_DATA / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim",
        "kcb_ndvi_mode": "sigmoid",
        "transpiration_cover_mode": "kcb",
        "veg_params": ("ndvi_k", "ndvi_0"),
        "config": f"{E0_EXAMPLE}/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml",
    },
    "unscaled_linear": {
        "display_label": "Unscaled linear",
        "run_label": "fao56",
        "par_csv": E0_RESULTS
        / "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56"
        / E0_MERGED_POSTERIOR,
        "container": E0_DATA
        / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr_fao56.swim",
        "kcb_ndvi_mode": "linear",
        "transpiration_cover_mode": "none",
        "veg_params": ("ndvi_alpha", "ndvi_beta"),
        "config": (
            f"{E0_EXAMPLE}/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56.toml"
        ),
    },
    "unscaled_sigmoid": {
        "display_label": "Unscaled sigmoid",
        "run_label": "fao56_sig",
        "par_csv": E0_RESULTS
        / "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56_sig"
        / E0_MERGED_POSTERIOR,
        "container": E0_DATA
        / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr_fao56_sig.swim",
        "kcb_ndvi_mode": "sigmoid",
        "transpiration_cover_mode": "none",
        "veg_params": ("ndvi_k", "ndvi_0"),
        "config": (
            f"{E0_EXAMPLE}/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56_sig.toml"
        ),
    },
}

# Panel (a) NDVI source and the E1-container check share the cover-scaled
# arm's container; the per-arm calibrated containers are what
# pooled_arm_compare ran each forecast-mode arm from.  Zarr directories cannot
# be single-file hashed; their content is pinned by the Table 3 reproduction
# assert in build_fig02_bootstrap.
E0_CONTAINER = E0_ARMS["cover_scaled_sigmoid"]["container"]
E0_ARM_CONTAINERS = {key: arm["container"] for key, arm in E0_ARMS.items()}

# Paired-arm comparison archives (examples/6_Flux_International/
# pooled_arm_compare.py output, --sites-file e0_disjoint/disjoint_sites.txt).
# The cover-scaled arm is arm A in the two reader-facing contrasts; the
# shape-only pair closes the triangle for cross-file consistency checks and
# never reaches a display table on its own.
E0_COMPARISON_ROOT = E0_RESULTS / "e0_disjoint"
E0_COMPARISONS = {
    "isolated_cover": {
        "dir": E0_COMPARISON_ROOT / "grassbasis_vs_fao56_sig" / "disjoint37",
        "arm_a": "cover_scaled_sigmoid",
        "arm_b": "unscaled_sigmoid",
    },
    "whole_formulation": {
        "dir": E0_COMPARISON_ROOT / "grassbasis_vs_fao56" / "disjoint37",
        "arm_a": "cover_scaled_sigmoid",
        "arm_b": "unscaled_linear",
    },
    "shape_only": {
        "dir": E0_COMPARISON_ROOT / "fao56_sig_vs_fao56" / "disjoint37",
        "arm_a": "unscaled_sigmoid",
        "arm_b": "unscaled_linear",
    },
}

# Fixed coefficients shared by every E0 arm.  examples/6_Flux_International/
# evaluate.py::run_calibrated_model requests the container's empirical kc_max,
# which src/swimrs/process/input.py::_resolve_kc_max floors at 1.35; the arm
# containers store derived/dynamics/kc_max = 1.35 at every E0 site (asserted in
# build_fig02), so the fixed 1.35 governs every arm.  kc_min is fixed at 0.15
# (input.py) and the clip semantics follow src/swimrs/process/loop_fast.py /
# cover_modes.py.
E0_KC_MAX = 1.35
E0_KC_MIN = 0.15
E0_FC_MAX = 0.99
E0_SIGMOID_EXP_CLIP = 20.0

# Non-vegetation parameters carried identically (same priors) by every arm.
E0_SHARED_PARAMS = ("aw", "ks_alpha", "kr_alpha", "mad", "swe_alpha", "swe_beta")

E0_NDVI_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
E0_SUPPORT_BINS = np.round(np.linspace(0.0, 1.0, 51), 2)
E0_MASK_ID = "e0_arm_paired_flux_mask_37sites_49289d_1514mo"
E0_EXPERIMENT_LABEL = (
    "E0 (vegetation-formulation comparison at the 37 E2 sites not in E1; "
    "examples/6_Flux_International/e0_disjoint)"
)

E0_SUPPORT_RULE = (
    "site-equal weighting: every finite merged-NDVI observation carries weight 1/n_site so "
    "each of the 37 E0 sites contributes equal total weight; density integrates to 1 over NDVI"
)

# The previous Figure 2 data package, kept as history outside the active
# manifest (copied there on 2026-09-20 before this build replaced it).
E0_PRIOR_PACKAGE_DIR = "superseded_fig02_2026-09-01"

# Table 3 (paper/text/main.md) at manuscript precision: daily values and KGE
# to three decimals, monthly RMSE/MBE to one.  Signs are plain ASCII here;
# main.md prints a typographic minus and an explicit plus.  The build fails
# when a frozen pooled value does not reproduce its manuscript string exactly.
# Values are the 2026-09-21 HWSD-AWC-units recalibration (frozen copies in
# paper/data/final/e2_closure_pool/e0_disjoint/); the cover-scaled arm wins
# 6/6 pooled metrics against the unscaled sigmoid and 5/6 against the unscaled
# linear arm, losing pooled monthly KGE 0.734 vs 0.737.
E0_TABLE3 = {
    ("cover_scaled_sigmoid", "daily", "rmse"): "1.144",
    ("cover_scaled_sigmoid", "daily", "mbe"): "-0.092",
    ("cover_scaled_sigmoid", "daily", "kge"): "0.697",
    ("cover_scaled_sigmoid", "monthly", "rmse"): "24.2",
    ("cover_scaled_sigmoid", "monthly", "mbe"): "-3.7",
    ("cover_scaled_sigmoid", "monthly", "kge"): "0.734",
    ("unscaled_linear", "daily", "rmse"): "1.407",
    ("unscaled_linear", "daily", "mbe"): "0.221",
    ("unscaled_linear", "daily", "kge"): "0.677",
    ("unscaled_linear", "monthly", "rmse"): "32.8",
    ("unscaled_linear", "monthly", "mbe"): "5.5",
    ("unscaled_linear", "monthly", "kge"): "0.737",
    ("unscaled_sigmoid", "daily", "rmse"): "1.456",
    ("unscaled_sigmoid", "daily", "mbe"): "-0.268",
    ("unscaled_sigmoid", "daily", "kge"): "0.591",
    ("unscaled_sigmoid", "monthly", "rmse"): "34.9",
    ("unscaled_sigmoid", "monthly", "mbe"): "-9.2",
    ("unscaled_sigmoid", "monthly", "kge"): "0.606",
}

# calc_metrics (examples/6_Flux_International/evaluate.py) returns NaN below
# ten paired values, so per-site monthly metrics exist only for sites with at
# least ten qualifying months even though pooled monthly sums admit sites from
# six months up.
E0_SITE_METRIC_MIN_MONTHS = 10
E0_POOLED_MIN_MONTHS = 6
E0_MONTH_MIN_PAIRED_DAYS = 20


def _e0_sites() -> list[str]:
    """The E0 site list in file order (the order pooled_arm_compare pooled them)."""
    sites = [ln.strip() for ln in E0_SITES_FILE.read_text().splitlines() if ln.strip()]
    require_count(len(sites), EXPECTED["E0_configured"], "fig02 E0 site list")
    if len(set(sites)) != len(sites):
        raise BuildError("fig02: duplicate site in the E0 site list")
    return sites


def _e0_table3_format(scale: str, metric: str, value: float) -> str:
    """Format a pooled value exactly as Table 3 prints it (sign retained)."""
    nd = 3 if (scale == "daily" or metric == "kge") else 1
    s = f"{value:.{nd}f}"
    if s == "-" + f"{0.0:.{nd}f}":
        s = s[1:]
    return s


def _e0_par_medians(arm_key: str, container_uids: list[str], sites: list[str]):
    """Posterior-median parameter vector per E0 site for one arm.

    Mirrors examples/6_Flux_International/evaluate.py::parse_pest_params: the
    median is taken across IES realizations excluding the ``base`` row, and
    parameter columns are matched to sites by lowercase suffix.  Every column
    must match one of the container's fields; the result is then restricted
    to ``sites``.  Returns ``({site: {param: median}}, n_realizations)``.
    """
    arm = E0_ARMS[arm_key]
    df = pd.read_csv(arm["par_csv"], index_col=0)
    reals = df.loc[df.index.astype(str) != "base"]
    n_real = len(reals)
    if n_real < 50:
        raise BuildError(f"fig02 {arm_key}: only {n_real} IES realizations in {arm['par_csv']}")
    med = reals.median()
    by_len = sorted(container_uids, key=len, reverse=True)
    by_site: dict[str, dict[str, float]] = {u: {} for u in container_uids}
    for col in df.columns:
        core = col.split("_ptype:")[0].replace("pname:p_", "").rsplit("_:0", 1)[0]
        site = next((u for u in by_len if core.lower().endswith("_" + u.lower())), None)
        if site is None:
            raise BuildError(f"fig02 {arm_key}: column {col!r} matches no container site")
        pname = core[: -(len(site) + 1)]
        if pname in by_site[site]:
            raise BuildError(f"fig02 {arm_key}: duplicate parameter {pname} for {site}")
        by_site[site][pname] = float(med[col])
    expected = set(E0_SHARED_PARAMS) | set(arm["veg_params"])
    for u in sites:
        if set(by_site[u]) != expected:
            raise BuildError(
                f"fig02 {arm_key}: site {u} carries {sorted(by_site[u])}, "
                f"expected {sorted(expected)}"
            )
    return {u: by_site[u] for u in sites}, n_real


def _e0_kt(arm_key: str, ndvi: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Potential canopy-transpiration multiplier K_T on an NDVI grid.

    Replicates the Kcb and fc arithmetic of src/swimrs/process/loop_fast.py
    (sigmoid exponent clip +-20, linear Kcb clip [0, Kc_max], fc clip
    [0, 0.99]) with Ks = 1 and the soil-evaporation term omitted.
    """
    arm = E0_ARMS[arm_key]
    if arm["kcb_ndvi_mode"] == "linear":
        kcb = p["ndvi_beta"] * ndvi + p["ndvi_alpha"]
    else:
        e = np.clip(-p["ndvi_k"] * (ndvi - p["ndvi_0"]), -E0_SIGMOID_EXP_CLIP, E0_SIGMOID_EXP_CLIP)
        kcb = E0_KC_MAX / (1.0 + np.exp(e))
    kcb = np.clip(kcb, 0.0, E0_KC_MAX)
    if arm["transpiration_cover_mode"] == "kcb":
        fc = np.clip((kcb - E0_KC_MIN) / (E0_KC_MAX - E0_KC_MIN), 0.0, E0_FC_MAX)
        return fc * kcb
    if arm["transpiration_cover_mode"] != "none":
        raise BuildError(f"fig02 {arm_key}: unhandled cover mode")
    return kcb


def build_fig02() -> None:
    import zarr

    srcs: dict[str, Path] = {"container": E0_CONTAINER, "sites": E0_SITES_FILE}
    for key, arm in E0_ARMS.items():
        srcs[f"par:{key}"] = arm["par_csv"]
        srcs[f"config:{key}"] = REPO / arm["config"]
        srcs[f"container:{key}"] = arm["container"]
    for key, comp in E0_COMPARISONS.items():
        srcs[f"gate:{key}"] = comp["dir"] / "pooled_gate.json"
        srcs[f"per_site:{key}"] = comp["dir"] / "pooled_per_site.csv"
        srcs[f"excluded:{key}"] = comp["dir"] / "sites_excluded.csv"
    for k, p in srcs.items():
        if not p.exists():
            raise BuildError(f"fig02 source missing: {k} -> {p}")

    # ---- E0 sites and their observed-NDVI distribution (panel a underlay) ----
    sites = _e0_sites()
    z = zarr.open(str(E0_CONTAINER), mode="r")
    uids = [str(x) for x in z["geometry/uid"][:]]
    require_count(len(uids), EXPECTED["E0_container_fields"], "fig02 E0 container fields")
    missing = sorted(set(sites) - set(uids))
    if missing:
        raise BuildError(f"fig02: E0 sites absent from the container: {missing}")
    cols = [uids.index(s) for s in sites]
    kc_max_store = np.asarray(z["derived/dynamics/kc_max"])[cols]
    if not np.all(kc_max_store == E0_KC_MAX):
        raise BuildError(
            f"fig02: container kc_max at an E0 site is not {E0_KC_MAX}: "
            f"{sorted(set(kc_max_store.tolist()))}"
        )
    ndvi_all = np.asarray(z["derived/merged_ndvi/no_mask"])
    if ndvi_all.ndim != 2 or ndvi_all.shape[1] != len(uids):
        raise BuildError(f"fig02 NDVI: merged NDVI shape {ndvi_all.shape} != (time, {len(uids)})")
    ndvi = np.ascontiguousarray(ndvi_all[:, cols])
    ndvi_hash = hashlib.sha256(ndvi.tobytes() + "|".join(sites).encode()).hexdigest()
    finite = np.isfinite(ndvi)
    per_site_obs = finite.sum(axis=0)
    if int(per_site_obs.min()) < 100:
        raise BuildError("fig02 NDVI: a site has fewer than 100 NDVI observations")
    lo, hi = float(np.nanmin(ndvi)), float(np.nanmax(ndvi))
    if lo < 0.0 or hi > 1.0:
        raise BuildError(f"fig02 NDVI: merged NDVI outside [0, 1] ({lo:.3f}, {hi:.3f})")

    width = float(E0_SUPPORT_BINS[1] - E0_SUPPORT_BINS[0])
    dens = np.zeros(len(E0_SUPPORT_BINS) - 1)
    n_obs = np.zeros(len(E0_SUPPORT_BINS) - 1, dtype=int)
    n_sites_bin = np.zeros(len(E0_SUPPORT_BINS) - 1, dtype=int)
    for j in range(len(sites)):
        h, _ = np.histogram(ndvi[finite[:, j], j], bins=E0_SUPPORT_BINS)
        dens += h / float(per_site_obs[j])
        n_obs += h
        n_sites_bin += (h > 0).astype(int)
    dens /= len(sites) * width
    if abs(float(dens.sum()) * width - 1.0) > 1e-9:
        raise BuildError("fig02 NDVI: site-equal density does not integrate to 1")
    if int(n_obs.sum()) != int(finite.sum()):
        raise BuildError("fig02 NDVI: binned observation count mismatch")
    support = pd.DataFrame(
        {
            "bin_left": E0_SUPPORT_BINS[:-1],
            "bin_right": E0_SUPPORT_BINS[1:],
            "density_site_equal": dens,
            "n_sites": n_sites_bin,
            "n_obs": n_obs,
            "support_rule": E0_SUPPORT_RULE,
            "source_sha256": ndvi_hash,
        }
    )

    # ---- fitted vegetation response distributions (panel a) ----
    par_hashes: dict[str, str] = {}
    veg_medians: dict[str, dict[str, dict[str, float]]] = {}
    n_reals: dict[str, int] = {}
    resp_frames = []
    for key, arm in E0_ARMS.items():
        by_site, n_real = _e0_par_medians(key, uids, sites)
        par_hashes[key] = sha256(arm["par_csv"])
        n_reals[key] = n_real
        veg_medians[key] = {u: {p: by_site[u][p] for p in arm["veg_params"]} for u in sites}
        src_label = f"posterior median over {n_real} IES realizations of {arm['par_csv']}"
        for u in sites:
            kt = _e0_kt(key, E0_NDVI_GRID, by_site[u])
            if not np.isfinite(kt).all() or kt.min() < 0.0 or kt.max() > E0_KC_MAX:
                raise BuildError(f"fig02 response: K_T out of [0, kc_max] for {key}/{u}")
            resp_frames.append(
                pd.DataFrame(
                    {
                        "formulation": key,
                        "site_id": u,
                        "ndvi": E0_NDVI_GRID,
                        "k_t": kt,
                        "parameter_source": src_label,
                        "source_sha256": par_hashes[key],
                    }
                )
            )
    resp = pd.concat(resp_frames, ignore_index=True)
    require_count(len(resp), 3 * len(sites) * len(E0_NDVI_GRID), "fig02 response rows")
    require_unique(resp, ["formulation", "site_id", "ndvi"], "fig02 response")

    # ---- pooled flux-ET agreement on the Table 3 mask (panel b) ----
    gates = {
        key: json.loads((comp["dir"] / "pooled_gate.json").read_text())
        for key, comp in E0_COMPARISONS.items()
    }
    for key, g in gates.items():
        comp = E0_COMPARISONS[key]
        for side in ("a", "b"):
            form = comp[f"arm_{side}"]
            arm = E0_ARMS[form]
            if g[f"arm_{side}"] != arm["run_label"]:
                raise BuildError(f"fig02 gate {key}: arm_{side} is not {arm['run_label']}")
            want_phys = {
                "kcb_ndvi_mode": arm["kcb_ndvi_mode"],
                "transpiration_cover_mode": arm["transpiration_cover_mode"],
            }
            if g[f"{side}_physics"] != want_phys:
                raise BuildError(f"fig02 gate {key}: {side}_physics != stated formulation")
            # recorded on the authoring machine; compare below the data root / clone
            if not reproduce.same_path(g[f"{side}_params"], arm["par_csv"]):
                raise BuildError(f"fig02 gate {key}: {side}_params != panel-(a) parameter source")
            if not reproduce.same_path(g[f"{side}_config"], REPO / arm["config"]):
                raise BuildError(f"fig02 gate {key}: {side}_config != stated arm config")
            if not reproduce.same_path(g[f"{side}_container"], arm["container"]):
                raise BuildError(f"fig02 gate {key}: {side}_container != stated arm container")
        if list(g["sites_requested"]) != sites:
            raise BuildError(f"fig02 gate {key}: sites_requested != the E0 site list")
        excluded = pd.read_csv(comp["dir"] / "sites_excluded.csv")
        if len(excluded):
            raise BuildError(f"fig02 gate {key}: {len(excluded)} E0 sites were excluded")
        require_count(g["n_sites"], EXPECTED["E0_pooled_sites"], f"fig02 gate {key} sites")
        require_count(g["n_daily"], EXPECTED["E0_pooled_daily"], f"fig02 gate {key} site-days")
        require_count(g["n_monthly"], EXPECTED["E0_pooled_monthly"], f"fig02 gate {key} months")

    pooled_vals: dict[tuple[str, str, str], list[float]] = {}
    for key, g in gates.items():
        comp = E0_COMPARISONS[key]
        for m in g["metrics"]:
            n_want = EXPECTED["E0_pooled_daily" if m["scale"] == "daily" else "E0_pooled_monthly"]
            require_count(m["n"], n_want, f"fig02 gate {key} {m['scale']} {m['metric']} n")
            for side in ("a", "b"):
                form = comp[f"arm_{side}"]
                run = E0_ARMS[form]["run_label"]
                pooled_vals.setdefault((form, m["scale"], m["metric"].lower()), []).append(
                    float(m[run])
                )

    gates_hash = hashlib.sha256(
        "".join(
            sha256(E0_COMPARISONS[k]["dir"] / "pooled_gate.json") for k in sorted(gates)
        ).encode()
    ).hexdigest()
    pooled_rows = []
    table3_record = {}
    for form in E0_ARMS:
        for scale in ("daily", "monthly"):
            for metric in ("kge", "rmse", "mbe"):
                vals = pooled_vals.pop((form, scale, metric))
                if len(vals) != 2 or vals[0] != vals[1]:
                    raise BuildError(
                        f"fig02 pooled: {form}/{scale}/{metric} inconsistent across "
                        f"comparison files: {vals}"
                    )
                v = vals[0]
                want = E0_TABLE3[(form, scale, metric)]
                got = _e0_table3_format(scale, metric, v)
                if got != want:
                    raise BuildError(
                        f"fig02 pooled: {form}/{scale}/{metric} = {v!r} formats to {got}, "
                        f"Table 3 says {want}"
                    )
                if (v < 0.0) != want.startswith("-"):
                    raise BuildError(f"fig02 pooled: {form}/{scale}/{metric} sign flip")
                table3_record[f"{form}|{scale}|{metric}"] = {"value": v, "manuscript": want}
                unit = (
                    "dimensionless"
                    if metric == "kge"
                    else ("mm d-1" if scale == "daily" else "mm month-1")
                )
                pooled_rows.append(
                    {
                        "formulation": form,
                        "scale": scale,
                        "metric": metric,
                        "value": v,
                        "unit": unit,
                        "n_sites": EXPECTED[
                            "E0_pooled_sites" if scale == "daily" else "E0_pooled_monthly_sites"
                        ],
                        "n_paired": EXPECTED[
                            "E0_pooled_daily" if scale == "daily" else "E0_pooled_monthly"
                        ],
                        "evaluation_mask_id": E0_MASK_ID,
                        "manuscript_value": want,
                        "source_sha256": gates_hash,
                    }
                )
    if pooled_vals:
        raise BuildError(f"fig02 pooled: unexpected extra entries {sorted(pooled_vals)}")
    pooled = pd.DataFrame(pooled_rows)
    require_count(len(pooled), 18, "fig02 pooled rows")

    # ---- paired site RMSE effects (panel c) ----
    ps_cols = [
        "fid",
        "n_daily",
        "rmse_a_daily",
        "rmse_b_daily",
        "n_monthly",
        "rmse_a_monthly",
        "rmse_b_monthly",
    ]
    eff_frames = []
    win_counts: dict[tuple[str, str], int] = {}
    scale_sets: dict[tuple[str, str], frozenset] = {}
    per_site_hashes: dict[str, str] = {}
    for comp_key in ("isolated_cover", "whole_formulation"):
        comp = E0_COMPARISONS[comp_key]
        ps_path = comp["dir"] / "pooled_per_site.csv"
        ps = pd.read_csv(ps_path)
        label = f"fig02 per-site {comp_key}"
        require_columns(ps, ps_cols, label)
        require_unique(ps, ["fid"], label)
        require_count(len(ps), EXPECTED["E0_effect_daily_sites"], f"{label} daily sites")
        if list(ps["fid"].astype(str)) != sites:
            raise BuildError(f"{label}: fids are not the E0 site list in file order")
        if ps["rmse_a_daily"].isna().any() or ps["rmse_b_daily"].isna().any():
            raise BuildError(f"{label}: missing daily RMSE")
        per_site_hashes[comp_key] = sha256(ps_path)
        # Sites contributing pooled monthly sums (>= 6 qualifying months); the
        # per-site monthly metric needs >= 10 (E0_effect_monthly_sites).
        require_count(
            int((ps["n_monthly"] >= E0_POOLED_MIN_MONTHS).sum()),
            EXPECTED["E0_pooled_monthly_sites"],
            f"{label} pooled-month sites",
        )
        mo_fin = ps["rmse_a_monthly"].notna() & ps["rmse_b_monthly"].notna()
        if (ps.loc[mo_fin, "n_monthly"] < E0_SITE_METRIC_MIN_MONTHS).any() or (
            ps.loc[~mo_fin, "n_monthly"] >= E0_SITE_METRIC_MIN_MONTHS
        ).any():
            raise BuildError(
                f"{label}: monthly-metric availability violates the "
                f">= {E0_SITE_METRIC_MIN_MONTHS} qualifying-months rule"
            )
        for scale, sub in (("daily", ps), ("monthly", ps[mo_fin])):
            if scale == "monthly":
                require_count(
                    len(sub), EXPECTED["E0_effect_monthly_sites"], f"{label} monthly sites"
                )
            scale_sets[(comp_key, scale)] = frozenset(sub["fid"].astype(str))
            a = sub[f"rmse_a_{scale}"].to_numpy(dtype=float)
            b = sub[f"rmse_b_{scale}"].to_numpy(dtype=float)
            win = a < b
            win_counts[(comp_key, scale)] = int(win.sum())
            eff_frames.append(
                pd.DataFrame(
                    {
                        "site_id": sub["fid"].astype(str).to_numpy(),
                        "scale": scale,
                        "comparator": comp_key,
                        "rmse_cover_scaled": a,
                        "rmse_unscaled": b,
                        "d_rmse": a - b,
                        "n_paired": sub[f"n_{scale}"].to_numpy(dtype=int),
                        "win_cover_scaled": win,
                        "source_sha256": per_site_hashes[comp_key],
                    }
                )
            )
    for scale in ("daily", "monthly"):
        if scale_sets[("isolated_cover", scale)] != scale_sets[("whole_formulation", scale)]:
            raise BuildError(f"fig02 effects: {scale} site sets differ between comparators")
    for comp_key, tag in (("isolated_cover", "iso"), ("whole_formulation", "whole")):
        for scale in ("daily", "monthly"):
            require_count(
                win_counts[(comp_key, scale)],
                EXPECTED[f"E0_{tag}_{scale}_wins"],
                f"fig02 {comp_key} {scale} wins",
            )
    eff = pd.DataFrame(pd.concat(eff_frames, ignore_index=True))
    require_count(
        len(eff),
        2 * (EXPECTED["E0_effect_daily_sites"] + EXPECTED["E0_effect_monthly_sites"]),
        "fig02 effect rows",
    )
    if not eff["d_rmse"].notna().all():
        raise BuildError("fig02 effects: non-finite d_rmse")
    n_d, n_m = EXPECTED["E0_effect_daily_sites"], EXPECTED["E0_effect_monthly_sites"]
    win_text = {
        comp_key: {
            "daily": f"{win_counts[(comp_key, 'daily')]}/{n_d}",
            "monthly": f"{win_counts[(comp_key, 'monthly')]}/{n_m}",
        }
        for comp_key in ("isolated_cover", "whole_formulation")
    }

    # ---- quarantine the superseded External-ET-agreement package ----
    quarantined = []
    qdir = paths.OUT / "superseded_fig02_et_agreement"
    for name in (
        "fig02_daily_site_metrics.csv",
        "fig02_monthly_site_metrics.csv",
        "fig02_site_effects.csv",
    ):
        p = paths.OUT / name
        if p.exists():
            qdir.mkdir(exist_ok=True)
            p.rename(qdir / name)
            quarantined.append(name)

    # ---- write tables, metadata, and manifest records ----
    nr = write_table(resp, "fig02_formulation_response.csv")
    ns = write_table(support, "fig02_ndvi_support.csv")
    np_ = write_table(pooled, "fig02_pooled_metrics.csv")
    ne = write_table(eff, "fig02_site_rmse_effects.csv")

    meta_json = {
        "figure": "Figure 2 -- vegetation formulation and flux-tower ET performance",
        "role": (
            "E0 vegetation-formulation comparison: three formulations calibrated to the "
            "remote-sensing ETf target with the E2 inputs and processing, then evaluated "
            "against withheld flux-tower ET at the 37 E2 cropland sites that are not in E1"
        ),
        "sites": {
            "n": len(sites),
            "list_file": str(E0_SITES_FILE),
            "list_sha256": sha256(E0_SITES_FILE),
            "ids": sites,
            "container_fields": len(uids),
        },
        "formulations": {
            key: {
                "display_label": arm["display_label"],
                "kcb_ndvi_mode": arm["kcb_ndvi_mode"],
                "transpiration_cover_mode": arm["transpiration_cover_mode"],
                "veg_params": list(arm["veg_params"]),
                "run_label_provenance_only": arm["run_label"],
                "par_csv": str(arm["par_csv"]),
                "par_sha256": par_hashes[key],
                "container": str(arm["container"]),
                "config": arm["config"],
                "n_ies_realizations": n_reals[key],
            }
            for key, arm in E0_ARMS.items()
        },
        "equations": {
            "kcb_sigmoid": "Kcb = Kc_max / (1 + exp(-ndvi_k (NDVI - ndvi_0)))",
            "kcb_linear": "Kcb = clip(ndvi_beta NDVI + ndvi_alpha, 0, Kc_max)",
            "fc": "fc = clip((Kcb - Kc_min) / (Kc_max - Kc_min), 0, 0.99)",
            "k_t": (
                "K_T = fc Kcb for the cover-scaled formulation and K_T = Kcb for the "
                "unscaled formulations, with Ks = 1 and the common soil-evaporation term "
                "omitted; K_T is the potential canopy-transpiration multiplier, not total Kc"
            ),
        },
        "constants": {
            "kc_max": E0_KC_MAX,
            "kc_min": E0_KC_MIN,
            "fc_max": E0_FC_MAX,
            "sigmoid_exp_clip": E0_SIGMOID_EXP_CLIP,
            "provenance": (
                "src/swimrs/process/input.py::_resolve_kc_max floors the requested "
                "empirical kc_max at 1.35 and the arm containers store "
                "derived/dynamics/kc_max = 1.35 at every E0 site (asserted), so kc_max is "
                "1.35 in every arm; kc_min is fixed at 0.15 (input.py); clip semantics "
                "from src/swimrs/process/loop_fast.py"
            ),
        },
        "priors_note": (
            "the unscaled-linear arm carries formulation-specific slope/intercept priors "
            "(ndvi_alpha, ndvi_beta) in place of the logistic (ndvi_k, ndvi_0), so its "
            "ordering against the unscaled sigmoid is not an isolated curve-shape ablation; "
            "all non-vegetation priors are identical across arms"
        ),
        "ndvi_grid": {"start": 0.0, "stop": 1.0, "step": 0.01, "points": len(E0_NDVI_GRID)},
        "ndvi_support": {
            "rule": E0_SUPPORT_RULE,
            "source": "derived/merged_ndvi/no_mask of " + str(E0_CONTAINER),
            "site_ids_source": "geometry/uid of the same container, restricted to the E0 sites",
            "sha256": ndvi_hash,
            "n_sites": len(sites),
            "n_obs_total": int(finite.sum()),
            "per_site_obs_min": int(per_site_obs.min()),
            "per_site_obs_max": int(per_site_obs.max()),
            "n_bins": len(E0_SUPPORT_BINS) - 1,
            "bin_width": width,
        },
        "evaluation_mask": {
            "id": E0_MASK_ID,
            "n_sites": EXPECTED["E0_pooled_sites"],
            "n_daily": EXPECTED["E0_pooled_daily"],
            "n_sites_monthly": EXPECTED["E0_pooled_monthly_sites"],
            "n_monthly": EXPECTED["E0_pooled_monthly"],
            "flux_source": (
                "closure-corrected per-site flux ET from the (network, et_col) pair declared "
                "in the E2 fields shapefile (examples/6_Flux_International/evaluate.py::"
                "load_flux_sources / load_flux_et); site minimum 90 flux days and 3 months "
                "with >= 20 flux days"
            ),
            "daily_rule": (
                "days with finite flux ET and finite simulated ET in all three formulations "
                "(three-arm common finite mask)"
            ),
            "pooled_month_rule": (
                "paired-day monthly sums over flux-valid days with >= "
                f"{E0_MONTH_MIN_PAIRED_DAYS} paired days per month "
                "(swimrs.calibrate.flux_utils.paired_monthly_sums), not full-calendar-month "
                f"totals; a site contributes monthly sums when it has >= {E0_POOLED_MIN_MONTHS} "
                "such months"
            ),
            "site_metric_month_rule": (
                f">= {E0_SITE_METRIC_MIN_MONTHS} qualifying months (calc_metrics minimum "
                f"n = 10); {EXPECTED['E0_effect_monthly_sites']} of "
                f"{EXPECTED['E0_effect_daily_sites']} evaluation sites qualify"
            ),
        },
        "site_effects": {
            "d_rmse_sign": (
                "d_rmse = rmse_cover_scaled - rmse_unscaled; negative favours the "
                "cover-scaled formulation"
            ),
            "comparators": {
                "isolated_cover": "cover_scaled_sigmoid minus unscaled_sigmoid",
                "whole_formulation": "cover_scaled_sigmoid minus unscaled_linear",
            },
            "win_counts": win_text,
        },
        "comparison_sources": {
            key: {
                "dir": str(comp["dir"]),
                "arm_a": comp["arm_a"],
                "arm_b": comp["arm_b"],
                "gate_sha256": sha256(comp["dir"] / "pooled_gate.json"),
                "per_site_sha256": sha256(comp["dir"] / "pooled_per_site.csv"),
            }
            for key, comp in E0_COMPARISONS.items()
        },
        "table3_reproduction": table3_record,
        "table3_sign_note": (
            "manuscript_value keeps plain ASCII signs; main.md prints a typographic minus "
            "and an explicit plus on signed MBE"
        ),
        "veg_param_site_medians": veg_medians,
        "builder_version": SCRIPT_VERSION,
        "prior_package": {
            "moved_to": str(paths.OUT / E0_PRIOR_PACKAGE_DIR),
            "note": (
                "the previous Figure 2 data package and its manifest, retained as history "
                "only and not part of the active manifest"
            ),
        },
        "superseded_package": {
            "moved_to": str(qdir),
            "files": quarantined,
            "note": (
                "former External-ET-agreement Figure 2 package (E1/E2 metric dashboard); "
                "retained as design provenance only, removed from the active manifest"
            ),
        },
    }
    (paths.OUT / "fig02_metadata.json").write_text(json.dumps(meta_json, indent=2))

    common_meta = dict(
        experiment=E0_EXPERIMENT_LABEL,
        evaluation_mask_id=E0_MASK_ID,
        sources={k: str(p) for k, p in srcs.items()},
        source_hashes={
            **{f"par:{k}": v for k, v in par_hashes.items()},
            **{f"per_site:{k}": v for k, v in per_site_hashes.items()},
            "gates_combined": gates_hash,
            "merged_ndvi": ndvi_hash,
            "sites": sha256(E0_SITES_FILE),
        },
    )
    MANIFEST.add(
        "fig02_formulation_response.csv",
        rows=nr,
        note=(
            "Fitted K_T response per formulation and site on the frozen NDVI grid, from "
            "posterior-median vegetation parameters. Display transformation only; no model "
            "rerun."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig02_ndvi_support.csv",
        rows=ns,
        note="Site-equal observed merged-NDVI density underlay for panel (a).",
        **common_meta,
    )
    MANIFEST.add(
        "fig02_pooled_metrics.csv",
        rows=np_,
        note=(
            "Pooled KGE/RMSE/signed-MBE per formulation and scale on the Table 3 arm-paired "
            "flux mask; values asserted to reproduce Table 3 at manuscript precision."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig02_site_rmse_effects.csv",
        rows=ne,
        note=(
            "Paired per-site RMSE effects; d_rmse = cover-scaled minus unscaled (negative "
            f"favours cover scaling). Isolated-cover wins {win_text['isolated_cover']['daily']} "
            f"daily and {win_text['isolated_cover']['monthly']} monthly; whole-formulation "
            f"wins {win_text['whole_formulation']['daily']} daily and "
            f"{win_text['whole_formulation']['monthly']} monthly."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig02_metadata.json",
        rows=None,
        note="Reader-facing labels, equations, provenance, rules, and assertion record.",
        **common_meta,
    )
    print(
        f"  fig02: response {nr} rows, NDVI {ns} bins ({int(finite.sum())} obs.), "
        f"pooled {np_} rows, effects {ne} rows; quarantined {quarantined or 'nothing'}"
    )


# ---- fig02 whole-site bootstrap (panel b interval rules) -------------------

E0_BOOT_N = 10_000
E0_BOOT_SEED = 42


def _e0_evaluate_module():
    """examples/6_Flux_International/evaluate.py under a collision-free module name.

    Several example directories ship an ``evaluate.py``; loading the Example 6
    one by path keeps it apart from any other already imported in this process.
    """
    import importlib.util

    path = REPO / E0_EXAMPLE / "evaluate.py"
    spec = importlib.util.spec_from_file_location("e0_flux_international_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e0_suff(obs: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Per-site sufficient statistics [n, So, Sm, Soo, Smm, Som]."""
    return np.array(
        [
            float(len(obs)),
            obs.sum(),
            sim.sum(),
            (obs * obs).sum(),
            (sim * sim).sum(),
            (obs * sim).sum(),
        ]
    )


def _e0_pooled_from_suff(s: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled KGE / RMSE / signed MBE from summed sufficient statistics.

    Matches evaluate.py::calc_metrics arithmetic: population moments (ddof 0),
    KGE-2009 with alpha = sd(model)/sd(obs) and beta = mean(model)/mean(obs),
    MBE = model - obs.  Vectorized over any leading dimensions of ``s``.
    """
    n, so, sm, soo, smm, som = (s[..., i] for i in range(6))
    mo, mm = so / n, sm / n
    var_o = soo / n - mo * mo
    var_m = smm / n - mm * mm
    r = (som / n - mo * mm) / np.sqrt(var_o * var_m)
    alpha = np.sqrt(var_m / var_o)
    beta = mm / mo
    return {
        "kge": 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2),
        "rmse": np.sqrt((soo - 2.0 * som + smm) / n),
        "mbe": mm - mo,
    }


def build_fig02_bootstrap() -> None:
    """95% whole-site bootstrap intervals for the Figure 2 panel (b) pooled metrics.

    Re-runs the three E0 arms in forecast mode at the 37 E0 sites (the
    examples/6_Flux_International/pooled_arm_compare.py recipe: closure-corrected
    per-site flux sources from the E2 fields shapefile, paired-day monthly sums
    with >= 20 paired days), rebuilds the arm-paired flux mask requiring all
    three arms jointly finite, asserts exact reproduction of the frozen pooled
    point values, the per-site RMSE table, and the Table 3 strings, then
    resamples the 37 evaluation sites with replacement.  A drawn site
    contributes all of its paired days and monthly sums; daily and monthly
    statistics share the same site draw.  Design frozen before any interval
    was inspected.
    """
    ev = _e0_evaluate_module()
    from swimrs.calibrate.flux_utils import paired_monthly_sums, passes_site_minimum
    from swimrs.container import SwimContainer
    from swimrs.swim.config import ProjectConfig

    frozen_path = paths.OUT / "fig02_pooled_metrics.csv"
    srcs: dict[str, Path] = {"frozen_pooled": frozen_path, "sites": E0_SITES_FILE}
    for key, arm in E0_ARMS.items():
        srcs[f"par:{key}"] = arm["par_csv"]
        srcs[f"config:{key}"] = REPO / arm["config"]
        srcs[f"container:{key}"] = arm["container"]
    for key, comp in E0_COMPARISONS.items():
        srcs[f"gate:{key}"] = comp["dir"] / "pooled_gate.json"
        srcs[f"per_site:{key}"] = comp["dir"] / "pooled_per_site.csv"
    for k, p in srcs.items():
        if not p.exists():
            raise BuildError(f"fig02 bootstrap source missing: {k} -> {p}")

    # round_trip: the frozen point values must re-serialize byte-identically.
    frozen = pd.read_csv(frozen_path, float_precision="round_trip")
    require_count(len(frozen), 18, "fig02 bootstrap frozen pooled rows")
    frozen_vals = {
        (r["formulation"], r["scale"], r["metric"]): float(r["value"]) for _, r in frozen.iterrows()
    }
    per_site_ref = {
        key: pd.read_csv(E0_COMPARISONS[key]["dir"] / "pooled_per_site.csv").set_index("fid")
        for key in ("isolated_cover", "whole_formulation")
    }

    # ---- forward re-runs, exactly as pooled_arm_compare ran each arm ----
    fids = _e0_sites()
    forms = list(E0_ARMS)
    cfgs: dict[str, object] = {}
    series: dict[str, dict] = {}
    for key, arm in E0_ARMS.items():
        cfg = ProjectConfig()
        cfg.read_config(str(REPO / arm["config"]), calibrate=True)
        cfgs[key] = cfg
        params = ev.parse_pest_params(str(arm["par_csv"]), fids)
        missing = [f for f in fids if f not in params]
        if missing:
            raise BuildError(f"fig02 bootstrap: {key} has no posterior for {missing}")
        container = SwimContainer.open(str(arm["container"]), mode="r")
        try:
            results = ev.run_calibrated_model(cfg, container, fids, params)
        finally:
            container.close()
        series[key] = {fid: df["et_act"] for fid, df in results.items()}
        if sorted(series[key]) != sorted(fids):
            raise BuildError(f"fig02 bootstrap: {key} forward run did not return every site")
        print(f"  fig02 bootstrap: {key} forward run done ({len(results)} sites)")

    shp = {k: (c.fields_shapefile, c.feature_id_col) for k, c in cfgs.items()}
    if len(set(shp.values())) != 1:
        raise BuildError(f"fig02 bootstrap: arms disagree on the fields shapefile: {shp}")
    cfg0 = cfgs["cover_scaled_sigmoid"]
    flux_sources = ev.load_flux_sources(cfg0.fields_shapefile, cfg0.feature_id_col)
    if any(fid not in flux_sources for fid in fids):
        raise BuildError("fig02 bootstrap: an E0 site has no declared flux source")

    # ---- arm-paired mask: flux and all three arms finite on the same day ----
    site_ids: list[str] = []
    d_obs: list[np.ndarray] = []
    m_obs: list[np.ndarray] = []
    d_sim: dict[str, list] = {k: [] for k in forms}
    m_sim: dict[str, list] = {k: [] for k in forms}
    d_suff: dict[str, list] = {k: [] for k in forms}
    m_suff: dict[str, list] = {k: [] for k in forms}
    n_site_monthly: dict[str, int] = {}
    et_cols: dict[str, str] = {}
    for fid in fids:
        flux_et = ev.load_flux_et(fid, flux_sources[fid])
        if flux_et.empty or not passes_site_minimum(flux_et):
            raise BuildError(f"fig02 bootstrap: {fid} fails the flux site minimum")
        et_cols[fid] = str(flux_et.attrs.get("et_col"))
        common = flux_et.index
        for k in forms:
            common = common.intersection(series[k][fid].index)
        obs = flux_et.loc[common].values
        sims = {k: series[k][fid].loc[common].values for k in forms}
        for k in forms:
            if not np.isfinite(sims[k]).all():
                raise BuildError(f"fig02 bootstrap: {k} simulated ET is not finite at {fid}")
        mask = np.isfinite(obs)
        for k in forms:
            mask &= np.isfinite(sims[k])
        if mask.sum() < 10:
            raise BuildError(f"fig02 bootstrap: {fid} has only {int(mask.sum())} paired days")
        site_ids.append(fid)
        d_obs.append(obs[mask])
        for k in forms:
            d_sim[k].append(sims[k][mask])
            d_suff[k].append(_e0_suff(obs[mask], sims[k][mask]))

        # E2 monthly basis: sums over flux-valid days, >= 20 paired days per
        # month, every arm integrating the identical day set.
        flux_daily = flux_et.loc[common]
        mo_sums = {}
        flux_mo = None
        for k in forms:
            s_mo, f_mo, _ = paired_monthly_sums(
                series[k][fid].loc[common], flux_daily, month_min_days=E0_MONTH_MIN_PAIRED_DAYS
            )
            mo_sums[k] = s_mo
            if flux_mo is None:
                flux_mo = f_mo
            elif not flux_mo.index.equals(f_mo.index) or not np.array_equal(
                flux_mo.values, f_mo.values, equal_nan=True
            ):
                raise BuildError(f"fig02 bootstrap: monthly flux sums differ by arm at {fid}")
        o_mo = flux_mo.values
        sm_mo = {k: mo_sums[k].reindex(flux_mo.index).values for k in forms}
        mmask = np.isfinite(o_mo)
        for k in forms:
            mmask &= np.isfinite(sm_mo[k])
        n_site_monthly[fid] = int(mmask.sum())
        if mmask.sum() >= E0_POOLED_MIN_MONTHS:
            m_obs.append(o_mo[mmask])
            for k in forms:
                m_sim[k].append(sm_mo[k][mmask])
                m_suff[k].append(_e0_suff(o_mo[mmask], sm_mo[k][mmask]))
        else:
            for k in forms:
                m_suff[k].append(np.zeros(6))

    require_count(len(site_ids), EXPECTED["E0_pooled_sites"], "fig02 bootstrap sites")
    n_daily = int(sum(len(v) for v in d_obs))
    require_count(n_daily, EXPECTED["E0_pooled_daily"], "fig02 bootstrap site-days")
    n_monthly = int(sum(len(v) for v in m_obs))
    require_count(n_monthly, EXPECTED["E0_pooled_monthly"], "fig02 bootstrap monthly sums")
    if any(c != "ET_corr" for c in et_cols.values()):
        raise BuildError(f"fig02 bootstrap: non-closure-corrected flux column in {et_cols}")

    # ---- per-site values must reproduce the panel (c) source tables ----
    arm_side = {
        "isolated_cover": {"cover_scaled_sigmoid": "a", "unscaled_sigmoid": "b"},
        "whole_formulation": {"cover_scaled_sigmoid": "a", "unscaled_linear": "b"},
    }
    m_i = 0
    for i, fid in enumerate(site_ids):
        has_mo = n_site_monthly[fid] >= E0_POOLED_MIN_MONTHS
        for comp_key, sides in arm_side.items():
            ref = per_site_ref[comp_key].loc[fid]
            if int(ref["n_daily"]) != len(d_obs[i]) or int(ref["n_monthly"]) != n_site_monthly[fid]:
                raise BuildError(f"fig02 bootstrap: {fid} paired counts differ from {comp_key}")
            for k, side in sides.items():
                got = ev.calc_metrics(d_obs[i], d_sim[k][i])["rmse"]
                if abs(float(got) - float(ref[f"rmse_{side}_daily"])) > 1e-9:
                    raise BuildError(f"fig02 bootstrap: {fid} daily RMSE drift vs {comp_key}")
                if has_mo:
                    got_m = ev.calc_metrics(m_obs[m_i], m_sim[k][m_i])["rmse"]
                    ref_m = float(ref[f"rmse_{side}_monthly"])
                    if np.isfinite(ref_m) and abs(float(got_m) - ref_m) > 1e-9:
                        raise BuildError(f"fig02 bootstrap: {fid} monthly RMSE drift vs {comp_key}")
                    if np.isfinite(ref_m) != (n_site_monthly[fid] >= E0_SITE_METRIC_MIN_MONTHS):
                        raise BuildError(f"fig02 bootstrap: {fid} monthly-metric rule mismatch")
        if has_mo:
            m_i += 1
    if m_i != len(m_obs):
        raise BuildError("fig02 bootstrap: monthly site bookkeeping mismatch")

    # ---- point values must reproduce the frozen package and Table 3 ----
    cat = {
        "daily": (np.concatenate(d_obs), {k: np.concatenate(d_sim[k]) for k in forms}),
        "monthly": (np.concatenate(m_obs), {k: np.concatenate(m_sim[k]) for k in forms}),
    }
    suff_arr = {("daily", k): np.vstack(d_suff[k]) for k in forms} | {
        ("monthly", k): np.vstack(m_suff[k]) for k in forms
    }

    point: dict[tuple[str, str, str], float] = {}
    for scale, (obs_v, sims_v) in cat.items():
        for k in forms:
            m = ev.calc_metrics(obs_v, sims_v[k])
            direct = {"kge": m["kge"], "rmse": m["rmse"], "mbe": m["bias"]}
            via_suff = _e0_pooled_from_suff(suff_arr[(scale, k)].sum(axis=0))
            for metric in ("kge", "rmse", "mbe"):
                v = float(direct[metric])
                got = _e0_table3_format(scale, metric, v)
                want = E0_TABLE3[(k, scale, metric)]
                if got != want:
                    raise BuildError(
                        f"fig02 bootstrap: {k}/{scale}/{metric} = {v!r} formats to {got}, "
                        f"Table 3 says {want}"
                    )
                fv = frozen_vals[(k, scale, metric)]
                if abs(v - fv) > 1e-9:
                    raise BuildError(
                        f"fig02 bootstrap: {k}/{scale}/{metric} recomputed {v!r} != frozen {fv!r}"
                    )
                if abs(float(via_suff[metric]) - v) > 1e-7:
                    raise BuildError(
                        f"fig02 bootstrap: sufficient-statistic {metric} disagrees with "
                        f"calc_metrics for {k}/{scale}"
                    )
                point[(k, scale, metric)] = fv

    # ---- whole-site resampling; one draw shared by every arm and scale ----
    rng = np.random.default_rng(E0_BOOT_SEED)
    draws = rng.integers(0, len(site_ids), size=(E0_BOOT_N, len(site_ids)))
    reps: dict[tuple[str, str, str], np.ndarray] = {}
    for scale in ("daily", "monthly"):
        for k in forms:
            sums = suff_arr[(scale, k)][draws].sum(axis=1)
            if scale == "monthly" and float(sums[:, 0].min()) < 60.0:
                raise BuildError("fig02 bootstrap: a monthly resample has < 60 sums")
            met = _e0_pooled_from_suff(sums)
            for metric in ("kge", "rmse", "mbe"):
                r = met[metric]
                if not np.isfinite(r).all():
                    raise BuildError(f"fig02 bootstrap: nonfinite {k}/{scale}/{metric} resample")
                reps[(k, scale, metric)] = r

    rows = []
    for k in forms:
        for scale in ("daily", "monthly"):
            for metric in ("kge", "rmse", "mbe"):
                r = reps[(k, scale, metric)]
                rows.append(
                    {
                        "formulation": k,
                        "scale": scale,
                        "metric": metric,
                        "value": point[(k, scale, metric)],
                        "ci95_lo": float(np.percentile(r, 2.5)),
                        "ci95_hi": float(np.percentile(r, 97.5)),
                        "n_resamples": E0_BOOT_N,
                        "seed": E0_BOOT_SEED,
                        "resample_unit": (
                            "whole site (all paired days and monthly sums retained "
                            "per sampled site)"
                        ),
                        "n_sites": len(site_ids),
                        "n_paired": n_daily if scale == "daily" else n_monthly,
                        "evaluation_mask_id": E0_MASK_ID,
                        "unit": (
                            "dimensionless"
                            if metric == "kge"
                            else ("mm d-1" if scale == "daily" else "mm month-1")
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    n = write_table(out, "fig02_pooled_bootstrap.csv")

    MANIFEST.add(
        "fig02_pooled_bootstrap.csv",
        rows=n,
        experiment=E0_EXPERIMENT_LABEL,
        evaluation_mask_id=E0_MASK_ID,
        sources={k: str(p) for k, p in srcs.items()},
        source_hashes={
            "frozen_pooled": sha256(frozen_path),
            "sites": sha256(E0_SITES_FILE),
            **{f"par:{k}": sha256(E0_ARMS[k]["par_csv"]) for k in forms},
            **{
                f"gate:{k}": sha256(c["dir"] / "pooled_gate.json")
                for k, c in E0_COMPARISONS.items()
            },
        },
        inclusion_rule=(
            "Arm-paired flux mask rebuilt at the 37 E0 sites with all three formulations "
            "required jointly finite; closure-corrected per-site flux sources and paired-day "
            "monthly sums (>= 20 paired days per month, >= 6 months per site) as in "
            "examples/6_Flux_International/pooled_arm_compare.py; asserted identical to the "
            "frozen Table 3 mask (37 sites, 49,289 site-days, 1,514 monthly sums), every "
            "pooled point value asserted to reproduce fig02_pooled_metrics.csv and Table 3 "
            "at manuscript precision, and every per-site RMSE asserted to reproduce the "
            "panel (c) source tables."
        ),
        bootstrap=(
            "Whole-site resampling with replacement, all paired days and monthly sums "
            "retained for each sampled site, 10,000 resamples, numpy default_rng(42); "
            "daily and monthly statistics share the same site draw. Design frozen before "
            "any interval was inspected."
        ),
        interval_caveat=(
            "Intervals describe pooled-magnitude uncertainty per formulation and are NOT "
            "paired formulation contrasts; paired site-level evidence lives in "
            "fig02_site_rmse_effects.csv."
        ),
        deterministic_seed=E0_BOOT_SEED,
        independent_unit="site",
    )
    print(f"  fig02 bootstrap: {n} rows, {E0_BOOT_N} whole-site resamples, seed {E0_BOOT_SEED}")
    for _, r in out.iterrows():
        print(
            f"    {r['formulation']:<22s} {r['scale']:<7s} {r['metric']:<4s} "
            f"{r['value']:>8.3f}  [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
        )

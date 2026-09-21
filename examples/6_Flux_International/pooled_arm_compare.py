"""Pooled two-arm vegetation-formulation comparison on the Example 6 (E2) footing.

Example 6 port of ``examples/5_Flux_Ensemble/pooled_arm_compare.py`` for the E0 disjoint
confirmation: the Ex5 formulation trio re-run on the E2 cohort, scored on the sites that
are NOT in the E1 (Ex5) calibration cohort. Two calibrated arms are run forward from their
own posterior under their own physics and paired against the same flux observations on
identical site-days; every site-day is pooled before RMSE / MBE / KGE are computed, and the
Ex5 gate (arm A wins >= 4 of 6 pooled daily+monthly metrics) is applied.

    arm A  cover-scaled sigmoid   kc_act = fc*Ks*Kcb + Ke,  Kcb = kc_max*sigmoid(NDVI)
           (6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr, no rerun)
    arm B  unscaled sigmoid       kc_act =    Ks*Kcb + Ke,  same Kcb          (..._fao56_sig)
    arm C  unscaled linear        kc_act =    Ks*Kcb + Ke,  Kcb = beta*NDVI+alpha (..._fao56)

Differences from the Ex5 script, all inherited from ``evaluate.py`` here: flux truth is the
per-site (network, et_col) source declared in the cohort shapefile (closure-corrected
ET_corr on the paper pool); the site minimum is VALIDATION_POLICY's 90 d / 3 mo; monthly
totals use ``paired_monthly_sums`` (flux-valid days only, >= 20 valid days per month), the
E2 monthly basis, not Ex5's ``full_month_paired_sums``. The paired-day mask is arm-vs-arm
(flux, A and B finite) and does not require the RS benchmark, so counts run slightly above
the canonical E2 evaluation tables.

Parameters come from ``--a-par`` / ``--b-par`` (a batch-runner ``merged_posterior.csv`` or a
PEST++ ``.par.csv``); without one, the arm container's ingested calibration is used.

Usage:
    uv run python examples/6_Flux_International/pooled_arm_compare.py \
        --a-name grassbasis --a-config 6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml \
        --a-par .../archive/4_pest_outputs/merged/merged_posterior.csv \
        --b-name fao56_sig --b-config 6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56_sig.toml \
        --b-par .../archive/4_pest_outputs/merged/merged_posterior.csv \
        --sites-file e0_disjoint/disjoint_sites.txt --out-dir .../comparison_disjoint
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from evaluate import (
    _default_container_path,
    _resolve_calibrated_params,
    calc_metrics,
    load_flux_et,
    load_flux_sources,
    run_calibrated_model,
)

from swimrs.calibrate.flux_utils import paired_monthly_sums, passes_site_minimum
from swimrs.container import SwimContainer
from swimrs.process.cover_modes import COVER_MODE_NAMES, resolve_cover_mode
from swimrs.process.kcb_modes import KCB_MODE_NAMES, resolve_kcb_mode
from swimrs.swim.config import ProjectConfig

HERE = os.path.dirname(os.path.abspath(__file__))

# rmse lower is better; bias closer to zero (|.|); kge higher
METRICS = [("rmse", "lower"), ("bias", "abs_lower"), ("kge", "higher")]


def load_arm_config(config_path):
    cfg = ProjectConfig()
    cfg.read_config(config_path, calibrate=True)
    return cfg


def effective_physics(cfg):
    """Physics actually used (absent keys resolve to the historical defaults)."""
    kcb = KCB_MODE_NAMES[resolve_kcb_mode(getattr(cfg, "kcb_ndvi_mode", None))]
    cover = COVER_MODE_NAMES[
        resolve_cover_mode(
            getattr(cfg, "transpiration_cover_mode", None),
            getattr(cfg, "transpiration_cover_scaling", None),
        )
    ]
    return {"kcb_ndvi_mode": kcb, "transpiration_cover_mode": cover}


def arm_series(cfg, container_path, par_csv, fids):
    """Run one arm forward under its own physics; return ({fid: et_act}, parameter source)."""
    container = SwimContainer.open(container_path, mode="r")
    try:
        params, source = _resolve_calibrated_params(container, fids, par_csv)
        missing = [f for f in fids if f not in params]
        if missing:
            print(f"  WARNING: no calibrated params for {missing}")
        fids = [f for f in fids if f in params]
        results = run_calibrated_model(cfg, container, fids, params)
    finally:
        container.close()
    return {fid: df["et_act"] for fid, df in results.items()}, source


def collect(fids, flux_sources, series_a, series_b):
    pooled = {k: [] for k in ("obs", "a", "b")}
    pooled_mo = {k: [] for k in ("obs", "a", "b")}
    per_site, excluded = [], []

    for fid in fids:
        flux_et = load_flux_et(fid, flux_sources.get(fid))
        if flux_et.empty:
            excluded.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            excluded.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue
        if fid not in series_a or fid not in series_b:
            excluded.append({"site": fid, "reason": "missing_in_one_arm"})
            continue

        a_et, b_et = series_a[fid], series_b[fid]
        common = flux_et.index.intersection(a_et.index).intersection(b_et.index)
        obs = flux_et.loc[common].values
        av = a_et.loc[common].values
        bv = b_et.loc[common].values
        mask = np.isfinite(obs) & np.isfinite(av) & np.isfinite(bv)
        if mask.sum() < 10:
            excluded.append({"site": fid, "reason": f"only_{int(mask.sum())}_paired_days"})
            continue

        pooled["obs"].append(obs[mask])
        pooled["a"].append(av[mask])
        pooled["b"].append(bv[mask])

        row = {
            "fid": fid,
            "n_daily": int(mask.sum()),
            "flux_et_col": flux_et.attrs.get("et_col"),
        }
        for arm, vals in (("a", av), ("b", bv)):
            m = calc_metrics(obs[mask], vals[mask])
            for k in ("rmse", "bias", "kge", "r2"):
                row[f"{k}_{arm}_daily"] = m[k]

        # E2 monthly basis: totals over flux-valid days, >= 20 valid days per month,
        # both arms integrating the identical day set
        flux_daily = flux_et.loc[common]
        a_mo, flux_mo, b_mo = paired_monthly_sums(a_et.loc[common], flux_daily, b_et.loc[common])
        idx = flux_mo.index
        o_mo = flux_mo.values
        am = a_mo.reindex(idx).values
        bm = b_mo.reindex(idx).values
        mmask = np.isfinite(o_mo) & np.isfinite(am) & np.isfinite(bm)
        row["n_monthly"] = int(mmask.sum())
        if mmask.sum() >= 6:
            pooled_mo["obs"].append(o_mo[mmask])
            pooled_mo["a"].append(am[mmask])
            pooled_mo["b"].append(bm[mmask])
            for arm, vals in (("a", am), ("b", bm)):
                m = calc_metrics(o_mo[mmask], vals[mmask])
                for k in ("rmse", "bias", "kge", "r2"):
                    row[f"{k}_{arm}_monthly"] = m[k]

        per_site.append(row)
        print(
            f"  {fid}: {row['n_daily']:>5d} d / {row['n_monthly']:>3d} mo   "
            f"KGE a={row.get('kge_a_daily', float('nan')):.3f} "
            f"b={row.get('kge_b_daily', float('nan')):.3f}"
        )

    cat = {k: (np.concatenate(v) if v else np.array([])) for k, v in pooled.items()}
    cat_mo = {k: (np.concatenate(v) if v else np.array([])) for k, v in pooled_mo.items()}
    return cat, cat_mo, pd.DataFrame(per_site), pd.DataFrame(excluded, columns=["site", "reason"])


def decide(name_a, name_b, pooled_daily, pooled_monthly):
    """Ex5 E0 gate: arm A wins >= 4 of 6 pooled metrics."""
    rows, wins_a = [], 0
    for scale, p in (("daily", pooled_daily), ("monthly", pooled_monthly)):
        ma = calc_metrics(p["obs"], p["a"])
        mb = calc_metrics(p["obs"], p["b"])
        for key, direction in METRICS:
            va, vb = ma[key], mb[key]
            if direction == "lower":
                a_wins = va < vb
            elif direction == "higher":
                a_wins = va > vb
            else:
                a_wins = abs(va) < abs(vb)
            wins_a += int(bool(a_wins))
            rows.append(
                {
                    "scale": scale,
                    "metric": {"bias": "MBE"}.get(key, key.upper()),
                    "n": int(ma["n"]),
                    name_a: va,
                    name_b: vb,
                    "winner": name_a if a_wins else name_b,
                    "delta_a_minus_b": va - vb,
                }
            )
    return pd.DataFrame(rows), wins_a, wins_a >= 4


def paired_site_deltas(per_site, name_a, name_b):
    """Per-site paired A−B deltas with the site-median: the per-site view of the same estimand."""
    rows = []
    for scale in ("daily", "monthly"):
        for key, _ in METRICS:
            ca, cb = f"{key}_a_{scale}", f"{key}_b_{scale}"
            if ca not in per_site or cb not in per_site:
                continue
            a, b = per_site[ca], per_site[cb]
            if key == "bias":
                d = a.abs() - b.abs()
            else:
                d = a - b
            d = d.dropna()
            better = (d < 0) if key in ("rmse", "bias") else (d > 0)
            rows.append(
                {
                    "scale": scale,
                    "metric": {"bias": "|MBE|"}.get(key, key.upper()),
                    "n_sites": int(len(d)),
                    "median_delta_a_minus_b": float(d.median()) if len(d) else np.nan,
                    f"{name_a}_better_sites": int(better.sum()),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-name", default="grassbasis")
    ap.add_argument(
        "--a-config",
        default=os.path.join(HERE, "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml"),
    )
    ap.add_argument("--a-container", default=None, help="default: [paths] container of --a-config")
    ap.add_argument("--a-par", default=None)
    ap.add_argument("--b-name", default="fao56_sig")
    ap.add_argument(
        "--b-config",
        default=os.path.join(
            HERE, "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr_fao56_sig.toml"
        ),
    )
    ap.add_argument("--b-container", default=None)
    ap.add_argument("--b-par", default=None)
    ap.add_argument("--sites", default=None, help="Comma-separated subset")
    ap.add_argument("--sites-file", default=None, help="One site id per line")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg_a = load_arm_config(args.a_config)
    cfg_b = load_arm_config(args.b_config)
    cont_a = args.a_container or _default_container_path(cfg_a)
    cont_b = args.b_container or _default_container_path(cfg_b)

    phys_a, phys_b = effective_physics(cfg_a), effective_physics(cfg_b)
    for name, phys in ((args.a_name, phys_a), (args.b_name, phys_b)):
        print(f"arm {name}: kcb={phys['kcb_ndvi_mode']} cover={phys['transpiration_cover_mode']}")
    if phys_a == phys_b:
        raise SystemExit("Both arms resolved to identical physics — check the configs")

    if args.sites and args.sites_file:
        raise SystemExit("pass --sites or --sites-file, not both")
    if args.sites:
        fids = args.sites.split(",")
    elif args.sites_file:
        fids = [ln.strip() for ln in open(args.sites_file) if ln.strip()]
    else:
        container = SwimContainer.open(cont_a, mode="r")
        try:
            fids = sorted(container.field_uids)
        finally:
            container.close()

    flux_sources = load_flux_sources(cfg_a.fields_shapefile, cfg_a.feature_id_col)

    print(f"\nRunning arm A ({args.a_name}) forward on {len(fids)} sites...")
    series_a, src_a = arm_series(cfg_a, cont_a, args.a_par, fids)
    print(f"Running arm B ({args.b_name}) forward on {len(fids)} sites...")
    series_b, src_b = arm_series(cfg_b, cont_b, args.b_par, fids)

    print("\nPairing...")
    pooled_daily, pooled_monthly, per_site, excluded = collect(
        fids, flux_sources, series_a, series_b
    )
    table, wins_a, passed = decide(args.a_name, args.b_name, pooled_daily, pooled_monthly)
    site_deltas = paired_site_deltas(per_site, args.a_name, args.b_name)

    os.makedirs(args.out_dir, exist_ok=True)
    per_site.to_csv(os.path.join(args.out_dir, "pooled_per_site.csv"), index=False)
    excluded.to_csv(os.path.join(args.out_dir, "sites_excluded.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, "pooled_gate.csv"), index=False)
    site_deltas.to_csv(os.path.join(args.out_dir, "paired_site_deltas.csv"), index=False)

    print("\n" + "=" * 84)
    print(f"POOLED TWO-ARM COMPARISON — {len(per_site)} sites ({len(excluded)} excluded)")
    print(f"  daily site-days pooled: {len(pooled_daily['obs']):,}")
    print(f"  monthly totals pooled : {len(pooled_monthly['obs']):,}")
    print("=" * 84)
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("-" * 84)
    print(site_deltas.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("-" * 84)
    print(
        f"{args.a_name} wins {wins_a} of 6 pooled metrics  ->  GATE {'PASS' if passed else 'FAIL'}"
    )
    print("=" * 84)

    with open(os.path.join(args.out_dir, "pooled_gate.json"), "w") as fh:
        json.dump(
            {
                "arm_a": args.a_name,
                "arm_b": args.b_name,
                "a_physics": phys_a,
                "b_physics": phys_b,
                "a_config": args.a_config,
                "b_config": args.b_config,
                "a_container": cont_a,
                "b_container": cont_b,
                "a_params": src_a,
                "b_params": src_b,
                "sites_requested": fids,
                "n_sites": int(len(per_site)),
                "n_daily": int(len(pooled_daily["obs"])),
                "n_monthly": int(len(pooled_monthly["obs"])),
                "wins_a": int(wins_a),
                "gate_rule": "arm A wins >= 4 of 6 pooled metrics",
                "passed": bool(passed),
                "metrics": table.to_dict(orient="records"),
                "paired_site_deltas": site_deltas.to_dict(orient="records"),
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {args.out_dir}/pooled_gate.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

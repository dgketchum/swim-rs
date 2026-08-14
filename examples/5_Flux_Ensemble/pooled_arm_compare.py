"""Pooled two-arm comparison: SWIM current formulation vs standard FAO-56.

Scores two calibrated arms against the same flux observations on the *same*
site-days, then pools every site-day into a single vector before computing
RMSE / MBE / KGE.  This is deliberately different from ``evaluate.py``, which
reports per-site metrics and then takes a median across sites: pooling lets the
long-record sites carry proportional weight, so an arm cannot win by edging out
a large number of short records.

The two arms differ only in transpiration physics:

    arm A (current)   kc_act = fc*Ks*Kcb + Ke,  Kcb = kc_max * sigmoid(NDVI)
    arm B (fao56_std) kc_act =    Ks*Kcb + Ke,  Kcb = ndvi_beta*NDVI + ndvi_alpha

Both arms are re-run in forecast mode from their own posterior parameter file
and their own project TOML, so each is evaluated under the physics it was
calibrated with.

The paired-day mask requires flux, arm A, and arm B all finite on the same day.
That is a pure arm-vs-arm basis and does NOT require OpenET finiteness, so the
day count runs slightly higher than ``evaluate.py``'s ensemble-paired basis and
the numbers here will not reconcile exactly with the canonical Run 22 table.
Both bases are reported so the difference is auditable.

Usage:
    python pooled_arm_compare.py \
        --a-name run22 --a-config 5_Flux_Ensemble.toml \
        --a-container .../5_Flux_Ensemble_run22.swim \
        --a-par .../results/run22/5_Flux_Ensemble.3.par.csv \
        --b-name RunFAO56 --b-config 5_Flux_Ensemble_fao56.toml \
        --b-container .../5_Flux_Ensemble_RunFAO56.swim \
        --b-par .../results/RunFAO56/5_Flux_Ensemble.3.par.csv \
        --out-dir .../results/RunFAO56/comparison
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from evaluate import (
    apply_exclusions,
    calc_metrics,
    find_par_csv,
    load_flux_et,
    parse_pest_params,
    resolve_flux_dir,
    run_calibrated_model,
)

from swimrs.calibrate.flux_utils import full_month_paired_sums, passes_site_minimum
from swimrs.container import SwimContainer
from swimrs.process.cover_modes import COVER_MODE_NAMES, resolve_cover_mode
from swimrs.process.kcb_modes import KCB_MODE_NAMES, resolve_kcb_mode
from swimrs.swim.config import ProjectConfig

HERE = os.path.dirname(os.path.abspath(__file__))

# Metric direction: how to decide which arm wins.
#   rmse  lower is better
#   bias  closer to zero is better (signed MBE, compared on |.|)
#   kge   higher is better
METRICS = [("rmse", "lower"), ("bias", "abs_lower"), ("kge", "higher")]


def load_arm_config(config_path):
    cfg = ProjectConfig()
    cfg.read_config(config_path, calibrate=True)
    return cfg


def effective_physics(cfg):
    """Resolve the physics actually used, not the raw (possibly absent) TOML keys.

    The Run 22 config predates both switches and omits them, so ``cfg`` reports
    ``None``/``None``.  That is NOT "no physics" — it resolves to the historical
    default (sigmoid Kcb, Eq. 76 cover weight).  Reporting the raw values would
    make an archived comparison unreadable a year from now.
    """
    kcb = KCB_MODE_NAMES[resolve_kcb_mode(getattr(cfg, "kcb_ndvi_mode", None))]
    cover = COVER_MODE_NAMES[
        resolve_cover_mode(
            getattr(cfg, "transpiration_cover_mode", None),
            getattr(cfg, "transpiration_cover_scaling", None),
        )
    ]
    return {"kcb_ndvi_mode": kcb, "transpiration_cover_mode": cover}


def arm_series(cfg, container_path, par_csv, fids):
    """Run one arm in forecast mode; return {fid: daily et_act Series}."""
    container = SwimContainer.open(container_path, mode="r")
    try:
        params = parse_pest_params(par_csv, fids)
        missing = [f for f in fids if f not in params]
        if missing:
            print(f"  WARNING: no calibrated params for {missing}")
        results = run_calibrated_model(cfg, container, fids, params)
    finally:
        container.close()
    return {fid: df["et_act"] for fid, df in results.items()}


def collect(fids, flux_dir, series_a, series_b):
    """Pair both arms against flux on identical days; return pooled + per-site."""
    pooled = {k: [] for k in ("obs", "a", "b")}
    pooled_mo = {k: [] for k in ("obs", "a", "b")}
    per_site = []

    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            print(f"  {fid}: no flux data, skipping")
            continue
        if not passes_site_minimum(flux_et):
            print(f"  {fid}: below VALIDATION_POLICY site minimum, skipping")
            continue
        if fid not in series_a or fid not in series_b:
            print(f"  {fid}: missing in one arm, skipping")
            continue

        a_et, b_et = series_a[fid], series_b[fid]
        common = flux_et.index.intersection(a_et.index).intersection(b_et.index)
        if len(common) < 10:
            print(f"  {fid}: only {len(common)} overlapping days, skipping")
            continue

        obs = flux_et.loc[common].values
        av = a_et.loc[common].values
        bv = b_et.loc[common].values
        mask = np.isfinite(obs) & np.isfinite(av) & np.isfinite(bv)
        if mask.sum() < 10:
            print(f"  {fid}: only {int(mask.sum())} paired days, skipping")
            continue

        pooled["obs"].append(obs[mask])
        pooled["a"].append(av[mask])
        pooled["b"].append(bv[mask])

        row = {"fid": fid, "n_daily": int(mask.sum())}
        for arm, vals in (("a", av), ("b", bv)):
            m = calc_metrics(obs[mask], vals[mask])
            for k in ("rmse", "bias", "kge", "r2"):
                row[f"{k}_{arm}_daily"] = m[k]

        # Monthly: full calendar-month totals gated on nearly-complete flux months.
        # flux_daily is identical for both arms, so the returned flux totals match.
        flux_daily = flux_et.loc[common]
        a_mo, flux_mo = full_month_paired_sums(a_et, flux_daily)
        b_mo, _ = full_month_paired_sums(b_et, flux_daily)
        months = flux_mo.index.intersection(a_mo.index).intersection(b_mo.index)
        o_mo = flux_mo.reindex(months).values
        am = a_mo.reindex(months).values
        bm = b_mo.reindex(months).values
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
    return cat, cat_mo, pd.DataFrame(per_site)


def decide(name_a, name_b, pooled_daily, pooled_monthly):
    """Apply the >=4-of-6 gate. Returns (rows, wins_a, passed)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-name", default="run22")
    ap.add_argument("--a-config", default=os.path.join(HERE, "5_Flux_Ensemble.toml"))
    ap.add_argument("--a-container", required=True)
    ap.add_argument("--a-par", default=None)
    ap.add_argument("--a-results", default=None, help="Dir to auto-find the .par.csv in")
    ap.add_argument("--b-name", default="RunFAO56")
    ap.add_argument("--b-config", default=os.path.join(HERE, "5_Flux_Ensemble_fao56.toml"))
    ap.add_argument("--b-container", required=True)
    ap.add_argument("--b-par", default=None)
    ap.add_argument("--b-results", default=None)
    ap.add_argument("--sites", default=None, help="Comma-separated subset")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg_a = load_arm_config(args.a_config)
    cfg_b = load_arm_config(args.b_config)

    par_a = args.a_par or find_par_csv(args.a_results, cfg_a.project_name)
    par_b = args.b_par or find_par_csv(args.b_results, cfg_b.project_name)
    if par_a is None or par_b is None:
        raise SystemExit(f"Posterior par.csv not found (a={par_a}, b={par_b})")

    phys_a, phys_b = effective_physics(cfg_a), effective_physics(cfg_b)
    print(
        f"arm A {args.a_name}: kcb={phys_a['kcb_ndvi_mode']} "
        f"cover={phys_a['transpiration_cover_mode']}"
    )
    print(
        f"arm B {args.b_name}: kcb={phys_b['kcb_ndvi_mode']} "
        f"cover={phys_b['transpiration_cover_mode']}"
    )
    if phys_a == phys_b:
        raise SystemExit("Both arms resolved to identical physics — check the configs")

    if args.sites:
        fids = args.sites.split(",")
    else:
        container = SwimContainer.open(args.a_container, mode="r")
        try:
            fids = sorted(container.field_uids)
        finally:
            container.close()
    fids = apply_exclusions(fids)

    flux_dir = resolve_flux_dir(cfg_a)
    print(f"\nRunning arm A ({args.a_name}) forward on {len(fids)} sites...")
    series_a = arm_series(cfg_a, args.a_container, par_a, fids)
    print(f"Running arm B ({args.b_name}) forward on {len(fids)} sites...")
    series_b = arm_series(cfg_b, args.b_container, par_b, fids)

    print("\nPairing...")
    pooled_daily, pooled_monthly, per_site = collect(fids, flux_dir, series_a, series_b)

    table, wins_a, passed = decide(args.a_name, args.b_name, pooled_daily, pooled_monthly)

    os.makedirs(args.out_dir, exist_ok=True)
    per_site.to_csv(os.path.join(args.out_dir, "pooled_per_site.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, "pooled_gate.csv"), index=False)

    print("\n" + "=" * 84)
    print(f"POOLED TWO-ARM COMPARISON — {len(per_site)} sites")
    print(f"  daily site-days pooled: {len(pooled_daily['obs']):,}")
    print(f"  monthly totals pooled : {len(pooled_monthly['obs']):,}")
    print("=" * 84)
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
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
                "par_a": par_a,
                "par_b": par_b,
                "n_sites": int(len(per_site)),
                "n_daily": int(len(pooled_daily["obs"])),
                "n_monthly": int(len(pooled_monthly["obs"])),
                "wins_a": int(wins_a),
                "gate_rule": "arm A wins >= 4 of 6 pooled metrics",
                "passed": bool(passed),
                "metrics": table.to_dict(orient="records"),
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {args.out_dir}/pooled_gate.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

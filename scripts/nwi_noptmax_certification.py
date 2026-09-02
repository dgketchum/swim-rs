"""Certify noptmax 2 against noptmax 3 from archived PEST++-IES trajectories.

A noptmax-2 run is the first two iterations of a noptmax-3 run: IES iterations
are sequential, the .pst carries no terminal-averaging option, and prior-data
conflicts are dropped once up front. So the archived ``*.2.par.csv`` *is* the
posterior a noptmax-2 run would have terminated with, for the same seed and the
same prior ensemble. Comparing it against ``*.3.par.csv`` is a paired
experiment; rerunning at noptmax 2 would confound the effect with a new seed.

Emits one row per (label, batch, fid, pargp) and prints the aggregate verdict.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIL_TOL = 0.01  # within 1% of a bound, in the parameter's own transform space
MATERIAL = 0.05  # median shift worth caring about, as a fraction of bound range

_PAR_ITER_RE = re.compile(r"\.(\d+)\.par\.csv$")


def _iter_par_files(batch_dir):
    """Map iteration number -> par.csv path for one archived batch."""
    out = {}
    for path in batch_dir.glob("*.par.csv"):
        m = _PAR_ITER_RE.search(path.name)
        if m:
            out[int(m.group(1))] = path
    return out


def _fid_of(parnme, pargp):
    """pname:p_aw_nv_690_:0_ptype:... + pargp 'aw' -> 'nv_690'."""
    head = parnme.split("_:", 1)[0]
    head = head.split("pname:p_", 1)[-1]
    prefix = f"{pargp}_"
    return head[len(prefix) :] if head.startswith(prefix) else head


def _positions(values, lb, ub, log):
    """Normalize parameter values to [0, 1] within their bounds."""
    values, lb, ub = np.asarray(values, float), np.asarray(lb, float), np.asarray(ub, float)
    if log:
        # log-transformed parameters are estimated in log space, so a "5% of
        # range" shift means 5% of the log range, not of the linear range.
        with np.errstate(divide="ignore", invalid="ignore"):
            values, lb, ub = np.log10(values), np.log10(lb), np.log10(ub)
    span = np.where(ub - lb == 0, np.nan, ub - lb)
    return (values - lb) / span


def certify_batch(batch_dir, label):
    """Return (per-parameter frame, phi record) for one archived batch."""
    pars = _iter_par_files(batch_dir)
    if len(pars) < 2:
        return None, None
    final = max(pars)
    if final - 1 not in pars:
        return None, None

    pdata = pd.read_csv(batch_dir / f"{label}.par_data.csv").set_index("parnme")
    short = pd.read_csv(pars[final - 1], index_col=0)
    full = pd.read_csv(pars[final], index_col=0)
    shared = [c for c in full.columns if c in short.columns and c in pdata.index]

    pdata = pdata.loc[shared]
    is_log = (pdata["partrans"] == "log").to_numpy()
    lb, ub = pdata["parlbnd"].to_numpy(), pdata["parubnd"].to_numpy()

    med_short = short[shared].median().to_numpy()
    med_full = full[shared].median().to_numpy()

    pos_short = np.where(
        is_log,
        _positions(med_short, lb, ub, True),
        _positions(med_short, lb, ub, False),
    )
    pos_full = np.where(
        is_log,
        _positions(med_full, lb, ub, True),
        _positions(med_full, lb, ub, False),
    )

    pargp = pdata["pargp"].to_numpy()
    frame = pd.DataFrame(
        {
            "label": label,
            "batch": batch_dir.name,
            "parnme": shared,
            "pargp": pargp,
            "fid": [_fid_of(p, g) for p, g in zip(shared, pargp)],
            "median_short": med_short,
            "median_full": med_full,
            "pos_short": pos_short,
            "pos_full": pos_full,
            "shift": np.abs(pos_full - pos_short),
            "rail_short": (np.minimum(pos_short, 1 - pos_short) <= RAIL_TOL),
            "rail_full": (np.minimum(pos_full, 1 - pos_full) <= RAIL_TOL),
        }
    )

    phi_path = batch_dir / f"{label}.phi.actual.csv"
    phi = None
    if phi_path.exists():
        pf = pd.read_csv(phi_path).set_index("iteration")
        if final in pf.index and final - 1 in pf.index:
            a, b = float(pf.loc[final - 1, "mean"]), float(pf.loc[final, "mean"])
            phi = {
                "label": label,
                "batch": batch_dir.name,
                "n_par": len(shared),
                "n_fid": frame["fid"].nunique(),
                "iter_short": final - 1,
                "iter_full": final,
                "phi_short": a,
                "phi_full": b,
                "phi_gain_pct": 100.0 * (a - b) / a if a else np.nan,
                "runs_short": int(pf.loc[final - 1, "total_runs"]),
                "runs_full": int(pf.loc[final, "total_runs"]),
            }
    return frame, phi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="/project/handily/swim/nwi")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames, phis = [], []
    for label in args.labels:
        archive = Path(args.run_root) / label / "pestrun" / "pest_archive"
        for batch_dir in sorted(archive.glob("batch_*")):
            frame, phi = certify_batch(batch_dir, label)
            if frame is None:
                print(f"skip {label}/{batch_dir.name}: no consecutive iteration pair")
                continue
            frames.append(frame)
            if phi:
                phis.append(phi)

    if not frames:
        raise SystemExit("no archived batches with a consecutive iteration pair")

    par = pd.concat(frames, ignore_index=True)
    phi = pd.DataFrame(phis)
    out = Path(args.out)
    par.to_csv(out, index=False)
    phi.to_csv(out.with_name(out.stem + "_phi.csv"), index=False)

    print(f"\n=== batches {len(phi)} | fields {par['fid'].nunique()} | parameters {len(par)} ===")

    print("\n--- phi: what the final iteration buys ---")
    print(f"mean gain   {phi['phi_gain_pct'].mean():6.2f}%")
    print(f"median gain {phi['phi_gain_pct'].median():6.2f}%")
    print(f"range       {phi['phi_gain_pct'].min():6.2f}% .. {phi['phi_gain_pct'].max():6.2f}%")
    cost = 100.0 * (phi["runs_full"] - phi["runs_short"]).sum() / phi["runs_full"].sum()
    print(f"cost of that iteration: {cost:.1f}% of all model runs")

    print("\n--- posterior median shift (fraction of bound range) ---")
    print(f"mean  {par['shift'].mean():.4f}   median {par['shift'].median():.4f}")
    print(f"p90   {par['shift'].quantile(0.90):.4f}   p99 {par['shift'].quantile(0.99):.4f}")
    print(f"max   {par['shift'].max():.4f}")
    n_mat = int((par["shift"] > MATERIAL).sum())
    print(
        f"parameters shifting >{MATERIAL:.0%} of range: {n_mat} / {len(par)} ({100 * n_mat / len(par):.2f}%)"
    )

    per_field = par.groupby(["label", "batch", "fid"])["shift"].max()
    n_field = int((per_field > MATERIAL).sum())
    print(
        f"fields with any such parameter:      {n_field} / {len(per_field)} ({100 * n_field / len(per_field):.2f}%)"
    )

    print("\n--- by parameter group ---")
    grp = par.groupby("pargp").agg(
        n=("shift", "size"),
        mean_shift=("shift", "mean"),
        p90_shift=("shift", lambda s: s.quantile(0.90)),
        max_shift=("shift", "max"),
        rail_short=("rail_short", "mean"),
        rail_full=("rail_full", "mean"),
    )
    grp["rail_delta"] = grp["rail_full"] - grp["rail_short"]
    print(grp.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- rail census (median within 1% of a bound) ---")
    print(f"noptmax {phi['iter_short'].iloc[0]}: {par['rail_short'].mean():.4%}")
    print(f"noptmax {phi['iter_full'].iloc[0]}: {par['rail_full'].mean():.4%}")
    newly = int((par["rail_full"] & ~par["rail_short"]).sum())
    freed = int((par["rail_short"] & ~par["rail_full"]).sum())
    print(f"newly railed by the final iteration: {newly}   freed: {freed}")

    print(f"\nwrote {out} and {out.with_name(out.stem + '_phi.csv')}")


if __name__ == "__main__":
    main()

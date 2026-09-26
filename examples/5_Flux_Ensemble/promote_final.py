"""Promote the Example 5 supporting analyses into ``paper/data/final`` (paper E1).

The headline E1 benchmark package (``paper/data/final/e1_openet_benchmark/``) is
written by ``evaluate.py`` and ``overpass_decomposition.py`` and frozen by
``rebuild_e1_benchmark_evidence.py``. The supporting products below used to be
copied into ``paper/data/final`` by hand; this script is their tracked producer.

<run> is the canonical run results dir (``ex5_paths.CANONICAL_RUN``); every
source lives under it so each run carries its own supporting products.

    <run>/spread_error/spread_error_{persite,quintiles,summary}.csv
        -> e2_spread_error_{persite,quintiles,summary}.csv          (byte copy; Fig. 4)
    <run>/within_e2_transfer/persite_{daily,monthly}.csv
        -> e2_within_transfer_{daily,monthly}_site_metrics.csv       (byte copy)
    <run>/within_e2_transfer_irrigation_stratified/summary_metrics.csv
        -> e2_irrigation_stratified_transfer_summary.csv             (adds the experiment column; Table S7, Fig. 5a)
    <run>/within_e2_transfer_irrigation_stratified/{transfer_vectors.json, class_fold_support.csv}
      + <run>/archive/3_problem_definition/parameter_bounds.csv
        -> e2_irrigation_stratified_fold_mad_domain.csv              (derived; Fig. 5a mad legality)

The ``e2_*`` names are the legacy namespace paper E1 evidence is frozen under;
the figure builder and the frozen hashes in ``e2_evidence_metadata.json`` key on
them, so they are kept. The default mode rebuilds every product in memory and
compares it byte-for-byte with the promoted file; ``--write`` replaces them.

Usage:
    uv run python examples/5_Flux_Ensemble/promote_final.py            # check
    uv run python examples/5_Flux_Ensemble/promote_final.py --write
"""

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ex5_paths  # noqa: E402

# Legacy experiment label carried by the frozen summary (paper E1).
EXPERIMENT_LABEL = "E2_within_held_out"
# Both relative to the run results dir (``ex5_paths.run_dir(CANONICAL_RUN)``).
STRATIFIED_DIR = "within_e2_transfer_irrigation_stratified"
# TODO(run23): no current script writes <run>/within_e2_transfer/persite_*.csv
# (the Run 22 copies at {project_ws}/results/within_e2_transfer/ predate
# within_e1_transfer.py's six-arm output); regenerate or derive before promoting.
POOLED_DIR = "within_e2_transfer"

FOLD_MAD_COLUMNS = [
    "arm",
    "fid",
    "region",
    "irr_class",
    "mad",
    "prior_lo",
    "prior_hi",
    "mad_in_class_prior",
    "kr_alpha",
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def stratified_summary(summary):
    """``summary_metrics.csv`` with the experiment label as the first column."""
    out = summary.copy()
    out.insert(0, "experiment", EXPERIMENT_LABEL)
    return out


def fold_mad_domain(vectors, support, bounds):
    """Per-arm, per-site ``mad`` legality against the site's own class prior.

    ``vectors``: ``transfer_vectors.json`` ({arm: {fid: {param: value}}}).
    ``support``: ``class_fold_support.csv`` (fid, region, irr_class, ...).
    ``bounds``: the Run 22 ``parameter_bounds.csv`` (param, site, lower_bound, upper_bound);
    the ``mad`` bounds are set per site by the calibration's irrigation status, so they are
    the class prior the transferred value is judged against.
    """
    sup = support.set_index("fid")
    mad_bounds = bounds[bounds["param"] == "mad"].set_index("site")[["lower_bound", "upper_bound"]]
    rows = []
    for arm, by_fid in vectors.items():
        for fid, params in by_fid.items():
            lo, hi = (float(v) for v in mad_bounds.loc[fid])
            mad = float(params["mad"])
            rows.append(
                {
                    "arm": arm,
                    "fid": fid,
                    "region": sup.loc[fid, "region"],
                    "irr_class": sup.loc[fid, "irr_class"],
                    "mad": mad,
                    "prior_lo": lo,
                    "prior_hi": hi,
                    "mad_in_class_prior": bool(lo <= mad <= hi),
                    "kr_alpha": float(params["kr_alpha"]),
                }
            )
    df = pd.DataFrame(rows, columns=FOLD_MAD_COLUMNS)
    classes = df.groupby("irr_class")[["prior_lo", "prior_hi"]].nunique()
    if (classes > 1).any().any():
        raise ValueError(f"mad prior bounds vary within an irrigation class:\n{classes}")
    return df


def csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def build_products(run_dir):
    """Return {final_name: bytes} for every promoted product."""
    run_dir = Path(run_dir)
    strat = run_dir / STRATIFIED_DIR
    pooled = run_dir / POOLED_DIR
    products = {}
    for part in ("persite", "quintiles", "summary"):
        products[f"e2_spread_error_{part}.csv"] = (
            run_dir / "spread_error" / f"spread_error_{part}.csv"
        ).read_bytes()
    for scale in ("daily", "monthly"):
        products[f"e2_within_transfer_{scale}_site_metrics.csv"] = (
            pooled / f"persite_{scale}.csv"
        ).read_bytes()
    products["e2_irrigation_stratified_transfer_summary.csv"] = csv_bytes(
        stratified_summary(pd.read_csv(strat / "summary_metrics.csv"))
    )
    with open(strat / "transfer_vectors.json") as fh:
        vectors = json.load(fh)
    products["e2_irrigation_stratified_fold_mad_domain.csv"] = csv_bytes(
        fold_mad_domain(
            vectors,
            pd.read_csv(strat / "class_fold_support.csv"),
            pd.read_csv(run_dir / "archive" / "3_problem_definition" / "parameter_bounds.csv"),
        )
    )
    return products


def compare(products, final_dir):
    """Return [(name, status, sha256)] where status is 'match', 'differs', or 'missing'."""
    rows = []
    for name, data in products.items():
        target = Path(final_dir) / name
        if not target.exists():
            status = "missing"
        else:
            status = "match" if target.read_bytes() == data else "differs"
        rows.append((name, status, sha256_bytes(data)))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help=f"default: the {ex5_paths.CANONICAL_RUN} results dir (ex5_paths.CANONICAL_RUN)",
    )
    ap.add_argument("--final-dir", default=str(ex5_paths.FINAL_DIR))
    ap.add_argument("--write", action="store_true", help="replace the promoted files")
    args = ap.parse_args()
    if args.run_dir is None:
        args.run_dir = ex5_paths.run_dir(cfg=ex5_paths.load_config())

    products = build_products(args.run_dir)
    rows = compare(products, args.final_dir)
    for name, status, digest in rows:
        print(f"{status:8s} {digest[:16]}  {name}")
    if args.write:
        os.makedirs(args.final_dir, exist_ok=True)
        for name, data in products.items():
            (Path(args.final_dir) / name).write_bytes(data)
        print(f"wrote {len(products)} files to {args.final_dir}")
    elif any(status != "match" for _, status, _ in rows):
        print("promoted files differ from the rebuilt products (run with --write to replace)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

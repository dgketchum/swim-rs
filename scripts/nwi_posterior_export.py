"""Export per-field posterior parameters and their distributions for an NWI run.

Reads the PEST++ IES posterior ensembles archived by the batch runner
(``{pest_run_dir}/pest_archive/batch_*/``, best-phi iteration per batch) and
the summary values ingested into the container's ``calibration/`` group, and
writes:

    posterior_ensemble_long.parquet   realization x field x parameter values
                                      (the full posterior distributions)
    posterior_field_distributions.csv per field x parameter: median, mean,
                                      std, q25, q75, iqr, cv, n_reals
    field_parameters_ingested.csv     per field: the container's ingested
                                      center (median) and std per parameter
    calibration.{json,html,png}       container calibration report

Existing output directories are never overwritten; a timestamped sibling is
used instead.

Usage:
    uv run python scripts/nwi_posterior_export.py --config /path/to/32009.toml
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.container import open_container
from swimrs.container.components.ingestor import _PEST_NAME_MAP, CALIBRATION_PARAMS
from swimrs.swim.config import ProjectConfig

_ITER_RE = re.compile(r"\.(\d+)\.par\.csv$")


def _best_par_csv(archive_dir: Path, max_iteration: int | None = None) -> Path | None:
    """Best-phi iteration .par.csv from an archived batch (min mean measured phi).

    ``max_iteration`` caps the iterations considered, matching
    ``batch_support.find_par_csv``. IES iterations are sequential, so a
    ``noptmax N`` archive restricted to iterations <= M is exactly what a
    ``noptmax M`` run would have produced — which is what makes the exported
    ensemble describe the same iteration the container ingested. Without the
    cap an archive from a longer run exports iteration 3 spread alongside
    iteration 2 medians.
    """
    par_files = [f for f in archive_dir.glob("*.par.csv") if _ITER_RE.search(f.name)]
    if max_iteration is not None:
        par_files = [f for f in par_files if int(_ITER_RE.search(f.name).group(1)) <= max_iteration]
    if not par_files:
        return None
    phi_files = list(archive_dir.glob("*.phi.meas.csv"))
    if phi_files:
        try:
            phi = pd.read_csv(phi_files[0]).dropna(subset=["mean"])
            if max_iteration is not None:
                phi = phi.loc[phi["iteration"] <= max_iteration]
            best = int(phi.loc[phi["mean"].idxmin(), "iteration"])
            for f in par_files:
                if int(_ITER_RE.search(f.name).group(1)) == best:
                    return f
        except Exception:
            pass
    return max(par_files, key=lambda f: int(_ITER_RE.search(f.name).group(1)))


def _melt_par_csv(par_csv: Path, fids: list[str], batch: str) -> pd.DataFrame:
    """Long-format posterior ensemble: one row per realization x field x parameter."""
    df = pd.read_csv(par_csv, index_col=0)
    df = df.loc[df.index != "base"]

    records = []
    for col in df.columns:
        parts = col.split("_ptype:")[0].replace("pname:p_", "").rsplit("_:0", 1)[0]
        matched_fid, param = None, None
        for fid in fids:
            if parts.lower().endswith(f"_{fid.lower()}"):
                matched_fid = fid
                param = _PEST_NAME_MAP.get(parts[: -(len(fid) + 1)], parts[: -(len(fid) + 1)])
                break
        if matched_fid is None or param not in CALIBRATION_PARAMS:
            continue
        records.append(
            pd.DataFrame(
                {
                    "realization": df.index,
                    "fid": matched_fid,
                    "param": param,
                    "value": df[col].values,
                    "batch": batch,
                }
            )
        )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="Default: {pest_run_dir}/posterior")
    ap.add_argument(
        "--max-iteration",
        type=int,
        default=None,
        help=(
            "cap the archived IES iteration exported, so the ensemble matches a "
            "container ingested with the same cap (e.g. 2 for noptmax 2)"
        ),
    )
    args = ap.parse_args()

    config = ProjectConfig()
    config.read_config(args.config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(config.pest_run_dir) / "posterior"
    if out_dir.exists():
        out_dir = out_dir.with_name(f"{out_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True)
    print(f"Writing posterior exports to {out_dir}")

    c = open_container(config.container_path, mode="r")
    fids = c.field_uids

    # Container-ingested center/std per field
    rows = {"fid": fids}
    for param in CALIBRATION_PARAMS:
        vpath = f"calibration/parameters/{param}"
        spath = f"calibration/uncertainty/{param}"
        if vpath in c._root:
            rows[param] = np.array(c._root[vpath])
        if spath in c._root:
            rows[f"{param}_std"] = np.array(c._root[spath])
    ingested = pd.DataFrame(rows)
    ingested.to_csv(out_dir / "field_parameters_ingested.csv", index=False)
    print(f"  field_parameters_ingested.csv: {len(ingested)} fields")
    c.close()

    # Full posterior ensembles from the pest archive
    archive_root = Path(config.pest_run_dir) / "pest_archive"
    frames = []
    for batch_dir in sorted(archive_root.glob("batch_*")):
        par_csv = _best_par_csv(batch_dir, max_iteration=args.max_iteration)
        if par_csv is None:
            print(f"  WARNING: no .par.csv in {batch_dir}")
            continue
        frames.append(_melt_par_csv(par_csv, fids, batch_dir.name))
        print(f"  {batch_dir.name}: {par_csv.name}")

    if not frames:
        print(f"No archived ensembles found under {archive_root}")
        return 1

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_parquet(out_dir / "posterior_ensemble_long.parquet", index=False)
    print(
        f"  posterior_ensemble_long.parquet: {len(long_df):,} rows "
        f"({long_df['fid'].nunique()} fields x {long_df['param'].nunique()} params)"
    )

    g = long_df.groupby(["fid", "param"])["value"]
    stats = g.agg(
        median="median",
        mean="mean",
        std="std",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        n_reals="count",
    ).reset_index()
    stats["iqr"] = stats["q75"] - stats["q25"]
    stats["cv"] = stats["std"] / stats["mean"].abs()
    stats.to_csv(out_dir / "posterior_field_distributions.csv", index=False)
    print(f"  posterior_field_distributions.csv: {len(stats)} field x param rows")

    # Container calibration report (json/html/png)
    c = open_container(config.container_path, mode="r")
    try:
        c.calibration_report(output_dir=str(out_dir))
        print("  calibration report written")
    except Exception as e:
        print(f"  WARNING: calibration report failed: {e}")
    finally:
        c.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

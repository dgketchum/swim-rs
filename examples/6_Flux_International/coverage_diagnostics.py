"""E3 Experiment C: does the ECOSTRESS JET treatment actually densify the ETf record?

Quantifies the *observation coverage* of the additional-date treatment relative to
the Landsat-only control. This is a sampling diagnostic on the calibration inputs
only: no flux benchmark is involved, no retrieval is scored for accuracy, and
nothing here compares ECOSTRESS and Landsat *quality*. The paired calibration
outcome lives in ``expc_paired_deltas.py``.

Reads the treatment container (which holds the Landsat control footing plus
``remote_sensing/etf/ecostress/ptjpl/no_mask``) and restricts to the publication
cohort named by the config's ``paths.fields_shapefile``. The container stores more
fields than the cohort, so the restriction is mandatory: on the 66-site cohort the
counts are 11,868 / 10,223 / 53 sites, whereas the full 75 stored fields give
12,989 / 11,208 / 59.

The zarr path says ``ecostress/ptjpl`` but the ingested product is
``ECO_L3T_JET.002_ETdaily`` -- the Collection 2 JET ensemble, whose ``ETdaily`` is
the median of PT-JPL-SM, STIC, MOD16 and BESS. The path name is a misnomer; see
``notes/etf_sensor_characterization/ETF_SENSOR_CHARACTERIZATION.md``.

PINNED DEFINITIONS
------------------
Two quantities are convention-dependent, and reasonable alternatives move them
materially. Both conventions are computed and emitted side by side so the choice
is auditable rather than implicit. The ``resample``/``interior`` pair is what the
manuscript quotes.

1. Empty ``WINDOW_DAYS``-day windows.
   - ``resample``: ``pd.Series.resample("16D")`` anchored at the first era day,
     which keeps a partial trailing bin. 11,220 windows, 2,907 Landsat-empty,
     814 rescued = 28.0%.
   - ``floor``: whole windows only, dropping the trailing stub. 11,154 windows,
     2,863 Landsat-empty, 814 rescued = 28.4%.
   The rescued numerator is identical; only the denominator moves.

2. Longest observation-free stretch per site-year.
   - ``interior``: gaps *between* consecutive captures, ``np.diff(idx).max()``,
     over site-years with >= 2 captures. Mean 38.3 -> 33.1 d, p90 64 -> 54 d.
   - ``edges``: additionally counts the distance from Jan 1 to the first capture
     and from the last capture to Dec 31. Mean 50.8 -> 42.6 d, p90 83 -> 71 d.

Usage:
    uv run python examples/6_Flux_International/coverage_diagnostics.py \
        --config examples/6_Flux_International/6_Flux_International_LSEnsemble_ECOSTRESSAddDates_POR_annual2yr.toml
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from swimrs.container import open_container
from swimrs.swim.config import ProjectConfig

WINDOW_DAYS = 16
ERA_START = "2018-08-01"

ETF_TEMPLATE = "remote_sensing/etf/{}/no_mask"
PRIMARY_SERIES = ("landsat/ssebop", "landsat/ptjpl")
AUXILIARY_SERIES = "ecostress/ptjpl"

SUMMARY_CSV = "coverage_diagnostics_summary.csv"
PERSITE_CSV = "coverage_diagnostics_persite.csv"


def _load_config(config_path=None):
    project_dir = Path(__file__).resolve().parent
    conf = Path(config_path) if config_path else project_dir / "6_Flux_International.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)
    return cfg, conf


def load_masks(cfg, container_path=None, era_start=ERA_START):
    """Boolean capture masks (time x cohort field) for the primary and auxiliary records.

    Returns ``(time_index, fids, primary, auxiliary)`` where ``primary`` is
    "either Landsat member retrieved" and ``auxiliary`` is "JET retrieved".
    """
    path = container_path or cfg.container_path
    container = open_container(path)
    root = container._root

    uids = [str(u) for u in container.field_uids]
    id_col = cfg.feature_id_col or "sid"
    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    cohort = set(gdf[id_col].astype(str))
    keep = np.array([u in cohort for u in uids])
    if not keep.any():
        raise ValueError(
            f"No container field matched {cfg.fields_shapefile} column {id_col!r}; "
            "cohort restriction would silently drop every site."
        )
    fids = [u for u, k in zip(uids, keep) if k]

    time = pd.DatetimeIndex(container.state._time_index)
    era = time >= pd.Timestamp(era_start)

    def read(series):
        arr = np.array(root[ETF_TEMPLATE.format(series)][:])
        return np.isfinite(arr[np.ix_(era, keep)])

    primary = np.zeros((int(era.sum()), len(fids)), dtype=bool)
    for series in PRIMARY_SERIES:
        primary |= read(series)
    auxiliary = read(AUXILIARY_SERIES)
    return time[era], fids, primary, auxiliary


def capture_counts(primary, auxiliary):
    """Auxiliary capture totals and the activated (primary-free) subset."""
    aux_total = int(auxiliary.sum())
    aux_only = int((auxiliary & ~primary).sum())
    overlap = int((auxiliary & primary).sum())
    with_aux = int(auxiliary.any(axis=0).sum())
    return {
        "aux_captures": aux_total,
        "aux_activated": aux_only,
        "aux_activated_frac": aux_only / aux_total if aux_total else np.nan,
        "aux_excluded_overlap": overlap,
        "primary_captures": int(primary.sum()),
        "sites_with_aux": with_aux,
        "sites_without_aux": int(auxiliary.shape[1] - with_aux),
    }


def window_coverage(time, primary, auxiliary, window_days=WINDOW_DAYS, convention="resample"):
    """Empty-window counts before and after adding the auxiliary record.

    See PINNED DEFINITIONS for the two conventions.
    """
    total = empty_primary = empty_joint = rescued = 0
    for col in range(primary.shape[1]):
        if convention == "resample":
            p = pd.Series(primary[:, col].astype(int), index=time)
            a = pd.Series(auxiliary[:, col].astype(int), index=time)
            pw = p.resample(f"{window_days}D").sum().values
            aw = a.resample(f"{window_days}D").sum().values
        elif convention == "floor":
            n = len(time) // window_days
            pw = primary[: n * window_days, col].reshape(n, window_days).sum(axis=1)
            aw = auxiliary[: n * window_days, col].reshape(n, window_days).sum(axis=1)
        else:
            raise ValueError(f"unknown window convention {convention!r}")
        total += len(pw)
        empty_primary += int((pw == 0).sum())
        empty_joint += int(((pw == 0) & (aw == 0)).sum())
        rescued += int(((pw == 0) & (aw > 0)).sum())
    return {
        "window_convention": convention,
        "windows_total": total,
        "windows_empty_primary": empty_primary,
        "windows_empty_joint": empty_joint,
        "windows_rescued": rescued,
        "windows_rescued_frac": rescued / empty_primary if empty_primary else np.nan,
        "empty_rate_primary": empty_primary / total if total else np.nan,
        "empty_rate_joint": empty_joint / total if total else np.nan,
    }


def _longest_runs(mask, time, convention):
    """Longest observation-free stretch (days) per site-year."""
    out = []
    years = time.year
    for col in range(mask.shape[1]):
        for year in np.unique(years):
            m = mask[years == year, col]
            idx = np.flatnonzero(m)
            if convention == "interior":
                if len(idx) < 2:
                    continue
                out.append(int(np.diff(idx).max()))
            elif convention == "edges":
                if len(idx) == 0:
                    continue
                runs = np.diff(idx) - 1 if len(idx) > 1 else np.array([0])
                out.append(int(max(runs.max(), idx[0], len(m) - 1 - idx[-1])))
            else:
                raise ValueError(f"unknown gap convention {convention!r}")
    return np.array(out)


def longest_gap(time, primary, auxiliary, convention="interior"):
    """Longest-gap distribution for the control and the treatment record."""
    rows = []
    for label, mask in (("primary_only", primary), ("primary_plus_aux", primary | auxiliary)):
        runs = _longest_runs(mask, time, convention)
        rows.append(
            {
                "gap_convention": convention,
                "record": label,
                "n_site_years": len(runs),
                "mean": float(runs.mean()) if len(runs) else np.nan,
                "median": float(np.median(runs)) if len(runs) else np.nan,
                "p90": float(np.percentile(runs, 90)) if len(runs) else np.nan,
                "max": int(runs.max()) if len(runs) else -1,
            }
        )
    return rows


def per_site_table(fids, primary, auxiliary):
    """Per-site capture counts, for auditing the treated/control split."""
    rows = []
    for j, fid in enumerate(fids):
        aux = auxiliary[:, j]
        pri = primary[:, j]
        rows.append(
            {
                "fid": fid,
                "primary_captures": int(pri.sum()),
                "aux_captures": int(aux.sum()),
                "aux_activated": int((aux & ~pri).sum()),
                "aux_excluded_overlap": int((aux & pri).sum()),
                "group": "treated" if aux.any() else "control",
            }
        )
    return pd.DataFrame(rows)


def cohort_capture_split(cfg, container_path=None, era_start=ERA_START):
    """``(treated, control)`` cohort fid lists by presence of any JET capture.

    Shared with ``expc_paired_deltas.py`` so the group sizes reported alongside
    the paired deltas are the same 53/13 as the coverage numbers.
    """
    _, fids, primary, auxiliary = load_masks(cfg, container_path, era_start)
    table = per_site_table(fids, primary, auxiliary)
    treated = table.loc[table["group"] == "treated", "fid"].tolist()
    control = table.loc[table["group"] == "control", "fid"].tolist()
    return treated, control


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="project TOML (AddDates treatment)")
    parser.add_argument("--container", default=None, help="override the container path")
    parser.add_argument("--out", default=None, help="output dir (default: results/<toml stem>)")
    parser.add_argument("--era-start", default=ERA_START, help="first day of the JET era")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    args = parser.parse_args()

    cfg, conf = _load_config(args.config)
    out_dir = args.out or os.path.join(cfg.project_ws, "results", conf.stem)
    os.makedirs(out_dir, exist_ok=True)

    time, fids, primary, auxiliary = load_masks(cfg, args.container, args.era_start)
    print(f"container : {args.container or cfg.container_path}")
    print(f"cohort    : {len(fids)} fields from {os.path.basename(cfg.fields_shapefile)}")
    print(f"era       : {time[0].date()} to {time[-1].date()} ({len(time)} days)")

    counts = capture_counts(primary, auxiliary)
    print("\n--- capture counts (cohort-restricted) ---")
    for k, v in counts.items():
        print(f"  {k:24s} {v:,.4g}" if isinstance(v, float) else f"  {k:24s} {v:,}")

    print(f"\n--- {args.window_days}-day window coverage ---")
    windows = [
        window_coverage(time, primary, auxiliary, args.window_days, c)
        for c in ("resample", "floor")
    ]
    print(pd.DataFrame(windows).to_string(index=False))

    print("\n--- longest observation-free stretch per site-year (days) ---")
    gaps = [r for c in ("interior", "edges") for r in longest_gap(time, primary, auxiliary, c)]
    print(pd.DataFrame(gaps).to_string(index=False))

    records = [{"metric": k, "value": v} for k, v in counts.items()]
    for w in windows:
        conv = w["window_convention"]
        records += [
            {"metric": f"{k}[{conv}]", "value": v} for k, v in w.items() if k != "window_convention"
        ]
    for g in gaps:
        stub = f"longest_gap[{g['gap_convention']}][{g['record']}]"
        records += [
            {"metric": f"{stub}.{k}", "value": g[k]}
            for k in ("n_site_years", "mean", "median", "p90", "max")
        ]
    summary = pd.DataFrame(records)
    summary.to_csv(os.path.join(out_dir, SUMMARY_CSV), index=False)

    table = per_site_table(fids, primary, auxiliary)
    table.to_csv(os.path.join(out_dir, PERSITE_CSV), index=False)
    print(f"\nwrote {os.path.join(out_dir, SUMMARY_CSV)}")
    print(f"wrote {os.path.join(out_dir, PERSITE_CSV)}")


if __name__ == "__main__":
    main()

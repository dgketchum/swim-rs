"""Completeness check for an Example 10 run container (run before calibrating).

Per field: finite meteorology (raw and corrected), SNODAS SWE, seasonal NDVI
coverage per instrument, at least one finite ETf capture per LST member, soils
and land-cover properties, and the water-balance irrigation classification.
Nothing is filled or dropped; incomplete fields are reported.

    uv run python examples/10_OR_Irrigation_Test/check_container.py --run e10cal
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from ex10_paths import LST_MODELS, load_config

from swimrs.container import open_container

MET_VARS = ("prcp", "tmax", "tmin", "srad", "ea", "u2", "eto", "etr", "eto_corr", "etr_corr")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="e10cal")
    args = p.parse_args()

    cfg = load_config()
    path = os.path.join(cfg.data_dir, f"{cfg.project_name}_{args.run}.swim")
    c = open_container(path, mode="r")
    root = c._root
    fids = list(c.field_uids)
    dates = pd.DatetimeIndex(np.array(root["time/daily"]))
    print(
        f"{path}\n{len(fids)} fields, {dates[0].date()} .. {dates[-1].date()} ({len(dates)} days)\n"
    )

    problems = []

    print("=== Meteorology (NaN count per variable, all fields) ===")
    for v in MET_VARS:
        arr = np.array(root[f"meteorology/gridmet/{v}"], dtype=np.float32)
        n_nan = int(np.isnan(arr).sum())
        print(
            f"  {v:9s} nan={n_nan:8d}  mean={np.nanmean(arr):.3f}  min={np.nanmin(arr):.3f}  max={np.nanmax(arr):.3f}"
        )
        if n_nan:
            bad = [fids[i] for i in np.where(np.isnan(arr).any(axis=0))[0]]
            problems.append(f"{v}: {n_nan} NaN cells in {len(bad)} fields")
    prcp = np.array(root["meteorology/gridmet/prcp"])
    annual_p = pd.Series(prcp[:, 0], index=dates).resample("YE").sum()
    print(
        f"  annual precipitation (cell 680670): mean {annual_p.mean():.0f} mm, range {annual_p.min():.0f}..{annual_p.max():.0f}"
    )

    print("\n=== SNODAS SWE ===")
    swe = np.array(root["snow/snodas/swe"], dtype=np.float32)
    first = dates[np.where(np.isfinite(swe).any(axis=1))[0][0]].date()
    per_field = np.isfinite(swe).sum(axis=0)
    print(
        f"  first finite date {first}; finite days per field min={per_field.min()} max={per_field.max()}; max SWE {np.nanmax(swe):.1f} mm"
    )

    print("\n=== NDVI (finite captures per field; growing-season Apr-Sep captures per year) ===")
    for inst in ("landsat", "sentinel"):
        arr = np.array(root[f"remote_sensing/ndvi/{inst}/no_mask"], dtype=np.float32)
        fin = np.isfinite(arr)
        n = fin.sum(axis=0)
        yrs = dates[fin.any(axis=1)].year
        gs_mask = (dates.month >= 4) & (dates.month <= 9)
        gs = pd.DataFrame(fin[gs_mask], index=dates[gs_mask]).groupby(dates[gs_mask].year).sum()
        gs = gs.loc[gs.index >= yrs.min()]
        print(
            f"  {inst:8s} years {yrs.min()}-{yrs.max()}; captures/field min={n.min()} max={n.max()}; "
            f"Apr-Sep captures/field-year min={gs.values.min()} median={np.median(gs.values):.0f}"
        )
        for i, f in enumerate(fids):
            if n[i] == 0:
                problems.append(f"{inst} NDVI: no captures for {f}")
    merged = np.array(root["derived/merged_ndvi/no_mask"], dtype=np.float32)
    print(
        f"  merged   finite/field min={np.isfinite(merged).sum(axis=0).min()} max={np.isfinite(merged).sum(axis=0).max()}"
    )

    print("\n=== ETf per member (finite captures per field) ===")
    rows = []
    for m in LST_MODELS + ("ensemble",):
        arr = np.array(root[f"remote_sensing/etf/landsat/{m}/no_mask"], dtype=np.float32)
        fin = np.isfinite(arr)
        n = fin.sum(axis=0)
        yrs = dates[fin.any(axis=1)].year
        rows.append(
            (m, int(n.min()), int(n.max()), int(yrs.min()), int(yrs.max()), float(np.nanmean(arr)))
        )
        for i, f in enumerate(fids):
            if n[i] == 0:
                problems.append(f"ETf {m}: no captures for {f}")
    for r in rows:
        print(
            f"  {r[0]:9s} captures/field {r[1]:4d}..{r[2]:4d}  years {r[3]}-{r[4]}  mean {r[5]:.3f}"
        )

    print("\n=== Properties ===")
    soils = {k: np.array(root[f"properties/soils/{k}"]) for k in ("awc", "ksat", "clay", "sand")}
    for k, v in soils.items():
        print(f"  {k:5s} nan={int(np.isnan(v).sum())} range {np.nanmin(v):.3f}..{np.nanmax(v):.3f}")
        if np.isnan(v).any():
            problems.append(f"soils {k}: NaN at {[fids[i] for i in np.where(np.isnan(v))[0]]}")
    glc = np.array(root["properties/land_cover/glc10"])
    modis = np.array(root["properties/land_cover/modis_lc"])
    cult = np.array(root["properties/land_cover/cdl_cultivated"])
    print(f"  glc10 classes {dict(zip(*np.unique(glc, return_counts=True)))}")
    print(f"  modis classes {dict(zip(*np.unique(modis, return_counts=True)))}")
    print(f"  cdl_cultivated {dict(zip(*np.unique(cult, return_counts=True)))}")
    cdl = np.array(root["properties/land_cover/cdl"])
    print(
        f"  cdl modal class per field-year, shape {cdl.shape}; classes {sorted(set(cdl.ravel().tolist()))}"
    )

    print(
        "\n=== Annual water balance seen by the classifier (ensemble ETf, interpolated, x eto_corr) ==="
    )
    etf = pd.DataFrame(
        np.array(root["remote_sensing/etf/landsat/ensemble/no_mask"]), index=dates, columns=fids
    )
    eto = pd.DataFrame(np.array(root["meteorology/gridmet/eto_corr"]), index=dates, columns=fids)
    pp = pd.DataFrame(np.array(root["meteorology/gridmet/prcp"]), index=dates, columns=fids)
    eta = etf.interpolate().bfill().ffill() * eto
    yrs_ok = (dates.year >= 1999) & (dates.year <= 2025)
    eta_y = eta[yrs_ok].groupby(dates[yrs_ok].year).sum()
    p_y = pp[yrs_ok].groupby(dates[yrs_ok].year).sum()
    ratio = eta_y / (p_y + 1.0)
    wb = pd.DataFrame(
        {
            "ETa_mm": eta_y.mean().round(0),
            "P_mm": p_y.mean().round(0),
            "ET/P_mean": ratio.mean().round(2),
            "yrs>1.1": (ratio > 1.1).sum(),
            "yrs>1.3": (ratio > 1.3).sum(),
            "ETa_min": eta_y.min().round(0),
            "ETa_max": eta_y.max().round(0),
        }
    )
    print(wb.to_string())

    print("\n=== Water-balance irrigation classification ===")
    irr = [json.loads(s) for s in np.array(root["derived/dynamics/irr_data"])]
    gws = [json.loads(s) for s in np.array(root["derived/dynamics/gwsub_data"])]
    props = {
        k: np.array(root[f"geometry/properties/{k}"])
        for k in ("ITYPE", "srctype", "IRR_EFF", "OWRD", "Acres")
    }
    table = []
    for i, f in enumerate(fids):
        d = irr[i]
        yrs_irr = sorted(
            int(y) for y, v in d.items() if isinstance(v, dict) and v.get("f_irr", 0) > 0
        )
        gw_years = (
            sorted(
                int(y) for y, v in gws[i].items() if isinstance(v, dict) and v.get("f_sub", 0) > 0
            )
            if isinstance(gws[i], dict)
            else []
        )
        table.append(
            dict(
                field=f,
                acres=round(float(props["Acres"][i]), 1),
                ITYPE=props["ITYPE"][i],
                srctype=props["srctype"][i],
                IRR_EFF=props["IRR_EFF"][i],
                OWRD=props["OWRD"][i],
                glc10=int(glc[i]),
                modis=int(modis[i]),
                n_irr_years=len(yrs_irr),
                irr_years=",".join(str(y) for y in yrs_irr) if yrs_irr else "-",
                n_gwsub_years=len(gw_years),
            )
        )
    tdf = pd.DataFrame(table)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(tdf.to_string(index=False))
    sample = irr[0]
    k0 = next(iter(sample))
    print(f"\n  irr_data entry example ({fids[0]}, {k0}): {sample[k0]}")

    c.close()
    print("\n=== Problems ===")
    if problems:
        for pr in problems:
            print("  " + pr)
    else:
        print("  none")


if __name__ == "__main__":
    main()

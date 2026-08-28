"""Generate the RUN_POLICY Category 5 (posterior summaries) archive for E1.

Produces ready-to-use per-site and per-LULC posterior parameter summaries and
boundary-hit rates from the iteration-3 PEST++ IES ensemble, so figures and the
manuscript's posterior-separation claims can be built without re-parsing the raw
ensemble CSV.

Convention: each site's posterior value for a parameter is the **median across
the non-`base` realizations**; LULC group statistics are then taken across those
per-site medians. LULC labels come from the `lc_class` field of the fields
shapefile. Parameter bounds come from the archived Cat 3 `par_data` table.

Usage:
    uv run python examples/4_Flux_Network/archive_posterior_summary.py \
        --par-csv    /data/ssd1/swim/4_Flux_Network/results/julyphysics/4_Flux_Network.3.par.csv \
        --par-data   /data/ssd1/swim/4_Flux_Network/results/julyphysics/archive/3_problem_definition/4_flux_network.par_data.csv \
        --fields     /data/ssd1/swim/4_Flux_Network/data/gis/flux_fields.shp \
        --archive-root /data/ssd1/swim/4_Flux_Network/results/julyphysics/archive \
        --run-name julyphysics
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def _fid_from_parnme(parnme, pargp):
    m = re.search(rf"p_{re.escape(pargp)}_(.+?)_:", parnme)
    return m.group(1).upper() if m else None


def _build_boundary_hit_rates(site_df, run_name):
    """Summarize boundary hits using each site's own parameter bounds."""
    hit_rows = []
    for parameter, sub in site_df.groupby("parameter", sort=True):
        groups = [("ALL", sub), *list(sub.groupby("lc_class", sort=True))]
        for lulc_group, group_df in groups:
            medians = group_df["median"].to_numpy(dtype=float)
            lower = group_df["parlbnd"].to_numpy(dtype=float)
            upper = group_df["parubnd"].to_numpy(dtype=float)
            tolerances = 0.01 * (upper - lower)
            lower_hits = medians <= lower + tolerances
            upper_hits = medians >= upper - tolerances
            bound_sets = group_df[["parlbnd", "parubnd"]].drop_duplicates()
            common_tolerance = (
                float(tolerances[0])
                if len(tolerances) and np.allclose(tolerances, tolerances[0])
                else np.nan
            )
            hit_rows.append(
                {
                    "run_name": run_name,
                    "parameter": parameter,
                    "lulc_group": lulc_group,
                    "n_sites": len(group_df),
                    "lower_hit_count": int(lower_hits.sum()),
                    "upper_hit_count": int(upper_hits.sum()),
                    "lower_hit_rate": float(lower_hits.mean()),
                    "upper_hit_rate": float(upper_hits.mean()),
                    "n_bound_sets": len(bound_sets),
                    "bound_tolerance": common_tolerance,
                    "bound_tolerance_min": float(tolerances.min()),
                    "bound_tolerance_max": float(tolerances.max()),
                }
            )
    return pd.DataFrame(hit_rows)


def build_posterior_summary(par_csv, par_data_csv, fields, archive_root, run_name):
    out = Path(archive_root) / "5_posterior_summaries"
    out.mkdir(parents=True, exist_ok=True)

    par = pd.read_csv(par_csv)
    par = par[par["real_name"].astype(str).str.lower() != "base"]  # drop base

    pd_meta = pd.read_csv(par_data_csv)
    bounds = pd_meta.set_index("parnme")[["parlbnd", "parubnd", "pargp"]]

    g = gpd.read_file(fields, engine="fiona")
    idc = next(c for c in g.columns if c.lower() in ("site_id", "siteid", "fid", "id"))
    g["SID"] = g[idc].astype(str).str.upper()
    lc = g.set_index("SID")["lc_class"]

    # Per-site, per-parameter posterior statistics across realizations.
    site_rows = []
    for col in par.columns:
        if col not in bounds.index:
            continue
        pargp = bounds.at[col, "pargp"]
        fid = _fid_from_parnme(col, pargp)
        if fid is None:
            continue
        v = par[col].to_numpy(dtype=float)
        med, mean, std = np.median(v), np.mean(v), np.std(v)
        q25, q75 = np.percentile(v, [25, 75])
        cv = std / abs(mean) if mean != 0 else np.nan
        site_rows.append(
            {
                "site": fid,
                "parameter": pargp,
                "lc_class": lc.get(fid, ""),
                "parlbnd": float(bounds.at[col, "parlbnd"]),
                "parubnd": float(bounds.at[col, "parubnd"]),
                "median": med,
                "mean": mean,
                "std": std,
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "cv": cv,
            }
        )
    site_df = pd.DataFrame(site_rows)
    site_output_columns = [
        "site",
        "parameter",
        "lc_class",
        "median",
        "mean",
        "std",
        "q25",
        "q75",
        "iqr",
        "cv",
    ]
    site_df[site_output_columns].to_csv(out / "posterior_site_summary.csv", index=False)

    # LULC-grouped summary across per-site posterior medians.
    lulc_df = (
        site_df.groupby(["lc_class", "parameter"])["median"]
        .agg(median="median", mean="mean", std="std", n_sites="count")
        .reset_index()
    )
    lulc_df.to_csv(out / "lulc_grouped_summary.csv", index=False)

    # Boundary-hit rates per parameter, per LULC group (+ ALL), from per-site medians.
    params = sorted(site_df["parameter"].unique())
    hit_df = _build_boundary_hit_rates(site_df, run_name)
    hit_df.to_csv(out / "boundary_hit_rates.csv", index=False)

    # irrigated_grouped_summary.csv: E1 is a multi-LULC ET-validity experiment,
    # not irrigation-partitioned. Emit the artifact with an explicit note so the
    # archive is complete and the non-applicability is documented, not silent.
    note = site_df.groupby("parameter")["median"].agg(
        median="median", mean="mean", std="std", n_sites="count"
    )
    note.insert(0, "group", "not_applicable_e1_not_irrigation_partitioned")
    note.reset_index().to_csv(out / "irrigated_grouped_summary.csv", index=False)

    # Console flag: any LULC group with a boundary-seeking parameter (>50%).
    flagged = hit_df[
        (hit_df["lulc_group"] != "ALL")
        & ((hit_df["lower_hit_rate"] > 0.5) | (hit_df["upper_hit_rate"] > 0.5))
    ]
    print(f"  Cat 5: {len(site_df)} site-parameter summaries, {len(params)} params")
    if len(flagged):
        print("  boundary-seeking (>50% of a LULC group at a bound):")
        for _, r in flagged.iterrows():
            print(
                f"    {r['parameter']:<10} {r['lulc_group']:<18} "
                f"lower={r['lower_hit_rate']:.2f} upper={r['upper_hit_rate']:.2f} (n={r['n_sites']})"
            )
    # Cropland/grass/shrub headline medians (used in the manuscript posterior text).
    for p in ["aw", "ndvi_0", "mad"]:
        row = lulc_df[lulc_df["parameter"] == p].set_index("lc_class")["median"]
        vals = {
            c: round(row.get(c, float("nan")), 3) for c in ["Croplands", "Grasslands", "Shrublands"]
        }
        print(f"  {p}: {vals}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--par-csv", required=True)
    ap.add_argument("--par-data", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--archive-root", required=True)
    ap.add_argument("--run-name", default="julyphysics")
    args = ap.parse_args()
    out = build_posterior_summary(
        args.par_csv, args.par_data, args.fields, args.archive_root, args.run_name
    )
    print(f"Cat 5 posterior summaries written -> {out}")


if __name__ == "__main__":
    main()

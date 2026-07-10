"""WP-B1 step 1: parse AmeriFlux BASE HR files at Mead into daily observed ET + theta.

The QAQC daily flux CSVs used by the first-pass Mead work carry theta but NO
LE/ET (verified in WP-A0), and the FLUXNET2015 ET ends pre-2016. So for the
OpenET-era (2016+) theta+ET overlap we go straight to the raw AmeriFlux BASE_HR
files, which carry both LE (LE_1_1_1, LE_PI_F_1_1_1) and multi-depth/position SWC
through 2024.

Outputs (per site) -> /data/ssd1/swim/soil_moisture/mead_base/US-Ne{1,2,3}_daily.parquet
with columns:
  et_le_raw     daily ET (mm) from raw hourly LE, valid-fraction gated
  et_le_pif     daily ET (mm) from gap-filled LE_PI_F (continuous)
  et_le_corr    daily ET (mm) energy-balance-closure corrected (Bowen-ratio),
                mirrors the QAQC ET_corr convention
  closure_frac  daily (Rn-G)/(H+LE) closure ratio (from gap-filled turbulent+available)
  et_valid_frac fraction of the 24 hours with a valid raw LE
  theta_<d>     profile-position-mean VWC (m3/m3) at depth index d (1=shallow)
  theta_mean    profile mean VWC (m3/m3) across all depths/positions
  theta_shallow theta at depth index 1
  theta_deep    theta at the deepest index
  theta_valid_n number of sensor-halfhours behind theta_mean

Conversion: ET(mm/day) = mean_daily_LE(W m-2) * 86400 / (lambda*rho_w),
lambda*rho_w = 2.45e6 J/kg * 1000 kg/m3 = 2.45e9 J/m3  ->  factor 86400/2.45e9 * 1000.
SWC(%) -> m3/m3 by /100. Depth index convention (AmeriFlux _H_V_R): V=1 shallowest.
No per-sensor VAR_INFO in the BADM; SOIL_DEPTH=180 cm; depths documented in the
UNL record are ~10/25/50/100 cm (Ne1/Ne2, 4 layers) plus deeper at Ne3 (5 layers).
Straight depth-mean profile is used (thickness weights unavailable) and reported as
such; anomaly-r and sigma-ratio comparisons are insensitive to a constant offset.
"""

import glob
import os

import numpy as np
import pandas as pd

SITES = ["US-Ne1", "US-Ne2", "US-Ne3"]
OUTDIR = "/data/ssd1/swim/soil_moisture/mead_base"
LAMBDA_RHO = 2.45e9  # J/m3 (latent heat * water density)
LE_TO_MM_DAY = 86400.0 / LAMBDA_RHO * 1000.0  # mean W/m2 -> mm/day  (~0.03526)
VALID_FRAC = 0.8  # require >=80% of 24 hours valid to trust a raw-LE daily mean
NA = -9999.0


def _base_file(site):
    s = site.split("-")[1]
    cands = glob.glob(
        f"/nas/climate/ameriflux/amf_new/AMF_US-{s}_BASE-BADM_*-5/AMF_US-{s}_BASE_HR_*.csv"
    )
    # prefer the highest version (latest span)
    return sorted(cands, key=lambda p: int(p.split("_HR_")[1].split("-")[0]))[-1]


def parse_site(site):
    f = _base_file(site)
    hdr = pd.read_csv(f, skiprows=2, nrows=0).columns.tolist()
    # gap-filled position/depth/replicate sensors: SWC_PI_F_<pos>_<depth>_<rep> (5 "_")
    swc_cols = [c for c in hdr if c.startswith("SWC_PI_F_") and c.count("_") == 5]
    energy = ["LE_1_1_1", "LE_PI_F_1_1_1", "H_1_1_1", "H_PI_F_1_1_1", "G_PI_F_1_1_1"]
    netrad = [c for c in ("NETRAD_1_1_1", "NETRAD_PI_F_1_1_1") if c in hdr]
    use = ["TIMESTAMP_START"] + [c for c in energy if c in hdr] + netrad + swc_cols
    df = pd.read_csv(f, skiprows=2, usecols=use, na_values=NA)
    ts = pd.to_datetime(df["TIMESTAMP_START"].astype(np.int64).astype(str), format="%Y%m%d%H%M")
    df.index = pd.DatetimeIndex(ts)
    date = df.index.floor("D")  # DatetimeIndex groupby key aligned to df

    # ---- ET from raw measured LE (LE_PI_F is empty in the OpenET era) ----
    # Raw components carried through 2024: LE_1_1_1, H_1_1_1, NETRAD_1_1_1
    # (Rn falls back to gap-filled NETRAD_PI_F; G uses gap-filled G_PI_F).
    le = df["LE_1_1_1"]
    h = df["H_1_1_1"]
    rn = df["NETRAD_1_1_1"].fillna(df["NETRAD_PI_F_1_1_1"])
    gflux = df["G_PI_F_1_1_1"]

    n_hr = df.groupby(date).size()
    et_valid_frac = le.notna().groupby(date).sum() / n_hr
    et_le_raw = le.groupby(date).mean() * LE_TO_MM_DAY
    et_le_raw[et_valid_frac < VALID_FRAC] = np.nan

    # Energy-balance closure (Bowen-ratio): scale LE by (Rn-G)/(H+LE), preserving
    # the Bowen ratio. Computed from hours where all four raw fluxes are present so
    # numerator and denominator share the same sampling, then gated on valid hours.
    comp = pd.DataFrame({"le": le, "h": h, "rn": rn, "g": gflux}, index=df.index).dropna()
    cdate = comp.index.floor("D")
    cmean = comp.groupby(cdate).mean()
    cvalid = comp.groupby(cdate).size() / n_hr.reindex(cmean.index)
    closure = (cmean["rn"] - cmean["g"]) / (cmean["h"] + cmean["le"])
    turb = cmean["h"] + cmean["le"]
    bad = (turb <= 20.0) | ~np.isfinite(closure) | (cvalid < 0.5)
    closure[bad] = np.nan
    closure = closure.reindex(et_le_raw.index)
    et_le_corr = et_le_raw * closure.clip(0.5, 2.0)

    out = pd.DataFrame(
        {
            "et_le_raw": et_le_raw,
            "et_le_corr": et_le_corr,
            "closure_frac": closure,
            "et_valid_frac": et_valid_frac,
        }
    )

    # ---- theta from SWC (positions x depths, gap-filled) ----
    depths = sorted({int(c.split("_")[4]) for c in swc_cols})
    theta_valid_n = pd.Series(0, index=out.index, dtype=float)
    depth_means = {}
    for d in depths:
        cols = [c for c in swc_cols if int(c.split("_")[4]) == d]
        sub = df[cols] / 100.0  # % -> m3/m3
        dm = sub.mean(axis=1)  # mean across horizontal positions
        depth_means[d] = dm.groupby(date).mean()
        out[f"theta_{d}"] = depth_means[d]
        theta_valid_n = theta_valid_n.add(sub.notna().sum(axis=1).groupby(date).sum(), fill_value=0)
    prof = pd.concat(depth_means, axis=1)
    out["theta_mean"] = prof.mean(axis=1)  # equal-weight depth mean
    out["theta_shallow"] = depth_means[depths[0]]
    out["theta_deep"] = depth_means[depths[-1]]
    out["theta_valid_n"] = theta_valid_n
    out.index.name = "date"
    out.attrs["depths"] = depths
    return out, depths, f


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    summary = []
    for site in SITES:
        out, depths, f = parse_site(site)
        p = os.path.join(OUTDIR, f"{site}_daily.parquet")
        out.to_parquet(p)
        oe = out[out.index.year >= 2016]
        summary.append(
            dict(
                site=site,
                base=os.path.basename(f),
                span=f"{out.index.min().date()}..{out.index.max().date()}",
                n_depth_idx=len(depths),
                n_days=len(out),
                et_raw_days=int(out["et_le_raw"].notna().sum()),
                et_corr_days=int(out["et_le_corr"].notna().sum()),
                theta_days=int(out["theta_mean"].notna().sum()),
                oe_et_raw=int(oe["et_le_raw"].notna().sum()),
                oe_et_corr=int(oe["et_le_corr"].notna().sum()),
                oe_theta=int(oe["theta_mean"].notna().sum()),
                oe_overlap=int((oe["et_le_raw"].notna() & oe["theta_mean"].notna()).sum()),
            )
        )
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUTDIR, "parse_summary.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()

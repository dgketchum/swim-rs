"""Validate one NWI data unit against the intake contract before a build.

Implements the checks in notes/nwi_intake_contract.md §3.2 so a collaborator
drop (or our own EE extraction) is proven complete before
nwi_build_container.py is allowed to consume it. Null-filled CSVs are the
recurring silent failure mode, so this reports and fails — it never patches.

Checks per unit (data/{label}/):
  1. file inventory per family: ETf (6 members + QC ensemble x 2 masks x
     1999-2025), Landsat NDVI (2 masks x POR), Sentinel NDVI (2 masks x
     2017+), ETo (POR), SNODAS SWE (2004+)
  2. field-ID set equality across every family, against properties/ssurgo
     and (when present) the unit's GIS roster
  3. shape/coverage: ETo day counts equal days-in-year; NDVI/ETf scene
     counts non-zero; per-file null fraction
  4. per-field coverage of the irr u inv_irr union (mask semantics make a
     single-mask all-null year expected; zero union coverage is a failure)

Writes data/{label}/intake_report.json and .md; exit code is nonzero when
any check fails.

Usage:
    uv run python scripts/nwi_intake.py --labels 32019a,32019b,32019c
"""

import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd

MASKS = ("irr", "inv_irr")
MEMBERS = ("ssebop", "sims", "geesebal", "eemetric", "ptjpl", "disalexi")
QC_MODEL = "ensemble"  # present in the drop, deliberately NOT ingested
ETF_START = 1999
# OpenET v2.1 per-model coverage starts; disalexi joins two years late, so its
# 1999-2000 absence is coverage, not a gap (the Esmeralda pilot is identical).
MODEL_START = {"disalexi": 2001}
# The EE props export emits LAT/LON header columns it never fills; the ingest
# does not read them. All-null in the pilot too — informational, not a failure.
KNOWN_EMPTY_PROP_COLS = ("LAT", "LON")
SWE_START = 2004
SENTINEL_START = 2017


class Report:
    def __init__(self, label):
        self.label = label
        self.items = []

    def add(self, family, name, ok, detail, **extra):
        self.items.append(
            {
                "family": family,
                "check": name,
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
                **extra,
            }
        )
        return ok

    @property
    def failures(self):
        return [i for i in self.items if i["status"] == "FAIL"]


def read_wide(path, id_col="NWI_ID"):
    df = pd.read_csv(path)
    ids = df[id_col].astype(str)
    vals = df.drop(columns=[id_col]).to_numpy(dtype=float)
    return ids, vals


def check_unit(label, data_root, por_start, por_end, strict_nulls):
    root = Path(data_root) / label
    rep = Report(label)
    years = list(range(por_start, por_end + 1))

    ss_path = root / "properties" / f"ssurgo_{label}.csv"
    if not ss_path.exists():
        rep.add("properties", "ssurgo present", False, f"missing {ss_path}")
        return rep, None
    roster = set(pd.read_csv(ss_path)["NWI_ID"].astype(str))
    rep.add("properties", "ssurgo roster", True, f"{len(roster)} fields")

    # 1/2/3. ETf ------------------------------------------------------------
    etf_years = [y for y in years if y >= ETF_START]
    union_cov = {}
    for model in MEMBERS:
        model_years = [y for y in etf_years if y >= MODEL_START.get(model, ETF_START)]
        for mask in MASKS:
            missing, empty, nullfrac = [], [], []
            for y in model_years:
                p = root / "etf" / mask / f"{model}_etf_{mask}_{y}.csv"
                if not p.exists():
                    missing.append(y)
                    continue
                ids, vals = read_wide(p)
                if set(ids) != roster:
                    rep.add(
                        "etf", f"{model}/{mask}/{y} ids", False, "field-ID set != ssurgo roster"
                    )
                if vals.shape[1] == 0:
                    empty.append(y)
                    continue
                nullfrac.append(float(np.isnan(vals).mean()))
                cov = np.isfinite(vals).sum(axis=1)
                key = (mask, y)
                prev = union_cov.get(key)
                union_cov[key] = (ids.to_numpy(), cov if prev is None else prev[1] + cov)
            rep.add(
                "etf",
                f"{model}/{mask} inventory",
                not missing and not empty,
                f"{len(model_years) - len(missing)}/{len(model_years)} years"
                + (f" (coverage starts {MODEL_START[model]})" if model in MODEL_START else "")
                + (f"; missing {missing}" if missing else "")
                + (f"; zero-scene {empty}" if empty else ""),
            )
            if nullfrac:
                rep.add(
                    "etf",
                    f"{model}/{mask} nulls",
                    True,
                    f"per-file null fraction median {np.median(nullfrac):.3f}",
                )

    qc = sum(
        1
        for mask in MASKS
        for y in etf_years
        if (root / "etf" / mask / f"{QC_MODEL}_etf_{mask}_{y}.csv").exists()
    )
    rep.add("etf", "QC ensemble present (not ingested)", True, f"{qc} files")

    # per-field union coverage across masks (union over both masks per year)
    per_field_union = {}
    for (mask, y), (ids, cov) in union_cov.items():
        for fid, c in zip(ids, cov):
            per_field_union[fid] = per_field_union.get(fid, 0) + int(c)
    zero_cov = [f for f in roster if per_field_union.get(f, 0) == 0]
    rep.add(
        "etf",
        "per-field union coverage",
        not zero_cov,
        f"{len(zero_cov)} fields with zero ETf observations across all members/masks"
        + (f": {zero_cov[:10]}" if zero_cov else ""),
    )

    # NDVI ------------------------------------------------------------------
    for sub, tag, yrs in (
        ("ndvi", "ndvi", years),
        ("ndvi/sentinel", "ndvi_sentinel", [y for y in years if y >= SENTINEL_START]),
    ):
        for mask in MASKS:
            missing, scenes = [], []
            for y in yrs:
                p = root / sub / mask / f"{tag}_{mask}_{y}.csv"
                if not p.exists():
                    missing.append(y)
                    continue
                ids, vals = read_wide(p)
                if set(ids) != roster:
                    rep.add("ndvi", f"{tag}/{mask}/{y} ids", False, "field-ID set != ssurgo roster")
                scenes.append(vals.shape[1])
            ok = not missing and all(s > 0 for s in scenes)
            rep.add(
                "ndvi",
                f"{tag}/{mask} inventory",
                ok,
                f"{len(yrs) - len(missing)}/{len(yrs)} years; scenes/yr min={min(scenes) if scenes else 0} "
                f"med={int(np.median(scenes)) if scenes else 0}"
                + (f"; missing {missing}" if missing else ""),
            )

    # ETo -------------------------------------------------------------------
    missing, bad_days, nulls = [], [], []
    for y in years:
        p = root / "met" / "eto" / f"eto_{y}.csv"
        if not p.exists():
            missing.append(y)
            continue
        ids, vals = read_wide(p)
        if set(ids) != roster:
            rep.add("eto", f"{y} ids", False, "field-ID set != ssurgo roster")
        expect = 366 if calendar.isleap(y) else 365
        if vals.shape[1] != expect:
            bad_days.append((y, vals.shape[1], expect))
        nf = float(np.isnan(vals).mean())
        if nf > 0:
            nulls.append((y, round(nf, 4)))
    rep.add(
        "eto",
        "inventory",
        not missing,
        f"{len(years) - len(missing)}/{len(years)} years"
        + (f"; missing {missing}" if missing else ""),
    )
    rep.add(
        "eto",
        "day counts",
        not bad_days,
        "all years match days-in-year" if not bad_days else str(bad_days),
    )
    rep.add(
        "eto",
        "nulls",
        not nulls or not strict_nulls,
        "zero NaN" if not nulls else f"NaN present: {nulls}",
    )

    # SNODAS ----------------------------------------------------------------
    swe_years = [y for y in years if y >= SWE_START]
    missing = [
        y
        for y in swe_years
        if not (root / "snow" / "snodas" / "extracts" / f"swe_{y}.csv").exists()
    ]
    rep.add(
        "snow",
        "inventory",
        not missing,
        f"{len(swe_years) - len(missing)}/{len(swe_years)} years"
        + (f"; missing {missing}" if missing else ""),
    )

    # properties ------------------------------------------------------------
    for kind in ("irr", "landcover"):
        p = root / "properties" / f"{kind}_{label}.csv"
        if not p.exists():
            rep.add("properties", f"{kind} present", False, f"missing {p}")
            continue
        df = pd.read_csv(p)
        same = set(df["NWI_ID"].astype(str)) == roster
        drop = ["NWI_ID"] + [c for c in KNOWN_EMPTY_PROP_COLS if c in df.columns]
        empty_known = [c for c in KNOWN_EMPTY_PROP_COLS if c in df.columns and df[c].isna().all()]
        nn = float(df.drop(columns=drop).isna().to_numpy().mean())
        rep.add(
            "properties",
            f"{kind}",
            same and nn == 0.0,
            f"{len(df)} rows, id_match={same}, null_frac={nn:.4f} (data columns)"
            + (f"; unused header columns empty as expected: {empty_known}" if empty_known else ""),
        )

    # GIS roster ------------------------------------------------------------
    shp = root / "gis" / f"nwi_fields_{label}_gfid.shp"
    gfids = None
    if shp.exists():
        import geopandas as gpd

        g = gpd.read_file(shp, engine="fiona")
        same = set(g["NWI_ID"].astype(str)) == roster
        has_gfid = "GFID" in g.columns and g["GFID"].notna().all()
        gfids = sorted(set(g["GFID"].astype(int))) if has_gfid else None
        rep.add(
            "gis",
            "roster + GFID",
            same and has_gfid,
            f"{len(g)} fields, id_match={same}, GFID complete={has_gfid}, cells={g['GFID'].nunique()}",
        )
    else:
        rep.add("gis", "roster + GFID", False, f"missing {shp}")

    return rep, gfids


def check_met(rep, met_dir, gfids, por_start, por_end):
    """Every GridMET cell the unit needs must be present and full-POR.

    THREDDS failures are silently swallowed upstream, so a short or
    NaN-bearing parquet is the expected failure mode, not a missing file.
    """
    met_dir = Path(met_dir)
    if gfids is None:
        rep.add("met", "gridmet parquets", False, "no GFID roster available (GIS check failed)")
        return
    expected_days = (pd.Timestamp(f"{por_end}-12-31") - pd.Timestamp(f"{por_start}-01-01")).days + 1
    missing, short, nanny = [], [], []
    for g in gfids:
        p = met_dir / f"{g}.parquet"
        if not p.exists():
            missing.append(g)
            continue
        df = pd.read_parquet(p)
        if len(df) != expected_days:
            short.append((g, len(df)))
        if df.isna().to_numpy().any():
            nanny.append(g)
    rep.add(
        "met",
        "gridmet parquets present",
        not missing,
        f"{len(gfids) - len(missing)}/{len(gfids)} cells"
        + (f"; missing {missing[:10]}" if missing else ""),
    )
    rep.add(
        "met",
        "full-POR row counts",
        not short,
        f"expected {expected_days} rows/cell" + (f"; short: {short[:10]}" if short else ""),
    )
    rep.add("met", "no NaN", not nanny, "zero NaN cells" if not nanny else f"NaN in {nanny[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Comma-separated unit labels")
    ap.add_argument("--data-root", default="/project/handily/swim/data")
    ap.add_argument("--por-start", type=int, default=1985)
    ap.add_argument("--por-end", type=int, default=2025)
    ap.add_argument("--strict-nulls", action="store_true", default=True)
    ap.add_argument(
        "--met-dir",
        default=None,
        help="Also verify the GridMET parquets each unit needs (shared store path)",
    )
    args = ap.parse_args()

    all_ok = True
    for label in [x.strip() for x in args.labels.split(",")]:
        rep, gfids = check_unit(
            label, args.data_root, args.por_start, args.por_end, args.strict_nulls
        )
        if args.met_dir:
            check_met(rep, args.met_dir, gfids, args.por_start, args.por_end)
        out = Path(args.data_root) / label
        out.mkdir(parents=True, exist_ok=True)
        (out / "intake_report.json").write_text(
            json.dumps({"label": label, "checks": rep.items}, indent=2)
        )

        lines = [f"# NWI intake report — {label}", ""]
        for fam in sorted({i["family"] for i in rep.items}):
            lines.append(f"## {fam}")
            for i in [x for x in rep.items if x["family"] == fam]:
                lines.append(f"- **{i['status']}** {i['check']}: {i['detail']}")
            lines.append("")
        verdict = (
            "GREEN — cleared for container build"
            if not rep.failures
            else f"RED — {len(rep.failures)} failures"
        )
        lines.insert(1, f"Verdict: **{verdict}**")
        (out / "intake_report.md").write_text("\n".join(lines))

        print(f"{label}: {len(rep.items)} checks, {len(rep.failures)} failures -> {verdict}")
        for f in rep.failures:
            print(f"    FAIL {f['family']}/{f['check']}: {f['detail']}")
        all_ok &= not rep.failures

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

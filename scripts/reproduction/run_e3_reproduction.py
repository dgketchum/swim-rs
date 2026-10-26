"""Example 7 (E3) reproduction, step by step, for walking in the debugger.

This is the "E3 (Example 7)" block of "3. Recompute the evidence" in
examples/README.md written as direct Python calls instead of command lines.
``evaluate_applied`` re-states ``evaluate_applied_water.main``: the config,
the metered-truth roster, the open container, the field list, the parameter
set, the daily series of every field (``run_applied``, the forward model run
with the irrigation scheduler), the annual sums, and the paired table are
locals you can inspect. The mapping builder keeps its body inside ``main`` and
parses its own arguments, so ``call_main`` sets ``sys.argv`` and calls it;
step into the call or break on its first line.

Run or debug it as-is: the data root and scratch directory are set in
``ROOT`` and ``SCRATCH`` below, and ``STEPS`` selects which steps run. The
repo-local package supplies ``examples/7_Applied_Water/data/metered_truth.csv``
and the IrrMapper control cache; the reference-final package supplies the
frozen transfer vectors and mapping under ``paper/data/final``.

Comparisons are byte-level, like the README's ``cmp``. The evaluator scores
every metered field it finds (100 fields, 730 field-years); the Table 6
statistics are computed on the frozen 50-field, 404-field-year SLV record,
which the last step replicates (``e3_period_split.gate``) before printing the
pre-/from-2016 split. A rebuild on other hardware may differ in the last digit
of a few statistics; a cohort, sign, or file-set difference is a failure.
"""

import filecmp
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EX7 = REPO / "examples" / "7_Applied_Water"
EX6 = REPO / "examples" / "6_Flux_International"  # the evaluator reuses its parse_pest_params

# Edit these three to walk the script from the debugger with no configuration.
ROOT = Path.home() / "data" / "swim"  # the data root the swim_data_* packages were unpacked into
SCRATCH = Path.home() / "data" / "swim_scratch"  # outside the clone and the data root
STEPS = []  # e.g. ["calibrated", "cmp"]; empty means every step in README order

REPS = 10000  # period-split bootstrap replicates (e3_period_split default --reps)
SEED = 42  # period-split bootstrap seed (e3_period_split default --seed)

os.environ["SWIM_ROOT"] = str(ROOT)  # the TOML root relocation every Example 7 module reads
for _p in (EX6, EX7):
    sys.path.insert(0, str(_p))  # the modules import each other as top-level names

import build_applied_irrigation_mapping as bam  # noqa: E402
import e3_period_split as ps  # noqa: E402
import evaluate_applied_water as eaw  # noqa: E402
import ex7_paths  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from swimrs.container import open_container  # noqa: E402

E7 = ROOT / "7_Applied_Water"
CONTAINER = E7 / "data" / "7_Applied_Water_e7cal.swim"  # the calibrated run container
POST = E7 / "results" / "e7cal" / "archive" / "4_pest_outputs" / "merged" / "merged_posterior.json"
FINAL = REPO / "paper" / "data" / "final"
MAPPING_FINAL = FINAL / "e4_irrigation_stratified_param_mapping.json"
SHIPPED_CAL = E7 / "results" / f"applied_{ex7_paths.LOCAL_LABEL}"
SHIPPED_TRANSFER = E7 / "results" / f"applied_{ex7_paths.TRANSFER_LABEL}"
S = SCRATCH / "e3"

DIFFERED = []  # (scratch file, shipped file) pairs that are not byte-identical


def compare(scratch_dir, shipped_dir, files):
    for f in files:
        a, b = Path(scratch_dir) / f, Path(shipped_dir) / f
        if a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False):
            print(f"  identical: {f}")
        else:
            print(f"  {'differs' if a.exists() and b.exists() else 'missing'}: {a} vs {b}")
            DIFFERED.append((a, b))


def call_main(module, argv):
    """Run ``module.main()`` as the README command line would; returns its exit status."""
    saved = sys.argv
    sys.argv = [f"{module.__name__}.py", *map(str, argv)]
    try:
        rc = module.main()  # step in: the whole script body is this one function
    except SystemExit as e:
        rc = e.code
    finally:
        sys.argv = saved
    return rc or 0


def evaluate_applied(params_json, label, out_dir):
    """evaluate_applied_water.main: one forward run, scored on the metered field-years."""
    cfg = eaw._load_config()
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = ex7_paths.drop_excluded_field_years(pd.read_csv(eaw.TRUTH))
    fields_gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")  # basin and crop per field
    meta = fields_gdf.drop_duplicates("site_id").set_index("site_id")
    container = open_container(str(CONTAINER), mode="r")
    try:
        fids = [f for f in container.field_uids if f in set(truth.site_id)]
        params = eaw._resolve_params(container, fids, None, str(params_json))
        series = eaw.run_applied(cfg, container, list(params), params)  # step in: the daily loop
    finally:
        container.close()

    rows = []  # annual simulated applied depth (+ ET) per field-year
    for fid, df in series.items():
        ann = df.groupby(df.index.year).agg(
            sim_applied_mm=("irr_sim", "sum"), sim_et_mm=("et_act", "sum")
        )
        for yr, r in ann.iterrows():
            rows.append(
                {
                    "site_id": fid,
                    "year": int(yr),
                    "sim_applied_mm": float(r.sim_applied_mm),
                    "sim_et_mm": float(r.sim_et_mm),
                }
            )
    sim = pd.DataFrame(rows)

    metered = truth[truth.metered_depth_mm > 0]  # join on (site_id, year)
    paired = metered.merge(sim, on=["site_id", "year"], how="inner")
    paired["basin"] = paired.site_id.map(meta.basin) if "basin" in meta else ""
    paired["crop"] = paired.site_id.map(meta.crop) if "crop" in meta else ""
    paired["efficiency"] = paired.sim_applied_mm / paired.metered_depth_mm.replace(0, np.nan)
    eaw._score(paired, out_dir, label)  # per_field_year.csv, summary_metrics.csv, scatter

    controls = truth[truth.source == "ESPA_rainfed_control"].site_id.unique()
    csim = sim[sim.site_id.isin(controls)]  # negative controls: simulated irrigation must be ~0
    ctl = {
        "n_control_fields": int(len(controls)),
        "n_control_field_years": int(len(csim)),
        "sim_applied_mm_mean": round(float(csim.sim_applied_mm.mean()) if len(csim) else 0.0, 2),
        "sim_applied_mm_max": round(float(csim.sim_applied_mm.max()) if len(csim) else 0.0, 2),
        "frac_years_gt_10mm": round(
            float((csim.sim_applied_mm > 10).mean()) if len(csim) else 0.0, 3
        ),
    }
    (out_dir / "negative_controls.json").write_text(json.dumps(ctl, indent=2))
    print("controls:", ctl)
    return paired


# --- 1. locally calibrated arm: the e7cal posterior medians ------------------
# README: evaluate_applied_water.py --container $E7/data/7_Applied_Water_e7cal.swim --params-json $POST --label calibrated --out $S/calibrated
def step_calibrated():
    return evaluate_applied(POST, ex7_paths.LOCAL_LABEL, S / "calibrated")


# --- 2. irrigation-stratified E1 -> E3 mapping, checked against the frozen one
# README: build_applied_irrigation_mapping.py --verify-keys $E7/results/applied_calibrated/per_field_year.csv --out-dir $S/mapping
#         cmp $S/mapping/e4_irrigation_stratified_param_mapping.json paper/data/final/e4_irrigation_stratified_param_mapping.json
def step_mapping():
    out = S / "mapping"
    call_main(bam, ["--verify-keys", SHIPPED_CAL / "per_field_year.csv", "--out-dir", out])
    compare(out, FINAL, [MAPPING_FINAL.name])


# --- 3. E1 irrigated-class transfer arm: the frozen per-field mapping ---------
# README: evaluate_applied_water.py --container ... --params-json paper/data/final/e4_irrigation_stratified_param_mapping.json --label transfer_run23_by_irrigation --out $S/transfer
def step_transfer():
    return evaluate_applied(MAPPING_FINAL, ex7_paths.TRANSFER_LABEL, S / "transfer")


# --- 4. compare both arms with the shipped results ---------------------------
# README: for a in calibrated:applied_calibrated transfer:applied_transfer_run23_by_irrigation; do cmp ... per_field_year.csv summary_metrics.csv negative_controls.json
def step_cmp():
    files = ["per_field_year.csv", "summary_metrics.csv", "negative_controls.json"]
    compare(S / "calibrated", SHIPPED_CAL, files)
    compare(S / "transfer", SHIPPED_TRANSFER, files)


# --- 5. period-split gate on the frozen Table 6 record (no --write) ----------
# README: e3_period_split.py
def step_split():
    metrics, diffs, gate = ps.build(REPS, SEED)  # gate raises PeriodSplitError when Table 6 differs
    print(gate)
    print(metrics.round(3).to_string(index=False))
    cols = ["treatment", "statistic", "estimate", "ci95_low", "ci95_high"]
    print(diffs[cols].round(3).to_string(index=False))
    return metrics, diffs


ALL_STEPS = {
    "calibrated": step_calibrated,
    "mapping": step_mapping,
    "transfer": step_transfer,
    "cmp": step_cmp,
    "split": step_split,
}


def main(steps):
    required = [
        CONTAINER,
        POST,
        MAPPING_FINAL,
        eaw.TRUTH,
        ex7_paths.TRANSFER_VECTORS,
        ex7_paths.ESPA_CONTROL_IRR,
        SHIPPED_CAL / "per_field_year.csv",
        ps.FIELD_YEARS,
    ]
    for p in required:
        if not p.exists():
            sys.exit(f"missing input: {p}")
    S.mkdir(parents=True, exist_ok=True)
    for name in steps:
        print(f"\n==== {name}")
        ALL_STEPS[name]()
    print("\n==== summary")
    if DIFFERED:
        print(
            f"{len(DIFFERED)} compared file(s) differ (diff each pair to see whether it is round-off):"
        )
        for a, b in DIFFERED:
            print(f"  {a}\n    vs {b}")
    else:
        print("every compared file is byte-identical to the shipped copy")
    return not DIFFERED


if __name__ == "__main__":
    chosen = sys.argv[1:] or STEPS or list(ALL_STEPS)
    unknown = [s for s in chosen if s not in ALL_STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from: {' '.join(ALL_STEPS)}")
    sys.exit(0 if main(chosen) else 1)

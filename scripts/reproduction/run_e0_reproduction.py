"""E0 (vegetation-formulation arms, Example 6) reproduction for walking in the debugger.

This is the six ``pooled_arm_compare.py`` invocations of the "E0 and E2
(Example 6)" block of "3. Recompute the evidence" in examples/README.md
(step 6 of examples/6_Flux_International/README.md) written as direct Python
calls. ``compare_arms`` re-states the script's ``main`` body: the two arm
configs, their effective physics, the forward run of each arm from its own
posterior (``arm_series`` -> ``run_calibrated_model``), the pooled pairing
(``collect``), the gate (``decide``), and the per-site deltas are locals you
can inspect. The three pairs run on the 37 disjoint sites and on the 47-site
closure pool, in README order.

Run or debug it as-is: the data root and scratch directory are set in
``ROOT`` and ``SCRATCH`` below, and ``STEPS`` selects which pair runs.

Outputs go under ``OUT_ROOT`` and are compared byte-for-byte with the shipped
copies under ``results/e0_disjoint``. The README writes in place there
instead, because ``promote_final.py`` reads that directory; set ``OUT_ROOT``
to ``RESULTS / "e0_disjoint"`` to do the same. The gate verdicts are also
checked: both grassbasis pairs pass, and fao56_sig vs fao56 fails, which is
the Table 3 result and not an error.
"""

import filecmp
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EX6 = REPO / "examples" / "6_Flux_International"
RUN = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
CFG = EX6 / f"{RUN}.toml"  # arm A: cover-scaled sigmoid Kcb (the published E2 run)
CFG_SIG = EX6 / f"{RUN}_fao56_sig.toml"  # arm B: unscaled sigmoid Kcb
CFG_LIN = EX6 / f"{RUN}_fao56.toml"  # arm C: unscaled linear Kcb

# Edit these three to walk the script from the debugger with no configuration.
ROOT = Path.home() / "data" / "swim"  # the data root the swim_data_* packages were unpacked into
SCRATCH = Path.home() / "data" / "swim_scratch"  # outside the clone and the data root
STEPS = []  # e.g. ["grassbasis_vs_fao56_sig"]; empty means every pair in README order

os.environ["SWIM_ROOT"] = str(ROOT)  # the TOML root relocation every Example 6 module reads
sys.path.insert(0, str(EX6))  # the modules import each other as top-level names

import pooled_arm_compare as pac  # noqa: E402

RESULTS = ROOT / "6_Flux_International" / "results"
SHIPPED = RESULTS / "e0_disjoint"
S = SCRATCH / "e0"
OUT_ROOT = S / "e0_disjoint"  # README: $RESULTS/e0_disjoint (in place)


def _posterior(run):
    return RESULTS / run / "archive" / "4_pest_outputs" / "merged" / "merged_posterior.csv"


ARMS = {  # name -> (config, posterior)
    "grassbasis": (CFG, _posterior(RUN)),
    "fao56_sig": (CFG_SIG, _posterior(f"{RUN}_fao56_sig")),
    "fao56": (CFG_LIN, _posterior(f"{RUN}_fao56")),
}
PAIRS = {  # pair -> (arm A, arm B, expected gate)
    "grassbasis_vs_fao56_sig": ("grassbasis", "fao56_sig", True),
    "grassbasis_vs_fao56": ("grassbasis", "fao56", True),
    "fao56_sig_vs_fao56": ("fao56_sig", "fao56", False),
}
SETS = {  # tag -> site list under examples/6_Flux_International/e0_disjoint
    "disjoint37": "disjoint_sites.txt",  # closure-pool sites outside the E1 cohort
    "pool47": "closure_pool_sites.txt",  # the whole closure pool, as a check
}
FILES = ["pooled_gate.csv", "pooled_per_site.csv", "paired_site_deltas.csv", "sites_excluded.csv"]

DIFFERED = []  # (scratch file, shipped file) pairs that are not byte-identical, or gate surprises


def compare(scratch_dir, shipped_dir, files):
    for f in files:
        a, b = Path(scratch_dir) / f, Path(shipped_dir) / f
        if a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False):
            print(f"  identical: {f}")
        else:
            print(f"  {'differs' if a.exists() and b.exists() else 'missing'}: {a} vs {b}")
            DIFFERED.append((a, b))


def compare_arms(name_a, name_b, sites_file, out_dir):
    """pooled_arm_compare.main for one pair and one site list; returns the gate verdict."""
    (config_a, par_a), (config_b, par_b) = ARMS[name_a], ARMS[name_b]
    cfg_a = pac.load_arm_config(str(config_a))
    cfg_b = pac.load_arm_config(str(config_b))
    cont_a = pac._default_container_path(cfg_a)  # [paths] container of each arm's TOML
    cont_b = pac._default_container_path(cfg_b)

    phys_a, phys_b = pac.effective_physics(cfg_a), pac.effective_physics(cfg_b)
    for name, phys in ((name_a, phys_a), (name_b, phys_b)):
        print(f"arm {name}: kcb={phys['kcb_ndvi_mode']} cover={phys['transpiration_cover_mode']}")
    if phys_a == phys_b:
        raise SystemExit("Both arms resolved to identical physics; check the configs")

    fids = [ln.strip() for ln in open(sites_file) if ln.strip()]
    flux_sources = pac.load_flux_sources(cfg_a.fields_shapefile, cfg_a.feature_id_col)

    print(f"\nRunning arm A ({name_a}) forward on {len(fids)} sites...")
    series_a, src_a = pac.arm_series(
        cfg_a, cont_a, str(par_a), fids
    )  # step in: run_calibrated_model
    print(f"Running arm B ({name_b}) forward on {len(fids)} sites...")
    series_b, src_b = pac.arm_series(cfg_b, cont_b, str(par_b), fids)

    print("\nPairing...")
    pooled_daily, pooled_monthly, per_site, excluded = pac.collect(
        fids, flux_sources, series_a, series_b
    )
    table, wins_a, passed = pac.decide(name_a, name_b, pooled_daily, pooled_monthly)
    site_deltas = pac.paired_site_deltas(per_site, name_a, name_b)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_site.to_csv(out_dir / "pooled_per_site.csv", index=False)
    excluded.to_csv(out_dir / "sites_excluded.csv", index=False)
    table.to_csv(out_dir / "pooled_gate.csv", index=False)
    site_deltas.to_csv(out_dir / "paired_site_deltas.csv", index=False)

    print(f"\nPOOLED TWO-ARM COMPARISON: {len(per_site)} sites ({len(excluded)} excluded)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(site_deltas.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"{name_a} wins {wins_a} of 6 pooled metrics  ->  GATE {'PASS' if passed else 'FAIL'}")

    with open(out_dir / "pooled_gate.json", "w") as fh:
        json.dump(
            {
                "arm_a": name_a,
                "arm_b": name_b,
                "a_physics": phys_a,
                "b_physics": phys_b,
                "a_config": str(config_a),
                "b_config": str(config_b),
                "a_container": cont_a,
                "b_container": cont_b,
                "a_params": src_a,
                "b_params": src_b,
                "sites_requested": fids,
                "n_sites": int(len(per_site)),
                "n_daily": int(len(pooled_daily["obs"])),
                "n_monthly": int(len(pooled_monthly["obs"])),
                "wins_a": int(wins_a),
                "gate_rule": "arm A wins >= 4 of 6 pooled metrics",
                "passed": bool(passed),
                "metrics": table.to_dict(orient="records"),
                "paired_site_deltas": site_deltas.to_dict(orient="records"),
            },
            fh,
            indent=2,
        )
    return passed


def run_pair(pair):
    name_a, name_b, expect_pass = PAIRS[pair]
    for tag, sites in SETS.items():
        print(f"\n-- {pair} on {tag}")
        out = OUT_ROOT / pair / tag
        passed = compare_arms(name_a, name_b, EX6 / "e0_disjoint" / sites, out)
        if passed != expect_pass:
            print(f"  UNEXPECTED: gate {'passed' if passed else 'failed'}")
            DIFFERED.append(
                (out / "pooled_gate.json", f"expected GATE {'PASS' if expect_pass else 'FAIL'}")
            )
        if out != SHIPPED / pair / tag:
            compare(out, SHIPPED / pair / tag, FILES)
        else:
            print("  written in place (compare with the frozen package via promote_final.py)")


ALL_STEPS = {pair: (lambda p=pair: run_pair(p)) for pair in PAIRS}


def main(steps):
    required = [p for cfg_par in ARMS.values() for p in cfg_par]
    required += [EX6 / "e0_disjoint" / f for f in SETS.values()]
    for p in required:
        if not p.exists():
            sys.exit(f"missing input: {p}")
    for cfg_path, _ in ARMS.values():  # each arm's container, before any forward run
        cont = pac._default_container_path(pac.load_arm_config(str(cfg_path)))
        if not Path(cont).exists():
            sys.exit(f"missing container: {cont}")
    S.mkdir(parents=True, exist_ok=True)
    for name in steps:
        print(f"\n==== {name}")
        ALL_STEPS[name]()
    print("\n==== summary")
    if DIFFERED:
        print(f"{len(DIFFERED)} check(s) differ (diff each pair to see whether it is round-off):")
        for a, b in DIFFERED:
            print(f"  {a}\n    vs {b}")
    else:
        print("every compared file is byte-identical to the shipped copy; every gate as expected")
    return not DIFFERED


if __name__ == "__main__":
    chosen = sys.argv[1:] or STEPS or list(ALL_STEPS)
    unknown = [s for s in chosen if s not in ALL_STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from: {' '.join(ALL_STEPS)}")
    sys.exit(0 if main(chosen) else 1)

"""Stage the data packages for an evaluation-only reproduction of the paper.

The packages carry everything the three paper experiments (E1 ``5_Flux_Ensemble``,
E0+E2 ``6_Flux_International``, E3 ``7_Applied_Water``) read from the SWIM data
root and from the untracked repo-local data directories, trimmed of the PEST++
ensemble files no evaluation or figure script opens. Unpacked under one directory
that ``SWIM_ROOT`` points at (and, for the repo-local package, at the clone root),
the READMEs under ``examples/`` run as written. See examples/README.md,
"Reproducing the paper".

    uv run python scripts/package_reproduction_data.py --dry-run
    uv run python scripts/package_reproduction_data.py --stage <dir> [--packages e1,e2,e3,repo,truth,reference]

``--dry-run`` lists each package's file count and size and writes nothing.
``--stage`` writes ``<name>.tar.gz`` (pigz when available) for every package plus
``MANIFEST.json`` (git sha, source root, file counts and sizes) and the upload command.

Package contents (paths relative to the data root unless noted):

``e1``   Ex5 containers (base, run23, run23ablation), the Volk v2.1 flux and OpenET
         deliveries, gis/properties, the Figure 1 legacy OpenET daily table, and
         ``results/run23`` plus both ablation arms and their summary. ``pestrun_kept/``
         is dropped. In ``archive/4_pest_outputs`` only the posterior observation
         ensemble read by ``conditioned_ensemble_uncertainty.py`` is kept.
``e2``   Ex6 canonical GrassBasis container plus the two E0 arm containers,
         gis/properties/awc_recal/e0_disjoint/e2_etf_refooting, the three run
         result directories, the run23 irrigation-stratified transfer, the frozen
         E0 disjoint comparison, the evaluator outputs of the superseded
         classifier-recal baseline and the Run 22 transfer summary that
         ``evaluation_summary.py`` compares against, and the 66-site subset of the
         flux-data-qaqc archive at
         ``6_Flux_International/data/flux_qaqc/<network>/<sid>_daily_data.csv``.
``e3``   Ex7 ``data/`` (both containers, gis, properties, metered_truth.csv) and the
         ``e7cal``, ``applied_calibrated``, ``applied_transfer_run23_by_irrigation``
         and ``rescore_404fy_20260926`` result directories.
``repo`` Untracked repo-local inputs, relative to the clone root:
         ``examples/5_Flux_Ensemble/data`` (OpenET ETo, flux_2pt1 masters, ETf tables,
         gis), ``examples/7_Applied_Water/data`` (metered_truth.csv, OpenET refet,
         ETf tables), ``data/cartographic`` (figure basemaps).
``truth``  Optional: the E3 metered-truth source pulls (``data/co_slv_wells``,
         ``data/idwr_wmis`` without their tracked code), so README step 1 of Ex7
         can be re-run offline.
``reference``  ``paper/data/final/`` exactly as frozen here. Sealed: the tester opens it
         only after regenerating the evidence (reproduction_test_plan rule 3).
"""

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EX5_TOML = REPO / "examples" / "5_Flux_Ensemble" / "5_Flux_Ensemble.toml"
EX6_SHAPEFILE = REPO / "examples" / "6_Flux_International" / "data" / "gis"  # tracked copy
EX6_COHORT_SHP = "flux_crop_pub_66_150m.shp"

# Never packaged, anywhere.
ALWAYS_EXCLUDE_DIRS = {"__pycache__", ".git"}
ALWAYS_EXCLUDE_DIR_GLOBS = ("pestrun*",)
ALWAYS_EXCLUDE_FILE_GLOBS = ("*.lock", ".DS_Store")
# PEST++ ensemble files under any archive that no evaluation or figure script reads.
ARCHIVE_EXCLUDE_FILE_GLOBS = ("*.obs.csv", "loc.mat", "*.obs+noise.csv", "*.rei", "*.jcb")
# ...except these (relative to the data root).
ARCHIVE_KEEP = {
    "5_Flux_Ensemble/results/run23/archive/4_pest_outputs/5_Flux_Ensemble.3.obs.csv",
}

GRASSBASIS = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
# the superseded classifier-recal baseline that evaluation_summary.py compares against
# (its per-site series and evaluator outputs only; no archives, no PEST++ files)
E2_BASELINE = "6_Flux_International_LSEnsemble_POR_annual2yr"

ROOT_PACKAGES = {
    "e1": {
        "name": "swim_data_e1",
        "items": [
            "5_Flux_Ensemble/data/5_Flux_Ensemble.swim",
            "5_Flux_Ensemble/data/5_Flux_Ensemble_run23.swim",
            "5_Flux_Ensemble/data/5_Flux_Ensemble_run23ablation.swim",
            "5_Flux_Ensemble/data/daily_flux_files_2pt1",
            "5_Flux_Ensemble/data/openet_flux_2pt1",
            "5_Flux_Ensemble/data/flux_2pt1",
            "5_Flux_Ensemble/data/gis",
            "5_Flux_Ensemble/data/properties",
            "5_Flux_Ensemble/data/station_metadata.csv",
            "5_Flux_Ensemble/data/_deprecated_january_delivery/openet_flux/daily_data",
            "5_Flux_Ensemble/results/run23",
            "5_Flux_Ensemble/results/ablation_run23_e1_spread",
            "5_Flux_Ensemble/results/ablation_run23_e2_fixed_sd",
            "5_Flux_Ensemble/results/ablation_run23_summary",
        ],
    },
    "e2": {
        "name": "swim_data_e2",
        "items": [
            "6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim",
            "6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr_fao56.swim",
            "6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr_fao56_sig.swim",
            "6_Flux_International/data/gis",
            "6_Flux_International/data/properties",
            "6_Flux_International/data/awc_recal",
            "6_Flux_International/data/e0_disjoint",
            "6_Flux_International/data/e2_etf_refooting",
            f"6_Flux_International/results/{GRASSBASIS}",
            f"6_Flux_International/results/{GRASSBASIS}_fao56",
            f"6_Flux_International/results/{GRASSBASIS}_fao56_sig",
            "6_Flux_International/results/e2_run23_transfer_by_irrigation_to_grassbasis",
            "6_Flux_International/results/e0_disjoint",
            f"6_Flux_International/results/{E2_BASELINE}/*.csv",
            f"6_Flux_International/results/{E2_BASELINE}/archive_recal20260702_classifier/6_evaluation",
            "6_Flux_International/results/e2_run22_transfer_by_irrigation_to_e3/transfer_comparison_summary.csv",
        ],
        "flux_subset": "6_Flux_International/data/flux_qaqc",
    },
    "e3": {
        "name": "swim_data_e3",
        "items": [
            "7_Applied_Water/data",
            "7_Applied_Water/results/e7cal",
            "7_Applied_Water/results/applied_calibrated",
            "7_Applied_Water/results/applied_transfer_run23_by_irrigation",
            "7_Applied_Water/results/rescore_404fy_20260926",
        ],
        "exclude_dirs": ["archive_superseded_*"],
    },
}

REPO_PACKAGES = {
    "repo": {
        "name": "swim_repo_local",
        "items": [
            "examples/5_Flux_Ensemble/data",
            "examples/7_Applied_Water/data",
            "data/cartographic",
            # the IrrMapper gate for the ESPA rainfed controls, read by the Example 7
            # transfer mapping (step 7); the rest of data/idwr_wmis is truth-build input
            "data/idwr_wmis/espa_control_irrmapper.csv",
        ],
        "exclude_dirs": ["_deprecated_january_delivery"],
    },
    "truth": {
        "name": "swim_e3_truth_sources",
        "items": ["data/co_slv_wells", "data/idwr_wmis"],
        "exclude_files": ["*.py", "README.md"],
    },
    "reference": {
        "name": "swim_reference_final",
        "items": ["paper/data/final"],
        # the obs_support diagnostics were dropped from the figure package (2026-10-15)
        "exclude_files": ["figs02_*"],
    },
}


def swim_root():
    """``SWIM_ROOT`` or ``root`` from the Ex5 TOML (the three examples share it)."""
    env = os.environ.get("SWIM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    with open(EX5_TOML, "rb") as fh:
        return Path(tomllib.load(fh)["root"])


def _dir_excluded(name, extra_dir_globs):
    if name in ALWAYS_EXCLUDE_DIRS:
        return True
    return any(fnmatch.fnmatch(name, g) for g in (*ALWAYS_EXCLUDE_DIR_GLOBS, *extra_dir_globs))


def _file_excluded(rel, name, in_archive, extra_file_globs):
    if any(fnmatch.fnmatch(name, g) for g in (*ALWAYS_EXCLUDE_FILE_GLOBS, *extra_file_globs)):
        return True
    if in_archive and rel not in ARCHIVE_KEEP:
        return any(fnmatch.fnmatch(name, g) for g in ARCHIVE_EXCLUDE_FILE_GLOBS)
    return False


def collect(base, items, exclude_dirs=(), exclude_files=()):
    """``[(relative path, bytes)]`` for every packaged file under ``base``.

    ``items`` are files, directories, or file globs (``dir/*.csv``: the matching
    files, no descent) relative to ``base``; a missing item or a glob that matches
    nothing is an error (the package would be silently incomplete). Directories
    are walked with the exclusion rules above; the archive rules apply below any
    ``archive`` directory component.
    """
    base = Path(base)
    out = []
    for item in items:
        if any(ch in item for ch in "*?["):
            hits = sorted(p for p in base.glob(item) if p.is_file())
            if not hits:
                raise FileNotFoundError(f"package glob matches nothing: {base / item}")
            out.extend((str(p.relative_to(base)), p.stat().st_size) for p in hits)
            continue
        src = base / item
        if not src.exists():
            raise FileNotFoundError(f"package item missing: {src}")
        if src.is_file():
            out.append((item, src.stat().st_size))
            continue
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = sorted(d for d in dirnames if not _dir_excluded(d, exclude_dirs))
            rel_dir = Path(dirpath).relative_to(base)
            in_archive = "archive" in rel_dir.parts
            for fn in sorted(filenames):
                rel = str(rel_dir / fn)
                if _file_excluded(rel, fn, in_archive, exclude_files):
                    continue
                out.append((rel, (Path(dirpath) / fn).stat().st_size))
    return out


def flux_subset(root, flux_dir_rel):
    """The 66-site ``<network>/<sid>_daily_data.csv`` files the Ex6 cohort reads."""
    import geopandas as gpd

    shp = root / "6_Flux_International" / "data" / "gis" / EX6_COHORT_SHP
    if not shp.exists():
        shp = EX6_SHAPEFILE / EX6_COHORT_SHP
    cohort = gpd.read_file(shp, engine="fiona")
    flux_root = root / flux_dir_rel
    out = []
    for sid, net in zip(cohort["sid"], cohort["network"]):
        rel = f"{flux_dir_rel}/{net}/{sid}_daily_data.csv"
        src = root / rel
        if not src.exists():
            raise FileNotFoundError(f"flux record missing for {sid} ({net}): {src}")
        out.append((rel, src.stat().st_size))
    if not flux_root.exists():
        raise FileNotFoundError(f"flux archive not under the data root: {flux_root}")
    return out


def write_tar(base, files, out_tar):
    """``tar -C base -T list`` through pigz when present; members keep their relative paths."""
    names = out_tar.with_name(out_tar.name + ".list")
    names.write_text("".join(f"{rel}\n" for rel, _ in files))
    compress = ["--use-compress-program=pigz"] if shutil.which("pigz") else ["-z"]
    cmd = ["tar", "-C", str(base), *compress, "-cf", str(out_tar), "-T", str(names)]
    subprocess.run(cmd, check=True)
    names.unlink()


def package_files(key, root):
    if key in ROOT_PACKAGES:
        spec = ROOT_PACKAGES[key]
        files = collect(
            root, spec["items"], spec.get("exclude_dirs", ()), spec.get("exclude_files", ())
        )
        if "flux_subset" in spec:
            files += flux_subset(root, spec["flux_subset"])
        return spec["name"], root, files
    spec = REPO_PACKAGES[key]
    files = collect(
        REPO, spec["items"], spec.get("exclude_dirs", ()), spec.get("exclude_files", ())
    )
    return spec["name"], REPO, files


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def git_state():
    """``(sha, dirty)`` for the clone; ``dirty`` is True when a tracked file is modified."""
    try:
        sha = _git("rev-parse", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", default=None, help="data root (default: SWIM_ROOT or the Ex5 TOML root)"
    )
    ap.add_argument("--stage", default=None, help="directory to write the packages into")
    ap.add_argument(
        "--packages",
        default=",".join([*ROOT_PACKAGES, *REPO_PACKAGES]),
        help="comma-separated subset of e1,e2,e3,repo,truth,reference",
    )
    ap.add_argument(
        "--bucket", default="gs://wudr/swim/reprod/", help="upload prefix echoed at the end"
    )
    ap.add_argument("--dry-run", action="store_true", help="list sizes only")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.stage:
        ap.error("--stage <dir> or --dry-run")

    root = Path(args.root).resolve() if args.root else swim_root()
    keys = [k.strip() for k in args.packages.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ROOT_PACKAGES and k not in REPO_PACKAGES]
    if unknown:
        ap.error(f"unknown package(s): {unknown}")

    sha, dirty = git_state()
    if dirty:
        print("WARNING: tracked files are modified; MANIFEST git_sha is not the packaged code")
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "data_root": str(root),
        "repo": str(REPO),
        "packages": {},
    }
    stage = Path(args.stage) if args.stage else None
    if stage:
        stage.mkdir(parents=True, exist_ok=True)
    for key in keys:
        name, base, files = package_files(key, root)
        total = sum(s for _, s in files)
        print(f"{name:24s} {len(files):7d} files {total / 1e9:8.2f} GB  (relative to {base})")
        entry = {
            "base": "data_root" if base == root else "clone_root",
            "n_files": len(files),
            "bytes": total,
        }
        if stage:
            out_tar = stage / f"{name}.tar.gz"
            entry["tar"] = out_tar.name
            write_tar(base, files, out_tar)
            entry["tar_bytes"] = out_tar.stat().st_size
            print(f"    -> {out_tar} ({entry['tar_bytes'] / 1e9:.2f} GB)")
        manifest["packages"][name] = entry
    if stage:
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nMANIFEST.json written to {stage}")
        print(f"upload:  gsutil -m cp {stage}/*.tar.gz {stage}/MANIFEST.json {args.bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

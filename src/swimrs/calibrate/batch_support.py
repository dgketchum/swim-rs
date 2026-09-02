"""Support utilities for batch calibration.

Provides batch log I/O, FID coercion, config-driven coverage detection,
manifest handling, run manifest creation, and resolved restart state persistence.
"""

import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Batch log I/O (crash-safe JSON via tmp+rename)
# ---------------------------------------------------------------------------


def read_batch_log(output_root):
    """Read batch_log.json, return dict keyed by batch_id string."""
    log_path = Path(output_root) / "batch_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def write_batch_log(output_root, log_data):
    """Atomic write of batch_log.json via tmp+rename."""
    log_path = Path(output_root) / "batch_log.json"
    tmp_path = log_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(log_data, indent=2))
    tmp_path.rename(log_path)


def update_batch_entry(output_root, batch_id, entry):
    """Read batch_log, update one entry, write back."""
    log_data = read_batch_log(output_root)
    log_data[str(batch_id)] = entry
    write_batch_log(output_root, log_data)


# ---------------------------------------------------------------------------
# Per-batch status shards
#
# batch_log.json is a single JSON object rewritten in full by every writer.
# That is safe for the serial runner but loses entries when array tasks write
# concurrently, so fanned-out tasks each write their own shard and a single
# serial reduce step merges them.
# ---------------------------------------------------------------------------

SHARD_DIRNAME = "batch_status"


def shard_path(output_root, batch_id):
    """Path to one batch's status shard."""
    return Path(output_root) / SHARD_DIRNAME / f"batch_{int(batch_id):03d}.json"


def write_batch_shard(output_root, batch_id, entry):
    """Atomically write one batch's status shard (tmp+rename)."""
    path = shard_path(output_root, batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(entry)
    payload.setdefault("batch_id", int(batch_id))
    payload.setdefault("timestamp", datetime.now().isoformat())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)
    return path


def read_batch_shard(output_root, batch_id):
    """Read one batch's status shard, or None when absent/unreadable."""
    path = shard_path(output_root, batch_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def read_batch_shards(output_root):
    """Read every status shard, keyed by batch id string."""
    shard_dir = Path(output_root) / SHARD_DIRNAME
    if not shard_dir.exists():
        return {}
    shards = {}
    for f in sorted(shard_dir.glob("batch_*.json")):
        try:
            entry = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        bid = entry.get("batch_id")
        if bid is None:
            continue
        shards[str(int(bid))] = entry
    return shards


def merge_shards_into_log(output_root):
    """Fold status shards into batch_log.json. Single-writer only.

    Shards win over existing log entries: they are written by the task that
    actually ran the batch. Returns the merged log.
    """
    log_data = read_batch_log(output_root)
    for bid, entry in read_batch_shards(output_root).items():
        log_data[bid] = entry
    write_batch_log(output_root, log_data)
    return log_data


# ---------------------------------------------------------------------------
# FID coercion
# ---------------------------------------------------------------------------


def coerce_fid(raw) -> str:
    """Normalize a field ID value to a clean string.

    Handles pandas int->float upcasting ("1.0" -> "1") without corrupting
    underscore-delimited IDs like "001_000001" or string IDs like "US-FPe".
    """
    s = str(raw)
    return str(int(float(s))) if s.replace(".", "", 1).isdigit() else s


# ---------------------------------------------------------------------------
# Config-driven coverage detection
# ---------------------------------------------------------------------------


def _resolve_etf_paths(etf_target_model, etf_ensemble_members, mask_mode, instrument="landsat"):
    """Derive container ETf paths from config settings.

    Returns a list of container paths to check for observation coverage.
    """
    masks = _masks_for_mode(mask_mode)
    paths = []

    if etf_target_model == "ensemble" and etf_ensemble_members:
        for model in etf_ensemble_members:
            for mask in masks:
                paths.append(f"remote_sensing/etf/{instrument}/{model}/{mask}")
    elif etf_target_model:
        for mask in masks:
            paths.append(f"remote_sensing/etf/{instrument}/{etf_target_model}/{mask}")

    return paths


def _resolve_ndvi_paths(mask_mode, instrument="landsat"):
    """Derive container NDVI paths from config settings."""
    masks = _masks_for_mode(mask_mode)
    return [f"remote_sensing/ndvi/{instrument}/{mask}" for mask in masks]


def _masks_for_mode(mask_mode):
    """Return mask suffixes for the given mask_mode."""
    if mask_mode in ("irrigation", "irr"):
        return ("irr", "inv_irr")
    return ("no_mask",)


def get_uncovered_fids(
    container, etf_target_model, mask_mode, etf_ensemble_members=None, instrument="landsat"
):
    """Return field UIDs with zero observations for NDVI and/or ETf.

    Unlike the external runner, this derives paths from the SWIM-RS config
    rather than hard-coding irr/inv_irr paths.

    Parameters
    ----------
    container : SwimContainer
        Open container (read mode).
    etf_target_model : str
        ETf model name (e.g., "ssebop") or "ensemble".
    mask_mode : str
        Mask mode from config ("none", "irrigation").
    etf_ensemble_members : list[str] or None
        Ensemble member names when etf_target_model == "ensemble".
    instrument : str
        Remote sensing instrument (default: "landsat").

    Returns
    -------
    dict with keys "ndvi", "etf", and "all" (union).
    Each value is a sorted list of field UID strings.
    """
    ndvi_paths = _resolve_ndvi_paths(mask_mode, instrument)
    etf_paths = _resolve_etf_paths(etf_target_model, etf_ensemble_members, mask_mode, instrument)

    check_paths = {"ndvi": ndvi_paths, "etf": etf_paths}

    field_uids = container._field_uids
    n = len(field_uids)
    uncovered: dict[str, list[str]] = {}
    checked_paths: dict[str, list[str]] = {}

    for var, paths in check_paths.items():
        total_obs = np.zeros(n, dtype=int)
        found_any = False
        found_paths = []
        for path in paths:
            if path in container._root:
                arr = container._root[path][:]
                total_obs += np.sum(~np.isnan(arr), axis=0)
                found_any = True
                found_paths.append(path)
        if found_any:
            zero_idx = np.where(total_obs == 0)[0]
            uncovered[var] = sorted(field_uids[i] for i in zero_idx)
        else:
            uncovered[var] = []
        checked_paths[var] = found_paths

    uncovered["all"] = sorted(set().union(*uncovered.values()))
    uncovered["_checked_paths"] = checked_paths
    return uncovered


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def partition_fields(
    shapefile,
    feature_id_col,
    batch_size=50,
    grouping_column=None,
    exclude_fids=None,
    include_fids=None,
):
    """Partition fields into batches, optionally grouping by a column.

    When ``grouping_column`` is present in the shapefile, groups fields by
    that column and greedy bin-packs groups into batches. Otherwise falls
    back to simple sequential packing.

    Parameters
    ----------
    shapefile : str or Path
        Path to fields shapefile.
    feature_id_col : str
        Column name for field identifiers (e.g., "site_id", "FID").
    batch_size : int
        Target number of fields per batch.
    grouping_column : str or None
        Column for grid-cell grouping (e.g., "GFID"). None = sequential.
    exclude_fids : set[str] or None
        Field IDs to omit from all batches.
    include_fids : set[str] or None
        When given, only field IDs in this set are eligible (typically the
        container's field UIDs — the shapefile may cover a larger network
        than the container, and a batch with no container fields cannot
        build).

    Returns
    -------
    list[list[str]]
        Each inner list is a batch of field ID strings.
    """
    exclude_fids = set(exclude_fids or [])
    include_fids = set(include_fids) if include_fids is not None else None

    def _keep(fid):
        if fid in exclude_fids:
            return False
        return include_fids is None or fid in include_fids

    gdf = gpd.read_file(str(shapefile), engine="fiona")

    if feature_id_col not in gdf.columns:
        raise ValueError(
            f"Feature ID column '{feature_id_col}' not found in {shapefile}. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.drop_duplicates(subset=feature_id_col, keep="first")
    has_grouping = grouping_column is not None and grouping_column in gdf.columns

    if has_grouping:
        groups: dict[str, list[str]] = {}
        for _, row in gdf.iterrows():
            fid = coerce_fid(row[feature_id_col])
            if not _keep(fid):
                continue
            gfid = coerce_fid(row[grouping_column])
            groups.setdefault(gfid, []).append(fid)

        try:
            sorted_keys = sorted(groups.keys(), key=int)
        except ValueError:
            sorted_keys = sorted(groups.keys())

        batches: list[list[str]] = []
        current_batch: list[str] = []
        for gfid in sorted_keys:
            fids = groups[gfid]
            if current_batch and len(current_batch) + len(fids) > batch_size:
                batches.append(current_batch)
                current_batch = []
            current_batch.extend(fids)
        if current_batch:
            batches.append(current_batch)
    else:
        all_fids = [
            coerce_fid(row[feature_id_col])
            for _, row in gdf.iterrows()
            if _keep(coerce_fid(row[feature_id_col]))
        ]
        batches = [all_fids[i : i + batch_size] for i in range(0, len(all_fids), batch_size)]

    return batches


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def read_manifest(output_root):
    """Read batch_manifest.csv, return DataFrame."""
    manifest_path = Path(output_root) / "batch_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def write_manifest(output_root, batches, feature_id_col="FID"):
    """Write batch_manifest.csv from a list of batches.

    Parameters
    ----------
    output_root : str or Path
        Output directory.
    batches : list[list[str]]
        Batches of field IDs.
    feature_id_col : str
        Column name for field IDs in the manifest.

    Returns
    -------
    Path to the written manifest.
    """
    output_root = Path(output_root)
    rows = [
        {"batch_id": i, feature_id_col: fid} for i, batch in enumerate(batches) for fid in batch
    ]
    manifest_path = output_root / "batch_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def load_batches_from_manifest(output_root, feature_id_col="FID"):
    """Load manifest and return list of (batch_id, [fids])."""
    manifest = read_manifest(output_root)
    # The manifest column may be the feature_id_col or "FID" as fallback
    fid_col = feature_id_col if feature_id_col in manifest.columns else "FID"
    batch_ids = sorted(manifest["batch_id"].unique())
    return [
        (bid, manifest.loc[manifest["batch_id"] == bid, fid_col].astype(str).tolist())
        for bid in batch_ids
    ]


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def create_run_manifest(
    output_root,
    container_path,
    toml_path,
    report,
    batches,
    noptmax,
    reals,
    workers,
    batch_size,
    override,
    feature_id_col,
    grouping_column,
    mask_mode,
    etf_target_model,
    project_name,
):
    """Write run_manifest.json at the start of calibrate_all."""
    run_id = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config_hash = None
    try:
        config_bytes = Path(toml_path).read_bytes()
        config_hash = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
    except Exception:
        pass

    if report is not None:
        fingerprint = report.container_fingerprint
        policy_version = report.policy_version
        gate_outcome = "PASS" if report.passed else ("OVERRIDE" if override else "FAIL")
        gate_failures = [c.to_dict() for c in report.failures]
        gate_warnings = [c.message for c in report.warnings]
    else:
        fingerprint = "skipped"
        policy_version = "skipped"
        gate_outcome = "SKIPPED"
        gate_failures = []
        gate_warnings = []

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "container_path": str(container_path),
        "container_fingerprint": fingerprint,
        "config_path": str(toml_path),
        "config_hash": config_hash,
        "policy_version": policy_version,
        "feature_id_column": feature_id_col,
        "grouping_column": grouping_column,
        "mask_mode": mask_mode,
        "etf_target_model": etf_target_model,
        "gate_outcome": gate_outcome,
        "gate_failures": gate_failures,
        "gate_warnings": gate_warnings,
        "override": override,
        "parameters": {
            "noptmax": noptmax,
            "reals": reals,
            "workers": workers,
            "batch_size": batch_size,
            "n_batches": len(batches),
            "n_fields": sum(len(b[1]) if isinstance(b, tuple) else len(b) for b in batches),
        },
    }

    manifest_path = Path(output_root) / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Run manifest: {manifest_path}")
    return run_id


# ---------------------------------------------------------------------------
# Ingested batch tracking
# ---------------------------------------------------------------------------


def ingested_batch_ids(container_path, output_root):
    """Return the set of batch IDs already ingested into the container or log."""
    from swimrs.container.container import SwimContainer

    ingested = {
        bid
        for bid, entry in read_batch_log(output_root).items()
        if entry.get("status") == "ingested"
    }
    try:
        container = SwimContainer.open(str(container_path), mode="r")
        try:
            if "calibration" in container._root:
                batches_str = container._root["calibration"].attrs.get("batches", "{}")
                ingested.update(json.loads(batches_str).keys())
        finally:
            container.close()
    except Exception:
        pass
    return ingested


def all_manifest_batches_ingested(container_path, output_root):
    """Check whether every batch in the manifest has been ingested."""
    manifest_path = Path(output_root) / "batch_manifest.csv"
    if not manifest_path.exists():
        return False, set()
    manifest = pd.read_csv(manifest_path)
    expected = {str(int(batch_id)) for batch_id in manifest["batch_id"].unique()}
    ingested = ingested_batch_ids(container_path, output_root)
    missing = expected - ingested
    return not missing, missing


# ---------------------------------------------------------------------------
# Resolved restart state
# ---------------------------------------------------------------------------


def persist_calibration_resolved_state(
    container_path, toml_path, output_root, *, command="batch_calibrate"
):
    """Persist the canonical post-calibration restart run in the container.

    Only runs fields that were actually calibrated (from batch manifest),
    excluding any with NaN calibration parameters.
    Only runs if all manifest batches are ingested.
    """
    import numpy as np

    from swimrs.container.container import SwimContainer
    from swimrs.swim.config import ProjectConfig

    all_ingested, missing = all_manifest_batches_ingested(container_path, output_root)
    if not all_ingested:
        missing_str = ", ".join(sorted(missing, key=lambda x: int(x) if x.isdigit() else x)[:10])
        print(
            "Skipping calibration resolved state: not all manifest batches are ingested"
            + (f" (missing: {missing_str})" if missing_str else "")
        )
        return False

    config = ProjectConfig()
    config.read_config(str(toml_path), calibrate=True)

    container = SwimContainer.open(str(container_path), mode="r+")
    try:
        all_uids = container.field_uids
        root = container._root
        met_source = getattr(config, "met_source", "gridmet") or "gridmet"

        # Resolve run fields: exclude uncalibrated (NaN aw) fields
        try:
            aw = root["calibration/parameters/aw"][:]
            run_uids = [uid for uid, val in zip(all_uids, aw) if not np.isnan(val)]
        except KeyError:
            run_uids = list(all_uids)

        if len(run_uids) < len(all_uids):
            n_skip = len(all_uids) - len(run_uids)
            print(
                f"Resolved state: running {len(run_uids)} calibrated fields "
                f"(skipping {n_skip} uncalibrated)"
            )

        run_kwargs = dict(
            run_id="calibration_resolved_state",
            profile="state_only",
            overwrite=True,
            engine="python",
            refet_type=getattr(config, "refet_type", "eto") or "eto",
            etf_model=getattr(config, "etf_target_model", "ssebop") or "ssebop",
            met_source=met_source,
            mask_mode=getattr(config, "mask_mode", "irrigation") or "irrigation",
            ndvi_mode="observed",
            max_irr_rate=getattr(config, "max_irr_rate", 100.0) or 100.0,
            fields=run_uids,
            command=command,
            run_attrs={
                "run_role": "resolved_restart",
                "source_context": "post_calibration",
            },
            use_default_restart=False,
        )

        try:
            container.run(**run_kwargs)
        except ValueError as exc:
            if "Non-finite state" not in str(exc):
                raise
            # Some fields produce NaN during the forward run due to NaN met,
            # NaN soil properties, or extreme calibrated parameter values.
            # Identify bad fields by inspecting container data directly
            # (running the model would also crash), then retry without them.
            print(f"WARNING: resolved state hit NaN state: {exc}")
            refet_type = run_kwargs["refet_type"]
            bad_fields = _find_nan_fields(root, all_uids, run_uids, met_source, refet_type)
            if bad_fields:
                safe_uids = [u for u in run_uids if u not in set(bad_fields)]
                if not safe_uids:
                    raise ValueError(
                        f"All {len(run_uids)} fields have NaN inputs — cannot run resolved state"
                    ) from exc
                print(f"  Dropping {len(bad_fields)} NaN-input fields: {bad_fields}")
                print(f"  Retrying with {len(safe_uids)} fields")
                run_kwargs["fields"] = safe_uids
                run_kwargs["overwrite"] = True
                container.run(**run_kwargs)
            else:
                raise

        container.runs.set_default_restart("calibration_resolved_state")
        container.save()
    finally:
        container.close()

    print("Persisted calibration resolved restart state: calibration_resolved_state")
    return True


def _find_nan_fields(root, all_uids, candidate_uids, met_source="gridmet", refet_type="eto"):
    """Identify fields with NaN in critical container arrays.

    Checks the active meteorology source, soil properties (AWC), and
    calibrated parameters for any NaN values that would cause the forward
    model to produce non-finite state. Does not run the model.

    Parameters
    ----------
    met_source : str
        Active met source ("era5" or "gridmet"). Only that source is checked.
    refet_type : str
        Reference ET type ("eto" or "etr"). Checks the corrected variant first
        (e.g. "eto_corr"), falling back to the raw variant — matching the
        precedence in build_swim_input.
    """
    bad = set()
    candidate_set = set(candidate_uids)

    # Check all met variables for the active source — any NaN can propagate.
    # For refET, check the corrected variant first (what the model actually uses).
    met_key = "era5" if met_source == "era5" else "gridmet"
    refet_variants = [f"{refet_type}_corr", refet_type]
    other_vars = ["prcp", "tmin", "tmax", "srad"]

    for var in refet_variants:
        try:
            arr = root[f"meteorology/{met_key}/{var}"][:]
            for i, uid in enumerate(all_uids):
                if uid not in candidate_set or uid in bad:
                    continue
                if np.any(np.isnan(arr[:, i])):
                    bad.add(uid)
            break  # found the refET array — don't check fallback
        except KeyError:
            continue

    for var in other_vars:
        try:
            arr = root[f"meteorology/{met_key}/{var}"][:]
            for i, uid in enumerate(all_uids):
                if uid not in candidate_set or uid in bad:
                    continue
                if np.any(np.isnan(arr[:, i])):
                    bad.add(uid)
        except KeyError:
            pass

    # Check AWC from properties
    try:
        awc = root["properties/soils/awc"][:]
        for i, uid in enumerate(all_uids):
            if uid not in candidate_set:
                continue
            if np.isnan(awc[i]) or awc[i] <= 0:
                bad.add(uid)
    except KeyError:
        pass

    return sorted(bad)


# ---------------------------------------------------------------------------
# NaN spinup FID parsing
# ---------------------------------------------------------------------------


def parse_nan_fids(exc_msg):
    """Parse bad field IDs from a NaN spinup ValueError message.

    Returns (bad_fids, n_expected) where n_expected is the count from the
    error message (may exceed len(bad_fids) if the list was truncated).
    """
    match = re.search(r"\[([^\]]+)\]", exc_msg)
    if match:
        bad_fids = re.findall(r"'([^']+)'", match.group(1))
    else:
        bad_fids = []

    count_match = re.search(r"(\d+) field\(s\)", exc_msg)
    n_expected = int(count_match.group(1)) if count_match else len(bad_fids)
    return bad_fids, n_expected


# ---------------------------------------------------------------------------
# Find par CSV
# ---------------------------------------------------------------------------


def _par_csv_iteration(path):
    """Extract numeric iteration from a .par.csv filename.

    Filenames follow the pattern ``project.N.par.csv`` where N is the
    PEST++ iteration number.  Returns N as int, or -1 if unparseable.
    """
    parts = path.stem.replace(".par", "").rsplit(".", 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


def _best_phi_iteration(master_dir, max_iteration=None):
    """Return the iteration with minimum mean measured phi, or None.

    Reads {project}.phi.meas.csv from the master dir. Returns None when the
    file is missing or unreadable so callers can fall back to the
    latest-iteration heuristic.

    ``max_iteration`` restricts the choice to iterations at or below N, which
    reproduces what a ``noptmax N`` run would have selected: IES iterations are
    sequential, so a shorter run is a prefix of a longer one.
    """
    phi_files = list(Path(master_dir).glob("*.phi.meas.csv"))
    if not phi_files:
        return None
    try:
        df = pd.read_csv(phi_files[0])
    except Exception:
        return None
    if df.empty or "iteration" not in df.columns or "mean" not in df.columns:
        return None
    valid = df.dropna(subset=["mean"])
    if max_iteration is not None:
        valid = valid.loc[valid["iteration"] <= max_iteration]
    if valid.empty:
        return None
    return int(valid.loc[valid["mean"].idxmin(), "iteration"])


def find_par_csv(batch_dir, max_iteration=None):
    """Find the best .par.csv for a batch.

    Accepts either a batch directory (reads its master/ subdirectory) or a
    flat pest_archive directory. Both .par.csv and .phi.meas.csv are in
    PEST_ARCHIVE_PATTERNS and survive every retention tier, so the archive
    supports the same selection once the batch directory is gone.

    Selects the iteration with minimum mean measured phi from
    {project}.phi.meas.csv (the last iteration is not always the best).
    Falls back to the highest-numbered iteration when no phi history is
    available.

    ``max_iteration`` caps the iterations considered, so an archive from a
    ``noptmax 3`` run yields exactly what a ``noptmax 2`` run would have
    ingested. PEST++-IES iterations are sequential and the seed is fixed, so
    the shorter run is a genuine prefix — re-ingesting under a cap is
    equivalent to recalibrating, at no compute cost.
    """
    batch_dir = Path(batch_dir)
    master = batch_dir / "master"
    if not master.exists():
        master = batch_dir
    par_files = list(master.glob("*.par.csv"))
    if max_iteration is not None:
        par_files = [f for f in par_files if 0 <= _par_csv_iteration(f) <= max_iteration]
    if not par_files:
        return None
    best_iter = _best_phi_iteration(master, max_iteration=max_iteration)
    if best_iter is not None:
        for f in par_files:
            if _par_csv_iteration(f) == best_iter:
                return f
    return max(par_files, key=_par_csv_iteration)


def batch_is_built(batch_dir):
    """Check if batch_dir/pest/*.pst exists."""
    pest_dir = Path(batch_dir) / "pest"
    return pest_dir.exists() and any(pest_dir.glob("*.pst"))


# ---------------------------------------------------------------------------
# PEST++ output archiving (RUN_POLICY Categories 3/4)
# ---------------------------------------------------------------------------

# Artifacts that must survive batch-directory deletion: the control file,
# record file, phi histories, ALL iteration parameter/observation ensembles
# (Category 4 requires the full trajectory, .0 prior through .noptmax final),
# prior-data conflict reports, and the localizer.
PEST_ARCHIVE_PATTERNS = [
    "*.pst",
    "*.rec",
    "*.phi.meas.csv",
    "*.phi.actual.csv",
    "*.phi.composite.csv",
    # per-observation-group phi: one column per field per obs type, the only
    # per-field fit-quality artifact PEST++ emits (~300 KB/batch)
    "*.phi.group.csv",
    "*.par.csv",
    "*.obs.csv",
    "*.pdc.csv",
    "loc.mat",
    "localizer_summary.json",
    "params.csv",
    # version-2 pst external tables — without these the archived .pst cannot
    # be reloaded by pyemu (obs weights/noise unverifiable after cleanup)
    "*.obs_data.csv",
    "*.par_data.csv",
    "*.pargp_data.csv",
    "*.insfile_data.csv",
    "*.tplfile_data.csv",
    "*.obs+noise.csv",
    "*.base.rei",
    # per-build ETf weight decomposition (PestBuilder.export_weight_audit);
    # required by RUN_POLICY Category 3 for auxiliary-source runs
    "weight_audit.csv",
]


def archive_pest_outputs(batch_dir, archive_dir):
    """Copy RUN_POLICY Category 3/4 PEST++ artifacts out of a batch build dir.

    Searches the batch's pest/ and master/ subdirectories for
    PEST_ARCHIVE_PATTERNS and copies matches into archive_dir. Must run
    BEFORE PestResults.cleanup() (which deletes intermediate iteration
    ensembles) and before any batch-directory deletion.

    Returns the sorted list of archived filenames.
    """
    batch_dir = Path(batch_dir)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    copied = set()
    for sub in ("pest", "master"):
        src_dir = batch_dir / sub
        if not src_dir.exists():
            continue
        for pattern in PEST_ARCHIVE_PATTERNS:
            for src in src_dir.glob(pattern):
                dst = archive_dir / src.name
                if src.name not in copied:
                    shutil.copy2(src, dst)
                    copied.add(src.name)
    return sorted(copied)


ARCHIVE_RETENTION_TIERS = ("full", "reference", "slim")

_OBS_ITER_RE = re.compile(r"\.(\d+)\.obs\.csv$")
_REI_ITER_RE = re.compile(r"\.(\d+)\.base\.rei$")
_PAR_ITER_RE = re.compile(r"\.(\d+)\.par\.csv$")

# Observation group names are emitted by the PEST builder as
# "oname:obs/obs_{type}_{fid}.np_otype:arr" (lowercased by PEST++).
_REI_GROUP_RE = re.compile(r"obs_(.+?)\.np")


def _parse_rei(path):
    """Read a PEST .rei residual file, keeping only weighted observations.

    The file has three preamble lines, then a header row of
    Name/Group/Measured/Modelled/Residual/Weight. Zero-weight rows are
    no-observation fill (Measured = -99) and carry no information.
    """
    df = pd.read_csv(path, sep=r"\s+", skiprows=3)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.loc[df["weight"] > 0]


def _rei_group_keys(groups):
    """Split PEST observation-group names into (obs_type, lowercase fid)."""
    bodies = groups.str.extract(_REI_GROUP_RE, expand=False)
    split = bodies.str.split("_", n=1, expand=True)
    return split[0], split[1]


def _fit_stats(df, fids):
    """Per (fid, obs type) fit statistics from a parsed .rei frame."""
    obs_type, fid_key = _rei_group_keys(df["group"])
    # PEST lowercases observation names; map back to the container's FIDs.
    lookup = {str(f).lower(): str(f) for f in fids}
    out = pd.DataFrame(
        {
            "fid": fid_key.map(lookup).fillna(fid_key),
            "obs_type": obs_type,
            "residual": df["residual"].to_numpy(),
            "phi": (df["weight"].to_numpy() * df["residual"].to_numpy()) ** 2,
        }
    )
    grouped = out.groupby(["fid", "obs_type"], sort=True)
    return pd.DataFrame(
        {
            "n_obs": grouped["residual"].size(),
            "phi": grouped["phi"].sum(),
            # PEST defines Residual = Measured - Modelled, so a positive bias
            # means the model runs dry against the observation.
            "bias": grouped["residual"].mean(),
            "rmse": grouped["residual"].apply(lambda r: float(np.sqrt((r**2).mean()))),
        }
    )


def write_field_fit_summary(archive_dir, fids):
    """Write per-field fit quality distilled from the archived residuals.

    The .rei residual files are 100-300 MB per batch and every retention tier
    below "full" discards them, but per-field fit quality is what makes a
    posterior parameter set usable as a training label downstream. This
    collapses them to a few KB: one row per field per observation type, with
    prior and posterior phi, RMSE and bias.

    Must run after archive_pest_outputs and BEFORE apply_archive_retention.
    Returns the path written, or None when no residuals are present.
    """
    archive_dir = Path(archive_dir)

    prior_path = None
    final_path = None
    final_iter = -1
    for f in archive_dir.glob("*.base.rei"):
        m = _REI_ITER_RE.search(f.name)
        if m is None:
            continue
        i = int(m.group(1))
        if i == 0:
            prior_path = f
        if i > final_iter:
            final_iter, final_path = i, f

    if final_path is None:
        # Fall back to the unnumbered final-iteration residual file.
        unnumbered = [
            f for f in archive_dir.glob("*.base.rei") if _REI_ITER_RE.search(f.name) is None
        ]
        if not unnumbered:
            return None
        final_path = unnumbered[0]

    summary = _fit_stats(_parse_rei(final_path), fids)
    summary = summary.rename(columns={"phi": "phi_post", "bias": "bias_post", "rmse": "rmse_post"})

    if prior_path is not None and prior_path != final_path:
        prior = _fit_stats(_parse_rei(prior_path), fids)[["phi", "rmse", "bias"]]
        prior.columns = ["phi_prior", "rmse_prior", "bias_prior"]
        summary = summary.join(prior)

    out_path = archive_dir / "field_fit_summary.csv"
    summary.reset_index().to_csv(out_path, index=False)
    return out_path


def apply_archive_retention(archive_dir, tier):
    """Prune a batch archive to its RUN_POLICY Category 4 retention tier.

    Must run only on a batch that converged cleanly (the caller gates on
    PestResults.is_successful) — a problem batch keeps its full archive for
    debugging regardless of tier.

    Tiers:
      full      -- keep everything (publication runs); no-op.
      reference -- drop intermediate-iteration obs ensembles, intermediate
                   residuals, and loc.mat; keep the prior (.0) and final
                   obs ensembles (Esmeralda/32009 demonstration archive).
      slim      -- additionally drop ALL obs ensembles, obs+noise, obs_data
                   tables, and residuals, and gzip the .rec run record.
                   Everything dropped is regenerable from the container +
                   archived config/SHA (NWI statewide default).

    Returns the sorted list of removed (or gzipped) filenames.
    """
    if tier not in ARCHIVE_RETENTION_TIERS:
        raise ValueError(
            f"archive_retention must be one of {ARCHIVE_RETENTION_TIERS}, got {tier!r}"
        )
    archive_dir = Path(archive_dir)
    if tier == "full":
        return []

    par_iters = [
        int(m.group(1)) for f in archive_dir.iterdir() if (m := _PAR_ITER_RE.search(f.name))
    ]
    if not par_iters:
        raise RuntimeError(
            f"No iteration .par.csv files in {archive_dir}; refusing to prune an "
            "archive whose iteration structure cannot be determined."
        )
    final_iter = max(par_iters)

    removed = []
    for f in sorted(archive_dir.iterdir()):
        name = f.name
        obs_m = _OBS_ITER_RE.search(name)
        rei_m = _REI_ITER_RE.search(name)
        drop = False
        if name == "loc.mat":
            drop = True
        elif obs_m:
            i = int(obs_m.group(1))
            drop = (0 < i < final_iter) if tier == "reference" else True
        elif rei_m:
            drop = int(rei_m.group(1)) < final_iter if tier == "reference" else True
        elif tier == "slim":
            if name.endswith(("obs+noise.csv", "obs_data.csv")) or name.endswith(".base.rei"):
                drop = True
            elif name.endswith(".rec"):
                with open(f, "rb") as src, gzip.open(f"{f}.gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                f.unlink()
                removed.append(name)
                continue
        if drop:
            f.unlink()
            removed.append(name)
    return sorted(removed)


def resolve_ingested_batches(container_ingested, batch_log, override, container_path, output_root):
    """Guard against stale-calibration contamination on resume (C-2).

    A container with ingested batches but no local batch log means its
    calibration/ group was created by a different run (e.g. a copied
    calibrated container). Trusting it would make --resume skip every batch.

    Returns the effective ingested-batch set (empty when overriding a stale
    group). Raises RuntimeError on the stale signature unless override=True.
    """
    if container_ingested and not batch_log:
        msg = (
            f"Container {container_path} already has {len(container_ingested)} "
            f"ingested batch(es) but {output_root}/batch_log.json has no record of "
            "them — its calibration/ group was created by a different run. Delete "
            "the container's calibration group (or use a clean container copy) "
            "before calibrating, or pass --override to ignore the stale group."
        )
        if not override:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
        print("WARNING: --override set; ignoring container batch state for resume.")
        return set()
    return container_ingested


def verify_pest_archive(archive_dir):
    """Verify an archive holds the minimum artifacts to permit batch-dir deletion.

    Requires at least one .pst, one .rec, and one .par.csv. Returns
    (ok, missing) where missing lists the absent artifact kinds.
    """
    archive_dir = Path(archive_dir)
    missing = []
    if not archive_dir.exists():
        return False, ["archive directory"]
    # .rec may be gzipped by the slim retention tier
    for kind, patterns in [
        ("pst", ["*.pst"]),
        ("rec", ["*.rec", "*.rec.gz"]),
        ("par.csv", ["*.par.csv"]),
    ]:
        if not any(any(archive_dir.glob(p)) for p in patterns):
            missing.append(kind)
    return not missing, missing


# ========================= EOF ====================================================================

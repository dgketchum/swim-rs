"""Native batch calibration runner for SWIM-RS.

Config-driven batch calibration that partitions fields into batches,
builds PEST++ setups, runs them, ingests results, and cleans up.

Usage:
    python -m swimrs.calibrate.batch_runner --config /path/to/project.toml --action calibrate-all
    python -m swimrs.calibrate.batch_runner --config /path/to/project.toml --action calibrate-all --resume
    python -m swimrs.calibrate.batch_runner --config /path/to/project.toml --action status
"""

import argparse
import json
import multiprocessing
import os
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Runtime context: resolved from ProjectConfig
# ---------------------------------------------------------------------------


@dataclass
class BatchContext:
    """Resolved runtime context for batch calibration.

    All values are derived from the SWIM-RS .toml config, with optional
    CLI overrides for container, output, and shapefile paths.
    """

    project_name: str
    container_path: str
    fields_shapefile: str
    feature_id_col: str
    grouping_column: str | None
    grouping_shapefile: str | None
    output_root: str
    mask_mode: str
    etf_target_model: str
    etf_ensemble_members: list[str] | None
    etf_target_instrument: str
    workers: int
    realizations: int
    noptmax: int
    batch_size: int
    toml_path: str
    prior_params_path: str | None = None
    archive_retention: str = "full"


def resolve_context(
    toml_path,
    *,
    container_override=None,
    output_override=None,
    shapefile_override=None,
    batch_size=None,
    workers=None,
    realizations=None,
    noptmax=None,
    prior_params_path=None,
) -> BatchContext:
    """Build a BatchContext from a SWIM-RS .toml config file.

    Parameters
    ----------
    toml_path : str
        Path to project TOML configuration.
    container_override, output_override, shapefile_override : str, optional
        CLI overrides for paths that would otherwise come from config.
    batch_size, workers, realizations, noptmax : int, optional
        Override the corresponding config values. When None, the TOML's
        [calibration] value is used, then the built-in default.
    """
    from swimrs.calibrate.batch_support import ARCHIVE_RETENTION_TIERS
    from swimrs.swim.config import ProjectConfig

    config = ProjectConfig()
    config.read_config(str(toml_path), calibrate=True)

    archive_retention = (getattr(config, "archive_retention", None) or "full").lower()
    if archive_retention not in ARCHIVE_RETENTION_TIERS:
        raise ValueError(
            f"archive_retention must be one of {ARCHIVE_RETENTION_TIERS}, got {archive_retention!r}"
        )

    container_path = container_override or config.container_path
    if not container_path:
        data_root = config.data_dir or os.path.dirname(os.path.abspath(toml_path))
        container_path = os.path.join(data_root, f"{config.project_name or 'swim'}.swim")

    fields_shp = shapefile_override or config.fields_shapefile

    # Grouping column: use gridmet_id_col (GFID) when available
    grouping_col = config.gridmet_id_col

    # Grouping shapefile: when a grouping column is configured, use the
    # gridmet mapping shapefile (which contains the GFID column) rather than
    # the plain fields shapefile.  Fall back to fields_shp if no mapping exists.
    grouping_shp = None
    if grouping_col and not shapefile_override:
        mapping_shp = getattr(config, "gridmet_mapping_shp", None)
        if mapping_shp and os.path.exists(mapping_shp):
            grouping_shp = mapping_shp

    # Output root: CLI override > config pest_run_dir > project workspace
    if output_override:
        output_root = output_override
    elif config.pest_run_dir:
        output_root = config.pest_run_dir
    else:
        output_root = os.path.join(config.project_ws or ".", "pestrun")

    return BatchContext(
        project_name=config.project_name or "swim",
        container_path=str(container_path),
        fields_shapefile=str(fields_shp),
        feature_id_col=config.feature_id_col,
        grouping_column=grouping_col,
        grouping_shapefile=grouping_shp,
        output_root=str(output_root),
        mask_mode=config.mask_mode or "none",
        etf_target_model=config.etf_target_model or "ssebop",
        etf_ensemble_members=config.etf_ensemble_members,
        etf_target_instrument=config.etf_target_instrument or "landsat",
        workers=workers or config.workers or 10,
        realizations=realizations or config.realizations or 200,
        noptmax=noptmax or config.noptmax or 3,
        batch_size=batch_size or config.batch_size or 50,
        toml_path=str(os.path.abspath(toml_path)),
        prior_params_path=str(os.path.abspath(prior_params_path)) if prior_params_path else None,
        archive_retention=archive_retention,
    )


# ---------------------------------------------------------------------------
# Preflight health gate
# ---------------------------------------------------------------------------


def preflight_gate(ctx: BatchContext, override=False):
    """Run container health check and block on FAIL.

    Writes health artifacts under output_root/health/<timestamp>/.
    Returns HealthReport. Raises on FAIL unless override=True.
    """
    from swimrs.container.container import SwimContainer
    from swimrs.container.health import health_report_output_dir
    from swimrs.swim.config import ProjectConfig

    config = ProjectConfig()
    config.read_config(ctx.toml_path, calibrate=True)
    container = SwimContainer.open(ctx.container_path, mode="r+")
    try:
        report = container.report(
            config=config,
            raise_on_fail=(not override),
            output_dir=str(
                health_report_output_dir(ctx.container_path, base_dir=Path(ctx.output_root))
            ),
            health_profile="calibration",
        )
        if not report.passed and override:
            print(f"WARNING: {len(report.failures)} FAIL(s) overridden by --override flag")
            override_record = {
                "timestamp": datetime.now().isoformat(),
                "failures": [c.to_dict() for c in report.failures],
                "user_override": True,
            }
            (Path(ctx.output_root) / "override_log.json").write_text(
                json.dumps(override_record, indent=2)
            )
        return report
    finally:
        container.close()


def _verify_prior_health(ctx: BatchContext):
    """Verify a prior health check exists in the container for --skip-health."""
    from swimrs.container.container import SwimContainer
    from swimrs.container.health import fingerprint_container

    c = SwimContainer.open(ctx.container_path, mode="r")
    try:
        last_hc = c._root.attrs.get("last_health_check")
        if not last_hc:
            raise RuntimeError(
                "No prior health check found in container. "
                "Run 'swim prep' first (without --skip-health), "
                "or remove --skip-health from this command."
            )

        current_fp = fingerprint_container(c._root, c._field_uids)
        stored_fp = last_hc.get("fingerprint", "")
        if current_fp != stored_fp:
            print(
                f"WARNING: container fingerprint changed since last health check "
                f"(stored={stored_fp[:8]}… current={current_fp[:8]}…). "
                f"Consider re-running without --skip-health."
            )

        if not last_hc.get("passed", False):
            print(
                f"WARNING: last health check FAILED "
                f"({last_hc.get('n_fail', '?')} failures at {last_hc.get('timestamp', '?')})"
            )

        print(
            f"Using prior health check from {last_hc.get('timestamp', '?')} "
            f"(fingerprint={stored_fp[:8]}…, passed={last_hc.get('passed')})"
        )
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Build a single batch
# ---------------------------------------------------------------------------


def _do_build(config, container, batch_id, noptmax, reals, prior_params_path=None):
    """Run the PestBuilder sequence. Returns on success, raises on failure."""
    from swimrs.calibrate.pest_builder import PestBuilder

    builder = PestBuilder(config, container)
    try:
        n = len(container._field_uids)
        print(f"  Batch {batch_id:03d}: spinup ({n} fields)...")
        builder.spinup()
        print(f"  Batch {batch_id:03d}: build_pest...")
        builder.build_pest(
            target_etf=config.etf_target_model or "ssebop",
            members=config.etf_ensemble_members,
        )
        if prior_params_path is not None:
            print(f"  Batch {batch_id:03d}: add_regularization (prior: {prior_params_path})...")
            builder.apply_prior_params(prior_params_path)
            builder.add_regularization()
        print(f"  Batch {batch_id:03d}: build_localizer...")
        builder.build_localizer()
        print(f"  Batch {batch_id:03d}: write_control_settings...")
        builder.write_control_settings(noptmax=noptmax, reals=reals)
        print(f"  Batch {batch_id:03d}: done.")
    finally:
        builder.close()


def _open_and_subset(toml_path, container_path, batch_dir, fids):
    """Open config and container, subset to the given FIDs."""
    from swimrs.container.container import SwimContainer
    from swimrs.swim.config import ProjectConfig

    config = ProjectConfig()
    config.read_config(str(toml_path), calibrate=True, calibration_dir_override=str(batch_dir))
    bd = str(batch_dir)
    config.pest_run_dir = bd
    config.obs_folder = str(Path(batch_dir) / "obs")
    config.spinup = str(Path(batch_dir) / "spinup.json")
    config.initial_values_csv = str(Path(batch_dir) / "params.csv")
    container = SwimContainer.open(str(container_path), mode="r")
    fid_set = set(fids)
    container._field_uids = [uid for uid in container._field_uids if uid in fid_set]
    if not container._field_uids:
        container.close()
        raise ValueError(
            f"None of the {len(fid_set)} batch field(s) exist in container "
            f"{container_path}; the batch manifest likely predates the container "
            f"or was partitioned from a shapefile covering a different network."
        )
    return config, container


def build_batch(ctx: BatchContext, batch_fids, batch_id):
    """Build PEST++ setup for a single batch of fields.

    Returns dict with status, n_fields, dropped_fids.
    Catches NaN spinup errors, drops bad FIDs, and retries once.
    """
    from swimrs.calibrate.batch_support import parse_nan_fids

    batch_dir = Path(ctx.output_root) / f"batch_{batch_id:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    dropped_fids = []

    config, container = _open_and_subset(ctx.toml_path, ctx.container_path, batch_dir, batch_fids)

    try:
        _do_build(
            config,
            container,
            batch_id,
            ctx.noptmax,
            ctx.realizations,
            prior_params_path=ctx.prior_params_path,
        )
        return {
            "status": "built",
            "n_fields": len(batch_fids),
            "dropped_fids": dropped_fids,
        }
    except ValueError as exc:
        if "NaN state" not in str(exc) and "Non-finite state" not in str(exc):
            return {
                "status": "build_failed",
                "n_fields": len(batch_fids),
                "dropped_fids": dropped_fids,
                "error": traceback.format_exc()[-4096:],
            }

        bad_fids, n_expected = parse_nan_fids(str(exc))

        if n_expected > len(bad_fids):
            return {
                "status": "build_failed",
                "n_fields": len(batch_fids),
                "dropped_fids": bad_fids,
                "error": f"Too many NaN fields ({n_expected}) to recover; skipping batch",
            }

        dropped_fids = bad_fids
        remaining = [f for f in batch_fids if f not in set(dropped_fids)]
        if not remaining:
            return {
                "status": "build_failed",
                "n_fields": len(batch_fids),
                "dropped_fids": dropped_fids,
                "error": "All fields had NaN spinup",
            }

        print(
            f"  Batch {batch_id:03d}: dropped {len(dropped_fids)} NaN FIDs "
            f"{dropped_fids}, retrying with {len(remaining)} fields..."
        )
    finally:
        container.close()

    # Retry with remaining fields (first container already closed above)
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    config, container = _open_and_subset(ctx.toml_path, ctx.container_path, batch_dir, remaining)
    try:
        _do_build(
            config,
            container,
            batch_id,
            ctx.noptmax,
            ctx.realizations,
            prior_params_path=ctx.prior_params_path,
        )
        return {
            "status": "built",
            "n_fields": len(remaining),
            "dropped_fids": dropped_fids,
        }
    except Exception:
        return {
            "status": "build_failed",
            "n_fields": len(remaining),
            "dropped_fids": dropped_fids,
            "error": traceback.format_exc()[-4096:],
        }
    finally:
        container.close()


def _build_batch_worker(queue, ctx_dict, batch_fids, batch_id):
    """Subprocess target: build a batch and put result on queue."""
    ctx = BatchContext(**ctx_dict)
    os.chdir(ctx.output_root)
    try:
        result = build_batch(ctx, batch_fids, batch_id)
        queue.put(("ok", batch_id, result))
    except Exception:
        queue.put(("error", batch_id, traceback.format_exc()[-4096:]))


# ---------------------------------------------------------------------------
# Run a single batch
# ---------------------------------------------------------------------------


def run_batch(batch_dir, num_workers=10, pst_name=None):
    """Run PEST++ IES for a single batch."""
    from swimrs.calibrate.run_pest import run_pst

    batch_dir = Path(batch_dir)
    pest_dir = batch_dir / "pest"
    master_dir = batch_dir / "master"
    workers_dir = batch_dir / "workers"

    if pst_name is None:
        pst_files = list(pest_dir.glob("*.pst"))
        if not pst_files:
            raise FileNotFoundError(f"No .pst file found in {pest_dir}")
        pst_name = pst_files[0].name

    print(f"Running PEST++ IES: {pest_dir / pst_name} with {num_workers} workers")
    run_pst(
        _dir=str(pest_dir),
        _cmd="pestpp-ies",
        pst_file=pst_name,
        num_workers=num_workers,
        worker_root=str(workers_dir),
        master_dir=str(master_dir),
    )


# ---------------------------------------------------------------------------
# Fanned-out batch task (stage 2 of prep -> array -> ingest)
# ---------------------------------------------------------------------------


def run_batch_task(ctx: BatchContext, batch_id, *, rebuild=False):
    """Build, run and archive one batch. Never opens the container for write.

    This is the body of a Slurm array task over batch ids. Everything that
    touches the container for write — parameter ingest and the `batches`
    attribute — is deferred to the serial `ingest-all` reduce step, because
    the zarr store takes a whole-container FileLock with a 3 s timeout and
    concurrent writers would simply fail.

    Idempotent: a batch whose shard is terminal and whose archive verifies is
    skipped, so a requeued or resubmitted task never redoes hours of PEST++.

    Returns 0 on success, 1 on failure.
    """
    from swimrs.calibrate.batch_support import (
        apply_archive_retention,
        archive_pest_outputs,
        batch_is_built,
        read_batch_shard,
        read_manifest,
        verify_pest_archive,
        write_batch_shard,
        write_field_fit_summary,
    )
    from swimrs.calibrate.pest_cleanup import PestResults

    output_root = Path(ctx.output_root)
    batch_dir = output_root / f"batch_{batch_id:03d}"
    archive_dir = output_root / "pest_archive" / f"batch_{batch_id:03d}"

    manifest = read_manifest(output_root)
    fid_col = ctx.feature_id_col if ctx.feature_id_col in manifest.columns else "FID"
    batch_fids = manifest.loc[manifest["batch_id"] == batch_id, fid_col].astype(str).tolist()
    if not batch_fids:
        raise ValueError(f"Batch {batch_id} has no fields in {output_root}/batch_manifest.csv")

    if not rebuild:
        shard = read_batch_shard(output_root, batch_id)
        if shard is not None and shard.get("status") in ("archived", "ingested"):
            archive_ok, _ = verify_pest_archive(archive_dir)
            if archive_ok:
                print(f"Batch {batch_id:03d}: already complete, skipping")
                return 0

    def _fail(status, error):
        write_batch_shard(
            output_root,
            batch_id,
            {
                "status": status,
                "n_fields": len(batch_fids),
                "dropped_fids": [],
                "error": error,
            },
        )
        print(f"Batch {batch_id:03d}: {status}")
        return 1

    # --- Build ---
    dropped_fids = []
    if rebuild or not batch_is_built(batch_dir):
        print(f"Batch {batch_id:03d}: building ({len(batch_fids)} fields)")
        build_result = build_batch(ctx, batch_fids, batch_id)
        if build_result["status"] != "built":
            return _fail("build_failed", build_result.get("error"))
        dropped_fids = build_result.get("dropped_fids", [])
        n_fields = build_result["n_fields"]
    else:
        print(f"Batch {batch_id:03d}: using existing build on disk")
        n_fields = len(batch_fids)

    # --- Run ---
    try:
        run_batch(batch_dir, num_workers=ctx.workers)
    except Exception:
        return _fail("run_failed", traceback.format_exc()[-4096:])

    pst_files = list((batch_dir / "pest").glob("*.pst")) or list(
        (batch_dir / "master").glob("*.pst")
    )
    if not pst_files:
        return _fail("run_failed", f"No .pst found under {batch_dir}")

    results = PestResults(
        str(batch_dir / "pest"), pst_files[0].stem, master_dir=str(batch_dir / "master")
    )
    success, issues = results.is_successful()
    summary = results.get_summary()

    # --- Archive (RUN_POLICY Cat 3/4) ---
    archived = archive_pest_outputs(batch_dir, archive_dir)
    print(f"Batch {batch_id:03d}: archived {len(archived)} artifact(s) -> {archive_dir}")

    fit_path = write_field_fit_summary(archive_dir, batch_fids)
    if fit_path is not None:
        print(f"Batch {batch_id:03d}: field fit summary -> {fit_path.name}")

    report = results.cleanup(archive_dir=str(archive_dir))
    print(f"Batch {batch_id:03d}: cleanup recovered {report['space_recovered_mb']:.1f} MB")

    if ctx.archive_retention != "full":
        if success:
            pruned = apply_archive_retention(archive_dir, ctx.archive_retention)
            print(f"Batch {batch_id:03d}: retention '{ctx.archive_retention}' pruned {len(pruned)}")
        else:
            print(
                f"Batch {batch_id:03d}: retention '{ctx.archive_retention}' skipped "
                "(did not converge, keeping full archive)"
            )

    if not success:
        return _fail("run_failed", f"PEST++ run did not succeed: {issues}")

    write_batch_shard(
        output_root,
        batch_id,
        {
            "status": "archived",
            "n_fields": n_fields,
            "dropped_fids": dropped_fids,
            "error": None,
            "archive_dir": str(archive_dir),
            "phi_initial": summary.get("phi_initial"),
            "phi_final": summary.get("phi_final"),
            "summary": summary,
        },
    )

    # Free the build directory only once the trajectory is safely archived.
    archive_ok, missing = verify_pest_archive(archive_dir)
    if archive_ok:
        shutil.rmtree(batch_dir)
        print(f"Batch {batch_id:03d}: archive verified, build directory removed")
    else:
        print(
            f"Batch {batch_id:03d}: WARNING — archive missing {missing}; "
            f"keeping build directory {batch_dir}"
        )
    return 0


# ---------------------------------------------------------------------------
# Ingest a single batch
# ---------------------------------------------------------------------------


def ingest_batch(ctx: BatchContext, batch_id, summary_stat="median", force=False):
    """Ingest calibrated parameters from one batch into the container.

    Refuses to ingest a batch whose PEST++ run did not succeed (per
    PestResults.is_successful) unless force=True.
    """
    from swimrs.calibrate.batch_support import (
        apply_archive_retention,
        archive_pest_outputs,
        find_par_csv,
        read_manifest,
        write_field_fit_summary,
    )
    from swimrs.calibrate.pest_cleanup import PestResults
    from swimrs.container.container import SwimContainer

    output_root = Path(ctx.output_root)
    manifest = read_manifest(output_root)
    fid_col = ctx.feature_id_col if ctx.feature_id_col in manifest.columns else "FID"
    batch_fids = manifest.loc[manifest["batch_id"] == batch_id, fid_col].astype(str).tolist()
    if not batch_fids:
        print(f"No fields found for batch {batch_id} in manifest.")
        return

    batch_dir = output_root / f"batch_{batch_id:03d}"
    master_dir = batch_dir / "master"
    pst_files = list((batch_dir / "pest").glob("*.pst"))
    if not pst_files:
        pst_files = list(master_dir.glob("*.pst"))

    # Gate on PEST++ success before touching the container.
    results = None
    if pst_files:
        project_name = pst_files[0].stem
        results = PestResults(str(batch_dir / "pest"), project_name, master_dir=str(master_dir))
        success, issues = results.is_successful()
        if not success and not force:
            raise RuntimeError(
                f"Batch {batch_id:03d}: PEST++ run did not succeed, refusing to ingest "
                f"(pass force=True / --force-ingest to override). Issues: {issues}"
            )
        if not success:
            print(f"Batch {batch_id:03d}: WARNING — ingesting despite issues: {issues}")

    par_csv = find_par_csv(batch_dir)
    if par_csv is None:
        print(f"No .par.csv found in {batch_dir}/master/")
        return

    container = SwimContainer.open(ctx.container_path, mode="r+")
    try:
        container.ingest.calibration(
            par_csv, fields=batch_fids, batch_id=batch_id, summary_stat=summary_stat
        )
        print(f"Batch {batch_id:03d}: ingested {len(batch_fids)} fields from {par_csv.name}")

        if results is not None:
            summary = results.get_summary()

            cal_group = container._root["calibration"]
            batches_meta = json.loads(cal_group.attrs.get("batches", "{}"))
            batches_meta[str(batch_id)] = {
                "n_fields": len(batch_fids),
                "status": summary.get("status", "unknown"),
                "phi_initial": summary.get("phi_initial"),
                "phi_final": summary.get("phi_final"),
                "phi_reduction_pct": summary.get("phi_reduction_pct"),
                "phi_history": summary.get("phi_history"),
                "noptmax": summary.get("noptmax"),
                "iterations_completed": summary.get("iterations_completed"),
            }
            cal_group.attrs["batches"] = json.dumps(batches_meta)

            phi_red = summary.get("phi_reduction_pct", 0)
            print(f"  Phi reduction: {phi_red:.1f}%")

            # Archive the full PEST++ trajectory (RUN_POLICY Cat 3/4) before
            # cleanup deletes intermediate iteration ensembles.
            archive_dir = output_root / "pest_archive" / f"batch_{batch_id:03d}"
            archived = archive_pest_outputs(batch_dir, archive_dir)
            print(f"  Archived {len(archived)} PEST++ artifact(s) -> {archive_dir}")

            # Distil per-field fit quality while the residuals still exist —
            # every retention tier below "full" discards them.
            fit_path = write_field_fit_summary(archive_dir, batch_fids)
            if fit_path is not None:
                print(f"  Field fit summary -> {fit_path.name}")

            report = results.cleanup(archive_dir=str(archive_dir))
            print(f"  Cleanup: {report['space_recovered_mb']:.1f} MB recovered")

            # Prune the archive to the configured retention tier (RUN_POLICY
            # Cat 4). Non-converged batches keep the full archive for debugging.
            if ctx.archive_retention != "full":
                if success:
                    pruned = apply_archive_retention(archive_dir, ctx.archive_retention)
                    print(
                        f"  Retention '{ctx.archive_retention}': "
                        f"pruned {len(pruned)} archive file(s)"
                    )
                else:
                    print(
                        f"  Retention '{ctx.archive_retention}' skipped: "
                        "batch did not converge, keeping full archive"
                    )
    finally:
        container.close()


def ingest_all(ctx: BatchContext, summary_stat="median", force=False):
    """Reduce step: fold every completed batch into the container.

    This is the sole container writer in the fanned-out pipeline. Array tasks
    (`run-batch-task`) leave a status shard and a verified pest_archive
    behind; this walks them in batch order under a single container open, so
    the zarr FileLock is taken once and never contended.

    Also handles the legacy layout where the batch build directory is still
    on disk (the serial `calibrate-all` path). Batches whose PEST++ run did
    not succeed are skipped with a warning unless force=True.
    """
    from swimrs.calibrate.batch_support import (
        find_par_csv,
        merge_shards_into_log,
        read_batch_shards,
        read_manifest,
        update_batch_entry,
    )
    from swimrs.calibrate.pest_cleanup import PestResults
    from swimrs.container.container import SwimContainer

    output_root = Path(ctx.output_root)
    manifest = read_manifest(output_root)
    batch_ids = sorted(manifest["batch_id"].unique())
    shards = read_batch_shards(output_root)
    merge_shards_into_log(output_root)

    container = SwimContainer.open(ctx.container_path, mode="r+")
    try:
        cal_group = container._root["calibration"] if "calibration" in container._root else None
        batches_meta = {}
        already_done = set()
        if cal_group is not None:
            batches_meta = json.loads(cal_group.attrs.get("batches", "{}"))
            already_done = set(batches_meta.keys())

        total_ingested = 0
        n_skipped_failed = 0
        ingested_ids = []
        fid_col = ctx.feature_id_col if ctx.feature_id_col in manifest.columns else "FID"

        for bid in batch_ids:
            if str(bid) in already_done:
                print(f"Batch {bid:03d}: already ingested, skipping")
                continue

            batch_dir = output_root / f"batch_{bid:03d}"
            archive_dir = output_root / "pest_archive" / f"batch_{bid:03d}"
            shard = shards.get(str(bid))

            summary = None
            if shard is not None and shard.get("status") in ("archived", "ingested"):
                # Stage 2 already ran, verified and archived this batch; the
                # build directory is gone by design.
                summary = shard.get("summary") or {}
                source = archive_dir
            else:
                source = batch_dir
                pst_files = list((batch_dir / "pest").glob("*.pst")) or list(
                    (batch_dir / "master").glob("*.pst")
                )
                if pst_files:
                    results = PestResults(
                        str(batch_dir / "pest"),
                        pst_files[0].stem,
                        master_dir=str(batch_dir / "master"),
                    )
                    success, issues = results.is_successful()
                    if not success:
                        if not force:
                            print(f"Batch {bid:03d}: run not successful, skipping — {issues}")
                            n_skipped_failed += 1
                            continue
                        print(f"Batch {bid:03d}: WARNING — ingesting despite issues: {issues}")
                    summary = results.get_summary()
                elif shard is not None and shard.get("status", "").endswith("failed"):
                    print(f"Batch {bid:03d}: {shard['status']}, skipping — {shard.get('error')}")
                    n_skipped_failed += 1
                    continue

            par_csv = find_par_csv(source)
            if par_csv is None:
                print(f"Batch {bid:03d}: no .par.csv under {source}, skipping")
                continue

            batch_fids = manifest.loc[manifest["batch_id"] == bid, fid_col].astype(str).tolist()
            container.ingest.calibration(
                par_csv, fields=batch_fids, batch_id=bid, summary_stat=summary_stat
            )
            total_ingested += len(batch_fids)
            ingested_ids.append(bid)
            print(f"Batch {bid:03d}: ingested {len(batch_fids)} fields from {par_csv.name}")

            if summary:
                batches_meta[str(bid)] = {
                    "n_fields": len(batch_fids),
                    "status": summary.get("status", "unknown"),
                    "phi_initial": summary.get("phi_initial"),
                    "phi_final": summary.get("phi_final"),
                    "phi_reduction_pct": summary.get("phi_reduction_pct"),
                    "phi_history": summary.get("phi_history"),
                    "noptmax": summary.get("noptmax"),
                    "iterations_completed": summary.get("iterations_completed"),
                }

        # One attribute write for the whole pass rather than one per batch.
        if batches_meta and "calibration" in container._root:
            container._root["calibration"].attrs["batches"] = json.dumps(batches_meta)

        print(f"\nTotal: {total_ingested} fields ingested across {len(batch_ids)} batches")
        if n_skipped_failed:
            print(
                f"Skipped {n_skipped_failed} unsuccessful batch(es); use --force-ingest to override"
            )
    finally:
        container.close()

    for bid in ingested_ids:
        entry = dict(shards.get(str(bid), {}))
        entry.update({"status": "ingested", "timestamp": datetime.now().isoformat()})
        entry.setdefault("batch_id", int(bid))
        update_batch_entry(output_root, bid, entry)


def show_status(ctx: BatchContext):
    """Print calibration status from the container and any pending shards."""
    import numpy as np

    from swimrs.calibrate.batch_support import read_batch_shards
    from swimrs.container.container import SwimContainer

    # Shards are written by array tasks that have run but not yet been
    # reduced into the container, so status is meaningful mid-fan-out.
    shards = read_batch_shards(ctx.output_root)
    if shards:
        counts = {}
        for entry in shards.values():
            counts[entry.get("status", "?")] = counts.get(entry.get("status", "?"), 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"Batch task shards: {len(shards)} ({summary})")
        pending = [b for b, e in shards.items() if e.get("status") == "archived"]
        if pending:
            print(f"  {len(pending)} archived batch(es) awaiting ingest-all")

    container = SwimContainer.open(ctx.container_path, mode="r")
    try:
        root = container._root
        if "calibration/metadata/calibrated" not in root:
            print("No calibration data in container.")
            return

        cal = np.asarray(root["calibration/metadata/calibrated"][:])
        n_cal = int(np.sum(cal > 0))
        n_total = len(cal)
        print(f"Calibrated: {n_cal}/{n_total} fields ({100 * n_cal / n_total:.1f}%)")

        if "calibration" in root:
            batches_str = root["calibration"].attrs.get("batches", "{}")
            batches = json.loads(batches_str)
            print(f"Batches completed: {len(batches)}")
            for bid, info in sorted(batches.items(), key=lambda x: int(x[0])):
                status = info.get("status", "?")
                n = info.get("n_fields", "?")
                phi_red = info.get("phi_reduction_pct")
                phi_str = f"phi_red={phi_red:.1f}%" if phi_red is not None else ""
                print(f"  Batch {int(bid):03d}: {n} fields, {status} {phi_str}")
    finally:
        container.close()


def cleanup_failed(ctx: BatchContext):
    """Remove batch directories for run_failed and ingest_failed batches."""
    from swimrs.calibrate.batch_support import read_batch_log, write_batch_log

    output_root = Path(ctx.output_root)
    batch_log = read_batch_log(output_root)

    cleaned = 0
    for bid_str, entry in batch_log.items():
        status = entry.get("status", "")
        if status in ("run_failed", "ingest_failed"):
            batch_dir = output_root / f"batch_{int(bid_str):03d}"
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
                print(f"Batch {bid_str}: removed {batch_dir}")
                cleaned += 1
            entry["status"] = "cleaned"
            entry["timestamp"] = datetime.now().isoformat()

    write_batch_log(output_root, batch_log)
    print(f"Cleaned {cleaned} failed batch directories")


# ---------------------------------------------------------------------------
# Main pipeline: calibrate_all
# ---------------------------------------------------------------------------


def calibrate_all(
    ctx: BatchContext,
    *,
    resume=False,
    override=False,
    skip_health=False,
    exclude_uncovered=False,
    skip_fids_path=None,
):
    """Pipelined batch calibration: build, run, ingest, cleanup one batch at a time.

    Pre-builds the next batch in a background process while the current
    batch's PEST++ run executes.
    """
    from swimrs.calibrate.batch_support import (
        batch_is_built,
        create_run_manifest,
        get_uncovered_fids,
        load_batches_from_manifest,
        partition_fields,
        persist_calibration_resolved_state,
        read_batch_log,
        update_batch_entry,
        write_manifest,
    )
    from swimrs.container.container import SwimContainer

    output_root = Path(ctx.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Preflight gate ---
    if skip_health:
        _verify_prior_health(ctx)
        report = None
    else:
        try:
            report = preflight_gate(ctx, override=override)
        except Exception as e:
            print(f"PREFLIGHT GATE BLOCKED: {e}")
            raise

    # --- Batch manifest: single source of truth ---
    manifest_path = output_root / "batch_manifest.csv"
    if manifest_path.exists():
        batches = load_batches_from_manifest(output_root, ctx.feature_id_col)
    else:
        exclude_set: set[str] = set()

        if exclude_uncovered:
            print("Scanning container for zero-coverage fields…")
            c = SwimContainer.open(ctx.container_path, mode="r")
            try:
                uncovered = get_uncovered_fids(
                    c,
                    ctx.etf_target_model,
                    ctx.mask_mode,
                    etf_ensemble_members=ctx.etf_ensemble_members,
                    instrument=ctx.etf_target_instrument,
                )
                exclude_set.update(uncovered["all"])
                if uncovered["all"]:
                    print(
                        f"  Excluding {len(uncovered['all'])} uncovered field(s): "
                        f"ndvi={len(uncovered.get('ndvi', []))}, "
                        f"etf={len(uncovered.get('etf', []))}"
                    )
            finally:
                c.close()

        if skip_fids_path is not None:
            skip_fids_path = Path(skip_fids_path)
            extra = {
                line.strip() for line in skip_fids_path.read_text().splitlines() if line.strip()
            }
            exclude_set.update(extra)
            print(f"  Excluding {len(extra)} additional field(s) from {skip_fids_path.name}")

        if exclude_set:
            excluded_record = {
                "timestamp": datetime.now().isoformat(),
                "n_excluded": len(exclude_set),
                "source": {
                    "exclude_uncovered": exclude_uncovered,
                    "skip_fids_path": str(skip_fids_path) if skip_fids_path else None,
                },
                "fids": sorted(exclude_set),
            }
            (output_root / "excluded_fids.json").write_text(json.dumps(excluded_record, indent=2))
            print(f"  Wrote excluded_fids.json ({len(exclude_set)} field(s))")

        # Use the grouping shapefile (gridmet mapping) when available so
        # the GFID column is present for grouped packing.
        grouping_col = ctx.grouping_column
        shapefile = ctx.grouping_shapefile or ctx.fields_shapefile

        # The shapefile may cover a larger network than the container; a
        # batch with no container fields cannot build.
        c = SwimContainer.open(ctx.container_path, mode="r")
        try:
            container_fids = set(c.field_uids)
        finally:
            c.close()
        print(f"  Restricting manifest to {len(container_fids)} container field(s)")

        raw_batches = partition_fields(
            shapefile,
            ctx.feature_id_col,
            ctx.batch_size,
            grouping_column=grouping_col,
            exclude_fids=exclude_set,
            include_fids=container_fids,
        )
        write_manifest(output_root, raw_batches, feature_id_col=ctx.feature_id_col)
        batches = list(enumerate(raw_batches))
        print(f"Created manifest with {len(batches)} batches: {manifest_path}")

    # --- Determine which batches to process ---
    batch_log = read_batch_log(output_root)
    container_ingested = set()
    try:
        c = SwimContainer.open(ctx.container_path, mode="r")
        try:
            if "calibration" in c._root:
                batches_str = c._root["calibration"].attrs.get("batches", "{}")
                container_ingested = set(json.loads(batches_str).keys())
        finally:
            c.close()
    except Exception:
        pass

    # --- Stale-calibration guard (C-2) ---
    from swimrs.calibrate.batch_support import resolve_ingested_batches

    container_ingested = resolve_ingested_batches(
        container_ingested, batch_log, override, ctx.container_path, output_root
    )

    to_process = []
    for batch_id, batch_fids in batches:
        bid_str = str(batch_id)
        if resume:
            log_entry = batch_log.get(bid_str, {})
            status = log_entry.get("status", "")
            if status == "ingested" or bid_str in container_ingested:
                print(f"Batch {batch_id:03d}: already ingested, skipping")
                continue
        to_process.append((batch_id, batch_fids))

    if not to_process:
        print("All batches already processed.")
        show_status(ctx)
        return 0

    # --- Run manifest ---
    create_run_manifest(
        output_root,
        ctx.container_path,
        ctx.toml_path,
        report,
        to_process,
        ctx.noptmax,
        ctx.realizations,
        ctx.workers,
        ctx.batch_size,
        override,
        ctx.feature_id_col,
        ctx.grouping_column,
        ctx.mask_mode,
        ctx.etf_target_model,
        ctx.project_name,
    )

    print(f"\nProcessing {len(to_process)} batches (pipeline mode)...\n")

    prebuild_proc = None
    prebuild_queue = None
    prebuild_batch_id = None

    # Serialize context for subprocess
    ctx_dict = {
        "project_name": ctx.project_name,
        "container_path": ctx.container_path,
        "fields_shapefile": ctx.fields_shapefile,
        "feature_id_col": ctx.feature_id_col,
        "grouping_column": ctx.grouping_column,
        "grouping_shapefile": ctx.grouping_shapefile,
        "output_root": ctx.output_root,
        "mask_mode": ctx.mask_mode,
        "etf_target_model": ctx.etf_target_model,
        "etf_ensemble_members": ctx.etf_ensemble_members,
        "etf_target_instrument": ctx.etf_target_instrument,
        "workers": ctx.workers,
        "realizations": ctx.realizations,
        "noptmax": ctx.noptmax,
        "batch_size": ctx.batch_size,
        "toml_path": ctx.toml_path,
    }

    n_failed = 0

    for step, (batch_id, batch_fids) in enumerate(to_process):
        batch_dir = output_root / f"batch_{batch_id:03d}"

        # --- PHASE A: Ensure this batch is built ---
        build_result = None

        if prebuild_proc is not None and prebuild_batch_id == batch_id:
            prebuild_proc.join(timeout=7200)
            if prebuild_proc.exitcode != 0:
                build_result = {
                    "status": "build_failed",
                    "n_fields": len(batch_fids),
                    "dropped_fids": [],
                    "error": f"Background build process exited with code {prebuild_proc.exitcode}",
                }
            elif not prebuild_queue.empty():
                tag, _, result = prebuild_queue.get_nowait()
                if tag == "ok":
                    build_result = result
                else:
                    build_result = {
                        "status": "build_failed",
                        "n_fields": len(batch_fids),
                        "dropped_fids": [],
                        "error": result,
                    }
            else:
                build_result = {
                    "status": "build_failed",
                    "n_fields": len(batch_fids),
                    "dropped_fids": [],
                    "error": "Background build produced no result",
                }
            prebuild_proc = None
            prebuild_queue = None
            prebuild_batch_id = None

        elif batch_is_built(batch_dir):
            print(f"Batch {batch_id:03d}: using existing build on disk")
            build_result = {
                "status": "built",
                "n_fields": len(batch_fids),
                "dropped_fids": [],
            }

        else:
            print(f"\n--- Building batch {batch_id:03d} (sync) ---")
            build_result = build_batch(ctx, batch_fids, batch_id)

        if build_result["status"] == "build_failed":
            print(f"Batch {batch_id:03d}: BUILD FAILED — {build_result.get('error', '')[:200]}")
            update_batch_entry(
                output_root,
                batch_id,
                {
                    "status": "build_failed",
                    "n_fields": build_result["n_fields"],
                    "dropped_fids": build_result.get("dropped_fids", []),
                    "error": build_result.get("error", ""),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            n_failed += 1
            continue

        # Update manifest if FIDs were dropped
        dropped = build_result.get("dropped_fids", [])
        if dropped:
            manifest = pd.read_csv(manifest_path)
            fid_col = ctx.feature_id_col if ctx.feature_id_col in manifest.columns else "FID"
            mask = (manifest["batch_id"] == batch_id) & (
                manifest[fid_col].astype(str).isin(set(dropped))
            )
            manifest = manifest[~mask]
            manifest.to_csv(manifest_path, index=False)
            batch_fids = [f for f in batch_fids if f not in set(dropped)]
            print(f"  Manifest updated: dropped FIDs {dropped}")

        # --- PHASE B: Start pre-building NEXT batch in background ---
        if step + 1 < len(to_process):
            next_batch_id, next_batch_fids = to_process[step + 1]
            next_batch_dir = output_root / f"batch_{next_batch_id:03d}"
            if not batch_is_built(next_batch_dir):
                prebuild_queue = multiprocessing.Queue()
                prebuild_proc = multiprocessing.Process(
                    target=_build_batch_worker,
                    args=(prebuild_queue, ctx_dict, next_batch_fids, next_batch_id),
                    daemon=True,
                )
                prebuild_proc.start()
                prebuild_batch_id = next_batch_id
                print(
                    f"  Pre-building batch {next_batch_id:03d} in background "
                    f"(PID {prebuild_proc.pid})"
                )

        # --- PHASE C: Run PEST++ (blocks) ---
        print(f"\n=== Running batch {batch_id:03d} ===")
        try:
            run_batch(batch_dir, num_workers=ctx.workers)
        except Exception:
            err = traceback.format_exc()[-4096:]
            print(f"Batch {batch_id:03d}: RUN FAILED — {err[:200]}")
            update_batch_entry(
                output_root,
                batch_id,
                {
                    "status": "run_failed",
                    "n_fields": build_result["n_fields"],
                    "dropped_fids": dropped,
                    "error": err,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            n_failed += 1
            continue

        # --- PHASE D: Ingest into container ---
        try:
            ingest_batch(ctx, batch_id)
        except Exception:
            err = traceback.format_exc()[-4096:]
            print(f"Batch {batch_id:03d}: INGEST FAILED — {err[:200]}")
            update_batch_entry(
                output_root,
                batch_id,
                {
                    "status": "ingest_failed",
                    "n_fields": build_result["n_fields"],
                    "dropped_fids": dropped,
                    "error": err,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            n_failed += 1
            continue

        # --- PHASE E: Log success + cleanup ---
        phi_initial = None
        phi_final = None
        try:
            c = SwimContainer.open(ctx.container_path, mode="r")
            try:
                if "calibration" in c._root:
                    bm = json.loads(c._root["calibration"].attrs.get("batches", "{}"))
                    info = bm.get(str(batch_id), {})
                    phi_initial = info.get("phi_initial")
                    phi_final = info.get("phi_final")
            finally:
                c.close()
        except Exception:
            pass

        update_batch_entry(
            output_root,
            batch_id,
            {
                "status": "ingested",
                "n_fields": build_result["n_fields"],
                "dropped_fids": dropped,
                "error": None,
                "timestamp": datetime.now().isoformat(),
                "phi_initial": phi_initial,
                "phi_final": phi_final,
            },
        )

        # Delete the build dir only after the PEST++ archive is verified
        # (RUN_POLICY Category 4: the iteration trajectory must survive).
        from swimrs.calibrate.batch_support import verify_pest_archive

        archive_dir = output_root / "pest_archive" / f"batch_{batch_id:03d}"
        archive_ok, missing = verify_pest_archive(archive_dir)
        if archive_ok:
            print(f"Batch {batch_id:03d}: archive verified, cleaning up build directory")
            shutil.rmtree(batch_dir)
        else:
            print(
                f"Batch {batch_id:03d}: WARNING — archive at {archive_dir} missing "
                f"{missing}; keeping build directory {batch_dir}"
            )

    # Join any lingering prebuild process
    if prebuild_proc is not None:
        prebuild_proc.join(timeout=60)

    if n_failed:
        print(f"\n=== Pipeline complete with {n_failed} failed batch(es) ===")
    else:
        print("\n=== Pipeline complete ===")
    os.chdir(output_root)
    persist_calibration_resolved_state(
        ctx.container_path,
        ctx.toml_path,
        str(output_root),
        command="calibrate-batch --action calibrate-all",
    )
    show_status(ctx)
    return n_failed


# ---------------------------------------------------------------------------
# Prep action (manifest-only)
# ---------------------------------------------------------------------------


def prep(
    ctx: BatchContext,
    *,
    exclude_uncovered=False,
    skip_fids_path=None,
    override=False,
    skip_health=False,
):
    """Stage 1 of the fanned-out pipeline: gate, partition, and record the run.

    Runs the preflight gate once and writes batch_manifest.csv plus
    run_manifest.json, so the array tasks that follow do no gating and no
    provenance capture of their own. Prints ``BATCH_COUNT=<n>`` for the
    submitting script to size its --array range.
    """
    from swimrs.calibrate.batch_support import (
        create_run_manifest,
        get_uncovered_fids,
        partition_fields,
        write_manifest,
    )
    from swimrs.container.container import SwimContainer

    output_root = Path(ctx.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if skip_health:
        _verify_prior_health(ctx)
        report = None
    else:
        try:
            report = preflight_gate(ctx, override=override)
        except Exception as e:
            print(f"PREFLIGHT GATE BLOCKED: {e}")
            raise

    # The coverage scan always runs and always reports; --exclude-uncovered only
    # decides whether to act on it. Keeping the two separate lets a run that
    # deliberately calibrates every field (e.g. reproducing an earlier run whose
    # partitioning must match) still measure what exclusion would have dropped.
    exclude_set: set[str] = set()
    print("Scanning container for zero-coverage fields…")
    c = SwimContainer.open(ctx.container_path, mode="r")
    try:
        uncovered = get_uncovered_fids(
            c,
            ctx.etf_target_model,
            ctx.mask_mode,
            etf_ensemble_members=ctx.etf_ensemble_members,
            instrument=ctx.etf_target_instrument,
        )
        n_uncovered = len(uncovered["all"])
        n_container = len(c.field_uids)
        frac = 100 * n_uncovered / n_container if n_container else 0.0
        print(
            f"  ndvi uncovered: {len(uncovered.get('ndvi', []))}, "
            f"etf uncovered: {len(uncovered.get('etf', []))}, "
            f"total uncovered: {n_uncovered}"
        )
        print(f"  UNCOVERED_FRACTION={frac:.2f}% ({n_uncovered}/{n_container})")
        if exclude_uncovered:
            exclude_set.update(uncovered["all"])
            print(f"  --exclude-uncovered: dropping those {n_uncovered} field(s)")
        else:
            print("  --exclude-uncovered not set: reporting only, all fields retained")
    finally:
        c.close()

    if skip_fids_path:
        extra = {
            line.strip() for line in Path(skip_fids_path).read_text().splitlines() if line.strip()
        }
        exclude_set.update(extra)
        print(f"  Additional skip-fids: {len(extra)}")

    if exclude_set:
        excluded_record = {
            "timestamp": datetime.now().isoformat(),
            "n_excluded": len(exclude_set),
            "fids": sorted(exclude_set),
        }
        excluded_path = output_root / "excluded_fids.json"
        excluded_path.write_text(json.dumps(excluded_record, indent=2))
        print(f"  Wrote {excluded_path}")

    c = SwimContainer.open(ctx.container_path, mode="r")
    try:
        container_fids = set(c.field_uids)
    finally:
        c.close()
    print(f"  Restricting manifest to {len(container_fids)} container field(s)")

    raw_batches = partition_fields(
        ctx.grouping_shapefile or ctx.fields_shapefile,
        ctx.feature_id_col,
        ctx.batch_size,
        grouping_column=ctx.grouping_column,
        exclude_fids=exclude_set,
        include_fids=container_fids,
    )
    manifest_path = write_manifest(output_root, raw_batches, feature_id_col=ctx.feature_id_col)
    print(f"Partitioned into {len(raw_batches)} batches:")
    for i, batch in enumerate(raw_batches):
        print(f"  Batch {i:03d}: {len(batch)} fields")
    print(f"\nWrote manifest: {manifest_path}")

    create_run_manifest(
        output_root,
        ctx.container_path,
        ctx.toml_path,
        report,
        raw_batches,
        ctx.noptmax,
        ctx.realizations,
        ctx.workers,
        ctx.batch_size,
        override,
        ctx.feature_id_col,
        ctx.grouping_column,
        ctx.mask_mode,
        ctx.etf_target_model,
        ctx.project_name,
    )

    # Parsed by the submitting script to size --array=0-(N-1).
    print(f"BATCH_COUNT={len(raw_batches)}")
    return len(raw_batches)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_batch_parser() -> argparse.ArgumentParser:
    """Build argparse parser for batch calibration."""
    parser = argparse.ArgumentParser(
        description="SWIM-RS batch PEST++ IES calibration",
    )
    parser.add_argument("--config", required=True, help="Path to SWIM-RS project TOML")
    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "prep",
            "build-all",
            "run-batch",
            "run-batch-task",
            "run-all",
            "ingest-batch",
            "ingest-all",
            "status",
            "calibrate-all",
            "cleanup-failed",
        ],
        help="Action to perform",
    )
    parser.add_argument(
        "--batch-id", type=int, help="Batch ID for run-batch/run-batch-task/ingest-batch"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="run-batch-task: rebuild and rerun even when the batch is already complete",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-ingested batches")
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override preflight gate failures (log and continue)",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip preflight health check (requires prior check in container)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Fields per batch (default: config, else 50)"
    )
    parser.add_argument("--workers", type=int, default=None, help="PEST workers per batch")
    parser.add_argument(
        "--noptmax", type=int, default=None, help="Max PEST iterations (default: config, else 3)"
    )
    parser.add_argument("--reals", type=int, default=None, help="Ensemble realizations")
    parser.add_argument(
        "--exclude-uncovered",
        action="store_true",
        help="Exclude fields with zero RS observations from the manifest",
    )
    parser.add_argument(
        "--skip-fids",
        type=str,
        default=None,
        help="Path to text file listing FIDs to exclude (one per line)",
    )
    parser.add_argument("--container", type=str, default=None, help="Override container path")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--shapefile", type=str, default=None, help="Override fields shapefile")
    parser.add_argument(
        "--prior-params",
        type=str,
        default=None,
        help="Path to JSON with LULC-specific prior parameter values for Tikhonov regularization",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Ingest batches even when the PEST++ run did not succeed",
    )
    return parser


def main(argv=None):
    """Entry point for ``python -m swimrs.calibrate.batch_runner``."""
    from swimrs.calibrate.batch_support import (
        find_par_csv,
        persist_calibration_resolved_state,
    )

    parser = build_batch_parser()
    args = parser.parse_args(argv)

    ctx = resolve_context(
        args.config,
        container_override=args.container,
        output_override=args.output,
        shapefile_override=args.shapefile,
        batch_size=args.batch_size,
        workers=args.workers,
        realizations=args.reals,
        noptmax=args.noptmax,
        prior_params_path=args.prior_params,
    )

    action = args.action

    if action == "prep":
        prep(
            ctx,
            exclude_uncovered=args.exclude_uncovered,
            skip_fids_path=args.skip_fids,
            override=args.override,
            skip_health=args.skip_health,
        )

    elif action == "build-all":
        from swimrs.calibrate.batch_support import load_batches_from_manifest, partition_fields
        from swimrs.container.container import SwimContainer

        manifest_path = Path(ctx.output_root) / "batch_manifest.csv"
        if manifest_path.exists():
            batches = load_batches_from_manifest(ctx.output_root, ctx.feature_id_col)
        else:
            c = SwimContainer.open(ctx.container_path, mode="r")
            try:
                container_fids = set(c.field_uids)
            finally:
                c.close()
            raw = partition_fields(
                ctx.grouping_shapefile or ctx.fields_shapefile,
                ctx.feature_id_col,
                ctx.batch_size,
                grouping_column=ctx.grouping_column,
                include_fids=container_fids,
            )
            batches = list(enumerate(raw))

        print(f"Building {len(batches)} batches...")
        for batch_id, batch_fids in batches:
            print(f"\n--- Batch {batch_id:03d} ({len(batch_fids)} fields) ---")
            build_batch(ctx, batch_fids, batch_id)

    elif action == "run-batch":
        if args.batch_id is None:
            parser.error("--batch-id required for run-batch")
        batch_dir = Path(ctx.output_root) / f"batch_{args.batch_id:03d}"
        if not batch_dir.exists():
            parser.error(f"Batch directory not found: {batch_dir}")
        run_batch(batch_dir, num_workers=ctx.workers)

    elif action == "run-batch-task":
        if args.batch_id is None:
            parser.error("--batch-id required for run-batch-task")
        return run_batch_task(ctx, args.batch_id, rebuild=args.rebuild)

    elif action == "run-all":
        batch_dirs = sorted(Path(ctx.output_root).glob("batch_*"))
        if not batch_dirs:
            parser.error(f"No batch directories found in {ctx.output_root}")
        print(f"Running {len(batch_dirs)} batches sequentially...")
        for bd in batch_dirs:
            if args.resume and find_par_csv(bd) is not None:
                print(f"\n=== {bd.name} === SKIP (has .par.csv)")
                continue
            print(f"\n=== {bd.name} ===")
            run_batch(bd, num_workers=ctx.workers)

    elif action == "ingest-batch":
        if args.batch_id is None:
            parser.error("--batch-id required for ingest-batch")
        ingest_batch(ctx, args.batch_id, force=args.force_ingest)

    elif action == "ingest-all":
        ingest_all(ctx, force=args.force_ingest)
        persist_calibration_resolved_state(
            ctx.container_path,
            ctx.toml_path,
            ctx.output_root,
            command="calibrate-batch --action ingest-all",
        )

    elif action == "status":
        show_status(ctx)

    elif action == "calibrate-all":
        n_failed = calibrate_all(
            ctx,
            resume=args.resume,
            override=args.override,
            skip_health=args.skip_health,
            exclude_uncovered=args.exclude_uncovered,
            skip_fids_path=args.skip_fids,
        )
        return min(n_failed, 1)

    elif action == "cleanup-failed":
        cleanup_failed(ctx)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
# ========================= EOF ====================================================================

"""Phase 9 (E2 re-footing plan §15): independent audit of the built inverse problem.

Reconstructs every ETf observation's target, spread, eligibility and weight directly from the
corrected container and the resolved config (no PestBuilder code path), then compares the
reconstruction with the ``.pst`` control files and ``weight_audit.csv`` written by
``batch_runner --action build-all``. It also re-derives the SWE weight scaling, checks the
manifest partition against the baseline run, reconciles weighted counts by site / year / member
count / batch, and classifies every baseline weighted date that is no longer weighted.

Gate G9 passes when the number of *unexplained* discrepancies is zero: every weight reproduces,
every lost baseline date is explained by the documented ETo floor or by the ingest bounds acting
on the basis-corrected SSEBop value, and the PT-JPL-only share is below the review threshold.

Outputs (QA root): ``objective_weight_audit.json`` (summary + gate), ``objective_weight_rows.csv``
(one row per ETf capture date: reproducible weight decomposition), ``objective_weight_losses.csv``
(baseline weighted dates lost, with category), ``objective_weight_counts.csv`` (weighted counts by
site / year / member count / batch vs baseline).

    uv run python examples/6_Flux_International/e2_refooting/phase9_objective_audit.py \
        --config examples/6_Flux_International/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml \
        --baseline-pest-archive /data/ssd1/swim/6_Flux_International/pestrun_ls_ensemble_por_annual2yr/pest_archive \
        --ledger /data/ssd1/swim/6_Flux_International/data/e2_etf_refooting/ssebop_conversion_ledger.csv \
        --out-dir /data/ssd1/swim/6_Flux_International/data/e2_etf_refooting
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

QA_ROOT = "/data/ssd1/swim/6_Flux_International/data/e2_etf_refooting"
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml",
)
DEFAULT_BASELINE_ARCHIVE = (
    "/data/ssd1/swim/6_Flux_International/pestrun_ls_ensemble_por_annual2yr/pest_archive"
)
DEFAULT_LEDGER = os.path.join(QA_ROOT, "ssebop_conversion_ledger.csv")

PTJPL_ONLY_REVIEW_SHARE = 0.10
WEIGHT_RTOL = 1e-6
OBSVAL_ATOL = 1e-6

# Loss categories that the plan documents; anything else is unexplained.
EXPLAINED_LOSS = {"eto_floor", "ingest_ceiling", "ingest_floor"}


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ---------------------------------------------------------------------------


def parse_obs_name(name: str) -> tuple[str, str, int]:
    """Split ``oname:obs_etf_us-ne1_otype:arr_i:123_j:0`` into (kind, fid, day index)."""
    head, rest = name.split("_otype:", 1)
    prefix = "oname:obs_"
    if not head.startswith(prefix):
        raise ValueError(f"unexpected observation name {name!r}")
    kind, fid = head[len(prefix) :].split("_", 1)
    idx = int(rest.split("_i:", 1)[1].split("_", 1)[0])
    return kind, fid, idx


def reconstruct_etf_weights(
    members: pd.DataFrame,
    eto: pd.Series,
    *,
    spread_floor: float,
    min_members: int,
    eto_floor: float | None,
    fixed_sd: float,
) -> pd.DataFrame:
    """Reproduce the pest_builder ``spread`` weighting from raw member series.

    ``members`` is indexed by date with one column per ensemble member (NaN = no retrieval).
    Rows with no finite member are not captures and are dropped. The target is the mean of the
    finite members, the spread is their sample SD (ddof 1, NaN for one member), the weight is
    target / (spread + floor) when the member count reaches ``min_members`` and the daily ETo
    reaches ``eto_floor`` (inclusive), else zero. The IES noise SD is spread + floor, falling
    back to ``fixed_sd`` where the spread is undefined.
    """
    finite = members.notna()
    captures = finite.any(axis=1)
    m = members.loc[captures]
    ct = finite.loc[captures].sum(axis=1)
    target = m.mean(axis=1)
    std = m.std(axis=1, ddof=1)
    eto_vals = eto.reindex(m.index).astype(float)
    if eto_floor is not None:
        if not np.isfinite(eto_vals.to_numpy()).all():
            bad = eto_vals.index[~np.isfinite(eto_vals.to_numpy())]
            raise ValueError(f"daily ETo missing on {len(bad)} capture date(s): {list(bad[:3])}")
        eto_ok = eto_vals >= eto_floor
    else:
        eto_ok = pd.Series(True, index=m.index)
    eligible = (ct >= min_members) & eto_ok
    weight = np.where(eligible, target / (std + spread_floor), 0.0)
    sd = (std + spread_floor).to_numpy(dtype=float)
    sd = np.where(np.isfinite(sd), sd, fixed_sd)
    out = pd.DataFrame(
        {
            "target": target,
            "member_count": ct.astype(int),
            "member_std": std,
            "eto": eto_vals,
            "eto_floor_excluded": ~eto_ok,
            "eligible": eligible,
            "weight": weight,
            "standard_deviation": sd,
        },
        index=m.index,
    )
    for col in members.columns:
        out[col] = m[col]
    return out


def swe_expected_weights(
    swe_obsvals: np.ndarray,
    etf_weights: np.ndarray,
    etf_sd: np.ndarray,
    *,
    sd_frac: float,
    sd_floor: float,
    phi_share: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reproduce ``_finalize_obs``: SWE sd = max(frac*swe, floor); weight = c / sd with c chosen
    so the expected SWE phi is ``phi_share`` of the expected primary-ETf phi.

    Returns (weights, sds, c). Positive-SWE observations only.
    """
    swe = np.asarray(swe_obsvals, dtype=float)
    sd = np.maximum(sd_frac * swe, sd_floor)
    etf_phi = float(
        ((np.asarray(etf_weights, dtype=float) * np.asarray(etf_sd, dtype=float)) ** 2).sum()
    )
    if len(swe) == 0 or etf_phi <= 0:
        return np.zeros_like(swe), sd, 0.0
    c = float(np.sqrt(phi_share / (1.0 - phi_share) * etf_phi / len(swe)))
    return c / sd, sd, c


def classify_loss(row: pd.Series, min_etf: float, max_etf: float) -> str:
    """Categorize a baseline weighted (site, date) that carries zero weight in the new problem.

    ``row`` carries the new-problem reconstruction for that date (``member_count``, ``eto``,
    ``eto_floor_excluded``, ``ssebop`` may be NaN) plus the conversion ledger's ``corrected``
    value and ``status`` for the site-date (NaN / '' when absent).
    """
    if bool(row.get("eto_floor_excluded", False)):
        return "eto_floor"
    if int(row.get("member_count", 0)) >= 2:
        return "unexplained_weighted_members_but_zero_weight"
    corrected = row.get("corrected", np.nan)
    if np.isfinite(corrected) and corrected > max_etf:
        return "ingest_ceiling"
    if np.isfinite(corrected) and corrected < min_etf:
        return "ingest_floor"
    if not np.isfinite(corrected):
        return "unexplained_no_ledger_row"
    return "unexplained_member_missing"


# ---------------------------------------------------------------------------
# Container / PEST readers
# ---------------------------------------------------------------------------


def _uid_list(root) -> list[str]:
    return [str(x) for x in np.asarray(root["geometry/uid"][:]).tolist()]


def load_member_frames(root, members: list[str], instrument: str, mask: str, dates, uids):
    arrays = {
        m: np.asarray(root[f"remote_sensing/etf/{instrument}/{m}/{mask}"][:]) for m in members
    }
    frames = {}
    for j, uid in enumerate(uids):
        frames[uid] = pd.DataFrame({m: arrays[m][:, j] for m in members}, index=dates)
    return frames


def read_pst_observations(pst_path: str) -> pd.DataFrame:
    from pyemu import Pst

    pst = Pst(str(pst_path))
    obs = pst.observation_data.copy()
    obs.index = obs.index.str.lower()
    kinds, fids, idxs = zip(*(parse_obs_name(n) for n in obs.index))
    obs["kind"] = kinds
    obs["fid_lower"] = fids
    obs["day_index"] = idxs
    meta = {
        "noptmax": int(pst.control_data.noptmax),
        "ies_num_reals": pst.pestpp_options.get("ies_num_reals"),
        "ies_reg_factor": pst.pestpp_options.get("ies_reg_factor"),
        "ies_localizer": pst.pestpp_options.get("ies_localizer"),
        "ies_drop_conflicts": pst.pestpp_options.get("ies_drop_conflicts"),
        "n_par": int(pst.npar),
        "n_par_adjustable": int(pst.npar_adj),
        "n_obs": int(pst.nobs),
        "n_obs_nonzero_weight": int(pst.nnz_obs),
        "parameter_groups": {
            str(g): {
                "n": int(n),
                "lower_min": float(
                    pst.parameter_data.loc[pst.parameter_data.pargp == g, "parlbnd"].min()
                ),
                "upper_max": float(
                    pst.parameter_data.loc[pst.parameter_data.pargp == g, "parubnd"].max()
                ),
            }
            for g, n in pst.parameter_data.pargp.value_counts().items()
        },
    }
    return obs, meta


def baseline_weighted_dates(archive_dir: Path, start: pd.Timestamp) -> tuple[set, pd.DataFrame]:
    """Weighted ETf (fid_lower, date) pairs and SWE obs of the baseline run across its batches."""
    weighted = set()
    swe_frames = []
    for pst_path in sorted(archive_dir.glob("batch_*/*.pst")):
        obs, _ = read_pst_observations(str(pst_path))
        etf = obs[(obs.kind == "etf") & (obs.weight > 0)]
        weighted.update(
            zip(etf.fid_lower, (start + pd.to_timedelta(etf.day_index, unit="D")).dt.normalize())
        )
        swe = obs[obs.kind == "swe"][["fid_lower", "day_index", "obsval", "weight"]].copy()
        swe["batch"] = pst_path.parent.name
        swe_frames.append(swe)
    return weighted, pd.concat(swe_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit(
    config_path, baseline_archive, ledger_path, out_dir, expected_noptmax, expected_reals
):
    import zarr

    from swimrs.container.schema import find_swe_path
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(config_path, calibrate=True)
    pest_run_dir = Path(cfg.pest_run_dir)
    members = list(cfg.etf_ensemble_members)
    instrument = cfg.etf_target_instrument or "landsat"
    if (cfg.mask_mode or "none") != "none":
        raise ValueError("audit reconstructs the mask_mode='none' problem only")
    mask = "no_mask"
    spread_floor = float(getattr(cfg, "etf_weighting_spread_floor", 0.1))
    min_members = int(getattr(cfg, "etf_weighting_min_members", 2))
    eto_floor = getattr(cfg, "etf_weighting_eto_floor", None)
    fixed_sd = float(getattr(cfg, "etf_weighting_fixed_sd", 0.33))
    sd_frac = float(getattr(cfg, "swe_weighting_sd_frac", 0.3))
    sd_floor = float(getattr(cfg, "swe_weighting_sd_floor", 10.0))
    phi_share = float(getattr(cfg, "swe_weighting_phi_share", 0.15))

    root = zarr.open_group(cfg.container_path, mode="r")
    uids = _uid_list(root)
    lower_to_uid = {u.lower(): u for u in uids}
    dates = pd.date_range(cfg.start_dt, cfg.end_dt, freq="D")
    n_days = len(dates)
    if root["time/daily"].shape[0] != n_days:
        raise ValueError("container time axis does not match the config date range")
    ingest_rules = (
        dict(root["remote_sensing/etf"].attrs).get("ingest_rules", {})
        if "remote_sensing/etf" in root
        else {}
    )
    min_etf = float(ingest_rules.get("min_etf", 0.05))
    max_etf = float(ingest_rules.get("max_etf", 2.0))

    frames = load_member_frames(root, members, instrument, mask, dates, uids)
    eto_arr = np.asarray(root[f"meteorology/{cfg.met_source}/eto"][:])
    swe_path = find_swe_path(root)
    swe_arr = np.asarray(root[swe_path][:])

    # Manifest / partition ------------------------------------------------------
    manifest = pd.read_csv(pest_run_dir / "batch_manifest.csv")
    fid_col = cfg.feature_id_col if cfg.feature_id_col in manifest.columns else "FID"
    manifest[fid_col] = manifest[fid_col].astype(str)
    excluded_path = pest_run_dir / "excluded_fids.json"
    excluded = (
        json.loads(excluded_path.read_text()).get("fids", []) if excluded_path.exists() else []
    )
    manifest_sites = manifest[fid_col].tolist()
    batch_of = dict(zip(manifest[fid_col], manifest.batch_id))
    baseline_manifest = pd.read_csv(Path(baseline_archive).parent / "batch_manifest.csv")
    bcol = fid_col if fid_col in baseline_manifest.columns else "FID"
    partition_identical = (
        manifest[["batch_id", fid_col]].values.tolist()
        == baseline_manifest[["batch_id", bcol]].astype({bcol: str}).values.tolist()
    )
    manifest_checks = {
        "n_sites": len(manifest_sites),
        "n_batches": int(manifest.batch_id.nunique()),
        "sites_match_container_minus_excluded": sorted(manifest_sites)
        == sorted(set(uids) - set(excluded)),
        "excluded": excluded,
        "partition_identical_to_baseline": bool(partition_identical),
    }

    # Ledger (corrected SSEBop before ingest bounds) --------------------------------
    ledger = pd.read_csv(
        ledger_path, usecols=["site", "date", "corrected", "status"], dtype={"date": str}
    )
    ledger["date"] = pd.to_datetime(ledger["date"], format="%Y%m%d")
    ledger = ledger.groupby(["site", "date"], as_index=False).agg(
        corrected=("corrected", "max"), status=("status", "first")
    )
    ledger_idx = ledger.set_index(["site", "date"])

    baseline_weighted, baseline_swe = baseline_weighted_dates(Path(baseline_archive), dates[0])

    # Per-batch comparison -----------------------------------------------------------
    rows = []
    swe_report = []
    discrepancies = []
    pst_meta = {}
    audit_csv_mismatch = 0
    for batch_dir in sorted(p for p in pest_run_dir.glob("batch_*") if p.is_dir()):
        pst_files = sorted((batch_dir / "pest").glob("*.pst"))
        if not pst_files:
            discrepancies.append({"batch": batch_dir.name, "issue": "no .pst built"})
            continue
        obs, meta = read_pst_observations(str(pst_files[0]))
        pst_meta[batch_dir.name] = meta
        if meta["noptmax"] != expected_noptmax:
            discrepancies.append(
                {
                    "batch": batch_dir.name,
                    "issue": f"noptmax {meta['noptmax']} != {expected_noptmax}",
                }
            )
        if int(meta["ies_num_reals"] or 0) != expected_reals:
            discrepancies.append(
                {
                    "batch": batch_dir.name,
                    "issue": f"ies_num_reals {meta['ies_num_reals']} != {expected_reals}",
                }
            )
        if not np.isfinite(obs.weight.to_numpy(dtype=float)).all():
            discrepancies.append({"batch": batch_dir.name, "issue": "non-finite weight in .pst"})

        audit_csv = batch_dir / "pest" / "weight_audit.csv"
        audit = pd.read_csv(audit_csv, parse_dates=["date"]) if audit_csv.exists() else None
        if audit is None:
            discrepancies.append({"batch": batch_dir.name, "issue": "weight_audit.csv missing"})

        etf_obs = obs[obs.kind == "etf"]
        batch_fids = sorted(etf_obs.fid_lower.unique())
        primary_w, primary_sd = [], []
        for fid_l in batch_fids:
            uid = lower_to_uid[fid_l]
            j = uids.index(uid)
            recon = reconstruct_etf_weights(
                frames[uid],
                pd.Series(eto_arr[:, j], index=dates),
                spread_floor=spread_floor,
                min_members=min_members,
                eto_floor=eto_floor,
                fixed_sd=fixed_sd,
            )
            o = etf_obs[etf_obs.fid_lower == fid_l].set_index("day_index").sort_index()
            if len(o) != n_days:
                discrepancies.append(
                    {
                        "batch": batch_dir.name,
                        "fid": uid,
                        "issue": f"{len(o)} ETf obs != {n_days} days",
                    }
                )
            o_dates = dates[o.index.to_numpy()]
            o = o.set_index(o_dates)
            primary_w.append(o.weight.to_numpy(dtype=float))
            primary_sd.append(o.standard_deviation.to_numpy(dtype=float))

            cap = recon.index
            non_cap = o.index.difference(cap)
            nc = o.loc[non_cap]
            n_bad_noncap = int(((nc.obsval != -99.0) | (nc.weight != 0.0)).sum())
            if n_bad_noncap:
                discrepancies.append(
                    {
                        "batch": batch_dir.name,
                        "fid": uid,
                        "issue": f"{n_bad_noncap} non-capture obs with obsval != -99 or weight != 0",
                    }
                )

            oc = o.loc[cap]
            d_obs = np.abs(oc.obsval.to_numpy(dtype=float) - recon.target.to_numpy(dtype=float))
            d_w = np.abs(oc.weight.to_numpy(dtype=float) - recon.weight.to_numpy(dtype=float))
            tol_w = WEIGHT_RTOL * np.maximum(1.0, np.abs(recon.weight.to_numpy(dtype=float)))
            d_sd = np.abs(
                oc.standard_deviation.to_numpy(dtype=float)
                - recon.standard_deviation.to_numpy(dtype=float)
            )
            bad_obs, bad_w, bad_sd = d_obs > OBSVAL_ATOL, d_w > tol_w, d_sd > 1e-6
            for label, bad in (
                ("obsval", bad_obs),
                ("weight", bad_w),
                ("standard_deviation", bad_sd),
            ):
                if bad.any():
                    discrepancies.append(
                        {
                            "batch": batch_dir.name,
                            "fid": uid,
                            "issue": f"{int(bad.sum())} {label} mismatches (max {float(np.nanmax(np.where(bad, {'obsval': d_obs, 'weight': d_w, 'standard_deviation': d_sd}[label], 0))):.3e})",
                        }
                    )

            if audit is not None:
                a = audit[audit.fid == uid].set_index("date").sort_index()
                if len(a) != len(cap) or not a.index.equals(cap):
                    audit_csv_mismatch += 1
                    discrepancies.append(
                        {
                            "batch": batch_dir.name,
                            "fid": uid,
                            "issue": f"weight_audit.csv has {len(a)} rows, reconstruction {len(cap)} captures",
                        }
                    )
                else:
                    if (
                        np.abs(
                            a.weight_final.to_numpy(dtype=float)
                            - recon.weight.to_numpy(dtype=float)
                        )
                        > tol_w
                    ).any():
                        audit_csv_mismatch += 1
                        discrepancies.append(
                            {
                                "batch": batch_dir.name,
                                "fid": uid,
                                "issue": "weight_audit.csv weight_final != reconstruction",
                            }
                        )
                    if (a.member_count.to_numpy() != recon.member_count.to_numpy()).any():
                        audit_csv_mismatch += 1
                        discrepancies.append(
                            {
                                "batch": batch_dir.name,
                                "fid": uid,
                                "issue": "weight_audit.csv member_count != reconstruction",
                            }
                        )

            r = recon.copy()
            r.insert(0, "fid", uid)
            r.insert(1, "batch", batch_dir.name)
            r["weight_pst"] = oc.weight.to_numpy(dtype=float)
            r["obsval_pst"] = oc.obsval.to_numpy(dtype=float)
            r["sd_pst"] = oc.standard_deviation.to_numpy(dtype=float)
            r["in_baseline_weighted"] = [(fid_l, d) in baseline_weighted for d in cap]
            rows.append(r.reset_index().rename(columns={"index": "date"}))

        # SWE ------------------------------------------------------------------------
        etf_w_all = np.concatenate(primary_w) if primary_w else np.array([])
        etf_sd_all = np.concatenate(primary_sd) if primary_sd else np.array([])
        swe_obs = obs[obs.kind == "swe"]
        pos = swe_obs[swe_obs.obsval > 0.0]
        exp_w, exp_sd, c = swe_expected_weights(
            pos.obsval.to_numpy(dtype=float),
            etf_w_all,
            etf_sd_all,
            sd_frac=sd_frac,
            sd_floor=sd_floor,
            phi_share=phi_share,
        )
        n_bad_swe_w = int(
            (
                np.abs(pos.weight.to_numpy(dtype=float) - exp_w)
                > 1e-6 * np.maximum(1.0, np.abs(exp_w))
            ).sum()
        )
        n_bad_swe_sd = int(
            (np.abs(pos.standard_deviation.to_numpy(dtype=float) - exp_sd) > 1e-6).sum()
        )
        zero_bad = int((swe_obs[swe_obs.obsval <= 0.0].weight != 0.0).sum())
        # SWE obsvals identical to container and to the baseline run for the same sites
        swe_ok_container = True
        for fid_l in batch_fids:
            j = uids.index(lower_to_uid[fid_l])
            s = (
                swe_obs[swe_obs.fid_lower == fid_l]
                .set_index("day_index")
                .sort_index()
                .obsval.to_numpy(dtype=float)
            )
            cont = swe_arr[:, j].astype(float)
            cont = np.where(np.isfinite(cont), cont, -99.0)
            if len(s) != n_days or not np.allclose(s, cont, atol=1e-6):
                swe_ok_container = False
        b = baseline_swe[baseline_swe.fid_lower.isin(batch_fids)]
        merged = swe_obs[["fid_lower", "day_index", "obsval", "weight"]].merge(
            b, on=["fid_lower", "day_index"], suffixes=("", "_base")
        )
        swe_obsval_identical_to_baseline = bool(
            len(merged) == len(swe_obs)
            and np.allclose(merged.obsval, merged.obsval_base, atol=1e-6)
        )
        bw = (
            merged.loc[merged.weight_base > 0, "weight"]
            / merged.loc[merged.weight_base > 0, "weight_base"]
        )
        if (
            n_bad_swe_w
            or n_bad_swe_sd
            or zero_bad
            or not swe_ok_container
            or not swe_obsval_identical_to_baseline
        ):
            discrepancies.append(
                {
                    "batch": batch_dir.name,
                    "issue": f"SWE: weight mismatches {n_bad_swe_w}, sd mismatches {n_bad_swe_sd}, nonzero weight on non-positive obs {zero_bad}, obsval==container {swe_ok_container}, obsval==baseline {swe_obsval_identical_to_baseline}",
                }
            )
        swe_report.append(
            {
                "batch": batch_dir.name,
                "n_swe_positive": int(len(pos)),
                "swe_c": c,
                "swe_weight_ratio_vs_baseline_median": float(bw.median()) if len(bw) else np.nan,
                "swe_weight_ratio_vs_baseline_min": float(bw.min()) if len(bw) else np.nan,
                "swe_weight_ratio_vs_baseline_max": float(bw.max()) if len(bw) else np.nan,
                "expected_primary_etf_phi": float(((etf_w_all * etf_sd_all) ** 2).sum()),
                "expected_swe_phi": float(((exp_w * exp_sd) ** 2).sum()),
                "swe_obsval_identical_to_baseline": swe_obsval_identical_to_baseline,
            }
        )

    table = pd.concat(rows, ignore_index=True)
    table["year"] = table.date.dt.year
    table["fid_lower"] = table.fid.str.lower()
    weighted = table[table.weight > 0]

    # Losses / gains vs baseline ------------------------------------------------------------
    new_weighted = set(zip(weighted.fid_lower, weighted.date))
    audited_lower = set(table.fid_lower)
    lost_keys = {k for k in baseline_weighted if k[0] in audited_lower} - new_weighted
    gained = new_weighted - baseline_weighted
    tab_idx = table.set_index(["fid_lower", "date"])
    loss_rows = []
    for fid_l, d in sorted(lost_keys):
        uid = lower_to_uid[fid_l]
        if (fid_l, d) in tab_idx.index:
            r = tab_idx.loc[(fid_l, d)]
            base = {
                "member_count": int(r.member_count),
                "eto": float(r.eto),
                "eto_floor_excluded": bool(r.eto_floor_excluded),
                "ssebop": r.get("ssebop", np.nan),
                "ptjpl": r.get("ptjpl", np.nan),
            }
        else:
            base = {
                "member_count": 0,
                "eto": float(eto_arr[dates.get_loc(d), uids.index(uid)]),
                "eto_floor_excluded": False,
                "ssebop": np.nan,
                "ptjpl": np.nan,
            }
        led = ledger_idx.loc[(uid, d)] if (uid, d) in ledger_idx.index else None
        base["corrected"] = float(led.corrected) if led is not None else np.nan
        base["status"] = str(led.status) if led is not None else ""
        cat = classify_loss(pd.Series(base), min_etf, max_etf)
        loss_rows.append({"fid": uid, "date": d.strftime("%Y-%m-%d"), **base, "category": cat})
    losses = pd.DataFrame(
        loss_rows,
        columns=[
            "fid",
            "date",
            "member_count",
            "eto",
            "eto_floor_excluded",
            "ssebop",
            "ptjpl",
            "corrected",
            "status",
            "category",
        ],
    )
    loss_counts = losses.category.value_counts().to_dict() if len(losses) else {}
    unexplained_losses = int(sum(v for k, v in loss_counts.items() if k not in EXPLAINED_LOSS))

    # Counts -------------------------------------------------------------------------------
    by_year = weighted.groupby("year").size()
    base_by_year = (
        pd.Series([d.year for _, d in baseline_weighted if _ in audited_lower])
        .value_counts()
        .sort_index()
    )
    counts = (
        pd.DataFrame({"weighted_new": by_year, "weighted_baseline": base_by_year})
        .fillna(0)
        .astype(int)
    )
    counts["delta"] = counts.weighted_new - counts.weighted_baseline
    by_site = weighted.groupby("fid").size().rename("weighted_new").to_frame()
    by_site["weighted_baseline"] = (
        pd.Series([f for f, _ in baseline_weighted])
        .map(lambda f: lower_to_uid.get(f, f))
        .value_counts()
    )
    by_site = by_site.fillna(0).astype(int)
    by_site["batch"] = by_site.index.map(lambda u: f"batch_{batch_of[u]:03d}")
    by_ct = (
        table.groupby("member_count")
        .agg(captures=("weight", "size"), weighted=("weight", lambda w: int((w > 0).sum())))
        .reset_index()
    )
    only_ptjpl = (
        int(((table.member_count == 1) & table.ptjpl.notna()).sum()) if "ptjpl" in table else 0
    )
    only_ssebop = (
        int(((table.member_count == 1) & table.ssebop.notna()).sum()) if "ssebop" in table else 0
    )
    n_captures = int(len(table))
    ptjpl_only_share = only_ptjpl / n_captures if n_captures else np.nan

    eligibility_rule_holds = bool(
        (
            (table.weight > 0) == ((table.member_count >= min_members) & ~table.eto_floor_excluded)
        ).all()
    )

    summary = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": os.path.abspath(config_path),
        "container": cfg.container_path,
        "pest_run_dir": str(pest_run_dir),
        "baseline_pest_archive": str(baseline_archive),
        "resolved": {
            "etf_target_model": cfg.etf_target_model,
            "etf_ensemble_members": members,
            "etf_target_instrument": instrument,
            "mask": mask,
            "etf_weighting_spread_floor": spread_floor,
            "etf_weighting_min_members": min_members,
            "etf_weighting_eto_floor": eto_floor,
            "etf_weighting_fixed_sd": fixed_sd,
            "swe_weighting_sd_frac": sd_frac,
            "swe_weighting_sd_floor": sd_floor,
            "swe_weighting_phi_share": phi_share,
            "prior_regularization_fraction": getattr(cfg, "prior_regularization_fraction", None),
            "ingest_rules": {"min_etf": min_etf, "max_etf": max_etf},
            "expected_noptmax": expected_noptmax,
            "expected_reals": expected_reals,
        },
        "manifest": manifest_checks,
        "pst": pst_meta,
        "etf": {
            "captures": n_captures,
            "weighted": int(len(weighted)),
            "zero_weight": int(n_captures - len(weighted)),
            "eto_floor_excluded": int(table.eto_floor_excluded.sum()),
            "eto_floor_excluded_with_two_members": int(
                (table.eto_floor_excluded & (table.member_count >= 2)).sum()
            ),
            "by_member_count": by_ct.to_dict(orient="records"),
            "ptjpl_only_captures": only_ptjpl,
            "ssebop_only_captures": only_ssebop,
            "ptjpl_only_share": ptjpl_only_share,
            "ptjpl_only_review_threshold": PTJPL_ONLY_REVIEW_SHARE,
            "ptjpl_only_below_threshold": bool(ptjpl_only_share < PTJPL_ONLY_REVIEW_SHARE),
            "eligibility_rule_holds": eligibility_rule_holds,
            "sum_weight": float(weighted.weight.sum()),
            "sum_weight_sq": float((weighted.weight**2).sum()),
            "weight_min_positive": float(weighted.weight.min()) if len(weighted) else np.nan,
            "weight_max": float(weighted.weight.max()) if len(weighted) else np.nan,
        },
        "vs_baseline": {
            "baseline_weighted": int(len({k for k in baseline_weighted if k[0] in audited_lower})),
            "new_weighted": int(len(new_weighted)),
            "retained": int(len(new_weighted & baseline_weighted)),
            "gained": int(len(gained)),
            "lost": int(len(lost_keys)),
            "lost_by_category": loss_counts,
            "unexplained_losses": unexplained_losses,
            "weighted_by_year": counts.reset_index()
            .rename(columns={"index": "year"})
            .to_dict(orient="records"),
            "early_period_2013_2017_new": int(
                counts.loc[counts.index <= 2017, "weighted_new"].sum()
            ),
            "early_period_2013_2017_baseline": int(
                counts.loc[counts.index <= 2017, "weighted_baseline"].sum()
            ),
        },
        "swe": swe_report,
        "weight_audit_csv_mismatches": audit_csv_mismatch,
        "discrepancies": discrepancies,
        "n_discrepancies": len(discrepancies),
    }
    summary["pass"] = bool(
        len(discrepancies) == 0
        and unexplained_losses == 0
        and eligibility_rule_holds
        and manifest_checks["sites_match_container_minus_excluded"]
        and summary["etf"]["ptjpl_only_below_threshold"]
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_out = table.drop(columns=["fid_lower"])
    table_out["date"] = table_out.date.dt.strftime("%Y-%m-%d")
    table_out.to_csv(out / "objective_weight_rows.csv", index=False)
    losses.to_csv(out / "objective_weight_losses.csv", index=False)
    by_site.reset_index().rename(columns={"index": "fid"}).to_csv(
        out / "objective_weight_counts.csv", index=False
    )
    (out / "objective_weight_audit.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    return summary


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o)}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--baseline-pest-archive", default=DEFAULT_BASELINE_ARCHIVE)
    p.add_argument("--ledger", default=DEFAULT_LEDGER)
    p.add_argument("--out-dir", default=QA_ROOT)
    p.add_argument("--noptmax", type=int, default=3, help="noptmax the .pst files must carry")
    p.add_argument("--reals", type=int, default=200, help="ies_num_reals the .pst files must carry")
    args = p.parse_args()
    s = run_audit(
        args.config, args.baseline_pest_archive, args.ledger, args.out_dir, args.noptmax, args.reals
    )
    e, v = s["etf"], s["vs_baseline"]
    print(
        f"captures {e['captures']}  weighted {e['weighted']}  eto-floor zeroed {e['eto_floor_excluded']}  ptjpl-only share {e['ptjpl_only_share']:.4f}"
    )
    print(
        f"vs baseline: retained {v['retained']}  gained {v['gained']}  lost {v['lost']} {v['lost_by_category']}  unexplained {v['unexplained_losses']}"
    )
    print(f"manifest: {s['manifest']}")
    for r in s["swe"]:
        print(
            f"SWE {r['batch']}: c={r['swe_c']:.4f} ratio vs baseline median {r['swe_weight_ratio_vs_baseline_median']:.4f} obsval==baseline {r['swe_obsval_identical_to_baseline']}"
        )
    for d in s["discrepancies"]:
        print("DISCREPANCY:", d)
    print("G9", "PASS" if s["pass"] else "FAIL")
    raise SystemExit(0 if s["pass"] else 1)


if __name__ == "__main__":
    main()

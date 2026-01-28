import json
import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def _load_counts_json(paths: List[str]) -> List[Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Load one or more observation-count JSON files.
    Each file format: {station_id: {mask: {year_str: count_int}}}
    Returns a list of dicts.
    """
    dicts = []
    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith('.json'):
                        fp = os.path.join(root, f)
                        try:
                            with open(fp, 'r') as h:
                                loaded = json.load(h)
                                instrument_hint = 'sentinel' if 'sentinel' in fp.lower() else (
                                    'landsat' if 'landsat' in fp.lower() else None)
                                # Preserve NDVI instrument by wrapping mask-level under 'none_<instrument>'
                                # or by renaming model 'none' -> 'none_<instrument>' for model-level inputs
                                try:
                                    level1 = next(iter((loaded or {}).values()))
                                    level2 = next(iter((level1 or {}).values())) if isinstance(level1, dict) else None
                                    inner = next(iter((level2 or {}).values())) if isinstance(level2, dict) else None
                                    is_mask_level = isinstance(level1, dict) and isinstance(level2, dict) and not isinstance(inner, dict)
                                except Exception:
                                    is_mask_level = False
                                if is_mask_level and instrument_hint is not None:
                                    wrapped = {}
                                    for sid, masks in (loaded or {}).items():
                                        wrapped[str(sid)] = {f'none_{instrument_hint}': masks}
                                    dicts.append(wrapped)
                                else:
                                    if instrument_hint is not None:
                                        # Rename model key 'none' if present
                                        renamed = {}
                                        for sid, models in (loaded or {}).items():
                                            if isinstance(models, dict) and 'none' in models:
                                                models_copy = dict(models)
                                                models_copy[f'none_{instrument_hint}'] = models_copy.pop('none')
                                                renamed[str(sid)] = models_copy
                                            else:
                                                renamed[str(sid)] = models
                                        dicts.append(renamed)
                                    else:
                                        dicts.append(loaded)
                        except Exception:
                            continue
        else:
            try:
                with open(p, 'r') as h:
                    loaded = json.load(h)
                    instrument_hint = 'sentinel' if 'sentinel' in p.lower() else (
                        'landsat' if 'landsat' in p.lower() else None)
                    # Preserve NDVI instrument by wrapping mask-level under 'none_<instrument>'
                    # or by renaming model 'none' -> 'none_<instrument>' for model-level inputs
                    try:
                        level1 = next(iter((loaded or {}).values()))
                        level2 = next(iter((level1 or {}).values())) if isinstance(level1, dict) else None
                        inner = next(iter((level2 or {}).values())) if isinstance(level2, dict) else None
                        is_mask_level = isinstance(level1, dict) and isinstance(level2, dict) and not isinstance(inner, dict)
                    except Exception:
                        is_mask_level = False
                    if is_mask_level and instrument_hint is not None:
                        wrapped = {}
                        for sid, masks in (loaded or {}).items():
                            wrapped[str(sid)] = {f'none_{instrument_hint}': masks}
                        dicts.append(wrapped)
                    else:
                        if instrument_hint is not None:
                            renamed = {}
                            for sid, models in (loaded or {}).items():
                                if isinstance(models, dict) and 'none' in models:
                                    models_copy = dict(models)
                                    models_copy[f'none_{instrument_hint}'] = models_copy.pop('none')
                                    renamed[str(sid)] = models_copy
                                else:
                                    renamed[str(sid)] = models
                            dicts.append(renamed)
                        else:
                            dicts.append(loaded)
            except Exception:
                continue
    return dicts


def _merge_counts(
    dicts: List[Dict[str, Dict[str, Dict[str, Union[int, Dict[str, int]]]]]]
) -> Dict[str, Dict[str, Dict[str, Dict[int, int]]]]:
    """
    Merge multiple observation-count dicts, keeping max count per (station, model, mask, year).
    Converts year keys to int.
    Accepts either model-level input {sid: {model: {mask: {year: count}}}} or older mask-level
    {sid: {mask: {year: count}}} which will be folded under model 'none'.
    Output: {station_id: {model: {mask: {year_int: count_int}}}}
    """
    merged: Dict[str, Dict[str, Dict[str, Dict[int, int]]]] = {}
    for d in dicts:
        for sid, level2 in (d or {}).items():
            sid_str = str(sid)
            merged.setdefault(sid_str, {})

            if not isinstance(level2, dict) or not level2:
                continue

            sample_val = next(iter(level2.values()))

            if isinstance(sample_val, dict):
                inner_sample_val = next(iter(sample_val.values())) if sample_val else None
                if isinstance(inner_sample_val, dict):
                    # model-level
                    for model, masks in level2.items():
                        model_str = str(model)
                        merged[sid_str].setdefault(model_str, {})
                        for msk, years in (masks or {}).items():
                            msk_str = str(msk)
                            merged[sid_str][model_str].setdefault(msk_str, {})
                            for y_str, cnt in (years or {}).items():
                                try:
                                    y = int(y_str)
                                    c = int(cnt)
                                except Exception:
                                    continue
                                prev = merged[sid_str][model_str][msk_str].get(y, 0)
                                merged[sid_str][model_str][msk_str][y] = max(prev, c)
                else:
                    # mask-level; fold under model 'none'
                    model_str = 'none'
                    merged[sid_str].setdefault(model_str, {})
                    for msk, years in level2.items():
                        msk_str = str(msk)
                        merged[sid_str][model_str].setdefault(msk_str, {})
                        for y_str, cnt in (years or {}).items():
                            try:
                                y = int(y_str)
                                c = int(cnt)
                            except Exception:
                                continue
                            prev = merged[sid_str][model_str][msk_str].get(y, 0)
                            merged[sid_str][model_str][msk_str][y] = max(prev, c)
            else:
                continue
    return merged


def _collect_all_years(merged: Dict[str, Dict[str, Dict[str, Dict[int, int]]]]) -> List[int]:
    years = set()
    for _, models in merged.items():
        for _, masks in models.items():
            for _, ymap in masks.items():
                years.update(ymap.keys())
    return sorted(years)


def summarize_observation_counts(
    json_paths: List[str],
    min_obs_per_year: int = 8,
    all_years: Optional[List[int]] = None,
    model_whitelist: Optional[List[str]] = None,
    mask_whitelist: Optional[List[str]] = None,
    station_whitelist: Optional[List[str]] = None,
    select_stations: Optional[List[str]] = None,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Build a compact diagnostic summary from one or more observation-count JSONs.

    Inputs
    - json_paths: list of file or directory paths containing JSON counts
    - min_obs_per_year: threshold below which a year's observations are flagged as low
    - all_years: optional explicit list of years to evaluate; if None, uses union of years found
    - mask_whitelist: optionally restrict masks (e.g., ['irr', 'inv_irr'])
    - station_whitelist: optionally restrict station IDs
    - output_csv: if provided, write a tabular summary CSV
    - output_json: if provided, write a dense JSON with details and flags

    Returns
    - summary_df: wide summary (one row per station per mask)
    - dense_dict: nested structure with per-station detail and flags
    """

    loaded = _load_counts_json(json_paths)
    merged = _merge_counts(loaded)

    stations_filter = select_stations if select_stations is not None else station_whitelist
    if stations_filter is not None:
        merged = {sid: merged[sid] for sid in merged if sid in set(stations_filter)}

    if model_whitelist is not None:
        for sid in list(merged.keys()):
            merged[sid] = {m: merged[sid].get(m, {}) for m in model_whitelist}

    if mask_whitelist is not None:
        for sid in list(merged.keys()):
            for model in list(merged[sid].keys()):
                merged[sid][model] = {m: merged[sid][model].get(m, {}) for m in mask_whitelist}

    years = all_years if all_years is not None else _collect_all_years(merged)
    years_set = set(years)

    rows = []
    dense: Dict[str, dict] = {}

    for sid, models in merged.items():
        dense[sid] = {"models": {}, "summary": {}}

        station_any_data = False
        station_mask_presence_overall = {}

        models_to_use = models.keys() if model_whitelist is None else model_whitelist

        for model in models_to_use:
            masks = models.get(model, {}) or {}
            dense[sid]["models"].setdefault(model, {"masks": {}, "summary": {}})

            masks_to_use = masks.keys() if mask_whitelist is None else mask_whitelist
            mask_presence = {}

            for msk in masks_to_use:
                year_counts = {int(y): int(c) for y, c in (masks.get(msk, {}) or {}).items() if isinstance(y, int)}
                normalized_counts = {y: int(year_counts.get(y, 0)) for y in years}

                years_with_data = [y for y, c in normalized_counts.items() if c > 0]
                years_low = [y for y, c in normalized_counts.items() if 0 < c < min_obs_per_year]
                years_zero = [y for y in years if normalized_counts.get(y, 0) == 0]

                has_any_data = len(years_with_data) > 0
                mask_presence[msk] = has_any_data
                station_mask_presence_overall[msk] = station_mask_presence_overall.get(msk, False) or has_any_data
                station_any_data = station_any_data or has_any_data

                flags = []
                if not has_any_data:
                    flags.append("no_data")
                if years_low:
                    flags.append(f"low_obs:{','.join(map(str, years_low))}")
                if years_zero and has_any_data:
                    flags.append(f"missing_years:{','.join(map(str, years_zero))}")

                dense[sid]["models"][model]["masks"][msk] = {
                    "year_counts": normalized_counts,
                    "years_with_data": years_with_data,
                    "years_below_threshold": years_low,
                    "years_zero": years_zero,
                    "flags": flags,
                }

                rows.append({
                    "station": sid,
                    "model": model,
                    "mask": msk,
                    "years_with_data_count": len(years_with_data),
                    "years_below_threshold_count": len(years_low),
                    "years_zero_count": len(years_zero),
                    "years_with_data": ",".join(map(str, years_with_data)) if years_with_data else "",
                    "years_below_threshold": ",".join(map(str, years_low)) if years_low else "",
                    "years_zero": ",".join(map(str, years_zero)) if years_zero else "",
                    "has_any_data": has_any_data,
                    "flags": ";".join(flags) if flags else "",
                })

            # Model-level summary for station
            masks_with_data = [m for m, present in mask_presence.items() if present]
            masks_without_data = [m for m, present in mask_presence.items() if not present]
            model_flags = []
            if not masks_with_data:
                model_flags.append("model_no_data")
            elif masks_without_data and masks_with_data:
                model_flags.append("partial_masks")

            dense[sid]["models"][model]["summary"] = {
                "masks_with_any_data": masks_with_data,
                "masks_with_no_data": masks_without_data,
                "flags": model_flags,
            }

        # Station-level summary across models
        station_flags = []
        if not station_any_data:
            station_flags.append("station_no_data")
        elif any(not present for present in station_mask_presence_overall.values()):
            station_flags.append("partial_masks_overall")

        dense[sid]["summary"] = {
            "any_data": station_any_data,
            "mask_presence_overall": station_mask_presence_overall,
            "flags": station_flags,
        }

    summary_df = pd.DataFrame(rows)
    # Sort for readability
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["station", "mask"]).reset_index(drop=True)

    if output_csv:
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        except Exception:
            pass
        summary_df.to_csv(output_csv, index=False)

    if output_json:
        try:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
        except Exception:
            pass
        with open(output_json, 'w') as fp:
            json.dump(dense, fp, indent=2)

    # Print stations that have no observations in both irr and inv_irr for any year
    # Aggregates across models: if any model has counts for a mask/year, it is considered present
    try:
        for sid, models in merged.items():
            missing_years: List[int] = []
            for y in years:
                irr_max = 0
                inv_irr_max = 0
                for _, masks in (models or {}).items():
                    if not isinstance(masks, dict):
                        continue
                    irr_max = max(irr_max, int((masks.get('irr') or {}).get(y, 0) or 0))
                    inv_irr_max = max(inv_irr_max, int((masks.get('inv_irr') or {}).get(y, 0) or 0))
                if irr_max == 0 and inv_irr_max == 0:
                    missing_years.append(y)
            if missing_years:
                years_str = ",".join(map(str, sorted(missing_years)))
                print(f"[rs_diagnostics] Station {sid} lacks irr/inv_irr observations in years: {years_str}")
    except Exception:
        # Do not fail the summary if printing diagnostics encounters edge cases
        pass

    # Build compact summary per instrument, with algorithm-level stats
    instrument_summary: Dict[str, Dict[str, Dict[str, Dict[str, Union[float, List[int]]]]]] = {}
    for sid, models in merged.items():
        # Collect per-instrument, per-algorithm yearly maxima
        landsat_alg_years: Dict[str, Dict[int, int]] = {}
        sentinel_alg_years: Dict[str, Dict[int, int]] = {}

        for model, masks in (models or {}).items():
            per_year = {y: 0 for y in years}
            for msk, ymap in (masks or {}).items():
                for y in years:
                    c = int((ymap or {}).get(y, 0) or 0)
                    if c > per_year[y]:
                        per_year[y] = c

            if model == 'none_landsat':
                landsat_alg_years['ndvi'] = per_year
            elif model == 'none_sentinel':
                sentinel_alg_years['ndvi'] = per_year
            elif model == 'none':
                # Ambiguous NDVI; assign to sentinel for visibility  # likely error: NDVI source ambiguous
                sentinel_alg_years['ndvi'] = per_year
            else:
                landsat_alg_years[model] = per_year

        # Compute stats per algorithm
        landsat_stats = {}
        for alg, ymap in landsat_alg_years.items():
            mean_obs = float(sum(ymap.values()) / len(years)) if years else 0.0
            zero_years = [y for y, c in ymap.items() if c == 0]
            landsat_stats[alg] = {'mean_obs': mean_obs, 'years_w_zero_obs': zero_years}

        sentinel_stats = {}
        for alg, ymap in sentinel_alg_years.items():
            mean_obs = float(sum(ymap.values()) / len(years)) if years else 0.0
            zero_years = [y for y, c in ymap.items() if c == 0 and y >= 2017]
            sentinel_stats[alg] = {'mean_obs': mean_obs, 'years_w_zero_obs': zero_years}

        instrument_summary[sid] = {
            'landsat': landsat_stats,
            'sentinel': sentinel_stats,
        }

    return summary_df, instrument_summary


def merge_counts_dict(json_paths: List[str], out_path: str) -> str:
    """
    Merge per-run observation count JSONs into a single nested structure and write to `out_path`.
    Accepts file paths or directories; directories are scanned recursively for .json files.

    Output JSON structure: {station: {model: {mask: {year_str: count}}}}
    (Older inputs without the model level are folded under model key 'none'.)
    Returns the written path.
    """
    dicts = _load_counts_json(json_paths)
    merged = _merge_counts(dicts)

    # Convert years back to strings for JSON stability
    serializable = {}
    for sid, models in merged.items():
        serializable[sid] = {}
        for model, masks in models.items():
            serializable[sid][model] = {}
            for msk, years in masks.items():
                serializable[sid][model][msk] = {str(y): int(c) for y, c in years.items()}

    with open(out_path, 'w') as fp:
        json.dump(serializable, fp, indent=2)
    return out_path


__all__ = ["summarize_observation_counts", "merge_counts_dict"]

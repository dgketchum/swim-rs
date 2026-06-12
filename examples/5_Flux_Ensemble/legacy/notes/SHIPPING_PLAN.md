# Plan: Ship Example 5 as Run 11 Reference

## Context

Run 11 is the reference calibration: 60 US cropland flux sites, 1995–2025,
8-param PEST++ IES against a 6-model unmasked Landsat ETf ensemble. Daily R²
0.654 mean, bias +0.101 (least biased model). Monthly R² 0.858 median.

The goal is to make Example 5 self-contained so users can clone the repo and
run calibration + evaluation without EE access. We ship the .swim container
(18MB packed) and flux tower validation CSVs (~22MB zipped). The Python scripts
must reproduce the exact Run 11 logic.

## Path Resolution

The TOML has `root = "/data/ssd1/swim"` (machine-specific). All scripts use a
sentinel check: if `/data/ssd2/swim` exists (dev machine), use TOML root
directly; otherwise, pass `project_root_override=str(project_dir.parent)` which
resolves all paths under `examples/5_Flux_Ensemble/`. This makes the example
portable without changing the TOML.

## Changes

### 1. Pack and ship the .swim container

```python
container = SwimContainer.open("/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim")
container.pack("examples/5_Flux_Ensemble/data/5_Flux_Ensemble.swim")
```

This creates a single zip file at `examples/5_Flux_Ensemble/data/5_Flux_Ensemble.swim`.
`SwimContainer.open()` auto-detects zip vs directory format.

### 2. Zip and ship flux tower validation data

The 60 flux CSVs live at `examples/5_Flux_Ensemble/data/flux/` (61MB raw,
~22MB zipped). But `evaluate.py` expects `cfg.data_dir + "/daily_flux_files/"`.

**Fix evaluate.py** to look for `flux/` instead of `daily_flux_files/`:
```python
flux_dir = os.path.join(cfg.data_dir, "flux")
```

Then zip the flux directory:
```bash
cd examples/5_Flux_Ensemble/data && zip -r flux_daily.zip flux/
```

Add auto-extract logic to evaluate.py that unzips `flux_daily.zip` → `flux/`
if the directory doesn't exist.

### 3. Update .gitignore

**File: `.gitignore`**

Remove the explicit ignores for container_prep.py and setup_shapefile.py (lines
193-194):
```
-/examples/5_Flux_Ensemble/container_prep.py
-/examples/5_Flux_Ensemble/setup_shapefile.py
```

Add whitelists for the container and flux zip:
```
!/examples/5_Flux_Ensemble/data/5_Flux_Ensemble.swim
!/examples/5_Flux_Ensemble/data/flux_daily.zip
```

### 4. Clean calibrate.py `__main__`

**File: `examples/5_Flux_Ensemble/calibrate.py`**

Change the `__main__` block from Run 11 dev state to a generic shipped form:

```python
if __name__ == "__main__":
    import time

    cfg = _load_config()
    DEBUG_FIELDS = None

    results = os.path.join(cfg.project_ws, "results", "calibration")
    t0 = time.time()
    run_pest_sequence(
        cfg,
        results,
        pdc_remove=False,
        debug_fields=DEBUG_FIELDS,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
```

Remove the `# Run 11: ...` comment and `run11_full_period` path.

### 5. Fix container_prep.py

**File: `examples/5_Flux_Ensemble/container_prep.py`**

- Fix `_load_config()` to match the sentinel pattern used by all other scripts:
  ```python
  def _load_config() -> ProjectConfig:
      project_dir = Path(__file__).resolve().parent
      conf = project_dir / "5_Flux_Ensemble.toml"
      cfg = ProjectConfig()
      if os.path.isdir("/data/ssd2/swim"):
          cfg.read_config(str(conf))
      else:
          cfg.read_config(str(conf), project_root_override=str(project_dir.parent))
      return cfg
  ```

- Remove stale `print("  python run.py")` at bottom — `run.py` does not exist.
  Replace with `print("  python calibrate.py")`.

### 6. Fix data_extract.py `__main__`

**File: `examples/5_Flux_Ensemble/data_extract.py`**

Uncomment all extraction steps so the full workflow is documented:
```python
extract_snodas(config)
extract_properties(config)
extract_ndvi(config, select_sites, get_sentinel=True)
extract_gridmet(config, select_sites)
extract_openet_etf_assets(...)
```

This makes the script self-documenting as the complete extraction workflow.
Users who ship-with-container skip this step entirely.

### 7. Fix evaluate.py flux path

**File: `examples/5_Flux_Ensemble/evaluate.py`**

- Change `flux_dir` from `daily_flux_files` to `flux`:
  ```python
  flux_dir = os.path.join(cfg.data_dir, "flux")
  ```

- Add auto-extract at the top of `__main__`:
  ```python
  flux_zip = os.path.join(cfg.data_dir, "flux_daily.zip")
  if not os.path.isdir(flux_dir) and os.path.isfile(flux_zip):
      import zipfile
      with zipfile.ZipFile(flux_zip, "r") as z:
          z.extractall(cfg.data_dir)
  ```

- Change default `--openet-source` from `"volk"` to `"diy"`. Volk CSVs are
  external benchmarking data that won't ship; `diy` computes OpenET ET from
  the container's ETf × ETo, which is self-contained.

### 8. Rewrite README.md

**File: `examples/5_Flux_Ensemble/README.md`**

Rewrite for Run 11. Structure:
- Title + summary: 60 US cropland flux stations, 1995–2025
- Quick start (clone → calibrate → evaluate, using shipped container)
- Data sources table
- Full workflow (for users who want to rebuild from scratch)
- Results summary (Run 11 daily + monthly tables)
- Files table (all shipped .py scripts)
- Configuration reference

### 9. Track container_prep.py and setup_shapefile.py

`git add` both files (currently gitignored by explicit rules in `.gitignore`).

### 10. No changes needed

- `5_Flux_Ensemble.toml` — correct (1995-2025, mask_mode=none, 6-model ensemble)
- `etf_asset_extract.py` — correct (CLI-driven, stale date defaults are
  harmless since users pass explicit args)
- `copy_openet_assets.py` — correct (same reasoning)
- `setup_shapefile.py` — correct (clean argparse, no machine paths)
- `params.csv` — correct (PestBuilder regenerates at runtime)

## File Summary

| File | Action |
|------|--------|
| `.gitignore` | Remove container_prep/setup_shapefile ignores, whitelist .swim + flux zip |
| `examples/5_Flux_Ensemble/data/5_Flux_Ensemble.swim` | Pack from DirectoryStore and git-track |
| `examples/5_Flux_Ensemble/data/flux_daily.zip` | Zip flux CSVs and git-track |
| `examples/5_Flux_Ensemble/calibrate.py` | Clean __main__ (remove run11 reference) |
| `examples/5_Flux_Ensemble/container_prep.py` | Fix _load_config, fix stale run.py print |
| `examples/5_Flux_Ensemble/data_extract.py` | Uncomment all extraction steps |
| `examples/5_Flux_Ensemble/evaluate.py` | flux path fix, auto-extract zip, default diy |
| `examples/5_Flux_Ensemble/README.md` | Rewrite for Run 11 |

## Verification

1. Pack container, verify it opens: `SwimContainer.open("examples/.../5_Flux_Ensemble.swim")`
2. Zip flux data, verify evaluate.py auto-extracts
3. `ruff check --fix . && ruff format .`
4. `pytest tests/ -v` (25 kc_max failures expected, no new failures)
5. Dry-run: simulate portable path resolution by checking config with
   project_root_override pointing to `examples/`
6. Verify all tracked Example 5 files are consistent with Run 11 configuration

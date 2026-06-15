"""Generate the Example 6 landcover_crop.csv that feeds the container LULC gate.

The irrigation classifier's cropland gate reads `properties/land_cover/glc10`,
ingested from this CSV's `glc10_lc` column (container_prep uses
`lulc_column="modis_lc"`, `extra_lulc_column="glc10_lc"`).

Provenance fix: the original CSV carried the *raw* GLC10 code per site (e.g.
US-Tw2=20 Forest, AU-Rgf=30 Grassland), so genuine cropland that GLC10
mislabels was gated out of the irrigation water balance. The cohort shapefile
already carries the curated cropland determination in `glc10_lulc` (=10 for
every retained site). This script writes that curated value into `glc10_lc`
for every cohort site, so the gate honors the cropland include/exclude decision.

This corrects the *cropland gate* only (a cropland include/exclude override,
which is permitted for Example 6). Irrigation status remains decided solely by
the internal water-balance classifier — `modis_lc` is preserved verbatim and no
irrigation override is written. Non-cohort rows keep their raw `glc10_lc` so
they stay correctly gated non-crop.

Usage:
    python landcover_crop.py [--config PATH]
"""

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

from swimrs.swim.config import ProjectConfig

DEFAULT_TOML = (
    Path(__file__).resolve().parent / "6_Flux_International_LSEnsemble_POR_annual2yr.toml"
)


def main(config_path: str | None = None) -> Path:
    cfg = ProjectConfig()
    cfg.read_config(str(Path(config_path) if config_path else DEFAULT_TOML))

    fid = cfg.feature_id_col
    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    assert "glc10_lulc" in gdf.columns, (
        f"cohort shapefile lacks curated 'glc10_lulc'; got {list(gdf.columns)}"
    )
    cohort = gdf.set_index(fid)["glc10_lulc"]

    csv_path = Path(cfg.lulc_csv)
    df = pd.read_csv(csv_path)
    assert {"glc10_lc", "modis_lc", fid}.issubset(df.columns), (
        f"{csv_path} missing required columns; got {list(df.columns)}"
    )

    # Preserve the raw-GLC10 extract once for provenance.
    raw_backup = csv_path.with_name(csv_path.stem + "_rawglc10.csv")
    if not raw_backup.exists():
        shutil.copy2(csv_path, raw_backup)
        print(f"Backed up raw-GLC10 CSV -> {raw_backup}")

    # Write the curated cropland code into glc10_lc for cohort sites only.
    mask = df[fid].isin(cohort.index)
    df.loc[mask, "glc10_lc"] = df.loc[mask, fid].map(cohort).astype(float)
    n_crop = int((df.loc[mask, "glc10_lc"] == 10).sum())
    print(f"Cohort sites: {mask.sum()}  cropland-gated (glc10_lc==10): {n_crop}")

    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Example 6 landcover_crop.csv")
    parser.add_argument("--config", default=None, help="TOML config (default: annual2yr POR)")
    args = parser.parse_args()
    main(config_path=args.config)

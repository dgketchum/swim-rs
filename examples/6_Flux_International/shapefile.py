"""Build publication-track shapefiles for Example 6.

Provenance chain:
    flux_crop_ag_96_150m.shp  (96 cropland flux sites, enriched with flux/LULC metadata)
    ├── flux_crop_pub_71_150m.shp  (legacy 71-site POR cohort)
    └── flux_crop_pub_66_150m.shp  (66-site POR publication cohort, current)

The 66-site cohort is the 96-site cropland pool minus three documented,
criterion-based exclusions:
  1. 21 sites lacking post-2013 flux tower data (the minimum required for the
     OLI-era LS Ensemble and Triple ETf POR experiments, 2013-2025).
  2. 4 sites whose authoritative flux-network IGBP / documented land use is
     natural, unmanaged vegetation rather than agriculture. The GLC10 cropland
     mask (=10) mislabels these as cropland, but they are not irrigated
     croplands and do not belong in a cropland-irrigation study. The flux
     IGBP is the ground-truth discriminator: the satellite mask and the IGBP
     *vegetation class* alone cannot separate farms from wild land (AU-RDF is
     IGBP=WSA yet is the "Red Dirt Melon Farm"; AU-Nim is IGBP=GRA yet is
     natural sub-alpine grassland), so the exclusion is keyed on the
     documented land use, not the satellite product that erred.
  3. 5 sites whose documented land use is non-cropland (woody plantation,
     restored prairie, managed pasture, desert) — both LC masks agree, or the
     flux metadata is unambiguous. Same inclusion criterion as (2): only
     cropland flux sites belong in the cropland-irrigation cohort.

These are *cohort* exclusions only. Irrigation status is decided solely by the
internal water-balance classifier (no per-site irrigation override). The
curated `glc10_lulc=10` flag carried by every retained site is a cropland
include/exclude determination, not an irrigation flag.

TOML references:
    - 6_Flux_International_LSEnsemble_POR_annual2yr.toml -> 66-site (current)
    - 6_Flux_International_LSEnsemble_POR.toml           -> 71-site (legacy)
    - 6_Flux_International_TripleETf_POR.toml            -> 71-site (legacy)

Usage:
    python shapefile.py [--gis-dir PATH]
"""

import argparse
from pathlib import Path

import geopandas as gpd

# 21 sites excluded from the 96-site cropland cohort because they lack
# any post-2013 flux tower data (required by the POR calibration window).
EXCLUDED_NO_POST2013_FLUX = {
    "CA-MA1",
    "CA-MA2",
    "CA-MA3",
    "CH-Oe1",
    "CN-Cng",
    "DE-Seh",
    "FI-Jok",
    "IT-PT1",
    "US-ARM",
    "US-Bo1",
    "US-Bo2",
    "US-Br1",
    "US-Br3",
    "US-Dia",
    "US-Dk1",
    "US-KS2",
    "US-Lin",
    "US-Pon",
    "US-SFP",
    "US-SP2",
    "US-Wi6",
}

# 4 sites the GLC10 cropland mask (=10) labels cropland but whose authoritative
# flux-network IGBP / documented land use is natural, unmanaged vegetation —
# not agriculture. Excluded from the cropland-irrigation cohort. (ES-LJu was
# additionally mis-flagged as irrigated 10/13 years by the classifier; the
# other three were already de-flagged but are excluded under the same rule for
# cohort consistency.)
EXCLUDED_NATURAL_LANDCOVER = {
    "ES-LJu",  # IGBP=OSH  Mediterranean karst open shrubland (Llano de los Juanes)
    "AU-Nim",  # IGBP=GRA  natural sub-alpine grassland (Nimmo)
    "US-Los",  # IGBP=WET  natural fen (Lost Creek)
    "CZ-wet",  # IGBP=WET  natural wetland (Trebon)
}

# 5 further sites dropped from the cohort because their documented land use is
# non-cropland (not annual agriculture). Identified during the 71-cohort gate
# audit: when the cropland gate is corrected to honor the curated cropland
# determination, these are the sites that remain genuinely non-cropland under
# both LC masks / unambiguous flux metadata. Same inclusion rule as
# EXCLUDED_NATURAL_LANDCOVER — a cohort exclusion, never an irrigation override.
EXCLUDED_NONCROP_LANDCOVER = {
    "US-Wpp",  # Willamette Poplar woody plantation (DBF); GLC10/MODIS non-crop
    "US-KL3",  # Lux Arbor Reserve restored prairie (GRA); GLC10=30
    "CH-Cha",  # Chamau managed grassland/pasture (GRA); GLC10=30
    "NZ-BFm",  # intensive dairy pasture; GLC10=20
    "US-UTV",  # UFLUX Desert View desert; both masks non-crop (GLC10=90 Barren)
}


def build_pub71(gis_dir: Path) -> Path:
    """Filter crop96 to the 71-site POR publication cohort."""
    crop96_path = gis_dir / "flux_crop_ag_96_150m.shp"
    out_path = gis_dir / "flux_crop_pub_71_150m.shp"

    gdf = gpd.read_file(crop96_path, engine="fiona")
    assert "sid" in gdf.columns, f"Expected 'sid' column, got {list(gdf.columns)}"

    excluded = EXCLUDED_NO_POST2013_FLUX | EXCLUDED_NATURAL_LANDCOVER
    pub = gdf[~gdf["sid"].isin(excluded)].copy()

    dropped = set(gdf["sid"]) - set(pub["sid"])
    assert dropped == excluded, (
        f"Exclusion mismatch: expected {len(excluded)} dropped, got {len(dropped)}"
    )
    assert len(pub) == 71, f"Expected 71 sites, got {len(pub)}"

    pub.to_file(out_path, engine="fiona")
    print(f"Wrote {len(pub)} sites to {out_path}")
    return out_path


def build_pub66(gis_dir: Path) -> Path:
    """Filter crop96 to the current 66-site POR publication cohort.

    The 71-site cohort minus the 5 EXCLUDED_NONCROP_LANDCOVER sites.
    """
    crop96_path = gis_dir / "flux_crop_ag_96_150m.shp"
    out_path = gis_dir / "flux_crop_pub_66_150m.shp"

    gdf = gpd.read_file(crop96_path, engine="fiona")
    assert "sid" in gdf.columns, f"Expected 'sid' column, got {list(gdf.columns)}"

    excluded = EXCLUDED_NO_POST2013_FLUX | EXCLUDED_NATURAL_LANDCOVER | EXCLUDED_NONCROP_LANDCOVER
    pub = gdf[~gdf["sid"].isin(excluded)].copy()

    dropped = set(gdf["sid"]) - set(pub["sid"])
    assert dropped == excluded, (
        f"Exclusion mismatch: expected {len(excluded)} dropped, got {len(dropped)}"
    )
    assert len(pub) == 66, f"Expected 66 sites, got {len(pub)}"

    pub.to_file(out_path, engine="fiona")
    print(f"Wrote {len(pub)} sites to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Example 6 publication shapefiles")
    default_gis = str(Path(__file__).resolve().parent / "data" / "gis")
    parser.add_argument(
        "--gis-dir",
        type=str,
        default=default_gis,
        help="GIS directory containing flux_crop_ag_96_150m.shp (default: in-repo data/gis/)",
    )
    args = parser.parse_args()
    build_pub71(Path(args.gis_dir))  # legacy experiments
    build_pub66(Path(args.gis_dir))  # current POR cohort

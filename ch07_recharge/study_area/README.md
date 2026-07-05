# Study Area — Oued Sikkak Watershed (nested under ch07_recharge; also used by Chapters 4 and 5)

## Data caveats, stated explicitly
- The shapefiles (`bassin_de_sikkak.shp`, `cours_eaux.shp`) were supplied as
  `.shp` geometry only, without their usual `.shx`/`.dbf`/`.prj` companions.
  Geometry reads correctly (pyshp reconstructs the index from `.shp` alone),
  but no attribute table exists, and the CRS (UTM Zone 30N, EPSG:32630) was
  confirmed only by comparing coordinate ranges against the rest of this
  project, not from a `.prj` file.
- **Basin area discrepancy:** the digitized boundary encloses ~458 km2,
  roughly double the 218 km2 figure (Bouanani et al. 2013) used for the
  Chapter 7 water-balance unit conversions. Most likely reflects two
  different delineation outlets (full basin vs. a gauged sub-basin), not
  confirmed against the original source. Both figures are reported, not
  silently reconciled.

## Contents
- `data/raw/bassin_de_sikkak.shp` — watershed boundary (geometry only)
- `data/raw/cours_eaux.shp` — stream network, 603 segments (geometry only)
- `scripts/01_offline_map.py` — matplotlib-only map (no internet needed)
- `scripts/02_colab_basemap.py` — same map with a real satellite basemap
  (contextily + Esri World Imagery, needs Colab's internet access)
- `notebooks/study_area_map.ipynb` — Colab notebook, reads all layers
  directly from this repository

## How to run
Open `notebooks/study_area_map.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.

# ============================================================
# Sikkak watershed study area map - WITH real basemap tiles
# RUN ON GOOGLE COLAB (needs internet access for basemap tiles)
# ============================================================
!pip install pyshp shapely contextily -q

import shapefile
from shapely.geometry import shape
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import contextily as cx

BASE = "https://raw.githubusercontent.com/Dr-LAOUFIAbdessalam/ai-hydrogeologists/main/"

# --- Download shapefiles (all 2 required parts: .shp geometry only, no .dbf/.shx available) ---
import urllib.request, io
def load_shp_geometry_only(url):
    data = urllib.request.urlopen(url).read()
    # pyshp can read shape geometry from an in-memory .shp stream alone
    sf = shapefile.Reader(shp=io.BytesIO(data))
    return sf

basin_sf = load_shp_geometry_only(BASE + "study_area/data/raw/bassin_de_sikkak.shp")
basin_geom = shape(basin_sf.shape(0).__geo_interface__)
basin_area_km2 = basin_geom.area / 1e6

streams_sf = load_shp_geometry_only(BASE + "study_area/data/raw/cours_eaux.shp")
stream_geoms = [shape(s.__geo_interface__) for s in streams_sf.shapes()]

print(f"Digitized basin area: {basin_area_km2:.1f} km2 "
      f"(literature sub-basin value used in Chapter 7: 218 km2 - see note there)")

water_points = pd.read_csv(BASE + "ch07_recharge/data/raw/points_d_eau_sikkak.csv")
water_points["type"] = water_points["Source"].apply(lambda s: "Spring" if s.startswith("Ain") else "Well (extraction)")
heads = pd.read_csv(BASE + "ch04_groundwater_level/data/raw/heads_clean.csv")
hydrochem = pd.read_csv(BASE + "ch05_quality_contamination/data/raw/hennaya_hydrochem_tidy.csv")

# =====================================================================
# Figure with real basemap (Esri World Imagery via contextily)
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 11))

if basin_geom.geom_type == "Polygon":
    xs, ys = basin_geom.exterior.xy
    ax.plot(xs, ys, color="yellow", linewidth=2,
            label=f"Sikkak basin outline ({basin_area_km2:.0f} km2, this shapefile)")
else:
    for part in basin_geom.geoms:
        xs, ys = part.exterior.xy
        ax.plot(xs, ys, color="yellow", linewidth=2)

for geom in stream_geoms:
    parts = [geom] if geom.geom_type == "LineString" else geom.geoms
    for part in parts:
        xs, ys = part.xy
        ax.plot(xs, ys, color="cyan", linewidth=0.7, alpha=0.8)
ax.plot([], [], color="cyan", linewidth=1, label="Stream network")

ax.scatter(heads["x"], heads["y"], c="white", s=8, alpha=0.6, edgecolor="grey",
           label=f"Hennaya piezometers (n={len(heads)})")
hc_unique = hydrochem.drop_duplicates("point")
ax.scatter(hc_unique["x"], hc_unique["y"], c="darkorange", s=45, marker="D",
           edgecolor="k", linewidth=0.6, label=f"Hydrochemistry points (n={len(hc_unique)})")
for t, color, marker in [("Spring", "dodgerblue", "^"), ("Well (extraction)", "red", "s")]:
    sub = water_points[water_points.type == t]
    ax.scatter(sub["x"], sub["y"], c=color, s=70, marker=marker,
               edgecolor="k", linewidth=0.6, label=f"{t} (n={len(sub)})")

ax.set_aspect("equal")
ax.set_title("Oued Sikkak watershed - study area (all chapters)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

# Add satellite basemap - source CRS is UTM 30N / EPSG:32630
cx.add_basemap(ax, crs="EPSG:32630", source=cx.providers.Esri.WorldImagery)

plt.tight_layout()
plt.savefig("sikkak_watershed_basemap.png", dpi=150)
plt.show()
print("Saved: sikkak_watershed_basemap.png")

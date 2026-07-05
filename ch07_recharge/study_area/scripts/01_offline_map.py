"""
Sikkak watershed context map: basin boundary, stream network, and all
water points used across Chapters 2, 4, 5, and 7 (Hennaya piezometers,
Hennaya hydrochemistry points, and the 22-point spring/well inventory).

NOTE ON BASIN AREA DISCREPANCY: the digitized boundary in
bassin_de_sikkak.shp encloses ~458 km2, roughly double the 218 km2 figure
from Bouanani et al. (2013) used for the Chapter 7 water-balance
conversions. This likely reflects two different delineation points (full
basin to an unknown outlet vs. a gauged sub-basin upstream of a specific
hydrometric station) and is flagged explicitly rather than silently
resolved. Both figures are shown on the map/labelled in the legend.
"""
import shapefile
from shapely.geometry import shape
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# --- Load basin boundary and stream network (geometry only, no .dbf) ---
basin_sf = shapefile.Reader("/mnt/user-data/uploads/bassin_de_sikkak.shp")
basin_geom = shape(basin_sf.shape(0).__geo_interface__)
basin_area_km2 = basin_geom.area / 1e6

streams_sf = shapefile.Reader("/mnt/user-data/uploads/cours_eaux.shp")
stream_geoms = [shape(s.__geo_interface__) for s in streams_sf.shapes()]

print(f"Digitized basin area: {basin_area_km2:.1f} km2")
print(f"Literature value (Bouanani et al. 2013): 218 km2")
print(f"Number of stream segments: {len(stream_geoms)}")

# --- Load point datasets already used elsewhere in the book ---
water_points = pd.read_csv("/home/claude/hennaya/points_d_eau_sikkak.csv")
water_points["type"] = water_points["Source"].apply(
    lambda s: "Spring" if s.startswith("Ain") else "Well (extraction)")

heads = pd.read_csv("/home/claude/hennaya/heads_clean.csv")
hydrochem = pd.read_csv("/home/claude/hennaya/hennaya_hydrochem_tidy.csv")

# =====================================================================
# Figure
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 11))

# Basin boundary
if basin_geom.geom_type == "Polygon":
    xs, ys = basin_geom.exterior.xy
    ax.plot(xs, ys, color="black", linewidth=1.5, label=f"Sikkak basin outline (this shapefile, {basin_area_km2:.0f} km2)")
else:
    for part in basin_geom.geoms:
        xs, ys = part.exterior.xy
        ax.plot(xs, ys, color="black", linewidth=1.5)

# Stream network
for geom in stream_geoms:
    if geom.geom_type == "LineString":
        xs, ys = geom.xy
        ax.plot(xs, ys, color="steelblue", linewidth=0.6, alpha=0.7)
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            xs, ys = part.xy
            ax.plot(xs, ys, color="steelblue", linewidth=0.6, alpha=0.7)
ax.plot([], [], color="steelblue", linewidth=1, label="Stream network (cours d'eau)")

# Hennaya piezometric wells (Chapter 4), all 3 campaigns pooled
ax.scatter(heads["x"], heads["y"], c="grey", s=8, alpha=0.5,
           label=f"Hennaya piezometers (n={len(heads)}, all campaigns)")

# Hennaya hydrochemistry points (Chapter 5)
hc_unique = hydrochem.drop_duplicates("point")
ax.scatter(hc_unique["x"], hc_unique["y"], c="darkorange", s=40, marker="D",
           edgecolor="k", linewidth=0.5, label=f"Hydrochemistry points (n={len(hc_unique)})")

# Springs and wells inventory (Chapter 7)
for t, color, marker in [("Spring", "blue", "^"), ("Well (extraction)", "red", "s")]:
    sub = water_points[water_points.type == t]
    ax.scatter(sub["x"], sub["y"], c=color, s=60, marker=marker,
               edgecolor="k", linewidth=0.5, label=f"{t} (n={len(sub)})")

ax.set_aspect("equal")
ax.set_xlabel("Easting (m, UTM Zone 30N)")
ax.set_ylabel("Northing (m, UTM Zone 30N)")
ax.set_title("Oued Sikkak watershed: study area and all sampling networks\n"
             "(basin area discrepancy with literature flagged - see notebook text)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

# Scale bar (simple, approximate)
scale_len_m = 5000
x0, y0 = ax.get_xlim()[0] + 0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]), ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0])
ax.plot([x0, x0+scale_len_m], [y0, y0], color="black", linewidth=2)
ax.text(x0+scale_len_m/2, y0+300, "5 km", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/sikkak_watershed_context_map.png", dpi=150)
plt.close()
print("\nSaved: sikkak_watershed_context_map.png")

# ============================================================
# Chapter 2 worked example - Step 3b (RUN THIS CELL ON GOOGLE COLAB)
# Fetch soil texture (clay/sand/silt, 0-5 cm) from SoilGrids v2.0
# as a locally-varying substitute for the (too coarse) global
# lithological map (GLiM), following ISRIC's 5 requests/minute
# fair-use policy.
# ============================================================

import pandas as pd
import numpy as np
import requests
import time

# --- 1. Load well coordinates (lon/lat already computed in Step 3a) ---
wells = pd.read_csv("wells_dem_landcover.csv")  # already has lon, lat

# --- 2. Reduce API calls: round to the ~250 m SoilGrids grid and
#        query each unique cell only once ---
# 250 m in degrees latitude is fixed (~0.00225); in longitude it varies
# with cos(latitude), but at this scale a fixed rounding is an
# acceptable approximation for deduplication purposes only.
GRID_DEG = 0.0025
wells["grid_lon"] = (wells["lon"] / GRID_DEG).round() * GRID_DEG
wells["grid_lat"] = (wells["lat"] / GRID_DEG).round() * GRID_DEG

unique_cells = wells[["grid_lon", "grid_lat"]].drop_duplicates().reset_index(drop=True)
print(f"{len(wells)} wells -> {len(unique_cells)} unique SoilGrids cells "
      f"({len(unique_cells)/len(wells)*100:.0f}% of original)")

# --- 3. Query SoilGrids v2.0 REST API (5 requests/minute fair use) ---
def fetch_soilgrids_point(lon, lat, properties=("clay", "sand", "silt"), depth="0-5cm"):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {"lon": lon, "lat": lat, "depth": depth, "value": "mean"}
    # requests library needs repeated keys for multiple 'property' values
    query = "&".join([f"property={p}" for p in properties])
    r = requests.get(f"{url}?{query}", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = {}
    for layer in data["properties"]["layers"]:
        name = layer["name"]
        depth_data = layer["depths"][0]["values"]["mean"]
        # SoilGrids returns values x10 (e.g. clay in g/kg *10) - convert to %
        out[f"{name}_pct"] = depth_data / 10.0
    return out

results = []
for i, row in unique_cells.iterrows():
    try:
        vals = fetch_soilgrids_point(row["grid_lon"], row["grid_lat"])
    except Exception as e:
        print(f"Failed at cell {i} ({row['grid_lon']},{row['grid_lat']}): {e}")
        vals = {"clay_pct": np.nan, "sand_pct": np.nan, "silt_pct": np.nan}
    vals.update({"grid_lon": row["grid_lon"], "grid_lat": row["grid_lat"]})
    results.append(vals)
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(unique_cells)} cells done, pausing 65s (fair-use limit)...")
        time.sleep(65)
    else:
        time.sleep(2)

soil_df = pd.DataFrame(results)
wells = wells.merge(soil_df, on=["grid_lon", "grid_lat"], how="left")

# --- 4. Simple USDA-style texture flag from clay/sand thresholds ---
def texture_class(row):
    if pd.isna(row["clay_pct"]):
        return "Unknown"
    if row["sand_pct"] >= 70 and row["clay_pct"] < 15:
        return "Sandy"
    if row["clay_pct"] >= 35:
        return "Clayey"
    return "Loamy"

wells["texture_class"] = wells.apply(texture_class, axis=1)

print("\nTexture summary:")
print(wells[["clay_pct", "sand_pct", "silt_pct"]].describe())
print(wells["texture_class"].value_counts())

wells.drop(columns=["grid_lon", "grid_lat"]).to_csv("wells_dem_landcover_soil.csv", index=False)
print("\nSaved: wells_dem_landcover_soil.csv")

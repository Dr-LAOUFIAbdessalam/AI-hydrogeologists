# Chapter 3 — Core ML Methods: Spatial Comparison Worked Example

## Scope note
The original book outline planned a temporal comparison (Random Forest /
XGBoost / LSTM) on a continuous monthly groundwater level series
(Section 3.6). The Hennaya dataset consists of three independent
piezometric snapshots (1981, 2012, 2022), not a continuous series, and
well field labels are not physically consistent across campaigns
(established in Chapter 2). A temporal model comparison is therefore not
meaningful here (see Chapter 4 for the correct temporal treatment of this
dataset).

This chapter instead demonstrates the same core methodological lesson of
Sections 3.5.1–3.5.2 — why naive random splits fail on spatially
autocorrelated hydrogeological data — using a spatial prediction task the
data actually support: predicting **depth to water table** from
physiographic covariates (elevation, soil texture, land cover, position),
comparing Random Forest against Linear Regression under two
cross-validation schemes.

## Why depth-to-water, not absolute head
Absolute piezometric head (HEAD) correlates almost trivially with surface
elevation (head ≈ elevation − a comparatively small depth), so any model
predicting HEAD from elevation would show spuriously high accuracy without
learning anything hydrogeologically meaningful. Depth to water is a harder,
more informative target.

## Key result: naive random split vs. spatially-honest split
| Model | Random CV (naive) | Spatial CV (honest, k-means blocks) |
|---|---|---|
| Linear Regression | R² = 0.66 | **R² = −0.43** |
| Random Forest | R² = 0.70 | **R² = −0.40** |

Both models look reasonably good under a random split, but **collapse
into negative R²** — worse than predicting the mean — under spatial
cross-validation, where test wells are never immediate neighbours of
training wells. This is a real-data confirmation, not a textbook
illustration, of Section 3.5.1's warning: nearby wells share enough
spatial autocorrelation that a random split lets a model "cheat" by
interpolating between near-duplicate neighbours. The spatially honest
estimate is the only trustworthy one for judging real-world
transferability to unmonitored locations.

## A noted data limitation
A few negative depth-to-water values appear in the dataset, an artefact
of the 30 m SRTM DEM's limited vertical accuracy near small local
depressions — flagged as a known limitation, not corrected, in keeping
with this book's approach of reporting data issues transparently rather
than silently smoothing them away.

## Contents
- `data/raw/hennaya_heads_ml_ready.csv` — the Chapter 2 output table, reused directly
- `scripts/01_spatial_ml_comparison.py` — standalone Python script
- `notebooks/ch03_spatial_ml_comparison.ipynb` — Colab notebook, reads
  data directly from this repository (no manual upload needed)

## How to run
Open `notebooks/ch03_spatial_ml_comparison.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.

# Chapter 7 — Recharge Estimation, Oued Sikkak Watershed

## Basin context (verified)
Oued Sikkak watershed, 218 km2, NW Algeria, includes the Hennaya plain
studied in Chapters 2-5. Mean annual precipitation 480.5 mm (206-800 mm
range), mean annual runoff ~104 mm (Bouanani et al., 2013, Journal of
Water Science).

## Scope note on GRACE-FO
GRACE-FO's native footprint spans several hundred km — far coarser than
this 218 km2 basin (same resolution mismatch as GLiM in Chapter 5). We use
the actual Zenata climate station record instead (45 years monthly,
1980/81-2024/25; the same station used in the published academic
literature for this basin).

## Method
Monthly climatic water balance using Thornthwaite PET, surplus = max(P-PET, 0),
as a first-order, UNCALIBRATED recharge+runoff proxy. Random Forest vs
Linear Regression comparison predicting the surplus from lagged
precipitation/temperature features, evaluated with year-blocked temporal
cross-validation (not random shuffling).

## Key results
- **Computed water balance surplus: 102.8 mm/yr — closely matches the
  independently published mean annual runoff (104 mm/yr)**, a striking
  plausibility check for a completely uncalibrated model. Not proof of
  accuracy: this proxy conflates runoff and recharge, and the closeness
  may partly be coincidental given the simple bucket approach.
- The random-vs-temporal CV gap here is much MILDER than the spatial CV
  collapse in Chapter 3 (R2 ~0.70->0.65 here vs ~0.70->-0.40 there) —
  the degree of optimistic bias from naive splitting depends on the
  correlation structure of the specific problem, not a universal constant.
- Linear Regression slightly outperforms Random Forest — model complexity
  is not automatically an advantage.
- Feature importance: 3-month cumulative precipitation dominates (0.46),
  consistent with a delayed recharge response to antecedent rainfall.

## Partial validation: field water-point inventory (22 locations)
A field inventory of 8 natural springs and 14 wells across the Sikkak
watershed gives a genuine, if partial, real-world check:
- **Springs (13.0 mm/yr) < water balance surplus (102.8 mm/yr)** — the
  expected direction: natural baseflow is only one discharge component of
  the total surplus (which also includes direct runoff and water
  intercepted by pumping upstream of any spring). Consistent, not proof.
- **Wells (35.5 mm/yr) measure human extraction, not recharge** — using
  them to "validate" the recharge number would be a category error. They
  instead inform a separate, practically important question: the
  naive exploitation ratio (~34%, a LOWER bound since these 14 wells are a
  partial inventory, not a full basin census).

## Contents
- `data/raw/Temperatures_mensuelles.xlsx` — Zenata monthly temperature
  (precipitation reused from `ch02_data_preparation/data/raw/`)
- `data/raw/sikkak_water_balance_monthly.csv` — computed water balance series
- `scripts/01_recharge_water_balance.py` — standalone Python script
- `notebooks/ch07_sikkak_recharge.ipynb` — Colab notebook, reads directly
  from this repository

## How to run
Open `notebooks/ch07_sikkak_recharge.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.

"""
Chapter 7 worked example: recharge estimation for the Oued Sikkak watershed
(218 km2, NW Algeria, includes the Hennaya plain) using a monthly climatic
water balance (Thornthwaite PET) as a recharge PROXY, then comparing Random
Forest against linear regression to predict that proxy from lagged climatic
inputs, with a properly time-ordered (not random) cross-validation.

Basin facts used for context (Bouanani et al., 2013, Journal of Water
Science; verified via web search): area 218 km2, mean annual precipitation
480.5 mm (206-800 mm range), mean annual runoff ~104 mm. Runoff is not
recharge, but provides an order-of-magnitude plausibility check on the
water balance surplus computed here.

NOTE ON SCOPE: GRACE-FO terrestrial water storage anomalies, mentioned in
the book outline, have a native footprint of several hundred km - far
larger than the 218 km2 Sikkak basin (the same resolution mismatch problem
identified for GLiM in Chapter 5). GRACE is therefore not used here; the
climatic water balance from the actual Zenata station (used in the
academic literature for this exact basin) is the appropriate real-data
approach at this scale.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

precip = pd.read_excel("/mnt/user-data/uploads/Précipitations_mensuelles.xlsx")
temp = pd.read_excel("/mnt/user-data/uploads/Températures_mensuelles.xlsx")

months_fr = ["Sep","Oct","Nov","Dec","Jan","Fev","Mars","Avr","Mai","Juin","Juill","Aout"]
months_en = ["Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
month_num = {  # calendar month number for each column, for day-length correction
    "Sep":9,"Oct":10,"Nov":11,"Dec":12,"Jan":1,"Fev":2,"Feb":2,"Mars":3,"Mar":3,
    "Avr":4,"Apr":4,"Mai":5,"May":5,"Juin":6,"Jun":6,"Juill":7,"Jul":7,"Aout":8,"Aug":8,
}

p_long = precip.melt(id_vars="Année_hydro", value_vars=months_fr,
                      var_name="month_fr", value_name="P_mm")
t_long = temp.melt(id_vars="Année_hydro", value_vars=months_en,
                    var_name="month_en", value_name="T_C")
p_long["month_num"] = p_long["month_fr"].map(month_num)
t_long["month_num"] = t_long["month_en"].map(month_num)

df = p_long.merge(t_long, on=["Année_hydro", "month_num"])
df["hydro_year_start"] = df["Année_hydro"].str.split("/").str[0].astype(int)

# Build a proper chronological date (hydrological year starts in September)
def to_date(row):
    year = row["hydro_year_start"] if row["month_num"] >= 9 else row["hydro_year_start"] + 1
    return pd.Timestamp(year=year, month=row["month_num"], day=1)
df["date"] = df.apply(to_date, axis=1)
df = df.sort_values("date").reset_index(drop=True)

# Drop the incomplete first hydrological year (1980/81 has only 4 months of
# precipitation on record, as established in Chapter 2/4)
df = df[df["hydro_year_start"] >= 1981].reset_index(drop=True)
print(f"n = {len(df)} months, {df['date'].min().date()} to {df['date'].max().date()}")

# =====================================================================
# Thornthwaite potential evapotranspiration (standard formulation)
# Latitude of Zenata station (~34.98 N, established in Chapter 2)
# =====================================================================
LAT = 34.98

def day_length_factor(month, lat=LAT):
    """Simplified monthly daylight correction factor (Thornthwaite tables,
    approximated via solar declination), unitless multiplier on PET."""
    lat_rad = np.radians(lat)
    day_of_year = {1:15,2:46,3:74,4:105,5:135,6:162,7:198,8:228,9:259,10:289,11:320,12:350}[month]
    decl = 0.4093 * np.sin(2*np.pi*day_of_year/365 - 1.405)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))
    N = 24/np.pi * ws  # max daylight hours
    days_in_month = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}[month]
    return (N/12) * (days_in_month/30)

annual = df.groupby("hydro_year_start")["T_C"].apply(
    lambda t: np.sum((np.maximum(t, 0)/5)**1.514))
df["I_annual"] = df["hydro_year_start"].map(annual)  # heat index per hydro year
a = (6.75e-7 * df["I_annual"]**3) - (7.71e-5 * df["I_annual"]**2) + (1.792e-2 * df["I_annual"]) + 0.49239

T_pos = np.maximum(df["T_C"], 0)
pet_unadjusted = 16 * (10 * T_pos / df["I_annual"])**a
df["PET_mm"] = pet_unadjusted * df["month_num"].apply(day_length_factor)
df.loc[df["T_C"] <= 0, "PET_mm"] = 0

# =====================================================================
# Monthly water balance: recharge proxy = max(P - PET, 0)
# A simple bucket approach (no soil storage carryover) - a first-order
# proxy, not a calibrated recharge model. Stated as such throughout.
# =====================================================================
df["WB_surplus_mm"] = np.maximum(df["P_mm"] - df["PET_mm"], 0)

annual_summary = df.groupby("hydro_year_start").agg(
    P_annual=("P_mm", "sum"), PET_annual=("PET_mm", "sum"),
    surplus_annual=("WB_surplus_mm", "sum"))
print("\nAnnual water balance summary (mean over record):")
print(annual_summary.mean().round(1))
print(f"\nPublished mean annual runoff for Sikkak basin (Bouanani et al. 2013): 104 mm")
print(f"Water balance surplus here is a recharge+runoff PROXY, not calibrated "
      f"recharge; order-of-magnitude comparison only.")

df.to_csv("/home/claude/hennaya/sikkak_water_balance_monthly.csv", index=False)

# =====================================================================
# Feature engineering: lag structure (Section 2.4.1 / 4.2.1 style)
# =====================================================================
for lag in [1, 2, 3, 6, 12]:
    df[f"P_lag{lag}"] = df["P_mm"].shift(lag)
    df[f"T_lag{lag}"] = df["T_C"].shift(lag)
df["P_cumul_3mo"] = df["P_mm"].rolling(3).sum()
df["P_cumul_6mo"] = df["P_mm"].rolling(6).sum()
df["month_sin"] = np.sin(2*np.pi*df["month_num"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month_num"]/12)

model_df = df.dropna().reset_index(drop=True)
feature_cols = [c for c in model_df.columns if c.startswith(("P_lag","T_lag","P_cumul","month_sin","month_cos"))]
X = model_df[feature_cols].values
y = model_df["WB_surplus_mm"].values
print(f"\nModelling dataset: n={len(model_df)} months, {len(feature_cols)} features")

# =====================================================================
# Temporal cross-validation (Section 3.5.3 / 3.5.1): blocked by year,
# NOT randomly shuffled, since months are strongly autocorrelated
# =====================================================================
years = model_df["hydro_year_start"].values
unique_years = np.sort(np.unique(years))
n_folds = 6
year_folds = np.array_split(unique_years, n_folds)

def temporal_cv(model, X, y, years, year_folds):
    rmses, r2s = [], []
    for fold_years in year_folds:
        test_idx = np.isin(years, fold_years)
        train_idx = ~test_idx
        if test_idx.sum() < 5:
            continue
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        r2s.append(r2_score(y[test_idx], pred))
    return np.array(rmses), np.array(r2s)

def random_cv(model, X, y, n_splits=6, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_splits)
    rmses, r2s = [], []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        r2s.append(r2_score(y[test_idx], pred))
    return np.array(rmses), np.array(r2s)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
}

print("\n" + "="*70)
print("RESULTS: naive random split vs year-blocked temporal split")
print("="*70)
for name, model in models.items():
    rmse_r, r2_r = random_cv(model, X, y)
    rmse_t, r2_t = temporal_cv(model, X, y, years, year_folds)
    print(f"\n{name}:")
    print(f"  Random CV   : RMSE = {rmse_r.mean():.1f} +/- {rmse_r.std():.1f} mm | R2 = {r2_r.mean():.2f}")
    print(f"  Temporal CV : RMSE = {rmse_t.mean():.1f} +/- {rmse_t.std():.1f} mm | R2 = {r2_t.mean():.2f}")

# Feature importance from the full-data RF fit (for interpretation only)
rf_full = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42).fit(X, y)
importances = pd.Series(rf_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances (Random Forest, full fit):")
print(importances.head(8).round(3))

# =====================================================================
# 5. Partial validation: springs (natural baseflow) vs wells (extraction)
# =====================================================================
print("\n" + "="*70)
print("PARTIAL VALIDATION: field water-point inventory (22 locations)")
print("="*70)

wp = pd.read_csv("/home/claude/hennaya/points_d_eau_sikkak.csv")
wp["type"] = wp["Source"].apply(lambda s: "spring" if s.startswith("Ain") else "well")

BASIN_AREA_KM2 = 218
SECONDS_PER_YEAR = 86400 * 365.25

def flow_to_mm_per_year(Q_Ls_sum, area_km2=BASIN_AREA_KM2):
    Q_m3s = Q_Ls_sum / 1000
    annual_m3 = Q_m3s * SECONDS_PER_YEAR
    return annual_m3 / (area_km2 * 1e6) * 1000

spring_sum = wp.loc[wp.type == "spring", "Q (l/s)"].sum()
well_sum = wp.loc[wp.type == "well", "Q (l/s)"].sum()
spring_mm = flow_to_mm_per_year(spring_sum)
well_mm = flow_to_mm_per_year(well_sum)

print(f"Springs (n={sum(wp.type=='spring')}): {spring_sum:.1f} L/s -> "
      f"{spring_mm:.1f} mm/yr equivalent over {BASIN_AREA_KM2} km2 (natural baseflow)")
print(f"Wells (n={sum(wp.type=='well')}): {well_sum:.1f} L/s -> "
      f"{well_mm:.1f} mm/yr equivalent over {BASIN_AREA_KM2} km2 (measured extraction)")
print(f"\nWater balance surplus estimated above: {annual_summary['surplus_annual'].mean():.1f} mm/yr")
print(f"\nInterpretation:")
print(f"  Springs ({spring_mm:.1f} mm/yr) < surplus ({annual_summary['surplus_annual'].mean():.1f} mm/yr): "
      f"{'CONSISTENT (baseflow is only one component of the surplus)' if spring_mm < annual_summary['surplus_annual'].mean() else 'INCONSISTENT - needs review'}")
print(f"  Wells measure human EXTRACTION, not natural recharge - not a valid")
print(f"  direct check on the recharge estimate, but informative for a")
print(f"  separate exploitation-ratio discussion (partial inventory only,")
print(f"  22 points do not cover the full basin's abstraction).")
print(f"  Naive exploitation ratio (partial wells / surplus): "
      f"{100*well_mm/annual_summary['surplus_annual'].mean():.0f}% "
      f"(likely a LOWER bound on true basin-wide exploitation)")

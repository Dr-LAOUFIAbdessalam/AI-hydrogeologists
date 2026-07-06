"""
Chapter 9 worked example: SHAP analysis of the Random Forest surrogate
model built in Chapter 8 (Exercise 2), directly targeting the
multicollinearity trap discovered there: raw feature_importances_ showed
near-zero credit for the calibrated K/S/porosity fields (combined 0.002)
while elevation-related features dominated (0.839), even though an
ablation test proved K/S alone still reach R2=0.944. Does SHAP give a
more informative, less misleading picture without needing a manual
ablation experiment?
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap

full = pd.read_csv("/home/claude/hennaya/modflow_surrogate_dataset.csv")
for col in ["Kx", "Ky", "Kz"]:
    full[f"log_{col}"] = np.log10(full[col].clip(lower=1e-12))

feature_cols = ["X", "Y", "Z", "Top", "Bot", "Thick",
                 "log_Kx", "log_Ky", "log_Kz", "Ss", "Sy", "PorEff", "PorTot", "year"]

train = full[full["year"].isin([1981, 2022])].reset_index(drop=True)
test = full[full["year"] == 2012].reset_index(drop=True)

X_train, y_train = train[feature_cols], train["Head"].values
X_test, y_test = test[feature_cols], test["Head"].values

rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
print(f"Validation R2 (2012, held out): {r2_score(y_test, rf.predict(X_test)):.4f}")

# =====================================================================
# SHAP values on the held-out validation set (TreeExplainer, exact for RF)
# =====================================================================
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
print("\nMean |SHAP value| per feature (global importance):")
print(mean_abs_shap.round(3))

# Group into the same two categories as the Chapter 8 impurity-importance comparison
ks_cols = ["log_Kx", "log_Ky", "log_Kz", "Ss", "Sy", "PorEff", "PorTot"]
elev_cols = ["Top", "Z", "Bot"]
ks_shap = mean_abs_shap[ks_cols].sum()
elev_shap = mean_abs_shap[elev_cols].sum()
total_shap = mean_abs_shap.sum()

print(f"\nSHAP-based K/S/porosity share: {ks_shap/total_shap:.3f} "
      f"(vs. impurity-based feature_importances_: 0.002)")
print(f"SHAP-based elevation share: {elev_shap/total_shap:.3f} "
      f"(vs. impurity-based feature_importances_: 0.839)")

print(f"""
HONEST FINDING - contrary to the initial hypothesis motivating this
chapter: SHAP does NOT resolve the multicollinearity trap either. The
SHAP-based K/S/porosity share (0.010) is nearly as small as the raw
impurity-based importance (0.002), and elevation still dominates
(0.803 by SHAP vs 0.839 by impurity). TreeSHAP's attributions are still
computed from the same underlying tree structure and split choices, so
when two features are highly correlated, SHAP can also fail to fully
disentangle their individual contributions - a documented limitation of
Shapley-value methods under multicollinearity, not unique to impurity-based
importance. The manual ablation test from Chapter 8 (refitting with K/S
alone, reaching R2=0.944) remains the more reliable diagnostic here.
The practical lesson: SHAP is a substantial improvement over raw
feature_importances_ for many purposes (Section 9.3.1), but it is not a
universal fix for correlated features, and domain-knowledge-driven
ablation tests remain an essential complementary tool (Section 9.6).
""")

# =====================================================================
# Local explanation: one high-conductivity cell, one low-conductivity cell
# =====================================================================
idx_high_k = X_test["log_Kx"].idxmax()
idx_low_k = X_test["log_Kx"].idxmin()

expected_value = explainer.expected_value
if hasattr(expected_value, "__len__"):
    expected_value = expected_value[0]

for label, idx in [("Highest-conductivity cell", idx_high_k), ("Lowest-conductivity cell", idx_low_k)]:
    print(f"\n--- {label} (row {idx}) ---")
    row_shap = pd.Series(shap_values[idx], index=feature_cols).sort_values(key=abs, ascending=False)
    print(f"Predicted head: {rf.predict(X_test.iloc[[idx]])[0]:.1f} m "
          f"(actual: {y_test[idx]:.1f} m, base value: {expected_value:.1f} m)")
    print("Top SHAP contributions:")
    print(row_shap.head(5).round(2))

np.savez("/home/claude/hennaya/ch9_shap_results.npz",
         feature_cols=feature_cols, shap_values=shap_values,
         mean_abs_shap=mean_abs_shap.values, X_test=X_test.values, y_test=y_test)
print("\nSaved: ch9_shap_results.npz")

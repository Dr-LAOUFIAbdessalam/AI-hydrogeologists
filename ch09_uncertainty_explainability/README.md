# Chapter 9 — SHAP Analysis of the Hennaya Surrogate Model

## Direct continuation of Chapter 8's feature-importance trap
Applies SHAP to the Random Forest surrogate from Chapter 8 (Exercise 2),
which validated at R2=0.966 on the held-out 2012 campaign but whose raw
`feature_importances_` gave the calibrated K/S/porosity fields almost no
credit (0.002 combined) while elevation dominated (0.839) - later shown
by manual ablation to be a multicollinearity artifact (K/S alone reach
R2=0.944).

## Key finding: SHAP does NOT resolve this trap either
Contrary to the hypothesis motivating this chapter, SHAP gives nearly the
same skewed picture: K/S/porosity SHAP share ~0.01 (vs 0.002 by impurity),
elevation ~0.80 (vs 0.839). TreeSHAP's attributions are computed from the
same underlying tree splits, so correlated features can still fail to be
disentangled - a documented limitation of Shapley-value methods under
multicollinearity, not unique to impurity-based importance. The manual
ablation test from Chapter 8 remains the more reliable diagnostic here.

**Practical lesson:** SHAP is a substantial improvement over raw
`feature_importances_` for many purposes (signed, per-observation,
theoretically grounded attributions), but it is not a universal fix for
correlated features. Domain-knowledge-driven ablation experiments remain
an essential complementary tool, not something SHAP makes obsolete.

## Local explanations
Even for the single highest-conductivity cell in the validation set, its
individual SHAP explanation is still dominated by elevation rather than
conductivity - a concrete illustration of why local explanations should
be read alongside a broader understanding of feature correlation, not
taken at face value in isolation.

## Contents
- `scripts/01_shap_analysis.py` — standalone Python script
- `notebooks/ch09_shap_analysis.ipynb` — Colab notebook, rebuilds the
  Chapter 8 surrogate model and dataset directly from the repository

## How to run
Open `notebooks/ch09_shap_analysis.ipynb` via
Google Colab → File → Open notebook → GitHub, then Run all.

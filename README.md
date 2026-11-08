# Divided Network Intrusion Detection notebooks

The original `majorproject(1).ipynb` has been split into focused notebooks:

1. `01_Logistic_Regression.ipynb`
2. `02_Decision_Tree.ipynb`
3. `03_Random_Forest.ipynb`
4. `04_XGBoost.ipynb`
5. `05_Model_Comparison.ipynb`
6. `06_Zero_Day_and_Hybrid_IDS.ipynb`
7. `07_SHAP_Explainability.ipynb`

## Important
- Each notebook repeats the original data-loading and preprocessing cells (0–8) so it can start from the beginning.
- Update `folder_path` in the data-loading cell to the folder containing your CSV files.
- Some advanced/summary cells in the original notebook depend on variables created in other model sections. Run cells top-to-bottom and check variable names if a summary cell raises an error.
- The split preserves the original code rather than silently rewriting its modeling logic.
- `pip install xgboost` and `%pip install shap` cells are kept where relevant. If needed, install dependencies in the active environment first.

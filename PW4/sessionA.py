import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

# Ensure output directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================================
# A2. Clean and Look
# ==========================================
df_raw = pd.read_csv("data/concrete.csv")

n_duplicates = df_raw.duplicated().sum()
n_missing = df_raw.isna().sum().sum()

print("--- A2. DATA CLEANING & SUMMARY ---")
print(f"Duplicates: {n_duplicates}")
print(f"Missing values: {n_missing}")

# Drop duplicate rows
df = df_raw.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["strength"])
y = df["strength"]

print("\n--- DATA STATS (df.describe()) ---")
print(df.describe())

# Scatter plots: strength vs cement and strength vs age
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(df["cement"], df["strength"], alpha=0.5, color="blue")
axes[0].set_xlabel("Cement (kg/m³)")
axes[0].set_ylabel("Strength (MPa)")
axes[0].set_title("Strength vs Cement")

axes[1].scatter(df["age"], df["strength"], alpha=0.5, color="green")
axes[1].set_xlabel("Age (days)")
axes[1].set_ylabel("Strength (MPa)")
axes[1].set_title("Strength vs Age")

plt.tight_layout()
plt.savefig("figures/A2_scatter.png")
plt.close()

# ==========================================
# A3. Train / Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- A3. SPLIT SIZES ---")
print(f"Train set rows: {X_train.shape[0]}, Test set rows: {X_test.shape[0]}")

# ==========================================
# A4. Linear Regression
# ==========================================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

n_train, p = X_train.shape
r2_tr = r2_score(y_train, y_train_pred)
r2_adj_tr = 1 - (1 - r2_tr) * (n_train - 1) / (n_train - p - 1)

rmse_tr = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_te = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_te = mean_absolute_error(y_test, y_test_pred)
r2_te = r2_score(y_test, y_test_pred)

print("\n--- A4. LINEAR REGRESSION RESULTS ---")
print(f"Train RMSE:   {rmse_tr:.3f} MPa")
print(f"Test RMSE:    {rmse_te:.3f} MPa")
print(f"Test MAE:     {mae_te:.3f} MPa")
print(f"Train R²:     {r2_tr:.3f}")
print(f"Train Adj R²: {r2_adj_tr:.3f}")
print(f"Test R²:      {r2_te:.3f}")

eq_terms = [f"{lr.intercept_:.3f}"] + [f"{coef:+.3f}*{col}" for coef, col in zip(lr.coef_, X.columns)]
print("\nFitted Equation:")
print("strength =", " ".join(eq_terms))

# Residual plot (log scale x-axis)
residuals_test = y_test - y_test_pred
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X_test["age"], residuals_test, alpha=0.6, color="purple")
ax.axhline(0, color="red", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("Age (days, log scale)")
ax.set_ylabel("Residuals (True - Predicted, MPa)")
ax.set_title("Test Residuals vs Age")
plt.tight_layout()
plt.savefig("figures/A4_residuals_vs_age.png")
plt.close()

# ==========================================
# A5. Polynomial Regression
# ==========================================
poly_results = []
for deg in [1, 2, 3, 4]:
    poly_pipe = make_pipeline(
        PolynomialFeatures(degree=deg, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    poly_pipe.fit(X_train, y_train)
    
    p_deg = poly_pipe[0].n_output_features_
    y_tr_p = poly_pipe.predict(X_train)
    y_te_p = poly_pipe.predict(X_test)
    
    r2_tr_p = r2_score(y_train, y_tr_p)
    r2_adj_p = 1 - (1 - r2_tr_p) * (n_train - 1) / (n_train - p_deg - 1)
    
    poly_results.append({
        "Degree": deg,
        "p (Features)": p_deg,
        "Train RMSE": round(np.sqrt(mean_squared_error(y_train, y_tr_p)), 3),
        "Test RMSE": round(np.sqrt(mean_squared_error(y_test, y_te_p)), 3),
        "Train R²": round(r2_tr_p, 3),
        "Train Adj R²": round(r2_adj_p, 3),
        "Test R²": round(r2_score(y_test, y_te_p), 3)
    })

poly_df = pd.DataFrame(poly_results)
print("\n--- A5. POLYNOMIAL REGRESSION TABLE ---")
print(poly_df.to_string(index=False))

# ==========================================
# A6. Classification: KNN & Decision Tree
# ==========================================
# Binary target classification for strength >= 40 MPa
y_class = (df["strength"] >= 40).astype(int)

# Stratified train/test split
Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 1. KNN Hyperparameter Sweep
k_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_tr_acc, knn_te_acc = [], []

for k in k_values:
    knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn_pipe.fit(Xc_train, c_train)
    knn_tr_acc.append(knn_pipe.score(Xc_train, c_train))
    knn_te_acc.append(knn_pipe.score(Xc_test, c_test))

# Plot KNN accuracy curves
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_values, knn_tr_acc, label="Train Accuracy", marker="o")
ax.plot(k_values, knn_te_acc, label="Test Accuracy", marker="s")
ax.set_xscale("log")
ax.set_xlabel("K (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("KNN Accuracy vs K")
ax.legend()
plt.tight_layout()
plt.savefig("figures/A6_knn_sweep.png")
plt.close()

# Scaled vs Unscaled comparison for chosen K = 7
chosen_k = 5
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_k))
knn_scaled.fit(Xc_train, c_train)

knn_unscaled = KNeighborsClassifier(n_neighbors=chosen_k)
knn_unscaled.fit(Xc_train, c_train)

print("\n--- A6. KNN SCALING COMPARISON (K=7) ---")
print(f"Scaled Test Accuracy:   {knn_scaled.score(Xc_test, c_test):.4f}")
print(f"Unscaled Test Accuracy: {knn_unscaled.score(Xc_test, c_test):.4f}")

# 2. Decision Tree Hyperparameter Sweep
depths = [1, 2, 3, 4, 5, 6, 8, None]
tree_tr_acc, tree_te_acc, leaves = [], [], []

for d in depths:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0)
    tree.fit(Xc_train, c_train)
    tree_tr_acc.append(tree.score(Xc_train, c_train))
    tree_te_acc.append(tree.score(Xc_test, c_test))
    leaves.append(tree.get_n_leaves())

tree_df = pd.DataFrame({
    "max_depth": [str(d) for d in depths],
    "Leaves": leaves,
    "Train Accuracy": [round(a, 4) for a in tree_tr_acc],
    "Test Accuracy":  [round(a, 4) for a in tree_te_acc]
})

print("\n--- A6. DECISION TREE SWEEP TABLE ---")
print(tree_df.to_string(index=False))

# Export top two levels of chosen Decision Tree (depth = 4)
chosen_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=1, random_state=0)
chosen_tree.fit(Xc_train, c_train)

print("\n--- A6. TREE TOP 2 LEVELS ---")
print(export_text(chosen_tree, feature_names=list(X.columns), max_depth=2))

# ==========================================
# A7. Metrics & Threshold Analysis
# ==========================================
models_dict = {
    "KNN_k7": knn_scaled,
    "Tree_d4": chosen_tree
}

metrics_list = []
for name, m in models_dict.items():
    preds = m.predict(Xc_test)
    cm = confusion_matrix(c_test, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics_list.append({
        "Model": name,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": round(accuracy_score(c_test, preds), 4),
        "Precision": round(precision_score(c_test, preds), 4),
        "Recall": round(recall_score(c_test, preds), 4),
        "F1": round(f1_score(c_test, preds), 4)
    })

metrics_df = pd.DataFrame(metrics_list)
print("\n--- A7. FINAL MODEL METRICS ---")
print(metrics_df.to_string(index=False))

# Save metrics to results/A_metrics.csv
metrics_df.to_csv("results/A_metrics.csv", index=False)

# Threshold Analysis for Decision Tree
probas_tree = chosen_tree.predict_proba(Xc_test)[:, 1]
thresh_list = []
for t in [0.3, 0.5, 0.7, 0.9]:
    preds_t = (probas_tree >= t).astype(int)
    cm_t = confusion_matrix(c_test, preds_t)
    tn, fp, fn, tp = cm_t.ravel()
    thresh_list.append({
        "Threshold": t,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(precision_score(c_test, preds_t, zero_division=0), 4),
        "Recall": round(recall_score(c_test, preds_t, zero_division=0), 4)
    })

thresh_df = pd.DataFrame(thresh_list)
print("\n--- A7. DECISION TREE THRESHOLD SWEEP ---")
print(thresh_df.to_string(index=False))

# ==========================================
# A8. Feature Importances
# ==========================================
raw_coefs = lr.coef_

# Standardized Linear Coefficients
std_pipe = make_pipeline(StandardScaler(), LinearRegression())
std_pipe.fit(X_train, y_train)
std_coefs = std_pipe[-1].coef_

# Tree Feature Importances
tree_imp = chosen_tree.feature_importances_

# Permutation Importances for KNN and Tree
perm_knn = permutation_importance(knn_scaled, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
perm_tree = permutation_importance(chosen_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Raw_Coef": np.round(raw_coefs, 4),
    "Std_Coef": np.round(std_coefs, 4),
    "Tree_Imp": np.round(tree_imp, 4),
    "Perm_KNN_Mean": np.round(perm_knn.importances_mean, 4),
    "Perm_Tree_Mean": np.round(perm_tree.importances_mean, 4)
})

print("\n--- A8. FEATURE IMPORTANCE TABLE ---")
print(imp_df.to_string(index=False))

# Bar Plot for Feature Importances
fig, ax = plt.subplots(figsize=(10, 5))
x_axis = np.arange(len(X.columns))
ax.bar(x_axis - 0.2, std_coefs, 0.2, label="Std Coef (Linear Reg)")
ax.bar(x_axis, tree_imp, 0.2, label="Tree Importance")
ax.bar(x_axis + 0.2, perm_knn.importances_mean, 0.2, label="Permutation Imp (KNN)")
ax.set_xticks(x_axis)
ax.set_xticklabels(X.columns, rotation=45)
ax.set_ylabel("Importance Value")
ax.set_title("Feature Importance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("figures/A8_feature_importances.png")
plt.close()

print("\nSession A execution complete! Metrics saved to 'results/A_metrics.csv'.")
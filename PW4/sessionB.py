import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Ensure output directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================================
# B1. Load and Hide Labels
# ==========================================
wine = load_wine(as_frame=True)
X = wine.data  # 178 rows x 13 measurements
cultivar = wine.target  # Set aside until B6

print("--- B1. DATASET SUMMARY ---")
print(f"Dataset shape: {X.shape}")
print("\nFeature Summary (X.describe()):")
print(X.describe().T[["mean", "std", "min", "max"]])

# ==========================================
# B2. PCA on Raw Data
# ==========================================
pca_raw = PCA()
pca_raw.fit(X)

print("\n--- B2. RAW PCA ---")
print(f"PC1 Explained Variance Ratio: {pca_raw.explained_variance_ratio_[0]:.4f}")
loadings_pc1_raw = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(ascending=False)
print("PC1 Loadings (Raw Data):")
print(loadings_pc1_raw)

# ==========================================
# B3. Scale, then PCA Again
# ==========================================
scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca_scaled = PCA()
pca_scaled.fit(Z)

exp_var = pca_scaled.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

print("\n--- B3. SCALED PCA ---")
print("Explained Variance Ratio per PC:")
for i, (ev, cv) in enumerate(zip(exp_var, cum_var), 1):
    print(f"PC{i}: {ev:.4f} (Cumulative: {cv:.4f})")

# Scree plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.6, label="Individual")
ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="red", label="Cumulative")
ax.axhline(0.9, color="black", linestyle="--", label="90% Threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Scree Plot (Scaled Data)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/B3_scree_plot.png")
plt.close()

# PC1 & PC2 Loadings
loadings_scaled = pd.DataFrame(
    pca_scaled.components_[:2].T,
    index=X.columns,
    columns=["PC1", "PC2"]
)
print("\nPC1 and PC2 Loadings (Scaled):")
print(loadings_scaled)

# Scatter on PC1-PC2 plane
Z_pca = pca_scaled.transform(Z)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.7, color="navy")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Wines on PC1-PC2 Plane (Unlabeled)")
plt.tight_layout()
plt.savefig("figures/B3_pc1_pc2_scatter.png")
plt.close()

# ==========================================
# B4. K-Means
# ==========================================
# (a) Check inertia variation for n_init=1 across random states
inertias_ninit1 = []
for seed in range(8):
    km_test = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=seed)
    km_test.fit(Z)
    inertias_ninit1.append(km_test.inertia_)

print("\n--- B4. K-MEANS INERTIA VARIATION (n_init=1) ---")
print(f"Inertias across seeds (0-7): {[round(i, 2) for i in inertias_ninit1]}")

# (b) Elbow curve for k=1..10
inertias_k = []
for k in range(1, 11):
    km_k = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km_k.fit(Z)
    inertias_k.append(km_k.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, 11), inertias_k, marker="o", color="darkgreen")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig("figures/B4_elbow_curve.png")
plt.close()

# ==========================================
# B5. What are the clusters?
# ==========================================
chosen_k = 3
km_final = KMeans(n_clusters=chosen_k, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_final.fit(Z)

counts = np.bincount(km_final.labels_)
centers_real = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=X.columns
)

print("\n--- B5. CLUSTER SIZES & CENTERS ---")
print(f"Cluster sizes: {counts}")
print("\nCluster Centers (Real Units):")
print(centers_real.round(2).T)

# Scatter plot colored by cluster
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km_final.labels_, cmap="viridis", alpha=0.8)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("K-Means Clusters on PC1-PC2 Plane")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("figures/B5_clusters_scatter.png")
plt.close()

# ==========================================
# B6. The Reveal (Agreement Score)
# ==========================================
def compute_agreement(true_labels, cluster_labels):
    ct = pd.crosstab(true_labels, cluster_labels)
    return ct.max(axis=0).sum() / len(true_labels)

# 1. Scaled Data K-Means
ag_scaled = compute_agreement(cultivar, km_final.labels_)

# 2. Raw Data K-Means
km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_raw.fit(X)
ag_raw = compute_agreement(cultivar, km_raw.labels_)

# 3. PCA (First 2 components) K-Means
km_pca2 = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_pca2.fit(Z_pca[:, :2])
ag_pca2 = compute_agreement(cultivar, km_pca2.labels_)

ag_df = pd.DataFrame([
    {"Representation": "Scaled Data (13 features)", "Agreement": round(ag_scaled, 4)},
    {"Representation": "Raw Data (13 features)", "Agreement": round(ag_raw, 4)},
    {"Representation": "PCA First 2 Components (Scaled)", "Agreement": round(ag_pca2, 4)}
])

print("\n--- B6. AGREEMENT SCORE TABLE ---")
print(ag_df.to_string(index=False))

# Save agreement table to results/B_agreement.csv
ag_df.to_csv("results/B_agreement.csv", index=False)

print("\nSession B execution complete! Results saved to 'results/B_agreement.csv'.")
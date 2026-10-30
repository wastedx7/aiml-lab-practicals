import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
original_component_names = cancer.feature_names

print("Original Dataset Shape:", X.shape)

# 2. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA - Retain Top 5 Components
n_retained = 5
pca_reduced = PCA(n_components=n_retained)
X_reduced = pca_reduced.fit_transform(X_scaled)

print("Reduced Dataset Shape:", X_reduced.shape)

# 4. Variance Calculations
var_ratio_reduced = pca_reduced.explained_variance_ratio_ * 100
cum_var_reduced = np.sum(var_ratio_reduced)

# 5. Original Dataset Variance
orig_var_ratio = np.full(
    shape=len(original_component_names),
    fill_value=100.0 / len(original_component_names)
)

# 6. Descriptive Names for Reduced Components
reduced_component_names = [
    "Size & Radius",
    "Texture & Perimeter",
    "Shape & Compactness",
    "Concavity & Symmetry",
    "Fractal Dimension"
]

# 7. Original Dataset Table
df_original = pd.DataFrame({
    "Index": range(1, len(original_component_names) + 1),
    "Original Feature Name": original_component_names
})

# 8. Reduced Dataset Table
df_reduced = pd.DataFrame({
    "Index": range(1, n_retained + 1),
    "Reduced Component Name": reduced_component_names,
    "Variance Explained (%)": np.round(var_ratio_reduced, 2),
    "Cumulative Variance (%)": np.round(
        np.cumsum(var_ratio_reduced), 2
    )
})

# 9. Display Original Dataset
print("\n" + "=" * 65)
print(" ORIGINAL DATASET (30 RAW FEATURES)")
print("=" * 65)
print(df_original.to_string(index=False))

# 10. Display Reduced Dataset
print("\n" + "=" * 65)
print(" REDUCED DATASET (5 PCA COMPONENTS)")
print("=" * 65)
print(df_reduced.to_string(index=False))

# 11. Variance Summary
print("\n" + "=" * 65)
print(" VARIANCE SUMMARY")
print("=" * 65)

for i in range(n_retained):
    print(
        f"PC{i + 1} - {reduced_component_names[i]} : "
        f"{var_ratio_reduced[i]:.2f}%"
    )

print(
    f"\nTotal Variance Explained by 5 Components: "
    f"{cum_var_reduced:.2f}%"
)

# 12. Side-by-Side Graphs
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(18, 7)
)

# Graph 1: Original Dataset
bars1 = ax1.bar(
    range(len(original_component_names)),
    orig_var_ratio,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8
)

ax1.set_xticks(range(len(original_component_names)))
ax1.set_xticklabels(
    original_component_names,
    rotation=75,
    ha="right",
    fontsize=7
)

ax1.set_ylabel(
    "Variance Contribution (%)",
    fontsize=11,
    fontweight="bold"
)

ax1.set_title(
    "Original Dataset (30 Raw Features)",
    fontsize=12,
    fontweight="bold"
)

ax1.set_ylim(0, max(10, max(var_ratio_reduced) + 5))
ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

for bar in bars1:
    ax1.annotate(
        "3.3%",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()
        ),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6
    )

# Graph 2: Reduced Dataset
bars2 = ax2.bar(
    reduced_component_names,
    var_ratio_reduced,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8
)

ax2.set_xticks(range(len(reduced_component_names)))
ax2.set_xticklabels(
    reduced_component_names,
    rotation=25,
    ha="right",
    fontsize=9,
    fontweight="bold"
)

ax2.set_xlabel(
    "Descriptive Reduced Component Names",
    fontsize=11,
    fontweight="bold"
)

ax2.set_ylabel(
    "Variance Contribution (%)",
    fontsize=11,
    fontweight="bold"
)

ax2.set_title(
    f"Reduced Dataset (5 Components = "
    f"{cum_var_reduced:.1f}% Total Variance)",
    fontsize=12,
    fontweight="bold"
)

ax2.set_ylim(0, max(10, max(var_ratio_reduced) + 5))
ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(
        f"{height:.1f}%",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            height
        ),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()

# 13. Optional PCA 2D Visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

for target in np.unique(y):
    plt.scatter(
        X_2d[y == target, 0],
        X_2d[y == target, 1],
        label=cancer.target_names[target],
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Breast Cancer Dataset - PCA 2D Visualization")
plt.legend()
plt.grid(True)
plt.show()
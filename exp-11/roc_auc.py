import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 1. Generate a Synthetic Imbalanced Dataset
# 1000 samples with 10% positive class
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42,
    n_features=4
)

# 2. Split into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 3. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict Probabilities
y_val_probs = model.predict_proba(X_val)[:, 1]

# Default Threshold Predictions (0.50)
y_val_pred_default = (y_val_probs >= 0.50).astype(int)

# 5. Evaluate at Default Threshold (0.50)
print("--- Validation at Threshold 0.50 ---")

tn, fp, fn, tp = confusion_matrix(
    y_val,
    y_val_pred_default
).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
fpr = fp / (fp + tn)

print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"FPR: {fpr:.2f}\n")

# 6. Calculate ROC and AUC Across All Thresholds
fpr_array, tpr_array, thresholds = roc_curve(
    y_val,
    y_val_probs
)

roc_auc = auc(fpr_array, tpr_array)

print("--- ROC-AUC Summary ---")
print(
    f"AUC Score: {roc_auc:.3f} "
    f"(Probability model ranks a random positive over a random negative)"
)

# 7. Plot the ROC Curve
plt.figure(figsize=(8, 6))

plt.plot(
    fpr_array,
    tpr_array,
    color="blue",
    lw=2,
    label=f"ROC curve (AUC = {roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    color="red",
    lw=2,
    linestyle="--",
    label="Random Classifier (AUC = 0.50)"
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC)")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
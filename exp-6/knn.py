import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# Step 1: Load Dataset
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "Error: 'data.csv' file not found in the current working directory."
    )

print("=" * 60)
print("WINE DATASET - k-NN CLASSIFICATION")
print("=" * 60)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows of Dataset:")
print(df.head())

# Step 2: Select Features and Target
features = ["Alcohol", "Malic acid", "Color intensity"]
target_col = "class"

missing_cols = [
    col for col in features + [target_col]
    if col not in df.columns
]

if missing_cols:
    raise KeyError(f"Missing columns in 'data.csv': {missing_cols}")

X = df[features]
y = df[target_col]

print("\nSelected Parameters:")
for feature in features:
    print("-", feature)

unique_classes = sorted(y.unique())

print("\nNumber of Classes:", len(unique_classes))

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 4: Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# Step 5: Take K from User
while True:
    try:
        k = int(input("\nEnter value of K: "))

        if k <= 0:
            print("K must be greater than 0.")
            continue

        if k > len(X_train):
            print(f"K cannot exceed total training samples ({len(X_train)}).")
            continue

        break

    except ValueError:
        print("Invalid entry. Please enter a valid integer for K.")

# Step 6: Create and Train k-NN Model
knn = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)

print("\nk-NN model trained successfully.")
print("Distance Metric: Euclidean")
print("K =", k)

# Step 7: Model Evaluation
y_pred = knn.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(cm)

print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print("Accuracy =", round(accuracy, 4))
print("Accuracy Percentage =", round(accuracy * 100, 2), "%")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

target_names = [f"Class {c}" for c in unique_classes]

print(
    classification_report(
        y_test,
        y_pred,
        labels=unique_classes,
        target_names=target_names,
        zero_division=0
    )
)

# Step 8: User Input for Custom Sample Prediction
print("\n" + "=" * 60)
print("PREDICT A NEW WINE SAMPLE")
print("=" * 60)


def prompt_float(field_name):
    while True:
        try:
            return float(input(f"Enter {field_name}: "))
        except ValueError:
            print("Invalid input! Please enter a numerical value.")


alcohol = prompt_float("Alcohol")
malic_acid = prompt_float("Malic Acid")
color_intensity = prompt_float("Color Intensity")

# Create DataFrame for New Sample
new_sample = pd.DataFrame(
    [[alcohol, malic_acid, color_intensity]],
    columns=features
)

new_sample_scaled = scaler.transform(new_sample)

# Predict Wine Class
prediction = knn.predict(new_sample_scaled)

# Display Prediction Result
print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("Alcohol:", alcohol)
print("Malic Acid:", malic_acid)
print("Color Intensity:", color_intensity)

print("\nPredicted Wine Class:", prediction[0])

print("=" * 60)
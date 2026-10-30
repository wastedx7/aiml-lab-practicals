import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 1: Load Dataset
data = pd.read_csv("data.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)

print(data.head())

# Step 2: Select Features and Target
features = ['area', 'perimeter', 'compactness']

X = data[features]
y = data.iloc[:, -1]

print("\nSelected Features:")
print(features)

print("\nTarget Column:")
print(y.name)

# Step 3: Check Missing Values
print("\nMissing Values:")
print(X.isnull().sum())

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 5: Create Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Step 6: Train Model
model.fit(X_train, y_train)

print("\nDecision Tree Training Completed!")

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 10: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Decision Tree Visualization
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=features,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Classifier")
plt.show()

# Step 12: Feature Importance
print("\nFeature Importance:")

for feature, value in zip(
    features,
    model.feature_importances_
):
    print(feature, ":", value)

# Step 13: Feature Importance Graph
plt.figure(figsize=(7, 5))

plt.bar(
    features,
    model.feature_importances_
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Step 14: User Input
print("\n================================")
print("     NEW SEED CLASSIFICATION")
print("================================")

area = float(input("Enter Area: "))
perimeter = float(input("Enter Perimeter: "))
compactness = float(input("Enter Compactness: "))

new_data = pd.DataFrame(
    [[area, perimeter, compactness]],
    columns=features
)

prediction = model.predict(new_data)[0]

# Step 15: Display Prediction
print("\n================================")
print("        PREDICTION RESULT")
print("================================")

print("Area:", area)
print("Perimeter:", perimeter)
print("Compactness:", compactness)

print("\nPredicted Class:", prediction)

class_names = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian"
}

if prediction in class_names:
    print("Predicted Wheat Variety:", class_names[prediction])

print("================================")
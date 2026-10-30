import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Load Dataset
data = pd.read_csv("admission.csv")

print("========== FIRST 5 ROWS OF DATASET ==========")
print(data.head())

# Step 2: Display Dataset Information
print("\n========== DATASET INFORMATION ==========")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# Step 3: Standardize Column Names
data.columns = data.columns.str.strip()

column_mapping = {}

for col in data.columns:
    c = col.lower().replace(" ", "").replace("_", "")

    if c == "gpa":
        column_mapping[col] = "GPA"

    elif c in ["satscore", "sat"]:
        column_mapping[col] = "SAT_Score"

    elif c in [
        "extracurricularactivities",
        "extracurricularactivity",
        "activities",
        "extracurricular"
    ]:
        column_mapping[col] = "Extracurricular_Activities"

    elif c in [
        "admissionstatus",
        "admission",
        "status",
        "acceptedrejected"
    ]:
        column_mapping[col] = "Admission_Status"

data.rename(columns=column_mapping, inplace=True)

# Step 4: Check Required Columns
required_columns = [
    "GPA",
    "SAT_Score",
    "Extracurricular_Activities",
    "Admission_Status"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    print("\nERROR: The following columns are missing:")
    print(missing_columns)
    print("\nAvailable columns are:")
    print(list(data.columns))
    raise SystemExit

# Step 5: Remove Missing Values
data = data[required_columns].dropna()

# Step 6: Convert Admission Status into Binary Values
def convert_status(status):
    status = str(status).strip().lower()

    if status in ["accepted", "accept", "1", "yes"]:
        return 1

    elif status in ["rejected", "reject", "0", "no"]:
        return 0

    else:
        return np.nan


data["Admission_Binary"] = data["Admission_Status"].apply(
    convert_status
)

data = data.dropna(subset=["Admission_Binary"])
data["Admission_Binary"] = data["Admission_Binary"].astype(int)

# Step 7: Select Input and Target Variables
X = data[
    [
        "GPA",
        "SAT_Score",
        "Extracurricular_Activities"
    ]
]

y = data["Admission_Binary"]

# Step 8: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# Step 9: Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 11: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# Step 13: Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

# Step 14: Display Actual vs Predicted Results
result = X_test.copy()

result["Actual"] = np.where(
    y_test == 1,
    "Accepted",
    "Rejected"
)

result["Predicted"] = np.where(
    y_pred == 1,
    "Accepted",
    "Rejected"
)

print("\n========== ACTUAL VS PREDICTED ==========")
print(result.head(10))

# Step 15: Take Input from User
print("\n==========================================")
print("       STUDENT ADMISSION PREDICTION")
print("==========================================")

gpa = float(input("Enter GPA (2.5 - 4.0): "))
sat = float(input("Enter SAT Score (900 - 1600): "))
activities = int(input("Enter Extracurricular Activities (0 or 1): "))

# Create Input DataFrame
new_student = pd.DataFrame(
    [[gpa, sat, activities]],
    columns=[
        "GPA",
        "SAT_Score",
        "Extracurricular_Activities"
    ]
)

# Scale Input
new_student_scaled = scaler.transform(new_student)

# Predict Class
prediction = model.predict(new_student_scaled)[0]

# Predict Probability
probability = model.predict_proba(new_student_scaled)[0][1]

# Step 16: Display Prediction
print("\n========== PREDICTION RESULT ==========")

if prediction == 1:
    print("Admission Status : ACCEPTED")
else:
    print("Admission Status : REJECTED")

print(f"Probability of Acceptance : {probability * 100:.2f}%")

print("\n========== EXPERIMENT COMPLETED ==========")
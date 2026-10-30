import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset
df = pd.read_csv("student_pass_fail_dataset.csv")

print("========= STUDENT DATASET ==========")
print(df)

# Step 2: Convert Result into Numbers
df["Result"] = df["Result"].map({
    "Pass": 1,
    "Fail": 0
})

# Step 3: Select Input and Output
X = df[
    [
        "Study_Hours",
        "Attendance_Percentage",
        "Previous_Marks"
    ]
]

y = df["Result"]

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Step 5: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Linear SVM
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)

linear_accuracy = accuracy_score(
    y_test,
    y_pred_linear
)

print("\n========== LINEAR SVM ==========")
print("Accuracy:", linear_accuracy * 100, "%")

# Step 7: Non-Linear SVM
nonlinear_svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

nonlinear_svm.fit(X_train, y_train)

y_pred_nonlinear = nonlinear_svm.predict(X_test)

nonlinear_accuracy = accuracy_score(
    y_test,
    y_pred_nonlinear
)

print("\n========== NON-LINEAR SVM ==========")
print("Accuracy:", nonlinear_accuracy * 100, "%")

# Step 8: Create 2D Data for Visualization
X_graph = df[
    [
        "Study_Hours",
        "Previous_Marks"
    ]
].values

y_graph = df["Result"].values

graph_scaler = StandardScaler()
X_graph_scaled = graph_scaler.fit_transform(X_graph)

# Step 9: Linear SVM for Graph
graph_linear_svm = SVC(kernel="linear")

graph_linear_svm.fit(
    X_graph_scaled,
    y_graph
)

# Step 10: Non-Linear SVM for Graph
graph_nonlinear_svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

graph_nonlinear_svm.fit(
    X_graph_scaled,
    y_graph
)

# Step 11: Decision Boundary Function
def plot_decision_boundary(model, X, y, title):
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    # Decision Regions
    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3
    )

    # Data Points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="black",
        s=70
    )

    # Support Vectors
    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Support Vectors"
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Previous Marks")
    plt.title(title)
    plt.legend()
    plt.grid(True)


# Step 12: Linear SVM Decision Boundary
plt.figure(figsize=(8, 6))

plot_decision_boundary(
    graph_linear_svm,
    X_graph_scaled,
    y_graph,
    "Linear SVM - Student Pass/Fail"
)

plt.show()

# Step 13: Non-Linear SVM Decision Boundary
plt.figure(figsize=(8, 6))

plot_decision_boundary(
    graph_nonlinear_svm,
    X_graph_scaled,
    y_graph,
    "Non-Linear SVM (RBF) - Student Pass/Fail"
)

plt.show()
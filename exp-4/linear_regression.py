import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

print("First 5 rows of the dataset:")
print(data.head())

print("\nColumn names:")
print(data.columns)

# Select input and output
X = data[["Weight"]]
Y = data["Height"]

# Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Display regression equation
slope = model.coef_[0]
intercept = model.intercept_

print("\nLinear Regression Equation:")
print(f"Height = {slope:.4f} × Weight + {intercept:.4f}")

# Plot actual data and regression line
plt.figure(figsize=(8, 5))

plt.scatter(
    X["Weight"],
    Y,
    color="blue",
    label="Actual Data"
)

plt.plot(
    X["Weight"],
    model.predict(X),
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.title("Linear Regression: Weight vs Height")
plt.legend()
plt.grid(True)
plt.show()

# Get weight input from user
while True:
    try:
        weight_input = float(input("\nEnter weight in kg: "))

        if weight_input <= 0:
            print("Please enter a positive weight.")
            continue

        break

    except ValueError:
        print("Invalid input! Please enter a number.")

# Predict height
input_data = pd.DataFrame({
    "Weight": [weight_input]
})

predicted_height = model.predict(input_data)

# Display prediction result
print("\n========================")
print("     PREDICTION RESULT")
print("========================")
print(f"Input Weight      : {weight_input:.2f} kg")
print(f"Predicted Height  : {predicted_height[0]:.2f} cm")
print("========================")
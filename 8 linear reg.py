import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv("C:/Users/kamini/Downloads/machine learning/Position_Salaries.csv")

# Extract the independent variable (X) and dependent variable (Y)
X = data["Level"].values.reshape(-1, 1)
Y = data["Salary"].values

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, Y)

# Make predictions using the model
Y_pred = model.predict(X)

# Plot the original data and the regression line
plt.scatter(X, Y, label="Original Data")
plt.plot(X, Y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Display the regression equation parameters
intercept = model.intercept_
slope = model.coef_[0]
print(f"Regression Equation: Y = {intercept:.4f} + {slope:.4f} * X")

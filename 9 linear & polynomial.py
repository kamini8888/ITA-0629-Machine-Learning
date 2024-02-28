import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("C:/Users/kamini/Downloads/machine learning/Position_Salaries.csv")
x = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
print("Linear Regression Prediction for position level 6.5:", lin_reg.predict([[6.5]]))
print("Polynomial Regression Prediction for position level 6.5:", lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

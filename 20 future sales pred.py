import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
sales_data = pd.read_csv("C:/Users/kamini/Downloads/machine learning/futuresale prediction.csv")
X = sales_data[['TV', 'Radio', 'Newspaper']]
y = sales_data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LinearRegression()
model.fit(X_train, y_train)
tv = float(input("Enter TV price: "))
radio = float(input("Enter radio price: "))
newspaper = float(input("Enter newspaper price: "))
predicted_sales = model.predict([[tv, radio, newspaper]])
print("Predicted sales:", predicted_sales[0])




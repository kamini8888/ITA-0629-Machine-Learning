import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df = pd.read_csv("C:/Users/kamini/Downloads/machine learning/mobile_prices.csv")
selected_features = ['battery_power', 'ram', 'n_cores', 'px_height']
X = df[selected_features]
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
new_mobile_data = pd.DataFrame({
    'battery_power': [float(input("Enter battery power: "))],
    'ram': [float(input("Enter RAM size: "))],
    'n_cores': [float(input("Enter number of cores: "))],
    'px_height': [float(input("Enter pixel height: "))],
})
predicted_price_range = knn.predict(new_mobile_data)
print("Predicted Price Range for the New Mobile:", predicted_price_range)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy",accuracy)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix",cm)

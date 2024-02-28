import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.80, random_state=72)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
predicted_species = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])
species_names = iris.target_names
predicted_species_name = species_names[predicted_species[0]]
print("Predicted species of the new flower:", predicted_species_name)

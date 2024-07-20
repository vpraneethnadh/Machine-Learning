import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

iris = pd.read_csv("IRISDataSet.csv")

X = iris[['sepal_width', 'petal_length', 'petal_width']]
y = iris['sepal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

test_data = np.array([[2.9, 1.4, 0.2]])
predicted_output = lr.predict(test_data)[0]
print("Predicted sepal length of the test datapoint:", predicted_output)

y_pred = lr.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared:", r2_score(y_test, y_pred))

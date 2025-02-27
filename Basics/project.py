import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')

if 'Name' in df.columns:
    df = df.drop(columns=['Name'])

X = df.drop(columns=['Result'])
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

arr1 = np.array([12, 14, 15, 67, 78, 25]).reshape(1, -1)
arr2 = np.array([34, 56, 78, 12, 34, 3]).reshape(1, -1)
arr3 = np.array([78, 90, 12, 43, 4, 19]).reshape(1, -1)
arr4 = np.array([45, 70, 81, 3, 5, 9]).reshape(1, -1)

predict1 = model.predict(arr1)
predict2 = model.predict(arr2)
predict3 = model.predict(arr3)
predict4 = model.predict(arr4)

print(predict1)
print(predict2)
print(predict3)
print(predict4)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Model Accuracy:", score)

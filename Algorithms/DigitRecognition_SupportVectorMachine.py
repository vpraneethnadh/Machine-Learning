from sklearn.datasets import load_digits
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as num

digits = load_digits()

x = digits.data
y = digits.target

plt.figure(figsize=(20, 4))

for index, (image, label) in enumerate(zip(x[0:5], y[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(num.reshape(image, (8, 8)))

plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = svm.SVC(max_iter=18000)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("The Predicted set is: ")
print(y_pred[0:5])
print("The tested set is: ")
print(y_test[0:5])
print("R2 Score Found Using Support Vector Machine (SVM): ")
print(r2_score(y_test, y_pred))

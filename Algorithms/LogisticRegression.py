from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import precision_score
import numpy as np

placement = pd.read_csv("Placement_Data_Full_Class.csv")

new_dataset = placement[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'status']]

X = new_dataset.iloc[:, :-1]
Y = new_dataset.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
clf = svm.SVC()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='weighted')
Recall = recall_score(Y_test, Y_pred, average='weighted')
Precision = precision_score(Y_test, Y_pred, average='weighted')
cm = confusion_matrix(Y_test, Y_pred)

# print("The Actual and the predicted values are: ")
# df = pd.DataFrame({'Actual': Y_test[0:20], 'Predicted': Y_pred[0:20]})
# print(df)
# print("\n")

input_data = (1, 67.00, 91.00, 58.99, 91)
np_input_data = np.asarray(input_data)
reshape_input_data = np_input_data.reshape(1, -1)

prediction = clf.predict(reshape_input_data)
print("Prediction is: ", prediction)
print("The accuracy of the predicted values is: ", accuracy_score(Y_test, Y_pred))
print("F1 Score of the model is: ", f1)
print("Precision Score of the model is: ", Precision)
print("Recall Score of the model is: ", Recall)
print("Confusion matrix for the model is: \n", cm)

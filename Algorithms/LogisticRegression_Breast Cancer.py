from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def svm1():
    data1 = load_breast_cancer()
    x = data1.data
    y = data1.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    clf = svm.SVC(max_iter=10000)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy_svm = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    p_svm = precision_score(y_test, y_pred)
    r_svm = recall_score(y_test, y_pred)
    print("The Predicted set is: ")
    print(y_pred)
    print("The tested set is: ")
    print(y_test)
    print("Accuracy Score Found Using Support Vector Machine (SVM): ")
    print(accuracy_svm)
    print("Confusion Matrix: ")
    print(cm)
    print("Precision Score: ")
    print(p_svm)
    print("Recall Score: ")
    print(r_svm)
    return accuracy_svm


def logistic_regression():
    data1 = load_breast_cancer()
    x = data1.data
    y = data1.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    lr = LogisticRegression(max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    accuracy_lr = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    p_lr = precision_score(y_test, y_pred)
    r_lr = recall_score(y_test, y_pred)
    print("The Predicted set is: ")
    print(y_pred)
    print("The tested set is: ")
    print(y_test)
    print("Accuracy Score Found Using Logistic Regression: ")
    print(accuracy_lr)
    print("Confusion Matrix: ")
    print(cm)
    print("Precision Score: ")
    print(p_lr)
    print("Recall Score: ")
    print(r_lr)
    return accuracy_lr


print("Support Vector Machine: ")
svm = svm1()
print("\n")
print("Logistic Regression:")
lor = logistic_regression()

if svm > lor:
    print("Accuracy using Support Vector Machine is higher")

if svm < lor:
    print("Accuracy using Logistic Regression is higher")
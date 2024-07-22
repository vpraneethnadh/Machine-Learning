import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("NavieBayes.csv")

categorical = [var for var in df.columns if df[var].dtype == 'O']
print('There are {} categorical variables'.format(len(categorical)))
print('The categorical variables are :\n', categorical)

df['workclass'].replace('?', np.NaN, inplace=True)
df.occupation.unique()
df.occupation.value_counts()
df['occupation'].replace('?', np.NaN, inplace=True)

df.native_country.unique()
df.native_country.value_counts()
df['occupation'].replace('?', np.NaN, inplace=True)

numerical = [var for var in df.columns if df[var].dtype != 'O']
print('There are {} numerical variables'.format(len(numerical)))
print('The numerical variables are :\n', numerical)

X = df.drop(['income'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

cols = X_train.columns
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred_train = gnb.predict(X_train)
null_accuracy = (1211 / (1211 + 389))
cm = confusion_matrix(y_test, y_pred)
print("\n")
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))
print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
print("\n")
print("Most Frequent class is: ", y_test.value_counts())
print('Null accuracy score: {0:0.4f}'.format(null_accuracy))
print("\n")
print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0, 0])
print('True Negatives(TN) = ', cm[1, 1])
print('False Positives(FP) = ', cm[0, 1])
print('False Negatives(FN) = ', cm[1, 0])

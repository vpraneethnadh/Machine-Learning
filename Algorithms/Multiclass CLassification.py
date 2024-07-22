import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('Articles.csv', encoding='latin-1')

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Heading'])
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)
y = data['NewsType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = LinearSVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification report:\n{report}")

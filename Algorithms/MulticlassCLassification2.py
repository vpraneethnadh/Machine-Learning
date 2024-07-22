# Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv('train.csv')

data['title'] = data['title'].apply(lambda x: x.lower())
data['title'] = data['title'].str.replace('[^\w\s]', '')
data['title'] = data['title'].str.replace('\d+', '')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

svm = SVC(kernel='linear', C=1.0, gamma='auto', decision_function_shape='ovr')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))

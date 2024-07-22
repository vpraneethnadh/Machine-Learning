import pandas as pd
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import re
import string as s
from sklearn.feature_extraction.text import TfidfVectorizer

data_train = pd.read_csv('NewTrain2.csv')
data_test = pd.read_csv('NewTest2.csv')

train_x = data_train.Description
test_x = data_test.Description
train_y = data_train.Class_Index
test_y = data_test.Class_Index


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


train_x = train_x.apply(remove_html)
test_x = test_x.apply(remove_html)


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


train_x = train_x.apply(remove_urls)
test_x = test_x.apply(remove_urls)


def word_tokenize(txt):
    tokens = re.findall("[\w']+", txt)
    return tokens


train_x = train_x.apply(word_tokenize)
test_x = test_x.apply(word_tokenize)


def remove_stopwords(lst):
    stop = stopwords.words('english')
    new_lst = []
    for i in lst:
        if i.lower() not in stop:
            new_lst.append(i)
    return new_lst

train_x = train_x.apply(remove_stopwords)
test_x = test_x.apply(remove_stopwords)


def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            i = i.replace(j, '')
        new_lst.append(i)
    return new_lst


train_x = train_x.apply(remove_punctuations)
test_x = test_x.apply(remove_punctuations)


def remove_numbers(lst):
    nodig_lst = []
    new_lst = []

    for i in lst:
        for j in s.digits:
            i = i.replace(j, '')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i != '':
            new_lst.append(i)
    return new_lst


train_x = train_x.apply(remove_numbers)
test_x = test_x.apply(remove_numbers)

train_x = train_x.apply(lambda x: ''.join(i + ' ' for i in x))
test_x = test_x.apply(lambda x: ''.join(i + ' ' for i in x))

tfidf = TfidfVectorizer(min_df=8, ngram_range=(1, 3))
train_X1 = tfidf.fit_transform(train_x)
test_X1 = tfidf.transform(test_x)

train_arr = train_X1.toarray()
test_arr = test_X1.toarray()

clf = svm.SVC(max_iter=18000)
clf.fit(train_arr, train_y)
pred = clf.predict(test_arr)

print("first 20 actual labels")
print(test_y.tolist()[:20])
print("first 20 predicted labels")
print(pred.tolist()[:20])

print("F1 score of the model", f1_score(test_y, pred))
print("Accuracy of the model", accuracy_score(test_y, pred))
report = classification_report(test_y, pred)
print("Classification Report is: ", report)

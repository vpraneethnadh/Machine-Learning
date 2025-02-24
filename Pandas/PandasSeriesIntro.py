import pandas as pd
import numpy as np

# data = pd.read_csv("dataset.csv")

# s = pd.Series([1,3,5,7,9, None])
# print(s)

# print(s.describe())

# print(s.mean())
# print(s)

# full = s.isnull()
# print(full)

# filled = s.fillna(s.mean())
# print(filled)

# full = filled.isnull()
# print(full)

s = pd.Series([10,20,30,40,50], index = ['a','b','c','d','e'])
print(s)

print(s.iloc[0])
print(s.loc['b': 'c'])

filtered = s[s > 25]
print(filtered)

print(s.sum())
print(s.cumsum())

aggregated = s.aggregate(['sum', 'mean', 'std'])
print(aggregated)
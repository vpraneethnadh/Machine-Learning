# Vudattu Praneethnadh - 21BCE7762
import pandas as pd
import numpy as np

data = pd.read_csv("data1.csv")
print(data)

attribute = np.array(data)[:, :-1]
print("\nThe attributes are: ")
print(attribute)

target = np.array(data)[:, -1]
print("\nThe target is: ", target)


def find_s(c, t):
    global specific
    for i, val in enumerate(t):
        if val == "Yes":
            specific = c[i].copy()
            break

    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific)):
                if val[x] != specific[x]:
                    specific[x] = '?'
                else:
                    pass

    return specific


print("The final hypothesis is:", find_s(attribute, target))

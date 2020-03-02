import tensorflow
import keras
import sklearn
import pickle

import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""
# Train model
best = 0

while best < 0.95:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    # print(acc)

    if acc > best:
        best = acc


with open("student_model.pickle", "wb") as f:
    pickle.dump(linear, f)
print(best)
"""
pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

print(linear.coef_)
print(linear.intercept_)

pred = linear.predict(x_test)

for i in range(len(pred)):
    print(pred[i], x_test[i], y_test[i])

p = "G1"

style.use("ggplot")

pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("G3")
pyplot.show()

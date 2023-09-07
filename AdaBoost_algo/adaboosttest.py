import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from adaboost import Adaboost

def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    print("Accuracy:", accuracy)
    
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# important as we are predicting -1 or 1
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Adaboost classification with 5 weak classifiers
clf = Adaboost(n_clf=5)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

acc = accuracy(y_test, y_hat)

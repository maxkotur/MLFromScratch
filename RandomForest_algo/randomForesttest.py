import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from randomForest import RandomForset

def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    print(accuracy)

data=datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RandomForset(n_trees=3)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy(y_test, predictions)

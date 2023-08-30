import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from LogisticRegression import LogisitcRegression

bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gets how many of the labels are correctly assigned
def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy
    
regressor = LogisitcRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

accuracy_val = accuracy(y_test, predictions)
print(accuracy_val)
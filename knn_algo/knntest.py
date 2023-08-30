import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn import KNN

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape) # (120, 4) so 120 samples with 4 features

# plt.figure()
# plt.scatter(X[:, 2], X[:, 3], c=y, s=20, edgecolor='k')
# plt.show()

clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions== y_test) / len(y_test)
print(accuracy)
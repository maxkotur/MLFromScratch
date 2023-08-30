import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# accuracy is calculated using MSE
def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

mse_val = mse(y_test, predictions)
print(mse_val)

y_pred_line = regressor.predict(X)
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X, y, color="b", s = 10)
plt.plot(X, y_pred_line, color="b", linewidth=2, label="Prediction")
plt.show()

print(X_train.shape) # Shape is (80, 1)

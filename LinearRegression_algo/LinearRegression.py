import numpy as np

class LinearRegression:
    
    def __init__(self, lr=0.001, iter=1000):
        self.lr = lr
        self.iter = iter
        self.weights = None
        self.bias = None
        
        
    def fit(self, X, y):
        # initial params
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iter):
            y_predicted = np.dot(X, self.weights) + self.bias # y_hat = wx + b
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) # derivative of weight
            db = (1/n_samples) * np.sum(y_predicted - y) # derivated of bias

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
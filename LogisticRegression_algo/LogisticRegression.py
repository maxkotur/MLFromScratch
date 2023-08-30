import numpy as np

class LogisitcRegression:
    
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
        
        # gradient decsent
        for _ in range(self.iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_model)
            
            # get derivatives of weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_hat-y))
            db = (1/n_samples) * np.sum((y_hat-y))
            
            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(linear_model)
        y_hat_predictions = [1 if i > 0.5 else 0 for i in y_hat]
        return y_hat_predictions
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
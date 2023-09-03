import numpy as np

class SVM:
    
    def __init__(self, lr=0.001, lambda_p = 0.01, iter = 1000):
        self.lr = lr
        self.lambda_p = lambda_p
        self.iter = iter
        self.weights = None
        self.bias = None
         
    def fit(self, X, y):
        # verify if y has 1 or -1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        # initial weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iter):
            for index, x_i in enumerate(X):
                condition = y_[index] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_p * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_p * self.weights - np.dot(x_i, y_[index]))
                    self.bias -= self.lr * y_[index]
                
    
    
    def predict(self, X):
        linear_out = np.dot(X, self.weights) - self.bias
        return np.sign(linear_out)
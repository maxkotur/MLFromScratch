import numpy as np

class Perceptron:
     
    def __init__(self, lr=0.01, iter=1000):
        self.lr = lr
        self.iter = iter
        self.act_fun = self.unit_step
        self.weights = None
        self.bias = None
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initial weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # insures we have 1 or 0 as we have 2 classes
        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.iter):
            for index, x_i in enumerate(X):
                linear_out = np.dot(x_i, self.weights) + self.bias
                y_hat = self.act_fun(linear_out)
                
                # update weights
                update = self.lr * (y_[index] - y_hat)
                self.weights += update * x_i
                self.bias += update
        
        
        
        
    def predict(self, X):
        linear_out = np.dot(X, self.weights) + self.bias
        y_hat = self.act_fun(linear_out)
        return y_hat
        
    def unit_step(self, x):
        return np.where(x>=0, 1, 0)
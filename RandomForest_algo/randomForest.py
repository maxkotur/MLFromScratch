import numpy as np
import DecisionTree_algo
from DecisionTree_algo import decisionTree as dt

def boostrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]
    

class RandomForset:
    
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = dt.DecisionTree(min_samples_split=self.min_samples_split)
                   
        
    
    def predict(self, X):
        pass
        


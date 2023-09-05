import numpy as np
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
            tree = dt.DecisionTree(min_samples_split=self.min_samples_split, 
                                   max_depth=self.max_depth, n_features=self.n_features)
            X_sample, y_sample = boostrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # switch to majority vote => [1111 0000 1111] -> [101 101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # majority vote
        y_hat = [dt.DecisionTree.most_common_label(self, tree_pred) for tree_pred in tree_preds]
        return np.array(y_hat)
        
    
        
        


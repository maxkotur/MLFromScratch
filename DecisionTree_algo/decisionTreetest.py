# Train algorithm = Build the tree
# Start at the top node and at each node select the best split based on the best information gain
# Using a greedy search where we loop all possible feature values
# Save the best split feature and split threshold at each node
# Apply some criteria to stop growing
# When we have a leaf node, store the most common class label of the node

# Predict = Traverse tree
# Traverse the tree recursively
# At each node look at the best split feature of the test feature
# vector x and go left or right --> depends on x[feature_index] <= threshold
# When we reach the leaf node, return the most common class label

import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    entropy = -np.sum([p * np.log2(p) for p in ps if p > 0])
    return entropy

class Node:
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    
    # determines if at leaf node
    def is_leafNode(self):
        return self.value is not None

class DecisionTree:
    
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    
    def fit(self, X, y):
        # grow the tree
        self.n_features = X.shape[1] if not self.n_features else min(self.n_feats, X.shape[1])
        self.root = self.grow_tree(X, y)
    
    def grow_tree(self, X, y, depth=0):
        n_samples, n_feautres = X.shape
        n_labels = len(np.unique(y))
        
        # stop
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feautres, self.n_features, replace=False)
        
        # greedy search
        best_feat, best_thresh = self.best_criteria(X, y, feat_idxs)
        
    
    def best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        
        return split_idx, split_thresh
    
    def information_gain(self, y, X_column, split_thresh):
        # parent entropy
        parent_entropy = entropy(y)
        # generate split
        left_idxs, right_idxs = self.split(X_column, split_thresh)
        # weighted average child entropy
        
        # return information_gain
    
    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()
        return left_idxs, right_idxs           
            
    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    
    def predict(self, X):
        
        
        
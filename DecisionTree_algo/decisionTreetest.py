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
    
    def __init__(self, min_sample_split=2, max_depth=100, n_f):
        
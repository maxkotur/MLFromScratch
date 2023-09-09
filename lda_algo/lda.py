# Approach :
# Calculate the scatter matrix S_B (between)
# Calculate the scatter matrix S_W (within)
# Calculate the eigenvalues of S_W^-1 . S_B
# Sort the eigenvectors according to their eigenvalues in decreasing order
# Choose first k eigenvectors and that will be the new k dimensions (linear discriminant)
# Transform the original n dimensional data points into k dimensions using the dot product

import numpy as np

class LDA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None
        
    def fit(self, X, y):
        n_features = X.shape[1] # 150, 4
        class_labels = np.unique(y)
        
        # Calculate S_W, S_B (4,4) matrix
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4, 4)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)
            
            n_c = X_c.shape[0]
            # (4, 1) * (1, 4) = (4, 4)
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1) # (4, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        A = np.linalg.inv(S_W).dot(S_B)
        
        # eigenvectors and eigenvalues and sort them
        eigenvectors, eigenvalues = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1] # [::-1] reverses the list
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]
        
    
    def transform(self, X):
        pass
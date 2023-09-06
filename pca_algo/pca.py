# Approach
# Subtract the mean from X
# Calculate the Cov(X, X)
# Calculate eigenvectors and eigenvalues of covariance matrix
# Sort the eigenvectors according to their eigenvalues in decreasing order
# Choose first k eigenvectors and that will the new k dimensions
# Transform the original n dimensional data points into k dimensions
import numpy as np

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    
    def fit(self, X):
        # mean and covariance
        self.mean = np.mean(X, axis=0)
        X = X - self.mean    
        cov = np.cov(X.T)
        
        # eigenvectors and eigenvalues and sort them
        eigenvectors, eigenvalues = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1] # [::-1] reverses the list
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]
        
    
    def predict(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)
        
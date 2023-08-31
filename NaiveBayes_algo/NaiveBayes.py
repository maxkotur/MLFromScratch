import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # inital mean, var, priors
        self.mean = np.zeroes((n_classes, n_features), dtype=np.float64)
        self.var = np.zeroes((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeroes(n_classes, dtype=np.float64)
        
        for c in self.classes:
            X_c = X[c==y]
            self.mean[c, :] = X_c.mean(axis=0)
            self.var[c, :] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / float(n_samples) # frequency
        
    
    def predict(self, X):
        
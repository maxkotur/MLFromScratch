import numpy as np

class DecisionStump:
    
    def __init__(self):
        self.polarity = 1 # tells us if a sample should be classified as -1 or +1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        
        predictions = np.ones(n_samples)
        if self.polarity == 1: # default case
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
            
        return predictions
    

class Adaboost:
    
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initalize weights
        w = np.full(n_samples, (1/n_samples)) # sets each value to 1/N
        
        for _ in range(self.n_clf):
            clf = DecisionStump()
            
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # calculate the error (sum of weights of misclassed samples)
                    missclassified = w[y != predictions]
                    error = sum(missclassified)
                    
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
        
        # calculate performance
        EPS = 1e-10 # so we dont divide by 0
        clf.alpha = 0.5 * np.log((1-min_error) / (min_error+EPS))
        
        # update the weights
        predictions = clf.predict(X)
        w *= np.exp(-clf.alpha * y * predictions)
        w /= np.sum(w) # normalize it
        
        self.clfs.append(clf)


    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_hat = np.sum(clf_preds, axis=0)
        y_hat = np.sign(y_hat)
        return y_hat
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    np.sqrt(np.sum((x1-x2)**2))

class KNN:
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    
    def predict(self, X):
        predicted_labels = [self.compute_distances(x) for x in X]
        return np.array(predicted_labels)
        
    def compute_distances(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest).most_common(1) # this gets first most common and number of occurences
        return most_common[0][0]
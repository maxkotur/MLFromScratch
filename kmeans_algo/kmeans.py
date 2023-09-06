# Initialize cluster centers (e.g. randomly)
# Repeat until converged:
# - Update cluster labels: Assign points to the nearest cluster center (centroid)
# - Update cluster centers (centroids): Set center to the mean of each cluster

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    
    def __init__(self, K=5, iter=100, plot_steps=False):
        self.K = K
        self.iter = iter,
        self.plot_steps = plot_steps
        
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # mean feature vector for each cluster
        self.centroids = []
        
    # no fit method as we have unlabeled unsupervised learning
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # optimization
        for _ in range(self.iter):
            # update clusters
            self.clusters = self.create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # update centroids
            centroid_old = self.centroids # to check for convergence later
            self.centroids = self.get_centroids(self.clusters)
            # check if converged
            if self.is_converged(centroid_old, self.centroids):
                break
        
        # return cluster labels
        return self.get_cluster_labels(self.clusters)
        
    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
            
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
            
        plt.show()
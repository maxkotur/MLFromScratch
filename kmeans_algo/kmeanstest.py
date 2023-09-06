import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from kmeans import KMeans

X, y = datasets.make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

# Set plot_steps to True if you want the plot of every iteration
k = KMeans(K=clusters, iter=150, plot_steps=False)
y_hat = k.predict(X)
print(y_hat)
k.plot()
import numpy as np

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for ix, x in enumerate(X):
        closest_centroid = 0
        min_dist = np.linalg.norm(x - centroids[0])
        for k in range(K):
            dist = np.linalg.norm(x - centroids[k])
            if dist < min_dist:
                closest_centroid = k
                min_dist = dist
        idx[ix] = closest_centroid
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        indices = np.where(idx == k)
        centroids[k, :] = np.mean(X[indices], axis=0)
    return centroids

def k_means_initialize_centroids(X: np.ndarray, K: int):
    centroids = []
    n = len(X)
    if n > K:
        indices = np.random.choice(a=list(range(n)), size=(K,), replace=False)
        centroids = [X[i] for i in indices]
    return np.array(centroids)
    
import numpy as np
import matplotlib.pyplot as plt

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

def plot_progress_k_means(X, centroids, previous, idx, K, i):
    plt.scatter(X[:, 0], X[:, 1], c=idx)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='r')
    if i > 0:
        plt.plot(previous[:, 0], previous[:, 1], 'k--', linewidth=0.5)
    plt.title("Iteration number %d" % (i+1))
    plt.show()

def run_k_means(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plot_progress_k_means(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx

def k_means_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return np.array(centroids)

def compress_image(X, centroids, idx, K):
    m, n = X.shape
    X_compressed = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            X_compressed[i][j] = centroids[idx[i]][j]
    return X_compressed
import numpy as np
import pickle

def load_data():
    """
    Load dummy dataset for testing ([throughput (mb/s), latency (ms)])
    """
    with open('data.pkl', 'rb') as f:
        X = pickle.load(f)
        return X
    return None

def estimate_gaussian(X):
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    s1 = np.zeros((n,), dtype=np.float_)
    s2 = np.zeros((n,), dtype=np.float_)
    mu = np.zeros((n,), dtype=np.float_)
    var = np.zeros((n,), dtype=np.float_)
    for j in range(m):
        for i in range(n):
            s1[i] += X[j][i]
            s2[i] += X[j][i] * X[j][i]
    for i in range(n):
        mu[i] = s1[i] / m
        var[i] = (s2[i] - (s1[i]*s1[i]) / m) / m
    return mu, var
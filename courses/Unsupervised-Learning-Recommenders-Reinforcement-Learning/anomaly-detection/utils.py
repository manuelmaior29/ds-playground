import numpy as np
import pickle

def load_data():
    """
    Load dummy dataset for testing ([throughput (mb/s), latency (ms)])
    """
    with open('data.pkl', 'rb') as f:
        X = pickle.load(f)
        return X

def load_example_val():
    """
    Load dummy validation preds/gts
    """
    y_val, p_val = None, None
    with open('y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    with open('p_val.pkl', 'rb') as f:
        p_val = pickle.load(f)
    return np.array(y_val), np.array(p_val)

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

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """
    best_epsilon = 0.0
    best_F1 = 0.0
    F1 = 0.0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        tp = np.sum((y_val == 1) & (p_val <= epsilon))
        fp = np.sum((y_val == 0) & (p_val <= epsilon))
        fn = np.sum((y_val == 1) & (p_val > epsilon))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * prec * rec) / (prec + rec)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1
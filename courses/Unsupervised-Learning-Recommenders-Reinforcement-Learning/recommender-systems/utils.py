import numpy as np
import tensorflow as tf
import pandas as pd

def load_ratings_small(path):
    """
      Args:
        path (str): path to the folder where the MovieLens (small) is located
      Returns:
        Y (ndarray (num_movies,num_users)): matrix of user ratings of movies
        R (ndarray (num_movies,num_users)): matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
    """
    ratings_df = pd.read_csv(path + '/ratings.csv')
    movies_df = pd.read_csv(path + '/movies.csv')
    nm, nu = movies_df['movieId'].nunique(), ratings_df['userId'].nunique()
    Y = np.zeros(shape=(nm, nu), dtype=np.float32)
    R = np.zeros(shape=(nm, nu), dtype=np.float32)
    user_ids = ratings_df['userId'].unique()
    for user_id in user_ids:
        movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId']
        for movie_id in movie_ids:
            Y[movie_id - 1][user_id - 1] = ratings_df[(ratings_df['movieId'] == movie_id) & (ratings_df['userId'] == user_id)]['rating'].values[0]
            R[movie_id - 1][user_id - 1] = 1
    return Y, R

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    _, k = W.shape
    for j in range(nu):
        for i in range(nm):
            J += R[i][j] * np.power(np.dot(W[j], X[i]) + b[0][j] - Y[i][j], 2)
    Rw, Rx = 0, 0
    for j in range(nu):
        Rw += np.sum(np.power(W[j], 2))
    for i in range(nm):
        Rx += np.sum(np.power(X[i], 2))
    J = J / 2 + (lambda_ / 2) * (Rw + Rx)
    return J

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J